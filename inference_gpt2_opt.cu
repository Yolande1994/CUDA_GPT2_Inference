/* ============================================================================
   inference_gpt2_opt.cu —— 优化版 GPT-2 推理主程序
   ============================================================================
   工程的主入口，把所有优化串成一条完整的解码链路：
     1. KV 缓存：每步只算新 token 的 K、V 并追加，历史部分直接复用；
     2. 手写 GEMV：单步全连接层走自研 kernel（GEMV.cuh / GEMV_w8.cuh），其余情况走 cublasLt，由本文件的分发函数选择；
     3. GPU 采样：softmax 与采样全部留在设备上，不把 logits 拷回主机；
     4. CUDA Graph：把一整步的 kernel 序列捕获成图重放，省掉每步的启动开销。可选关闭；
     5. 可选 8 位权重（--int8/--fp8）与 8 位 KV（编译开关，默认关）。

   两种运行方式：
     普通模式：每步由主机逐个发起 kernel；
     --graph：先捕获 small/large 两张整步图（按序列长度分界），再按当前长度选图重放。
   单步前向统一放在 forward_kv_graph_ready，普通模式和捕获模式共用，能跑的就能被捕获。
   ============================================================================ */
#include <unistd.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

#include "Inference_gpt2/common.h"
#include "Inference_gpt2/layernorm.cuh"
#include "Inference_gpt2/matmul.cuh"
#include "Inference_gpt2/utils.h"
#include "Inference_gpt2/tokenizer_sampler.h"

#include "Inference_gpt2/gpt2_model.h"
#include "Inference_gpt2/encoder.cuh"
#include "Inference_gpt2/attention_kvcache.cuh"
#include "Inference_gpt2/attention_kvcache_w8.cuh"
#include "Inference_gpt2/sampler_gpu.cuh"

#include "Inference_gpt2/GEMV.cuh"
#ifdef USE_W8
#include "Inference_gpt2/GEMV_w8.cuh"
#endif

cudaDeviceProp deviceProp;
cudaStream_t main_stream;

// 两张图：短序列用 small、长序列用 large，按下面的动态阈值切换
cudaGraphExec_t g_exec_small = nullptr;
cudaGraphExec_t g_exec_large = nullptr;

// g_cur_len 的设备地址缓存，避免每次调用都重新查询
static int* d_cur_len_ptr = nullptr;

#ifdef USE_W8

// 8 位 Linear 权重的分组粒度（1 = 逐通道；保留常量位便于扩展）
static constexpr int W8_G_K768  = 1;
static constexpr int W8_G_K3072 = 1;
// 5 个会被量化的 Linear 权重指针及各自 scale，其余参数仍为 BF16
static uint8_t* g_w8_wte = nullptr;
static uint8_t* g_w8_qkvw = nullptr;
static uint8_t* g_w8_attprojw = nullptr;
static uint8_t* g_w8_fcw = nullptr;
static uint8_t* g_w8_fcprojw = nullptr;
static float* g_w8_wte_s = nullptr;
static float* g_w8_qkvw_s = nullptr;
static float* g_w8_attprojw_s = nullptr;
static float* g_w8_fcw_s = nullptr;
static float* g_w8_fcprojw_s = nullptr;
static bool g_use_w8 = false;
#if defined(USE_KV_W8)
static float* g_kv_scale = nullptr;
#endif
#endif

#include "Inference_gpt2/dump_utils.h"

// 全局初始化：流、cublasLt 句柄与工作区；FP32 档且硬件支持时启用 TF32
void common_start(bool print_device_info = true) {
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    if (print_device_info) {
        printf("[系统]\n");
        printf("设备 0：%s\n", deviceProp.name);
    }
    cudaCheck(cudaStreamCreate(&main_stream));  // 主工作流
    nvtxNameCudaStreamA(main_stream, "main stream");
    cublasCheck(cublasLtCreate(&cublaslt_handle));  // cublasLt 句柄
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));  // 工作区
    bool enable_tf32 = PRECISION_MODE == PRECISION_FP32 && deviceProp.major >= 8;

    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;  // 选累加模式
}

void common_free() {
    cudaCheck(cudaStreamDestroy(main_stream));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
}

// ============================================================================
// 全连接层分发：单 token 解码走手写 GEMV，其余走 cublasLt
// ============================================================================

/* 解码时 B*T==1，正好是 GEMV 形态，自研 kernel 比通用库更贴合；
   预填充等多元素场景矩阵更大，交给 cublasLt。USE_CUBLASLT 定义时强制全走库。 */
static inline void matmul_forward(floatX* out, floatX* inp, floatX* weight, floatX* bias,
                                  int B, int T, int C, int OC, cudaStream_t stream,
                                  floatX* pre_gelu = NULL, int gelu_fusion = 1) {
#ifndef USE_CUBLASLT
    if (B * T == 1 && gelu_fusion >= 1) {  // 单 token 解码走手写融合 GEMV
        launch_GEMV6_fused<floatX>(out, pre_gelu, weight, inp, bias, OC, C, pre_gelu != NULL, stream);
        return;
    }
#endif
    matmul_forward_cublaslt(out, inp, weight, bias, B, T, C, OC, stream, pre_gelu, gelu_fusion);  // 其余走 cublasLt
}

#ifdef USE_W8

// 8 位版分发：单 token 走量化 GEMV
static inline void matmul_forward_w8(floatX* out, floatX* inp, const uint8_t* w8, const float* scale,
                                     floatX* bias, int B, int T, int C, int OC, cudaStream_t stream,
                                     floatX* pre_gelu = NULL, int gelu_fusion = 1) {
    if (B * T == 1) {
        launch_GEMV6_w8x2(out, pre_gelu, w8, inp, scale, bias, OC, C,
                          pre_gelu != NULL, stream);
        return;
    }
    fprintf(stderr, "8 位 matmul_forward: 仅支持 decode（B*T==1）\n");
    exit(1);
}

// ============================================================================
// 8 位权重加载：混合格式检查点（5 个 Linear 为字节，其余 11 个为 BF16）
// ============================================================================

/* ----------------------------------------------------------------------------
   权重文件里 16 个参数按统一顺序排列，每个参数标注元素数和是否量化。
   量化的 Linear 读进独立的 1 字节显存与 scale 显存；
   非量化参数读进模型整块参数显存（按 BF16 每元素 2 字节）。
   最后反量化词嵌入表：嵌入查找按随机 id 取行，不适合在线解码，直接把整张浮点表还原到参数槽里。
   ---------------------------------------------------------------------------- */
static void gpt2_build_w8(GPT2* model, const char* w8_path, const char* scale_path) {
    FILE* f = fopen(w8_path, "rb");
    if (!f) { perror("fopen w8"); exit(1); }
    int header[256];
    if (fread(header, sizeof(int), 256, f) != 256) { fprintf(stderr, "fp8 header read fail\n"); exit(1); }  // 256 整数文件头
    if (header[0] != 20251103 || header[1] != 5) { fprintf(stderr, "fp8 bin 版本错误\n"); exit(1); }
    GPT2Config* cfg = &model->config;
    // 从文件头还原模型尺寸
    cfg->max_seq_len = header[2]; cfg->vocab_size = header[3]; cfg->num_layers = header[4];
    cfg->num_heads = header[5]; cfg->channels = header[6]; cfg->padded_vocab_size = header[7];
    int maxT = cfg->max_seq_len, L = cfg->num_layers, C = cfg->channels, Vp = cfg->padded_vocab_size;
    gpt2_allocate_weights(model);

    // 16 个参数的元素数与“是否为量化 Linear”标记，顺序与文件排布一致
    struct { size_t elems; int linear; } lay[16] = {
        { (size_t)Vp*C, 1 }, { (size_t)maxT*C, 0 }, { (size_t)L*C, 0 }, { (size_t)L*C, 0 },
        { (size_t)L*3*C*C, 1 }, { (size_t)L*3*C, 0 }, { (size_t)L*C*C, 1 }, { (size_t)L*C, 0 },
        { (size_t)L*C, 0 }, { (size_t)L*C, 0 }, { (size_t)L*4*C*C, 1 }, { (size_t)L*4*C, 0 },
        { (size_t)L*C*4*C, 1 }, { (size_t)L*C, 0 }, { (size_t)C, 0 }, { (size_t)C, 0 },
    };

    // 5 个量化 Linear 的独立显存
    size_t w8_bytes[5] = { lay[0].elems, lay[4].elems, lay[6].elems, lay[10].elems, lay[12].elems };
    void* w8_mem[5];
    uint8_t** w8_ptr[5] = { &g_w8_wte, &g_w8_qkvw, &g_w8_attprojw, &g_w8_fcw, &g_w8_fcprojw };
    for (int i = 0; i < 5; i++) { cudaCheck(cudaMalloc(&w8_mem[i], w8_bytes[i])); *w8_ptr[i] = (uint8_t*)w8_mem[i]; }  // 逐块申请并挂指针

    // 非量化参数在整块参数显存中的字节偏移（按 BF16 2 字节/元素累计）
    size_t bf16_elems[16]; size_t acc = 0;
    for (int i = 0; i < 16; i++) { bf16_elems[i] = acc; acc += lay[i].elems; }
    char* pmem = (char*)model->params_memory;
    // 参数下标 → 5 个量化 Linear 的编号，-1 表示非量化
    int linear_idx[16] = { 0, -1, -1, -1, 1, -1, 2, -1, -1, -1, 3, -1, 4, -1, -1, -1 };
    for (int i = 0; i < 16; i++) {
        size_t bytes = lay[i].elems * (lay[i].linear ? 1 : 2);  // 量化 1 字节、BF16 2 字节
        void* h_buf = malloc(bytes);
        if (!h_buf) { fprintf(stderr, "malloc fail\n"); exit(1); }
        if (fread(h_buf, 1, bytes, f) != bytes) { fprintf(stderr, "w8 layer %d read fail\n", i); exit(1); }
        if (lay[i].linear) {
            // 量化 Linear：进独立显存
            cudaCheck(cudaMemcpyAsync(w8_mem[linear_idx[i]], h_buf, bytes, cudaMemcpyHostToDevice));
        } else {
            // 非量化参数：进整块参数显存对应偏移
            cudaCheck(cudaMemcpyAsync(pmem + bf16_elems[i] * 2, h_buf, bytes, cudaMemcpyHostToDevice));
        }
        free(h_buf);
    }
    fclose(f);

    // 5 组 scale 的元素数与上传
    size_t sc_elems[5] = { (size_t)Vp * W8_G_K768, (size_t)L * 3 * C * W8_G_K768,
                           (size_t)L * C * W8_G_K768, (size_t)L * 4 * C * W8_G_K768,
                           (size_t)L * C * W8_G_K3072 };
    size_t sc_total = 0; for (int i = 0; i < 5; i++) sc_total += sc_elems[i];
    float* h_sc = (float*)malloc(sc_total * sizeof(float));
    FILE* fs = fopen(scale_path, "rb");
    if (!fs) { perror("fopen scales"); exit(1); }
    if (fread(h_sc, sizeof(float), sc_total, fs) != sc_total) { fprintf(stderr, "scales read fail\n"); exit(1); }
    fclose(fs);
    float* sc_mem[5];
    float** sc_ptr[5] = { &g_w8_wte_s, &g_w8_qkvw_s, &g_w8_attprojw_s, &g_w8_fcw_s, &g_w8_fcprojw_s };
    size_t sc_off = 0;
    for (int i = 0; i < 5; i++) {
        cudaCheck(cudaMalloc(&sc_mem[i], sc_elems[i] * sizeof(float)));
        cudaCheck(cudaMemcpyAsync(sc_mem[i], h_sc + sc_off, sc_elems[i] * sizeof(float), cudaMemcpyHostToDevice));
        *sc_ptr[i] = sc_mem[i];
        sc_off += sc_elems[i];
    }
    free(h_sc);

    // 词嵌入表整体反量化，后续嵌入查找直接读浮点参数槽
    dequant_wte_kernel<<<Vp, 256>>>(g_w8_wte, g_w8_wte_s, model->params.wte, Vp, C);  // 每 block 一行
    cudaCheck(cudaDeviceSynchronize());
    printf("[W8] 权重加载完成：8 位 Linear 权重 %.1f MB + scales %.2f MB\n",
           (double)(w8_bytes[0]+w8_bytes[1]+w8_bytes[2]+w8_bytes[3]+w8_bytes[4]) / 1048576.0,
           (double)sc_total * 4 / 1048576.0);
}
#endif


// ============================================================================
// forward_kv_graph_ready：单步解码前向（普通模式与 Graph 捕获共用）
// ============================================================================

/* ----------------------------------------------------------------------------
   每一步只处理当前一个新 token。结构按层展开，层内顺序：
     LN1 → QKV 投影 → K/V 写入缓存 → 读缓存做注意力 → 输出投影 → 残差+LN2 → MLP 两层 → 残差+下一层 LN1（最后一层是最终 LN）。
   注意力路径的选择：
     长度低于阈值走 fused single；高于阈值走 split+merge；
     is_capturing 为真时强制 split+merge（cooperative kernel 无法被 Graph捕获，而 split 与 merge 是两个普通 kernel，可以安全进图）
   函数内不做依赖数据的主机判断，全部是可记录的设备操作，因此整张图可捕获。
   ---------------------------------------------------------------------------- */
void forward_kv_graph_ready(GPT2* model, int cur_len_host, floatX* kv_cache,
                            float* part_out, float* part_lse, float* part_max,
                            int S, cudaStream_t stream, bool is_capturing = false) {
    const size_t B = model->batch_size, T = 1;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;
    const size_t HS = C / NH;
    const size_t maxT = model->config.max_seq_len;

    // 短/长序列的分界长度，B 不同阈值不同
    const int dynamic_threshold = (B == 1) ? 150 : 400;

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;

    if (d_cur_len_ptr == nullptr) {
        cudaCheck(cudaGetSymbolAddress((void**)&d_cur_len_ptr, g_cur_len));  // 缓存 g_cur_len 地址
    }
    // 解码版嵌入：位置由 kernel 自己读设备端 g_cur_len
    encoder_forward_kv(acts.encoded, model->inputs, params.wte, params.wpe, d_cur_len_ptr, B, C, stream);

    // 第 0 层的 LN1 先算，后续层的 LN1 由上一层末尾融合完成
    layernorm_forward(acts.ln1, acts.ln1_mean, acts.ln1_rstd, acts.encoded, params.ln1w, params.ln1b, B, T, C, stream);

    for (int l = 0; l < L; l++) {
        // 本层激活的起始地址
        floatX* l_ln1 = acts.ln1 + l * B * C;
        floatX* l_atty = acts.atty + l * B * C;
        floatX* l_residual2 = acts.residual2 + l * B * C;
        floatX* l_ln2 = acts.ln2 + l * B * C;
        float* l_ln2_mean = acts.ln2_mean + l * B;
        float* l_ln2_rstd = acts.ln2_rstd + l * B;
        floatX* l_fch = acts.fch + l * B * 4 * C;
        floatX* l_fch_gelu = acts.fch_gelu + l * B * 4 * C;
        floatX* l_residual3 = acts.residual3 + l * B * C;
        floatX* scratch = (floatX*)acts.output;
        floatX* l_qkv = scratch;  // QKV 投影先写进 scratch

        // 本层权重起始地址
        floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_qkvb = params.qkvb + l * 3*C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_attprojb = params.attprojb + l * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcb = params.fcb + l * 4*C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        floatX* l_fcprojb = params.fcprojb + l * C;

        // 本层残差输入：第 0 层是嵌入输出，其余是上一层残差
        floatX* residual = l == 0 ? acts.encoded : (acts.residual3 + (l - 1) * B * C);

#ifdef USE_W8
        if (g_use_w8) {
            matmul_forward_w8(l_qkv, l_ln1, g_w8_qkvw + (size_t)l * 3*C * C, g_w8_qkvw_s + (size_t)l * 3*C * W8_G_K768,
                               l_qkvb, B, T, C, 3*C, stream);
        } else
#endif
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, stream);

#if defined(USE_KV_W8)
        if (g_use_w8) {
            // 本层的量化 KV 缓存与 scale 区域
            uint8_t* layer_kv_cache8 = (uint8_t*)kv_cache + (size_t)l * (2 * B * NH * maxT * HS);
            kv_cache_write_kernel_w8<<<B * NH * 2, 256, 0, stream>>>(
                layer_kv_cache8, g_kv_scale + (size_t)l * (2 * B * NH * maxT), l_qkv, B, NH, HS, maxT);
            if (cur_len_host < dynamic_threshold) {
                attention_forward_kv_small_w8(l_atty, l_qkv, layer_kv_cache8,
                                               g_kv_scale + (size_t)l * (2 * B * NH * maxT),
                                               B, NH, HS, maxT, stream);
            } else if (is_capturing) {
                // 捕获模式：强制用 split+merge，两个普通 kernel 可进图
                attention_kv_split_kernel_w8<<<B * NH * S, 256, 0, stream>>>(
                    part_out, part_lse, part_max, l_qkv, layer_kv_cache8,
                    g_kv_scale + (size_t)l * (2 * B * NH * maxT), B, NH, HS, maxT, S, 1.f/sqrtf(HS));
                attention_kv_merge_kernel_w8<<<B * NH, 64, 0, stream>>>(
                    l_atty, part_out, part_lse, part_max, NH, HS, S);
            } else {
                attention_forward_kv_large_w8(l_atty, l_qkv, layer_kv_cache8,
                                               g_kv_scale + (size_t)l * (2 * B * NH * maxT),
                                               B, NH, HS, maxT, stream,
                                               part_out, part_lse, part_max, S);
            }
        } else
#endif
        {
        floatX* layer_kv_cache = kv_cache + l * (2 * B * NH * maxT * HS);  // 本层 KV 槽位
        {
            int num_threads = 256;
            int num_blocks = CEIL_DIV(B * NH * HS, num_threads);
            // 当前 K/V 追加到缓存
            kv_cache_write_kernel<<<num_blocks, num_threads, 0, stream>>>(layer_kv_cache, l_qkv, B, NH, HS, maxT);
        }
        if (cur_len_host < dynamic_threshold) {
            // 短序列：用单 kernel 融合
            attention_forward_kv_small(l_atty, l_qkv, layer_kv_cache, B, NH, HS, maxT, stream);
        } else {

            if (is_capturing) {
                // 同上：捕获时用 split+merge 替代 cooperative
                attention_kv_split_kernel<<<B * NH * S, 256, 0, stream>>>(
                    part_out, part_lse, part_max, l_qkv, layer_kv_cache, B, NH, HS, maxT, S, 1.f/sqrtf(HS));
                attention_kv_merge_kernel<<<B * NH, 64, 0, stream>>>(
                    l_atty, part_out, part_lse, part_max, NH, HS, S);
            } else {
                attention_forward_kv_large(l_atty, l_qkv, layer_kv_cache, B, NH, HS, maxT, stream,
                                           part_out, part_lse, part_max, S);
            }
        }
        }

#ifdef USE_W8
        if (g_use_w8) {
            matmul_forward_w8(scratch, l_atty, g_w8_attprojw + (size_t)l * C * C, g_w8_attprojw_s + (size_t)l * C * W8_G_K768,
                               l_attprojb, B, T, C, C, stream);
        } else
#endif
        matmul_forward(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, stream);
        // 注意力输出加残差并做 LN2，融合在一个 kernel
        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd, residual, scratch, l_ln2w, l_ln2b,
                                (int)(B*T), (int)C, stream);

#ifdef USE_W8
        if (g_use_w8) {
            matmul_forward_w8(l_fch_gelu, l_ln2, g_w8_fcw + (size_t)l * 4*C * C, g_w8_fcw_s + (size_t)l * 4*C * W8_G_K768,
                               l_fcb, B, T, C, 4*C, stream, l_fch, model->gelu_fusion);
        } else
#endif
        matmul_forward(l_fch_gelu, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, stream, l_fch, model->gelu_fusion);

#ifdef USE_W8
        if (g_use_w8) {
            matmul_forward_w8(scratch, l_fch_gelu, g_w8_fcprojw + (size_t)l * C * 4*C, g_w8_fcprojw_s + (size_t)l * C * W8_G_K3072,
                               l_fcprojb, B, T, 4*C, C, stream);
        } else
#endif
        matmul_forward(scratch, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, stream);

        // 层末尾：残差 + 下一层 LN1（最后一层是最终 LN）
        if (l + 1 != L) {
            floatX* nl_ln1 = acts.ln1 + (l + 1) * B * C;
            float* nl_ln1_mean = acts.ln1_mean + (l + 1) * B;
            float* nl_ln1_rstd = acts.ln1_rstd + (l + 1) * B;
            const floatX* nl_ln1w = params.ln1w + (l + 1) * C;
            const floatX* nl_ln1b = params.ln1b + (l + 1) * C;
            fused_residual_forward5(l_residual3, nl_ln1, nl_ln1_mean, nl_ln1_rstd, l_residual2, scratch, nl_ln1w, nl_ln1b,
                                   (int)(B*T), (int)C, stream);
        } else {
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd, l_residual2, scratch,
                                   params.lnfw, params.lnfb, (int)(B*T), (int)C, stream);
        }
    }

#ifdef USE_W8
    if (g_use_w8) {
        // 词表投影复用量化词嵌入表，输出固定 float
        launch_GEMV6_w8x2<float, floatX>(acts.logits, (float*)nullptr, g_w8_wte, acts.lnf, g_w8_wte_s, (floatX*)nullptr,
                                         Vp, C, false, stream);
    } else
#endif
    // 词表投影：float32 输出，权重绑定复用词嵌入表
    matmul_forward_cublaslt_f32(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp, stream);
}

// 普通模式单步入口：上传输入、更新设备端长度，再跑可捕获前向
void gpt2_forward_kv(GPT2 *model, const int* inputs, int cur_len,
                     floatX* kv_cache,
                     float* part_out, float* part_lse, float* part_max,
                     int S) {
    cudaCheck(cudaMemcpy(model->inputs, inputs, model->batch_size * sizeof(int), cudaMemcpyHostToDevice));  // 上传当前 token
    tokenCheck(inputs, model->batch_size, model->config.vocab_size);
    cudaCheck(cudaMemcpyToSymbol(g_cur_len, &cur_len, sizeof(int)));  // 更新设备端长度
    forward_kv_graph_ready(model, cur_len, kv_cache, part_out, part_lse, part_max, S, main_stream, false);  // 可捕获前向
    cudaCheck(cudaDeviceSynchronize());
}

// ============================================================================
// graph_capture_all：预热 + 捕获 small/large 两张整步图
// ============================================================================

/* 先空跑一次让所有 kernel 完成初始化；再分别在“阈值−1”和“阈值”两个长度开始捕获，得到两张内部 kernel 序列不同的图。
   捕获前后的长度通过 g_cur_len 写入，图内 kernel 自己读它，因此重放时长度仍可逐变化。 */
void graph_capture_all(GPT2* model, floatX* kv_cache,
                       float* part_out, float* part_lse, float* part_max,
                       int S, cudaStream_t stream) {
    int dynamic_threshold = (model->batch_size == 1) ? 150 : 400;

    // 预热：用略大于阈值的长度完整跑一遍，避免捕获时才触发初始化
    int w = dynamic_threshold + 10;
    cudaCheck(cudaMemcpyToSymbol(g_cur_len, &w, sizeof(int)));
    forward_kv_graph_ready(model, w, kv_cache, part_out, part_lse, part_max, S, stream, true);
    cudaCheck(cudaStreamSynchronize(stream));

    cudaGraph_t graph;
    // 第一张：阈值以下的长度
    int cap = dynamic_threshold - 1;
    cudaCheck(cudaMemcpyToSymbol(g_cur_len, &cap, sizeof(int)));
    cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));  // 开始记录
    forward_kv_graph_ready(model, cap, kv_cache, part_out, part_lse, part_max, S, stream, true);
    cudaCheck(cudaStreamEndCapture(stream, &graph));  // 结束记录得到图
    cudaCheck(cudaGraphInstantiate(&g_exec_small, graph, 0));  // 实例化可执行图
    cudaCheck(cudaGraphDestroy(graph));

    // 第二张：阈值及以上的长度
    cap = dynamic_threshold;
    cudaCheck(cudaMemcpyToSymbol(g_cur_len, &cap, sizeof(int)));
    cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    forward_kv_graph_ready(model, cap, kv_cache, part_out, part_lse, part_max, S, stream, true);
    cudaCheck(cudaStreamEndCapture(stream, &graph));
    cudaCheck(cudaGraphInstantiate(&g_exec_large, graph, 0));
    cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaStreamSynchronize(stream));
}

// ============================================================================
// 生成循环（普通模式）
// ============================================================================
static double run_generation_once(GPT2* model, Tokenizer* tokenizer, int* gen_tokens,
                                  int* d_token,
                                  int genT, int eot_token, bool quiet,
                                  floatX* kv_cache,
                                  float* part_out, float* part_lse, float* part_max,
                                  int S) {
    int B = model->batch_size;
    for (int i = 0; i < B * (model->config.max_seq_len + 1); i++) gen_tokens[i] = eot_token;

    if (!quiet) printf("开始生成：\n---\n");
    cudaEvent_t gen_start, gen_stop;
    cudaCheck(cudaEventCreate(&gen_start));
    cudaCheck(cudaEventCreate(&gen_stop));
    cudaCheck(cudaEventRecord(gen_start, main_stream));
    auto wall_start = std::chrono::steady_clock::now();
    const unsigned long long sample_seed = 1337;

    for (int t = 1; t < genT && t <= model->config.max_seq_len; t++) {
        int* current_inputs = (int*)malloc(B * sizeof(int));
        // 每步只喂上一步选出的新 token
        for (int b = 0; b < B; b++) current_inputs[b] = gen_tokens[b * model->config.max_seq_len + (t - 1)];

        gpt2_forward_kv(model, current_inputs, t - 1, kv_cache, part_out, part_lse, part_max, S);  // 单步前向
        free(current_inputs);

        if (t == 1) dump_logits_t1(model);
        // 仅在用户指定的步骤落盘，避免每步写文件
        if (!quiet && want_dump_step(t)) {
            dump_logits_step(model, DEFAULT_DUMP_PREFIX, t);
            dump_inputs_step(gen_tokens, DEFAULT_DUMP_PREFIX, t);
        }

        // 采样全程在 GPU 上完成，只把最终 token 拷回
        gpu_sample_softmax(d_token, model->acts.logits, B, model->config.vocab_size, sample_seed, t, main_stream);

        int* next_tokens = (int*)malloc(B * sizeof(int));
        cudaCheck(cudaMemcpy(next_tokens, d_token, B * sizeof(int), cudaMemcpyDeviceToHost));  // 只回传 token

        for (int b = 0; b < B; b++) gen_tokens[b * model->config.max_seq_len + t] = next_tokens[b];  // 追加到序列
        free(next_tokens);

        if (g_ppl_len > 0 && t <= g_ppl_len) ppl_record_nll(model, gen_tokens[t]);

        if (!quiet && B == 1) {
            if (tokenizer->init_ok) safe_printf(tokenizer_decode(tokenizer, gen_tokens[t]));
            else printf("%d ", gen_tokens[t]);
            fflush(stdout);
        }
    }
    if (!quiet) printf("\n---\n");

    cudaCheck(cudaEventRecord(gen_stop, main_stream));
    cudaCheck(cudaEventSynchronize(gen_stop));
    float gpu_ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&gpu_ms, gen_start, gen_stop));
    auto wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
    cudaCheck(cudaEventDestroy(gen_start));
    cudaCheck(cudaEventDestroy(gen_stop));

    if (!quiet) {
        int gen_count = genT - 1;
        printf("\n[性能]\n生成 %d 个 token：\n  GPU 耗时：%.1f ms（平均 %.2f ms/token）\n  墙钟耗时：%.1f ms（平均 %.2f ms/token）\n", gen_count, gpu_ms, gpu_ms / gen_count, wall_ms, wall_ms / gen_count);
    }
    if (g_ppl_len > 0) ppl_write(DEFAULT_DUMP_PREFIX);
    return wall_ms;
}

// ============================================================================
// 生成循环（Graph 模式）：每步只更新长度与输入，然后选图重放
// ============================================================================
static double run_generation_once_graph(GPT2* model, Tokenizer* tokenizer, int* gen_tokens,
                                        int* d_token,
                                        int genT, int eot_token, bool quiet,
                                        floatX* kv_cache,
                                        float* part_out, float* part_lse, float* part_max,
                                        int S) {
    int B = model->batch_size;
    for (int i = 0; i < B * (model->config.max_seq_len + 1); i++) gen_tokens[i] = eot_token;

    if (!quiet) printf("开始生成（CUDA Graph 回放）：\n---\n");
    cudaEvent_t gen_start, gen_stop;
    cudaCheck(cudaEventCreate(&gen_start));
    cudaCheck(cudaEventCreate(&gen_stop));
    cudaCheck(cudaEventRecord(gen_start, main_stream));
    auto wall_start = std::chrono::steady_clock::now();
    const unsigned long long sample_seed = 1337;

    const int dynamic_threshold = (B == 1) ? 150 : 400;

    for (int t = 1; t < genT && t <= model->config.max_seq_len; t++) {
        const int cur_len = t - 1;

        // 图重放前仍要更新设备长度和当前输入，图内 kernel 读它们
        cudaCheck(cudaMemcpyToSymbol(g_cur_len, &cur_len, sizeof(int)));

        int* current_inputs = (int*)malloc(B * sizeof(int));
        for (int b = 0; b < B; b++) current_inputs[b] = gen_tokens[b * model->config.max_seq_len + (t - 1)];
        cudaCheck(cudaMemcpy(model->inputs, current_inputs, B * sizeof(int), cudaMemcpyHostToDevice));
        free(current_inputs);

        // 按当前长度选 small/large 图重放
        cudaCheck(cudaGraphLaunch(cur_len < dynamic_threshold ? g_exec_small : g_exec_large, main_stream));
        if (t == 1) dump_logits_t1(model);

        if (!quiet && want_dump_step(t)) {
            dump_logits_step(model, DEFAULT_DUMP_PREFIX, t);
            dump_inputs_step(gen_tokens, DEFAULT_DUMP_PREFIX, t);
        }

        gpu_sample_softmax(d_token, model->acts.logits, B, model->config.vocab_size, sample_seed, t, main_stream);

        int* next_tokens = (int*)malloc(B * sizeof(int));
        cudaCheck(cudaMemcpy(next_tokens, d_token, B * sizeof(int), cudaMemcpyDeviceToHost));

        for (int b = 0; b < B; b++) gen_tokens[b * model->config.max_seq_len + t] = next_tokens[b];
        free(next_tokens);

        if (g_ppl_len > 0 && t <= g_ppl_len) ppl_record_nll(model, gen_tokens[t]);

        if (!quiet && B == 1) {
            if (tokenizer->init_ok) safe_printf(tokenizer_decode(tokenizer, gen_tokens[t]));
            else printf("%d ", gen_tokens[t]);
            fflush(stdout);
        }
    }
    if (!quiet) printf("\n---\n");

    cudaCheck(cudaEventRecord(gen_stop, main_stream));
    cudaCheck(cudaEventSynchronize(gen_stop));
    float gpu_ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&gpu_ms, gen_start, gen_stop));
    auto wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
    cudaCheck(cudaEventDestroy(gen_start));
    cudaCheck(cudaEventDestroy(gen_stop));

    if (!quiet) {
        int gen_count = genT - 1;
        printf("\n[性能]\n生成 %d 个 token：\n  GPU 耗时：%.1f ms（平均 %.2f ms/token）\n  墙钟耗时：%.1f ms（平均 %.2f ms/token）\n", gen_count, gpu_ms, gpu_ms / gen_count, wall_ms, wall_ms / gen_count);
    }
    if (g_ppl_len > 0) ppl_write(DEFAULT_DUMP_PREFIX);
    return wall_ms;
}


int main(int argc, char *argv[]) {
    const char* model_path = "assets/weights/gpt2_124M.bin";
    const char* tokenizer_path = "assets/weights/gpt2_tokenizer.bin";
    int genT = 1001;
    int B = 1;
    bool bench_mode = false;
    bool use_graph = false;
    int bench_runs = 6;
    int positional = 0;
#ifdef USE_W8
    bool use_w8 = false;
#endif

    // 命令行参数解析：开关带参数的就地取下一个 argv
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) {
            bench_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                int r = atoi(argv[i + 1]);
                if (r > 1) { bench_runs = r; i++; }
            }
        } else if (strcmp(argv[i], "--graph") == 0) {
            use_graph = true;
#ifdef USE_W8
#if defined(USE_INT8)
        // 8 位档位名跟随编译格式：INT8 编译识别 --int8，FP8 编译识别 --fp8
        } else if (strcmp(argv[i], "--int8") == 0) {
            use_w8 = true;
#else
        } else if (strcmp(argv[i], "--fp8") == 0) {
            use_w8 = true;
#endif
#endif
        } else if (strcmp(argv[i], "--batch") == 0 || strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) { B = atoi(argv[i + 1]); i++; }
        } else if (strcmp(argv[i], "--dump-steps") == 0) {

            // 逗号分隔的步骤列表，逐个解析进 g_dump_steps
            if (i + 1 < argc) {
                char* p = strtok(argv[i + 1], ",");
                while (p) {
                    int s = atoi(p);
                    if (s >= 1) g_dump_steps.push_back(s);
                    p = strtok(NULL, ",");
                }
                i++;
            }
        } else if (strcmp(argv[i], "--dump-prefix") == 0) {

            if (i + 1 < argc) { g_dump_prefix = argv[i + 1]; i++; }
        } else if (strcmp(argv[i], "--ppl") == 0) {

            if (i + 1 < argc) { g_ppl_len = atoi(argv[i + 1]); if (g_ppl_len < 1) g_ppl_len = 1; i++; }
        } else if (positional == 0) { model_path = argv[i]; positional++; }
        else if (positional == 1) { tokenizer_path = argv[i]; positional++; }
        else { genT = atoi(argv[i]); positional++; }
    }

    if (B <= 0) { fprintf(stderr, "无效的 Batch Size，将设为 1\n"); B = 1; }

    common_start();
    GPT2 model;
    gpt2_init_common(&model);
#ifdef USE_W8
    if (use_w8) {
        g_use_w8 = true;
        // scale 文件约定与权重同名、后缀替换为 _scales.bin
        std::string w8_path = model_path;
        std::string scale_path = w8_path.substr(0, w8_path.size() - 4) + "_scales.bin";
        gpt2_build_w8(&model, w8_path.c_str(), scale_path.c_str());
    } else
#endif
    gpt2_build_from_checkpoint(&model, model_path);

    // 解码每步只处理 1 个 token，T 固定为 1
    int T = 1;
    gpt2_allocate_state(&model, B, T);

    printf("[GPT-2 模型]\n");
    printf("最大序列长度：%d\n", model.config.max_seq_len);
    printf("词表大小：%d\n", model.config.vocab_size);
    printf("层数：%d\n", model.config.num_layers);
    printf("注意力头数：%d\n", model.config.num_heads);
    printf("通道数：%d\n", model.config.channels);
    printf("参数量：%zu\n", model.num_parameters);
    printf("Batch Size：%d\n", B);
    if (!g_dump_steps.empty()) {
        printf("[精度验证] 将 dump 步：");
        for (size_t i = 0; i < g_dump_steps.size(); i++) printf("%d%s", g_dump_steps[i], i + 1 < g_dump_steps.size() ? ", " : "");
        printf("\n");
    }

    // KV 缓存：2(K/V) × 层 × B × 头 × 最大长度 × 头维度
    floatX* kv_cache;
    size_t kv_bytes = 2 * (size_t)model.config.num_layers * B * model.config.num_heads *
                      model.config.max_seq_len * (model.config.channels / model.config.num_heads) * sizeof(floatX);
#if defined(USE_KV_W8)

    if (g_use_w8) kv_bytes /= 2;  // 量化 KV 每元素 1 字节，缓存减半
#endif
    cudaCheck(cudaMalloc((void**)&kv_cache, kv_bytes));  // 申请 KV 缓存
#if defined(USE_KV_W8)
    if (g_use_w8) {
        // 量化 KV 的 scale 区域：每个 (K/V, 层, B, 头, 位置) 一个 float
        size_t sc_bytes = 2 * (size_t)model.config.num_layers * B * model.config.num_heads *
                          model.config.max_seq_len * sizeof(float);
        cudaCheck(cudaMalloc((void**)&g_kv_scale, sc_bytes));
    }
#endif
    printf("生成模式：KV Cache 增量前向（缓存占用 %zu MiB 显存）\n", kv_bytes / 1024 / 1024);

    // Flash Decoding 固定切 8 段（实测最佳段数）
    const int S = 8;
    float* part_out; float* part_lse; float* part_max;
    // 三样部分结果的显存
    cudaCheck(cudaMalloc((void**)&part_out, B * model.config.num_heads * S * (model.config.channels / model.config.num_heads) * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&part_lse, B * model.config.num_heads * S * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&part_max, B * model.config.num_heads * S * sizeof(float)));

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, tokenizer_path);

    int* gen_tokens = (int*)mallocCheck(B * (model.config.max_seq_len + 1) * sizeof(int));
    int* d_token;
    cudaCheck(cudaMalloc((void**)&d_token, B * sizeof(int)));

    int eot_token = tokenizer.eot_token;
    for (int i = 0; i < B * (model.config.max_seq_len + 1); i++) gen_tokens[i] = eot_token;

    if (bench_mode) {
        if (use_graph) {
            graph_capture_all(&model, kv_cache, part_out, part_lse, part_max, S, main_stream);
            printf("[CUDA Graph] 已捕获 2 张整步图\n");
        }
        // 第 1 次当预热丢弃，其余墙钟时间排序取中位数
        std::vector<double> wall_times;
        for (int r = 0; r < bench_runs; r++) {
            double w;
            if (use_graph) {
                w = run_generation_once_graph(&model, &tokenizer, gen_tokens, d_token, genT, eot_token, true, kv_cache, part_out, part_lse, part_max, S);
            } else {
                w = run_generation_once(&model, &tokenizer, gen_tokens, d_token, genT, eot_token, true, kv_cache, part_out, part_lse, part_max, S);
            }
            if (r >= 1) wall_times.push_back(w);
            printf("  第 %d/%d 次：%.1f ms\n", r + 1, bench_runs, w);
        }
        std::sort(wall_times.begin(), wall_times.end());
        double median = wall_times[wall_times.size() / 2];
        int gen_count = genT - 1;
        printf("\n[性能汇总]\n生成 %d 个 token（丢弃第 1 次，取 %d 次中位数）：\n  中位数：%.1f ms（平均 %.2f ms/token）\n", gen_count, (int)wall_times.size(), median, median / gen_count);
    } else {
        if (use_graph) {
            graph_capture_all(&model, kv_cache, part_out, part_lse, part_max, S, main_stream);
            printf("[CUDA Graph] 已捕获 2 张整步图\n");
            run_generation_once_graph(&model, &tokenizer, gen_tokens, d_token, genT, eot_token, false, kv_cache, part_out, part_lse, part_max, S);
        } else {
            run_generation_once(&model, &tokenizer, gen_tokens, d_token, genT, eot_token, false, kv_cache, part_out, part_lse, part_max, S);
        }
    }

    free(gen_tokens);
    cudaCheck(cudaFree(d_token));
    cudaCheck(cudaFree(kv_cache));
    cudaCheck(cudaFree(part_out));
    cudaCheck(cudaFree(part_lse));
    cudaCheck(cudaFree(part_max));
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    common_free();
    return 0;
}
