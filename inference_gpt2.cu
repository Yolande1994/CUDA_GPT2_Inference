/* ============================================================================
   inference_gpt2.cu —— 基线推理程序
   ============================================================================
   这是优化前的原始实现，作为性能与精度的对照基线。
   它和优化版 inference_gpt2_opt.cu 的区别：
     1. 没有 KV 缓存；
     2. 矩阵乘法全部走 cublasLt（优化版在 decode 场景改走自研 GEMV kernel）；
     3. 采样在 CPU 上做：每步把 logits 拷回主机，由主机 softmax 采样（优化版采样全程留在 GPU 上）；
     4. 没有 CUDA Graph：每个 kernel 都由主机逐个发起，每步都有启动开销（优化版把一整步捕获成图，重放即可）；
     5. 不支持 8 位量化：只有 FP32/BF16 两种编译精度，没有 int8/fp8 权重，也没有 8 位 KV；
     6. logits 与其他张量一样是 floatX：优化版的词表投影输出固定 float32。
   程序流程：
     解析参数 → 加载权重 → 分配激活 → 生成循环（全量前向 → 拷回 logits → CPU 采样）→ 打印耗时；--bench 模式跑多次、弃首跑取中位数。
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

#include "Inference_gpt2/encoder.cuh"
#include "Inference_gpt2/layernorm.cuh"
#include "Inference_gpt2/matmul.cuh"
#include "Inference_gpt2/attention.cuh"

#include "Inference_gpt2/utils.h"
#include "Inference_gpt2/tokenizer_sampler.h"

// 全局变量：设备属性、主工作流、权重流式上传的分块大小（32 MB）
cudaDeviceProp deviceProp;
cudaStream_t main_stream;
constexpr size_t IO_BUF_SIZE = 32 * 1024 * 1024;

// ============================================================================
// 模型蓝图（一）：配置尺寸 + 参数（权重）张量
// ============================================================================
typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int channels;
} GPT2Config;

constexpr int NUM_PARAMETER_TENSORS = 16;
typedef struct {
    floatX* wte;
    floatX* wpe;
    floatX* ln1w;
    floatX* ln1b;
    floatX* qkvw;
    floatX* qkvb;
    floatX* attprojw;
    floatX* attprojb;
    floatX* ln2w;
    floatX* ln2b;
    floatX* fcw;
    floatX* fcb;
    floatX* fcprojw;
    floatX* fcprojb;
    floatX* lnfw;
    floatX* lnfb;
} ParameterTensors;
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

// ============================================================================
// 参数尺寸：算出 16 个权重张量各自的元素数；带 L 的每层一份
// ============================================================================
void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C;
    param_sizes[1] = maxT * C;
    param_sizes[2] = L * C;
    param_sizes[3] = L * C;
    param_sizes[4] = L * (3 * C) * C;
    param_sizes[5] = L * (3 * C);
    param_sizes[6] = L * C * C;
    param_sizes[7] = L * C;
    param_sizes[8] = L * C;
    param_sizes[9] = L * C;
    param_sizes[10] = L * (4 * C) * C;
    param_sizes[11] = L * (4 * C);
    param_sizes[12] = L * C * (4 * C);
    param_sizes[13] = L * C;
    param_sizes[14] = C;
    param_sizes[15] = C;

    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);
    }
}

// ============================================================================
// 参数显存：一次 cudaMalloc 申请整块，16 个命名指针依次指向块内位置
// ============================================================================
void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_elements, size_t *param_sizeof) {
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters_bytes += param_elements[i] * param_sizeof[i];  // 累加总字节
    }

    void* params_memory;
    cudaCheck(cudaMalloc((void**)&params_memory, num_parameters_bytes));  // 一次申请整块

    floatX** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    char* params_memory_iterator = (char*)params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)params_memory_iterator;  // 指针落在当前游标
        params_memory_iterator += param_elements[i] * param_sizeof[i];  // 再推进该张量字节
    }
    return params_memory;
}

// ============================================================================
// 模型蓝图（二）：激活（中间结果）张量
// ============================================================================
constexpr int NUM_ACTIVATION_TENSORS = 21;
typedef struct {
    floatX* encoded;
    floatX* ln1;
    float* ln1_mean;
    float* ln1_rstd;
    floatX* atty;
    floatX* att;
    floatX* residual2;
    floatX* ln2;
    float* ln2_mean;
    float* ln2_rstd;
    floatX* fch;
    floatX* fch_gelu;
    floatX* residual3;
    floatX* lnf;
    float* lnf_mean;
    float* lnf_rstd;
    float* losses;
    floatX* qkv;

    floatX* output;

    floatX* scratch_bt4c;
    floatX* scratch_btc;
} ActivationTensors;

// 激活的统一描述：指针 + 元素数 + 数据类型，让分配逻辑可以统一处理
struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};

#define TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)};

// ============================================================================
// 激活尺寸：逐个填 21 个张量的元素数；recompute 档位可把部分张量置 0
// ============================================================================
void fill_in_activation_sizes(const ActivationTensors* data, TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS], size_t B, size_t T, GPT2Config config, int recompute) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    tensors[0] = TENSOR_SPEC(data->encoded, B * T * C);
    tensors[1] = TENSOR_SPEC(data->ln1,  (recompute < 2) ? L * B * T * C : 0);
    tensors[2] = TENSOR_SPEC(data->ln1_mean, L * B * T);
    tensors[3] = TENSOR_SPEC(data->ln1_rstd, L * B * T);
    tensors[4] = TENSOR_SPEC(data->atty, L * B * T * C);
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * T * T);
    tensors[6] = TENSOR_SPEC(data->residual2, L * B * T * C);
    tensors[7] = TENSOR_SPEC(data->ln2,  (recompute < 2) ? L * B * T * C : 0);
    tensors[8] = TENSOR_SPEC(data->ln2_mean, L * B * T);
    tensors[9] = TENSOR_SPEC(data->ln2_rstd, L * B * T);
    tensors[10] = TENSOR_SPEC(data->fch, L * B * T * 4*C);
    tensors[11] = TENSOR_SPEC(data->fch_gelu, (recompute < 1) ? L * B * T * 4*C : B * T * 4*C);
    tensors[12] = TENSOR_SPEC(data->residual3, L * B * T * C);
    tensors[13] = TENSOR_SPEC(data->lnf, B * T * C);
    tensors[14] = TENSOR_SPEC(data->lnf_mean, B * T);
    tensors[15] = TENSOR_SPEC(data->lnf_rstd, B * T);
    tensors[16] = TENSOR_SPEC(data->losses, B * T);
    tensors[17] = TENSOR_SPEC(data->qkv, L * B * T * 3*C);

    tensors[18] = TENSOR_SPEC(data->output, B * T * max(3*C, max(NH*T, Vp)));
    tensors[19] = TENSOR_SPEC(data->scratch_bt4c, B * T * 4 * C);
    tensors[20] = TENSOR_SPEC(data->scratch_btc, B * T * C);
}

// ============================================================================
// 激活显存：一次 cudaMalloc 申请整块并清零，零尺寸张量给 NULL
// ============================================================================
void* malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]) {
    size_t bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);  // 累加激活总字节
    }
    printf("正在为激活值分配 %d MiB 显存\n", (int)round(bytes / (1024 * 1024)));

    void* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, bytes));

    cudaCheck(cudaMemset(acts_memory, 0, bytes));  // 清零

    char* acts_memory_iterator = (char*)acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        if(tensors[i].size == 0) {
            *(tensors[i].ptr) = NULL;  // 零尺寸给 NULL
        } else {
            *(tensors[i].ptr) = acts_memory_iterator;  // 游标切块
            acts_memory_iterator += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }
    return acts_memory;
}

// ============================================================================
// 顶层 GPT2 结构：把配置、参数、激活、显存句柄和运行参数全部打包
// ============================================================================
typedef struct {
    GPT2Config config;
    ParameterTensors params;
    size_t param_elements[NUM_PARAMETER_TENSORS];
    size_t param_sizeof[NUM_PARAMETER_TENSORS];
    void* params_memory;
    size_t num_parameters;
    size_t num_parameters_bytes;
    ActivationTensors acts;
    TensorSpec acts_specs[NUM_ACTIVATION_TENSORS];
    void* acts_memory;
    int batch_size;
    int seq_len;
    int* inputs;
    int recompute;
    int gelu_fusion;
} GPT2;

// ============================================================================
// 公共启动/释放：流、cublasLt 句柄与工作区
//   TF32 仅在 FP32 且硬件支持时启用，其他精度走标准 FP32 累加
// ============================================================================
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

    // TF32 用较低精度换吞吐，只在 FP32 编译档考虑，其他精度走标准 FP32 累加
    bool enable_tf32 = PRECISION_MODE == PRECISION_FP32 && deviceProp.major >= 8;
    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}

void common_free() {
    cudaCheck(cudaStreamDestroy(main_stream));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
}

// ============================================================================
// 模型生命周期：初始化字段 → 分配权重 → 分配激活 → 从检查点加载
// ============================================================================

// 先把句柄和档位字段清零，避免后面读到未初始化值
void gpt2_init_common(GPT2 *model) {
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->params_memory = NULL;
    model->recompute = 0;
    model->gelu_fusion = 0;
}

// 算参数尺寸、统计总字节，再一次性申请权重显存
void gpt2_allocate_weights(GPT2 *model) {
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    model->num_parameters = 0;
    model->num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_elements[i];
        model->num_parameters_bytes += model->param_elements[i] * model->param_sizeof[i];
    }
    assert(model->params_memory == nullptr);
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_elements, model->param_sizeof);
}

// 按当前 B、T 分配激活显存和输入缓冲，再打印一次显存占用
void gpt2_allocate_state(GPT2* model, int B, int T) {
    model->batch_size = B;
    model->seq_len = T;
    fill_in_activation_sizes(&model->acts, model->acts_specs, B, T, model->config, model->recompute);
    model->acts_memory = malloc_and_point_activations(model->acts_specs);
    cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
    size_t free, total;
    cudaCheck(cudaMemGetInfo(&free, &total));
    printf("显存使用情况：%zd MiB / %zd MiB\n", (total-free) / 1024 / 1024, total / 1024 / 1024);
}

// 读文件头、校验魔数/版本与精度匹配，再分配权重并流式上传
void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {
    if (PRECISION_MODE == PRECISION_FP16) {
        fprintf(stderr, "从检查点构建目前不支持 fp16。\n");
        exit(EXIT_FAILURE);
    }

    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20251103) { printf("模型文件魔数错误\n"); exit(EXIT_FAILURE); }
    int version = model_header[1];
    if (!(version == 3 || version == 5)) {
        fprintf(stderr, "模型文件版本号错误\n");
        exit(EXIT_FAILURE);
    }

    if (PRECISION_MODE == PRECISION_BF16 && version != 5) {
        fprintf(stderr, "编译精度为 BF16，但 %s 中的模型不是该精度。\n", checkpoint_path);
        fprintf(stderr, "提示：请加载 _bf16.bin 文件\n");
        exit(EXIT_FAILURE);
    }
    if (PRECISION_MODE == PRECISION_FP32 && version != 3) {
        fprintf(stderr, "编译精度为 FP32，但 %s 中的模型不是该精度。\n", checkpoint_path);
        fprintf(stderr, "提示：请加载文件名中不含 _bf16 的 .bin 文件\n");
        exit(EXIT_FAILURE);
    }

    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    gpt2_allocate_weights(model);
    file_to_device(model->params_memory, model_file, model->num_parameters_bytes, IO_BUF_SIZE, main_stream);
    fcloseCheck(model_file);

    cudaCheck(cudaDeviceSynchronize());
}


// ============================================================================
// gpt2_forward：完整前向
// ============================================================================
void gpt2_forward(GPT2 *model, const int* inputs, size_t B, size_t T) {
    NVTX_RANGE_FN();

    if (model->params_memory == NULL) {
        printf("错误：模型未正确初始化。\n");
        exit(EXIT_FAILURE);
    }

    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    if (B > (size_t)model->batch_size || T > (size_t)model->seq_len) {
        printf("模型初始：B=%d T=%d，请求：B=%zu T=%zu\n", model->batch_size, model->seq_len, B, T);
        exit(EXIT_FAILURE);
    }

    // 输入 token 上传到设备
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(inputs, B*T, V);

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;

    // 嵌入层：token 编号变向量
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, main_stream);

    // 第 0 层的 LN1 在循环外先算，后续层的 LN1 由上一层末尾的融合 kernel 顺带完成
    layernorm_forward(acts.ln1, acts.ln1_mean, acts.ln1_rstd, acts.encoded, params.ln1w, params.ln1b, B, T, C, main_stream);

    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);

        // 本层的残差输入：第 0 层是嵌入输出，其余是上一层末尾的残差
        floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // 本层各权重/激活的起始地址：按层偏移
        floatX* l_qkvw = params.qkvw + l * 3*C * C;  // 本层 QKV 权重
        floatX* l_qkvb = params.qkvb + l * 3*C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_attprojb = params.attprojb + l * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcb = params.fcb + l * 4*C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        floatX* l_fcprojb = params.fcprojb + l * C;

        floatX* l_ln1 = acts.ln1 + l * B * T * C;  // 本层激活起始地址
        floatX* l_qkv = acts.qkv + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch = acts.fch + l * B * T * 4*C;
        floatX* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        floatX* l_residual3 = acts.residual3 + l * B * T * C;
        floatX* l_att = acts.att + l * B * NH * T * T;
        floatX* scratch = (floatX*)acts.output;

        // 短于最大长度时先清零注意力缓冲，避免上一轮残留参与计算
        if (T != (size_t)model->seq_len) {
            cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        }
        // 子层 1：QKV 投影 → 完整注意力（无缓存）→ 输出投影
        matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward(l_atty, l_qkv, l_att, scratch, B, T, C, NH, main_stream);

        // 残差 + LN2 融合：一个 kernel 同时完成加残差和归一化
        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);
        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd, residual, scratch, l_ln2w, l_ln2b, B*T, C, main_stream);

        // 子层 2：MLP 两层（第一层可带 GELU 融合）
        matmul_forward_cublaslt(l_fch_gelu, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, main_stream, l_fch, model->gelu_fusion);
        matmul_forward_cublaslt(scratch, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, main_stream);

        // 层末尾：残差 + 下一层 LN1（最后一层则是最终 LN），为下一层备好输入
        if(l+1 != L) {
            floatX* nl_ln1 = acts.ln1 + (l + 1) * B * T * C;
            float* nl_ln1_mean = acts.ln1_mean + (l + 1) * B * T;
            float* nl_ln1_rstd = acts.ln1_rstd + (l + 1) * B * T;
            const floatX* nl_ln1w = params.ln1w + (l + 1) * C;
            const floatX* nl_ln1b = params.ln1b + (l + 1) * C;
            fused_residual_forward5(l_residual3, nl_ln1, nl_ln1_mean, nl_ln1_rstd, l_residual2, scratch, nl_ln1w, nl_ln1b,
                                   B * T, C, main_stream);
        } else {
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd, l_residual2, scratch,
                                   params.lnfw, params.lnfb, B * T, C, main_stream);
        }
    }

    // 词表投影：权重直接复用词嵌入表（权重绑定，weight tying），省一份参数
    matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);
    cudaCheck(cudaDeviceSynchronize());
}


// ============================================================================
// 模型释放：归还权重、激活两块显存和输入缓冲
// ============================================================================
void gpt2_free(GPT2 *model) {
    cudaFreeCheck(&model->params_memory);
    cudaFreeCheck(&model->acts_memory);
    cudaCheck(cudaFree(model->inputs));
}

// ============================================================================
// 生成循环：每步全量前向 → 拷回 logits → CPU 采样
// ============================================================================
static double run_generation_once(GPT2* model, Tokenizer* tokenizer, int* gen_tokens,
                                  floatX* cpu_logits_raw, float* cpu_logits,
                                  int genT, int eot_token, bool quiet) {
    for (int i = 0; i < model->config.max_seq_len; i++) {
        gen_tokens[i] = eot_token;
    }

    if (!quiet) printf("开始生成：\n---\n");

    cudaEvent_t gen_start, gen_stop;
    cudaCheck(cudaEventCreate(&gen_start));
    cudaCheck(cudaEventCreate(&gen_stop));
    cudaCheck(cudaEventRecord(gen_start, main_stream));
    auto wall_start = std::chrono::steady_clock::now();
    unsigned long long sample_rng_state = 1337;

    int gen_limit = min(genT - 1, model->config.max_seq_len);
    for (int t = 1; t <= gen_limit; t++) {
        // 按 256 向上取整：前向长度分段增长，避免每步都按最大长度重算
        int chunk = min(model->config.max_seq_len, 256);
        int forwardT = min(model->config.max_seq_len, CEIL_DIV(t, chunk) * chunk);  // 本步前向长度
        gpt2_forward(model, gen_tokens, 1, forwardT);

        // 取当前位置（第 t-1 行）的 logits，其余行是历史不需要
        floatX* logits = model->acts.output + (t - 1) * model->config.padded_vocab_size;

        // 当前行 logits 拷回主机
        cudaCheck(cudaMemcpy(cpu_logits_raw, logits, model->config.vocab_size * sizeof(floatX), cudaMemcpyDeviceToHost));

        // 第 1 步额外落盘一份 t1 logits，供精度验证使用
        if (t == 1) {
            for (int i = 0; i < model->config.vocab_size; i++) {
                cpu_logits[i] = (float)cpu_logits_raw[i];
            }
            mkdir("assets", 0755); mkdir("assets/dumps", 0755);
            FILE* f = fopen("assets/dumps/cuda_logits_t1.bin", "wb");
            if (f) { fwrite(cpu_logits, sizeof(float), model->config.vocab_size, f); fclose(f); }
        }

        for (int i = 0; i < model->config.vocab_size; i++) {
            cpu_logits[i] = (float)cpu_logits_raw[i];
        }

        // CPU 采样：随机数种子固定，结果可复现
        float coin = random_f32(&sample_rng_state);  // 0~1 随机数
        int next_token = sample_softmax(cpu_logits, model->config.vocab_size, coin);  // 按概率抽 token
        gen_tokens[t] = next_token;  // 追加到生成序列，下一步作为输入
        if (!quiet) {
            if (tokenizer->init_ok) {
                const char* token_str = tokenizer_decode(tokenizer, next_token);
                safe_printf(token_str);
            } else {
                printf("%d ", next_token);
            }
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
        int gen_count = gen_limit;
        printf("\n[性能]\n");
        printf("生成 %d 个 token：\n", gen_count);
        printf("  GPU 耗时：%.1f ms（平均 %.2f ms/token）\n", gpu_ms, gpu_ms / gen_count);
        printf("  墙钟耗时：%.1f ms（平均 %.2f ms/token）\n", wall_ms, wall_ms / gen_count);
    }
    return wall_ms;
}


// ============================================================================
// main：解析参数 → 构建模型 → 跑生成循环（普通/bench 两档）→ 清理退出
// ============================================================================
int main(int argc, char *argv[]) {
    const char* model_path = "assets/weights/gpt2_124M.bin";
    const char* tokenizer_path = "assets/weights/gpt2_tokenizer.bin";
    int genT = 1001;
    bool bench_mode = false;
    int bench_runs = 6;
    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) {
            bench_mode = true;
            // 后面若跟了数字就是次数
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                int r = atoi(argv[i + 1]);
                if (r > 1) { bench_runs = r; i++; }
            }
        } else if (positional == 0) {
            model_path = argv[i]; positional++;  // 第一个位置参数：模型路径
        } else if (positional == 1) {
            tokenizer_path = argv[i]; positional++;  // 第二个：分词器路径
        } else {
            genT = atoi(argv[i]); positional++;  // 第三个：生成长度
        }
    }

    common_start();

    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, model_path);

    int B = 1;
    int T = model.config.max_seq_len;
    gpt2_allocate_state(&model, B, T);

    printf("[GPT-2 模型]\n");
    printf("最大序列长度：%d\n", model.config.max_seq_len);
    printf("词表大小：%d\n", model.config.vocab_size);
    printf("层数：%d\n", model.config.num_layers);
    printf("注意力头数：%d\n", model.config.num_heads);
    printf("通道数：%d\n", model.config.channels);
    printf("参数量：%zu\n", model.num_parameters);
    printf("生成模式：全量重算（每步对整个序列重做前向，无 KV Cache）\n");

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, tokenizer_path);

    int* gen_tokens = (int*)mallocCheck(B * (T + 1) * sizeof(int));
    floatX* cpu_logits_raw = (floatX*)mallocCheck(model.config.vocab_size * sizeof(floatX));
    float* cpu_logits = (float*)mallocCheck(model.config.vocab_size * sizeof(float));

    int eot_token = tokenizer.eot_token;
    for (int i = 0; i < B * T; i++) {
        gen_tokens[i] = eot_token;
    }

    if (bench_mode) {
        // 第 1 次当作预热丢弃，后面的墙钟时间排序取中位数，抗偶发抖动
        std::vector<double> wall_times;
        for (int r = 0; r < bench_runs; r++) {
            double w = run_generation_once(&model, &tokenizer, gen_tokens, cpu_logits_raw,
                                           cpu_logits, genT, eot_token, true);
            if (r >= 1) wall_times.push_back(w);
            printf("  第 %d/%d 次：%.1f ms\n", r + 1, bench_runs, w);
        }
        std::sort(wall_times.begin(), wall_times.end());
        double median = wall_times[wall_times.size() / 2];
        int gen_count = min(genT - 1, model.config.max_seq_len);
        printf("\n[性能汇总]\n");
        printf("生成 %d 个 token（丢弃第 1 次，取 %d 次中位数）：\n", gen_count, (int)wall_times.size());
        printf("  中位数：%.1f ms（平均 %.2f ms/token）\n", median, median / gen_count);
    } else {
        run_generation_once(&model, &tokenizer, gen_tokens, cpu_logits_raw, cpu_logits,
                            genT, eot_token, false);
    }

    free(cpu_logits_raw);
    free(cpu_logits);
    free(gen_tokens);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    common_free();
    return 0;
}
