/* ============================================================================
   gpt2_model.h —— GPT-2 的模型蓝图与显存管理
   ============================================================================
   包含四块内容：
     GPT2Config         模型尺寸（层数、头数、通道数、词表大小等）；
     ParameterTensors   16 个“参数”指针：训练好的权重和偏置，推理全程只读，数量与输入无关；
     ActivationTensors  22 个“激活值”指针：前向过程中产生的中间结果，大小随 batch 和序列长度变化；
     GPT2               顶层结构，把配置、参数、激活、显存句柄全部打包。

   显存策略：一次 cudaMalloc 申请一整块连续显存，再把 16/22 个指针依次指向块内的对应位置，而不是每个张量单独申请。
   好处是分配次数少、地址连续、释放时只还一块；激活块内每个张量按 16 字节对齐。

   精度说明：绝大多数张量是 floatX（随编译精度切换 BF16/FP32），但 logits 固定为 float：词表投影后的打分要保持全精度，采样和数值稳定都依赖它。
   ============================================================================ */
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string>
#include "common.h"
#include "utils.h"

extern cudaDeviceProp deviceProp;

// 模型尺寸：全部从权重文件头读取，不写死
typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int channels;
} GPT2Config;

// ============================================================================
// 参数（权重）：16 个命名指针
// ============================================================================
constexpr int NUM_PARAMETER_TENSORS = 16;
typedef struct {
    floatX* wte;       // 词嵌入表
    floatX* wpe;       // 位置嵌入表
    floatX* ln1w;      // 各层 LN1 权重/偏置
    floatX* ln1b;
    floatX* qkvw;      // 各层 QKV 投影权重/偏置
    floatX* qkvb;
    floatX* attprojw;  // 各层注意力输出投影权重/偏置
    floatX* attprojb;
    floatX* ln2w;      // 各层 LN2 权重/偏置
    floatX* ln2b;
    floatX* fcw;       // 各层 MLP 第一层权重/偏置
    floatX* fcb;
    floatX* fcprojw;   // 各层 MLP 第二层权重/偏置
    floatX* fcprojb;
    floatX* lnfw;      // 最终 LayerNorm 权重/偏置
    floatX* lnfb;
} ParameterTensors;
// 编译期兜底：指针数必须正好 16，增删字段时这里会立刻报错
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

// 每个参数张量的元素个数：带 L 的是每层一份、共 L 层
void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;  // 别名，下面公式更短
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C;               // WTE 词嵌入表
    param_sizes[1] = maxT * C;             // WPE 位置嵌入表
    param_sizes[2] = L * C;                // LN1 权重
    param_sizes[3] = L * C;                // LN1 偏置
    param_sizes[4] = L * (3 * C) * C;      // QKV 投影权重，3C 是拼在一起的 Q/K/V
    param_sizes[5] = L * (3 * C);          // QKV 偏置
    param_sizes[6] = L * C * C;            // 注意力输出投影权重
    param_sizes[7] = L * C;
    param_sizes[8] = L * C;                // LN2 权重/偏置
    param_sizes[9] = L * C;
    param_sizes[10] = L * (4 * C) * C;     // MLP 第一层，4C 是中间层宽度
    param_sizes[11] = L * (4 * C);
    param_sizes[12] = L * C * (4 * C);     // MLP 第二层
    param_sizes[13] = L * C;
    param_sizes[14] = C;                   // 最终 LN 只有一份
    param_sizes[15] = C;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);  // 参数全部按 floatX 存储
    }
}

// 申请一整块参数显存，再把 16 个命名指针依次切到块内位置
void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_elements, size_t *param_sizeof) {
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters_bytes += param_elements[i] * param_sizeof[i];  // 累加总字节数
    }
    void* params_memory;
    cudaCheck(cudaMalloc((void**)&params_memory, num_parameters_bytes));  // 一次申请整块
    // 指针列表顺序必须与 fill_in_parameter_sizes 的下标顺序一一对应
    floatX** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    char* it = (char*)params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)it;  // 切块：指针落在当前游标，再推进该张量的字节数
        it += param_elements[i] * param_sizeof[i];
    }
    return params_memory;
}

// ============================================================================
// 激活值（中间结果）：22 个指针，含混合精度
// ============================================================================
constexpr int NUM_ACTIVATION_TENSORS = 22;
typedef struct {
    floatX* encoded;       // 嵌入层输出
    floatX* ln1;           // LN1 输出
    float* ln1_mean;       // LN1 的均值/标准差（float）
    float* ln1_rstd;
    floatX* atty;          // 注意力输出
    floatX* att;           // 注意力打分
    floatX* residual2;     // 残差
    floatX* ln2;           // LN2 输出
    float* ln2_mean;       // LN2 的均值/标准差
    float* ln2_rstd;
    floatX* fch;           // MLP 第一层输出
    floatX* fch_gelu;      // GELU 后输出
    floatX* residual3;     // 残差
    floatX* lnf;           // 最终 LN 输出
    float* lnf_mean;       // 最终 LN 的均值/标准差
    float* lnf_rstd;
    float* losses;
    floatX* qkv;           // QKV 投影输出
    floatX* output;
    float* logits;         // 词表打分，固定 float 全精度

    floatX* scratch_bt4c;  // 通用临时缓冲（4 倍宽）
    floatX* scratch_btc;   // 通用临时缓冲
} ActivationTensors;

// 通用张量描述：指针 + 元素数 + 数据类型，让分配逻辑可以统一处理混合精度
struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};
#define TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)};

// 每个激活张量的元素数；recompute 档位可把部分张量大小置 0 以省显存
void fill_in_activation_sizes(const ActivationTensors* data, TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS], size_t B, size_t T, GPT2Config config, int recompute) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    tensors[0] = TENSOR_SPEC(data->encoded, B * T * C);
    tensors[1] = TENSOR_SPEC(data->ln1,  (recompute < 2) ? L * B * T * C : 0);  // 高档重算时不存 LN1
    tensors[2] = TENSOR_SPEC(data->ln1_mean, L * B * T);
    tensors[3] = TENSOR_SPEC(data->ln1_rstd, L * B * T);
    tensors[4] = TENSOR_SPEC(data->atty, L * B * T * C);
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * T * T);
    tensors[6] = TENSOR_SPEC(data->residual2, L * B * T * C);
    tensors[7] = TENSOR_SPEC(data->ln2,  (recompute < 2) ? L * B * T * C : 0);
    tensors[8] = TENSOR_SPEC(data->ln2_mean, L * B * T);
    tensors[9] = TENSOR_SPEC(data->ln2_rstd, L * B * T);
    tensors[10] = TENSOR_SPEC(data->fch, L * B * T * 4*C);
    tensors[11] = TENSOR_SPEC(data->fch_gelu, (recompute < 1) ? L * B * T * 4*C : B * T * 4*C);  // 融合后只留一份
    tensors[12] = TENSOR_SPEC(data->residual3, L * B * T * C);
    tensors[13] = TENSOR_SPEC(data->lnf, B * T * C);
    tensors[14] = TENSOR_SPEC(data->lnf_mean, B * T);
    tensors[15] = TENSOR_SPEC(data->lnf_rstd, B * T);
    tensors[16] = TENSOR_SPEC(data->losses, B * T);
    tensors[17] = TENSOR_SPEC(data->qkv, L * B * T * 3*C);
    // output 要兼容三种最大用途，取其中最大者
    tensors[18] = TENSOR_SPEC(data->output, B * T * max(3*C, max(NH*T, Vp)));
    tensors[19] = TENSOR_SPEC(data->scratch_bt4c, B * T * 4 * C);
    tensors[20] = TENSOR_SPEC(data->scratch_btc, B * T * C);

    tensors[21] = TENSOR_SPEC(data->logits, B * T * Vp);  // 全精度词表打分
}

// 申请一整块激活显存：16 字节对齐切块，零尺寸张量给 NULL
void* malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]) {
    constexpr size_t ALIGN = 16;
    auto align_up = [](size_t x) { return (x + ALIGN - 1) / ALIGN * ALIGN; };  // 向上对齐到 16 倍数

    size_t bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        if (tensors[i].size == 0) continue;
        bytes = align_up(bytes);   // 先对齐再累加
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);
    }
    bytes = align_up(bytes);
    printf("正在为激活值分配 %d MiB 显存\n", (int)round(bytes / (1024 * 1024)));
    void* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, bytes));
    cudaCheck(cudaMemset(acts_memory, 0, bytes));  // 清零，避免读到未初始化数据
    char* it = (char*)acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        if (tensors[i].size == 0) {
            *(tensors[i].ptr) = NULL;  // 重算档位不分配，指针置空
        } else {
            it = (char*)align_up((size_t)it);  // 游标先对齐
            *(tensors[i].ptr) = it;
            it += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }
    return acts_memory;
}

// ============================================================================
// 顶层 GPT2 结构与生命周期函数
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

void gpt2_init_common(GPT2 *model) {
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->params_memory = NULL;
    model->recompute = 0;
    model->gelu_fusion = 1;
}

// 计算参数尺寸并一次性分配权重显存
void gpt2_allocate_weights(GPT2 *model) {
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    model->num_parameters = 0;
    model->num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_elements[i];  // 元素总数
        model->num_parameters_bytes += model->param_elements[i] * model->param_sizeof[i];  // 字节总数
    }
    assert(model->params_memory == nullptr);
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_elements, model->param_sizeof);
}

// 按当前 B、T 分配激活显存和输入缓冲
void gpt2_allocate_state(GPT2 *model, int B, int T) {
    model->batch_size = B;
    model->seq_len = T;
    fill_in_activation_sizes(&model->acts, model->acts_specs, B, T, model->config, model->recompute);
    model->acts_memory = malloc_and_point_activations(model->acts_specs);
    cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
    size_t free, total;
    cudaCheck(cudaMemGetInfo(&free, &total));
    printf("显存使用情况：%zd MiB / %zd MiB\n", (total-free) / 1024 / 1024, total / 1024 / 1024);
}

/* 从权重文件构建模型：先读 256 整数的文件头，校验魔数和版本，再从文件头解析模型尺寸，分配权重后把参数一次性流式上传。
   版本号同时标识精度，必须与编译精度匹配，否则读出来的字节会被按错误格式解释。 */
void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {
    if (PRECISION_MODE == PRECISION_FP16) {
        fprintf(stderr, "从检查点构建目前不支持 fp16。\n");
        exit(EXIT_FAILURE);
    }
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);  // 固定 256 整数文件头
    if (model_header[0] != 20251103) { printf("模型文件魔数错误\n"); exit(EXIT_FAILURE); }
    int version = model_header[1];
    if (!(version == 3 || version == 5)) {
        fprintf(stderr, "模型文件版本号错误\n");
        exit(EXIT_FAILURE);
    }
    // 版本 3 = FP32、版本 5 = BF16 系，与编译精度逐一核对
    if (PRECISION_MODE == PRECISION_BF16 && version != 5) {
        fprintf(stderr, "编译精度为 BF16，但 %s 中的模型不是该精度。\n", checkpoint_path);
        exit(EXIT_FAILURE);
    }
    if (PRECISION_MODE == PRECISION_FP32 && version != 3) {
        fprintf(stderr, "编译精度为 FP32，但 %s 中的模型不是该精度。\n", checkpoint_path);
        exit(EXIT_FAILURE);
    }
    model->config.max_seq_len = model_header[2];  // 从文件头逐项还原模型尺寸
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];
    gpt2_allocate_weights(model);

    // 分块流式上传（每块 32 MB），避免一次性占过多主机内存
    file_to_device(model->params_memory, model_file, model->num_parameters_bytes, 32*1024*1024, 0);
    fcloseCheck(model_file);
    cudaCheck(cudaDeviceSynchronize());
}

// 释放三块显存；内部调用带重复释放保护
void gpt2_free(GPT2 *model) {
    cudaFreeCheck(&model->params_memory);
    cudaFreeCheck(&model->acts_memory);
    cudaFreeCheck(&model->inputs);
}
