/* ============================================================================
   dump_utils.h —— 调试转储工具：把中间结果落盘，供离线精度分析
   ============================================================================
   推理过程中 GPU 上的数据看不见，这个文件负责在指定步骤把它们拷回主机、写成二进制文件，verify_precision.py 再读这些文件与参考实现逐项对比。
   提供三类功能：
     1. 转储指定步骤的 logits、以及当时的输入 token 前缀；
     2. 转储第 1 步的基准 logits（整套精度验证的锚点）；
     3. 逐步记录真实 token 的负对数似然，最终汇总困惑度 PPL。
   ============================================================================ */
#ifndef DUMP_UTILS_H
#define DUMP_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>
#include <string>
#include <vector>

#include "common.h"
#include "gpt2_model.h"

// 需要转储的步骤列表；只在这些步骤落盘，避免每步都写文件
static std::vector<int> g_dump_steps;

// 用户指定的文件名前缀，空串表示用带精度名的默认前缀
static std::string g_dump_prefix = "";

// 按编译精度选择文件名后缀，保证不同精度的 dump 不会互相覆盖
#if defined(USE_INT8)
#define DUMP_PRECISION_SUFFIX "int8"
#elif defined(USE_FP8)
#define DUMP_PRECISION_SUFFIX "fp8"
#elif defined(ENABLE_FP32)
#define DUMP_PRECISION_SUFFIX "fp32"
#else
#define DUMP_PRECISION_SUFFIX "bf16"
#endif
#define DEFAULT_DUMP_PREFIX "opt_" DUMP_PRECISION_SUFFIX

static const char* dump_prefix(const char* def) {
    return g_dump_prefix.empty() ? def : g_dump_prefix.c_str();
}

// 判断当前步骤是否在转储列表里
static bool want_dump_step(int step) {
    return std::find(g_dump_steps.begin(), g_dump_steps.end(), step) != g_dump_steps.end();
}

static void mkdir_p(const char* dir);
static int g_ppl_len = 0;
static std::vector<float> g_ppl_nlls;
static std::vector<int> g_ppl_toks;

// ============================================================================
// PPL 记录：计算“真实下一个 token”的负对数似然（NLL）
// ============================================================================

/* 把 logits 拷回主机，用 log-sum-exp 稳定地算出归一化分母，再取真实 token 的概率：NLL = -log P(真实token) = logΣexp(logits) − logits[token]。
   先减最大值再取指数，防止 exp 溢出。 */
static void ppl_record_nll(GPT2* model, int next_token) {
    float* h_logits = (float*)malloc(model->config.vocab_size * sizeof(float));
    // 全精度 logits 拷回主机
    cudaCheck(cudaMemcpy(h_logits, model->acts.logits,
                         model->config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    float mx = h_logits[0];
    // 先找词表上的最大值，下面减掉它防溢出
    for (int i = 1; i < model->config.vocab_size; i++) mx = fmaxf(mx, h_logits[i]);
    double s = 0.0;
    // log-sum-exp：减最大值后求指数和
    for (int i = 0; i < model->config.vocab_size; i++) s += exp((double)h_logits[i] - (double)mx);
    // NLL = log 归一化分母 − 真实 token 的打分
    float nll = (float)((double)mx + log(s) - (double)h_logits[next_token]);
    g_ppl_nlls.push_back(nll);     // 记录本步 NLL
    g_ppl_toks.push_back(next_token);
    free(h_logits);
}

// 写出全部 NLL 和 token，并打印平均 NLL 与 PPL（PPL = exp(平均NLL)）
static void ppl_write(const char* prefix) {
    if (g_ppl_nlls.empty()) return;
    mkdir_p("assets/dumps");
    char path[256];
    snprintf(path, sizeof(path), "assets/dumps/ppl_%s_nll.bin", dump_prefix(prefix));
    FILE* f = fopen(path, "wb");
    fwrite(g_ppl_nlls.data(), sizeof(float), g_ppl_nlls.size(), f);  // 逐 NLL 落盘
    fclose(f);
    snprintf(path, sizeof(path), "assets/dumps/ppl_%s_tokens.bin", dump_prefix(prefix));
    f = fopen(path, "wb");
    fwrite(g_ppl_toks.data(), sizeof(int), g_ppl_toks.size(), f);  // 对应真实 token 落盘
    fclose(f);
    double sum = 0.0;
    for (float v : g_ppl_nlls) sum += v;  // 求平均 NLL
    // PPL 直观理解：模型在“多少个候选词之间犹豫”，越接近 1 越确定
    printf("[PPL] %zu 步：平均 NLL=%.4f，困惑度 PPL=%.4f（越小越好；HF 参考见 verify_precision.py --ppl）\n",
           g_ppl_nlls.size(), sum / g_ppl_nlls.size(), exp(sum / g_ppl_nlls.size()));
}

// 逐级创建目录：逐段把路径截到每个 / 处调用 mkdir，已存在不报错
static void mkdir_p(const char* dir) {
    char tmp[256];
    snprintf(tmp, sizeof(tmp), "%s", dir);
    // 从第二个字符开始扫描路径
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';          // 临时截断到当前 /
            mkdir(tmp, 0755);  // 创建这一级，已存在不报错
            *p = '/';          // 恢复 / 继续
        }
    }
    mkdir(tmp, 0755);  // 最后一级目录
}

// 转储指定步骤的完整 logits
static void dump_logits_step(GPT2* model, const char* prefix, int step) {
    float* h_logits = (float*)malloc(model->config.vocab_size * sizeof(float));
    cudaCheck(cudaMemcpy(h_logits, model->acts.logits,
                         model->config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    char path[256];
    mkdir_p("assets/dumps");
    snprintf(path, sizeof(path), "assets/dumps/%s_logits_step%d.bin", dump_prefix(prefix), step);
    FILE* f = fopen(path, "wb");
    fwrite(h_logits, sizeof(float), model->config.vocab_size, f);  // 完整 logits 二进制落盘
    fclose(f);
    free(h_logits);
}

// 转储到当前步骤为止的输入 token 前缀（验证时需要知道每步喂了什么）
static void dump_inputs_step(const int* gen_tokens, const char* prefix, int step) {
    char path[256];
    snprintf(path, sizeof(path), "assets/dumps/%s_inputs_step%d.bin", dump_prefix(prefix), step);
    FILE* f = fopen(path, "wb");
    // 逐个写入到当前步为止的输入 token
    for (int t = 0; t < step; t++) {
        int tok = gen_tokens[t];
        fwrite(&tok, sizeof(int), 1, f);
    }
    fclose(f);
}

// 转储第 1 步基准 logits：固定前缀，作为整套精度验证的对照锚点
static void dump_logits_t1(GPT2* model) {
    float* h_logits = (float*)malloc(model->config.vocab_size * sizeof(float));
    cudaCheck(cudaMemcpy(h_logits, model->acts.logits,
                         model->config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    mkdir_p("assets/dumps");
    char path[256];
    snprintf(path, sizeof(path), "assets/dumps/%s_logits_t1.bin", dump_prefix(DEFAULT_DUMP_PREFIX));
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(h_logits, sizeof(float), model->config.vocab_size, f); fclose(f); }
    free(h_logits);
}

#endif
