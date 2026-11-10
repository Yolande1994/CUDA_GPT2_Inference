/* ============================================================================
   tokenizer_sampler.h —— GPT-2 分词器与采样工具
   ============================================================================

   说明：
     本文件包含两类工具：
       1. 分词器（tokenizer）：仅支持解码（decode），把 token（整数）转换为字符串。
          若以后要支持输入提示（prompt），则需要增加编码（encode）.
          由于涉及正则表达式，在纯 C 中实现会比较麻烦，留待以后研究。
       2. 采样器（sampler）：根据 logits 的概率分布采样出下一个 token，
          使用简单的 xorshift 伪随机数生成器。
   ============================================================================ */

#pragma once

#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
// 自有工具：fopenCheck、freadCheck、fcloseCheck、fseekCheck、mallocCheck
#include "utils.h"

// ============================================================================
// 分词器
// ============================================================================

/* Tokenizer —— 分词器结构体：
   vocab_size  词表大小；
   token_table 每个 token 对应的字符串；
   init_ok     是否成功初始化；
   eot_token   文本结束符 <|endoftext|> 的 id。 */
typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token; // <|endoftext|> 的 token id
} Tokenizer;

/* safe_printf —— 安全打印 token：
   token 是原始字节，其中可能包含控制符、退格等不可打印字节，只打印可打印或空白字符，避免输出乱码。 */
void safe_printf(const char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // 处理单字节 token（每个 token 至少一个字节，piece[1] 可安全访问）
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // 异常字节，不打印
        }
    }
    printf("%s", piece);
}

/* tokenizer_init —— 从分词器文件加载词表
   文件结构：256 个 uint32 的头（魔数、版本、词表大小、EOT id），随后逐个 token：先 1 字节长度，再对应长度的字节内容。 */
void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        // 分词器文件缺失时给出提示（不致命，后续按 id 打印）
        printf("---\n");
        printf("警告：无法打开分词器文件 %s\n", filename);
        printf("重新运行 python export_weights.py 以生成该文件\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }
    // 读取文件头
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20251103);
    int version = header[1];
    tokenizer->vocab_size = header[2];
    if (version == 1) {
        // 版本 1 不包含 EOT id，按 GPT-2 约定取 50256
        assert(tokenizer->vocab_size == 50257); // 防御性检查
        tokenizer->eot_token = 50256;
    } else if (version == 2) {
        tokenizer->eot_token = header[3];
    } else {
        fprintf(stderr, "分词器模型文件 %s 版本号错误：%d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // 读取全部 token
    unsigned char length;
    tokenizer->token_table = (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // 每个 token 至少一个字符
        char *token_bytes = (char *)mallocCheck(length + 1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // 加字符串结束符，便于打印
        tokenizer->token_table[i] = token_bytes;
    }
    // 清理
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

// tokenizer_decode —— 把 token id 解码为字符串；id 越界或未初始化时返回 NULL。
const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        printf("无效的 token id：%u\n", token_id);
        return NULL;
    }
}

// tokenizer_free —— 释放词表占用的内存
void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}


// ============================================================================
// 采样工具
// ============================================================================

// ----------------------------------------------------------------------------
// xorshift 随机数生成器
// ----------------------------------------------------------------------------

/* random_u32 —— 简单的 xorshift* 伪随机数生成器，返回一个 32 位随机整数；
   通过对状态做固定的移位异或混合，速度快、周期长。
   参考：https://en.wikipedia.org/wiki/Xorshift#xorshift.2A */
unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

/* random_f32 —— 返回 [0, 1) 区间的随机 float；
   取随机整数的高 24 位除以 2^24，保证均匀且精度足够。 */
float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

// ----------------------------------------------------------------------------
// 按 softmax 概率采样
// ----------------------------------------------------------------------------

/* ----------------------------------------------------------------------------
   sample_softmax —— 从 logits 经 softmax 得到的概率分布中采样一个下标
   ----------------------------------------------------------------------------
   原理：把 logits 转成概率后做轮盘赌采样（累积分布 CDF）。
   参数：
     logits 未归一化的打分结果；
     n      候选个数；
     coin   [0,1) 的随机数。
   做法：
     1. 先求所有 exp(logits) 的总和 norm（即 softmax 的分母）；
     2. 不逐个除以 norm，而是直接把 coin 乘以 norm，等价于在概率空间采样；
     3. 累加 exp(logits) 作为 CDF，当 coin 落入对应区间即返回该下标。
   返回：采样到的下标；末尾兜底返回 n-1，防止浮点误差导致未命中。
---------------------------------------------------------------------------- */
int sample_softmax(const float* logits, int n, float coin) {
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // 直接把 coin 乘以总和，等价于用归一化概率采样
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // 应对浮点舍入误差的兜底
}
