/* ============================================================================
   encoder.cuh —— 嵌入层：把 token 编号变成模型内部的向量
   ============================================================================
   GPT-2 的输入不是文字，而是一连串整数 token 编号。嵌入层做两件事：
     1. 词嵌入（token embedding，WTE）：按 token 编号从一张大表里查出该 token 对应的向量，同编号永远查到同一向量；
     2. 位置嵌入（position embedding，WPE）：按 token 在句子中的位置查出一个位置向量，让模型知道谁在前、谁在后。
   两个向量逐元素相加，就是后续网络真正的输入。

   本文件提供两个入口，对应推理的两个阶段：
     encoder_forward     预填充（prefill）：一次处理一整段 B*T 个 token；
     encoder_forward_kv  逐 token 解码（decode）：每个请求只处理当前这一个新 token，位置从设备端的 g_cur_len 读取。
   解码入口的位置不写死在参数里，是因为推理会被 CUDA Graph 捕获重放，重放时主机参数不变，只能让 kernel 自己去读设备上随步数更新的位置。

   访存方式：每个线程通过 x128 辅助类型一次处理 128 bit（一整组向量），合并加载、合并写回，而非逐元素搬运。
   ============================================================================ */
#pragma once

#include <assert.h>
#include <stdint.h>
#include "common.h"

// ============================================================================
// 预填充路径：一次处理完整序列
// ============================================================================

// 每个线程负责一个 128-bit 小块。先把扁平下标拆成 (batch, 位置, 通道)，再分别查词嵌入和位置嵌入，相加后整块写回。
__global__ void encoder_forward_kernel3(floatX* out, const int* inp, const floatX* wte, const floatX* wpe, int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;  // 本线程负责的起始通道下标
    int N = B * T * C;
    if (idx >= N) { return; }  // 越界退出

    int bt = idx / C;  // 扁平下标 → 第几个序列位置
    int b = bt / T;    // 第几个 batch
    int t = bt % T;    // 序列内位置
    int c = idx % C;   // 通道内偏移

    int ix = inp[b * T + t];  // 该位置上的 token 编号，词嵌入表的行号
    floatX* out_btc = out + b * T * C + t * C + c;
    const floatX* wte_ix = wte + ix * C + c;   // 词嵌入：按 token 编号取行
    const floatX* wpe_tc = wpe + t * C + c;    // 位置嵌入：按位置取行

    x128 packed_out;
    x128 wte128 = load128cs(wte_ix);  // 128-bit 合并加载词嵌入
    x128 wpe128 = load128cs(wpe_tc);  // 128-bit 合并加载位置嵌入
    for (int k = 0; k < x128::size; k++) {
        packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);  // 词嵌入 + 位置嵌入
    }
    store128(out_btc, packed_out);  // 128-bit 合并写回
}

void encoder_forward(floatX* out, const int* inp, const floatX* wte, const floatX* wpe, int B, int T, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int N = B * T * C;
    // 每个线程处理 x128::size 个元素，网格按此换算
    const int grid_size = CEIL_DIV(N, (int)(block_size * x128::size));
    encoder_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// 解码路径：每步只处理当前一个新 token
// ============================================================================

/* 与预填充 kernel 的计算完全相同，区别只在位置来源：位置不是下标 t，而是去设备内存读 g_cur_len，保证 CUDA Graph 重放时也能拿到当前步数。
   二维网格的 y 维区分 batch 中的不同请求。 */
__global__ void encoder_forward_kv_kernel(floatX* out, const int* inp,
                                          const floatX* wte, const floatX* wpe,
                                          const int* g_cur_len_ptr,
                                          int B, int C) {
    const int position = *g_cur_len_ptr;  // 设备端当前长度，随每个解码步更新
    int b = blockIdx.y;  // 二维网格 y 维区分 batch
    if (b >= B) return;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= C) return;

    int ix = inp[b];  // 解码步每请求只有一个 token
    x128 wte128 = load128cs(wte + ix * C + idx);
    x128 wpe128 = load128cs(wpe + position * C + idx);  // 位置用 g_cur_len，不是下标
    x128 packed;
    for (int k = 0; k < x128::size; k++) {
        packed[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
    }
    store128(out + b * C + idx, packed);
}

void encoder_forward_kv(floatX* out, const int* inp, const floatX* wte, const floatX* wpe,
                        const int* g_cur_len_ptr, int B, int C, cudaStream_t stream) {
    const int block = 256;
    const int grid_x = CEIL_DIV(C, block * (int)x128::size);
    dim3 grid(grid_x, B);  // x 维覆盖通道，y 维覆盖 batch
    encoder_forward_kv_kernel<<<grid, block, 0, stream>>>(out, inp, wte, wpe, g_cur_len_ptr, B, C);
    cudaCheck(cudaGetLastError());
}
