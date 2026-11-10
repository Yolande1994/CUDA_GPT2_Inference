/* ============================================================================
   gelu.cuh —— GELU（高斯误差线性单元）激活函数
   ============================================================================

   算子说明：
     GELU 是一种平滑的非线性激活函数，GPT-2 使用其 tanh 近似版本：
       GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
     相比 ReLU，它在 0 附近平滑可导，训练更稳定。
   ============================================================================ */
#pragma once

#include <assert.h>
// llm 内部依赖
#include "common.h"

// ============================================================================
// CUDA Kernel
// ============================================================================

// GELU 公式中的缩放因子 sqrt(2/π)
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

/* ----------------------------------------------------------------------------
   gelu_forward_kernel2 —— 128 位向量化 GELU kernel
   ----------------------------------------------------------------------------
   每个线程负责一个数据包（x128），流式加载输入（只读一次，不留在缓存），逐元素计算 GELU 后写回。
   写回用普通 store128 而非流式 store128cs：GELU 输出紧接着会被下一个算子读取，保留在缓存中可能更有利。
---------------------------------------------------------------------------- */
__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx); // 流式加载，不保留在缓存
    for(int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        float cube = 0.044715f * xi * xi * xi; // x^3 修正项
        packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
    }
    // 用普通 store128 而非流式 store128cs：GELU 之后的数据可能对下一个算子有用，保留在缓存中更有利
    store128(out + idx, packed_out);
}

// ============================================================================
// Kernel 启动封装
// ============================================================================

/* ----------------------------------------------------------------------------
   gelu_forward_tail_kernel —— 标量版 GELU kernel（处理向量化装不下的尾部）
   ----------------------------------------------------------------------------
   背景：gelu_forward_kernel2 要求元素总数 N 是 block_size*x128::size 的整数倍；
         当 N 不满足时（典型：KV 增量模式 T=1，FFN 元素数 3072 < 2048 的倍数），
         尾部 N - mainN 个元素由本 kernel 逐元素处理，带边界检查。
   每个线程处理一个元素（无向量化约束，因此任意 N 均可）。
---------------------------------------------------------------------------- */
__global__ void gelu_forward_tail_kernel(floatX* out, const floatX* inp, int start, int N) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) { return; } // 边界保护
    float xi = (float)inp[idx];
    float cube = 0.044715f * xi * xi * xi; // x^3 修正项
    out[idx] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
}

/* ----------------------------------------------------------------------------
   gelu_forward —— GELU 主机端启动函数
   ----------------------------------------------------------------------------
   block_size=512，每线程负责一个数据包，按此计算 grid 大小。
   分两段：
     1. 向量化主体：前 mainN 个元素（vec=block_size*x128::size 的整数倍），
        用 gelu_forward_kernel2 精确覆盖（mainN 是 vec 倍数，grid 恰为整数）；
     2. 尾部：mainN..N 之间的零头，用标量 kernel 逐个处理。
   全量模式（T 为 256 倍数）下 N 恒为 vec 整数倍，只走第 1 段，行为与原来一致。
---------------------------------------------------------------------------- */
void gelu_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    const int vec = block_size * x128::size; // 每 block 处理的元素数（FP32 下 2048）
    const int mainN = (N / vec) * vec;       // 向量化主体：不超过 N 的最大 vec 倍数
    if (mainN > 0) {
        gelu_forward_kernel2<<<mainN / vec, block_size, 0, stream>>>(out, inp);
    }
    if (mainN < N) {
        // 尾部不足一个数据包，逐元素处理
        gelu_forward_tail_kernel<<<CEIL_DIV(N - mainN, 256), 256, 0, stream>>>(out, inp, mainN, N);
    }
    cudaCheck(cudaGetLastError());
}
