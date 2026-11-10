/* ============================================================================
   GEMV.cuh —— 推理工程使用的手写 GEMV 算子（不依赖 cuBLAS）
   ============================================================================
   提供两个 kernel：
     GEMV6        纯 GEMV：out = W·x；
     GEMV6_fused  融合版：out = GELU?(W·x + bias)，并可额外输出 GELU 之前的线性值，省掉单独的 bias / 激活 kernel。
   二者都由 CPU 端的 launch_* 函数启动，推理主程序只调用 launch 层。

   精度约定：
     权重和输入可以是 FP32 或 BF16（模板参数 T），但无论哪种，累加都在 FP32 下进行，输出始终是 float，避免低精度累加带来误差。

   kernel 的线程分工：
     一个 warp（32 线程）负责一行，每轮发两个 128-bit 向量加载（FP32 为两个 float4、BF16 为两个 uint4），合并访存 + 双路读请求并行；
     block 固定 512 线程，即每个 block 处理 16 行。
   ============================================================================ */
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// 向上取整除法：算网格需要多少个 block
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// CUDA API 返回值检查：失败即打印位置和原因并退出
#define GEMV_CUDA_CHECK(err) { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } \
}

// ============================================================================
// GEMV6：纯矩阵向量乘
// ============================================================================

/* ----------------------------------------------------------------------------
   每个线程的工作：
     row/lane 把全局线程号映射到 “warp 负责的行” 和 “warp 内编号”；
     主循环里每个 lane 读两个 128-bit 块并累加，整轮覆盖 256（FP32）或 512（BF16）个元素；
     末尾不足一轮的部分用单块兜底；
     最后 warp 内 shuffle 归约，lane 0 写出整行结果。
   BF16 没有专用 16 字节向量类型，用 union 把 uint4 重新解释成 8 个 BF16。
   ---------------------------------------------------------------------------- */
template <typename T>
__global__ void GEMV6(float* out, const T* W, const T* x, int M, int K) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;  // 每 32 线程一个 warp，整除得行号
    int lane = threadIdx.x % 32;  // warp 内编号 0~31
    if (row >= M) return;
    const T* wrow = W + row * K;  // 本行起始权重
    float sum = 0.0f;

    if constexpr (sizeof(T) == 4) {
        const int BLOCKS = K / 256;  // 整轮数，每轮覆盖 256 元素
        const int BASE = lane * 4;   // 本 lane 在块内的起始偏移
        #pragma unroll 2
        for (int r = 0; r < BLOCKS; r++) {
            int off = r * 256 + BASE;
            const float4 w0 = *(const float4*)(wrow + off);  // 第一块，128-bit 向量加载
            const float4 x0 = *(const float4*)(x + off);
            const float4 w1 = *(const float4*)(wrow + off + 128);  // 第二个块与第一个块独立，两路加载并行
            const float4 x1 = *(const float4*)(x + off + 128);
            sum += w0.x * x0.x + w0.y * x0.y + w0.z * x0.z + w0.w * x0.w;
            sum += w1.x * x1.x + w1.y * x1.y + w1.z * x1.z + w1.w * x1.w;
        }

        const int rem = K % 256;  // 尾部兜底：余数非 0 时再读一个单块
        if (rem != 0) {
            int off = BLOCKS * 256 + lane * 4;
            if (off + 3 < K) {
                const float4 w0 = *(const float4*)(wrow + off);
                const float4 x0 = *(const float4*)(x + off);
                sum += w0.x * x0.x + w0.y * x0.y + w0.z * x0.z + w0.w * x0.w;
            }
        }
    } else {
        const int BLOCKS = K / 512;  // BF16 整轮数，每轮 512 元素
        const int BASE = lane * 8;
        #pragma unroll 2
        for (int r = 0; r < BLOCKS; r++) {
            int off = r * 512 + BASE;
            union { uint4 u; __nv_bfloat16 h[8]; } w0, x0, w1, x1;
            w0.u = *(const uint4*)(wrow + off);  // union 把 uint4 重解释成 8 个 BF16
            x0.u = *(const uint4*)(x + off);
            w1.u = *(const uint4*)(wrow + off + 256);  // 第二块，双路并行
            x1.u = *(const uint4*)(x + off + 256);
            #pragma unroll
            for (int i = 0; i < 8; i++) sum += (float)w0.h[i] * (float)x0.h[i];
            #pragma unroll
            for (int i = 0; i < 8; i++) sum += (float)w1.h[i] * (float)x1.h[i];
        }

        const int rem = K % 512;  // BF16 尾部兜底
        if (rem != 0) {
            int off = BLOCKS * 512 + lane * 8;
            if (off + 7 < K) {
                union { uint4 u; __nv_bfloat16 h[8]; } w0, x0;
                w0.u = *(const uint4*)(wrow + off);
                x0.u = *(const uint4*)(x + off);
                #pragma unroll
                for (int i = 0; i < 8; i++) sum += (float)w0.h[i] * (float)x0.h[i];
            }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off /= 2)  // warp 内归约：5 轮 shuffle 后总和汇聚到 lane 0
        sum += __shfl_down_sync(0xffffffffu, sum, off);

    if (lane == 0) out[row] = sum;  // lane 0 写回整行
}

template <typename T>
inline void launch_GEMV6(float* out, const T* W, const T* x, int M, int K) {
    dim3 grid(ceil_div(M, 16));  // 每 block 512 线程处理 16 行

    GEMV6<T><<<grid, 512>>>(out, W, x, M, K);
    GEMV_CUDA_CHECK(cudaGetLastError());
}


// ============================================================================
// GEMV6_fused：GEMV + bias + 可选 GELU 融合
// ============================================================================

/* ----------------------------------------------------------------------------
   计算部分与 GEMV6 相同，区别在收尾（仅 lane 0 执行）：
     1. 加 bias；
     2. 若给了 pre_gelu，把 GELU 前的线性值存一份（残差连接要用）；
     3. do_gelu 为真时套用 GPT-2 的 tanh 近似 GELU。
   尾部用循环逐块补齐而不是单块判断，因此对任意 K 都成立，不要求余数对齐。
   ---------------------------------------------------------------------------- */
template <typename T, typename OT = T, typename BT = T>
__global__ void GEMV6_fused(OT* out, OT* pre_gelu, const T* W, const T* x, const BT* bias,
                            int M, int K, bool do_gelu) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= M) return;
    const T* wrow = W + row * K;
    float sum = 0.0f;

    if constexpr (sizeof(T) == 4) {
        const int BLOCKS = K / 256;
        const int BASE = lane * 4;
        #pragma unroll 2
        for (int r = 0; r < BLOCKS; r++) {
            int off = r * 256 + BASE;
            const float4 w0 = *(const float4*)(wrow + off);
            const float4 x0 = *(const float4*)(x + off);
            const float4 w1 = *(const float4*)(wrow + off + 128);
            const float4 x1 = *(const float4*)(x + off + 128);
            sum += w0.x * x0.x + w0.y * x0.y + w0.z * x0.z + w0.w * x0.w;
            sum += w1.x * x1.x + w1.y * x1.y + w1.z * x1.z + w1.w * x1.w;
        }

        const int rem_start = BLOCKS * 256 + BASE;
        for (int off = rem_start; off + 3 < K; off += 128) {  // 循环兜底：任意 K 都能覆盖完
            const float4 w0 = *(const float4*)(wrow + off);
            const float4 x0 = *(const float4*)(x + off);
            sum += w0.x * x0.x + w0.y * x0.y + w0.z * x0.z + w0.w * x0.w;
        }

    } else {
        const int BLOCKS = K / 512;
        const int BASE = lane * 8;
        #pragma unroll 2
        for (int r = 0; r < BLOCKS; r++) {
            int off = r * 512 + BASE;
            union { uint4 u; __nv_bfloat16 h[8]; } w0, x0, w1, x1;
            w0.u = *(const uint4*)(wrow + off);
            x0.u = *(const uint4*)(x + off);
            w1.u = *(const uint4*)(wrow + off + 256);
            x1.u = *(const uint4*)(x + off + 256);
            #pragma unroll
            for (int i = 0; i < 8; i++) sum += (float)w0.h[i] * (float)x0.h[i];
            #pragma unroll
            for (int i = 0; i < 8; i++) sum += (float)w1.h[i] * (float)x1.h[i];
        }

        const int rem_start = BLOCKS * 512 + BASE;
        for (int off = rem_start; off + 7 < K; off += 256) {
            union { uint4 u; __nv_bfloat16 h[8]; } w0, x0;
            w0.u = *(const uint4*)(wrow + off);
            x0.u = *(const uint4*)(x + off);
            #pragma unroll
            for (int i = 0; i < 8; i++) sum += (float)w0.h[i] * (float)x0.h[i];
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) {
        float pre = sum;
        if (bias) pre += (float)bias[row];     // 加偏置
        if (pre_gelu) pre_gelu[row] = (OT)pre; // GELU 前的线性值存一份，残差连接要用
        if (do_gelu) {
            // GPT-2 使用的 tanh 近似 GELU：0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
            float cube = 0.044715f * pre * pre * pre;
            pre = 0.5f * pre * (1.0f + tanhf(0.7978845608028654f * (pre + cube)));
        }
        out[row] = (OT)pre;  // 写回融合结果
    }
}

template <typename T, typename OT = T, typename BT = T>
inline void launch_GEMV6_fused(OT* out, OT* pre_gelu, const T* W, const T* x, const BT* bias,
                               int M, int K, bool do_gelu, cudaStream_t stream = 0) {
    dim3 grid(ceil_div(M, 16));
    GEMV6_fused<T, OT, BT><<<grid, 512, 0, stream>>>(out, pre_gelu, W, x, bias, M, K, do_gelu);
    GEMV_CUDA_CHECK(cudaGetLastError());
}
