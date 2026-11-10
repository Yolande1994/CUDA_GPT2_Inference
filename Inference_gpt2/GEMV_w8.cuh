/* ============================================================================
   GEMV_w8.cuh —— 8 位权重版 GEMV（INT8 / FP8 由编译宏二选一）
   ============================================================================
   量化版 GEMV 与普通 GEMV 的区别：
     权重不再是浮点数，而是每行一个字节的整数码 + 一个浮点 scale（缩放系数）。
     一个字节只有 1 字节，相比 FP32 的 4 字节，权重搬运量直接降到 1/4，这是带宽瓶颈下的提速来源。
     还原关系：真实值 = 字节解码后的数 × 该行的 scale。

   两种格式（编译期互斥，不能同时定义）：
     USE_INT8：字节是有符号整数，解码就是直接转成浮点数；
     USE_FP8 ：字节是 E4M3 浮点格式，需要硬件指令先转成 BF16 再转 FP32。

   本文件提供：
     GEMV6_w8x2 融合 kernel：一个 warp 同时处理两行，内含 bias、可选 GELU 和 GELU 前线性值输出；
     launch_GEMV6_w8x2 对应的启动函数；
     dequant_wte_kernel 把词嵌入表整体反量化成浮点（嵌入查找路径使用）。
   ============================================================================ */
#ifndef GEMV_W8_CUH
#define GEMV_W8_CUH

#include <cstdint>
#include "common.h"

// 编译期格式检查：两个宏必须恰好定义一个，配错了直接报错，不默默走错路径
#if defined(USE_FP8) && defined(USE_INT8)
#error "USE_FP8 与 USE_INT8 互斥，只能选一个"
#elif !defined(USE_FP8) && !defined(USE_INT8)
#error "8 位路径需要定义 USE_INT8 或 USE_FP8 之一"
#elif defined(USE_FP8)
#include <cuda_fp8.h>
#endif

// ============================================================================
// w8_to_f32：把一个量化字节解码成它代表的浮点数（还没乘 scale）
// ============================================================================
#ifdef USE_FP8
__device__ __forceinline__ float w8_to_f32(uint8_t b) {
    // E4M3 字节 → BF16 原始位 → BF16 数 → FP32，全程走硬件转换指令
    __nv_fp8_storage_t s = b;
    __nv_bfloat16_raw r = __nv_cvt_fp8_to_bf16raw(s, __NV_E4M3);
    __nv_bfloat16 h = *reinterpret_cast<__nv_bfloat16*>(&r);
    return __bfloat162float(h);
}
#else
__device__ __forceinline__ float w8_to_f32(uint8_t b) {
    // INT8 只是有符号整数编码，重新解释成 int8 再转浮点即可
    return (float)((int8_t)b);
}
#endif

// ============================================================================
// GEMV6_w8x2：8 位权重融合 GEMV
// ============================================================================

/* ----------------------------------------------------------------------------
   并行设计：一个 warp 同时负责两行（row0、row1）。
     输入向量 x 只加载一次，两行的权重各自加载，乘加时复用同一批 x，
     既提高指令级并行，又让 x 的带宽被两行摊薄。
   每个 lane 一轮覆盖 32 个字节：权重按两个 uint4（共 32 字节）读入，x 按四个 uint4 读入；
   K 不是 1024 整倍数时由后面的循环逐块补齐。
   行 scale 在累加结束后才乘一次，而不是每个元素都乘——乘法次数从 K 次降到 1 次，数学等价。
   ---------------------------------------------------------------------------- */
template <typename OT, typename BT>
__global__ void GEMV6_w8x2(OT* out, OT* pre_gelu,
                           const uint8_t* W, const __nv_bfloat16* x, const float* scale,
                           const BT* bias, int M, int K, bool do_gelu) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    // 每个 warp 负责两行，乘 2 得到第一行行号
    int row0 = (blockIdx.x * (blockDim.x / 32) + warp) * 2;
    if (row0 >= M) return;
    int row1 = row0 + 1;
    bool has1 = (row1 < M);  // 行数为奇数时，最后一个 warp 的第二行不存在
    const uint8_t* w0r = W + (size_t)row0 * K;   // 第一行的字节权重
    const uint8_t* w1r = W + (size_t)row1 * K;   // 第二行的字节权重
    const float s0 = scale[row0];                 // 第一行的 scale
    const float s1 = has1 ? scale[row1] : 0.0f;  // 第二行 scale，不存在给 0
    float sum0 = 0.0f, sum1 = 0.0f;              // 两行各自的 FP32 累加器

    const int BLOCKS = K / 1024;  // 整块数，每块覆盖 1024 字节
    const int BASE = lane * 32;   // 本 lane 在块内的起始偏移
    #pragma unroll 2
    for (int r = 0; r < BLOCKS; r++) {
        int off = r * 1024 + BASE;
        union { uint4 u; uint8_t b[16]; } a0, a1, b0, b1;
        union { uint4 u; __nv_bfloat16 h[8]; } x0, x1, x2, x3;
        a0.u = *(const uint4*)(w0r + off);       // 第一行前 16 字节
        b0.u = *(const uint4*)(w1r + off);       // 第二行前 16 字节
        x0.u = *(const uint4*)(x + off);         // 输入向量前 8 元素
        a1.u = *(const uint4*)(w0r + off + 16);  // 第一行后 16 字节
        b1.u = *(const uint4*)(w1r + off + 16);  // 第二行后 16 字节
        x1.u = *(const uint4*)(x + off + 8);
        x2.u = *(const uint4*)(x + off + 16);
        x3.u = *(const uint4*)(x + off + 24);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // 同一批 x 同时喂给两行的权重，x 的加载被复用
            float xv0 = (float)x0.h[i], xv1 = (float)x1.h[i];
            float xv2 = (float)x2.h[i], xv3 = (float)x3.h[i];
            sum0 += w8_to_f32(a0.b[i])     * xv0;  // 第一行前 16 字节
            sum1 += w8_to_f32(b0.b[i])     * xv0;  // 第二行前 16 字节
            sum0 += w8_to_f32(a0.b[8 + i]) * xv1;
            sum1 += w8_to_f32(b0.b[8 + i]) * xv1;
            sum0 += w8_to_f32(a1.b[i])     * xv2;
            sum1 += w8_to_f32(b1.b[i])     * xv2;
            sum0 += w8_to_f32(a1.b[8 + i]) * xv3;
            sum1 += w8_to_f32(b1.b[8 + i]) * xv3;
        }
    }
    // 尾部兜底：剩余不足 1024 的部分按同样方式逐块处理
    for (int off = BLOCKS * 1024 + BASE; off + 31 < K; off += 1024) {
        union { uint4 u; uint8_t b[16]; } a0, a1, b0, b1;
        union { uint4 u; __nv_bfloat16 h[8]; } x0, x1, x2, x3;
        a0.u = *(const uint4*)(w0r + off);
        b0.u = *(const uint4*)(w1r + off);
        x0.u = *(const uint4*)(x + off);
        a1.u = *(const uint4*)(w0r + off + 16);
        b1.u = *(const uint4*)(w1r + off + 16);
        x1.u = *(const uint4*)(x + off + 8);
        x2.u = *(const uint4*)(x + off + 16);
        x3.u = *(const uint4*)(x + off + 24);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float xv0 = (float)x0.h[i], xv1 = (float)x1.h[i];
            float xv2 = (float)x2.h[i], xv3 = (float)x3.h[i];
            sum0 += w8_to_f32(a0.b[i])     * xv0;
            sum1 += w8_to_f32(b0.b[i])     * xv0;
            sum0 += w8_to_f32(a0.b[8 + i]) * xv1;
            sum1 += w8_to_f32(b0.b[8 + i]) * xv1;
            sum0 += w8_to_f32(a1.b[i])     * xv2;
            sum1 += w8_to_f32(b1.b[i])     * xv2;
            sum0 += w8_to_f32(a1.b[8 + i]) * xv3;
            sum1 += w8_to_f32(b1.b[8 + i]) * xv3;
        }
    }
    sum0 *= s0;  // 行 scale 提到循环外：这里乘一次等价于循环内每个元素都乘
    sum1 *= s1;
    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {  // 两行各自做 warp 内归约
        sum0 += __shfl_down_sync(0xffffffffu, sum0, off);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, off);
    }
    if (lane == 0) {
        float pre0 = sum0, pre1 = sum1;
        if (bias) { pre0 += (float)bias[row0]; pre1 += has1 ? (float)bias[row1] : 0.0f; }  // 加偏置
        if (pre_gelu) { pre_gelu[row0] = (OT)pre0; if (has1) pre_gelu[row1] = (OT)pre1; }  // GELU 前线性值
        if (do_gelu) {
            // GPT-2 的 tanh 近似 GELU，两行分别处理
            float c0 = 0.044715f * pre0 * pre0 * pre0;
            pre0 = 0.5f * pre0 * (1.0f + tanhf(0.7978845608028654f * (pre0 + c0)));
            if (has1) {
                float c1 = 0.044715f * pre1 * pre1 * pre1;
                pre1 = 0.5f * pre1 * (1.0f + tanhf(0.7978845608028654f * (pre1 + c1)));
            }
        }
        out[row0] = (OT)pre0;                 // 写回第一行
        if (has1) out[row1] = (OT)pre1;      // 写回第二行
    }
}

template <typename OT, typename BT>
inline void launch_GEMV6_w8x2(OT* out, OT* pre_gelu,
                              const uint8_t* W, const __nv_bfloat16* x, const float* scale,
                              const BT* bias, int M, int K, bool do_gelu,
                              cudaStream_t stream = 0) {
    // 每个 block 8 个 warp、每 warp 两行，即一个 block 处理 16 行
    dim3 grid(CEIL_DIV(M, 16));
    GEMV6_w8x2<OT, BT><<<grid, 256, 0, stream>>>(out, pre_gelu, W, x, scale, bias, M, K, do_gelu);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "GEMV6_w8x2 launch error: %s\n", cudaGetErrorString(err)); exit(1); }
}

// ============================================================================
// dequant_wte_kernel：词嵌入表整表反量化
// ============================================================================

/* 嵌入查找按 token id 随机取行，不适合在 GEMV 里在线解码，
   所以先把整张表还原成浮点：每个 block 负责一行，逐元素解码并乘上行 scale。 */
__global__ void dequant_wte_kernel(const uint8_t* w8, const float* scale, floatX* out,
                                   int Vp, int C) {
    int row = blockIdx.x;                       // 每个 block 负责一行
    const uint8_t* w = w8 + (size_t)row * C;  // 本行字节权重
    floatX* o = out + (size_t)row * C;         // 本行浮点输出
    float s = scale[row];                       // 本行 scale
    // 块内线程按步长 blockDim 覆盖整行
    for (int i = threadIdx.x; i < C; i += blockDim.x)
        o[i] = (floatX)(w8_to_f32(w[i]) * s);   // 解码后乘 scale 还原真实值
}

#endif
