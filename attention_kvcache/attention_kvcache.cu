/* ============================================================================
   Attention KV Cache Decode CUDA Kernel
   ============================================================================

   算子说明：
     本文件是一个自包含的 KV Cache 增量注意力（Decode 阶段）微基准测试。
     使用随机 Q/K/V 数据模拟 GPT-2 124M 的注意力形状（B=8, NH=12, hs=64, maxT=1024），
     对比多种从朴素实现到 Flash Decoding / FlashAttention 架构的演进。

   版本说明：
     版本1：朴素基线版（Baseline）。
             单 Block 单头处理；点积依赖块内共享内存树状归约；两遍式 Softmax，存在大量 __syncthreads 同步开销。
     版本2：融合优化版（Fused）。
             融合 QK 打分、Softmax 与加权；采用 float4 128-bit 向量化读 K；
             使用 Warp Shuffle 替代树状归约；预计算 Exp 消除 SFU 冗余，4 组并行求和。
     版本3：Flash Decoding 经典两阶段版。
             引入 Split-KV 架构，沿序列切分为 S 段，多 Block 并行计算局部结果（m, lse, O）；
             独立的 Merge Kernel 利用 Log-Sum-Exp 跨段重缩放合并。
     版本4：Cooperative 单核合并版。
             在 v3 基础上，利用 cooperative groups 的 grid.sync() 将 Split 和 Merge 合并进单一 Kernel，消除两次 Kernel Launch 带来的固定开销。
     版本5：Flash Decoding++ 对照实验版。
             预计算全局 Max，使 Merge 阶段退化为简单求和。
             刻意采用 3 个 Kernel 的拆分设计，作为“增加全局读写反而变慢”的反面教材。
     版本6：FlashAttention 在线优化版。
             采用经典在线 Softmax（Online Softmax）与 cp.async 异步双缓冲流水线。
             核心状态（m, l, O）全量驻留寄存器，TILE=64 动态共享内存，最大程度重叠搬运与计算。

   编译示例：
     nvcc -O3 --use_fast_math -arch=native attention_kvcache.cu -o attention_kvcache

   运行命令：
     ./attention_kvcache            # 一次跑全部版本：正确性 + 性能对比
     ./attention_kvcache 2          # 测指定版本，比如版本 2
     ./attention_kvcache 0 profile  # profile 模式：供 ncu/nsys 采集全部版本

   测量稳定性：
     每次测量前 memset 一个 L2 大小的缓冲刷新 L2 保证每次都在相同冷缓存条件下跑；
     每个版本重复 100 次单次计时，报告中位数。

    性能说明：本文件以 v3 为通用最优，v4 只能服务 B=1 场景，v5、v6 仅作思想参考，不适用该文件数据环境。
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>   // v4 grid_group（跨 block 同步）
namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define MAX_SPLIT_S 32            // 显存分配的 Flash Decoding 最大安全切分段数

#define cudaCheck(err) \
    do { \
        cudaError_t e = (err); \
        if (e != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        } \
    } while (0)


// ============================================================================
// CPU 参考实现
// ============================================================================

/* CPU 参考实现：增量注意力（打分 → softmax → 加权）
   逐 (b,h) 头：本头 Q 与缓存 0..seq_len-1 全部位置（含自己，即因果对角线）打分，
   先减行最大分保证数值稳定（避免 exp 上溢），再 exp 加权归一化。
   与各 GPU kernel 语义严格一致，作为正确性基准。 */
void attention_kv_cpu(float* out, const float* qkv, const float* kv_cache,
                      int B, int seq_len, int NH, int hs, int maxT, float scale) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            float scores[1024];
            float maxval = -FLT_MAX;
            for (int j = 0; j < seq_len; j++) {
                float sum = 0.0f;
                for (int d = 0; d < hs; d++) {
                    sum += qkv[b * 3 * NH * hs + h * hs + d] * kv_cache[(b * NH + h) * maxT * hs + j * hs + d];
                }
                scores[j] = sum * scale;
                maxval = fmaxf(maxval, scores[j]);
            }
            // 先减最大值再 exp，保证数值稳定性
            const float* v_cache = kv_cache + B * NH * maxT * hs + (b * NH + h) * maxT * hs;
            for (int d = 0; d < hs; d++) {
                float acc = 0.0f;
                float expsum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    float expv = expf(scores[j] - maxval);
                    acc += expv * v_cache[j * hs + d];
                    expsum += expv;
                }
                out[(b * NH + h) * hs + d] = acc / expsum;
            }
        }
    }
}


// ============================================================================
// GPU Kernel 实现
// ============================================================================

/* ----------------------------------------------------------------------------
   版本1：朴素版
   ----------------------------------------------------------------------------
   定位：最直接的 KV Cache 增量注意力实现，作为性能基线（Baseline）。
   并行策略：1 个 block 负责 1 个 (b,h) 头，blockDim=hs=64。
   性能瓶颈：
     每个位置 j 的点积都要靠「各线程算一维 + 块内树状归约」完成；
     逐 j 串行执行，每个 j 都需要一轮归约（log2(hs) 次 __syncthreads），
     访存延迟与同步开销随 seq_len 线性增长。
   ---------------------------------------------------------------------------- */
__global__ void attention_kv_kernel(float* out, const float* qkv, const float* kv_cache,
                                    int B, int seq_len, int NH, int hs, int maxT, float scale) {
    int b = blockIdx.x / NH;   // 该 block 负责的 batch
    int h = blockIdx.x % NH;   // 该 block 负责的头
    int i = threadIdx.x;       // 本线程负责的通道号 0..hs-1

    __shared__ float reduce_buf[64]; // 树状归约的临时缓冲区
    __shared__ float scores[1024];   // 存储当前头所有历史位置的打分结果

    float q_val = qkv[b * 3 * NH * hs + h * hs + i]; // 读新 token 的 Q 值到寄存器

    const float* k_cache = kv_cache + (b * NH + h) * maxT * hs;
    for (int j = 0; j < seq_len; j++) {
        reduce_buf[i] = q_val * k_cache[j * hs + i];
        __syncthreads(); // 等待所有线程将乘积写入共享内存
        for (int stride = hs / 2; stride > 0; stride >>= 1) {
            if (i < stride) { reduce_buf[i] += reduce_buf[i + stride]; }
            __syncthreads(); // 每一步折半累加后，必须做块级同步，防止读写冲突
        }
        if (i == 0) { scores[j] = reduce_buf[0] * scale; }
        __syncthreads(); // 确保 scores[j] 写入后，其他线程才能进入下一轮循环
    }

    float maxval = -FLT_MAX;
    for (int j = 0; j < seq_len; j++) { maxval = fmaxf(maxval, scores[j]); }

    const float* v_cache = kv_cache + B * NH * maxT * hs + (b * NH + h) * maxT * hs;
    float acc = 0.0f;    // softmax 分子
    float expsum = 0.0f; // softmax 分母
    for (int j = 0; j < seq_len; j++) {
        float expv = expf(scores[j] - maxval);
        acc += expv * v_cache[j * hs + i];
        expsum += expv;
    }
    out[(b * NH + h) * hs + i] = acc / expsum;
}


/* ----------------------------------------------------------------------------
   版本2：融合版（float4 向量化打分 + exp 预计算消冗余）
   ----------------------------------------------------------------------------
   针对 v1 发现的两处冗余：
     1. 逐 float 读 K 行（64 条 load），改用 float4 读（16 条 load）。
     2. 64 个通道线程在 j 循环里各算一遍 expf，改为用 64 线程预计算 exp 存共享内存，
        全局指数和也只归约一次；阶段三直接读 exp_buf，只做 FMA。
   ---------------------------------------------------------------------------- */
__global__ void attention_kv_fused_v2_kernel(float* out, const float* qkv, const float* kv_cache,
                                             int B, int seq_len, int NH, int hs, int maxT, float scale) {
    const int b = blockIdx.x / NH;
    const int h = blockIdx.x % NH;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    __shared__ float scores[1024];  // 存储所有位置的打分结果
    __shared__ float q_s[64];       // 缓存 Q 向量，避免重复访问全局内存
    __shared__ float max_buf[2];    // 两个 warp 各写一个局部最大
    __shared__ float exp_buf[1024]; // 预计算的 exp(分-最大分)
    __shared__ float esum_buf[2];   // 全局指数和的 warp 归约缓冲区
    __shared__ float part_acc[256]; // 4 组 × 64 通道的部分加权和（分子）

    if (tid < hs) { q_s[tid] = qkv[b * 3 * NH * hs + h * hs + tid]; }
    __syncthreads();

    const float* k_cache = kv_cache + (b * NH + h) * maxT * hs;
    for (int j = tid; j < seq_len; j += block_size) {
        float sum = 0.0f;
        // float4 强转：强制 GPU 使用 128-bit 的 LDG 指令加载，提升带宽利用率
        const float4* k4 = (const float4*)(k_cache + j * hs);
        for (int d4 = 0; d4 < hs / 4; d4++) {
            float4 kv = k4[d4];
            sum += q_s[d4 * 4 + 0] * kv.x;
            sum += q_s[d4 * 4 + 1] * kv.y;
            sum += q_s[d4 * 4 + 2] * kv.z;
            sum += q_s[d4 * 4 + 3] * kv.w;
        }
        scores[j] = sum * scale;
    }
    __syncthreads();

    float maxval = -FLT_MAX;
    if (tid < hs) {
        for (int j = tid; j < seq_len; j += hs) { maxval = fmaxf(maxval, scores[j]); }
        // __shfl_xor_sync：Warp 级寄存器数据交换指令，用蝶形算法实现高速规约
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, offset));
        }
        if (tid % WARP_SIZE == 0) { max_buf[tid / WARP_SIZE] = maxval; }
    }
    __syncthreads();
    const float global_max = fmaxf(max_buf[0], max_buf[1]);

    if (tid < hs) {
        float esum = 0.0f;
        for (int j = tid; j < seq_len; j += hs) {
            float ev = expf(scores[j] - global_max);
            exp_buf[j] = ev; // 存到共享内存，供阶段三复用，消除 64 倍的 SFU 冗余
            esum += ev;
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            esum += __shfl_xor_sync(0xffffffff, esum, offset);
        }
        if (tid % WARP_SIZE == 0) { esum_buf[tid / WARP_SIZE] = esum; }
    }
    __syncthreads();
    const float expsum = esum_buf[0] + esum_buf[1];

    {
        const int group = tid / hs;      // 0..3，每组负责一段 j 范围
        const int chan = tid % hs;       // 0..63，负责哪个输出通道
        const float* v_cache = kv_cache + B * NH * maxT * hs + (b * NH + h) * maxT * hs;
        const int total = seq_len;
        const int chunk = CEIL_DIV(total, 4); // 将 j 范围均分给 4 个组
        const int j_begin = group * chunk;
        const int j_end = (j_begin + chunk < total) ? j_begin + chunk : total;

        float acc = 0.0f;
        for (int j = j_begin; j < j_end; j++) {
            acc += exp_buf[j] * v_cache[j * hs + chan]; // 只有 FMA 指令，极快
        }
        part_acc[tid] = acc;
    }
    __syncthreads();

    if (tid < hs) {
        float acc = 0.0f;
        for (int g = 0; g < 4; g++) { acc += part_acc[g * hs + tid]; }
        out[(b * NH + h) * hs + tid] = acc / expsum;
    }
}


/* ----------------------------------------------------------------------------
   版本3：Flash Decoding（沿 KV 序列切分到多 block 并行 + merge 归约）
   ----------------------------------------------------------------------------
   背景：v1~v2 都是「一个头一个 block」，而在 Decode 阶段 batch 小、序列长时，GPU 的大部分 SM 无事可做。
   核心改进：借鉴 GEMM 的 split-K 技巧，沿 KV 序列切 S 段，多 block 并行，最后用 merge kernel 合并。
   ---------------------------------------------------------------------------- */
__global__ void attention_kv_split_kernel(float* part_out, float* part_lse, float* part_max,
                                          const float* qkv, const float* kv_cache,
                                          int B, int seq_len, int NH, int hs, int maxT,
                                          int S, float scale) {
    // 拆解一维 Grid 为三维索引 (b, h, s)
    const int b = blockIdx.x / (NH * S);
    const int h = (blockIdx.x / S) % NH;
    const int s = blockIdx.x % S;
    const int tid = threadIdx.x;

    const int total = seq_len;
    const int chunk = CEIL_DIV(total, S);  // 每段应该分配多少个位置
    const int j_start = s * chunk;
    const int j_end = (j_start + chunk < total) ? j_start + chunk : total;
    const int n_local = j_end - j_start;  // 本段实际负责的 KV 数量

    // 空段处理：写中性值覆盖垃圾数据，merge 时 exp(-inf)=0，自然忽略
    if (n_local <= 0) {
        if (tid == 0) {
            part_max[(b * NH + h) * S + s] = -FLT_MAX;
            part_lse[(b * NH + h) * S + s] = 0.0f;
        }
        if (tid < hs) part_out[((b * NH + h) * S + s) * hs + tid] = 0.0f;
        return;
    }

    __shared__ float q_s[64];        // 本头 Q 向量
    __shared__ float scores[256];    // 本段打分（S=4 时每段 ≤ 256）
    __shared__ float warp_max[2];    // 两个 warp 各写一个局部最大
    __shared__ float exp_local[256]; // 本段 exp 预计算
    __shared__ float warp_lse[2];    // 两个 warp 各写一个局部指数和

    if (tid < hs) { q_s[tid] = qkv[b * 3 * NH * hs + h * hs + tid]; }
    __syncthreads();

    const float* k_cache = kv_cache + (b * NH + h) * maxT * hs;
    for (int jj = tid; jj < n_local; jj += blockDim.x) {
        int j = j_start + jj;
        float sum = 0.0f;
        const float4* k4 = (const float4*)(k_cache + j * hs);
        for (int d4 = 0; d4 < hs / 4; d4++) {
            float4 kv = k4[d4];
            sum += q_s[d4 * 4 + 0] * kv.x;
            sum += q_s[d4 * 4 + 1] * kv.y;
            sum += q_s[d4 * 4 + 2] * kv.z;
            sum += q_s[d4 * 4 + 3] * kv.w;
        }
        scores[jj] = sum * scale;
    }
    __syncthreads();

    if (tid < hs) {
        float mv = -FLT_MAX;
        for (int jj = tid; jj < n_local; jj += hs) { mv = fmaxf(mv, scores[jj]); }
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffff, mv, off));
        }
        if (tid % WARP_SIZE == 0) { warp_max[tid / WARP_SIZE] = mv; }
    }
    __syncthreads();
    const float local_max = fmaxf(warp_max[0], warp_max[1]);

    if (tid < hs) {
        float ls = 0.0f;
        for (int jj = tid; jj < n_local; jj += hs) {
            float ev = expf(scores[jj] - local_max);
            exp_local[jj] = ev;
            ls += ev;
        }
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            ls += __shfl_xor_sync(0xffffffff, ls, off);
        }
        if (tid % WARP_SIZE == 0) { warp_lse[tid / WARP_SIZE] = ls; }
    }
    __syncthreads();
    const float local_lse = warp_lse[0] + warp_lse[1];

    __shared__ float group_acc[256];
    {
        const int group = tid / hs;  // 0..3，负责本段内哪一小段
        const int chan = tid % hs;   // 0..63，负责哪个输出通道
        const float* v_cache = kv_cache + B * NH * maxT * hs + (b * NH + h) * maxT * hs;
        const int gchunk = CEIL_DIV(n_local, 4);  // 把总任务平均分成 4 份
        const int gstart = group * gchunk;  // 本组的起始任务
        const int gend = (gstart + gchunk < n_local) ? gstart + gchunk : n_local; // 本组结束任务
        float acc = 0.0f;
        for (int jj = gstart; jj < gend; jj++) {
            acc += exp_local[jj] * v_cache[(j_start + jj) * hs + chan];
        }
        group_acc[tid] = acc;
    }
    __syncthreads();

    if (tid < hs) {
        float acc = 0.0f;
        for (int g = 0; g < 4; g++) { acc += group_acc[g * hs + tid]; }
        part_out[((b * NH + h) * S + s) * hs + tid] = acc;
        if (tid == 0) {
            part_max[(b * NH + h) * S + s] = local_max;
            part_lse[(b * NH + h) * S + s] = local_lse;
        }
    }
}

__global__ void attention_kv_merge_kernel(float* out, const float* part_out,
                                          const float* part_lse, const float* part_max,
                                          int NH, int hs, int S) {
    const int b = blockIdx.x / NH;
    const int h = blockIdx.x % NH;
    const int tid = threadIdx.x;
    if (tid >= hs) return;

    float m_global = -FLT_MAX;
    for (int s = 0; s < S; s++) {
        m_global = fmaxf(m_global, part_max[(b * NH + h) * S + s]); // 跨段求全局最大
    }

    float O_final = 0.0f;
    float l_final = 0.0f;
    for (int s = 0; s < S; s++) {
        // 跨段 rescale 合并：各段均对齐到全局最大值的基准上
        float rescale = expf(part_max[(b * NH + h) * S + s] - m_global);
        O_final += rescale * part_out[((b * NH + h) * S + s) * hs + tid];  // 分子合并
        l_final += rescale * part_lse[(b * NH + h) * S + s];                // 分母合并
    }
    out[(b * NH + h) * hs + tid] = O_final / l_final;
}


/* ============================================================================
   版本4：Cooperative Flash Decoding（单 kernel 合并 split+merge）
   ----------------------------------------------------------------------------
   动机：v3 大序列是 split、merge 两个 kernel（2 次 launch，~6-8μs 固定开销）。
   本版本用 cooperative groups 的 grid.sync() 做跨 block 全局同步，仅 1 次 launch。

   限制：
    grid.sync() 要求 Grid 内所有 Block 必须同时驻留在 GPU 上。
    若总 Block 数超出硬件并发上限，cudaLaunchCooperativeKernel 将直接报错 (cudaErrorCooperativeLaunchTooLarge)。
    因此 Host 端需利用 Occupancy API 预检硬件容量，超限则直接跳过。
============================================================================ */
__global__ void attention_kv_cooperative_kernel(float* out,
                                               float* part_out, float* part_lse, float* part_max,
                                               const float* qkv, const float* kv_cache,
                                               int B, int seq_len, int NH, int hs, int maxT,
                                               int S, float scale) {
    cg::grid_group grid = cg::this_grid(); // 获取整个 Grid 的句柄，用于全局屏障同步
    const int b = blockIdx.x / (NH * S);
    const int h = (blockIdx.x / S) % NH;
    const int s = blockIdx.x % S;
    const int tid = threadIdx.x;

    const int total = seq_len;
    const int chunk = CEIL_DIV(total, S);
    const int j_start = s * chunk;
    const int j_end = (j_start + chunk < total) ? j_start + chunk : total;
    const int n_local = j_end - j_start;

    __shared__ float q_s[64];
    __shared__ float scores[256];
    __shared__ float warp_max[2];
    __shared__ float exp_local[256];
    __shared__ float warp_lse[2];
    __shared__ float group_acc[256];

    if (tid < hs) { q_s[tid] = qkv[b * 3 * NH * hs + h * hs + tid]; }
    __syncthreads();

    if (n_local > 0) {
        const float* k_cache = kv_cache + (b * NH + h) * maxT * hs;
        for (int jj = tid; jj < n_local; jj += blockDim.x) {
            int j = j_start + jj;
            float sum = 0.0f;
            const float4* k4 = (const float4*)(k_cache + j * hs);
            for (int d4 = 0; d4 < hs / 4; d4++) {
                float4 kv = k4[d4];
                sum += q_s[d4*4+0]*kv.x + q_s[d4*4+1]*kv.y
                     + q_s[d4*4+2]*kv.z + q_s[d4*4+3]*kv.w;
            }
            scores[jj] = sum * scale;
        }
        __syncthreads();

        if (tid < hs) {
            float mv = -FLT_MAX;
            for (int jj = tid; jj < n_local; jj += hs) { mv = fmaxf(mv, scores[jj]); }
            for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                mv = fmaxf(mv, __shfl_xor_sync(0xffffffff, mv, off));
            }
            if (tid % WARP_SIZE == 0) { warp_max[tid / WARP_SIZE] = mv; }
        }
        __syncthreads();
        const float local_max = fmaxf(warp_max[0], warp_max[1]);

        if (tid < hs) {
            float ls = 0.0f;
            for (int jj = tid; jj < n_local; jj += hs) {
                float ev = expf(scores[jj] - local_max);
                exp_local[jj] = ev;
                ls += ev;
            }
            for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                ls += __shfl_xor_sync(0xffffffff, ls, off);
            }
            if (tid % WARP_SIZE == 0) { warp_lse[tid / WARP_SIZE] = ls; }
        }
        __syncthreads();
        const float local_lse = warp_lse[0] + warp_lse[1];

        {
            const int group = tid / hs;
            const int chan = tid % hs;
            const float* v_cache = kv_cache + B * NH * maxT * hs + (b * NH + h) * maxT * hs;
            const int gchunk = CEIL_DIV(n_local, 4);
            const int gstart = group * gchunk;
            const int gend = (gstart + gchunk < n_local) ? gstart + gchunk : n_local;
            float acc = 0.0f;
            for (int jj = gstart; jj < gend; jj++) {
                acc += exp_local[jj] * v_cache[(j_start + jj) * hs + chan];
            }
            group_acc[tid] = acc;
        }
        __syncthreads();

        if (tid < hs) {
            float acc = 0.0f;
            for (int g = 0; g < 4; g++) { acc += group_acc[g * hs + tid]; }
            part_out[((b * NH + h) * S + s) * hs + tid] = acc;
            if (tid == 0) {
                part_max[(b * NH + h) * S + s] = local_max;
                part_lse[(b * NH + h) * S + s] = local_lse;
            }
        }
    } else {
        // 空段：写中性值，不能提前 return，必须走到 grid.sync，否则死锁
        if (tid == 0) {
            part_max[(b * NH + h) * S + s] = -FLT_MAX;
            part_lse[(b * NH + h) * S + s] = 0.0f;
        }
        if (tid < hs) { part_out[((b * NH + h) * S + s) * hs + tid] = 0.0f; }
    }

    // ---- 跨 block 全局同步：确保所有段的 Split 计算全部完成 ----
    grid.sync();

    // ---- merge 阶段：每头只让 s==0 的 block（64 线程）做跨段合并 ----
    if (s == 0 && tid < hs) {
        float m_global = -FLT_MAX;
        for (int sp = 0; sp < S; sp++) { m_global = fmaxf(m_global, part_max[(b * NH + h) * S + sp]); }
        float O_final = 0.0f;
        float l_final = 0.0f;
        for (int sp = 0; sp < S; sp++) {
            float rescale = expf(part_max[(b * NH + h) * S + sp] - m_global);
            O_final += rescale * part_out[((b * NH + h) * S + sp) * hs + tid];
            l_final += rescale * part_lse[(b * NH + h) * S + sp];
        }
        out[(b * NH + h) * hs + tid] = O_final / l_final;
    }
}


/* ----------------------------------------------------------------------------
   版本5：Flash Decoding++（对照实验）
   ----------------------------------------------------------------------------
   背景：v3/v4 各 split 先算「局部 max」，merge 需要 rescale 合并。
   核心改进：先确定全局统一 max，merge 退化为简单求和。但增加了 kernel 数和读写。
   ---------------------------------------------------------------------------- */
__global__ void attention_kv_premax_kernel(float* global_scores, float* global_max,
                                           const float* qkv, const float* kv_cache,
                                           int B, int seq_len, int NH, int hs, int maxT,
                                           float scale) {
    const int b = blockIdx.x / NH;
    const int h = blockIdx.x % NH;
    const int tid = threadIdx.x;

    __shared__ float q_s[64];
    __shared__ float warp_max[2];
    if (tid < hs) { q_s[tid] = qkv[b * 3 * NH * hs + h * hs + tid]; }
    __syncthreads();

    const float* k_cache = kv_cache + (b * NH + h) * maxT * hs;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float sum = 0.0f;
        const float4* k4 = (const float4*)(k_cache + j * hs);
        for (int d4 = 0; d4 < hs / 4; d4++) {
            float4 kv = k4[d4];
            sum += q_s[d4 * 4 + 0] * kv.x;
            sum += q_s[d4 * 4 + 1] * kv.y;
            sum += q_s[d4 * 4 + 2] * kv.z;
            sum += q_s[d4 * 4 + 3] * kv.w;
        }
        global_scores[(b * NH + h) * maxT + j] = sum * scale; // 写回全局内存，带来额外带宽开销
    }
    __syncthreads();

    if (tid < hs) {
        float mv = -FLT_MAX;
        for (int j = tid; j < seq_len; j += hs) {
            mv = fmaxf(mv, global_scores[(b * NH + h) * maxT + j]);
        }
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffff, mv, off));
        }
        if (tid % WARP_SIZE == 0) { warp_max[tid / WARP_SIZE] = mv; }
    }
    __syncthreads();
    if (tid == 0) { global_max[b * NH + h] = fmaxf(warp_max[0], warp_max[1]); }
}

__global__ void attention_kv_split_unified_kernel(float* part_out, float* part_lse,
                                                  const float* global_scores,
                                                  const float* global_max,
                                                  const float* kv_cache,
                                                  int B, int seq_len, int NH, int hs,
                                                  int maxT, int S) {
    const int b = blockIdx.x / (NH * S);
    const int h = (blockIdx.x / S) % NH;
    const int s = blockIdx.x % S;
    const int tid = threadIdx.x;

    const float m_unified = global_max[b * NH + h]; // 读入统一全局 Max

    const int total = seq_len;
    const int chunk = CEIL_DIV(total, S);
    const int j_start = s * chunk;
    const int j_end = (j_start + chunk < total) ? j_start + chunk : total;
    const int n_local = j_end - j_start;

    if (n_local <= 0) {
        if (tid < hs) { part_out[((b * NH + h) * S + s) * hs + tid] = 0.0f; }
        if (tid == 0) { part_lse[(b * NH + h) * S + s] = 0.0f; }
        return;
    }

    __shared__ float exp_local[256];
    __shared__ float warp_lse[2];
    __shared__ float group_acc[256];

    if (tid < hs) {
        float ls = 0.0f;
        for (int jj = tid; jj < n_local; jj += hs) {
            float ev = expf(global_scores[(b * NH + h) * maxT + (j_start + jj)] - m_unified);
            exp_local[jj] = ev;
            ls += ev;
        }
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            ls += __shfl_xor_sync(0xffffffff, ls, off);
        }
        if (tid % WARP_SIZE == 0) { warp_lse[tid / WARP_SIZE] = ls; }
    }
    __syncthreads();
    const float lse_local = warp_lse[0] + warp_lse[1];

    {
        const int group = tid / hs;
        const int chan = tid % hs;
        const float* v_cache = kv_cache + B * NH * maxT * hs + (b * NH + h) * maxT * hs;
        const int gchunk = CEIL_DIV(n_local, 4);
        const int gstart = group * gchunk;
        const int gend = (gstart + gchunk < n_local) ? gstart + gchunk : n_local;
        float acc = 0.0f;
        for (int jj = gstart; jj < gend; jj++) {
            acc += exp_local[jj] * v_cache[(j_start + jj) * hs + chan];
        }
        group_acc[tid] = acc;
    }
    __syncthreads();

    if (tid < hs) {
        float acc = 0.0f;
        for (int g = 0; g < 4; g++) { acc += group_acc[g * hs + tid]; }
        part_out[((b * NH + h) * S + s) * hs + tid] = acc;
        if (tid == 0) { part_lse[(b * NH + h) * S + s] = lse_local; }
    }
}

__global__ void attention_kv_merge_unified_kernel(float* out, const float* part_out,
                                                  const float* part_lse,
                                                  int NH, int hs, int S) {
    const int b = blockIdx.x / NH;
    const int h = blockIdx.x % NH;
    const int tid = threadIdx.x;
    if (tid >= hs) return;

    // 统一 max 下，merge 直接求和，无 rescale 计算
    float O_final = 0.0f;
    for (int s = 0; s < S; s++) { O_final += part_out[((b * NH + h) * S + s) * hs + tid]; }
    float lse_final = 0.0f;
    for (int s = 0; s < S; s++) { lse_final += part_lse[(b * NH + h) * S + s]; }
    out[(b * NH + h) * hs + tid] = O_final / lse_final;
}


/* ============================================================================
   版本6：FlashAttention （在线 softmax + cp.async 双缓冲）
   ----------------------------------------------------------------------------
   优化重点：
     1. TILE 放大到 64，减少流水线的同步开销和循环分支。
     2. 废弃了冗余的共享内存中转，m_state、l_state 以及输出 o_acc，全部提升为寄存器级别维护（仅前 64 个线程持有）。
     3. 共享内存改为动态分配（extern __shared__），突破 48KB 的静态上限。
============================================================================ */

__device__ __forceinline__ void cp_async_cg_f4(float* smem, const float* gmem) {
    // cp.async 是 Ampere+ 架构的硬件异步 DMA 指令，绕过 L1 寄存器，直接内存对拷
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"((unsigned int)__cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__global__ void attention_kv_v6_split_kernel(float* part_out, float* part_lse, float* part_max,
                                             const float* qkv, const float* kv_cache,
                                             int B, int seq_len, int NH, int hs, int maxT, int S, float scale) {
    const int b = blockIdx.x / (NH * S);
    const int h = (blockIdx.x / S) % NH;
    const int s = blockIdx.x % S;
    const int tid = threadIdx.x;

    const int total = seq_len;
    const int chunk = CEIL_DIV(total, S);
    const int j_start = s * chunk;
    const int j_end = (j_start + chunk < total) ? j_start + chunk : total;
    const int n_local = j_end - j_start;

    if (n_local <= 0) {
        if (tid == 0) {
            part_max[(b * NH + h) * S + s] = -FLT_MAX;
            part_lse[(b * NH + h) * S + s] = 0.f;
        }
        if (tid < hs) part_out[((b * NH + h) * S + s) * hs + tid] = 0.f;
        return;
    }

    // 寄存器级状态维护，仅被前 64 个线程使用，极大降低访存开销
    float m_state_reg = -FLT_MAX;
    float l_state_reg = 0.0f;
    float o_acc_reg = 0.0f;

    const int TILE = 64;
    extern __shared__ float sram[]; // 动态共享内存，突破 48KB 的静态限制

    // 手动划界共享内存
    float* q_s = sram;
    float* k_tile = q_s + 64;
    float* v_tile = k_tile + 2 * TILE * 64;
    float* s_tile = v_tile + 2 * TILE * 64;
    float* p_tile = s_tile + TILE;
    float* g_acc  = p_tile + TILE;
    float* w2     = g_acc + 256;

    if (tid < hs) {
        // Q 向量化加载
        const float4* q4 = (const float4*)(qkv + b*3*NH*hs + h*hs);
        float4* qs4 = (float4*)q_s;
        qs4[tid/4] = q4[tid/4];
    }

    const float* k_base = kv_cache + (b*NH+h)*maxT*hs;
    const float* v_base = kv_cache + B*NH*maxT*hs + (b*NH+h)*maxT*hs;
    const int nb = CEIL_DIV(n_local, TILE);

    // 预取第一个 tile 的 K/V 数据
    {
        int tn0 = (n_local < TILE) ? n_local : TILE;
        int nf4 = tn0 * (hs/4);
        for (int x = tid; x < nf4; x += 256) {
            int jj = x/16, d4 = x%16;
            cp_async_cg_f4(k_tile + 0*TILE*64 + jj*hs + d4*4, k_base + (j_start + jj)*hs + d4*4);
            cp_async_cg_f4(v_tile + 0*TILE*64 + jj*hs + d4*4, v_base + (j_start + jj)*hs + d4*4);
        }
        cp_async_commit(); // 提交 DMA 搬运任务
    }

    for (int bi = 0; bi < nb; bi++) {
        const int cur = bi & 1;
        const int nxt = cur ^ 1;
        const int t0 = bi * TILE;
        int tn = TILE;
        if (n_local - t0 < TILE) tn = n_local - t0;

        // 异步预取下一个 tile 的 K/V 数据，与当前 tile 计算重叠
        if (bi + 1 < nb) {
            int t0n = (bi+1)*TILE;
            int tnn = (n_local - t0n < TILE) ? n_local - t0n : TILE;
            int nf4n = tnn * (hs/4);
            for (int x = tid; x < nf4n; x += 256) {
                int jj = x/16, d4 = x%16;
                cp_async_cg_f4(k_tile + nxt*TILE*64 + jj*hs + d4*4, k_base + (j_start + t0n + jj)*hs + d4*4);
                cp_async_cg_f4(v_tile + nxt*TILE*64 + jj*hs + d4*4, v_base + (j_start + t0n + jj)*hs + d4*4);
            }
            cp_async_commit();  // 提交下一块搬运
            cp_async_wait<1>(); // 等待当前块完成，保留下一块在途（真正的流水线重叠）
        } else {
            cp_async_wait<0>(); // 最后一块，等待所有搬运完成
        }
        __syncthreads();

        // 阶段 1：打分
        if (tid < tn) {
            float sum = 0.f;
            const float4* k4 = (const float4*)(k_tile + cur*TILE*64 + tid * hs);
            const float4* q4 = (const float4*)q_s;
            for (int d4 = 0; d4 < hs/4; d4++) {
                float4 kv = k4[d4];
                float4 qv = q4[d4];
                sum += qv.x*kv.x + qv.y*kv.y + qv.z*kv.z + qv.w*kv.w;
            }
            s_tile[tid] = sum * scale;
        }
        __syncthreads();

        // 阶段 2：求当前 tile 的局部最大 m_new
        float mv = -FLT_MAX;
        if (tid < hs) {
            for (int jj = tid; jj < tn; jj += hs) mv = fmaxf(mv, s_tile[jj]);
            for (int off=16; off>0; off>>=1)
                mv = fmaxf(mv, __shfl_xor_sync(0xffffffff, mv, off));
            if (tid % 32 == 0) w2[tid / 32] = mv;
        }
        __syncthreads();
        float m_new = fmaxf(w2[0], w2[1]); // 合并两个 warp 的局部最大值

        // 阶段 3：更新 p_tile，并对寄存器状态进行 Rescale (在线 Softmax 核心)
        float lv = 0.f;
        if (tid < hs) {
            float r_old = expf(m_state_reg - m_new); // 计算旧状态的缩放系数
            o_acc_reg *= r_old;
            l_state_reg *= r_old;
            m_state_reg = m_new;

            for (int jj = tid; jj < tn; jj += hs) {
                float e = expf(s_tile[jj] - m_new);
                p_tile[jj] = e;
                lv += e;
            }
            for (int off=16; off>0; off>>=1)
                lv += __shfl_xor_sync(0xffffffff, lv, off);
            if (tid % 32 == 0) w2[tid / 32] = lv;
        }
        __syncthreads();
        if (tid < hs) {
            l_state_reg += w2[0] + w2[1]; // 累加当前的 l
        }

        // 阶段 4：4 组并行加权 V
        {
            int group = tid / hs;
            int chan = tid % hs;
            int gchunk = CEIL_DIV(tn, 4);
            int gstart = group * gchunk;
            int gend = (gstart + gchunk < tn) ? gstart + gchunk : tn;
            float a = 0.f;
            for (int jj = gstart; jj < gend; jj++)
                a += p_tile[jj] * v_tile[cur*TILE*64 + jj*hs + chan];
            g_acc[tid] = a;
        }
        __syncthreads();

        // 阶段 5：累加进输出寄存器
        if (tid < hs) {
            float a = 0.f;
            for (int g = 0; g < 4; g++) a += g_acc[g*hs + tid];
            o_acc_reg += a;
        }
    }

    if (tid < hs) part_out[((b * NH + h) * S + s) * hs + tid] = o_acc_reg;
    if (tid == 0) {
        part_max[(b * NH + h) * S + s] = m_state_reg;
        part_lse[(b * NH + h) * S + s] = l_state_reg;
    }
}


// ============================================================================
// host 封装：按版本号启动对应 kernel
// ============================================================================
void attention_kv_forward(int version, float* out, const float* qkv, const float* kv_cache,
                          int B, int seq_len, int NH, int hs, int maxT, cudaStream_t stream,
                          float* part_out, float* part_lse, float* part_max, int S,
                          float* global_scores, float* global_max) {
    float scale = 1.f / sqrtf(hs);
    switch (version) {
        case 1: attention_kv_kernel<<<B * NH, hs, 0, stream>>>(out, qkv, kv_cache, B, seq_len, NH, hs, maxT, scale); break;
        case 2: attention_kv_fused_v2_kernel<<<B * NH, 256, 0, stream>>>(out, qkv, kv_cache, B, seq_len, NH, hs, maxT, scale); break;
        case 3: {
            // v3 走完整 Flash Decoding 路径（split + merge 两 kernel）
            attention_kv_split_kernel<<<B * NH * S, 256, 0, stream>>>(
                part_out, part_lse, part_max, qkv, kv_cache,
                B, seq_len, NH, hs, maxT, S, scale);
            attention_kv_merge_kernel<<<B * NH, 64, 0, stream>>>(
                out, part_out, part_lse, part_max, NH, hs, S);
            break;
        }
        case 4: {
            // v4 走完整 cooperative 路径（单 kernel，grid.sync 合并 split+merge）。
            dim3 grid(B * NH * S), block(256);
            void* args[] = {
                (void*)&out, (void*)&part_out, (void*)&part_lse, (void*)&part_max,
                (void*)&qkv, (void*)&kv_cache, (void*)&B, (void*)&seq_len,
                (void*)&NH, (void*)&hs, (void*)&maxT, (void*)&S, (void*)&scale
            };
            cudaError_t err = cudaLaunchCooperativeKernel(
                (void*)attention_kv_cooperative_kernel, grid, block, args, 0, stream);

            // v4 依赖于整个 Grid 的所有 Block 同时驻留。若超出硬件并发上限，
            // 需捕获错误并清理状态，避免影响后续 Kernel 启动。外层已提前拦截。
            if (err != cudaSuccess) {
                cudaGetLastError();
            }
            break;
        }
        case 5: {
            // v5 走完整 Flash Decoding++ 路径（premax + split + merge 三 kernel）
            attention_kv_premax_kernel<<<B * NH, 256, 0, stream>>>(
                global_scores, global_max, qkv, kv_cache,
                B, seq_len, NH, hs, maxT, scale);
            attention_kv_split_unified_kernel<<<B * NH * S, 256, 0, stream>>>(
                part_out, part_lse, global_scores, global_max, kv_cache,
                B, seq_len, NH, hs, maxT, S);
            attention_kv_merge_unified_kernel<<<B * NH, 64, 0, stream>>>(
                out, part_out, part_lse, NH, hs, S);
            break;
        }
        case 6: {
            // v6 走完整 FlashAttention 流水路径
            const int TILE = 64;
            size_t smem_size = (64 + 2*TILE*64 + 2*TILE*64 + TILE + TILE + 256 + 2) * sizeof(float);

            // 突破硬件默认的 48KB 限制，提前设置动态共享内存上限
            cudaFuncSetAttribute(attention_kv_v6_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            attention_kv_v6_split_kernel<<<B * NH * S, 256, smem_size, stream>>>(
                part_out, part_lse, part_max, qkv, kv_cache,
                B, seq_len, NH, hs, maxT, S, scale);
            attention_kv_merge_kernel<<<B * NH, 64, 0, stream>>>(
                out, part_out, part_lse, part_max, NH, hs, S);
            break;
        }
        default:
            fprintf(stderr, "无效版本号：%d（可选 1/2/3/4/5/6）\n", version);
            exit(1);
    }
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// 数据生成与校验工具
// ============================================================================

// 生成 n 个 [-1, 1) 的随机浮点数
float* make_random_float(int n) {
    float* arr = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 10000) / 5000.0f - 1.0f;
    }
    return arr;
}

// 校验：打印该 seq_len 下的最大绝对误差，统计超差元素数
int validate_result(int seq_len, const float* got, const float* want, int n, float tol) {
    float max_abs = 0.0f;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(got[i] - want[i]);
        if (diff > max_abs) max_abs = diff;
        if (diff > tol) bad++;
    }
    printf("  seq_len=%4d | 最大绝对误差 %.3e%s\n", seq_len, max_abs, bad ? "  <-- 有超差" : "");
    return bad;
}

// 升序比较（qsort 用）
int cmp_float(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

// 基于经验公式动态计算最优切分段数 S
// 启发式规则：每段大约包含 128 个元素，最少 4 段，最多不超过 MAX_SPLIT_S
int calculate_split_S(int seq_len) {
    int s = (seq_len + 127) / 128;
    if (s < 4) s = 4;
    if (s > MAX_SPLIT_S) s = MAX_SPLIT_S;
    return s;
}


// ============================================================================
// 主程序：正确性验证 + 性能基准（冷缓存 + 中位数）+ profile 模式
// ============================================================================
int main(int argc, char** argv) {
    srand(0); // 固定种子，结果可复现

    // 模拟 GPT-2 124M 注意力形状：B=8, NH=12, hs=64, maxT=1024
    const int B = 1, NH = 12, hs = 64, maxT = 1024;
    const int qkv_elems = B * 3 * NH * hs;          // 单 token 的 QKV
    const int cache_elems = 2 * B * NH * maxT * hs; // K 区 + V 区
    const int out_elems = B * NH * hs;

    int version = 0; // 0 = 全部版本
    if (argc > 1) { version = atoi(argv[1]); }
    int profile = (argc > 2 && strcmp(argv[2], "profile") == 0);

    cudaCheck(cudaSetDevice(0));
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, 0));
    printf("设备：%s\n\n", prop.name);

    // ---- 动态检测 v4 硬件限制 ----
    // v4 (Cooperative Launch) 要求 Grid 内所有 Block 必须能同时驻留在 GPU 上。
    // 预先计算当前硬件的最大驻留容量，如果不满足直接跳过，避免刷屏与无效计时。
    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, (const void*)attention_kv_cooperative_kernel, 256, 0);
    int max_active_blocks = max_blocks_per_sm * prop.multiProcessorCount;

    printf("  [提示] v4 硬件并行驻留上限为 %d 个 Block。\n\n", max_active_blocks);

    // ---- 造数据（全部随机，不加载任何权重）----
    float* qkv = make_random_float(qkv_elems);
    float* kv_cache = make_random_float(cache_elems);
    float* out_cpu = (float*)malloc(out_elems * sizeof(float));
    float* out_gpu_h = (float*)malloc(out_elems * sizeof(float));

    float* d_qkv; float* d_kv_cache; float* d_out;
    cudaCheck(cudaMalloc(&d_qkv, qkv_elems * sizeof(float)));
    cudaCheck(cudaMalloc(&d_kv_cache, cache_elems * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, out_elems * sizeof(float)));

    // 部分结果缓冲：根据 MAX_SPLIT_S 预分配安全空间
    float* d_part_out; float* d_part_lse; float* d_part_max;
    cudaCheck(cudaMalloc(&d_part_out, B * NH * MAX_SPLIT_S * hs * sizeof(float)));
    cudaCheck(cudaMalloc(&d_part_lse, B * NH * MAX_SPLIT_S * sizeof(float)));
    cudaCheck(cudaMalloc(&d_part_max, B * NH * MAX_SPLIT_S * sizeof(float)));

    // 全局 scores 与全局 max 缓冲
    float* d_global_scores; float* d_global_max;
    cudaCheck(cudaMalloc(&d_global_scores, B * NH * maxT * sizeof(float)));
    cudaCheck(cudaMalloc(&d_global_max, B * NH * sizeof(float)));
    cudaCheck(cudaMemcpy(d_qkv, qkv, qkv_elems * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_kv_cache, kv_cache, cache_elems * sizeof(float), cudaMemcpyHostToDevice));

    int versions[6] = {1, 2, 3, 4, 5, 6};
    const char* vnames[6] = {"朴素版", "融合版 v2", "Flash Decoding v3",
                             "Cooperative v4", "Flash Decoding++ v5", "FlashAttention v6"};
    const char* enames[6] = {"naive v1", "fused v2", "flash-dec v3",
                             "coop v4", "flash-dec++ v5", "flash-attn v6"};

    // 如果不是 profile 模式，则执行标准验证与测试流程
    if (!profile) {
        // ---- 正确性验证：对代表性 seq_len 与 CPU 参考对比 ----
        int check_seq_lens[] = {1, 64, 256, 512, 1024};
        int n_checks = sizeof(check_seq_lens) / sizeof(int);
        for (int vi = 0; vi < 6; vi++) {
            int v = versions[vi];
            if (version != 0 && version != v) continue;

            printf("=== 版本 %d（%s）正确性 ===\n", v, vnames[vi]);
            int total_bad = 0;
            for (int ci = 0; ci < n_checks; ci++) {
                int cl = check_seq_lens[ci];
                int dynamic_S = calculate_split_S(cl);

                // 动态检测 v4 是否能在此特定 S 下运行
                if (v == 4) {
                    int grid_blocks = B * NH * dynamic_S;
                    if (grid_blocks > max_active_blocks) {
                        printf("  seq_len=%4d | 跳过 v4 (需要 %d Blocks > 硬件上限 %d)\n", cl, grid_blocks, max_active_blocks);
                        continue;
                    }
                }

                attention_kv_cpu(out_cpu, qkv, kv_cache, B, cl, NH, hs, maxT, 1.f / sqrtf(hs));
                attention_kv_forward(v, d_out, d_qkv, d_kv_cache, B, cl, NH, hs, maxT, 0,
                                     d_part_out, d_part_lse, d_part_max, dynamic_S,
                                     d_global_scores, d_global_max);
                cudaCheck(cudaDeviceSynchronize());
                cudaCheck(cudaMemcpy(out_gpu_h, d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost));
                total_bad += validate_result(cl, out_gpu_h, out_cpu, out_elems, 1e-4f);
            }
            printf("  -> %s\n\n", total_bad ? "正确性未通过！" : "全部通过");
        }

        // ---- 性能基准：冷缓存（每次刷新 L2）+ 单次计时 + 中位数 ----
        int bench_seq_lens[] = {64, 256, 512, 1024};
        int n_bench = sizeof(bench_seq_lens) / sizeof(int);
        const int repeats = 100;

        // 确定要测的版本列表
        int test_vers[6]; int n_test = 0;
        for (int vi = 0; vi < 6; vi++) {
            if (version == 0 || version == versions[vi]) {
                test_vers[n_test++] = versions[vi];
            }
        }

        void* flush_buf;
        cudaCheck(cudaMalloc(&flush_buf, prop.l2CacheSize));

        cudaEvent_t t0, t1;
        cudaCheck(cudaEventCreate(&t0));
        cudaCheck(cudaEventCreate(&t1));

        printf("=== 性能基准（冷缓存，每点重复 %d 次取中位数，单位: us）===\n\n", repeats);

        printf("%7s", "seq_len");
        for (int ti = 0; ti < n_test; ti++) {
            printf(" | %14s", enames[test_vers[ti] - 1]);
        }
        printf("\n");

        printf("%7s", "-------");
        for (int ti = 0; ti < n_test; ti++) {
            printf(" | %14s", "--------------");
        }
        printf("\n");

        // 数据行
        for (int ci = 0; ci < n_bench; ci++) {
            int cl = bench_seq_lens[ci];
            int dynamic_S = calculate_split_S(cl);
            printf("%7d", cl);
            for (int ti = 0; ti < n_test; ti++) {
                int v = test_vers[ti];

                if (v == 4) {
                    int grid_blocks = B * NH * dynamic_S;
                    if (grid_blocks > max_active_blocks) {
                        printf(" | %14s", "N/A (爆)");
                        continue;
                    }
                }

                float* samples = (float*)malloc(repeats * sizeof(float));
                for (int r = 0; r < repeats; r++) {
                    cudaCheck(cudaMemset(flush_buf, 0, prop.l2CacheSize)); // 刷新 L2 缓存
                    cudaCheck(cudaEventRecord(t0));
                    attention_kv_forward(v, d_out, d_qkv, d_kv_cache, B, cl, NH, hs, maxT, 0,
                                         d_part_out, d_part_lse, d_part_max, dynamic_S,
                                         d_global_scores, d_global_max);
                    cudaCheck(cudaEventRecord(t1));
                    cudaCheck(cudaEventSynchronize(t1));
                    float ms = 0.0f;
                    cudaCheck(cudaEventElapsedTime(&ms, t0, t1));
                    samples[r] = ms * 1000.0f;
                }
                qsort(samples, repeats, sizeof(float), cmp_float);
                float med = samples[repeats / 2];

                printf(" | %14.2f", med);
                free(samples);
            }
            printf("\n");
        }
        printf("\n（冷缓存中位数：抗后台干扰与异常值，可复现；小 seq_len 时 launch 开销占比较大）\n");

        // 释放资源
        cudaCheck(cudaFree(flush_buf));
        cudaCheck(cudaEventDestroy(t0));
        cudaCheck(cudaEventDestroy(t1));
        free(qkv); free(kv_cache); free(out_cpu); free(out_gpu_h);
        cudaCheck(cudaFree(d_qkv));
        cudaCheck(cudaFree(d_kv_cache));
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_part_out));
        cudaCheck(cudaFree(d_part_lse));
        cudaCheck(cudaFree(d_part_max));
        cudaCheck(cudaFree(d_global_scores));
        cudaCheck(cudaFree(d_global_max));
        return 0;
    }

    // ========== 性能采集模式（配合 NCU / NSYS 使用） ==========
    {
        const int profile_seq_len = 512;
        const int dynamic_S = calculate_split_S(profile_seq_len);
        const int warmup_rounds = 2;

        // 预热 2 轮
        for (int round = 0; round < warmup_rounds; round++) {
            for (int ver = 1; ver <= 6; ver++) {
                if (ver == 4 && (B * NH * dynamic_S > max_active_blocks)) continue;
                attention_kv_forward(ver, d_out, d_qkv, d_kv_cache, B, profile_seq_len, NH, hs, maxT, 0,
                                     d_part_out, d_part_lse, d_part_max, dynamic_S,
                                     d_global_scores, d_global_max);
            }
        }
        cudaCheck(cudaDeviceSynchronize());

        // 正式采集 1 轮
        for (int ver = 1; ver <= 6; ver++) {
            if (ver == 4 && (B * NH * dynamic_S > max_active_blocks)) continue;
            printf("正在采集版本 %d ...\n", ver);
            attention_kv_forward(ver, d_out, d_qkv, d_kv_cache, B, profile_seq_len, NH, hs, maxT, 0,
                                 d_part_out, d_part_lse, d_part_max, dynamic_S,
                                 d_global_scores, d_global_max);
        }
        cudaCheck(cudaDeviceSynchronize());

        free(qkv); free(kv_cache); free(out_cpu); free(out_gpu_h);
        cudaCheck(cudaFree(d_qkv));
        cudaCheck(cudaFree(d_kv_cache));
        cudaCheck(cudaFree(d_out));
        cudaCheck(cudaFree(d_part_out));
        cudaCheck(cudaFree(d_part_lse));
        cudaCheck(cudaFree(d_part_max));
        cudaCheck(cudaFree(d_global_scores));
        cudaCheck(cudaFree(d_global_max));
        return 0;
    }
}