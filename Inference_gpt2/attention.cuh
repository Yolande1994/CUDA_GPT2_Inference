/* ============================================================================
   attention.cuh —— 自注意力（无 KV 缓存的完整版，基线程序使用）
   ============================================================================
   自注意力（self-attention）做什么：
     每个 token 拿着自己的查询向量 Q（query，提问），去和序列里所有 token 的键向量 K（key，标签）做比对，得到一组“关注程度”打分；
     打分经 softmax 归一化成权重，再对所有值向量 V（value，内容）做加权求和，输出就是这个 token 融合了上下文之后的新向量。

   本文件一次性处理整段序列的完整 attention，流程为：
     1. 拆分：QKV 投影输出是三块交织在一起的，拆成独立的 q、k、v，并把“头”维度提前（permute）；
     2. 打分：Q·Kᵀ / √(头维度)，用 cublasLt 批量矩阵乘；
     3. 掩码 softmax：因果掩码让每个位置只能看见自己之前的 token，在线 softmax 保证数值稳定；
     4. 加权：打分 × V，同样用 cublasLt；
     5. 合回：把布局还原（unpermute）。
   逐 token 解码的优化版（带 KV 缓存）在 attention_kvcache.cuh 中。
   ============================================================================ */
#pragma once
#include <assert.h>
#include "common.h"

// ============================================================================
// 布局变换：QKV 交织布局 → 独立的 q/k/v
// ============================================================================

/* QKV 投影输出的内存顺序是 [B, T, 3(Q/K/V), NH, hs]，而矩阵乘要求 q、k、v 各自连续、顺序为 [B, NH, T, hs]。
   这里把目标下标拆成 (b, 头, 位置, 通道)，再换算回交织布局里的地址，把三个值分别搬到 q、k、v。 */
__global__ void permute_kernel(floatX* q, floatX* k, floatX* v,
                               const floatX* inp,
                               int B, int N, int NH, int hs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 目标布局里的扁平下标
    if (idx >= B * NH * N * hs) { return; }  // 越界退出
    int b = idx / (NH * N * hs);
    int rest = idx % (NH * N * hs);
    int nh_ = rest / (N * hs);
    rest = rest % (N * hs);
    int n = rest / hs;
    int d_ = rest % hs;
    // 换算回交织布局 [B,T,3,NH,hs] 里的地址
    int inp_idx = (b * N * 3 * NH * hs) + (n * 3 * NH * hs) + (0 * NH * hs) + (nh_ * hs) + d_;
    q[idx] = __ldcs(&inp[inp_idx]);  // 同一交织位置上偏移 0/1/2 份即 Q/K/V
    k[idx] = __ldcs(&inp[inp_idx + NH * hs]);
    v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * hs)]);
}

// permute 的逆操作：[B, NH, T, hs] 还原回 [B, T, NH, hs]
__global__ void unpermute_kernel(floatX* inp, floatX *out, int B, int N, int NH, int hs) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= B * NH * N * hs) { return; }
    int b = idx / (NH * N * hs);
    int rest = idx % (NH * N * hs);
    int nh_ = rest / (N * hs);
    rest = rest % (N * hs);
    int n = rest / hs;
    int d_ = rest % hs;
    int other_idx = (b * NH * N * hs) + (n * NH * hs) + (nh_ * hs) + d_;  // 目标交织布局地址
    out[other_idx] = __ldcs(&inp[idx]);  // 从头连续布局搬回交织布局
}

// ============================================================================
// 因果掩码 + 在线 softmax
// ============================================================================

/* ----------------------------------------------------------------------------
   一行（一个查询位置对所有位置的打分）由一个 warp 处理。
   两个关键点：
     因果掩码：第 own_pos 行只处理下标 ≤ own_pos 的元素，未来位置不参与，对应“生成时不能偷看还没写出的 token”；
     在线 softmax：边扫边维护当前最大值 maxval 和指数和 sumval。每出现更大的值，就用 exp(旧max − 新max) 把旧的指数和整体修正，避免直接对大数取指数导致溢出。
                   最后再扫一遍做归一化输出。
   每次处理 4 个元素，减少循环与取指开销。
   ---------------------------------------------------------------------------- */
__global__ void softmax_forward_kernel5(floatX* out, float inv_temperature, const floatX* inp, int N, int T) {
    assert(T % 4  == 0);
    int lane_id = threadIdx.x % WARP_SIZE;  // warp 内编号
    int warp_id = threadIdx.x / WARP_SIZE;  // 块内 warp 编号
    int num_warps = blockDim.x / WARP_SIZE;
    // 反向映射行号，让相邻 block 处理的行在内存上错开
    int idx = (gridDim.x - blockIdx.x - 1) * num_warps + warp_id;
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;       // 本 warp 负责的查询位置
    int pos_by_4 = own_pos / 4;  // 前面能凑成多少个 4 元素组
    const floatX* x = inp + idx * T;  // 本行的打分起点
    const float flt_max = 340282346638528859811704183484516925440.0f;
    float maxval = -flt_max;  // 在线 softmax 维护的当前最大值
    float sumval = 0.0f;      // 在线 softmax 维护的指数和

    const floatX* x_aligned = reinterpret_cast<const floatX*>(__builtin_assume_aligned(x, 16));
    for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
        float regarray[4];
        for (int k = 0; k < 4; ++k) {
            regarray[k] = (float)x_aligned[4*i + k];  // 一次读 4 个打分
        }
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, regarray[k]);  // 更新本 lane 最大值
        }

        // 最大值更新后，旧指数和整体乘修正系数，再把新元素的指数加进来
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (regarray[k] - maxval));
        }
    }

    // 最后不足 4 个的零头位置单独处理
    if(4*pos_by_4 + lane_id <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, (float)x[4*pos_by_4 + lane_id]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * ((float)x[4*pos_by_4 + lane_id] - maxval));
    }

    float global_maxval = warpReduceMax(maxval);  // 全 warp 统一最大值，各 lane 口径一致
    sumval *= expf(inv_temperature * (maxval - global_maxval));  // 把各 lane 口径统一到全局最大

    float sum = warpReduceSum(sumval);  // 归一化分母
    float norm = 1.f / sum;

    // 用统一的最大值和分母重算一遍指数，写出归一化结果
    for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
        float ev = expf(inv_temperature * ((float)__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, (floatX)(ev * norm));
    }
}

// ============================================================================
// attention_forward 完整自注意力编排入口
// ============================================================================
void attention_forward(floatX* out, floatX* qkv, floatX* att,
                       floatX* inp,
                       int B, int T, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();

    const int block_size = 256;
    const int hs = C / NH;

    floatX *q, *k, *v;
    q = qkv + 0 * B * T * C;  // Q 段起点
    k = qkv + 1 * B * T * C;  // K 段起点
    v = qkv + 2 * B * T * C;  // V 段起点
    int total_threads = B * NH * T * hs;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size, 0, stream>>>(q, k, v, inp, B, T, NH, hs);

    floatX* preatt = inp;
    // 批量 GEMM：每个 (batch, 头) 一份 Q·Kᵀ，batch 数 = B*NH
    matmul_cublaslt(preatt, k, q, nullptr, T, T, hs, stream, true, false, B * NH, T * hs, T * hs, T * T);

    float scale = 1.f / sqrtf(hs);  // 缩放防止头维度大时点积过大、softmax 梯度消失
    int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
    softmax_forward_kernel5<<<grid_size, block_size, 0, stream>>>(att, scale, preatt, B * NH, T);
    floatX* vaccum = inp;

    // 第二个批量 GEMM：权重 × V
    matmul_cublaslt(vaccum, v, att, nullptr, hs, T, T, stream, false, false, B * NH, T * hs, T * T, T * hs);

    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size, 0, stream>>>(vaccum, out, B, T, NH, hs);  // 还原布局
    cudaCheck(cudaGetLastError());
}
