/* ============================================================================
   attention_kvcache.cuh —— 带 KV 缓存的解码注意力（BF16 存储 / FP32 累加）
   ============================================================================
   为什么需要 KV 缓存：
     解码第 t 步时，新 token 的注意力要用到前面所有 token 的 K 和 V。如果每一步都重新计算整段历史的 K、V，总工作量会随步数平方增长。
     KV 缓存把历史 token 的 K、V 存下来，每步只算新 token 自己的 K、V 并追加，历史部分直接复用。

   缓存布局（一块连续显存，K 在前、V 在后）：
     [0, B*NH*maxT*hs)               存放全部 K
     [B*NH*maxT*hs, 2*B*NH*maxT*hs)  存放全部 V
     每个 (batch, 头) 占一条长度 maxT、宽度 hs 的槽位；
     第 j 个位置的 K 从 (b*NH+h)*maxT*hs + j*hs 起，连续 hs 个元素；
     V 用同样公式，再整块偏移 B*NH*maxT*hs。

   三种读取策略，按序列长度和 batch 选择（各自的网格尺寸写在该 kernel 上方）：
     策略一 fused_single   一个 block 负责一个 (batch, 头)，全流程在共享内存内完成，适合短序列；
     策略二 split + merge  Flash Decoding：把序列切成 S 段，每段一个 block 并行，
                           产出「段内加权和 + 段最大值 + 段指数和」三样部分结果，
                           再由 merge kernel 按统一基准合并。长序列时更多 SM 能同时干活；
     策略三 cooperative    与 split+merge 等价，但用 grid 级同步把两步合进一个 kernel，少一次启动；无法被 CUDA Graph 捕获。

   线程分工约定（三种策略一致，hs 取 64）：
     一个 (batch, 头) 固定配 256 个线程 = 8 个 warp × 32，恰好等于 4 组 × hs；
     tid < hs 的线程一人负责一个通道，做归约和写回；
     全部 256 线程按 tid / hs 分成 4 组，每组负责一段位置对 V 加权，用 ILP 掩盖访存延迟。
   ============================================================================ */
#pragma once

#include <float.h>
#include <cooperative_groups.h>
#include "common.h"

namespace cg = cooperative_groups;

// 设备端当前序列长度：每一步由主机 cudaMemcpyToSymbol 更新。
// 做成设备变量而不是 kernel 参数，是为了 CUDA Graph 重放时长度仍能逐 step 变化。
#ifndef KV_G_CUR_LEN_DEFINED
#define KV_G_CUR_LEN_DEFINED
__device__ int g_cur_len = 0;
#endif

// ============================================================================
// KV 缓存写入：把当前 token 的 K、V 追加到缓存的第 cur_len 个位置
//   网格：CEIL_DIV(B*NH*hs, 256) 个 block × 256 线程，一个线程写一个元素
// ============================================================================
__global__ void kv_cache_write_kernel(floatX* kv_cache, const floatX* qkv,
                                      int B, int NH, int hs, int maxT) {
    const int cur_len = g_cur_len;                      // 本步要写的列号 = 已生成的历史长度
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局元素号 0..B*NH*hs-1
    if (idx >= B * NH * hs) { return; }

    // 把全局元素号拆成 (batch, 头, 通道)
    int b = idx / (NH * hs);
    int rest = idx % (NH * hs);
    int h = rest / hs;
    int d_ = rest % hs;

    // 源是当前 token 的 QKV 交织数据，目标是缓存槽位中的第 cur_len 列
    int src = (b * 3 * NH * hs) + h * hs + d_;               // QKV 里 Q 段起点
    int dst = (b * NH + h) * maxT * hs + cur_len * hs + d_; // 槽位第 cur_len 列

    kv_cache[dst] = qkv[src + NH * hs];                             // K 段：紧跟 Q 之后
    kv_cache[B * NH * maxT * hs + dst] = qkv[src + 2 * NH * hs];  // V 段：整块偏移一份
}

// ============================================================================
// 策略一：fused single —— 单个 block 完成一个 (batch, 头) 的全部计算
//   网格：B*NH 个 block × 256 线程，blockIdx.x = b*NH + h
// ============================================================================

/* ----------------------------------------------------------------------------
   256 个线程分四步走，中间数据全部落在共享内存：
     1. 算出当前 Q 与每个历史位置 K 的打分（128-bit 向量加载）；
     2. 两个 warp 各归约一个局部最大，再合并成全头最大值（数值稳定所需）；
     3. 用这个最大值算指数，得到每个位置的 softmax 分子，同时累加出 softmax 分母；
     4. 线程分成 4 组各管一段位置，对 V 加权求和后再汇总，除分母。
   共享内存 scores 上限 1024，因此只用于短序列；更长的序列走策略二。
   ---------------------------------------------------------------------------- */
__global__ void attention_kv_fused_single_kernel(floatX* out, const floatX* qkv,
                                                 const floatX* kv_cache,
                                                 int B, int NH,
                                                 int hs, int maxT, float scale) {
    const int cur_len = g_cur_len;
    const int b = blockIdx.x / NH;  // 一维 Grid 拆成 (b, h)：先除头数得 batch
    const int h = blockIdx.x % NH;  // 余数即头号
    const int tid = threadIdx.x;
    const int total = cur_len + 1;  // 参与注意力的位置总数（含当前 token）

    __shared__ float q_s[64];        // 本头 Q 向量，搬一次供 256 线程共用
    __shared__ float scores[1024];   // 存储当前头所有历史位置的打分结果
    __shared__ float warp_max[2];    // 两个 warp 各写一个局部最大
    __shared__ float exp_buf[1024];  // 预计算的 exp(打分成 - 最大分)，即 softmax 分子
    __shared__ float warp_lse[2];    // 两个 warp 各写一个局部指数和
    __shared__ float group_acc[256]; // 4 组 × 64 通道的部分加权和（分子）

    if (tid < hs) { q_s[tid] = (float)qkv[b * 3 * NH * hs + h * hs + tid]; }  // 把当前 Q 搬进共享内存
    __syncthreads();

    // 第一步：Q 与每个历史位置的 K 做点积，乘 1/√hs 缩放
    const floatX* k_cache = kv_cache + (b * NH + h) * maxT * hs;  // 本 (batch,头) 的 K 槽位起点
    for (int j = tid; j < total; j += blockDim.x) {                // 一个线程包干若干位置，步长 = 线程数
        float sum = 0.0f;
        const floatX* k_row = k_cache + j * hs;   // 第 j 个位置的 K 向量
        for (int d4 = 0; d4 < hs / x128::size; d4++) {
            x128 kv = load128cs(k_row + d4 * x128::size);  // 128-bit 向量加载
            for (int k = 0; k < x128::size; k++) {
                sum += q_s[d4 * x128::size + k] * (float)kv[k];  // Q·K 逐元素乘加
            }
        }
        scores[j] = sum * scale;  // 缩放后存入共享内存，下标 j 就是位置号
    }
    __syncthreads();

    // 第二步：两个 warp 各自归约最大值，再取全头最大
    if (tid < hs) {
        float mv = -FLT_MAX;                                           // 中性初值：任何数都比 -inf 大
        for (int j = tid; j < total; j += hs) { mv = fmaxf(mv, scores[j]); }  // 先串行走一遍自己负责的位置
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {         // 蝶形归约：偏移 16/8/4/2/1
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffff, mv, off));   // warp 内寄存器交换，比共享内存归约快
        }
        if (tid % WARP_SIZE == 0) { warp_max[tid / WARP_SIZE] = mv; }  // 每个 warp 的 lane 0 写局部最大
    }
    __syncthreads();
    const float gmax = fmaxf(warp_max[0], warp_max[1]);  // 两个 warp 再取全头最大值

    // 第三步：统一最大值下算指数，既写出 softmax 分子又累加出 softmax 分母
    if (tid < hs) {
        float ls = 0.0f;
        for (int j = tid; j < total; j += hs) {
            float ev = expf(scores[j] - gmax);  // 减去最大值，指数不会溢出
            exp_buf[j] = ev;
            ls += ev;
        }
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            ls += __shfl_xor_sync(0xffffffff, ls, off);
        }
        if (tid % WARP_SIZE == 0) { warp_lse[tid / WARP_SIZE] = ls; }
    }
    __syncthreads();
    const float glse = warp_lse[0] + warp_lse[1];  // 全局指数和，即 softmax 分母

    // 第四步：4 组线程各管 1/4 位置对 V 加权，结果先存组内
    {
        const int group = tid / hs;  // 0..3，每组负责一段 j 范围
        const int chan = tid % hs;   // 0..hs-1，负责哪个输出通道
        const floatX* v_cache = kv_cache + B * NH * maxT * hs + (b * NH + h) * maxT * hs;  // V 槽位：K 之后整块偏移
        const int chunk = CEIL_DIV(total, 4);   // 将 j 范围均分给 4 个组
        const int js = group * chunk;           // 本组的起始任务
        const int je = (js + chunk < total) ? js + chunk : total;  // 本组结束任务，末组可能不满
        float acc = 0.0f;
        for (int j = js; j < je; j++) {
            acc += exp_buf[j] * (float)v_cache[j * hs + chan];  // 权重 × V，只有 FMA 指令
        }
        group_acc[tid] = acc;   // 按 tid 落位，tid = group*hs + chan
    }
    __syncthreads();

    // 汇总四组结果，除以指数和完成归一化
    if (tid < hs) {
        float acc = 0.0f;
        for (int g = 0; g < 4; g++) { acc += group_acc[g * hs + tid]; }  // 同一通道跨 4 组相加
        out[(b * NH + h) * hs + tid] = (floatX)(acc / glse);  // 写回最终输出
    }
}


// ============================================================================
// 策略二：split —— Flash Decoding，每个 block 只处理序列的一段
//   网格：B*NH*S 个 block × 256 线程，blockIdx.x 拆成 (b, h, s)
// ============================================================================

/* ----------------------------------------------------------------------------
   把一维 Grid 拆成 (b, h, s) 三维索引，每个 block 只看自己那一段。
   段内的计算与 fused single 完全相同，但输出的是三样「部分结果」：
     part_out  段内 V 的加权和，未归一化（数值上等于分子）；
     part_max  段内打分最大值；
     part_lse  段内指数和（数值上等于分母）。
   后两者是合并的钥匙：不同段的最大值不同，直接相加会错，必须带着它们才能在 merge 阶段按统一基准重新缩放。
   空段写中性值（max=-∞、lse=0、out=0），merge 时 exp(-inf)=0，自然被忽略。

   三组部分结果的索引（S 是切分段数）：
     part_max / part_lse : (b*NH + h)*S + s             每 (batch,头) 连续 S 个槽
     part_out            : ((b*NH + h)*S + s)*hs + d    再按通道展开一层
   ---------------------------------------------------------------------------- */
__global__ void attention_kv_split_kernel(float* part_out, float* part_lse, float* part_max,
                                          const floatX* qkv, const floatX* kv_cache,
                                          int B, int NH, int hs,
                                          int maxT, int S, float scale) {
    const int cur_len = g_cur_len;
    const int b = blockIdx.x / (NH * S); // 先除「头 × 段」，得 batch
    const int h = (blockIdx.x / S) % NH; // 再除「段」取模，得头号
    const int s = blockIdx.x % S;        // 余数即段号 0..S-1
    const int tid = threadIdx.x;

    // 把 [0, total) 均分成 S 段，取本 block 对应的起止位置
    const int total = cur_len + 1;
    const int chunk = CEIL_DIV(total, S);
    const int j_start = s * chunk;  // 本段在全局序列里的起点
    const int j_end = (j_start + chunk < total) ? j_start + chunk : total; // 末段可能不满
    const int n_local = j_end - j_start; // 本段实际负责的 KV 数量

    if (n_local <= 0) {
        // 空段处理：写中性值覆盖垃圾数据，merge 时 exp(-inf)=0，自然忽略
        if (tid == 0) { part_max[(b * NH + h) * S + s] = -FLT_MAX; part_lse[(b * NH + h) * S + s] = 0.0f; }
        if (tid < hs) { part_out[((b * NH + h) * S + s) * hs + tid] = 0.0f; }
        return;   // 普通 kernel 不受 grid 同步约束，可以直接退出
    }

    __shared__ float q_s[64];        // 本头 Q 向量
    __shared__ float scores[256];    // 本段打分（S=4 时每段 ≤ 256）
    __shared__ float warp_max[2];    // 两个 warp 各写一个局部最大
    __shared__ float exp_local[256]; // 本段 exp 预计算，即 softmax 分子
    __shared__ float warp_lse[2];    // 两个 warp 各写一个局部指数和
    __shared__ float group_acc[256]; // 4 组 × 64 通道的部分加权和

    if (tid < hs) { q_s[tid] = (float)qkv[b * 3 * NH * hs + h * hs + tid]; }
    __syncthreads();

    // 段内点积打分（下标 jj 是段内编号，访问缓存时要加 j_start 换成全局位置）
    const floatX* k_cache = kv_cache + (b * NH + h) * maxT * hs;
    for (int jj = tid; jj < n_local; jj += blockDim.x) {
        float sum = 0.0f;
        const floatX* k_row = k_cache + (j_start + jj) * hs;  // 全局位置 = j_start + jj
        for (int d4 = 0; d4 < hs / x128::size; d4++) {
            x128 kv = load128cs(k_row + d4 * x128::size);
            for (int k = 0; k < x128::size; k++) {
                sum += q_s[d4 * x128::size + k] * (float)kv[k];
            }
        }
        scores[jj] = sum * scale;
    }
    __syncthreads();

    // 段内最大值归约：两个 warp 各出一个局部最大
    if (tid < hs) {
        float mv = -FLT_MAX;   // 空段已被提前拦下，这里至少有一个有效位置
        for (int jj = tid; jj < n_local; jj += hs) { mv = fmaxf(mv, scores[jj]); }
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffff, mv, off));
        }
        if (tid % WARP_SIZE == 0) { warp_max[tid / WARP_SIZE] = mv; }
    }
    __syncthreads();
    const float local_max = fmaxf(warp_max[0], warp_max[1]);  // 本段最大值

    // 段内指数和与 softmax 分子
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
    const float local_lse = warp_lse[0] + warp_lse[1];  // 本段指数和

    // 段内 4 组线程对 V 加权
    {
        const int group = tid / hs;  // 0..3，每组负责一段 jj 范围
        const int chan = tid % hs;   // 0..hs-1，负责哪个输出通道
        const floatX* v_cache = kv_cache + B * NH * maxT * hs + (b * NH + h) * maxT * hs;
        const int gchunk = CEIL_DIV(n_local, 4);  // 把总任务平均分成 4 份
        const int gs = group * gchunk;            // 本组的起始任务
        const int ge = (gs + gchunk < n_local) ? gs + gchunk : n_local;  // 本组结束任务
        float acc = 0.0f;
        for (int jj = gs; jj < ge; jj++) {
            acc += exp_local[jj] * (float)v_cache[(j_start + jj) * hs + chan];
        }
        group_acc[tid] = acc;
    }
    __syncthreads();

    // 写出三样部分结果，交给 merge kernel
    if (tid < hs) {
        float acc = 0.0f;
        for (int g = 0; g < 4; g++) { acc += group_acc[g * hs + tid]; }
        part_out[((b * NH + h) * S + s) * hs + tid] = acc;  // 段内未归一化加权和，即分子
        if (tid == 0) {
            part_max[(b * NH + h) * S + s] = local_max;     // 段最大值
            part_lse[(b * NH + h) * S + s] = local_lse;     // 段指数和，即分母
        }
    }
}
/* ----------------------------------------------------------------------------
   策略二：merge —— 把 S 段部分结果合并成最终输出
     网格：B*NH 个 block × 64 线程，一个 block 合并一个 (batch, 头)，只有 tid < hs 的 64 个线程参与，其余线程直接退出。

   合并公式（在线 softmax 的标准拼接）：
     先跨段求全局最大 m_global；
     每段 rescale 系数 = exp(段最大 − m_global)，段分子、段分母都乘它，各段就对齐到全局最大值的基准上；
     最终值 = 缩放后分子之和 / 缩放后分母之和。
   ---------------------------------------------------------------------------- */
__global__ void attention_kv_merge_kernel(floatX* out, const float* part_out,
                                          const float* part_lse,
                                          const float* part_max,
                                          int NH, int hs, int S) {
    const int b = blockIdx.x / NH;  // block 负责的 batch
    const int h = blockIdx.x % NH;  // block 负责的头
    const int tid = threadIdx.x;
    if (tid >= hs) return;  // 只有 hs 个通道线程干活，其余直接退出

    float m_global = -FLT_MAX;
    // 先扫所有段，取全局最大
    for (int s = 0; s < S; s++) { m_global = fmaxf(m_global, part_max[(b * NH + h) * S + s]); }
    float O_final = 0.0f;
    float l_final = 0.0f;
    for (int s = 0; s < S; s++) {
        float rescale = expf(part_max[(b * NH + h) * S + s] - m_global);  // 各段对齐到全局最大值的基准上
        O_final += rescale * part_out[((b * NH + h) * S + s) * hs + tid]; // 分子合并
        l_final += rescale * part_lse[(b * NH + h) * S + s];               // 分母合并
    }
    out[(b * NH + h) * hs + tid] = (floatX)(O_final / l_final);  // 分子/分母，得到归一化输出
}


// ============================================================================
// 策略三：cooperative —— split 与 merge 合进一个 kernel，grid 级同步
// ============================================================================

/* ----------------------------------------------------------------------------
   计算内容与 split kernel 逐段相同，区别是所有段写完后执行一次 grid.sync()
   让全网格的 block 在此汇合，段号 0 的 block 随后直接做跨段合并，省掉 merge kernel 的一次启动。
   代价：grid 级同步要求 cooperative launch，且该 kernel 无法被 CUDA Graph 捕获，所以只在 B=1 且不走 Graph 的路径由主机选择。
   ---------------------------------------------------------------------------- */
__global__ void attention_kv_cooperative_kernel(floatX* out,
                                               float* part_out, float* part_lse, float* part_max,
                                               const floatX* qkv, const floatX* kv_cache,
                                               int B, int NH, int hs, int maxT,
                                               int S, float scale) {
    cg::grid_group grid = cg::this_grid();  // 拿到全网格的同步句柄
    const int cur_len = g_cur_len;
    const int b = blockIdx.x / (NH * S);  // 一维 Grid 拆成 (b, h, s)：先除「头 × 段」
    const int h = (blockIdx.x / S) % NH;  // 再除「段」取模，得头号
    const int s = blockIdx.x % S;         // 余数即段号
    const int tid = threadIdx.x;

    const int total = cur_len + 1;
    const int chunk = CEIL_DIV(total, S);  // 每段应该分配多少个位置
    const int j_start = s * chunk;          // 本段起点
    const int j_end = (j_start + chunk < total) ? j_start + chunk : total;  // 本段终点
    const int n_local = j_end - j_start;   // 本段实际负责的 KV 数量

    __shared__ float q_s[64];
    __shared__ float scores[256];
    __shared__ float warp_max[2];
    __shared__ float exp_local[256];
    __shared__ float warp_lse[2];
    __shared__ float group_acc[256];

    if (tid < hs) { q_s[tid] = (float)qkv[b * 3 * NH * hs + h * hs + tid]; }
    __syncthreads();

    if (n_local > 0) {
        const floatX* k_cache = kv_cache + (b * NH + h) * maxT * hs;
        for (int jj = tid; jj < n_local; jj += blockDim.x) {
            int j = j_start + jj;  // 段内编号换成全局位置
            float sum = 0.0f;
            const floatX* k_row = k_cache + j * hs;
            for (int d4 = 0; d4 < hs / x128::size; d4++) {
                x128 kv = load128cs(k_row + d4 * x128::size);
                for (int k = 0; k < x128::size; k++) {
                    sum += q_s[d4 * x128::size + k] * (float)kv[k];
                }
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
        const float local_max = fmaxf(warp_max[0], warp_max[1]);  // 本段最大值

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
        const float local_lse = warp_lse[0] + warp_lse[1];  // 本段指数和

        {
            const int group = tid / hs;  // 0..3，每组负责一段 jj 范围
            const int chan = tid % hs;   // 0..hs-1，负责哪个输出通道
            const floatX* v_cache = kv_cache + B * NH * maxT * hs + (b * NH + h) * maxT * hs;
            const int gchunk = CEIL_DIV(n_local, 4);  // 把总任务平均分成 4 份
            const int gs = group * gchunk;             // 本组起始任务
            const int ge = (gs + gchunk < n_local) ? gs + gchunk : n_local;  // 本组结束任务
            float acc = 0.0f;
            for (int jj = gs; jj < ge; jj++) {
                acc += exp_local[jj] * (float)v_cache[(j_start + jj) * hs + chan];
            }
            group_acc[tid] = acc;
        }
        __syncthreads();

        if (tid < hs) {
            float acc = 0.0f;
            for (int g = 0; g < 4; g++) { acc += group_acc[g * hs + tid]; }
            part_out[((b * NH + h) * S + s) * hs + tid] = acc;  // 段内未归一化加权和，即分子
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

    grid.sync();  // 跨 block 全局同步：确保所有段的部分结果都写完

    // merge 阶段：每头只让 s==0 的 block（hs 个线程）做跨段合并，公式与 merge kernel 相同
    if (s == 0 && tid < hs) {
        float m_global = -FLT_MAX;
        for (int sp = 0; sp < S; sp++) { m_global = fmaxf(m_global, part_max[(b * NH + h) * S + sp]); }  // 跨段求全局最大
        float O_final = 0.0f;
        float l_final = 0.0f;
        for (int sp = 0; sp < S; sp++) {
            float rescale = expf(part_max[(b * NH + h) * S + sp] - m_global);  // 各段对齐到全局最大值的基准上
            O_final += rescale * part_out[((b * NH + h) * S + sp) * hs + tid]; // 分子合并
            l_final += rescale * part_lse[(b * NH + h) * S + sp];               // 分母合并
        }
        out[(b * NH + h) * hs + tid] = (floatX)(O_final / l_final);
    }
}

// ============================================================================
// 主机入口：短序列走 fused single
// ============================================================================
inline void attention_forward_kv_small(floatX* out, const floatX* qkv, floatX* kv_cache,
                                       int B, int NH, int hs, int maxT, cudaStream_t stream) {
    float scale = 1.f / sqrtf(hs);
    attention_kv_fused_single_kernel<<<B * NH, 256, 0, stream>>>(
        out, qkv, kv_cache, B, NH, hs, maxT, scale);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// 主机入口：长序列走 split+merge（B>1）或 cooperative（B=1）
// ============================================================================
inline void attention_forward_kv_large(floatX* out, const floatX* qkv, floatX* kv_cache,
                                       int B, int NH, int hs, int maxT, cudaStream_t stream,
                                       float* part_out, float* part_lse, float* part_max, int S) {
    float scale = 1.f / sqrtf(hs);
    if (B == 1) {
        // B=1 且非 Graph 路径：cooperative launch 把两段合成一个 kernel
        dim3 grid(B * NH * S), block(256);  // 与 split 相同的 Grid 形状
        // 按 kernel 参数顺序打包成 void* 数组，cooperative launch 只接受这种形式
        void* args[] = {
            (void*)&out, (void*)&part_out, (void*)&part_lse, (void*)&part_max,
            (void*)&qkv, (void*)&kv_cache, (void*)&B, (void*)&NH, (void*)&hs, (void*)&maxT,
            (void*)&S, (void*)&scale
        };
        cudaCheck(cudaLaunchCooperativeKernel(
            (void*)attention_kv_cooperative_kernel, grid, block, args, 0, stream));
    } else {
        // B>1：split 与 merge 两个独立 kernel，可在默认流上依次执行
        attention_kv_split_kernel<<<B * NH * S, 256, 0, stream>>>(
            part_out, part_lse, part_max, qkv, kv_cache, B, NH, hs, maxT, S, scale);
        attention_kv_merge_kernel<<<B * NH, 64, 0, stream>>>(
            out, part_out, part_lse, part_max, NH, hs, S);
    }
    cudaCheck(cudaGetLastError());
}
