/* ============================================================================
   attention_kvcache_w8.cuh —— 8 位 KV 缓存版解码注意力
   ============================================================================
   这是 attention_kvcache.cuh 的量化版本：缓存里的 K、V 不再是 BF16（每个元素 2 字节），而是 1 字节的 INT8 / FP8，缓存体积减半。

   量化方式：逐行动态量化。
   写入每一行（某个 token 某个头的整条 K 或 V）时，先在这一行里找绝对值最大值 amax，再得到 scale = amax / 格式上限（INT8 为 127，FP8 E4M3 为 448）
   每个元素除以 scale 后编码成字节；读取时解码并乘回 scale。
   scale 没有统一常量，而是每行一个、随数据动态计算，所以另开一块 float 数组 kv_scale 与字节缓存并排存放。

   与非量化版的三处代码差异：
     1. 编码：INT8 四舍五入后夹到 [-127,127]；FP8 用硬件饱和转换；
     2. 解码：INT8 直接转整数；FP8 经硬件指令转回 BF16 再转 FP32；
     3. 点积：行 scale 提到元素循环外，一行只乘一次。

   三种读取策略的并行结构（网格形状、线程分工）与非量化版完全相同，

   整个文件包在 USE_KV_W8 宏内，默认配置不编译。
   ============================================================================ */
#pragma once

#include <float.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include "common.h"

#if defined(USE_KV_W8)

namespace cg = cooperative_groups;

// 设备端当前序列长度：与非量化头共用同一份状态（同一步的序列长度）。
// 用宏守卫保证只定义一次，两个头可以任意顺序 include、也可以单独使用。
#ifndef KV_G_CUR_LEN_DEFINED
#define KV_G_CUR_LEN_DEFINED
__device__ int g_cur_len = 0;
#endif

// 格式互斥检查：两个宏必须恰好定义一个，由它决定字节如何编码解码
#if defined(USE_FP8) && defined(USE_INT8)
#error "USE_FP8 与 USE_INT8 互斥，KV 量化同样只能选一个"
#elif !defined(USE_FP8) && !defined(USE_INT8)
#error "KV 量化需要定义 USE_INT8 或 USE_FP8 之一（决定用哪种 8 位格式）"
#endif

// 格式上限：量化时把 amax 映射到这个值，决定 scale 的换算分母
#ifdef USE_FP8
#define KV_W8_LIMIT 448.0f
#else
#define KV_W8_LIMIT 127.0f
#endif

// ============================================================================
// 编码：浮点数 → 量化字节（输入已经除以本行 scale）
// ============================================================================
#ifdef USE_FP8
__device__ __forceinline__ uint8_t kv_w8_encode(float x, float scale) {
    // 饱和转换：超出 E4M3 表示范围的值会被钳到最大可表示值，而不是溢出
    return (uint8_t)__nv_cvt_float_to_fp8(x / scale, __NV_SATFINITE, __NV_E4M3);
}
#else
__device__ __forceinline__ uint8_t kv_w8_encode(float x, float scale) {
    int q = __float2int_rn(x / scale);              // 四舍五入到整数码，比截断少一半量化误差
    q = (q < -127) ? -127 : ((q > 127) ? 127 : q); // 手动夹断，防止整数码越界
    return (uint8_t)(int8_t)q;
}
#endif

// ============================================================================
// 解码：量化字节 → 浮点数（还没乘回 scale）
// ============================================================================
#ifdef USE_FP8
__device__ __forceinline__ float kv_w8_decode(uint8_t b) {
    __nv_fp8_storage_t s = b;
    __nv_bfloat16_raw r = __nv_cvt_fp8_to_bf16raw(s, __NV_E4M3);  // 硬件指令：FP8 → BF16
    __nv_bfloat16 h = *reinterpret_cast<__nv_bfloat16*>(&r);
    return __bfloat162float(h);  // BF16 → FP32，参与后续累加
}
#else
__device__ __forceinline__ float kv_w8_decode(uint8_t b) {
    return (float)((int8_t)b); // 补码字节转回有符号整数，再升 FP32
}
#endif

/* ----------------------------------------------------------------------------
   kv_dot16：16 个量化字节上的点积
     q    指向 Q 里的 16 个连续元素
     p    指向本行 K 的 16 个连续字节
     ks   本行的 scale
   一次 uint4 合并读入 16 字节，逐元素解码后与 Q 相乘，最后整段乘一次行 scale。
   ---------------------------------------------------------------------------- */
__device__ __forceinline__ float kv_dot16(const float* q, const uint8_t* p, float ks) {
    uint4 w = *reinterpret_cast<const uint4*>(p);      // 一条 128-bit 指令取 16 字节
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&w);
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        s += q[i] * kv_w8_decode(b[i]);  // 解码后与 Q 对应位相乘，循环展开成 16 条 FMA
    }
    return s * ks;  // 行 scale 提到循环外，省掉 15 次乘
}


// ============================================================================
// 缓存写入：逐行求 amax → 算 scale → 编码整行
//    网格：B*NH*2 个 block × 256 线程，一个 block 负责一行（前半 K、后半 V）
// ============================================================================

/* ----------------------------------------------------------------------------
   blockIdx.x 直接就是行号 0..B*NH*2-1：
     行号 <  B*NH  属于 K 区，kh = 行号；
     行号 >= B*NH  属于 V 区，kh = 行号 − B*NH，再折算成 (b, h)。
   amax 的归约路径：各线程先取自己通道的绝对值 → warp 内 shuffle 归约 → 各 warp 最大值写共享内存 → 全 block 再取一次，得到整行 amax。
   随后 scale 由线程 0 写入 kv_scale，所有线程按它编码各自的通道。
   ---------------------------------------------------------------------------- */
__global__ void kv_cache_write_kernel_w8(uint8_t* kv_cache, float* kv_scale,
                                         const floatX* qkv, int B, int NH,
                                         int hs, int maxT) {
    const int cur_len = g_cur_len;  // 本步写入第几列
    const int row = blockIdx.x;     // 网格共 B*NH*2 行，前半 K 后半 V
    if (row >= B * NH * 2) { return; }
    const bool is_v = row >= B * NH;  // 行号过了 K 区就是 V
    const int kh = row - (is_v ? B * NH : 0);  // 折算成 (batch,头) 编号 0..B*NH-1
    const int b = kh / NH, h = kh % NH;

    // 源数据：当前 token QKV 里的 K 或 V 段
    const floatX* src_base = qkv + (b * 3 * NH * hs) + h * hs
                           + (is_v ? 2 * NH * hs : NH * hs);   // V 段比 K 段再往后错一个头矩阵
    // 字节缓存目标：第 cur_len 列；V 区整体加一块 B*NH*maxT*hs
    uint8_t* dst_base = kv_cache + (is_v ? (size_t)B * NH * maxT * hs : 0) + (size_t)kh * maxT * hs + (size_t)cur_len * hs;
    // scale 数组目标，与字节缓存并排，粒度是「每行一个」
    float* sc_base = kv_scale + (is_v ? (size_t)B * NH * maxT : 0) + (size_t)kh * maxT + cur_len;

    // 第一步：线程内先取自己负责通道的绝对值最大
    float mx = 0.0f;
    for (int d = threadIdx.x; d < hs; d += blockDim.x) {
        mx = fmaxf(mx, fabsf((float)src_base[d]));
    }
    __shared__ float s_mx[16];  // 最多 16 个 warp 的局部最大（256 线程 = 8 个 warp，够用）
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)  // 第二步：warp 内蝶形归约
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) s_mx[threadIdx.x >> 5] = mx;  // 每个 warp 的 lane 0 落位
    __syncthreads();
    mx = 0.0f;
    for (int w = 0; w < (int)(blockDim.x >> 5); w++) mx = fmaxf(mx, s_mx[w]);  // 第三步：跨 warp 取整行 amax

    const float scale = mx / KV_W8_LIMIT;  // 行 scale = amax / 格式上限
    if (threadIdx.x == 0) *sc_base = scale;  // 行 scale 落盘，读取侧按它还原

    // 全 0 行没有有效 scale，直接写 0 字节，避免除零
    for (int d = threadIdx.x; d < hs; d += blockDim.x) {
        dst_base[d] = (scale > 0.0f) ? kv_w8_encode((float)src_base[d], scale)
                                     : (uint8_t)0;
    }
}

// ============================================================================
// 策略一：fused single —— 单个 block 完成一个 (batch, 头) 的全部计算
// ============================================================================

/* 四步流程（打分 → 全头最大 → 指数与 softmax 分子 → 分组对 V 加权汇总）与 attention_kv_fused_single_kernel 完全一致。
   区别只在：K 打分走 kv_dot16，行 scale 已含在其中；V 的每个元素解码后要乘该位置自己的 V scale（逐位置不同）。 */
__global__ void attention_kv_fused_single_kernel_w8(floatX* out, const floatX* qkv,
                                                    const uint8_t* kv_cache,
                                                    const float* kv_scale,
                                                    int B, int NH,
                                                    int hs, int maxT, float scale) {
    const int cur_len = g_cur_len;
    const int b = blockIdx.x / NH;   // 一维 Grid 拆成 (b, h)：先除头数得 batch
    const int h = blockIdx.x % NH;   // 余数即头号
    const int tid = threadIdx.x;
    const int total = cur_len + 1;   // 参与注意力的位置总数（含当前 token）

    __shared__ float q_s[64];        // 本头 Q 向量，搬一次供 256 线程共用
    __shared__ float scores[1024];   // 存储当前头所有历史位置的打分结果
    __shared__ float warp_max[2];    // 两个 warp 各写一个局部最大
    __shared__ float exp_buf[1024];  // 预计算的 exp(打分成 - 最大分)，即 softmax 分子
    __shared__ float warp_lse[2];    // 两个 warp 各写一个局部指数和
    __shared__ float group_acc[256]; // 4 组 × 64 通道的部分加权和（分子）

    if (tid < hs) { q_s[tid] = (float)qkv[b * 3 * NH * hs + h * hs + tid]; }
    __syncthreads();

    const uint8_t* k_cache = kv_cache + (b * NH + h) * maxT * hs;  // 本 (batch,头) 的 K 字节槽位
    const float* k_sc = kv_scale + (b * NH + h) * maxT;             // 每个位置的 K scale
    for (int j = tid; j < total; j += blockDim.x) {
        float sum = 0.0f;
        const uint8_t* k_row = k_cache + j * hs;  // 第 j 个位置的 K 字节行
        const float ks = k_sc[j];                  // 本位置的行 scale
        #pragma unroll
        for (int d4 = 0; d4 < hs / 16; d4++) {
            // 每 16 字节走一次量化点积，scale 已含在内
            sum += kv_dot16(q_s + d4 * 16, k_row + d4 * 16, ks);
        }
        scores[j] = sum * scale;  // 再乘注意力缩放 1/√hs
    }
    __syncthreads();

    // 两个 warp 各自归约最大值，再取全头最大
    if (tid < hs) {
        float mv = -FLT_MAX;  // 中性初值
        for (int j = tid; j < total; j += hs) { mv = fmaxf(mv, scores[j]); }  // 先串行走一遍自己负责的位置
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)  // 蝶形归约：偏移 16/8/4/2/1
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffff, mv, off));
        if (tid % WARP_SIZE == 0) { warp_max[tid / WARP_SIZE] = mv; }  // 每个 warp 的 lane 0 落位
    }
    __syncthreads();
    const float gmax = fmaxf(warp_max[0], warp_max[1]);  // 两个 warp 再取全头最大值

    // 统一最大值下算指数，既写出 softmax 分子又累加出 softmax 分母
    if (tid < hs) {
        float ls = 0.0f;
        for (int j = tid; j < total; j += hs) {
            float ev = expf(scores[j] - gmax);  // 减去最大值，指数不会溢出
            exp_buf[j] = ev;
            ls += ev;
        }
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            ls += __shfl_xor_sync(0xffffffff, ls, off);
        if (tid % WARP_SIZE == 0) { warp_lse[tid / WARP_SIZE] = ls; }
    }
    __syncthreads();
    const float glse = warp_lse[0] + warp_lse[1];  // 全局指数和，即 softmax 分母
    {
        const int group = tid / hs;  // 0..3，每组负责一段 j 范围
        const int chan = tid % hs;   // 0..hs-1，负责哪个输出通道
        const uint8_t* v_cache = kv_cache + (size_t)B * NH * maxT * hs + (b * NH + h) * maxT * hs;  // V 字节槽位
        const float* v_sc = kv_scale + (size_t)B * NH * maxT + (b * NH + h) * maxT;                  // 每位置 V scale
        const int chunk = CEIL_DIV(total, 4);  // 将 j 范围均分给 4 个组
        const int js = group * chunk;           // 本组起始任务
        const int je = (js + chunk < total) ? js + chunk : total;  // 本组结束任务
        float acc = 0.0f;
        for (int j = js; j < je; j++) {
            // V scale 逐位置不同，必须每个元素单独乘
            acc += exp_buf[j] * (kv_w8_decode(v_cache[j * hs + chan]) * v_sc[j]);
        }
        group_acc[tid] = acc;
    }
    __syncthreads();

    if (tid < hs) {
        float acc = 0.0f;
        for (int g = 0; g < 4; g++) { acc += group_acc[g * hs + tid]; }  // 同一通道跨 4 组相加
        out[(b * NH + h) * hs + tid] = (floatX)(acc / glse);             // 除分母完成归一化
    }
}


// ============================================================================
// 策略二（上）：split —— Flash Decoding，每 block 处理一段，输出部分结果
// ============================================================================

/* 段内流程与非量化 split kernel 相同，量化差异同上：K 走 kv_dot16，V 逐元素解码并乘位置 scale；
   空段写中性值（max=-∞、lse=0、out=0），merge 时 exp(-inf)=0，自然被忽略。

   三组部分结果的索引（S 是切分段数）：
     part_max / part_lse : (b*NH + h)*S + s             每 (batch,头) 连续 S 个槽
     part_out            : ((b*NH + h)*S + s)*hs + d    再按通道展开一层
   与非量化版的唯一区别是：量化版的 V 在解码后还要乘逐位置的 scale。 */
__global__ void attention_kv_split_kernel_w8(float* part_out, float* part_lse, float* part_max,
                                             const floatX* qkv, const uint8_t* kv_cache,
                                             const float* kv_scale,
                                             int B, int NH, int hs,
                                             int maxT, int S, float scale) {
    const int cur_len = g_cur_len;
    const int b = blockIdx.x / (NH * S);  // 先除「头 × 段」，得 batch
    const int h = (blockIdx.x / S) % NH;  // 再除「段」取模，得头号
    const int s = blockIdx.x % S;         // 余数即段号 0..S-1
    const int tid = threadIdx.x;

    // 把 [0, total) 均分成 S 段，取本 block 对应的起止位置
    const int total = cur_len + 1;
    const int chunk = CEIL_DIV(total, S);  // 每段应该分配多少个位置
    const int j_start = s * chunk;          // 本段在全局序列里的起点
    const int j_end = (j_start + chunk < total) ? j_start + chunk : total;  // 末段可能不满
    const int n_local = j_end - j_start;   // 本段实际负责的 KV 数量

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

    // 段内打分（下标 jj 是段内编号，访问缓存时要加 j_start 换成全局位置）
    const uint8_t* k_cache = kv_cache + (b * NH + h) * maxT * hs;
    const float* k_sc = kv_scale + (b * NH + h) * maxT;
    for (int jj = tid; jj < n_local; jj += blockDim.x) {
        float sum = 0.0f;
        const uint8_t* k_row = k_cache + (j_start + jj) * hs;  // 全局位置 = j_start + jj
        const float ks = k_sc[j_start + jj];                    // 该位置的行 scale

        #pragma unroll
        for (int d4 = 0; d4 < hs / 16; d4++) {
            sum += kv_dot16(q_s + d4 * 16, k_row + d4 * 16, ks);
        }
        scores[jj] = sum * scale;
    }
    __syncthreads();

    // 段内最大值归约：两个 warp 各出一个局部最大
    if (tid < hs) {
        float mv = -FLT_MAX;   // 空段已被提前拦下，这里至少有一个有效位置
        for (int jj = tid; jj < n_local; jj += hs) { mv = fmaxf(mv, scores[jj]); }
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            mv = fmaxf(mv, __shfl_xor_sync(0xffffffff, mv, off));
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
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            ls += __shfl_xor_sync(0xffffffff, ls, off);
        if (tid % WARP_SIZE == 0) { warp_lse[tid / WARP_SIZE] = ls; }
    }
    __syncthreads();
    const float local_lse = warp_lse[0] + warp_lse[1];  // 本段指数和，即分母
    {
        const int group = tid / hs;  // 0..3，每组负责一段 jj 范围
        const int chan = tid % hs;   // 0..hs-1，负责哪个输出通道
        const uint8_t* v_cache = kv_cache + (size_t)B * NH * maxT * hs + (b * NH + h) * maxT * hs;  // V 字节槽位
        const float* v_sc = kv_scale + (size_t)B * NH * maxT + (b * NH + h) * maxT;                  // 每位置 V scale
        const int gchunk = CEIL_DIV(n_local, 4);  // 把总任务平均分成 4 份
        const int gs = group * gchunk;            // 本组起始任务
        const int ge = (gs + gchunk < n_local) ? gs + gchunk : n_local;  // 本组结束任务
        float acc = 0.0f;
        for (int jj = gs; jj < ge; jj++) {
            // 逐元素解码 V，再乘该位置自己的 scale
            acc += exp_local[jj] * (kv_w8_decode(v_cache[(j_start + jj) * hs + chan]) * v_sc[j_start + jj]);
        }
        group_acc[tid] = acc;
    }
    __syncthreads();

    // 写出三样部分结果，交给本文件的 merge kernel
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
   策略二（下）：merge —— 把 S 段部分结果合并成最终输出

   合并公式（在线 softmax 的标准拼接）：
     先跨段求全局最大 m_global；
     每段 rescale 系数 = exp(段最大 − m_global)，段分子、段分母都乘它，各段就对齐到全局最大值的基准上；
     最终值 = 缩放后分子之和 / 缩放后分母之和。

   这段完全是浮点算术，只吃 part_out / part_lse / part_max 三组 float，与 K、V 是 INT8 还是 FP8 无关，所以量化版与非量化版公式一模一样，
   ---------------------------------------------------------------------------- */
__global__ void attention_kv_merge_kernel_w8(floatX* out, const float* part_out,
                                             const float* part_lse,
                                             const float* part_max,
                                             int NH, int hs, int S) {
    const int b = blockIdx.x / NH;  // block 负责的 batch
    const int h = blockIdx.x % NH;  // block 负责的头
    const int tid = threadIdx.x;
    if (tid >= hs) return;          // 只有 hs 个通道线程干活，其余直接退出

    float m_global = -FLT_MAX;
    // 先扫所有段，取全局最大
    for (int s = 0; s < S; s++) { m_global = fmaxf(m_global, part_max[(b * NH + h) * S + s]); }

    float O_final = 0.0f;
    float l_final = 0.0f;
    for (int s = 0; s < S; s++) {
        float rescale = expf(part_max[(b * NH + h) * S + s] - m_global);  // 各段对齐到全局最大值的基准上

        O_final += rescale * part_out[((b * NH + h) * S + s) * hs + tid];  // 分子合并
        l_final += rescale * part_lse[(b * NH + h) * S + s];               // 分母合并
    }
    out[(b * NH + h) * hs + tid] = (floatX)(O_final / l_final);  // 分子 / 分母，得到归一化输出
}


// ============================================================================
// 策略三：cooperative —— 段计算 + grid 同步 + 段号 0 合并
// ============================================================================

/* 与非量化 cooperative kernel 的结构逐段相同，量化差异仍只在 K 点积和
   V 逐元素 scale 两处；合并公式不变，所以段号 0 的 block 直接照同一套归并写法做。 */
__global__ void attention_kv_cooperative_kernel_w8(floatX* out,
                                                   float* part_out, float* part_lse, float* part_max,
                                                   const floatX* qkv, const uint8_t* kv_cache,
                                                   const float* kv_scale,
                                                   int B, int NH, int hs, int maxT,
                                                   int S, float scale) {
    cg::grid_group grid = cg::this_grid();  // 拿到全网格的同步句柄
    const int cur_len = g_cur_len;
    const int b = blockIdx.x / (NH * S);
    const int h = (blockIdx.x / S) % NH;
    const int s = blockIdx.x % S;
    const int tid = threadIdx.x;

    const int total = cur_len + 1;
    const int chunk = CEIL_DIV(total, S);  // 每段应该分配多少个位置
    const int j_start = s * chunk;         // 本段起点
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
        const uint8_t* k_cache = kv_cache + (b * NH + h) * maxT * hs;
        const float* k_sc = kv_scale + (b * NH + h) * maxT;
        for (int jj = tid; jj < n_local; jj += blockDim.x) {
            int j = j_start + jj; // 段内编号换成全局位置
            float sum = 0.0f;
            const uint8_t* k_row = k_cache + j * hs;
            const float ks = k_sc[j];
            #pragma unroll
            for (int d4 = 0; d4 < hs / 16; d4++) {
                sum += kv_dot16(q_s + d4 * 16, k_row + d4 * 16, ks);
            }
            scores[jj] = sum * scale;
        }
        __syncthreads();

        if (tid < hs) {
            float mv = -FLT_MAX;
            for (int jj = tid; jj < n_local; jj += hs) { mv = fmaxf(mv, scores[jj]); }
            for (int off = WARP_SIZE/2; off > 0; off >>= 1)
                mv = fmaxf(mv, __shfl_xor_sync(0xffffffff, mv, off));
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
            for (int off = WARP_SIZE/2; off > 0; off >>= 1)
                ls += __shfl_xor_sync(0xffffffff, ls, off);
            if (tid % WARP_SIZE == 0) { warp_lse[tid / WARP_SIZE] = ls; }
        }
        __syncthreads();
        const float local_lse = warp_lse[0] + warp_lse[1];  // 本段指数和

        {
            const int group = tid / hs;  // 0..3，每组负责一段 jj 范围
            const int chan = tid % hs;   // 0..hs-1，负责哪个输出通道
            const uint8_t* v_cache = kv_cache + (size_t)B * NH * maxT * hs + (b * NH + h) * maxT * hs;
            const float* v_sc = kv_scale + (size_t)B * NH * maxT + (b * NH + h) * maxT;
            const int gchunk = CEIL_DIV(n_local, 4);  // 把总任务平均分成 4 份
            const int gs = group * gchunk;            // 本组起始任务
            const int ge = (gs + gchunk < n_local) ? gs + gchunk : n_local;  // 本组结束任务
            float acc = 0.0f;
            for (int jj = gs; jj < ge; jj++) {
                acc += exp_local[jj] * (kv_w8_decode(v_cache[(j_start + jj) * hs + chan]) * v_sc[j_start + jj]);
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
        for (int sp = 0; sp < S; sp++) m_global = fmaxf(m_global, part_max[(b * NH + h) * S + sp]);  // 跨段求全局最大
        float O_final = 0.0f;
        float l_final = 0.0f;
        for (int sp = 0; sp < S; sp++) {
            float rescale = expf(part_max[(b * NH + h) * S + sp] - m_global);  // 各段对齐到全局最大值的基准上
            O_final += rescale * part_out[((b * NH + h) * S + sp) * hs + tid];  // 分子合并
            l_final += rescale * part_lse[(b * NH + h) * S + sp];               // 分母合并
        }
        out[(b * NH + h) * hs + tid] = (floatX)(O_final / l_final);  // 分子/分母，得到归一化输出
    }
}


// ============================================================================
// 主机入口：短序列走 fused single
// ============================================================================
inline void attention_forward_kv_small_w8(floatX* out, const floatX* qkv, uint8_t* kv_cache,
                                          const float* kv_scale,
                                          int B, int NH, int hs, int maxT, cudaStream_t stream) {
    float scale = 1.f / sqrtf(hs);  // 1/√hs，点积后统一缩放
    attention_kv_fused_single_kernel_w8<<<B * NH, 256, 0, stream>>>(
        out, qkv, kv_cache, kv_scale, B, NH, hs, maxT, scale);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// 主机入口：长序列（B=1 cooperative，B>1 split + merge，两者都在本文件内）
// ============================================================================
inline void attention_forward_kv_large_w8(floatX* out, const floatX* qkv, uint8_t* kv_cache,
                                          const float* kv_scale,
                                          int B, int NH, int hs, int maxT, cudaStream_t stream,
                                          float* part_out, float* part_lse, float* part_max, int S) {
    float scale = 1.f / sqrtf(hs);
    if (B == 1) {
        // B=1 且非 Graph 路径：cooperative launch 把两段合成一个 kernel
        dim3 grid(B * NH * S), block(256);  // 与 split 相同的 Grid 形状
        // 按 kernel 参数顺序打包成 void* 数组，cooperative launch 只接受这种形式
        void* args[] = {
            (void*)&out, (void*)&part_out, (void*)&part_lse, (void*)&part_max,
            (void*)&qkv, (void*)&kv_cache, (void*)&kv_scale,
            (void*)&B, (void*)&NH, (void*)&hs, (void*)&maxT,
            (void*)&S, (void*)&scale
        };
        cudaCheck(cudaLaunchCooperativeKernel(
            (void*)attention_kv_cooperative_kernel_w8, grid, block, args, 0, stream));
    } else {
        // B>1：split 与 merge 两个独立 kernel，可在默认流上依次执行
        attention_kv_split_kernel_w8<<<B * NH * S, 256, 0, stream>>>(
            part_out, part_lse, part_max, qkv, kv_cache, kv_scale, B, NH, hs, maxT, S, scale);
        attention_kv_merge_kernel_w8<<<B * NH, 64, 0, stream>>>(
            out, part_out, part_lse, part_max, NH, hs, S);
    }
    cudaCheck(cudaGetLastError());
}

#endif
