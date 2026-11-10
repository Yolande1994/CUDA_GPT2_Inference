/* ============================================================================
   layernorm.cuh —— LayerNorm（层归一化）与残差算子
   ============================================================================

   算子说明：
     LayerNorm 对每一行（一个 token 的 C 个通道）做归一化：
       1. 计算均值 m 与方差 v；
       2. 归一化：n = (x - m) / sqrt(v + eps)；
       3. 缩放平移：out = n * weight + bias。
     残差（Residual）把两个张量逐元素相加，常与 LayerNorm 融合在一起计算，
     因此本文件同时包含这两类算子。

   说明：
     mean（均值）和 rstd（标准差倒数）仍会缓存写出，但它们并非产生前向输出所必需，仅用于查看或后续复用。
   ============================================================================ */

#pragma once

#include <assert.h>
#include "common.h"

// ============================================================================
// CUDA Kernels
// ============================================================================

/* ----------------------------------------------------------------------------
   layernorm_forward_kernel3 —— 无共享内存的 LayerNorm kernel
   ----------------------------------------------------------------------------
   并行策略：一个 warp（32 线程）负责一行（一个 token 的 C 个通道），各 lane 用跨步访问分摊元素，再用 warpReduceSum 做 warp 内规约。
   流程：
     1. 各 lane 累加自己负责的元素，规约求和后除以 C 得均值；
     2. 各 lane 累加 (x-m)^2，规约求和后除以 C 得方差，再用 rsqrtf 求标准差倒数（含 eps 防止除零）；
     3. 各 lane 完成归一化和缩放平移并写回。
   访存提示：读写使用 .cs 流式指令，表明数据不会很快复用，让共享的 weight/bias 获得更高缓存命中率。
   用途：作为 kernel6 无法使用大共享内存时的回退方案。
---------------------------------------------------------------------------- */
__global__ void layernorm_forward_kernel3(floatX* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const floatX*  __restrict__ inp, const floatX*  __restrict__ weight,
                                    const floatX* __restrict__ bias, int N, int C) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    int idx = blockIdx.x * num_warps + warp_id;
    if(idx >= N) { return; } // 越界保护
    // 该 warp 负责的输入行
    const floatX* x = inp + idx * C;

    // 计算均值
    float sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        sum += (float)x[i];
    }
    sum = warpReduceSum(sum);
    float m = sum / C;
    if(lane_id == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    // 计算标准差倒数 rstd
    sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        float diff = (float)x[i] - m;
        sum += diff * diff;
    }
    sum = warpReduceSum(sum);
    float s = rsqrtf(sum / C + 1e-5f);
    if(lane_id == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // 最终归一化，再乘以 weight 加 bias
    floatX* o = out + idx * C;
    for (int c = lane_id; c < C; c += WARP_SIZE) {
        // 用 .cs 流式提示编译器：该数据不会很快复用，可直接流过缓存，从而让共享的 weight 和 bias 参数获得更多缓存命中
        float n = s * ((float)__ldcs(x+c) - m);
        __stcs(o+c, (floatX)(n * (float)weight[c] + (float)bias[c]));
    }
}

/* ----------------------------------------------------------------------------
   layernorm_forward_kernel6 —— 共享内存版 LayerNorm kernel
   ----------------------------------------------------------------------------
   并行策略：block 的 x 维度固定为一个 warp（32 线程），y 维度的每一行负责一个 token，从而一个 block 同时处理多个 token。
   共享内存布局：
     - s_weight / s_bias：存放本 block 复用的 weight 和 bias；
     - s_in：按 y 行存放各 token 的输入数据。
   流程：
     1. 所有线程先把 weight、bias 加载进共享内存（必须在任何线程退出前完成）；
     2. 各 y 行把自己负责的输入读入共享内存；
     3. warp 内规约求均值、方差、标准差倒数；
     4. 从共享内存读取数据完成归一化与缩放平移，流式写回。
   优点：weight/bias 只需从显存读一次，再在共享内存中被多个 token 复用。
---------------------------------------------------------------------------- */
__global__ void layernorm_forward_kernel6(floatX* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const floatX*  __restrict__ inp, const floatX*  __restrict__ weight,
                                    const floatX* __restrict__ bias, int N, int C) {
    assert(blockDim.x == WARP_SIZE);
    // 把 weight 和 bias 加载进共享内存；必须在允许任何线程退出之前完成！
    extern __shared__ char* params[];
    // 当类型为 floatX* 时，load128/store128 有时会生成多条指令，因此这里统一用 x128 类型处理
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_in = reinterpret_cast<x128*>(params) + ((2 + threadIdx.y) * C / x128::size);

    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx >= N) { return; } // 越界保护

    // 把指针调整到当前 token
    inp += idx * C;
    out += idx * C;
    const float eps = 1e-5f;
    float sum = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = load128cs(inp + c);
        for(int k = 0; k < x128::size; ++k) {
            sum += (float)in_data[k];
        }
        s_in[c / x128::size] = in_data;
    }

    sum = warpReduceSum(sum);
    float m = sum / C;
    float v = 0.f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)in_data[k] - m) * ((float)in_data[k] - m);
        }
    }

    v = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out_data;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)in_data[k] - m); // 归一化结果
            float o = n * (float)w[k] + (float)b[k]; // 缩放平移
            out_data[k] = (floatX)o;
        }
        store128cs(out + c, out_data);
    }
    // 缓存均值和标准差倒数，供后续使用
    if(threadIdx.x == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    if(threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
}

/* ----------------------------------------------------------------------------
   fused_residual_forward_kernel5 —— 融合「残差 + LayerNorm」kernel
   ----------------------------------------------------------------------------
   一次 kernel 同时完成：
     1. 残差：residual = inp1 + inp2；
     2. LayerNorm：对残差结果归一化并缩放平移，输出 normed。
   并行策略与 kernel6 相同：x 为一个 warp，y 每行负责一个 token。
   共享内存：s_weight/s_bias 存权重，s_res 存各 token 的残差结果。
   优点：残差结果直接在共享内存中被后续归一化复用，省去一次往返显存。
---------------------------------------------------------------------------- */
__global__ void fused_residual_forward_kernel5(floatX* residual, floatX* normed, float* mean, float* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C) {
    assert(blockDim.x == WARP_SIZE);
    // 把 weight 和 bias 加载进共享内存；必须在允许任何线程退出之前完成！
    extern __shared__ char* params[];
    // 统一用 x128 类型处理，避免生成多条指令
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_res = reinterpret_cast<x128*>(params) + ((2 + threadIdx.y) * C / x128::size);

    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx >= N) return;  // 越界保护：必须是 >= N（N 不是 blockDim.y 倍数时，尾部多余线程必须退出，
                            // 否则会以越界行号继续读写，最终层会通过 residual 指针污染 acts.lnf 行 0）
    // 把指针调整到当前 token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;
    const float eps = 1e-5f;
    float sum = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (float)in1[k] + (float)in2[k];
            sum += (float)out[k];
        }
        store128cs(residual + c, out);
        s_res[c / x128::size] = out;
    }

    sum = warpReduceSum(sum);
    float m = sum / C;
    float v = 0.f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 res = s_res[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)res[k] - m) * ((float)res[k] - m);
        }
    }

    v = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 res = s_res[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // 归一化结果
            float o = n * (float)w[k] + (float)b[k]; // 缩放平移
            out[k] = o;
        }
        store128cs(normed + c, out);
    }
    // 缓存均值和标准差倒数
    if(threadIdx.x == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

/* ----------------------------------------------------------------------------
   residual_forward_kernel —— 128 位向量化残差相加 kernel
   ----------------------------------------------------------------------------
   每个线程负责一个数据包，流式加载两个输入并相加后写回。
---------------------------------------------------------------------------- */
__global__ void residual_forward_kernel(floatX* out, const floatX* inp1, const floatX* inp2) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    x128 packed_out;
    x128 packed_inp1 = load128cs(inp1 + idx);
    x128 packed_inp2 = load128cs(inp2 + idx);
    for (int k = 0; k < packed_inp1.size; k++) {
        packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
    }
    store128(out + idx, packed_out);
}

/* ----------------------------------------------------------------------------
   residual_forward_kernel_scalar —— 逐元素残差相加 kernel（兜底）
   ----------------------------------------------------------------------------
   当数据规模无法被 128 位向量宽度整除时使用（本项目常规配置不会走到）。
---------------------------------------------------------------------------- */
__global__ void residual_forward_kernel_scalar(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

// ============================================================================
// Kernel 启动封装
// ============================================================================

/* ----------------------------------------------------------------------------
   layernorm_forward —— LayerNorm 主机端启动函数
   ----------------------------------------------------------------------------
   block 为 (32, block_y)，block_y = 256/32 = 8，即一个 block 处理 8 个 token。
   共享内存可能超过默认 48 KiB 上限，需先调用 cudaFuncSetAttribute 申请；
   若申请失败，则回退到无需共享内存的 kernel3。
---------------------------------------------------------------------------- */
void layernorm_forward(floatX* out, float* mean, float* rstd,
                       floatX* inp, const floatX* weight, const floatX* bias,
                       int B, int T, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    int block_y = block_size / WARP_SIZE;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(floatX);

    // 要使用超过 48 KiB 的共享内存，必须调用 cudaFuncSetAttribute；该调用可能失败，此时回退到不使用共享内存的实现
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(layernorm_forward_kernel6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaCheck(cudaGetLastError());
    if (status == cudaSuccess) {
        layernorm_forward_kernel6<<<grid_size, dim3(WARP_SIZE, block_y), smem, stream>>>(out, mean, rstd, inp, weight, bias, N, C);
    } else {
        // 回退到不使用共享内存的版本
        const int grid_size_fb = CEIL_DIV(N * WARP_SIZE, block_size);
        layernorm_forward_kernel3<<<grid_size_fb, block_size, 0, stream>>>(out, mean, rstd, inp, weight, bias, N, C);
    }
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   residual_forward —— 残差主机端启动函数
   ----------------------------------------------------------------------------
   按 block_size=256、每线程一个数据包计算 grid 大小。
---------------------------------------------------------------------------- */
void residual_forward(floatX* out, const floatX* inp1, const floatX* inp2, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    // 128 位向量化要求 N 能被 block_size * x128::size 整除；
    // 数据规模不满足时（例如单 token 的残差）逐级缩小 block 直到可整除，因此任意 N 都能正确执行（C 是向量宽度的整数倍时，block=1 恒可整除）。
    int block_size = 256;
    while (block_size > 1 && N % (block_size * (int)x128::size) != 0) {
        block_size /= 2;
    }
    if (N % (int)x128::size == 0) {
        const int grid_size = N / (block_size * (int)x128::size);
        residual_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp1, inp2);
    } else {
        // 连最小向量宽度都无法整除时，退回逐元素处理（非常规配置兜底）
        const int grid_size = CEIL_DIV(N, 256);
        residual_forward_kernel_scalar<<<grid_size, 256, 0, stream>>>(out, inp1, inp2, N);
    }
    cudaCheck(cudaGetLastError());
}

/* ----------------------------------------------------------------------------
   fused_residual_forward5 —— 融合残差+LayerNorm 主机端启动函数
   ----------------------------------------------------------------------------
   与 layernorm_forward 类似，先尝试申请大共享内存启动融合 kernel；
   若失败则退化为「先单独做残差、再单独做 LayerNorm」两步。
---------------------------------------------------------------------------- */
void fused_residual_forward5(floatX* residual, floatX* normed, float* mean, float* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, cudaStream_t stream) {
    const int block_size = 256;
    int block_y = block_size / WARP_SIZE;
    const int grid_size = CEIL_DIV(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(floatX);

    // 要使用超过 48 KiB 的共享内存，必须调用 cudaFuncSetAttribute；该调用可能失败，此时回退到分步实现
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(fused_residual_forward_kernel5, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaCheck(cudaGetLastError());
    if(status == cudaSuccess) {
        fused_residual_forward_kernel5<<<grid_size, dim3(WARP_SIZE, block_y), smem, stream>>>(residual, normed,
                                                                                              mean, rstd, inp1, inp2,
                                                                                              weight, bias, N, C);
    } else {
        residual_forward(residual, inp1, inp2, N*C, stream);
        layernorm_forward(normed, mean, rstd, residual, weight, bias, N, 1, C, stream);
    }
    cudaCheck(cudaGetLastError());
}
