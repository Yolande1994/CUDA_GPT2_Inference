/* ============================================================================
   sampler_gpu.cuh —— 纯 GPU 端的高效采样器（Softmax + 随机数采样）
   ============================================================================ */
#pragma once

#include <curand_kernel.h>
#include <float.h>
#include "common.h"

__global__ void gpu_sample_softmax_kernel(int* out_token, const float* logits, int B, int V,
                                          unsigned long long seed, int step) {
    const int b = blockIdx.x;
    if (b >= B) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_rand;
    __shared__ float seg[256];
    __shared__ float warp_buf[8];
    __shared__ int s_result;

    const float* b_logits = logits + b * V;

    float local_max = -FLT_MAX;
    for (int i = tid; i < V; i += nthreads) {
        local_max = fmaxf(local_max, (float)b_logits[i]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
    }
    if (tid % 32 == 0) warp_buf[tid / 32] = local_max;
    __syncthreads();
    if (tid < 8) {
        float v = warp_buf[tid];
        for (int off = 4; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xff, v, off));
        }
        if (tid == 0) s_max = v;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < V; i += nthreads) {
        local_sum += expf((float)b_logits[i] - s_max);
    }
    seg[tid] = local_sum;
    float ws = local_sum;
    for (int off = 16; off > 0; off >>= 1) {
        ws += __shfl_xor_sync(0xffffffff, ws, off);
    }
    if (tid % 32 == 0) warp_buf[tid / 32] = ws;
    __syncthreads();
    if (tid < 8) {
        float v = warp_buf[tid];
        for (int off = 4; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xff, v, off);
        }
        if (tid == 0) { s_sum = v; s_result = V - 1; }
    }
    __syncthreads();

    if (tid == 0) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, /*sequence=*/b, /*offset=*/step, &state);
        s_rand = curand_uniform(&state) * s_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float acc = 0.0f;
        for (int j = 0; j < nthreads; j++) {
            float v = seg[j];
            seg[j] = acc;
            acc += v;
        }
    }
    __syncthreads();
    float my_start = seg[tid];
    float my_end = my_start + local_sum;

    if (s_rand >= my_start && s_rand < my_end) {
        float acc = my_start;
        for (int i = tid; i < V; i += nthreads) {
            acc += expf((float)b_logits[i] - s_max);
            if (acc >= s_rand) { s_result = i; break; }
        }
    }
    __syncthreads();
    if (tid == 0) out_token[b] = s_result;
}

inline void gpu_sample_softmax(int* d_token, const float* logits, int B, int V,
                               unsigned long long seed, int step, cudaStream_t stream) {
    gpu_sample_softmax_kernel<<<B, 256, 0, stream>>>(d_token, logits, B, V, seed, step);
    cudaCheck(cudaGetLastError());
}