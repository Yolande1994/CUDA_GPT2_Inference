/* ============================================================================
   GEMV 算子优化实验：矩阵 × 向量（FP32 / BF16 双精度）
   ============================================================================
   GEMV（General Matrix-Vector multiplication，通用矩阵向量乘）：
     计算 out = W·x。W 是 (M, K) 的行主序矩阵，x 是长度 K 的向量，
     out 是长度 M 的向量，即 out[m] = W 第 m 行与 x 的点积。

   为何单独写它：
     大模型逐 token 生成时一次只输出一个 token，这个阶段所有全连接层都从"矩阵乘矩阵"退化成"矩阵乘向量"，GEMV 是推理链路上出现最多的算子。

   它是带宽瓶颈：
     每做一次乘加只需 2 次浮点运算，却要从显存读一个权重元素。
     以 (2304, 768) 这个尺寸为例：全部计算约 0.02 微秒，而读权重约 10 微秒，搬运比计算慢几百倍。
     所以这里的优化目标不是少算，而是让显存带宽一刻也不闲。

   本文件包含 6 个手写版本（逐步优化）+ 2 个官方库对照：
     GEMV1  每线程一行，访存不合并
     GEMV2  一个 warp 合作一行，合并访存 + warp 内归约
     GEMV3  在 GEMV2 基础上改用 128-bit 向量加载
     GEMV4  软件流水：寄存器预取下一段，边算边等内存
     GEMV5  每轮发两个 128-bit 加载（ILP=2），更多读请求同时在途
     GEMV6  与 GEMV5 同形态，调参后的启动配置（部署版）
     GEMV7  cuBLAS 官方库对照
     GEMV8  cuBLASLt 官方库对照

   编译（XX 替换为目标 GPU 的计算能力编号）：
     nvcc -O3 --use_fast_math -std=c++17 -Xcompiler -fopenmp -arch=sm_XX GEMV.cu -o GEMV -lcublas -lcublasLt

   运行：
     ./GEMV          热缓存，遍历全部尺寸与两种精度
     ./GEMV cold     冷缓存（每次计时前清空 L2，模拟权重不在缓存里）
     ./GEMV fp32     只测 FP32
     ./GEMV bf16     只测 BF16
   ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>

// ============================================================================
// 公共工具：错误检查与向上取整
// ============================================================================

// 检查 CUDA API 返回值：失败就打印出错位置和原因并退出，避免错误被静默吞掉
#define cudaCheck(err) { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { printf("CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } \
}

// 同上，检查 cuBLAS/cuBLASLt 的返回值
#define cublasCheck(err) { \
    cublasStatus_t _e = (err); \
    if (_e != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS 错误 %s:%d 代码=%d\n", __FILE__, __LINE__, _e); exit(1); } \
}

// 向上取整除法：ceil_div(100, 32) = 4，常用于算需要多少个 block
inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ============================================================================
// 设备信息：打印理论显存带宽，作为性能对比的参照上限
// ============================================================================
void print_device_props() {
    cudaDeviceProp p;
    cudaCheck(cudaGetDeviceProperties(&p, 0));
    int clock_khz = 0, bus_bits = 0;
    // 显存时钟和总线位宽，用于下面手算理论带宽
    cudaCheck(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, 0));
    cudaCheck(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    // DDR 显存在时钟上下沿各传一次，所以乘 2；除以 8 把 bit 换算成字节
    double bw = (double)clock_khz * 2.0 * bus_bits / 8.0 / 1e6;
    printf("设备: %s\n", p.name);
    printf("SM 数: %d | L2 缓存: %.1f MB | 显存: %.1f GB\n", p.multiProcessorCount, p.l2CacheSize / 1e6, p.totalGlobalMem / 1e9);
    printf("显存时钟: %d MHz | 位宽: %d bit | 理论带宽: %.1f GB/s\n\n", clock_khz / 1000, bus_bits, bw);
}

// ============================================================================
// CPU 参考实现：结果作为校验 GPU kernel 的标准答案
// ============================================================================
template <typename T>
void GEMV_cpu(float* out, const T* W, const T* x, int M, int K) {
    #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += (float)W[m * K + k] * (float)x[k];
        out[m] = sum;
    }
}

// ============================================================================
// 随机数据与结果校验
// ============================================================================

// 生成 n 个取值在 [-1, 1) 的随机数
inline float* make_random_float(int n) {
    float* p = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) p[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    return p;
}

// 把 GPU 结果拷回主机，与 CPU 答案逐元素比最大误差，超容差就中止
inline void validate_result(float* dev, float* host, const char* name, int n, float tol) {
    float* tmp = (float*)malloc(n * sizeof(float));
    // GPU 结果拷回主机再和 CPU 答案逐元素比较
    cudaCheck(cudaMemcpy(tmp, dev, n * sizeof(float), cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = tmp[i] - host[i];
        if (d < 0) d = -d;                // 取绝对误差
        if (d > max_err) max_err = d;    // 保留最大误差
    }
    free(tmp);
    if (max_err > tol) {
        printf("  验证失败 %s: 最大误差=%.6f (tol=%.3f)\n", name, max_err, tol);
        exit(1);
    }
}

// ============================================================================
// L2 缓存冲刷：冷缓存模式下，把权重从 L2 里挤出去，强制走显存
// ============================================================================
static char*  g_flush_buf  = nullptr;
static size_t g_flush_size = 0;

__global__ void flush_l2_kernel(char* buffer, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // 网格跨步循环写满整个缓冲，占住所有缓存行
    for (int i = idx; i < n; i += stride) {
        buffer[i] = (char)(i % 256);  // 写什么不重要，占住缓存行才是目的
    }
}

void init_flush_buffer(size_t bytes) {
    g_flush_size = bytes;
    cudaCheck(cudaMalloc(&g_flush_buf, g_flush_size));
}

void destroy_flush_buffer() {
    if (g_flush_buf) {
        cudaCheck(cudaFree(g_flush_buf));
        g_flush_buf  = nullptr;
        g_flush_size = 0;
    }
}

void flush_l2_cache() {
    if (g_flush_buf == nullptr) return;
    flush_l2_kernel<<<4096, 256>>>(g_flush_buf, (int)g_flush_size);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// 128-bit 向量块辅助：一次处理 16 字节（FP32 4 个 / BF16 8 个元素）
// ============================================================================

/* ----------------------------------------------------------------------------
   accumulate_block：读入一个 128-bit 块的 W 和 x，乘加进累加器。
   BF16 没有专用的 16 字节向量类型，用 union 把 uint4 的同一块内存
   重新解释成 8 个 BF16，既享受向量加载、又能按元素取值。
   ---------------------------------------------------------------------------- */
template <typename T>
__device__ __forceinline__ void accumulate_block(float& sum, const T* wrow, const T* xrow, int off) {
    if constexpr (sizeof(T) == 4) {
        const float4 w = *(const float4*)(wrow + off);
        const float4 xv = *(const float4*)(xrow + off);
        sum += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
    } else {
        union { uint4 u; __nv_bfloat16 h[8]; } w, xv;
        w.u = *(const uint4*)(wrow + off);
        xv.u = *(const uint4*)(xrow + off);
        #pragma unroll
        for (int i = 0; i < 8; i++) sum += (float)w.h[i] * (float)xv.h[i];
    }
}

// 一行能切成多少个 128-bit 块：FP32 每块 128 元素，BF16 每块 256 元素
template <typename T>
__device__ __forceinline__ int block_rounds(int K) {
    return (sizeof(T) == 4) ? K / 128 : K / 256;
}


// ============================================================================
// 手写 kernel：GEMV1 - GEMV6
// ============================================================================

/* ----------------------------------------------------------------------------
   GEMV1：朴素版，一个线程算一整行。
   同一时刻相邻线程访问的是不同行、地址相隔 K 个元素，硬件无法把这些读请求合并，这是它慢的根本原因，作为优化前后的对照。
   ---------------------------------------------------------------------------- */
template <typename T>
__global__ void GEMV1(float* out, const T* W, const T* x, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const T* wrow = W + row * K;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        // 相邻线程读不同行、地址相隔 K，访存不合并，每条读请求都要单独发
        sum += (float)wrow[k] * (float)x[k];
    out[row] = sum;
}

template <typename T>
void launch_GEMV1(float* out, const T* W, const T* x, int M, int K) {
    GEMV1<T><<<ceil_div(M, 256), 256>>>(out, W, x, M, K);
    cudaCheck(cudaGetLastError());
}


/* ----------------------------------------------------------------------------
   GEMV2：warp-per-row，一个 warp 合作算一行。
   32 个 lane 按步长 32 错开取数，同一时刻读的是连续地址，天然合并；
   各 lane 先算自己的部分和，最后用 shuffle 在 warp 内归约成总和。
   ---------------------------------------------------------------------------- */
template <typename T>
__global__ void GEMV2(float* out, const T* W, const T* x, int M, int K) {
    // 每 32 个线程组成一个 warp 合作一行，整除 32 得到行号
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;  // 本线程在 warp 内的编号 0~31
    if (row >= M) return;
    const T* wrow = W + row * K;
    float sum = 0.0f;

    // lane 按步长 32 错开取数，32 个 lane 同时读连续地址，访存合并
    for (int k = lane; k < K; k += 32)
        sum += (float)wrow[k] * (float)x[k];

    // warp 内归约：线程之间直接交换寄存器，不走共享内存，5 轮后汇聚到 lane 0
    #pragma unroll
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) out[row] = sum;     // 只有 lane 0 持有总和，负责写回
}

template <typename T>
void launch_GEMV2(float* out, const T* W, const T* x, int M, int K) {
    dim3 grid(ceil_div(M, 8));
    GEMV2<T><<<grid, 256>>>(out, W, x, M, K);
    cudaCheck(cudaGetLastError());
}


/* ----------------------------------------------------------------------------
   GEMV3：在 GEMV2 基础上把单元素加载换成 128-bit 向量加载。
   同样的数据量，访存指令数降到原来的 1/4（FP32）或 1/8（BF16），
   指令发射压力小了，带宽才更容易跑满。
   ---------------------------------------------------------------------------- */
template <typename T>
__global__ void GEMV3(float* out, const T* W, const T* x, int M, int K) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= M) return;
    const T* wrow = W + row * K;
    float sum = 0.0f;
    const int BLOCKS = block_rounds<T>(K);  // 一行能切成多少个 128-bit 块
    const int STRIDE = (sizeof(T) == 4) ? 128 : 256;  // 相邻两块之间的元素步长
    // 每个 lane 负责块内的起始偏移：FP32 4 元素、BF16 8 元素
    const int BASE = (sizeof(T) == 4) ? lane * 4 : lane * 8;
    #pragma unroll 4
    for (int r = 0; r < BLOCKS; r++) {
        int off = r * STRIDE + BASE;     // 本 lane 在当前块内的绝对偏移
        accumulate_block(sum, wrow, x, off);  // 一次读 128-bit 并乘加
    }
    #pragma unroll
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) out[row] = sum;
}

template <typename T>
void launch_GEMV3(float* out, const T* W, const T* x, int M, int K) {
    dim3 grid(ceil_div(M, 8));
    GEMV3<T><<<grid, 256>>>(out, W, x, M, K);
    cudaCheck(cudaGetLastError());
}


/* ----------------------------------------------------------------------------
   GEMV4：软件流水。把下一段的加载提前发出去，同时计算当前段，
   让 “等内存” 和 “做计算” 在时间上重叠，隐藏显存延迟。
   带宽瓶颈下计算太少、收益有限，但这是通用的延迟隐藏手法。
   ---------------------------------------------------------------------------- */
template <typename T>
__global__ void GEMV4(float* out, const T* W, const T* x, int M, int K) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= M) return;
    const T* wrow = W + row * K;
    const int BLOCKS = block_rounds<T>(K);
    const int STRIDE = (sizeof(T) == 4) ? 128 : 256;
    const int BASE = (sizeof(T) == 4) ? lane * 4 : lane * 8;
    float sum = 0.0f;

    // 当前段的权重/输入指针
    const T* w_cur = wrow + BASE;
    const T* x_cur =    x + BASE;

    #pragma unroll 4
    for (int r = 0; r < BLOCKS; r++) {
        // 最后一段没有“下一段”，指针指向自己，避免越界
        const T* w_nxt = (r + 1 < BLOCKS) ? wrow + (r + 1) * STRIDE + BASE : w_cur;
        const T* x_nxt = (r + 1 < BLOCKS) ?    x + (r + 1) * STRIDE + BASE : x_cur;

        accumulate_block(sum, w_cur, x_cur, 0);  // 算当前段；下一段的加载已在途
        w_cur = w_nxt; x_cur = x_nxt;            // 推进到下一段
    }
    #pragma unroll
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) out[row] = sum;
}

template <typename T>
void launch_GEMV4(float* out, const T* W, const T* x, int M, int K) {
    dim3 grid(ceil_div(M, 8));
    GEMV4<T><<<grid, 256>>>(out, W, x, M, K);
    cudaCheck(cudaGetLastError());
}

// 为什么不用共享内存：GEMV 的权重只流式读一遍、没有复用，搬进共享内存纯属多此一举；寄存器预取已经是这种场景的最优解。


/* ----------------------------------------------------------------------------
   GEMV5：每轮发两个相互独立的 128-bit 加载（ILP=2，指令级并行）。
   同时在途的读请求翻倍，GPU 可以一边等第一批数据、一边发第二批，
   尾部不足一整轮时用单块兜底。
   ---------------------------------------------------------------------------- */
template <typename T>
__global__ void GEMV5(float* out, const T* W, const T* x, int M, int K) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= M) return;
    const T* wrow = W + row * K;
    float sum = 0.0f;

    if constexpr (sizeof(T) == 4) {
        const int BLOCKS = K / 256;  // 每轮处理 256 元素（两个 128-bit 块）
        const int BASE = lane * 4;
        #pragma unroll 4
        for (int r = 0; r < BLOCKS; r++) {
            int off = r * 256 + BASE;
            // 第一块
            const float4 w0 = *(const float4*)(wrow + off);
            const float4 x0 = *(const float4*)(x + off);
            // 第二块：与第一块互不依赖，两个加载同时在途（ILP=2）
            const float4 w1 = *(const float4*)(wrow + off + 128);
            const float4 x1 = *(const float4*)(x + off + 128);
            sum += w0.x * x0.x + w0.y * x0.y + w0.z * x0.z + w0.w * x0.w;
            sum += w1.x * x1.x + w1.y * x1.y + w1.z * x1.z + w1.w * x1.w;
        }

        if constexpr (true) {
            const int rem = K % 256;  // 尾部不足一整轮的元素数
            if (rem != 0) {
                int off = BLOCKS * 256 + lane * 4;
                if (off + 3 < K) {   // 单块兜底，够 4 个元素才读
                    const float4 w0 = *(const float4*)(wrow + off);
                    const float4 x0 = *(const float4*)(x + off);
                    sum += w0.x * x0.x + w0.y * x0.y + w0.z * x0.z + w0.w * x0.w;
                }
            }
        }
    } else {
        const int BLOCKS = K / 512;  // BF16 每轮 512 元素
        const int BASE = lane * 8;
        #pragma unroll 4
        for (int r = 0; r < BLOCKS; r++) {
            int off = r * 512 + BASE;
            union { uint4 u; __nv_bfloat16 h[8]; } w0, x0, w1, x1;
            w0.u = *(const uint4*)(wrow + off);
            x0.u = *(const uint4*)(x + off);
            // 第二块，同样双加载并行
            w1.u = *(const uint4*)(wrow + off + 256);
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
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) out[row] = sum;
}

template <typename T>
void launch_GEMV5(float* out, const T* W, const T* x, int M, int K) {
    dim3 grid(ceil_div(M, 8));
    GEMV5<T><<<grid, 256>>>(out, W, x, M, K);
    cudaCheck(cudaGetLastError());
}


/* ----------------------------------------------------------------------------
   GEMV6：部署版。计算结构与 GEMV5 完全相同，区别只在启动参数：
   block 512（每个 block 16 个 warp）、循环展开 2 份，
   来自对近百个配置的网格扫描，在多种尺寸上表现最稳定。
   ---------------------------------------------------------------------------- */
template <typename T>
__global__ void GEMV6(float* out, const T* W, const T* x, int M, int K) {
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

        const int rem = K % 256;
        if (rem != 0) {
            int off = BLOCKS * 256 + lane * 4;
            if (off + 3 < K) {
                const float4 w0 = *(const float4*)(wrow + off);
                const float4 x0 = *(const float4*)(x + off);
                sum += w0.x * x0.x + w0.y * x0.y + w0.z * x0.z + w0.w * x0.w;
            }
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

        const int rem = K % 512;
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
    for (int off = 16; off > 0; off /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) out[row] = sum;
}

template <typename T>
void launch_GEMV6(float* out, const T* W, const T* x, int M, int K) {
    dim3 grid(ceil_div(M, 16));
    GEMV6<T><<<grid, 512>>>(out, W, x, M, K);
    cudaCheck(cudaGetLastError());
}


// ============================================================================
// GEMV7：cuBLAS 官方库对照
// ============================================================================
static cublasHandle_t cublas_handle;

void cublas_init() {
    cublasCheck(cublasCreate(&cublas_handle));
    // 默认数学模式，不允许 TF32 等加速，保证和手写 kernel 同精度口径
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
}
void cublas_cleanup() { cublasCheck(cublasDestroy(cublas_handle)); }

/* ----------------------------------------------------------------------------
   FP32 版：cublasSgemv。
   cuBLAS 只认列主序，而行主序的 W(M,K) 在内存排布上恰好等价于一个列主序的 (K,M) 矩阵，
   用 CUBLAS_OP_T 转置后就是 (M,K)，算出来正好是 W·x，不需要真的搬动数据。
   ---------------------------------------------------------------------------- */
void gemv_cublas(float* out, const float* W, const float* x, int M, int K) {
    float alpha = 1.0f, beta = 0.0f;   // alpha=1、beta=0 即纯乘法，不累加旧值
    // 列主序转置技巧：CUBLAS_OP_T 把行主序 W 当作转置后的列主序矩阵
    cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_T, K, M, &alpha, W, K, x, 1, &beta, out, 1));
}

/* BF16 版没有专用的 GEMV 接口，改用 cublasGemmEx 把 N（列数）设成 1，即退化成 GEMV；
   输入 BF16、FP32 累加，与手写 kernel 的精度口径一致。 */
void gemv_cublas(float* out, const __nv_bfloat16* W, const __nv_bfloat16* x, int M, int K) {
    float alpha = 1.0f, beta = 0.0f;

    // N=1 的 GEMM 即 GEMV；输入 BF16、FP32 累加
    cublasCheck(cublasGemmEx(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N, M, 1, K,
        &alpha,
        W, CUDA_R_16BF, K,   // 输入矩阵类型 BF16
        x, CUDA_R_16BF, K,
        &beta,
        out, CUDA_R_32F, M,  // 输出固定 FP32
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
}


// ============================================================================
// GEMV8：cuBLASLt 官方库对照
// ============================================================================
static cublasLtHandle_t lt_handle;
static void* lt_workspace = nullptr;
static size_t lt_workspace_size = 4 * 1024 * 1024;

void cublaslt_init() {
    cublasCheck(cublasLtCreate(&lt_handle));
    // 工作区给启发式选中的 kernel 当临时空间
    cudaCheck(cudaMalloc(&lt_workspace, lt_workspace_size));
}

/* ----------------------------------------------------------------------------
   cuBLASLt 是 cuBLAS 的低层接口：先描述矩阵布局和偏好，再由启发式搜索自动挑选 kernel。
   这里同样用 N=1 的 GEMM 形式 + 列主序转置技巧。
   ---------------------------------------------------------------------------- */
template <typename T>
void gemv_cublaslt(float* out, const T* W, const T* x, int M, int K) {
    float alpha = 1.0f, beta = 0.0f;
    cudaDataType_t ab_type = (sizeof(T) == 4) ? CUDA_R_32F : CUDA_R_16BF;
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t A_desc, B_desc, C_desc;
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;

    cublasCheck(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));  // FP32 累加
    cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;  // 同样的列主序转置技巧
    cublasCheck(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    cublasCheck(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    // 三个矩阵布局，最后一个参数是 leading dimension（行主序即每行的跨度）
    cublasCheck(cublasLtMatrixLayoutCreate(&A_desc, ab_type, K, M, K));
    cublasCheck(cublasLtMatrixLayoutCreate(&B_desc, ab_type, K, 1, K));
    cublasCheck(cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_32F, M, 1, M));
    cublasCheck(cublasLtMatmulPreferenceCreate(&pref));
    // 告诉启发式搜索可用的工作区上限
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &lt_workspace_size, sizeof(lt_workspace_size)));
    // 启发式搜索：按描述和偏好挑一个 kernel
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(lt_handle, matmulDesc, A_desc, B_desc, C_desc, C_desc,
        pref, 1, &heuristic, &returned));
    // 执行矩阵乘
    cublasCheck(cublasLtMatmul(lt_handle, matmulDesc, &alpha,
        W, A_desc, x, B_desc, &beta, out, C_desc, out, C_desc,
        &heuristic.algo, lt_workspace, lt_workspace_size, 0));
    cublasCheck(cublasLtMatmulDescDestroy(matmulDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(A_desc));
    cublasCheck(cublasLtMatrixLayoutDestroy(B_desc));
    cublasCheck(cublasLtMatrixLayoutDestroy(C_desc));
    cublasCheck(cublasLtMatmulPreferenceDestroy(pref));
}


// ============================================================================
// 版本分发表：版本号 → 对应的启动函数
// ============================================================================
template <typename T>
using Launcher = void(*)(float*, const T*, const T*, int, int);

template <typename T>
Launcher<T> get_launcher(int kernel_num) {
    switch (kernel_num) {
        case 1: return launch_GEMV1<T>;
        case 2: return launch_GEMV2<T>;
        case 3: return launch_GEMV3<T>;
        case 4: return launch_GEMV4<T>;
        case 5: return launch_GEMV5<T>;
        case 6: return launch_GEMV6<T>;
        default: printf("无效的内核编号: %d\n", kernel_num); exit(1);
    }
}

const char* kernel_name(int n) {
    switch (n) {
        case 1: return "GEMV1 朴素(非合并)";
        case 2: return "GEMV2 warp-per-row";
        case 3: return "GEMV3 +128bit向量化";
        case 4: return "GEMV4 软件流水";
        case 5: return "GEMV5 双128bit/轮";
        case 6: return "GEMV6 双128bit·tune512";
        case 7: return "GEMV7 cuBLAS";
        case 8: return "GEMV8 cuBLASLt";
        default: return "?";
    }
}

// ============================================================================
// 计时与统计：CUDA Event 计时 + 中位数汇总
// ============================================================================
struct BenchResult {
    float avg;
    float median;
};

// 单次计时：冷缓存模式先冲刷 L2，再用 CUDA Event 在 GPU 时间轴上打点
template <typename F, typename... Args>
inline void bench_once(F func, cudaEvent_t s, cudaEvent_t e, float& out, Args... args) {
    if (g_flush_buf != nullptr) flush_l2_cache();
    cudaCheck(cudaEventRecord(s));
    func(args...);
    cudaCheck(cudaEventRecord(e));
    cudaCheck(cudaEventSynchronize(e));
    cudaCheck(cudaEventElapsedTime(&out, s, e));
}

// 排序后取中位数：中位数比平均值更不容易被偶发抖动带偏
inline BenchResult summarize(float* samples, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) sum += samples[i];  // 求和算平均

    // 冒泡排序，把样本按耗时从小到大排
    for (int i = 0; i < count - 1; i++)
        for (int j = 0; j < count - 1 - i; j++)
            if (samples[j] > samples[j+1]) {
                float tmp = samples[j]; samples[j] = samples[j+1]; samples[j+1] = tmp;
            }
    // 奇数个取正中间，偶数个取中间两个的平均
    float median = (count % 2 == 1)
        ? samples[count / 2]
        : (samples[count / 2 - 1] + samples[count / 2]) * 0.5f;
    return {sum / count, median};
}

// ============================================================================
// run_benchmark：单个尺寸的完整流程（准备数据 → 校验 → 计时 → 打印表格）
// ============================================================================
template <typename T>
void run_benchmark(int M, int K, const char* dtype_name) {
    constexpr int N_VER = 8;
    constexpr int BLOCKS = 5;
    constexpr int SAMP = 10;
    constexpr int TOTAL = 2 * BLOCKS * SAMP;   // 两个 pass 各 50 样本
    static float samples[9][512];

    float* h_Wf = make_random_float(M * K);    // 主机随机权重
    float* h_xf = make_random_float(K);        // 主机随机输入向量
    float* h_C = (float*)malloc(M * sizeof(float));
    T* d_W = nullptr; T* d_x = nullptr;
    float* d_out;
    cudaCheck(cudaMalloc(&d_out, (size_t)M * sizeof(float)));  // 设备输出
    if (sizeof(T) == 4) {
        // FP32：直接分配并上传
        cudaCheck(cudaMalloc(&d_W, (size_t)M * K * sizeof(float)));
        cudaCheck(cudaMalloc(&d_x, (size_t)K * sizeof(float)));
        cudaCheck(cudaMemcpy(d_W, h_Wf, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_x, h_xf, (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        // BF16：主机侧先转成 BF16 再上传
        __nv_bfloat16* h_Wbf = (__nv_bfloat16*)malloc((size_t)M * K * sizeof(__nv_bfloat16));
        __nv_bfloat16* h_xbf = (__nv_bfloat16*)malloc((size_t)K * sizeof(__nv_bfloat16));
        for (int i = 0; i < M * K; i++) h_Wbf[i] = __float2bfloat16(h_Wf[i]);
        for (int i = 0; i < K; i++) h_xbf[i] = __float2bfloat16(h_xf[i]);
        cudaCheck(cudaMalloc(&d_W, (size_t)M * K * sizeof(__nv_bfloat16)));
        cudaCheck(cudaMalloc(&d_x, (size_t)K * sizeof(__nv_bfloat16)));
        cudaCheck(cudaMemcpy(d_W, h_Wbf, (size_t)M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_x, h_xbf, (size_t)K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        free(h_Wbf); free(h_xbf);
    }

    // CPU 参考用的是同一份设备数据拷回，确保输入完全一致
    T* h_Wt = (T*)malloc((size_t)M * K * sizeof(T));
    T* h_xt = (T*)malloc((size_t)K * sizeof(T));
    cudaCheck(cudaMemcpy(h_Wt, d_W, (size_t)M * K * sizeof(T), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_xt, d_x, (size_t)K * sizeof(T), cudaMemcpyDeviceToHost));
    GEMV_cpu(h_C, h_Wt, h_xt, M, K);

    printf("========== GEMV 尺寸 M=%d  K=%d  (dtype=%s) ==========\n\n", M, K, dtype_name);

    // 预热：让库完成初始化、GPU 时钟进入稳态，避免第一批样本偏慢
    for (int n = 1; n <= N_VER; n++) {
        if (n == 7) { for (int i = 0; i < 5; i++) gemv_cublas(d_out, d_W, d_x, M, K); continue; }
        if (n == 8) { for (int i = 0; i < 5; i++) gemv_cublaslt(d_out, d_W, d_x, M, K); continue; }
        Launcher<T> launch = get_launcher<T>(n);
        for (int i = 0; i < 5; i++) launch(d_out, d_W, d_x, M, K);
    }
    cudaCheck(cudaDeviceSynchronize());

    // 正确性校验：每个版本都要对得上 CPU 答案
    for (int n = 1; n <= N_VER; n++) {
        if (n == 7) gemv_cublas(d_out, d_W, d_x, M, K);
        else if (n == 8) gemv_cublaslt(d_out, d_W, d_x, M, K);
        else { Launcher<T> l = get_launcher<T>(n); l(d_out, d_W, d_x, M, K); }
        cudaCheck(cudaDeviceSynchronize());
        // FP32 容差更紧，BF16 受精度影响容差放宽
        float tol = (sizeof(T) == 4) ? 0.0002f : 0.001f;
        validate_result(d_out, h_C, kernel_name(n), M, tol);
    }
    printf("  全部版本正确性验证通过 (tol=%s)\n\n", (sizeof(T) == 4) ? "0.0002" : "0.001");

    /* 计时采样：两个 pass 分别按正序、倒序各跑 50 个样本。
       这样 GPU 时钟/温度的漂移被均摊给所有版本，没有谁系统性占便宜。 */
    cudaEvent_t s, e;
    cudaCheck(cudaEventCreate(&s));
    cudaCheck(cudaEventCreate(&e));
    for (int pass = 0; pass < 2; pass++) {
        for (int k = 0; k < N_VER; k++) {
            int n = (pass == 0) ? (k + 1) : (N_VER - k);  // pass0 正序、pass1 倒序
            for (int i = 0; i < BLOCKS * SAMP; i++) {
                int slot = pass * BLOCKS * SAMP + i;  // 样本数组中的列位置
                if (n == 7) {
                    auto f = [&](float* out, const T* w, const T* x, int m, int k) { gemv_cublas(out, w, x, m, k); };
                    bench_once(f, s, e, samples[n][slot], d_out, d_W, d_x, M, K);
                }
                else if (n == 8) {
                    auto f = [&](float* out, const T* w, const T* x, int m, int k) { gemv_cublaslt(out, w, x, m, k); };
                    bench_once(f, s, e, samples[n][slot], d_out, d_W, d_x, M, K);
                }
                else { Launcher<T> l = get_launcher<T>(n); bench_once(l, s, e, samples[n][slot], d_out, d_W, d_x, M, K); }
            }
        }
    }
    cudaCheck(cudaEventDestroy(s));
    cudaCheck(cudaEventDestroy(e));

    // 表格输出
    auto pad = [](const char* s, int w, bool left) {
        int n = 0;
        for (const char* p = s; *p; ++p)
            if (((unsigned char)*p & 0xC0) != 0x80)
                n += ((unsigned char)*p < 0x80) ? 1 : 2;
        if (!left) for (; n < w; ++n) putchar(' ');
        printf("%s", s);
        if (left)  for (; n < w; ++n) putchar(' ');
    };
    printf("%-4s ", "版本");
    pad("名称", 20, true);
    printf("%9s %9s %8s %9s %5s\n", "平均(ms)", "中位(ms)", "GB/s", "占cuBLAS", "次数");
    printf("--------------------------------------------------------------------------\n");

    // 以 cuBLASLt 为分母，给出每个版本相对官方库的速度比
    BenchResult cublas_res = summarize(samples[8], TOTAL);
    BenchResult best_res = {1e30f, 1e30f};
    int best_n = 0;
    for (int n = 1; n <= N_VER; n++) {
        BenchResult res = summarize(samples[n], TOTAL);
        // 搬运字节数 = 权重 + 输入向量 + 输出
        double bytes = ((double)M * K + K) * sizeof(T) + M * 4.0;
        double gbps = bytes / (res.median * 1e-3) / 1e9;  // 实测带宽
        // 以 cuBLASLt 耗时为分母，>100% 表示比官方库快
        float pct = cublas_res.median / res.median * 100.0f;
        printf("%-4d ", n);
        pad(kernel_name(n), 20, true);
        printf("%9.3f %9.3f %8.0f %9.1f%% %5d\n", res.avg, res.median, gbps, pct, TOTAL);
        if (res.median < best_res.median) { best_res = res; best_n = n; }
    }
    double best_bytes = ((double)M * K + K) * sizeof(T) + M * 4.0;
    printf("  -> 本尺寸最快: %s (中位 %.3f ms, %.0f GB/s)\n",
           kernel_name(best_n), best_res.median, best_bytes / (best_res.median * 1e-3) / 1e9);

    free(h_Wf); free(h_xf); free(h_C); free(h_Wt); free(h_xt);
    cudaFree(d_W); cudaFree(d_x); cudaFree(d_out);
}


// ============================================================================
// main：解析参数，遍历 GPT-2 推理中的 5 个核心 GEMV 尺寸
// ============================================================================
int main(int argc, char** argv) {
    srand(0);
    cudaCheck(cudaSetDevice(0));
    print_device_props();

    bool cold = false, do_fp32 = true, do_bf16 = true;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "cold") == 0) cold = true;
        if (strcmp(argv[i], "fp32") == 0) do_bf16 = false;  // 只跑 FP32
        if (strcmp(argv[i], "bf16") == 0) do_fp32 = false;  // 只跑 BF16
    }
    if (cold) {
        // 冲刷缓冲要大于 L2 容量，才能保证把权重缓存行全部挤出去
        init_flush_buffer(64ULL * 1024 * 1024);
        printf("模式: 冷缓存（每样本前清空 L2）\n\n");
    }
    cublas_init();
    cublaslt_init();

    // 这些尺寸对应 GPT-2 一次推理里的全部 GEMV：QKV、注意力输出投影、MLP 的两层全连接、最后的词表投影
    struct Dim { int M, K; };
    Dim sizes[] = {
        {2304, 768}, {768, 768}, {3072, 768}, {768, 3072}, {50257, 768}
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("自动遍历全部 %d 种 (M,K) 组合\n\n", n_sizes);
    for (int i = 0; i < n_sizes; i++) {
        if (do_fp32) run_benchmark<float>(sizes[i].M, sizes[i].K, "FP32");
        if (do_bf16) run_benchmark<__nv_bfloat16>(sizes[i].M, sizes[i].K, "BF16");
        printf("\n");
    }

    cublas_cleanup();
    cublasCheck(cublasLtDestroy(lt_handle));
    if (lt_workspace) cudaFree(lt_workspace);
    destroy_flush_buffer();
    return 0;
}
