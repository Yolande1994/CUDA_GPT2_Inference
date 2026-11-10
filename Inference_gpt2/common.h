/* ============================================================================
   common.h —— CUDA 算子公共工具库
   ============================================================================

   用途：
     本头文件提供所有算子共用的基础工具，避免每个文件重复写通用初始化、错误检查、规约、向量化访存等代码，
     让各算子文件专注 kernel 本身的优化。

   提供的工具列表：
     1. 全局宏与设备属性（WARP_SIZE、CEIL_DIV、True/False、MAX_1024_THREADS_BLOCKS）
     2. 编译期精度切换（FP32 / FP16 / BF16，floatX）与 cuBLAS 精度（CUBLAS_LOWP）
     3. cuBLAS / cuBLASLt 全局句柄与工作区
     4. 错误检查宏（cudaCheck、cudaFreeCheck、cublasCheck）
     5. 低精度流式访存兼容（旧版 nvcc 的 __ldcs/__stcs）
     6. NVTX 性能区间工具
     7. 128 位向量化读写封装（Packed128、load128、store128 等）
     8. 数据类型 DType 支持与类型转换
     9. Warp / Block 级规约（warpReduceSum、warpReduceMax、blockReduce）
     10. 条件性托管内存分配
     11. 随机舍入所需的随机数生成
     12. 显存与文件之间的双缓冲读写（device_to_file / file_to_device）
   ============================================================================ */
#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <string>
#include <type_traits>      // std::bool_constant
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "utils.h"

// ============================================================================
// 全局宏与设置
// ============================================================================

// 本进程使用的 CUDA 设备属性；
// 这里用 extern 声明，因为各 kernel 都希望使用它，而它实际在主程序文件中被创建和实例化
extern cudaDeviceProp deviceProp;

// Warp 大小不是编译期常量，但在此定义成宏，可能有助于编译器做更好的优化
#define WARP_SIZE 32U

// 在 sm_800 及更新架构上尽量让一个 SM 同时容纳 2 个大 block，以提升对延迟的容忍度；
// 这必须用宏定义而非运行时查询，才能用于 __launch_bounds__
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

// 计算 kernel grid/block 尺寸的便捷宏：整数除法向上取整
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 可作为函数参数使用的编译期布尔值快捷方式
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<false> False;

// ============================================================================
// CUDA 精度设置
// ============================================================================

// 精度模式枚举
enum PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// 根据启用的精度做相应配置，floatX 为全工程统一的浮点类型
#if defined(ENABLE_FP32)
typedef float floatX;
#define PRECISION_MODE PRECISION_FP32
// 使用 fp16（可能需要梯度缩放，当前未实现）
#elif defined(ENABLE_FP16)
typedef half floatX;
#define PRECISION_MODE PRECISION_FP16
#else // 默认使用 bfloat16
typedef __nv_bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16
#endif

// ============================================================================
// cuBLAS 精度与全局句柄
// ============================================================================

// CUBLAS_LOWP 为矩阵 IO 使用的低精度数据类型
#if defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#elif defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#else // 默认使用 bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// 工作区硬编码为 32 MiB（只有 Hopper 需要 32，其他架构 4 MiB 即可）
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
void* cublaslt_workspace = NULL;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ============================================================================
// 错误检查
// ============================================================================

/* ----------------------------------------------------------------------------
   cudaCheck_ —— CUDA 错误检查函数
   ----------------------------------------------------------------------------
   失败时打印文件名、行号与错误字符串并退出。
   函数名加下划线，使其既可以通过宏调用，也可以直接调用。
---------------------------------------------------------------------------- */
inline void cudaCheck_(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA 错误] 位于文件 %s:%d：\n%s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
// 自动带入调用处的文件名和行号
#define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

/* ----------------------------------------------------------------------------
   cudaFreeCheck —— 释放显存并检查错误、重置指针
   ----------------------------------------------------------------------------
   类似 cudaFree，但额外检查错误，并把指针置为空，避免悬空指针。
---------------------------------------------------------------------------- */
template<class T>
inline void cudaFreeCheck(T** ptr, const char *file, int line) {
    cudaError_t error = cudaFree(*ptr);
    if (error != cudaSuccess) {
        printf("[CUDA 错误] 位于文件 %s:%d：\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#define cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

/* ----------------------------------------------------------------------------
   cublasCheck —— cuBLAS 错误检查
   ----------------------------------------------------------------------------
   失败时打印状态码、文件名、行号并退出。
---------------------------------------------------------------------------- */
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS 错误]：%d 位于文件 %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
// 自动带入调用处的文件名和行号
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ============================================================================
// 带缓存提示的流式读写（兼容旧版 nvcc）
// ============================================================================
// 旧版 nvcc 没有为 bfloat16 提供 __ldcs / __stcs，尽管它们本质上就是 unsigned short。
// 这里仅在系统版本不存在时才定义自己的版本，
// 否则编译器会报“无可用重载”（sm52）或“函数已存在”（sm_80）。
#if defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(const floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
    return __nv_bfloat16_raw{bf};
}

__device__ void __stcs(floatX* address, floatX value) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)value).x);
}
#endif

// ============================================================================
// NVTX 性能分析工具
// ============================================================================

// NvtxRange —— NVTX 区间的 RAII 封装：构造时压入区间，析构时弹出，便于在 Nsight 中按区间查看耗时。
class NvtxRange {
 public:
    NvtxRange(const char* s) { nvtxRangePush(s); }
    // 带编号的区间名，例如 "Layer 0"
    NvtxRange(const std::string& base_str, int number) {
        std::string range_string = base_str + " " + std::to_string(number);
        nvtxRangePush(range_string.c_str());
    }
    ~NvtxRange() { nvtxRangePop(); }
};
// 在函数入口使用，自动以函数名建立 NVTX 区间
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ============================================================================
// 128 位向量化读写封装
// ============================================================================
// 强制编译器在支持的 GPU 上使用 128 位加载/存储（LDG.128、STG.128 指令）。
// 类似 32 位浮点中使用 float4，但支持任意精度。

/* ----------------------------------------------------------------------------
   Packed128 —— 16 字节对齐的打包数据结构
   ----------------------------------------------------------------------------
   设计：
     - alignas(16) 强制 16 字节对齐，硬件才能发射 128 位指令；
     - 元素个数由 sizeof(int4)/sizeof(ElementType) 编译期推导；
     - 用 int4 做原始载体，通过 memcpy 做类型双关，避免严格别名违规。
---------------------------------------------------------------------------- */
template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    // 从显存读出的 int4 构造，memcpy 按字节拷贝，保证类型安全的同时零开销
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    // 生成所有元素均为 value 的包
    __device__  static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }
    // 全 0 包
    __device__ static Packed128 zeros() {
        return constant(0.f);
    }
    // 全 1 包
    __device__ static Packed128 ones() {
        return constant(1.f);
    }

    __device__ ElementType& operator[](int index) {
        return payload[index];
    }
    __device__ const ElementType& operator[](int index) const {
        return payload[index];
    }
    // 写回显存时转回 int4，配合 STG.128
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};

// 从对齐地址加载一个 Packed128（走完整缓存路径）
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
// 从对齐地址流式加载（.cs 提示，绕过 L1，适合只读一次）
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// 向对齐地址写入一个 Packed128（走完整缓存路径）
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
// 向对齐地址流式写入（.cs 提示，绕过 L1）
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// 向对齐地址写入并保留在 L2、绕过 L1（.cg 提示，适合中间结果）
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

// 简短类型别名
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

// ============================================================================
// 数据类型 DType 支持
// ============================================================================

// 用于标识张量数据类型的枚举
enum class DType : uint8_t {
    FP32, FP16, BF16
};

/* ----------------------------------------------------------------------------
   sizeof_dtype —— 返回某数据类型单个标量的字节数
---------------------------------------------------------------------------- */
size_t sizeof_dtype(DType type) {
    switch (type) {
        case DType::FP32:
            return sizeof(float);
        case DType::FP16:
            return sizeof(half);
        case DType::BF16:
            return sizeof(nv_bfloat16);
        default: // 处理未知类型，避免编译器警告
            fprintf(stderr, "未知数据类型\n");
            exit(EXIT_FAILURE);
    }
}

// 由指针类型推导 DType
DType dtype_of(float* f) { return DType::FP32; }
DType dtype_of(nv_bfloat16 * f) { return DType::BF16; }
DType dtype_of(half * f) { return DType::FP16; }

// ============================================================================
// 拷贝与类型转换
// ============================================================================

// 设备端类型转换函数的模板声明
template<typename Td, typename Ts>
__device__ Td cast_value(Ts val);

// float -> float
template<>
__device__ float cast_value<float, float>(float val) {
    return val;
}

// half -> float
template<>
__device__ float cast_value<float, half>(half val) {
    return __half2float(val);
}

// bfloat16 -> float
template<>
__device__ float cast_value<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

/* copy_and_cast_kernel —— 拷贝并在拷贝过程中完成类型转换；
   stride_dst、stride_src 支持带步长的分块（如分批）拷贝。 */
template<typename Td, typename Ts>
__global__ void copy_and_cast_kernel(Td* dst, const Ts* src, size_t n, ptrdiff_t stride_dst, ptrdiff_t stride_src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 后续可改用 grid stride 循环进一步提升性能
    if (idx < n) {
        dst[idx + stride_dst * blockIdx.y] = cast_value<Td, Ts>(src[idx + stride_src * blockIdx.y]);
    }
}

// ============================================================================
// Warp / Block 通信原语
// ============================================================================

/* ----------------------------------------------------------------------------
   warpReduceSum —— Warp 内求和规约
   ----------------------------------------------------------------------------
   蝴蝶规约，用 __shfl_xor_sync 在 warp 内直接交换寄存器，无需共享内存。
---------------------------------------------------------------------------- */
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
/* ----------------------------------------------------------------------------
   warpReduceMax —— Warp 内求最大值规约
   ----------------------------------------------------------------------------
   同样用蝴蝶规约，每步用 fmaxf 取较大值。
---------------------------------------------------------------------------- */
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
/* ----------------------------------------------------------------------------
   blockReduce —— Block 内两级规约
   ----------------------------------------------------------------------------
   要求 warp 内 32 个线程都活跃，但可用于任意 block 大小。
   第一级：warp 内 shuffle 规约；
   第二级：各 warp 结果写非动态共享内存，第 0 个 warp 再规约。
   共享内存是唯一的，因此末尾可省去一次 __syncthreads；
   但若在循环中调用、共享内存被隐式复用，应把 final_sync 置为 1。
---------------------------------------------------------------------------- */
using reduction_func_t = float (*) (float);
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync=false, float out_of_bounds=0.0f) {
    // 两级规约：1) warp 内 shuffle；2) 跨 warp 共享内存；3) 再 warp 内 shuffle
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // 仅在循环中复用共享内存等场景需要
    }
    return block_val;
}

/* ----------------------------------------------------------------------------
   global_sum_single_block_kernel —— 确定性全局求和 kernel
   ----------------------------------------------------------------------------
   确定性通过要求只使用单个 block 实现（多 block 求和顺序不确定会有浮点差异）。
   各线程用 grid stride 方式累加，再用 blockReduce 规约。
---------------------------------------------------------------------------- */
template<class Float>
__global__ void global_sum_single_block_kernel(float* result, const Float* values, size_t count) {
    assert(gridDim.x == 1); // 只能单个 block
    float thread_sum = 0;
    for(size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)values[index];
    }
    float reduction = blockReduce<warpReduceSum>(thread_sum, true);
    if(threadIdx.x == 0) {
        *result = reduction;
    }
}

// 确定性全局求和的主机端封装，固定 1 个 block、1024 线程
template<class Float>
void global_sum_deterministic(float* result, const Float* values, int count, cudaStream_t stream) {
    global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values, count);
    cudaCheck(cudaGetLastError());
}


// ============================================================================
// 内存管理
// ============================================================================

/* ----------------------------------------------------------------------------
   cudaMallocConditionallyManaged —— 条件性托管内存分配
   ----------------------------------------------------------------------------
   优先在设备上分配；若显存不足（OOM），则回退到托管内存（managed memory），速度更慢但至少不会崩溃。
   返回状态码：0=成功，1=回退到了托管内存。
   同时用 cudaMemAdvise 把托管内存的首选位置设为 CPU。
---------------------------------------------------------------------------- */
int cudaMallocConditionallyManaged(void** out, size_t bytes, const char *file, int line) {
    // 先尝试普通设备分配
    cudaError_t err = cudaMalloc(out, bytes);
    if(err == cudaErrorMemoryAllocation) {
        // 显存不足时回退到托管内存
        cudaGetLastError(); // 在下一次调用前重置错误
        cudaCheck_(cudaMallocManaged(out, bytes), file, line);
#if CUDART_VERSION >= 13000
        // CUDA 13 起第 4 参数从 int 设备号改为 cudaMemLocation 结构体；
        // type=Host 表示该托管内存偏好 CPU 位置
        cudaMemLocation location = {cudaMemLocationTypeHost, 0};
        cudaCheck_(cudaMemAdvise(*out, bytes, cudaMemAdviseSetPreferredLocation, location), file, line);
#else
        // CUDA 12 及更早：cudaCpuDeviceId（-1）表示 CPU 位置
        cudaCheck_(cudaMemAdvise(*out, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId), file, line);
#endif
        return 1;
    } else {
        cudaCheck_(err, file, line);
        return 0;
    }
}

#define cudaMallocConditionallyManaged(out, bytes)\
(cudaMallocConditionallyManaged((void**)out, bytes, __FILE__, __LINE__))

// ============================================================================
// 随机舍入所需的随机数生成
// ============================================================================

/* ----------------------------------------------------------------------------
   SquirrelNoise5 —— Squirrel 噪声工具（第 5 版）
   ----------------------------------------------------------------------------
   由 threadIdx/blockIdx 加一个全 GPU 统一的种子生成随机数。
   通过一系列固定的位噪声常量做乘、加、异或混合，输出高质量伪随机整数。
   参考：http://eiserloh.net/noise/SquirrelNoise5.hpp
---------------------------------------------------------------------------- */
__device__ __host__ constexpr unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed)
{
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B;
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;
    unsigned int mangledBits = positionX;
    mangledBits *= SQ5_BIT_NOISE1;
    mangledBits += seed;
    mangledBits ^= (mangledBits >> 9);
    mangledBits += SQ5_BIT_NOISE2;
    mangledBits ^= (mangledBits >> 11);
    mangledBits *= SQ5_BIT_NOISE3;
    mangledBits ^= (mangledBits >> 13);
    mangledBits += SQ5_BIT_NOISE4;
    mangledBits ^= (mangledBits >> 15);
    mangledBits *= SQ5_BIT_NOISE5;
    mangledBits ^= (mangledBits >> 17);
    return mangledBits;
}

// Get2dNoiseUint —— 由二维坐标（x、y）生成噪声，用一个大质数把 y 混入，避免二维上出现规律。
__device__ __host__ constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr unsigned int PRIME_NUMBER = 198491317u; // 位模式不单调的大质数
    unsigned int x = static_cast<unsigned int>(indexX);
    unsigned int y = static_cast<unsigned int>(indexY);

    return SquirrelNoise5(x + (PRIME_NUMBER * y), seed);
}

/* ----------------------------------------------------------------------------
   stochastic_rounding —— 随机舍入（bfloat16 版）
   ----------------------------------------------------------------------------
   目的：把 float 转成 bfloat16 时，不总是四舍五入，而是按小数部分的大小以概率向上/向下舍入，使大量转换在统计上无偏（减少系统性误差）。
   做法：
     1. 由线程坐标生成随机数，保证每个线程随机值不同；
     2. 取随机数的低 16 位作为概率阈值；
     3. 比较 float 尾数的低 16 位与阈值，决定向上还是向下舍入。
---------------------------------------------------------------------------- */
__device__ __forceinline__ void stochastic_rounding(float in, __nv_bfloat16 *out, unsigned int seed) {
    // 保证每个线程得到不同的随机数
    unsigned int random = Get2dNoiseUint(threadIdx.x, blockIdx.x * blockDim.x + blockIdx.y, seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = __float_as_uint(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    *out = __float2bfloat16_rn(__uint_as_float(float_bits));
}
// half 版随机舍入（尚未实现）
__device__ __forceinline__ void stochastic_rounding(float in, half *out, unsigned int random) {
    *out = (float)in; // 待实现
}
// float 版为空操作（FP32 模式下不需要舍入）
__device__ __forceinline__ void stochastic_rounding(float in, float *out, unsigned int random) {
    *out = in;
}


// ============================================================================
// 显存与文件之间的读写工具
// ============================================================================

/* ----------------------------------------------------------------------------
   device_to_file —— 把显存数据双缓冲写入文件
   ----------------------------------------------------------------------------
   使用两个缓冲区（double buffering）在给定 stream 上运行：
     一个缓冲区从显存异步拷出时，另一个缓冲区同时写磁盘，两者重叠，隐藏拷贝与磁盘 IO 的延迟。
   缓冲区使用页锁定内存（pinned memory），传输更快且支持异步。
---------------------------------------------------------------------------- */
inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
    // 分配页锁定缓冲区，获得更快的异步传输
    char* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size));
    // 把缓冲区分成两半
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    // 先填充读缓冲区；第一次拷贝必须等待
    char* gpu_read_ptr = (char*)src;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    gpu_read_ptr += copy_amount;

    std::swap(read_buffer, write_buffer);
    // 主循环：只要还有数据
    while(rest_bytes > 0) {
        // 启动下一次读
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
        // 同时把写缓冲区的内容写入磁盘
        fwriteCheck(write_buffer, 1, write_buffer_size, dest);
        cudaCheck(cudaStreamSynchronize(stream));  // 等待两个缓冲区都就绪

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // 确保最后剩余的写缓冲区也写入文件
    fwriteCheck(write_buffer, 1, write_buffer_size, dest);
    cudaCheck(cudaFreeHost(buffer_space));
}

/* ----------------------------------------------------------------------------
   file_to_device —— 把文件数据双缓冲读入显存
   ----------------------------------------------------------------------------
   与 device_to_file 对称：一个缓冲区从磁盘读入时，另一个缓冲区异步上传到显存，两者重叠。
   页锁定缓冲区使用 WriteCombined（写结合）标志：适合 CPU 写、设备读的场景，可减少对缓存的占用。
---------------------------------------------------------------------------- */
inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
     // 分配页锁定缓冲区，获得更快的异步传输
     // WriteCombined 内存适合 CPU 写、设备通过映射内存或 H2D 传输读取的场景
    char* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size, cudaHostAllocWriteCombined));
    // 把缓冲区分成两半
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    // 先填充读缓冲区
    char* gpu_write_ptr = (char*)dest;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    freadCheck(read_buffer, 1, copy_amount, src);

    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    std::swap(read_buffer, write_buffer);

    // 主循环：只要还有数据
    while(rest_bytes > 0) {
        // 启动下一次读
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
        gpu_write_ptr += write_buffer_size;
        // 同时从磁盘读入
        freadCheck(read_buffer, 1, copy_amount, src);
        cudaCheck(cudaStreamSynchronize(stream));  // 等待两个缓冲区都就绪

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
    }

    // 把最后剩余的写缓冲区上传到 GPU
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaFreeHost(buffer_space));
}
