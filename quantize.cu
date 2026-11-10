/* ============================================================================
   quantize.cu —— 离线量化工具：把 BF16 权重压成 8 位混合格式
   ============================================================================
   输入一个 BF16 权重文件，输出两个文件：
     8 位权重 bin：文件头不变，16 个参数按原顺序排列，但 5 个 Linear 权重被压成每元素 1 字节，其余参数（偏置、LayerNorm、位置嵌入等）仍保持 BF16；
     scale bin：每个量化行一个 float 缩放因子，推理时按行还原数值。

   量化方案：逐行对称量化。一行内先找绝对值最大的数 amax，令 scale = amax / 满量程（INT8 用 127，FP8 E4M3 用 448），
   再把每个数除以 scale 后编码。还原时 数值 ≈ 字节码 × scale。
   之所以只量化权重矩阵：它们占了模型绝大部分体积，而偏置等小张量量化收益极小、还会引入额外误差，因此保留原精度。

   注：FP8 硬件转换指令要求较新架构，低架构上可用 INT8 路径（默认就是 INT8）。
   ============================================================================ */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define cudaCheck(e) do { \
    cudaError_t _e = (e); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA 错误: %s，位置 %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define QFMT_INT8 0
#define QFMT_FP8  1

#define FP8_MAX_VALUE 448.0f
#define INT8_MAX_VALUE 127.0f

/* ----------------------------------------------------------------------------
   quant_rows：逐行量化 kernel，一个 block 处理一行
   第一步：线程并行扫本行，各自记局部绝对值最大值，再用共享内存做树形归约，得到全行 amax；
   第二步：由 amax 算本行 scale 并写回；
   第三步：每个线程把自己负责的元素除以 scale，按目标格式编码成字节。
   ---------------------------------------------------------------------------- */
__global__ void quant_rows(const float* W, uint8_t* W8, float* scale,
                           int M, int K, int fmt) {
    int row = blockIdx.x;   // 一个 block 处理一行
    if (row >= M) return;
    int tid = threadIdx.x;
    int nthreads = 256;
    __shared__ float s_max[256];
    // 网格跨步遍历本行，先求每个线程的局部最大值
    float local = 0.0f;
    for (int i = tid; i < K; i += nthreads)
        local = fmaxf(local, fabsf(W[row * K + i]));  // 取绝对值
    s_max[tid] = local;
    __syncthreads();
    // 共享内存树形归约：相邻两半反复取大，最终 amax 落在 0 号线程
    for (int off = nthreads / 2; off > 0; off >>= 1) {
        if (tid < off) s_max[tid] = fmaxf(s_max[tid], s_max[tid + off]);
        __syncthreads();
    }

    float full_scale = (fmt == QFMT_FP8) ? FP8_MAX_VALUE : INT8_MAX_VALUE;
    // 全零行直接给 scale=1，避免除零；该行还原后仍是零
    float s = (s_max[0] == 0.0f) ? 1.0f : (s_max[0] / full_scale);
    if (tid == 0) scale[row] = s;  // 行 scale 写回

    if (fmt == QFMT_FP8) {
        for (int i = tid; i < K; i += nthreads) {
            float v = W[row * K + i] / s;  // 除以 scale 得到待编码值
            // 硬件指令直接转 FP8 E4M3，SATFINITE 表示越界自动夹到最大可表示值
            uint8_t q = (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
            W8[row * K + i] = q;
        }
    } else {
        for (int i = tid; i < K; i += nthreads) {
            float v = W[row * K + i] / s;
            // 四舍五入到整数，再夹到对称范围 [-127,127]
            int q = __float2int_rn(v);
            q = min(max(q, -127), 127);
            W8[row * K + i] = (uint8_t)(int8_t)q;
        }
    }
}

int main(int argc, char** argv) {
    int fmt = QFMT_INT8;
    int n_pos = 0;
    const char* pos[3] = {nullptr, nullptr, nullptr};
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            const char* v = argv[++i];
            if (strcmp(v, "fp8") == 0) fmt = QFMT_FP8;
            else if (strcmp(v, "int8") == 0) fmt = QFMT_INT8;
            else { fprintf(stderr, "未知格式 %s（可用 int8 / fp8）\n", v); return 1; }
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "未知选项 %s\n", argv[i]); return 1;
        } else if (n_pos < 3) {
            pos[n_pos++] = argv[i];  // 三个位置参数：输入、8位输出、scale 输出
        }
    }
    if (n_pos < 3) {
        fprintf(stderr, "用法: %s <输入bf16> <输出8位bin> <输出scale bin> [--format int8|fp8]\n", argv[0]);
        return 1;
    }
    const char* in_path = pos[0];
    const char* out8_path = pos[1];
    const char* out_s_path = pos[2];
    printf("格式: %s（满量程 %.0f，每行一个 scale）\n",
           fmt == QFMT_FP8 ? "FP8 E4M3" : "INT8",
           fmt == QFMT_FP8 ? FP8_MAX_VALUE : INT8_MAX_VALUE);

    // 读入 BF16 检查点并校验文件头
    FILE* f = fopen(in_path, "rb");
    if (!f) { perror("打开输入文件失败"); return 1; }
    int header[256];
    if (fread(header, sizeof(int), 256, f) != 256) { fprintf(stderr, "读取文件头失败\n"); return 1; }
    if (header[0] != 20251103 || header[1] != 5) { fprintf(stderr, "不是 BF16 v5 格式的 bin 文件\n"); return 1; }
    int maxT = header[2], L = header[4], C = header[6], Vp = header[7];

    // 各参数的元素数，用于计算偏移和总长度
    size_t n_wte   = (size_t)Vp * C;
    size_t n_wpe   = (size_t)maxT * C;
    size_t n_ln1w  = (size_t)L * C;
    size_t n_qkvw  = (size_t)L * 3 * C * C;
    size_t n_qkvb  = (size_t)L * 3 * C;
    size_t n_attpw = (size_t)L * C * C;
    size_t n_attpb = (size_t)L * C;
    size_t n_ln2w  = (size_t)L * C;
    size_t n_fcw   = (size_t)L * 4 * C * C;
    size_t n_fcb   = (size_t)L * 4 * C;
    size_t n_fcpw  = (size_t)L * C * 4 * C;
    size_t n_fcpb  = (size_t)L * C;
    size_t n_lnfw  = C;

    size_t total_elems = n_wte + n_wpe + n_ln1w + n_ln1w + n_qkvw + n_qkvb + n_attpw + n_attpb +
                         n_ln2w + n_ln2w + n_fcw + n_fcb + n_fcpw + n_fcpb + n_lnfw + n_lnfw;

    printf("模型: Vp=%d C=%d L=%d maxT=%d 总元素数=%zu\n", Vp, C, L, maxT, total_elems);

    uint16_t* h_bf16 = (uint16_t*)malloc(total_elems * sizeof(uint16_t));
    if (fread(h_bf16, sizeof(uint16_t), total_elems, f) != total_elems) { fprintf(stderr, "读取权重失败\n"); return 1; }
    fclose(f);

    // BF16 转 FP32：把 16 位数据移到高 16 位、低 16 位补零，即对应浮点值
    float* h_f = (float*)malloc(total_elems * sizeof(float));
    for (size_t i = 0; i < total_elems; i++) {
        uint32_t u = ((uint32_t)h_bf16[i]) << 16;  // 左移 16 位
        memcpy(&h_f[i], &u, 4);
    }

    // 5 个待量化 Linear 在整块权重中的偏移与形状
    size_t off_qkvw = n_wte + n_wpe + 2 * n_ln1w;
    size_t off_attpw = off_qkvw + n_qkvw + n_qkvb;
    size_t off_fcw = off_attpw + n_attpw + n_attpb + 2 * n_ln2w;
    size_t off_fcpw = off_fcw + n_fcw + n_fcb;
    struct { size_t off; int M; int K; const char* name; } layers[] = {
        { 0,           Vp,       C,      "wte"      },
        { off_qkvw,    L*3*C,    C,      "qkvw"     },
        { off_attpw,   L*C,      C,      "attprojw" },
        { off_fcw,     L*4*C,    C,      "fcw"      },
        { off_fcpw,    L*C,      4*C,    "fcprojw"  },
    };
    const int NL = 5;

    cudaCheck(cudaSetDevice(0));
    float *d_f, *d_scale;
    uint8_t* d_w8;
    size_t total_q8 = 0, total_scales = 0;
    for (int i = 0; i < NL; i++) { total_q8 += (size_t)layers[i].M * layers[i].K; total_scales += layers[i].M; }
    cudaCheck(cudaMalloc(&d_f, total_elems * sizeof(float)));  // 浮点权重
    cudaCheck(cudaMalloc(&d_scale, total_scales * sizeof(float)));
    cudaCheck(cudaMalloc(&d_w8, total_q8));  // 字节权重
    cudaCheck(cudaMemcpy(d_f, h_f, total_elems * sizeof(float), cudaMemcpyHostToDevice));  // 上传

    uint8_t* h_w8 = (uint8_t*)malloc(total_q8);
    float* h_scales = (float*)malloc(total_scales * sizeof(float));
    size_t q8_off = 0, scale_off = 0;
    for (int i = 0; i < NL; i++) {
        int M = layers[i].M, K = layers[i].K;
        printf("量化 %s: M=%d K=%d 字节数=%zu\n", layers[i].name, M, K, (size_t)M * K);
        // 一个 block 一行，共 M 个 block
        quant_rows<<<M, 256>>>(d_f + layers[i].off, d_w8 + q8_off, d_scale + scale_off, M, K, fmt);
        cudaCheck(cudaGetLastError());
        q8_off += (size_t)M * K;  // 推进字节偏移
        scale_off += M;           // 推进 scale 偏移
    }
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(h_w8, d_w8, total_q8, cudaMemcpyDeviceToHost));  // 结果拷回
    cudaCheck(cudaMemcpy(h_scales, d_scale, total_scales * sizeof(float), cudaMemcpyDeviceToHost));

    // 按原 16 参数布局重组输出：量化位标记每个参数是否为字节
    struct { size_t elems; int linear; } layout[16] = {
        { n_wte,  1 }, { n_wpe, 0 }, { n_ln1w, 0 }, { n_ln1w, 0 },
        { n_qkvw, 1 }, { n_qkvb, 0 }, { n_attpw, 1 }, { n_attpb, 0 },
        { n_ln2w, 0 }, { n_ln2w, 0 }, { n_fcw, 1 }, { n_fcb, 0 },
        { n_fcpw, 1 }, { n_fcpb, 0 }, { n_lnfw, 0 }, { n_lnfw, 0 },
    };
    int linear_idx[16] = { 0, -1, -1, -1, 1, -1, 2, -1, -1, -1, 3, -1, 4, -1, -1, -1 };
    FILE* fo = fopen(out8_path, "wb");
    if (!fo) { perror("打开输出权重文件失败"); return 1; }
    fwrite(header, sizeof(int), 256, fo);
    // 每个量化 Linear 在字节缓冲区中的起始偏移
    size_t q8_off_lut[5] = { 0, 0, 0, 0, 0 };
    for (int i = 1; i < NL; i++) q8_off_lut[i] = q8_off_lut[i-1] + (size_t)layers[i-1].M * layers[i-1].K;
    // 每个参数在 FP32 缓冲区中的起始偏移
    size_t hf_off[16];
    size_t acc = 0;
    for (int i = 0; i < 16; i++) { hf_off[i] = acc; acc += layout[i].elems; }
    for (int i = 0; i < 16; i++) {
        if (layout[i].linear) {
            int li = linear_idx[i];
            // 量化参数：直接写字节
            fwrite(h_w8 + q8_off_lut[li], 1, layout[i].elems, fo);
        } else {
            size_t n = layout[i].elems;
            uint16_t* bf = (uint16_t*)malloc(n * sizeof(uint16_t));
            // 非量化参数：FP32 转回 BF16（截掉低 16 位）
            for (size_t j = 0; j < n; j++) {
                uint32_t u; memcpy(&u, &h_f[hf_off[i] + j], 4);
                bf[j] = (uint16_t)(u >> 16);
            }
            fwrite(bf, sizeof(uint16_t), n, fo);
            free(bf);
        }
    }
    fclose(fo);
    // scale 单独成文件，顺序与 5 个 Linear 一一对应
    FILE* fs = fopen(out_s_path, "wb");
    if (!fs) { perror("打开输出 scale 文件失败"); return 1; }
    fwrite(h_scales, sizeof(float), total_scales, fs);  // 全部 scale 落盘
    fclose(fs);

    size_t out_bytes = (size_t)1024 + total_q8 + (total_elems - total_q8) * 2;
    printf("完成: %s (%zu 字节), %s (%zu 个 float)\n", out8_path, out_bytes, out_s_path, total_scales);
    return 0;
}
