/* ============================================================================
   matmul.cuh —— 矩阵乘法（基于 cuBLASLt）
   ============================================================================

   算子说明：
     矩阵乘法（GEMM）是 Transformer 中占比最高的计算，
     本文件封装 NVIDIA 官方高度优化的 cuBLASLt 库，并在其 epilogue（收尾阶段）融合 bias、GELU 等操作。
   ============================================================================ */
#pragma once

#include <assert.h>
#include <type_traits>
#include "common.h"
// GELU 既可以用 cuBLASLt 融合，也可以用 gelu.cuh 单独实现
#include "gelu.cuh"

// ============================================================================
// Kernel 启动封装
// ============================================================================

void matmul_cublaslt(void* d, const floatX* a, const floatX* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false,
                     bool out_f32=false)
// d 用 void* 是为了同一个函数既能写 floatX（BF16）也能写 float32（见最后的 out_f32）：
//   cuBLASLt 的 D 指针本来就是 void*，输出类型由 DLayout 的元素类型决定，与指针声明无关。
//   out_f32=true 时 D 必须指向一块 float32 缓冲（每元素 4 字节），否则会写坏内存。
{
    NVTX_RANGE_FN();
    bool has_bias = (bias != NULL);
    bool has_gelu = (pre_gelu != NULL);
    // 校验对齐（部分模式支持非对齐，但对齐总是性能最好）
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        printf("所有 cuBLASLt 指针必须对齐！\n");
        exit(EXIT_FAILURE);
    }

    // 创建操作描述符
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA)  ? &opTranspose : &opNoTranspose,   sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose   : &opNoTranspose, sizeof(opNoTranspose)));

    // 定义矩阵布局
    cublasLtMatrixLayout_t ALayout, BLayout, DLayout, CLayout;
    if (transA) {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m));
    }
    if (transB) {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k));
    }
    //  C/D 的元素类型：默认与 floatX 同宽（BF16 / FP32）；
    //  out_f32=true 时强制成 float32——A、B 仍是 BF16，只有结果按 4 字节落盘，
    //  这样 kernel 内部的乘加精度（32 位累加）能完整保留到显存，不再被 BF16 截断。
    cublasDataType_t cd_type = out_f32 ? CUDA_R_32F : ((sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP);
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, cd_type, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, cd_type, m, n, m));

    // 分批 strided batched GEMM
    if (batch_count) {
        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
    }

    // 设置 epilogue
    cublasLtEpilogue_t epilogue;
    if (has_gelu) {
        int64_t gelu_ld = m;
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        if (backward) {
            assert(!has_bias);
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if(has_bias){
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        cublasDataType_t bias_data_type = (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP;
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // 创建偏好设置，并指定最大工作区大小
    cublasLtMatmulPreference_t preference;
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                               preference, 1, &heuristic, &returnedResults));
    if (returnedResults == 0) {
        printf("未找到 cuBLASLt 算法：m：%d，n：%d，k：%d，bias：%d\n", n, m, k, has_bias);
        exit(EXIT_FAILURE);
    }

    // 设置是否累加（D += C）；注意算法选择并不考虑这一项
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;

    // 调用矩阵乘法
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

    // 清理描述符
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}

void matmul_forward_cublaslt(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL, int gelu_fusion=1) {
    if (gelu_fusion < 1 && pre_gelu) {
        // 不融合：先 GEMM 得到 GELU 前结果，再单独做 GELU
        matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false);
        gelu_forward(out, pre_gelu, B*T*OC, stream);
    } else {
        // 直接在 cuBLASLt 中融合 GELU
        matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false);
    }
}

// 与上面的区别只有一个：结果写成 float32（4 字节）而不是 floatX（BF16，2 字节）。
// 目前只有 lm_head 用它——logits 要拿去和 HuggingFace 的 float32 输出做数值对比，
// 如果在写回显存时就被截成 BF16，对比里会凭空多出一份"不是量化造成的"误差。
void matmul_forward_cublaslt_f32(float* out,
                                 floatX* inp, floatX* weight, floatX* bias,
                                 int B, int T, int C, int OC, cudaStream_t stream) {
    // 参数顺序与 matmul_cublaslt 一致：行数=OC（输出通道）、列数=B*T、内积维度=C
    matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false,
                    0, 0, 0, 0, false, NULL, false, true);
}