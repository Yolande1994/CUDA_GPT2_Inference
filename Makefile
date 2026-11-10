# ============================================================================
# GPT-2 推理构建 Makefile：一个文件覆盖所有编译和运行指令
# ============================================================================
#
# 【如何选配置】三个开关，各自独立：
#   PRECISION: 模型精度（决定加载哪一份权重）
#     - INT8（默认）  → 加载 gpt2_124M_int8.bin + 同目录 *_int8_scales.bin（权重减半，推荐）
#     - FP8         → 加载 gpt2_124M_fp8.bin + 同目录 *_fp8_scales.bin（同样权重减半，作为可选格式）
#     - BF16        → 加载 gpt2_124M_bf16.bin（不量化，全精度对照）
#     - FP32        → 加载 gpt2_124M.bin（不量化，全精度对照）
#
#   GEMV: 矩阵乘实现（只影响计算方式，不影响权重文件）
#     - CUSTOM（默认）→ 手写 GEMV 内核（最快）
#     - cuBLASLt     → NVIDIA 官方库
#
#   KV_W8=1: 把 KV 缓存也压成 8 位（可选，PRECISION=INT8 或 FP8 时生效）
#
# 【编译 + 运行，成对使用】（先编译，再按对应行运行）
#   1. 默认：INT8 + 手写 GEMV（推荐，直接 make 即可）
#        make
#        ./inference_gpt2_opt assets/weights/gpt2_124M_int8.bin assets/weights/gpt2_tokenizer.bin 1000 --graph --int8 --bench 6
#
#   2. 可选格式：FP8 + 手写 GEMV
#        make PRECISION=FP8
#        ./inference_gpt2_opt assets/weights/gpt2_124M_fp8.bin assets/weights/gpt2_tokenizer.bin 1000 --graph --fp8 --bench 6
#
#   3. 全精度对照：BF16 + 手写 GEMV
#        make PRECISION=BF16
#        ./inference_gpt2_opt assets/weights/gpt2_124M_bf16.bin assets/weights/gpt2_tokenizer.bin 1000 --graph --bench 6
#
#   4. 官方库对照：BF16 + cuBLASLt（与 3 同权重，只换了矩阵乘实现）
#        make PRECISION=BF16 GEMV=cuBLASLt
#        ./inference_gpt2_opt assets/weights/gpt2_124M_bf16.bin assets/weights/gpt2_tokenizer.bin 1000 --graph --bench 6
#
#   5. 8 位 KV 缓存（可选功能，默认关）：KV Cache 从 36 MiB 压到 18 MiB
#        格式跟随 PRECISION：INT8 编出 INT8 KV，FP8 编出 FP8 KV（同一份内核，只有编解码不同）
#        make PRECISION=INT8 KV_W8=1        # 推荐：整数 8 位 KV，精度高于 FP8 KV，速度基本相同
#        ./inference_gpt2_opt assets/weights/gpt2_124M_int8.bin assets/weights/gpt2_tokenizer.bin 1000 --graph --int8 --bench 6
#        make PRECISION=FP8 KV_W8=1         # 可选：FP8 格式 KV，用于与 INT8 KV 对照
#        ./inference_gpt2_opt assets/weights/gpt2_124M_fp8.bin assets/weights/gpt2_tokenizer.bin 1000 --graph --fp8 --bench 6
#
# 【权重准备】一键下载 + 导出 + 量化，在本目录下创建 assets/weights/ 文件夹并落在其中
#        make weights              # 等价于 python export_weights.py，并量化成 INT8（默认）
#        make weights QUANT=fp8    # 同一套流程，但量化成 FP8 E4M3
#
# 【性能跑分】--bench N = 共跑 N 次，丢弃第 1 次，取 N-1 次中位数
#        ./inference_gpt2_opt <权重> <分词器> 1000 --graph [--int8/--fp8] --bench 6
#
# 【精度验证】对比 HuggingFace 官方权重，详见 verify_precision.py
#
# 【其他】
#   - 基础推理版（无 CUDA Graph 等优化）：./inference_gpt2 <权重> <分词器> 1000
#       例：./inference_gpt2 assets/weights/gpt2_124M_bf16.bin assets/weights/gpt2_tokenizer.bin 1000 --bench 6
#   - GPU 架构自动探测，也可显式指定： make GPU_COMPUTE_CAPABILITY=120
# ============================================================================


# ----------------------------------------------------------------------------
# 1. 配置开关与条件编译
# ----------------------------------------------------------------------------
# 默认精度：INT8（8 位整数权重，又快又准）；可切换 FP8 / BF16 / FP32
PRECISION ?= INT8
VALID_PRECISIONS := FP32 BF16 FP8 INT8

# 默认矩阵乘实现：手写 GEMV；可切换 cuBLASLt（官方库对照）
GEMV ?= CUSTOM
VALID_GEMVS := CUSTOM cuBLASLt

# 校验开关取值
ifeq ($(filter $(PRECISION),$(VALID_PRECISIONS)),)
  $(error 无效精度 $(PRECISION)，有效值为 $(VALID_PRECISIONS))
endif
ifeq ($(filter $(GEMV),$(VALID_GEMVS)),)
  $(error 无效 GEMV 实现 $(GEMV)，有效值为 $(VALID_GEMVS))
endif

# 精度宏：FP32 用 FP32；8 位（INT8/FP8）与 BF16 都用 BF16 激活（8 位只量化权重）
ifeq ($(PRECISION), FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION), INT8)
  # INT8：启用 8 位权重骨架 + INT8 格式
  PFLAGS = -DENABLE_BF16 -DUSE_W8 -DUSE_INT8
else ifeq ($(PRECISION), FP8)
  # FP8：启用 8 位权重骨架 + FP8 格式
  PFLAGS = -DENABLE_BF16 -DUSE_W8 -DUSE_FP8
else
  # BF16：不量化
  PFLAGS = -DENABLE_BF16
endif

# GEMV 开关：cuBLASLt 时定义 USE_CUBLASLT，回退到 NVIDIA 官方矩阵乘
ifeq ($(GEMV), cuBLASLt)
  PFLAGS += -DUSE_CUBLASLT
endif

# KV Cache 8 位量化（默认不开启）。格式跟随 PRECISION：
#   PRECISION=INT8 + KV_W8=1 → INT8 KV（整数，精度高、sm_75+ 可跑）
#   PRECISION=FP8  + KV_W8=1 → FP8 KV（E4M3；硬件转换指令 sm_89+ 才有，更老的卡走软件模拟）
# 默认 0 = 不开：KV 缓存保持 BF16
KV_W8 ?= 0
ifeq ($(KV_W8), 1)
  ifeq ($(filter INT8 FP8,$(PRECISION)),)
    $(error KV_W8=1 需要 PRECISION=INT8 或 FP8（BF16/FP32 没有 8 位权重，8 位 KV 无从谈起）)
  endif
  PFLAGS += -DUSE_KV_W8
endif


# ----------------------------------------------------------------------------
# 2. 工具链与 GPU 架构自动检测
# ----------------------------------------------------------------------------
# 查找 nvcc 路径
ifeq ($(OS), Windows_NT)
  NVCC := $(shell where nvcc 2>nul)
else
  NVCC := $(shell which nvcc 2>/dev/null)
endif
OUTPUT_FILE = -o $@

# 自动探测 GPU 计算能力（如 12.0 -> 120），生成对应 PTX/SM 选项。
# 多卡时取最低算力，保证编出的二进制在所有卡上都能跑。
# 探测失败（如没有 nvidia-smi 的环境）时回退到 120；也可显式指定 make GPU_COMPUTE_CAPABILITY=xx
GPU_COMPUTE_CAPABILITY ?= $(strip $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/\.//g' | sort -n | head -n 1))
ifeq ($(GPU_COMPUTE_CAPABILITY),)
  GPU_COMPUTE_CAPABILITY := 120
endif
ifneq ($(GPU_COMPUTE_CAPABILITY),)
  ARCH_FLAGS = --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
endif


# ----------------------------------------------------------------------------
# 3. NVCC 编译与链接选项
# ----------------------------------------------------------------------------
# --use_fast_math：允许浮点近似（如快速除法、近似超越函数），换吞吐
# -t=0：让 nvcc 并行编译多个目标；-O3：最高优化
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O3 $(ARCH_FLAGS)
NVCC_LDFLAGS = -lcublas -lcublasLt
ifneq ($(OS), Windows_NT)
  NVCC_LDFLAGS += -lnvidia-ml
endif


# ----------------------------------------------------------------------------
# 4. 构建目标
# ----------------------------------------------------------------------------
TARGETS = inference_gpt2 inference_gpt2_opt

$(info ---------------------------------------------)
ifeq ($(NVCC),)
  $(info x 未找到 nvcc，无法构建 GPU 目标)
else
  $(info o nvcc 就绪：精度=$(PRECISION)，GEMV=$(GEMV)，gpu_cc=$(GPU_COMPUTE_CAPABILITY)，KV_W8=$(KV_W8))
endif
$(info ---------------------------------------------)

# 编译参数指纹：PRECISION / GEMV / KV_W8 / GPU 架构任一变化时自动强制重编。
BUILD_KEY := $(PRECISION)|$(GEMV)|$(KV_W8)|$(GPU_COMPUTE_CAPABILITY)
# 构建元数据，放 build/ 下（已被 .gitignore 忽略）
BUILD_FINGERPRINT := build/.build_fingerprint

.PHONY: fingerprint
fingerprint:
	@mkdir -p build; if [ "$$(cat $(BUILD_FINGERPRINT) 2>/dev/null)" != "$(BUILD_KEY)" ]; then \
		echo "* 编译参数变化 -> $(BUILD_KEY)，强制重新编译"; \
		rm -f $(TARGETS) quantize; \
		echo "$(BUILD_KEY)" > $(BUILD_FINGERPRINT); \
	fi

# 默认目标必须显式指定为 all。GNU make 会把“第一个规则”当默认目标，
# 而第一个规则是 .PHONY 的 fingerprint，不指定的话无参数 make 只会跑指纹检查、不编译任何东西。
.DEFAULT_GOAL := all

all: fingerprint $(TARGETS)

# 目标 1：基础推理版（无 CUDA Graph）
inference_gpt2: inference_gpt2.cu Inference_gpt2
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) inference_gpt2.cu $(NVCC_LDFLAGS) $(OUTPUT_FILE)

# 目标 2：优化成品版（CUDA Graph + 手写 GEMV + 8 位权重 INT8/FP8 开关）
inference_gpt2_opt: inference_gpt2_opt.cu Inference_gpt2/GEMV.cuh Inference_gpt2/GEMV_w8.cuh \
                     Inference_gpt2/attention_kvcache.cuh Inference_gpt2/attention_kvcache_w8.cuh Inference_gpt2
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) inference_gpt2_opt.cu $(NVCC_LDFLAGS) $(OUTPUT_FILE)

# 目标 3：8 位量化器（BF16 bin → INT8 或 FP8 per-channel bin + scales）
#   一份源码同时支持两种格式，运行时用 --format 选择；默认 INT8。
quantize: quantize.cu
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) quantize.cu $(NVCC_LDFLAGS) $(OUTPUT_FILE)


# ----------------------------------------------------------------------------
# 5. 辅助指令
# ----------------------------------------------------------------------------
# 一键权重：下载 HF 原格式 + 导出 BF16 + 量化 8 位（等价 python export_weights.py）
#   QUANT 决定量化格式：int8（默认）/ fp8；产物文件名跟着格式走
QUANT ?= int8
weights: assets quantize
	./quantize assets/weights/gpt2_124M_bf16.bin \
	           assets/weights/gpt2_124M_$(QUANT).bin \
	           assets/weights/gpt2_124M_$(QUANT)_scales.bin \
	           --format $(QUANT)

assets:
	python export_weights.py

clean:
	rm -f $(TARGETS) quantize $(BUILD_FINGERPRINT)

.PHONY: all clean assets weights quantize
