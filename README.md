<h1 align="center">CUDA GPT-2 推理引擎</h1>

**纯 CUDA 从零实现的 GPT-2（124M）完整推理：**

不调用任何高层深度学习框架，无任何封装，涵盖 Transformer 的推理底层算子、量化（INT8/FP8）、KV Cache、CUDA Graph、自回归生成主循环。  
在同口径单请求场景下，平均 token 延迟跑赢 cuBLASLt、HuggingFace transformers 与 vLLM。  

**本仓库既是一个能跑的推理引擎，也是一套 CUDA 算子优化教程：**

13 个独立算子各自从最朴素的实现开始，一版一版地进行优化，每一步都配了 Nsight Compute（NVIDIA 官方性能剖析工具，简称 NCU）的实测数据，并写 README 详细记录优化过程。

<br>

## 项目说明

- **纯底层实现，无调库拼装**：从权重参数、激活指针的显存布局，到每个 kernel 的线程映射、计算流程等等，全在代码里清晰可见。
- **一条流水线可切换四种精度**：FP32 / BF16 / INT8 / FP8，外加一个自研量化器（`quantize.cu`），权重体积可从 475 MB 压到 120 MB。
- **两套主程序分工明确**：`inference_gpt2` 是将各算子直白的拼接、易于理解的基础版本，`inference_gpt2_opt` 是带 CUDA Graph、算子融合、自研 GEMV、8 位量化的优化版本。
- **工程操作完整**：权重下载与导出、量化、性能测试、精度验证、二进制格式校验，都有对应脚本，一键运行完成。
- **性能数据可复现**：在我的测量口径里（见下方），INT8 达到 **0.77 ms/token**，同口径领先 vLLM 的性能约 **1.5** 倍。

<br>

## 性能表现

### 先说测量口径

- 场景：batch=1、单请求、单卡，自回归生成，统计每个 token 的平均耗时（ms/token）。
- 生成长度：1000 个 token。
- 统计方法：每个配置连跑 6 次，丢掉第 1 次预热，取后 5 次的中位数。
- 手写 vs cuBLASLt 的端到端用的是同一份权重，只换矩阵乘实现，保证对照公平。

### 与主流推理框架的对比（ms/token）

| 引擎 | FP32 | BF16 | FP8 | INT8 |
| --- | --- | --- | --- | --- |
| 本项目 | 1.82 | 1.11 | 0.77 | 0.77 |
| vLLM | 2.043 | 1.347 | 1.159 | 1.156 |
| transformers（SDPA） | — | 5.45 | — | — |

- 对比 vLLM 的领先倍数：FP32 约 **1.12** 倍、BF16 约 **1.21** 倍、FP8 约 **1.5** 倍、INT8 约 **1.5** 倍。
- 对比 transformers BF16：约 **7** 倍。

### 只看优化版性能（ms/token）

| 精度 | 手写 GEMV | 调用 cuBLASLt | 相差幅度 |
| --- | --- | --- | --- |
| FP32 | 1.82 | 1.99 | 8.5% |
| BF16 | 1.11 | 1.27 | 12.6% |
| INT8（weight-only） | 0.77 | — | — |
| FP8（weight-only） | 0.77 | — | — |
| INT8 + 8位KV缓存 | 0.77 | — | — |
| FP8 + 8位KV缓存 | 0.77 | — | — |

> 基础版（各算子的直接拼接）的耗时是 FP32 15.85、BF16 8.64 ms/token。

### 为何能赢，边界在哪

解码是**纯访存瓶颈**：每生成一个 token 都要把全部权重从显存读一遍，而计算量很小，GPU 计算单元多数时间是闲着等数据的。  
手写版针对此特性做了算子融合、自研 GEMV 和 CUDA Graph 整步重放，在算子上领先 cuBLASLt（多数尺寸下），在架构上省掉通用框架的任务调度、分页 KV 管理和 kernel 启动开销。

但是：vLLM 的强项是多请求并发，请求多了它才能靠 PagedAttention 把显存和算力拉高，此时吞吐量会超过单请求实现。  
本项目没有做并发调度（留待以后），暂时不打算在这个维度上和它比，而是看看单请求时能将其甩开多远。

<br>

## 目录结构

```
.
├── inference_gpt2.cu             # 基础版主程序
├── inference_gpt2_opt.cu         # 优化版主程序
├── quantize.cu                   # 量化器：BF16 → INT8/FP8
├── Makefile                      # 全部编译/运行指令，复制它，运行所有组合
├── common.h                      # 独立算子共用的工具头
├── export_weights.py             # 下载 HuggingFace 权重 → 导出扁平 bin → 量化 → vLLM 检查点
├── verify_precision.py           # 对比 HuggingFace 的精度验证（cosine/KL/Top-k/PPL）
├── vllm_gpt2_bench.py            # vLLM 单请求测速基准
├── transformers_gpt2_bench.py    # transformers 单请求测速基准
├── check_fp8_bin.py              # 8 位二进制文件格式校验
├── Inference_gpt2/               # 推理引擎核心头文件
│   ├── README.md                 #   本目录导航
│   ├── gpt2_model.h              #   模型配置、权重/激活内存布局、checkpoint 加载
│   ├── GEMV.cuh                  #   全精度手写 GEMV（含融合 bias/GELU 版）
│   ├── GEMV_w8.cuh               #   8 位 GEMV（INT8/FP8 择一编译）
│   ├── attention.cuh             #   预填充阶段注意力
│   ├── attention_kvcache.cuh     #   解码阶段 KV 注意力
│   ├── attention_kvcache_w8.cuh  # 8 位 KV 缓存注意力
│   ├── layernorm.cuh / gelu.cuh / encoder.cuh / matmul.cuh
│   ├── sampler_gpu.cuh           #   设备端采样（Philox，全程不回传中间结果）
│   ├── tokenizer_sampler.h       #   分词器 + CPU 采样（基础版用）
│   └── common.h / utils.h / dump_utils.h
├── gemv/ attention/ attention_kvcache/ fused/ gelu/ layernorm/
├── softmax/ trimat/ residual/ other_kernels/   # 13 个独立算子优化实验
└── images/                       # 各目录 README 引用的性能截图、示意图
```

<br>

## 核心实现

### 1. 首要瓶颈

GPT-2 参数维度小，且逐 token 自回归生成，每算一步都要把全部权重从显存读一遍，权重搬运占了每步耗时的绝大部分。  
优化的核心，重点关注是否减少了显存读取量，而非看浮点运算次数。这是多数设计的出发点。

### 2. 显存管理

核心目标是：少浪费空间，让数据读得更快。

- **整块切分，减少碎片**：用一次 `cudaMalloc` 申请一整块连续显存，切分出 16 组权重、22 组激活（基础版 21 组），激活按 16 字节对齐摆放。连续布局既减少显存碎片，也更利于 GPU 缓存预取。
- **流式搬运，并行重叠**：权重按 32 MB 分块上传，配合双缓冲和 WriteCombined 页锁定内存，让磁盘读取与显存上传重叠。
- **分层精度**：中间张量随编译档位切换精度（BF16 档即用 BF16），最终词概率 logits 固定 FP32。

### 3. GEMV

这是解码路上出现最多的算子：

逐 token 生成时，所有全连接层都从 “矩阵乘矩阵”（GEMM）退化成 “矩阵乘向量”（GEMV）。  
拿一次显存（1 个权重），只换两次浮点计算（1 次 FMA），典型的带宽瓶颈，优化重点是拉满显存带宽。

本项目部署版 `GEMV6` 的关键设计是：
- 一个 warp（32 线程）负责输出矩阵的一行，块内 512 线程一次处理 16 行。
- 用双 128-bit 向量加载（FP32 走 `float4`，BF16 走 `uint4`），把访存请求做大、合并。
- 用软件流水，将访存和计算延迟重叠。
- 中间计算全程 FP32 累加，warp 内用 5 轮 shuffle 做归约。
- 融合版 `GEMV6_fused` 把 bias、tanh 近似 GELU 并到同一个 kernel，省掉中间结果的来回读写，顺手输出 GELU 前的线性值供残差使用。

> 热点尺寸已跑到硬件上限的 94%（363/384 GB/s，FP32）

8 位 `GEMV6_w8x2` 版让一个 warp 处理两行，两行复用同一份输入向量，把读 x 的带宽摊薄到两行，靠指令级并行（ILP）让更多内存请求同时在途。

### 4. KV Cache 与注意力

解码阶段每步只处理一个新 token，但要读到当前位置之前的全部历史 K/V。工程里实现了三种策略，根据序列长度动态切换选择：

- **fused_single（短序列）**：单个 block 用共享内存装下全部注意力分数（上限 1024），实现最简单、无额外归约。
- **split + merge（长序列，Flash Decoding）**：把序列切成多段并行（本机实测8段最好），每段产出各自的加权和、最大值、指数和，再用在线 softmax 合并起来。用更多线程块一起分担，避免长序列全压在少数 block 上拖慢计算速度。
- **cooperative（grid.sync）**：在上版的基础上用 `cudaLaunchCooperativeKernel` 让整个网格同步，合并两独立 kernel 为一个，省一次启动开销，只能在非 Graph 的 batch=1 下用。

### 5. 量化器

`quant_rows` 实现逐行对称量化：一个 block 负责一行，在共享内存里做树形归约求出该行绝对值最大值 amax，再算 `scale = amax / 满量程`（INT8 满量程是 127，FP8 E4M3 是 448）。

只量化 5 个关键 Linear 权重，其余参数保持 BF16。

### 6. CUDA Graph

正常推理一个 token 要启动几十个 kernel，每次启动都有几微秒固定开销。

这里把整个前向推理捕获成 small/large 两张图（根据序列的长度切换），生成时直接重放，把大量启动开销消掉。

### 7. 权重绑定与采样

权重绑定：最终词表投影直接复用词嵌入表，省掉一份参数。

设备端采样 `gpu_sample_softmax`：采样计算全程在 GPU 内执行，只把最终选出的 token 传回 CPU 主机，避免把完整的 logits 大数组传回 CPU 带来的传输开销。

<br>

## 独立算子的优化导览

所有 `.cu` 都能独立编译运行，每个文件的头部注释里写好了编译命令和版本说明；

子目录 README 是完整优化记录与教程，大致结构是：算子简介、张量布局/显存开销、性能总览、逐版本演进（含 NCU 剖析、为什么慢/快）；

> 全部内容涵盖 CUDA 编程常见优化思想和手法，读完已然入门。

| 目录/文件 | 版本数量 | 关键知识点 |
| --- | --- | --- |
| `gemv/GEMV.cu` | 8 | 矩阵向量乘，从每线程一行，到 warp 协作、128-bit 加载、软件流水、ILP，含 cuBLAS/cuBLASLt 对照 |
| `attention/attention.cu` | 11 | 多头因果注意力，含 FlashAttention v1/2 演进、warp 级优化、cuBLAS 混合，以及 cuDNN 融合版 |
| `attention_kvcache/attention_kvcache.cu` | 6 | 带 KV 缓存的解码注意力：朴素、融合、Flash Decoding、grid.sync、在线 + cp.async |
| `fused/fused.cu` | 6 | 残差 + LayerNorm 融合，含共享内存、向量化、流式访存、锯齿循环与生产级自适应 |
| `layernorm/layernorm.cu` | 6 | 层归一化，单趟统计、warp/块级归约与向量化 |
| `softmax/softmax.cu` | 9 | 常规 softmax 与在线 softmax，逐步消除对全局中间缓存的依赖 |
| `trimat/trimat.cu` | 5 | 三角矩阵乘，融合因果掩码直接跳过上三角，含 cuBLAS 对照 |
| `residual/residual.cu` | 2 | 残差连接的朴素与向量化写法 |
| `gelu/gelu.cu` | 2 | GELU 激活及其近似实现 |
| `other_kernels/matmul.cu` | 5 | 矩阵乘：朴素 GEMM、基础手写分块，再到 cuBLAS/cuBLASLt |
| `other_kernels/encoder.cu` | 3 | 词嵌入查找 |
| `other_kernels/classifier_fused.cu` | 3 | Softmax + 交叉熵 + logit 梯度融合 |
| `other_kernels/permute.cu` | — | 四维张量维度置换 |


<br>

## 精度验证

`verify_precision.py` 会把手写 CUDA 推理的输出，和 HuggingFace 原版高精度 eager 实现逐项对照，输出余弦相似度、平均偏差、KL 散度、Top-1/Top-5 匹配率和困惑度（PPL）。

下面是实测数据（INT8 和 FP8 都是 weight-only，只压权重，激活保持 BF16）：

| 验证层级 | 指标 | INT8 | FP8 |
| --- | --- | --- | --- |
| 格式理论上限 | 单值最大相对误差 | 理论上限 0.5/127，实测 0.39% | 理论上限 6.25%，实测 1.72% |
| 权重本身误差（全部权重逐个检查） | 最大相对误差 | 0.3934% | 1.7155% |
| 整条序列、逐个预测位置 | KL 平均值 | 0.0907 | 1.1839 |
| 整条序列、逐个预测位置 | Top-1 一致率 | 79.6% | 48.8% |
| CUDA 端到端（首步输出） | 余弦相似度 | 0.999994 | 0.999938 |
| CUDA 端到端（首步输出） | KL | 0.0148 | 0.1098 |

GPT2 weight 的数值分布很稳定，在 “预测下个 token 概率分布是否一致” 这个决定生成质量的口径上，INT8 优于 FP8；FP8 唯一占优的是个别极值位置的绝对偏差 —— 它有指数位，天生擅长表示大数。


<br>

## 测试环境

| 条目 | 数据 |
| --- | --- |
| GPU | RTX 5060 Laptop（Blackwell，`sm_120`） |
| SM 数 / 最大时钟 | 26 / 3090 MHz |
| 显存 / 位宽 / 时钟 | 8151 MiB / 128 bit / 12001 MHz，算出理论带宽 384 GB/s |
| L2 | 32 MB |
| 每 SM 共享内存 / 寄存器 / 线程 | 100 KB（block 默认 48 KB，可申请到 99 KB）/ 65536 / 1536 |
| 每 block 线程上限 / 每 SM block 上限 | 1024 / 24 |

65536÷1536≈42，单线程超过 42 个寄存器会掉占用率；

L2 占 32 MB，小于它的数据量测出来的是 L2 带宽、非显存带宽；

26 个 SM，grid < 26 时一定有 SM 闲着。并行度满足率 = block 总数 ÷ (SM 数 × 每 SM 可驻留 block 数)

环境：Ubuntu 24.04.4（WSL2）+ 驱动 616.92 + CUDA 13.4 + Nsight Compute 2026.3.0。


<br>

## 快速开始

### 环境要求

- NVIDIA GPU + CUDA 工具链（开发环境用的是 CUDA 13.x，`nvcc` 可在 PATH 中找到）。
- 一份带 `torch` 的 Python 3 环境，用于跑权重导出和精度脚本。
- GPU 架构由 Makefile 自动探测，也可显式指定：`make GPU_COMPUTE_CAPABILITY=120`。

#### 第一步：准备权重

权重文件由一条命令全自动完成「下载 HuggingFace 原始权重 → 导出扁平 bin → 量化」，产物统一落在 `assets/weights/`（同目录下）：

```bash
make weights              # 默认量化出 INT8
make weights QUANT=fp8    # 同一套流程，量化成 FP8 E4M3
```

直接运行 `python export_weights.py` 也是同样的效果。它会依次产出 FP32、BF16、INT8、FP8 四档权重、分词器文件，以及供 vLLM 对照用的检查点。

#### 第二步：编译

三个互相独立的开关：`PRECISION`（精度，默认 INT8）、`GEMV`（矩阵乘实现，默认手写）、`KV_W8`（是否量化 KV 缓存，默认=0 不量化）。

```bash
make                          # 默认：INT8 + 手写 GEMV
make PRECISION=FP8            # FP8 + 手写 GEMV
make PRECISION=BF16           # BF16 + 手写 GEMV
make PRECISION=BF16 GEMV=cuBLASLt   # 同权重，换 cuBLASLt 官方库（做对照）
make PRECISION=FP32           # FP32 全精度
make PRECISION=INT8 KV_W8=1   # 开启 8 位 KV 缓存
```

#### 第三步：运行

编译和运行命令成对出现（运行时的 `--int8` / `--fp8` 档位要和编译精度匹配，不匹配会报错）：

```bash
# INT8（默认推荐）
./inference_gpt2_opt assets/weights/gpt2_124M_int8.bin assets/weights/gpt2_tokenizer.bin 1000 --graph --int8 --bench 6

# FP8
./inference_gpt2_opt assets/weights/gpt2_124M_fp8.bin assets/weights/gpt2_tokenizer.bin 1000 --graph --fp8 --bench 6

# BF16
./inference_gpt2_opt assets/weights/gpt2_124M_bf16.bin assets/weights/gpt2_tokenizer.bin 1000 --graph --bench 6
```

assets/weights：权重路径  |  gpt2_tokenizer.bin：分词器路径  |  1000：生成 token 数

常用开关：`--graph`（启用 CUDA Graph）、`--bench N`（跑 N 轮取中位数）、`-b/--batch`。  
其他开关：`--ppl N`（困惑度）、`--dump-steps` / `--dump-prefix`（导出中间张量做对拍）。

基础版本的命令：

```bash
make PRECISION=BF16
./inference_gpt2 assets/weights/gpt2_124M_bf16.bin assets/weights/gpt2_tokenizer.bin 1000 --bench 6
```

#### 第四步：验证精度

直接运行 verify_precision.py，或指定：

```bash
python verify_precision.py --dtype int8   # 可选 bf16/fp32/fp8/int8
```

<br>

## 实测过但不采用的方向（记录结论）

| 方向 | 不采用的原因 | 实测影响 |
| --- | --- | --- |
| MLP 三层融合为一个 kernel | 中间数据仅约 6 KB，已留在 L2、不走显存，没东西可省；引入同步开销 | ± 0.5%（噪声范围） |
| LayerNorm + QKV 融合 | 同理，中间数据太小，融合开销大于收益 | 慢 0.8% |
| 投机解码 | 接受率约 0.82、验证成本约 1.11，收益 0.74 < 1；用计算换带宽，方向与带宽墙相反 | 数学上必亏 |
| 持久化 kernel 减少启动 | 单次 kernel 启动仅 2~3 微秒，可省的上限太小 | 无收益 |
| cp.async 异步搬运 | 每行仅 768 个元素，一次就能搬完，没有流水线可重叠，反而引入管理开销 | 慢 17% |
| 单线程处理多行 GEMV | 寄存器容量不足，数据溢出到慢速内存 | 慢 5.5% |

<br>

## 致谢与许可

本项目在学习过程中参考了 Andrej Karpathy 的 [llm.c](https://github.com/karpathy/llm.c) 中的 `11m.c`（MIT License）的基础版本。

所有优化kernel、优化版本、性能数据、README文档等均为本人独立编写。

底层代码沿用 MIT License，详见 [LICENSE](./LICENSE)。
