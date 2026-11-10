# Inference_gpt2 目录导航

本目录是整个推理引擎的核心组成区，共 15 个头文件，GPT-2 推理需要的组件都在这：模型数据结构、显存管理、算子、矩阵乘、注意力、量化、分词和采样。

<br>

## 分组预览

#### 基础设施层

| 文件 | 职责 |
| --- | --- |
| [`utils.h`](utils.h) | 最底层工具：封装 `fopenCheck`、`freadCheck` 等 C 标准库检查宏，出错就打印并退出 |
| [`common.h`](common.h) | CUDA 公共工具库：通用宏、精度切换（FP32/BF16）、cuBLAS 句柄、规约和向量化访存，几乎所有算子都依赖它 |
| [`gpt2_model.h`](gpt2_model.h) | 模型蓝图：定义配置、权重指针、激活指针，以及「一次 `cudaMalloc` 整块切分」的显存管理和 checkpoint 加载 |

#### 矩阵乘

| 文件 | 职责 |
| --- | --- |
| [`GEMV.cuh`](GEMV.cuh) | 全精度手写矩阵向量乘（General Matrix-Vector multiplication）：`GEMV6` 纯乘 + `GEMV6_fused` 融合 bias/GELU，全程 FP32 累加 |
| [`GEMV_w8.cuh`](GEMV_w8.cuh) | 8 位权重版 GEMV：INT8/FP8 由编译宏二选一，一个 warp 处理两行（ILP） |
| [`matmul.cuh`](matmul.cuh) | 官方库路线：封装 cuBLASLt 矩阵乘，在收尾阶段（epilogue）融合 bias、GELU，手写 kernel 的对照 |

#### 注意力机制

| 文件 | 职责 |
| --- | --- |
| [`attention.cuh`](attention.cuh) | 无 KV 缓存的完整自注意力（self-attention），一次性处理整段序列，基础版程序和预填充阶段使用 |
| [`attention_kvcache.cuh`](attention_kvcache.cuh) | 带 KV 缓存的解码注意力：历史 K/V 存下来复用，含单 block、Flash Decoding、grid.sync 三种并行策略，按序列长度切换 |
| [`attention_kvcache_w8.cuh`](attention_kvcache_w8.cuh) | 8 位 KV 缓存版：缓存里的 K/V 压成 1 字节、体积减半，逐行动态量化，默认关闭 |

#### 嵌入与逐元素算子

| 文件 | 职责 |
| --- | --- |
| [`encoder.cuh`](encoder.cuh) | 嵌入层（embedding）：词嵌入（WTE）加位置嵌入（WPE），把 token 编号变成模型内部向量，提供预填充和解码两个入口 |
| [`layernorm.cuh`](layernorm.cuh) | 层归一化（LayerNorm）与残差（Residual），二者常融合在一个 kernel 里算 |
| [`gelu.cuh`](gelu.cuh) | GELU 激活函数的 tanh 近似版本，公式和实现都在里面 |

#### 采样、分词与调试

| 文件 | 职责 |
| --- | --- |
| [`sampler_gpu.cuh`](sampler_gpu.cuh) | 纯 GPU 采样器：softmax 加 Philox 随机数，全程留在设备、只回传最终 token，优化版使用 |
| [`tokenizer_sampler.h`](tokenizer_sampler.h) | 分词器（tokenizer，负责把 token 转字符串）和 CPU 采样器（xorshift 随机），基础版使用 |
| [`dump_utils.h`](dump_utils.h) | 调试转储：把中间 logits、token 前缀、困惑度数据落盘，供 `verify_precision.py` 离线对拍 |

<br>

## 两个主程序各用了哪些头文件

| 头文件 | 基础版 `inference_gpt2.cu` | 优化版 `inference_gpt2_opt.cu` |
| --- | :---: | :---: |
| `common.h` / `utils.h` | 是 | 是 |
| `encoder.cuh` / `layernorm.cuh` / `matmul.cuh` | 是 | 是 |
| `tokenizer_sampler.h` | 是 | 是 |
| `attention.cuh`（完整注意力） | 是 | 否 |
| `gpt2_model.h`（显存蓝图） | 否 | 是 |
| `GEMV.cuh` / `GEMV_w8.cuh`（手写 GEMV） | 否 | 是 |
| `attention_kvcache.cuh` / `..._w8.cuh`（KV 缓存） | 否 | 是 |
| `sampler_gpu.cuh`（GPU 采样） | 否 | 是 |
| `dump_utils.h`（调试转储） | 否 | 是 |

基础版只保留最直观的实现（全精度、无图、无 KV 缓存、无量化），是 GPT-2 前向的骨架；

优化版在此之上叠加了性能工程，是和官方库对比的成品。
