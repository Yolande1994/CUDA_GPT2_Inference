# ============================================================================
# transformers_gpt2_bench.py —— HuggingFace Transformers 官方实现性能基准
# ============================================================================
# 用 transformers 库直接加载 GPT-2 跑生成，作为手写引擎之外的第二个参考（另一个是 vLLM）。
# 支持 BF16/FP32 两档，注意力实现可在 SDPA 与 eager 之间切换。计时口径与其他基准一致：先生成一次预热，再跑若干次取中位数。
#
# 权重从统一的本地目录读取；目录不存在时自动下载。
# ============================================================================
import sys, time, os, statistics, argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 本地权重目录：脚本所在目录下的 assets/weights/hf
HF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "weights", "hf")

# 目录缺失时下载需要的文件，显式过滤掉 onnx 等用不到的格式
def ensure_hf_dir():

    if os.path.isdir(HF_DIR):
        return  # 本地已存在直接用
    from huggingface_hub import snapshot_download
    print(f"[权重] 未找到统一权重目录 {HF_DIR}，正在下载 openai-community/gpt2 ...")
    snapshot_download(repo_id="openai-community/gpt2", local_dir=HF_DIR,
                      allow_patterns=[  # 只下载需要的文件
                          "config.json", "generation_config.json", "merges.txt",
                          "model.safetensors", "tokenizer.json",
                          "tokenizer_config.json", "vocab.json",
                      ],
                      ignore_patterns=["onnx/*", "*.tflite", "*.onnx"],
                      endpoint="https://hf-mirror.com")  # 镜像站
    print(f"[权重] 下载完成：{HF_DIR}")

p = argparse.ArgumentParser()
p.add_argument("--genT", type=int, default=1000, help="生成 token 数（默认 1000）")
p.add_argument("--runs", type=int, default=6, help="正式计时次数（默认 6：丢弃第 1 次，取 5 次中位数）")
p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="精度（默认 bf16）")
# SDPA = PyTorch 2.0+ 的缩放点积注意力 API，具体后端（flash / memory-efficient / math）由 PyTorch 自动挑选；eager 是最朴素的实现
p.add_argument("--impl", choices=["sdpa", "eager"], default="sdpa",
               help="attention 实现（默认 sdpa = PyTorch 内置融合注意力算子，零依赖；eager 为基线）")
args = p.parse_args()

dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
impl = args.impl

print(f"加载：dtype={args.dtype}, attn_implementation={impl}")
ensure_hf_dir()
model = GPT2LMHeadModel.from_pretrained(
    HF_DIR, torch_dtype=dtype,
    attn_implementation=impl,
).to("cuda").eval()  # 加载到 GPU，推理模式
tokenizer = GPT2Tokenizer.from_pretrained(HF_DIR)
# 所有基准统一用结束符作为起始输入，保证起点一致
start = tokenizer("<|endoftext|>", return_tensors="pt").to("cuda")

# min/max 设成同一个数，每次生成长度严格相等，计时才可比较
GEN_KW = dict(
    min_new_tokens=args.genT - 1,
    max_new_tokens=args.genT - 1,
    do_sample=True, temperature=1.0, top_k=None,
    pad_token_id=model.config.eos_token_id,
)

# 预热：不计时，只让库完成首次算子后端选择与显存分配
with torch.no_grad():
    _ = model.generate(**start, **GEN_KW)

times = []
actuals = []
for i in range(args.runs):
    # 计时前后各同步一次，把 GPU 异步执行的时间完整包进墙钟区间
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():  # 推理关掉梯度
        out = model.generate(**start, **GEN_KW)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    actual = out.shape[1] - start.input_ids.shape[1]  # 实际生成 token 数
    times.append((t1 - t0) * 1000)
    actuals.append(actual)
    print(f"第 {i+1}/{args.runs} 次：{(t1-t0)*1000:.1f} ms（{(t1-t0)*1000/actual:.2f} ms/token，实际 {actual} token）")

# 丢弃第 1 次，结果为空时（极端情况）退回全量数据
stable = times[1:] or times
stable_actual = actuals[1:] or actuals
median_ms = statistics.median(stable)  # 中位数
median_actual = statistics.median(stable_actual)

print("\n[性能汇总]")
print(f"配置：dtype={args.dtype}, attn={impl}, 生成 {args.genT-1} token（min=max 等长）")
print(f"  中位数（丢第1次）：{median_ms:.1f} ms（{median_ms/median_actual:.2f} ms/token，实际 {median_actual} token）")
print(f"  最小：{min(stable):.1f} ms / 最大：{max(stable):.1f} ms")
print("---")
print(tokenizer.decode(out[0]))
print("---")
