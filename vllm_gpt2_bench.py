# ============================================================================
# vllm_gpt2_bench.py —— vLLM 官方引擎性能基准（手写引擎的主要对照对象）
# ============================================================================
# 用 vLLM 加载同一个 GPT-2，逐档输出单请求延迟：FP32、BF16，以及加载预量化 checkpoint 的 FP8/INT8 W8A16 档；
# 还可通过 --kv-cache-dtype 切换 KV 缓存精度。
# 计时口径与手写引擎一致：预热 1 次，其余次数取中位数。
# ============================================================================
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"       # 压掉无关日志，只保留基准输出
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# 按优先级找可用的 CUDA 安装目录：环境变量 → torch 自带路径 → PATH 里的 nvcc
def _resolve_cuda_home() -> str | None:
    import shutil
    for var in ("CUDA_HOME", "CUDA_PATH"):
        home = os.environ.get(var)
        if home and os.path.isfile(os.path.join(home, "bin", "nvcc")):
            return home  # 环境变量命中
    try:
        from torch.utils.cpp_extension import CUDA_HOME as _torch_cuda
        if _torch_cuda and os.path.isfile(os.path.join(_torch_cuda, "bin", "nvcc")):
            return _torch_cuda  # torch 自带路径命中
    except ImportError:
        pass

    nvcc = shutil.which("nvcc")
    if nvcc:
        # nvcc 在 <root>/bin 下，取上两级才是 CUDA 根目录
        return os.path.dirname(os.path.dirname(nvcc))
    return None

# vLLM 部分后端需要 JIT 编译，ninja 缺失时把当前解释器目录加进 PATH
def _ensure_ninja() -> None:
    import shutil
    import sys
    if shutil.which("ninja"):
        return
    bindir = os.path.dirname(sys.executable)
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")

_cuda_home = _resolve_cuda_home()
if _cuda_home:
    os.environ["CUDA_HOME"] = _cuda_home

_ensure_ninja()

import statistics
import time

from vllm import LLM, SamplingParams

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HF_DIR = os.path.join(BASE_DIR, "assets", "weights", "hf")

# 两份预量化 checkpoint：由 export_weights.py 生成，vLLM 直接加载无需在线量化
CT_CKPT_DIR = os.path.join(BASE_DIR, "assets", "weights", "ct_fp8")
CT_INT8_DIR = os.path.join(BASE_DIR, "assets", "weights", "ct_int8")

# 本地权重目录缺失时自动下载，过滤掉用不到的格式
def ensure_hf_dir():
    if os.path.isdir(HF_DIR):
        return
    from huggingface_hub import snapshot_download
    print(f"[权重] 未找到统一权重目录 {HF_DIR}，正在下载 openai-community/gpt2 ...")
    snapshot_download(repo_id="openai-community/gpt2", local_dir=HF_DIR,
                      allow_patterns=[
                          "config.json", "generation_config.json", "merges.txt",
                          "model.safetensors", "tokenizer.json",
                          "tokenizer_config.json", "vocab.json",
                      ],
                      ignore_patterns=["onnx/*", "*.tflite", "*.onnx"],
                      endpoint="https://hf-mirror.com")
    print(f"[权重] 下载完成：{HF_DIR}")

ensure_hf_dir()

MODEL = HF_DIR
PROMPT = "<|endoftext|>"   # 与手写引擎相同的起点，保证场景可比
TEMPERATURE = 1.0
SEED = 42

DTYPES = {
    "float32": {"label": "FP32"},
    "bfloat16": {"label": "BF16"},
}

# 执行一次完整生成并返回总耗时；校验 token 数，防止参数错位导致不可比
def run_once(llm: LLM, prompt: str, params: SamplingParams, n_tokens: int) -> float:
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], params, use_tqdm=False)  # 跑一次生成
    total_ms = (time.perf_counter() - t0) * 1000
    gen = len(outputs[0].outputs[0].token_ids)
    assert gen == n_tokens, f"生成了 {gen} tokens，预期 {n_tokens}（采样参数未对齐）"  # 校验长度
    return total_ms

# ----------------------------------------------------------------------------
# bench：加载一档 vLLM 引擎，预热后计时，结束时释放显存
# ----------------------------------------------------------------------------
def bench(dtype: str, n_tokens: int, n_runs: int,
          quant: str | None = None, ct_ckpt: str | None = None,
          kv_cache_dtype: str | None = None) -> float:
    if quant == "compressed-tensors":
        # 从 checkpoint 路径名判断是 INT8 还是 FP8，标注在结果里
        is_int8_ckpt = bool(ct_ckpt) and "int8" in os.path.basename(ct_ckpt.rstrip("/")).lower()
        label = ("INT8 · 打包 W8A16（与手写同规格）" if is_int8_ckpt
                 else "FP8 · 静态 W8A16（与手写同规格）")
    else:
        label = DTYPES[dtype]["label"] + (f" + {quant}" if quant else "")
    if kv_cache_dtype:
        label += f" + KV={kv_cache_dtype}"
    model = ct_ckpt if ct_ckpt else MODEL

    print(f"[{label}] 加载引擎（dtype={dtype}" + (f", quant={quant}" if quant else "") + "）...")
    kwargs = dict(
        model=model,
        dtype=dtype,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
    )
    if quant:
        kwargs["quantization"] = quant  # 在线量化方案
    if kv_cache_dtype:
        kwargs["kv_cache_dtype"] = kv_cache_dtype  # KV 精度
    llm = LLM(**kwargs)  # 加载引擎

    # ignore_eos 保证每次都生成满 n_tokens，长度一致计时才可比
    params = SamplingParams(
        max_tokens=n_tokens,
        ignore_eos=True,
        temperature=TEMPERATURE,
        seed=SEED,
    )

    print(f"[{label}] 预热 1 次（{n_tokens} tokens）...")
    run_once(llm, PROMPT, params, n_tokens)  # 预热不计时

    totals_ms = []
    for i in range(n_runs - 1):
        total_ms = run_once(llm, PROMPT, params, n_tokens)
        totals_ms.append(total_ms)
        print(f"[{label}] run {i + 1}: {total_ms / n_tokens:.3f} ms/token")
    per_token = statistics.median(totals_ms) / n_tokens  # 中位数每 token
    print(f"[{label}] 中位数: {per_token:.3f} ms/token")

    # 释放本档占用的显存，避免多档连续跑时显存叠加
    llm.sleep(level=2, mode="abort")
    return per_token

def main():
    import argparse
    ap = argparse.ArgumentParser(description="vLLM GPT-2 单请求延迟基准（默认输出三档）")
    ap.add_argument("--quantization",
                    choices=["fp8", "fp8_per_channel", "compressed-tensors"],
                    default=None,
                    help="指定 8 位档的实现；不指定时默认用静态 ckpt（缺失回退在线量化）")
    ap.add_argument("--ct-ckpt", default=None,
                    help="compressed-tensors 预量化 checkpoint 目录（配合 --quantization compressed-tensors）")
    ap.add_argument("--int8-ckpt", default=CT_INT8_DIR,
                    help=f"INT8 预量化 checkpoint 目录（默认 {CT_INT8_DIR}；不存在则跳过该档）")
    ap.add_argument("--skip-int8", action="store_true",
                    help="默认多档跑完时跳过 INT8 档")
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16"],
                    help="目标模型精度（默认 BF16）")
    ap.add_argument("--n-tokens", type=int, default=1000,
                    help="每轮生成 token 数（默认 1000，与 CUDA 基准同场景）")
    ap.add_argument("--n-runs", type=int, default=6,
                    help="预热 1 次 + 实测次数（默认 6：预热 1 + 实测 5）")
    # auto = 跟随模型 dtype，BF16 模型即不量化；其余为各档 KV 量化格式
    ap.add_argument("--kv-cache-dtype", default=None,
                    choices=["auto", "fp8_e4m3", "fp8_e5m2", "int8_per_token_head",
                             "fp8_per_token_head", "int4_per_token_head"],
                    help="KV cache 存储精度；不指定 = vLLM 默认 auto（跟随模型 dtype，即 BF16 不量化）")
    args = ap.parse_args()

    n_tokens, n_runs = args.n_tokens, args.n_runs
    results = {}

    if args.quantization is None:
        # 默认计划：FP32、BF16 两档，加上能找到的预量化 checkpoint
        plan = [("float32", None, None),
                ("bfloat16", None, None)]
        if os.path.isdir(CT_CKPT_DIR):
            plan.append(("bfloat16", "compressed-tensors", CT_CKPT_DIR))  # FP8 静态档
        else:
            # 静态 checkpoint 缺失时退回在线量化，规格与手写不同，结果仅供参考
            print(f"⚠ 未找到静态 FP8 checkpoint（{CT_CKPT_DIR}），FP8 档回退在线量化；"
                  f"需要静态 ckpt 时先运行 python export_weights.py 生成"
                  "（W8A8，与手写规格不同、速度也更慢，对比结果仅供参考）")
            plan.append(("bfloat16", "fp8", None))
        if not args.skip_int8 and os.path.isdir(args.int8_ckpt):
            plan.append(("bfloat16", "compressed-tensors", args.int8_ckpt))
    else:
        # 指定量化档时：一个 BF16 基线 + 一个目标量化档
        target_dtype = args.dtype or "bfloat16"
        plan = [("bfloat16", None, None),
                (target_dtype, args.quantization, args.ct_ckpt)]

    for dtype, quant, ckpt in plan:
        results[(dtype, quant, ckpt)] = bench(
            dtype, n_tokens, n_runs, quant, ckpt, args.kv_cache_dtype)

    print("\n" + "=" * 60)
    print(f"汇总（ms/token，batch=1，{n_tokens} tokens，{n_runs - 1} 次中位数）")
    print("=" * 60)
    for (dtype, quant, ckpt), val in results.items():
        if quant == "compressed-tensors":
            is_int8_ckpt = bool(ckpt) and "int8" in os.path.basename(ckpt.rstrip("/")).lower()
            name = ("INT8 · 打包 W8A16（与手写同规格）" if is_int8_ckpt
                    else "FP8 · 静态 W8A16（与手写同规格）")
        else:
            name = DTYPES[dtype]["label"] + (f" + {quant}" if quant else "")
        if args.kv_cache_dtype:
            name += f" + KV={args.kv_cache_dtype}"
        print(f"  vLLM {name}: {val:.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
