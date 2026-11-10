# ============================================================================
# verify_precision.py —— 精度验证工具：以 HuggingFace 为基准给各实现打分
# ============================================================================
# 读取 C++ 程序落盘的 dump 文件，和 HF 官方模型在相同输入下的输出逐项对比，
# 输出一份可读的验证报告。分两部分：
#   1. logits 对比：余弦相似度、平均偏差、KL 散度、Top-1/Top-5 候选，
#      并按容差给出“完美/基本/量化容差内通过/失败”的结论；
#   2. 困惑度 PPL 对比（检测到数据时自动执行）。
#
# 参考模型固定用 eager 注意力：它和 C++ 全量重算的数学过程完全一致，避免注意力实现差异混进精度结论。
# 量化档使用更宽松的独立容差。
# ============================================================================
import argparse
import glob
import os
import re

import numpy as np
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DUMP_DIR = os.path.join(BASE_DIR, "assets", "dumps")
HF_DIR = os.path.join(BASE_DIR, "assets", "weights", "hf")
MODEL_NAME = "openai-community/gpt2"
EOT_TOKEN = 50256

DTYPE_MAP = {"bf16": torch.bfloat16, "fp32": torch.float32}
DTYPE_LABEL = {"bf16": "BF16 (gpt2_124M_bf16.bin)", "fp32": "FP32 (gpt2_124M.bin)",
               "fp8": "FP8 量化版 (gpt2_124M_fp8.bin)",
               "int8": "INT8 量化版 (gpt2_124M_int8.bin)"}

# 量化档的参考模型仍用 BF16
FP8_REF_DTYPE = torch.bfloat16

# 三档判定容差：完美档（全精度）、基本档（全精度兜底）、量化档（INT8/FP8）
PERFECT_COSINE = 0.99
PERFECT_MEAN_ABS = 0.25

PERFECT_KL = 0.01
BASIC_COSINE = 0.95
BASIC_MEAN_ABS = 2.0

QUANT_COSINE = 0.99
QUANT_MEAN_ABS = 3.0
QUANT_KL = 0.3

# dump 前缀 → 产生它的源文件，用于报告里标注被测实现
PREFIX_LABEL = {
    "cuda": "inference_gpt2.cu",
    "graph": "inference_gpt2_opt.cu",
    "gemv": "inference_gpt2_opt.cu",
    "gemv_fp32": "inference_gpt2_opt.cu",
    "gemv_lmhead": "inference_gpt2_opt.cu",
    "fp8": "inference_gpt2_opt.cu",
    "int8": "inference_gpt2_opt.cu",
}

DTYPE_SUFFIX = {"bf16": "BF16", "fp32": "FP32", "fp8": "FP8", "int8": "INT8"}

# 从 dump 前缀解析（实现, 精度）；opt/base 形式的前缀自带精度
def parse_dump_prefix(prefix):
    parts = prefix.split("_")
    if len(parts) == 2 and parts[0] in ("opt", "base") and parts[1] in ("bf16", "fp8", "fp32", "int8"):
        impl = "inference_gpt2_opt.cu" if parts[0] == "opt" else "inference_gpt2.cu"
        return impl, parts[1].upper()
    return PREFIX_LABEL.get(prefix, prefix), None

# 报告里展示的实现名：前缀自带精度用自带的，否则用命令行 --dtype
def impl_name(prefix, dtype):
    impl, prec = parse_dump_prefix(prefix)
    if prec:
        return f"{impl}（{prec}）"
    return f"{impl}（{DTYPE_SUFFIX[dtype]}）"

# 判断一组对比是否应按量化容差判定
def is_quant(prefix, dtype):
    _, prec = parse_dump_prefix(prefix)
    if prec:
        return prec in ("INT8", "FP8")
    return prefix in ("fp8", "int8") or dtype in ("fp8", "int8")

TOKENIZER = None

# 把 token id 转成可读记号；BPE 特殊字符替换成直观符号
def decode_token(tok_id):
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(HF_DIR)
    tok = TOKENIZER.convert_ids_to_tokens([tok_id])
    if not tok or tok[0] is None:
        return f"<id:{tok_id}>"
    # Ġ 表示前导空格、Ċ 表示换行，替换后在报告里更直观
    return tok[0].replace("Ġ", "␣").replace("Ċ", "\\n")[:10]

# 数值稳定的 softmax：先减最大值再取指数
def softmax_probs(logits):
    z = logits - np.max(logits)  # 减最大值
    e = np.exp(z)
    return e / e.sum()  # 归一化成概率

# 本地权重目录缺失时自动下载
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

# 加载 HF 参考模型；eager 不可用时（版本差异）退回默认实现
def load_hf_model(dtype, model_dir=HF_DIR):
    try:
        model = GPT2LMHeadModel.from_pretrained(
            model_dir, torch_dtype=dtype,
            attn_implementation="eager",
        ).eval()
    except TypeError:
        model = GPT2LMHeadModel.from_pretrained(
            model_dir, torch_dtype=dtype,
        ).eval()
    return model

# 对给定输入做一次前向，返回最后位置的 logits（numpy float32）
def forward_hf_logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids).logits[0, -1, :].float().numpy()  # 最后位置 logits 转 numpy

def save_fp32(path, arr):
    arr.astype(np.float32).tofile(path)

def load_fp32(path):
    return np.fromfile(path, dtype=np.float32)

# ----------------------------------------------------------------------------
# 为每个 dump 生成对应的 HF 基准 logits
# ----------------------------------------------------------------------------
def generate_hf_baseline(model, dtype_name):
    print("=== 步骤 1/2：生成 HuggingFace 基准 Logits ===")
    print(f"模型: {MODEL_NAME}  dtype={dtype_name}  attn=eager")

    # 第 1 步基准：单一起始 token
    logits = forward_hf_logits(model, torch.tensor([[EOT_TOKEN]]))
    save_fp32(os.path.join(DUMP_DIR, "hf_logits_t1.bin"), logits)  # 落盘
    print(f"-> hf_logits_t1.bin (Top-1={np.argmax(logits)})")

    # 多步基准：对每个输入 dump 用相同 token 序列跑一次 HF
    for inputs_path in sorted(glob.glob(os.path.join(DUMP_DIR, "*_inputs_step*.bin"))):
        m = re.search(r"^(.*)_inputs_step(\d+)\.bin$", os.path.basename(inputs_path))
        if m is None:
            continue
        prefix, step = m.group(1), int(m.group(2))
        logits_path = os.path.join(DUMP_DIR, f"{prefix}_logits_step{step}.bin")
        if not os.path.exists(logits_path):
            continue
        toks = np.fromfile(inputs_path, dtype=np.int32)
        if len(toks) == 0:
            continue
        logits = forward_hf_logits(model, torch.tensor([toks.tolist()]))
        save_fp32(os.path.join(DUMP_DIR, f"hf_{prefix}_logits_step{step}.bin"), logits)
        print(f"-> hf_{prefix}_logits_step{step}.bin (Top-1={np.argmax(logits)}, 前缀长度={len(toks)})")

# 稳定排序取 Top-k：并列时索引小者在前，保证结果可复现
def stable_topk(logits, k=5):
    return np.argsort(-logits, kind="stable")[:k]  # 负号降序，稳定排序取前 k

# ----------------------------------------------------------------------------
# 计算两组 logits 的全部指标。a=参考，b=被测
# ----------------------------------------------------------------------------
def compute_metrics(a, b):
    a64, b64 = a.astype(np.float64), b.astype(np.float64)  # 64 位减少计算误差
    # 余弦：只比向量方向，不受整体尺度影响；加小常数防除零
    cosine = float(np.clip(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12),
                           -1.0, 1.0))
    abs_diff = np.abs(a - b)  # 逐元素绝对差
    rel_diff = abs_diff / (np.abs(a) + 1e-5)
    order_a = stable_topk(a)
    order_b = stable_topk(b)
    # 两侧各自归一化成概率分布，再算 KL 散度（a 相对 b 的信息增益）
    sa = np.exp(a - a.max()); sa /= sa.sum()
    sb = np.exp(b - b.max()); sb /= sb.sum()
    kl = float(np.sum(sa * np.log(np.clip(sa / (sb + 1e-12), 1e-12, None))))
    return {
        "top1_ok": bool(order_a[0] == order_b[0]),
        "top1_a": int(order_a[0]),
        "top1_b": int(order_b[0]),
        "top5_overlap": len(set(order_a) & set(order_b)),
        "cosine": cosine,
        "max_abs": float(np.max(abs_diff)),
        "mean_abs": float(np.mean(abs_diff)),
        "rel_p50": float(np.percentile(rel_diff, 50) * 100),
        "rel_p95": float(np.percentile(rel_diff, 95) * 100),
        "rel_p99": float(np.percentile(rel_diff, 99) * 100),
        "mean_rel": float(np.mean(rel_diff) * 100),
        "kl": kl,
    }

# 按指标和档位给出结论；量化档用独立的更宽松阈值
def verdict(m, quant=False):
    if quant:
        if (m["top1_ok"] and m["top5_overlap"] >= 4
                and m["cosine"] >= QUANT_COSINE
                and m["mean_abs"] < QUANT_MEAN_ABS
                and m["kl"] < QUANT_KL):
            return "量化容差内通过", True  # 量化宽松档
        return "失败", False
    if (m["top1_ok"] and m["top5_overlap"] >= 4
            and m["cosine"] >= PERFECT_COSINE
            and m["mean_abs"] < PERFECT_MEAN_ABS
            and m["kl"] < PERFECT_KL):
        return "完美通过", True
    if (m["top1_ok"] and m["top5_overlap"] >= 4
            and m["cosine"] >= BASIC_COSINE
            and m["mean_abs"] < BASIC_MEAN_ABS):
        return "基本通过", True
    return "失败", False

# 打印前 5 候选的逐名次对比表
def print_top5_detail(a, b):
    pa, pb = softmax_probs(a), softmax_probs(b)
    order_a = stable_topk(a)
    order_b = stable_topk(b)
    print("     前 5 候选逐名次对比（概率为 softmax 归一化后，✓=同名次相同）:")
    print("       名次  参考词(概率)              被测词(概率)")
    for rank, (ia, ib) in enumerate(zip(order_a, order_b), 1):
        same = "✓" if ia == ib else "✗"
        print(f"       #{rank}   {decode_token(ia):<10s} id{ia:<6d} {pa[ia]*100:4.1f}%   "
              f"{decode_token(ib):<10s} id{ib:<6d} {pb[ib]*100:4.1f}%  {same}")

# ----------------------------------------------------------------------------
# 单次对比：算指标、下结论，并用白话解释每个数字
# ----------------------------------------------------------------------------
def compare(title, a, b, show_top5=False, quant=False):
    if len(a) != len(b):
        print(f"  [✗ 失败] {title}: 两侧 logits 长度不一致 A={len(a)} B={len(b)}")
        return False, "失败"
    m = compute_metrics(a, b)  # 算指标
    label, passed = verdict(m, quant=quant)  # 下结论
    mark = "✅" if passed else "❌"
    tok_a, tok_b = decode_token(m["top1_a"]), decode_token(m["top1_b"])
    if m["top1_ok"]:
        top1_txt = f'一致：双方都预测 "{tok_a}" (id {m["top1_a"]})'
    else:
        top1_txt = f'不一致：参考预测 "{tok_a}" (id {m["top1_a"]})，被测预测 "{tok_b}" (id {m["top1_b"]})'
    print(f"  {mark} {label}  {title}")
    print(f"     · 预测的最可能 token : {top1_txt}")
    overlap_txt = "（完全一致）" if m["top5_overlap"] == 5 else ""
    print(f"     · 前 5 候选重合      : {m['top5_overlap']}/5{overlap_txt}")
    print(f"     · 余弦相似度         : {m['cosine']:.6f}（1.0=方向完全一致；不受 logits 整体尺度影响）")
    print(f"     · 打分平均偏差       : {m['mean_abs']:.4f}（全部 50257 个打分差值的平均，0=完全相同）")
    print(f"     · 概率分布差异 KL    : {m['kl']:.4f}（0=完全相同）")
    # 默认只在失败时展开明细，避免报告过长
    if show_top5 or not passed:
        print_top5_detail(a, b)
    return passed, label

# 报告横幅：列出本次对比的精度、参考模型和实际存在的被测实现
def banner(dtype_name, present_prefixes):
    bar = "=" * 68
    print(bar)
    print(" GPT-2 CUDA 推理精度验证报告")
    # 只有单一精度时用 dump 自带精度，否则跟随命令行 --dtype
    precs = {pr for _, pr in (parse_dump_prefix(x) for x in present_prefixes) if pr}
    eff = precs.pop().lower() if len(precs) == 1 else dtype_name
    print(f"   对比精度 : {DTYPE_LABEL[eff]}")
    print(f"   HuggingFace : {MODEL_NAME}（官方权重，eager 注意力）")
    if present_prefixes:
        impls = " · ".join(impl_name(p, dtype_name) for p in present_prefixes)
        print(f"   被测实现 : {impls}")
    else:
        print("   被测实现 : （dumps 中暂无待比对文件，先运行 C++ 程序生成）")
    print(bar)

# ----------------------------------------------------------------------------
# 比对全部 logits dump：t1 对参考，以及各实现之间互比
# ----------------------------------------------------------------------------
def run_compare(dtype_name, show_top5=False):
    print("\n=== 步骤 2/2：比对 Logits ===")
    results = []
    def load(name):
        p = os.path.join(DUMP_DIR, name)
        return load_fp32(p) if os.path.exists(p) else None
    hf_t1 = load("hf_logits_t1.bin")
    if hf_t1 is None:
        print("⚠️ 缺少 hf_logits_t1.bin（先不带 --no-hf 运行一次生成基准）")
    # 收集所有 t1 dump（排除 HF 自己的）
    t1_files = sorted(
        f for f in os.listdir(DUMP_DIR)
        if re.fullmatch(r"[A-Za-z0-9_]+_logits_t1\.bin", f) and not f.startswith("hf_")
    )
    if not t1_files:
        print("⚠️ 缺少 *_logits_t1.bin（先运行对应 C++ 程序生成 dump）")
    t1_map = {}
    for fn in t1_files:
        arr = load(fn)
        if arr is not None:
            t1_map[fn[: -len("_logits_t1.bin")]] = arr  # 去掉后缀做键

    # 每个实现的 t1 先和 HF 参考比
    for name in sorted(t1_map):
        if hf_t1 is not None:
            ok, label = compare(f"HuggingFace vs {impl_name(name, dtype_name)}",
                                hf_t1, t1_map[name], show_top5, quant=is_quant(name, dtype_name))
            results.append((f"HuggingFace vs {impl_name(name, dtype_name)}", label, ok))

    # 各实现之间两两互比：输入相同，差异只可能来自实现本身
    names = sorted(t1_map)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ok, label = compare(f"{impl_name(names[i], dtype_name)} vs {impl_name(names[j], dtype_name)}（同一输入，专查实现差异）",
                                t1_map[names[i]], t1_map[names[j]], show_top5,
                                quant=(is_quant(names[i], dtype_name) or is_quant(names[j], dtype_name)))
            results.append((f"{impl_name(names[i], dtype_name)} vs {impl_name(names[j], dtype_name)}（同输入）", label, ok))

    if results:
        print("\n" + "-" * 68)
        print(f" 汇总（共 {len(results)} 项）:")
        for i, (title, label, ok) in enumerate(results, 1):
            mark = "✅" if ok else "❌"
            print(f"   {i:2d}. {title}  ——  {mark} {label}")
        print("-" * 68)
        all_ok = all(ok for _, _, ok in results)
        if all_ok:
            print(" 总体结论: ✅ 全部通过 —— 被测实现与参考模型的输出一致")
        else:
            print(" 总体结论: ❌ 存在未通过项（见上方对应对比的明细）")
    return all(ok for _, _, ok in results) if results else False

# HF 侧 PPL：对整条 token 序列一次性前向，交叉熵即平均 NLL
def compute_hf_ppl(model, tokens):
    ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids).logits
    # 错位对齐：前一位置的预测对应当前位置的真实 token
    logits = logits[0, :-1, :].float()
    target = torch.tensor(tokens[1:], dtype=torch.long)
    ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")  # 逐位置交叉熵
    mean_nll = float(ce.mean().item())
    return mean_nll, float(np.exp(mean_nll))  # PPL = exp(平均 NLL)

# ----------------------------------------------------------------------------
# PPL 对比：CUDA 逐步 NLL 轨迹 vs HF 对同一 token 序列的一次性前向
# ----------------------------------------------------------------------------
def run_ppl(args):
    print("\n=== PPL（困惑度）对比 ===")
    print("  口径：CUDA 逐步 NLL（自回归采样轨迹）vs HF 一次性前向（同 token 序列）")
    files = sorted(glob.glob(os.path.join(DUMP_DIR, "ppl_*_nll.bin")))
    if not files:
        print("  ⚠ 未找到 ppl_*_nll.bin——先运行 CUDA 生成：")
        print("    ./inference_gpt2_opt <权重> <分词器> <N+1> --graph [--fp8] --ppl <N> [--dump-prefix xxx]")
        return
    ensure_hf_dir()
    model = load_hf_model(torch.bfloat16, args.model)
    rows = []
    for nll_path in files:
        stem = os.path.basename(nll_path)[len("ppl_"):-len("_nll.bin")]
        toks_path = os.path.join(DUMP_DIR, f"ppl_{stem}_tokens.bin")
        if not os.path.exists(toks_path):
            print(f"  ⚠ 缺少 {toks_path}，跳过 {stem}")
            continue
        nlls = np.fromfile(nll_path, dtype=np.float32)
        toks = np.fromfile(toks_path, dtype=np.int32).tolist()
        cuda_ppl = float(np.exp(np.mean(nlls)))
        hf_nll, hf_ppl_val = compute_hf_ppl(model, toks)
        delta = (cuda_ppl - hf_ppl_val) / hf_ppl_val * 100
        rows.append((impl_name(stem, args.dtype), len(nlls), cuda_ppl, hf_nll, hf_ppl_val, delta))
    del model
    if not rows:
        return
    print(f"\n  {'实现':<22s} {'步数':>5s} {'CUDA PPL':>10s} {'HF NLL':>10s} "
          f"{'HF PPL(同序列)':>14s} {'偏差':>8s}")
    for stem, n, cp, cn, hp, d in rows:
        print(f"  {stem:<22s} {n:>5d} {cp:>10.3f} {cn:>10.3f} {hp:>14.3f} {d:>+7.1f}%")
    print("  量化健康度以 CUDA PPL 绝对值 + 短步 logits 对比 + 文本保真共同判定。")
    print("  参考：openai-community/gpt2 论文口径在 WikiText-2 的 PPL ≈ 41.5（绝对值随序列变化，以对比为准）")

# 自动检测 PPL 数据：有就对比，没有就提示怎么生成
def auto_ppl(args):
    if glob.glob(os.path.join(DUMP_DIR, "ppl_*_nll.bin")):
        run_ppl(args)
    else:
        print("\n  （未检测到 PPL 数据：运行推理时加 --ppl <N> 即可生成并自动附上困惑度对比。")
        print("   示例（FP8）：./inference_gpt2_opt assets/weights/gpt2_124M_fp8.bin \\")
        print("                assets/weights/gpt2_tokenizer.bin 1000 --graph --fp8 --ppl 100）")

def main():
    parser = argparse.ArgumentParser(description="GPT-2 CUDA 推理精度验证（HF 基准 dtype 对齐）")
    parser.add_argument("--dtype", choices=["bf16", "fp32", "fp8", "int8"], default="bf16",
                        help="对比精度：bf16/fp32 须与 CUDA 编译精度一致；fp8/int8 为量化版（参考仍用 BF16 模型）")
    parser.add_argument("--no-hf", action="store_true",
                        help="跳过 HF 基准生成，只对比已有的 dump 文件")
    parser.add_argument("--top5", action="store_true",
                        help="打印每个对比组的 Top-5 逐名次明细（默认仅失败组打印）")
    parser.add_argument("--ppl", action="store_true",
                        help="只做困惑度对比（需先运行 CUDA --ppl <N> 生成 NLL/tokens 落盘）")
    parser.add_argument("--model", default=HF_DIR,
                        help="HF 模型目录/名（默认统一权重目录 assets/weights/hf）")
    args = parser.parse_args()

    os.makedirs(DUMP_DIR, exist_ok=True)
    ensure_hf_dir()
    # 扫描现有 dump，确定本次报告里实际存在的被测实现
    present = set()
    for fn in os.listdir(DUMP_DIR):
        m = re.match(r"^(.+)_logits_t1\.bin$", fn)
        if m and not fn.startswith("hf_"):
            present.add(m.group(1))
        m = re.match(r"^(.+)_inputs_step\d+\.bin$", fn)
        if m:
            present.add(m.group(1))
    banner(args.dtype, sorted(present))

    if args.ppl:
        run_ppl(args)
        return

    # 量化档参考模型固定 BF16；全精度档跟随对应 dtype
    ref_dtype = FP8_REF_DTYPE if args.dtype in ("fp8", "int8") else DTYPE_MAP[args.dtype]
    if not args.no_hf:
        model = load_hf_model(ref_dtype, args.model)  # 加载参考模型
        generate_hf_baseline(model, args.dtype)       # 生成基准
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 释放显存

    run_compare(args.dtype, show_top5=args.top5)
    auto_ppl(args)

if __name__ == "__main__":
    main()
