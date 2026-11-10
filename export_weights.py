# ============================================================================
# export_weights.py —— 权重导出流水线：从 HuggingFace 到工程自有格式
# ============================================================================
# 一条命令把官方 GPT-2 处理成工程需要的全部文件，含六步：
#   1. 下载 HF 权重到统一本地目录；
#   2. 导出 FP32/BF16 扁平二进制（本工程的权重格式，带 256 整数文件头）；
#   3. 导出分词器 tokenizer.bin；
#   4. 调用 quantize 生成 INT8/FP8 权重与 scale；
#   5、6. 导出 vLLM 可直接加载的预量化 checkpoint（ct_fp8/ct_int8）。
# 每一步都可用命令行开关跳过，方便在已有产物上增量执行。
#
# 关键背景：HF 的 GPT-2 用 Conv1D 存权重，布局是 [输入, 输出]，而本工程按 [输出, 输入] 理解矩阵，导出时必须转置；词表还要补齐到 padded 大小。
# ============================================================================
import argparse
import os
import struct

import torch
from huggingface_hub import snapshot_download
from transformers import GPT2LMHeadModel, GPT2Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# GPT-2 124M 的固定尺寸，同时写进文件头供 C++ 侧读取
N_LAYERS = 12
N_HEADS = 12
VOCAB_SIZE = 50257
C = 768
MAX_SEQ = 1024
PADDED_VOCAB = 50304
EOT_TOKEN = 50256

# 模型与分词器文件的魔数、版本号，C++ 侧据此校验格式
MODEL_MAGIC = 20251103
MODEL_VERSION_FP32 = 3
MODEL_VERSION_BF16 = 5
MODEL_HEADER_WORDS = 256

TOKENIZER_MAGIC = 20251103
TOKENIZER_VERSION = 2
TOKENIZER_HEADER_WORDS = 256

# 以 FP32 加载 HF 模型并返回 state_dict，保持导出前的原始精度
def _state_dict_from_hf(hf_dir: str) -> dict:

    model = GPT2LMHeadModel.from_pretrained(hf_dir, torch_dtype=torch.float32)
    return model.state_dict()

# ----------------------------------------------------------------------------
# 按工程固定的 16 张量顺序拼接扁平权重，顺序必须与 C++ 侧布局严格一致
# ----------------------------------------------------------------------------
def export_flat_weights(state_dict: dict, out_path: str, dtype: torch.dtype, version: int) -> int:
    chunks = []
    def push(name: str) -> None:
        chunks.append(state_dict[name].reshape(-1))  # 单个张量展平入列

    def push_layers(key_fmt: str, transpose: bool = False) -> None:
        for i in range(N_LAYERS):  # 逐层取
            t = state_dict[key_fmt % i]
            if transpose:
                # Conv1D 权重转置成工程约定的 [输出, 输入]
                t = t.t().contiguous()
            chunks.append(t.reshape(-1))

    # 词嵌入补齐到 padded 词表，多出的行填零
    wte = torch.zeros(PADDED_VOCAB, C, dtype=torch.float32)
    wte[:VOCAB_SIZE] = state_dict["transformer.wte.weight"]  # 原始行填入
    chunks.append(wte.reshape(-1))
    push("transformer.wpe.weight")

    # 顺序与 ParameterTensors 的 16 个字段一一对应
    push_layers("transformer.h.%d.ln_1.weight")
    push_layers("transformer.h.%d.ln_1.bias")
    push_layers("transformer.h.%d.attn.c_attn.weight", transpose=True)
    push_layers("transformer.h.%d.attn.c_attn.bias")
    push_layers("transformer.h.%d.attn.c_proj.weight", transpose=True)
    push_layers("transformer.h.%d.attn.c_proj.bias")
    push_layers("transformer.h.%d.ln_2.weight")
    push_layers("transformer.h.%d.ln_2.bias")
    push_layers("transformer.h.%d.mlp.c_fc.weight", transpose=True)
    push_layers("transformer.h.%d.mlp.c_fc.bias")
    push_layers("transformer.h.%d.mlp.c_proj.weight", transpose=True)
    push_layers("transformer.h.%d.mlp.c_proj.bias")
    push("transformer.ln_f.weight")
    push("transformer.ln_f.bias")

    flat = torch.cat(chunks).to(dtype).contiguous()  # 拼接后转目标精度

    # 文件头：魔数、版本、六个尺寸，其余位置补零
    header = [MODEL_MAGIC, version, MAX_SEQ, VOCAB_SIZE, N_LAYERS, N_HEADS, C, PADDED_VOCAB] \
        + [0] * (MODEL_HEADER_WORDS - 8)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<256i", *header))  # 256 个 int 文件头

        # BF16 没有独立的 numpy 类型，按 int16 视图落盘、字节不变
        if dtype == torch.bfloat16:
            flat = flat.view(torch.int16)
        flat.detach().cpu().numpy().tofile(f)
    return flat.numel()

# GPT-2 BPE 的反向映射：HF 词表里的 unicode 显示字符 → 原始字节值
def _byte_decoder() -> dict:

    # 可打印字符保持原样，其余字节映射到 256 以上的 unicode 位置
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)          # 不可打印字节
            cs.append(256 + n)    # 映射到可显示 unicode
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

# 导出分词器：每个 token 写成“1 字节长度 + 原始字节”，C++ 侧按此顺序读
def export_tokenizer(hf_dir: str, out_path: str) -> int:

    tok = GPT2Tokenizer.from_pretrained(hf_dir)
    vocab = tok.get_vocab()
    byte_decoder = _byte_decoder()

    n_vocab = len(vocab)
    # 先按 id 排好字符串，再逐个写回，保证下标即 token id
    id_to_str = [""] * n_vocab
    for token_str, token_id in vocab.items():
        id_to_str[token_id] = token_str

    header = [TOKENIZER_MAGIC, TOKENIZER_VERSION, n_vocab, EOT_TOKEN] \
        + [0] * (TOKENIZER_HEADER_WORDS - 4)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<256I", *header))
        for i in range(n_vocab):
            raw = bytes(byte_decoder[c] for c in id_to_str[i])  # 显示字符还原成原始字节
            assert 0 < len(raw) < 256, f"token {i} 长度异常: {len(raw)}"
            f.write(struct.pack("<B", len(raw)))  # 1 字节长度
            f.write(raw)
    return n_vocab

# 把字节数换算成带单位的易读字符串
def _human_size(num_bytes: int) -> str:

    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"

# 读扁平 bin，返回文件头和数据字节；skip_bytes=0 表示无文件头
def _read_bin(path: str, skip_bytes: int = 1024):

    with open(path, "rb") as f:
        data = f.read()
    head = None
    if skip_bytes:
        head = list(struct.unpack("<256i", data[:skip_bytes]))
        data = data[skip_bytes:]
    return head, data

# 把 [输出, 输入] 的 INT8 矩阵每 4 个字节打包进一个 int32（vLLM pack 格式）
def _pack_to_int32(mat):

    rows, cols = mat.shape
    assert cols % 4 == 0, f"输入维必须是 4 的倍数才能打包，当前是 {cols}"
    # 有符号值先加 128 转成无符号字节，避免移位时符号位干扰
    unsigned = (mat.to(torch.int16) + 128).to(torch.uint8)
    groups = unsigned.reshape(rows, cols // 4, 4).to(torch.int32)
    shifts = torch.arange(4, dtype=torch.int32) * 8  # 0/8/16/24 位

    # 四个字节按小端移位后相加，得到一个打包整数
    return (groups << shifts).sum(dim=2, dtype=torch.int32)

# ----------------------------------------------------------------------------
# export_ct_ckpt：把三个 bin 重组为 vLLM compressed-tensors checkpoint
# ----------------------------------------------------------------------------
def export_ct_ckpt(out_root: str, hf_dir: str, fmt: str) -> str:
    import json
    import numpy as np
    from safetensors.torch import save_file

    # 输入：BF16 整权重、量化权重、scale，三者都由前面步骤生成
    quant_bin = os.path.join(out_root, f"gpt2_124M_{fmt}.bin")
    scale_bin = os.path.join(out_root, f"gpt2_124M_{fmt}_scales.bin")
    bf16_bin = os.path.join(out_root, "gpt2_124M_bf16.bin")
    for p in (bf16_bin, quant_bin, scale_bin):
        if not os.path.isfile(p):
            raise RuntimeError(f"缺少 {p}，请先完成量化步骤（make weights）")

    h_bf16, bf = _read_bin(bf16_bin)
    h_q8, q8 = _read_bin(quant_bin)
    _, scales_raw = _read_bin(scale_bin, skip_bytes=0)
    if int(h_bf16[0]) != MODEL_MAGIC or int(h_q8[0]) != MODEL_MAGIC:
        raise RuntimeError("bin 文件头魔数不对，权重文件可能损坏")
    maxT, V, L, C, Vp = int(h_bf16[2]), int(h_bf16[3]), int(h_bf16[4]), int(h_bf16[6]), int(h_bf16[7])

    # 16 个张量的元素数与是否量化，用于计算各段偏移
    n = [Vp * C, maxT * C, L * C, L * C, L * 3 * C * C, L * 3 * C, L * C * C, L * C,
         L * C, L * C, L * 4 * C * C, L * 4 * C, L * C * 4 * C, L * C, C, C]
    lin = [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]

    # 量化文件的字节分段（量化段 1 字节、非量化段 2 字节）
    seg, off = [], 0
    for i, cnt in enumerate(n):
        seg.append((off, cnt, lin[i]))  # 记录段起点、元素数、是否量化
        off += cnt * (1 if lin[i] else 2)

    # BF16 文件按 2 字节/元素累计偏移
    bf_seg, acc = [], 0
    for cnt in n:
        bf_seg.append(acc)
        acc += cnt * 2

    # 从 BF16 数据段取元素，返回 bfloat16 张量
    def take_bf16(seg_idx: int, elem_start: int, count: int):
        byte_off = bf_seg[seg_idx] + elem_start * 2
        raw = np.frombuffer(bf, dtype="<u2", count=count, offset=byte_off).copy()  # 视图后 copy
        return torch.from_numpy(raw).view(torch.bfloat16)

    # 从量化数据段取整矩阵；FP8 需转置回 HF 布局并按 E4M3 类型解释
    def take_quant(seg_idx: int, rows: int, cols: int):
        byte_off, count, _ = seg[seg_idx]
        raw = np.frombuffer(q8, dtype=np.uint8, count=count, offset=byte_off)
        mat = raw.reshape(rows, cols)
        if fmt == "fp8":
            return torch.from_numpy(mat.T.copy()).view(torch.float8_e4m3fn)  # 转置回 HF 布局
        return torch.from_numpy(mat.copy())

    if fmt == "int8":
        # INT8 走 pack-quantized，张量名带 _packed 后缀
        w_suffix = "weight_packed"
        def take_rows(mat, start: int, count: int):
            return _pack_to_int32(mat[start:start + count])
    else:
        # FP8 走 float-quantized，直接切片保留原字节
        w_suffix = "weight"
        def take_rows(mat, start: int, count: int):
            return mat[:, start:start + count].contiguous()

    # scale 按五个 Linear 分段，每段整理成 [通道数, 1]
    sc = np.frombuffer(scales_raw, dtype="<f4")
    s0 = Vp  # 五段边界
    s1 = s0 + L * 3 * C
    s2 = s1 + L * C
    s3 = s2 + L * 4 * C
    sc = [sc[0:s0], sc[s0:s1], sc[s1:s2], sc[s2:s3], sc[s3:]]

    def make_scale(arr, count: int):

        return torch.from_numpy(np.array(arr[:count], dtype=np.float32).reshape(count, 1))

    # 按 HF 命名重建全部张量
    ts = {}

    # 非量化的嵌入与最终 LN
    ts["transformer.wte.weight"] = take_bf16(0, 0, Vp * C)[: V * C].reshape(V, C)
    ts["transformer.wpe.weight"] = take_bf16(1, 0, maxT * C).reshape(maxT, C)
    ts["transformer.ln_f.weight"] = take_bf16(14, 0, C)
    ts["transformer.ln_f.bias"] = take_bf16(15, 0, C)

    # 四个量化矩阵先整块取出，再按层切片
    qkv = take_quant(4, L * 3 * C, C)
    apw = take_quant(6, L * C, C)
    fcw = take_quant(10, L * 4 * C, C)
    fcp = take_quant(12, L * C, 4 * C)

    for l in range(L):
        b = l * C
        q = l * 3 * C
        f4 = l * 4 * C

        # 非量化的 LN 权重偏置
        ts[f"transformer.h.{l}.ln_1.weight"] = take_bf16(2, b, C)
        ts[f"transformer.h.{l}.ln_1.bias"] = take_bf16(3, b, C)
        ts[f"transformer.h.{l}.ln_2.weight"] = take_bf16(8, b, C)
        ts[f"transformer.h.{l}.ln_2.bias"] = take_bf16(9, b, C)

        # 量化权重 + 对应 scale + 非量化偏置
        ts[f"transformer.h.{l}.attn.c_attn.{w_suffix}"] = take_rows(qkv, q, 3 * C)
        ts[f"transformer.h.{l}.attn.c_attn.weight_scale"] = make_scale(sc[1][q:], 3 * C)
        ts[f"transformer.h.{l}.attn.c_attn.bias"] = take_bf16(5, q, 3 * C)

        ts[f"transformer.h.{l}.attn.c_proj.{w_suffix}"] = take_rows(apw, b, C)
        ts[f"transformer.h.{l}.attn.c_proj.weight_scale"] = make_scale(sc[2][b:], C)
        ts[f"transformer.h.{l}.attn.c_proj.bias"] = take_bf16(7, b, C)

        ts[f"transformer.h.{l}.mlp.c_fc.{w_suffix}"] = take_rows(fcw, f4, 4 * C)
        ts[f"transformer.h.{l}.mlp.c_fc.weight_scale"] = make_scale(sc[3][f4:], 4 * C)
        ts[f"transformer.h.{l}.mlp.c_fc.bias"] = take_bf16(11, f4, 4 * C)

        ts[f"transformer.h.{l}.mlp.c_proj.{w_suffix}"] = take_rows(fcp, b, C)
        ts[f"transformer.h.{l}.mlp.c_proj.weight_scale"] = make_scale(sc[4][b:], C)
        ts[f"transformer.h.{l}.mlp.c_proj.bias"] = take_bf16(13, b, C)

    # 在 HF config 基础上追加 compressed-tensors 量化配置
    with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    # INT8 用 pack-quantized+整数类型，FP8 用 float-quantized+浮点类型
    ct_format, wtype = (("pack-quantized", "int") if fmt == "int8"
                        else ("float-quantized", "float"))
    cfg["quantization_config"] = {
        "format": ct_format,
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {
                    "type": wtype,
                    "strategy": "channel",        # 逐通道 scale，与手写 per-channel 一致
                    "symmetric": True,
                    "dynamic": False,            # 静态量化，scale 预先算好
                    "num_bits": 8,
                },
                "targets": ["Linear"],
                "input_activations": None,       # W8A16：只量化权重，激活不量化
            }
        },
        "ignore": ["lm_head"],                   # 词表投影不量化
    }

    out_dir = os.path.join(out_root, f"ct_{fmt}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    save_file({k: v.contiguous() for k, v in ts.items()},
              os.path.join(out_dir, "model.safetensors"))  # 落 safetensors

    # 分词器相关文件直接复制，保证 checkpoint 自包含
    import shutil
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        src = os.path.join(hf_dir, name)
        if os.path.isfile(src):
            shutil.copy(src, out_dir)  # 复制
    return out_dir

def main() -> None:
    # 模块 docstring 已被剥离，这里描述直接写死
    parser = argparse.ArgumentParser(description="从 HuggingFace 导出 GPT-2 权重到统一目录")
    parser.add_argument("--repo", default="openai-community/gpt2",
                        help="HuggingFace 仓库名（默认 openai-community/gpt2）")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "assets", "weights"),
                        help="输出目录（默认 <工程>/assets/weights）")
    parser.add_argument("--endpoint", default="https://hf-mirror.com",
                        help="HuggingFace 镜像端点（默认 hf-mirror.com，可改 https://huggingface.co）")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过下载（hf/ 已存在时使用）")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="跳过 8 位量化（默认自动用 quantize 从 BF16 导出 INT8 与 FP8）")
    parser.add_argument("--skip-vllm-ckpt", action="store_true",
                        help="跳过 vLLM checkpoint 导出（默认自动生成 ct_int8/ 与 ct_fp8/）")
    args = parser.parse_args()

    out_root = os.path.abspath(args.out)
    hf_dir = os.path.join(out_root, "hf")
    os.makedirs(hf_dir, exist_ok=True)

    print("=" * 60)
    print("权重下载与导出")
    print("=" * 60)
    print(f"仓库      : {args.repo}")
    print(f"输出目录  : {out_root}")
    print(f"镜像端点  : {args.endpoint}")
    print()

    # 第 1 步：下载；setdefault 保证用户已设端点时不覆盖
    if not args.skip_download:
        os.environ.setdefault("HF_ENDPOINT", args.endpoint)
        print(f"[1/6] 下载 {args.repo} → {hf_dir} ...")
        snapshot_download(repo_id=args.repo, local_dir=hf_dir,
                          allow_patterns=[
                              "config.json", "generation_config.json", "merges.txt",
                              "model.safetensors", "tokenizer.json",
                              "tokenizer_config.json", "vocab.json",
                          ],
                          ignore_patterns=["onnx/*", "*.tflite", "*.onnx"],
                          endpoint=args.endpoint)
    else:
        print("[1/6] 跳过下载（--skip-download），使用已有 hf/ 目录")

    # 第 2 步：导出两档扁平权重
    print(f"[2/6] 加载权重并导出扁平二进制 ...")
    state_dict = _state_dict_from_hf(hf_dir)
    n_fp32 = export_flat_weights(state_dict, os.path.join(out_root, "gpt2_124M.bin"),
                                 torch.float32, MODEL_VERSION_FP32)
    n_bf16 = export_flat_weights(state_dict, os.path.join(out_root, "gpt2_124M_bf16.bin"),
                                 torch.bfloat16, MODEL_VERSION_BF16)
    del state_dict

    # 第 3 步：分词器
    print(f"[3/6] 导出分词器 tokenizer.bin ...")
    export_tokenizer(hf_dir, os.path.join(out_root, "gpt2_tokenizer.bin"))

    print()
    print("导出完成，文件如下：")
    total = 0
    for name in ("gpt2_124M.bin", "gpt2_124M_bf16.bin", "gpt2_tokenizer.bin"):
        p = os.path.join(out_root, name)
        size = os.path.getsize(p)
        total += size
        print(f"  {name:<22} {_human_size(size)}")
    for name in sorted(os.listdir(hf_dir)):
        p = os.path.join(hf_dir, name)
        if os.path.isfile(p):
            total += os.path.getsize(p)
    print(f"  总占用：{_human_size(total)}（含 hf/ 原始格式）")
    print()
    print(f"元素总数校验：FP32 {n_fp32} / BF16 {n_bf16}（应为 124475904）")

    # 第 4 步：调用 quantize 二进制做两档量化
    quantize_bin = os.path.join(BASE_DIR, "quantize")
    bf16_bin = os.path.join(out_root, "gpt2_124M_bf16.bin")
    plans = [("int8", "gpt2_124M_int8.bin", "gpt2_124M_int8_scales.bin"),
             ("fp8", "gpt2_124M_fp8.bin", "gpt2_124M_fp8_scales.bin")]
    if args.skip_quantize:
        print("[4/6] 跳过 8 位量化（--skip-quantize）；需要时执行 make weights 或手动 ./quantize")
    else:
        if not os.path.isfile(quantize_bin):

            # 量化器缺失时现场编译；没有 nvcc 则跳过
            import shutil, subprocess
            if shutil.which("nvcc"):
                print("[4/6] 未找到 quantize 二进制，先编译量化器 ...")
                subprocess.run(["make", "quantize"], cwd=BASE_DIR, check=True)
            else:
                print("[4/6] 环境无 nvcc，跳过 8 位量化；请在有 CUDA 的环境执行 make weights")
        if os.path.isfile(quantize_bin):
            import subprocess
            for fmt, wname, sname in plans:
                w_path = os.path.join(out_root, wname)
                s_path = os.path.join(out_root, sname)
                print(f"[4/6] 量化 {fmt.upper()}（bf16.bin → {wname} + {sname}）...")

                subprocess.run([quantize_bin, bf16_bin, w_path, s_path,
                                "--format", fmt], check=True)
                print(f"  {wname:<28} {_human_size(os.path.getsize(w_path))}")
                print(f"  {sname:<28} {_human_size(os.path.getsize(s_path))}")

    # 第 5、6 步：两份 vLLM checkpoint
    for step, fmt in ((5, "fp8"), (6, "int8")):
        if args.skip_vllm_ckpt:
            print(f"[{step}/6] 跳过 vLLM {fmt.upper()} checkpoint 导出（--skip-vllm-ckpt）")
            continue
        if not os.path.isfile(os.path.join(out_root, f"gpt2_124M_{fmt}.bin")):
            print(f"[{step}/6] 跳过：gpt2_124M_{fmt}.bin 不存在（量化未执行）")
            continue
        print(f"[{step}/6] 导出 vLLM {fmt.upper()} checkpoint"
              f"（W8A16 per-channel，与手写 {fmt.upper()} 同规格）...")
        out_dir = export_ct_ckpt(out_root, hf_dir, fmt)
        p = os.path.join(out_dir, "model.safetensors")
        print(f"  {os.path.basename(out_dir)}/model.safetensors {_human_size(os.path.getsize(p))}"
              f"（供 vllm_gpt2_bench.py 加载）")

    # 结尾打印后续构建命令，方便直接复制
    print()
    print(f"  cd {BASE_DIR}")
    print("  make clean && make")
    print("  ./inference_gpt2_opt assets/weights/gpt2_124M_int8.bin assets/weights/gpt2_tokenizer.bin 1000 --graph --int8")
    print("  # FP8 推理版（可选）：make PRECISION=FP8 && ./inference_gpt2_opt ...fp8.bin ... 1000 --graph --fp8")

if __name__ == "__main__":
    main()
