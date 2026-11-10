#!/usr/bin/env python3
"""检查 FP8 权重 .bin 文件的格式完整性（头部字段 + 各层数据尺寸）。

用法：
    python check_fp8_bin.py                  # 默认检查 <本脚本目录>/assets/weights/gpt2_124M_fp8.bin
    python check_fp8_bin.py 路径/别的.bin    # 检查任意 FP8 bin

文件由 quantize 生成：1024 字节头部（256 个 int32）+ 逐层 E4M3/FP32 数据。
"""
import argparse
import os
import struct


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_bin = os.path.join(base_dir, "assets", "weights", "gpt2_124M_fp8.bin")

    parser = argparse.ArgumentParser(
        description="检查 FP8 权重 .bin 文件（头部 + 各层数据尺寸）")
    parser.add_argument("path", nargs="?", default=default_bin,
                        help="FP8 .bin 文件路径（默认 <本脚本目录>/assets/weights/gpt2_124M_fp8.bin）")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"找不到文件：{args.path}\n"
              "先运行 make weights 或 python export_weights.py 生成 FP8 权重。")
        return 1

    with open(args.path, "rb") as f:
        hdr = struct.unpack("<256i", f.read(1024))
        print("magic", hdr[0], "ver", hdr[1], "Vp", hdr[7], "C", hdr[6],
              "L", hdr[4], "maxT", hdr[2])
        Vp, C, L, maxT = hdr[7], hdr[6], hdr[4], hdr[2]
        lay = [(Vp * C, 1), (maxT * C, 0), (L * C, 0), (L * C, 0),
               (L * 3 * C * C, 1), (L * 3 * C, 0), (L * C * C, 1), (L * C, 0),
               (L * C, 0), (L * C, 0), (L * 4 * C * C, 1), (L * 4 * C, 0),
               (L * C * 4 * C, 1), (L * C, 0), (C, 0), (C, 0)]
        tot = 0
        for i, (e, lin) in enumerate(lay):
            b = e * (1 if lin else 2)
            d = f.read(b)
            ok = len(d) == b
            print(f"layer {i}: elems={e} lin={lin} bytes={b} "
                  f"read={len(d)} {'OK' if ok else 'FAIL'}")
            tot += b
        print("total data", tot, "file data size", os.path.getsize(args.path) - 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
