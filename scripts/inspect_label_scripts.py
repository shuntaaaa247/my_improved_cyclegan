"""YOLO ラベルの bbox 長辺分布を集計し、--local_ctx_sizes の決め方を示す。

Usage:
    python scripts/inspect_label_scales.py <labels_dir> [--img_size 640] [--cls 1] [--ctx_pad 16]
"""
import argparse
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_dir", help="YOLO ラベル(.txt)のルート（再帰的に探索）")
    parser.add_argument("--img_size", type=int, default=640, help="画像の1辺のピクセル数")
    parser.add_argument("--cls", type=int, default=None, help="このクラスIDだけ集計（既定は全クラス）")
    parser.add_argument("--ctx_pad", type=int, default=16, help="bbox周囲に確保するコンテキスト幅")
    parser.add_argument("--ctx_sizes", type=str, default="64,128,192", help="検討したい窓サイズ候補")
    opt = parser.parse_args()

    longs = []
    for path in glob.glob(os.path.join(opt.labels_dir, "**", "*.txt"), recursive=True):
        for line in open(path, encoding="utf-8"):
            parts = line.split()
            if len(parts) != 5:
                continue
            c, _, _, w, h = (float(v) for v in parts)
            if opt.cls is not None and int(c) != opt.cls:
                continue
            longs.append(max(w, h) * opt.img_size)

    if not longs:
        print("ラベルが見つかりませんでした")
        return
    longs = np.array(longs)
    print(f"bbox 数: {len(longs)}")
    print("長辺(px) の分位点: " + "  ".join(f"p{q}={np.percentile(longs, q):.0f}" for q in (10, 25, 50, 75, 90, 95, 99)))
    print(f"最大: {longs.max():.0f}")

    sizes = sorted(int(v) for v in opt.ctx_sizes.split(",") if v.strip())
    need = longs + opt.ctx_pad
    print("\n--local_ctx_sizes の候補ごとの内訳（窓px: 割り当てられるbboxの割合）")
    prev = 0
    for s in sizes:
        frac = ((need > prev) & (need <= s)).mean()
        print(f"  {s:4d}: {frac * 100:5.1f}%")
        prev = s
    print(f"  除外 (>{sizes[-1]}): {(need > sizes[-1]).mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
