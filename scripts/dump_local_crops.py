"""学習を回す前に、局所クロップが正しく切り出せているかを確認する。

データローダとクロップ抽出だけを動かし（ネットワークは作らない）、
  1. 実際に局所Dへ渡るクロップを bbox 枠つきのタイル画像として保存
  2. 採用/除外の内訳を集計して表示
する。学習オプションをそのまま渡せるので、本番と同じ設定で確認できる。

Usage:
    python scripts/dump_local_crops.py --dataroot DATA --dataset_mode unaligned_labeled \
        --preprocess crop --crop_size 256 --batch_size 4 --num_threads 0 \
        --lambda_local 1.0 --local_classes 1 --num_classes 10 --local_ctx_sizes 64,128 \
        --object_aware_crop_p 0.5 --n_batches 20 --out_dir ./local_crop_check
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import create_dataset  # noqa: E402
from models.cycle_gan_model import CycleGANModel  # noqa: E402
from options.train_options import TrainOptions  # noqa: E402
from util import util  # noqa: E402


class CropProbe:
    """CycleGANModel からクロップ抽出まわりのメソッドだけを借りた軽量シム。"""

    _extract_crops = CycleGANModel._extract_crops
    _local_crop_grid = CycleGANModel._local_crop_grid
    _draw_box = staticmethod(CycleGANModel._draw_box)
    _init_local_stats = CycleGANModel._init_local_stats
    _accumulate_local_stats = CycleGANModel._accumulate_local_stats
    _format_local_stats = CycleGANModel._format_local_stats

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cpu")
        self.local_ctx_sizes = sorted(int(v) for v in str(opt.local_ctx_sizes).split(",") if v.strip())
        self.local_class_ids = None
        if opt.local_classes.strip():
            ids = [int(c) for c in opt.local_classes.split(",") if c.strip()]
            self.local_class_ids = torch.tensor(ids, dtype=torch.long)
        self._init_local_stats()


def main():
    # 学習と同じオプションをそのまま受け取り、このスクリプト固有の2つだけ追加する
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--n_batches", type=int, default=20, help="確認に使うバッチ数")
    extra.add_argument("--out_dir", type=str, default="./local_crop_check", help="タイル画像の保存先")
    known, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    opt = TrainOptions().parse()
    opt.device = torch.device("cpu")
    os.makedirs(known.out_dir, exist_ok=True)

    dataset = create_dataset(opt)
    probe = CropProbe(opt)
    print(f"\n窓サイズ候補: {probe.local_ctx_sizes}  対象クラス: "
          f"{'全て' if probe.local_class_ids is None else probe.local_class_ids.tolist()}\n")

    for i, data in enumerate(dataset):
        if i >= known.n_batches:
            break
        # --direction を反映して real_A / real_B を決める（学習時の set_input と同じ規則）
        AtoB = opt.direction == "AtoB"
        real_A, real_B = data["A" if AtoB else "B"], data["B" if AtoB else "A"]
        ka, kb = ("labels_A", "labels_B") if AtoB else ("labels_B", "labels_A")
        labels_A, labels_B = data[ka], data[kb]

        # 統計は A 側 / B 側それぞれ 1 回だけ数える
        probe._extract_crops(real_A, labels_A, stats_tag="A")
        probe._extract_crops(real_B, labels_B, stats_tag="B")

        if i < 5:  # 先頭数バッチだけ画像を保存
            for name, imgs, labs in (("A", real_A, labels_A), ("B", real_B, labels_B)):
                grid = probe._local_crop_grid(imgs, labs)
                path = os.path.join(known.out_dir, f"batch{i:03d}_{name}.png")
                util.save_image(util.tensor2im(grid), path)
                print(f"  保存: {path}")

    probe.opt.local_log_freq = known.n_batches
    print("\n" + probe._format_local_stats(known.n_batches))
    print(f"\nタイル画像を {known.out_dir} に保存しました。")
    print("確認ポイント: 緑枠が各タイルの中央にあり、その中に信号機が丸ごと写っていること。")
    print("  枠の中が空 / 枠が信号機からずれている → ラベルか座標変換を疑う")
    print("  『画像外』の除外が多い → --object_aware_crop_p を上げるか --crop_size を大きくする")
    print("  『窓なし』の除外が多い → --local_ctx_sizes に大きい窓を足す")


if __name__ == "__main__":
    main()
