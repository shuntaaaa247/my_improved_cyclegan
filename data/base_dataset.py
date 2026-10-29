"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""

import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_transformed_size(opt, size):
    """get_transform と同じ規則で「リサイズ直後のサイズ」と「最終出力サイズ」を返す。

    ラベル側の座標変換を画像側と厳密に一致させるために使う。

    Parameters:
        opt     -- オプション（preprocess / load_size / crop_size を参照）
        size    -- 元画像サイズ (w, h)
    Returns:
        (rw, rh, ow, oh) -- リサイズ後の (幅, 高さ) と、クロップ後の最終 (幅, 高さ)
    """
    w, h = size

    # 1) リサイズ系（get_transform の resize / scale_width / none 分岐に対応）
    if "resize" in opt.preprocess:
        rw = rh = opt.load_size
    elif "scale_width" in opt.preprocess:
        if w == opt.load_size and h >= opt.crop_size:  # __scale_width の早期 return と揃える
            rw, rh = w, h
        else:
            rw = opt.load_size
            rh = int(max(opt.load_size * h / w, opt.crop_size))
    elif opt.preprocess == "none":
        rw = int(round(w / 4) * 4)  # __make_power_2(base=4)
        rh = int(round(h / 4) * 4)
    else:  # '--preprocess crop' 等、リサイズなし
        rw, rh = w, h

    # 2) クロップ（__crop は画像が crop_size 以下ならそのまま返す）
    if "crop" in opt.preprocess and (rw > opt.crop_size or rh > opt.crop_size):
        return rw, rh, opt.crop_size, opt.crop_size
    return rw, rh, rw, rh


def _sample_crop_origin(p, max_origin, crop_size, margin):
    """点 p が [origin + margin, origin + crop_size - margin] に入る origin を一様サンプルする。

    Parameters:
        p           -- クロップに含めたい点の座標（リサイズ後ピクセル、1 軸分）
        max_origin  -- クロップ原点の最大値（= リサイズ後サイズ - crop_size）
        crop_size   -- クロップサイズ
        margin      -- 点をクロップ端から最低これだけ離す（局所コンテキスト窓の半径を想定）
    Returns:
        int -- クロップ原点
    """
    lo = int(np.clip(np.ceil(p + margin - crop_size), 0, max_origin))
    hi = int(np.clip(np.floor(p - margin), 0, max_origin))
    if lo > hi:  # margin が crop_size に対して大きすぎる等の異常時の保険
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _select_target_box(labels, rw, rh, crop_size, ctx_sizes, ctx_pad):
    """object-aware crop の狙い先ボックスを1つ選び、(中心x, 中心y, 必要margin) を返す。

    「局所損失に実際に使えるボックス」だけを候補にするのが要点。最大窓に収まらない
    ボックスや、窓がクロップに収まりきらないボックスを狙っても、モデル側の
    _extract_crops で除外されるだけで object-aware crop が空振りする。

    Parameters:
        labels      -- (N, 5) 正規化 YOLO ラベル
        rw, rh      -- リサイズ後の画像サイズ
        crop_size   -- クロップサイズ
        ctx_sizes   -- 局所窓サイズの候補（昇順）。None なら窓サイズ = 必要サイズ
        ctx_pad     -- bbox 周囲に確保するコンテキスト幅
    Returns:
        (xc, yc, margin) or None -- 候補が無ければ None
    """
    cands = []
    for row in labels:
        need = max(row[3] * rw, row[4] * rh) + ctx_pad
        if ctx_sizes:
            fits = [s for s in ctx_sizes if s >= need]
            if not fits:
                continue  # 最大窓に収まらない = 局所損失では使えない
            win = fits[0]
        else:
            win = need
        margin = int(win / 2) + 1
        if 2 * margin > crop_size:
            continue  # 窓がクロップに収まらない
        cands.append((row[1], row[2], margin))
    return random.choice(cands) if cands else None


def get_params(opt, size, labels=None, object_aware_p=0.0, object_margin=32, ctx_sizes=None, ctx_pad=16):
    """画像とラベルに共通して適用する幾何変換パラメータ（クロップ位置・左右反転）を決める。

    labels を渡して object_aware_p > 0 にすると、その確率で「GT ボックスを 1 つ選び、
    その局所コンテキスト窓ごと確実にクロップ内に入る」ように原点をサンプルする
    （object-aware crop sampling）。信号機のような小物体は素のランダムクロップだと
    高頻度で切り落とされ、局所損失に使えるサンプルが枯れてしまうため。

    A 側 / B 側の両方に同じポリシーを適用すること。片側だけだとクロップ位置の分布が
    ドメイン間でずれ、大域 D が「物体の有無」でドメインを見分けるショートカットを学習する。

    Parameters:
        opt             -- オプション
        size            -- 元画像サイズ (w, h)
        labels          -- (N, 5) [cls, xc, yc, w, h] 正規化 YOLO ラベル。None なら一様ランダム
        object_aware_p  -- 物体を含むようにクロップする確率（0 で従来どおり一様ランダム）
        object_margin   -- 物体中心をクロップ端から離す最小距離(px)。実際にはその物体に
                           割り当てられる窓サイズの半分まで自動的に引き上げられる
        ctx_sizes       -- 局所窓サイズの候補（昇順）。モデル側の --local_ctx_sizes と揃えること
        ctx_pad         -- bbox 周囲に確保するコンテキスト幅。--local_ctx_pad と揃えること
    Returns:
        dict -- {"crop_pos": (x, y), "flip": bool}
    """
    rw, rh, _, _ = get_transformed_size(opt, size)

    max_x = int(np.maximum(0, rw - opt.crop_size))
    max_y = int(np.maximum(0, rh - opt.crop_size))

    target = None
    if labels is not None and len(labels) > 0 and random.random() < object_aware_p:
        target = _select_target_box(labels, rw, rh, opt.crop_size, ctx_sizes, ctx_pad)

    if target is not None:
        xc, yc, margin = target
        margin = max(object_margin, margin)
        x = _sample_crop_origin(xc * rw, max_x, opt.crop_size, margin)
        y = _sample_crop_origin(yc * rh, max_y, opt.crop_size, margin)
    else:
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if "resize" in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif "scale_width" in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if "crop" in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params["crop_pos"], opt.crop_size)))

    if opt.preprocess == "none":
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params["flip"]:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params["flip"])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __transforms2pil_resize(method):
    mapper = {
        transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
        transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
        transforms.InterpolationMode.NEAREST: Image.NEAREST,
        transforms.InterpolationMode.LANCZOS: Image.LANCZOS,
    }
    return mapper[method]


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print("The image size needs to be a multiple of 4. " "The loaded image size was (%d, %d), so it was adjusted to " "(%d, %d). This adjustment will be done to all images " "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

def load_yolo_txt(path, num_fields=5):
    """YOLO 形式のラベルファイルを読む。

    Parameters:
        path       -- YOLO 形式のラベルファイルのパス
        num_fields -- ラベルファイル1行に含まれる要素数

    Returns:
        (N, 5) float32 ndarray -- [cls, xc, yc, w, h]（すべて正規化座標）
        ファイルが存在しない / 中身が空なら (0, 5) を返す（背景のみの画像は正当なサンプル）
    """
    if not os.path.isfile(path):
        return np.zeros((0, num_fields), dtype=np.float32)

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != num_fields:
                raise ValueError(
                    f"{path}:{lineno}: expected {num_fields} fields "
                    f"[cls xc yc w h], got {len(parts)}: {line!r}"
                )
            try:
                rows.append([float(v) for v in parts])
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: failed to parse {line!r}") from e

    if not rows:
        return np.zeros((0, num_fields), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)

def transform_yolo_labels(labels, params, orig_size, opt, min_visibility=0.3):
    """YOLO ラベルに、画像と同一の幾何変換（リサイズ → クロップ → 左右反転）を適用する。

    get_transform / get_params と厳密に対応させてある。特に params["crop_pos"] は
    「リサイズ後」の座標系である点に注意（--preprocess resize_and_crop で元画像座標の
    まま引き算するとボックスが大きくずれる）。--preprocess crop ではリサイズが無いので
    リサイズ後 == 元画像座標になる。

    Parameters:
        labels          -- (N, 5) ndarray [cls, xc, yc, w, h]（YOLO 正規化形式）
        params          -- get_params の戻り値（crop_pos, flip）
        orig_size       -- 元画像サイズ (w, h)
        opt             -- オプション
        min_visibility  -- クロップ後に残った面積の割合がこれ以下の bbox は捨てる
    Returns:
        (M, 5) ndarray -- 変換後の座標系での同形式ラベル
    """
    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    rw, rh, ow, oh = get_transformed_size(opt, orig_size)

    cls = labels[:, 0]
    # 正規化 → リサイズ後ピクセルの xyxy
    xc, yc = labels[:, 1] * rw, labels[:, 2] * rh
    bw, bh = labels[:, 3] * rw, labels[:, 4] * rh
    x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
    area_before = np.maximum(bw * bh, 1e-6)

    # クロップ原点を引いて枠内にクリップ（実際にクロップが行われた場合のみ）
    if (ow, oh) != (rw, rh):
        cx, cy = params["crop_pos"]
        x1, x2 = np.clip(x1 - cx, 0, ow), np.clip(x2 - cx, 0, ow)
        y1, y2 = np.clip(y1 - cy, 0, oh), np.clip(y2 - cy, 0, oh)

    # 見切れすぎた bbox を除去
    keep = ((x2 - x1) * (y2 - y1)) / area_before > min_visibility
    cls, x1, y1, x2, y2 = cls[keep], x1[keep], y1[keep], x2[keep], y2[keep]
    if len(cls) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    # 出力サイズで再正規化
    xc, yc = (x1 + x2) / 2 / ow, (y1 + y2) / 2 / oh
    bw, bh = (x2 - x1) / ow, (y2 - y1) / oh

    # 左右反転（get_transform の __flip と条件を揃える）
    if not opt.no_flip and params["flip"]:
        xc = 1.0 - xc

    return np.stack([cls, xc, yc, bw, bh], axis=1).astype(np.float32)
