import os
from data.base_dataset import BaseDataset, get_transform, get_params, load_yolo_txt, transform_yolo_labels
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
from torch.utils.data._utils.collate import default_collate


def yolo_collate_fn(batch):
    """可変長の YOLO ラベルを含むバッチをまとめる collate 関数。

    "labels_" で始まるキーだけを特別扱いし、(N_i, 5) を縦に連結して
    先頭に batch_idx 列を足した (総bbox数, 6) にする。
    それ以外のキー（A, B, A_paths, B_paths）は default_collate に任せる。

    Parameters:
        batch -- __getitem__ の戻り値 dict のリスト（長さ batch_size）
    Returns:
        dict -- labels_* が (M, 6) [batch_idx, cls, xc, yc, w, h] になったバッチ
    """
    label_keys = [k for k in batch[0] if k.startswith("labels_")]

    # ラベル以外は通常どおり stack / list 化
    collated = default_collate([{k: v for k, v in sample.items() if k not in label_keys}
                                for sample in batch])

    for key in label_keys:
        rows = []
        for i, sample in enumerate(batch):
            labels = sample[key]                                   # (N_i, 5)
            batch_idx = torch.full((labels.shape[0], 1), float(i), dtype=torch.float32)
            rows.append(torch.cat([batch_idx, labels], dim=1))     # (N_i, 6)
        collated[key] = torch.cat(rows, dim=0)                     # (ΣN_i, 6)

    return collated

class UnalignedLabeledDataset(BaseDataset):
    """
    このデータセットクラスはyoloラベル付きの unaligned/unpaired データセットを読み込む。

    このデータセットは学習のために2つの画像ディレクトリ
    ドメインA '/path/to/data/trainA', ドメインB '/path/to/data/trainB'
    と2つのラベルディレクトリ（こちらはラベルフラグ '--label_dir_A /your/original/path/to/labels_A, '--label_dir_B /your/original/path/to/labels_B' にて個別に指定可能）
    ドメインA '/path/to/data/trainA_labels', ドメインB '/path/to/data/trainB_labels'
    が必要です。
    また、テストの際には、テスト用の画像フォルダ
    '/path/to/data/testA', '/path/to/data/testB'
    が必要です。
    """

    collate_fn = staticmethod(yolo_collate_fn)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """このデータセット固有のコマンドラインオプションを追加する。"""
        parser.add_argument("--label_dir_A", type=str, default="",
                            help="A側ラベル(.txt)のルート。空なら <dataroot>/<phase>A_labels、それも無ければ画像と同じディレクトリ")
        parser.add_argument("--label_dir_B", type=str, default="",
                            help="B側ラベル(.txt)のルート。空なら <dataroot>/<phase>B_labels、それも無ければ画像と同じディレクトリ")
        parser.add_argument("--object_aware_crop_p", type=float, default=0.5,
                            help="この確率で、GTボックスが必ず入るようにクロップ位置を選ぶ（0で従来の一様ランダム）")
        parser.add_argument("--object_crop_margin", type=int, default=32,
                            help="object-aware crop時、物体中心をクロップ端から離す最小距離(px)。局所コンテキスト窓の半径 local_ctx_size//2 に合わせる")
        parser.add_argument("--min_visibility", type=float, default=0.3,
                            help="クロップで見切れたbboxを捨てる閾値（クロップ後面積 / クロップ前面積）")
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # モデル側の局所窓設定を共有する（--local_ctx_sizes / --local_ctx_pad は
        # cycle_gan_model が登録するが、パース後は同じ opt に載っている）。
        # object-aware crop のマージンをこれに合わせないと、大きいボックスを狙っても
        # 窓がクロップからはみ出してモデル側で捨てられる。
        raw = str(getattr(opt, "local_ctx_sizes", "") or "")
        self.ctx_sizes = sorted(int(v) for v in raw.split(",") if v.strip()) or None
        self.ctx_pad = int(getattr(opt, "local_ctx_pad", 16))

        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image

        # ここでself.transform_{A, B} を定義せずに、下の__getitem__関数で定義する
        # allined_dataset.pyのように、__getitem__関数内でget_params関数でクロップ位置などを取得し、それをparamsに指定した上でget_transformを呼ぶ & そのクロップ位置をラベルにも適用する
        # self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        # ラベルが格納されたディレクトリのパスを作成
        if opt.label_dir_A:
            self.dir_A_labels = opt.label_dir_A
        else:
            cand = os.path.join(opt.dataroot, opt.phase + "A_labels")
            self.dir_A_labels = cand if os.path.isdir(cand) else self.dir_A

        if opt.label_dir_B:
            self.dir_B_labels = opt.label_dir_B
        else:
            cand = os.path.join(opt.dataroot, opt.phase + "B_labels")
            self.dir_B_labels = cand if os.path.isdir(cand) else self.dir_B

        # 対応するラベルが見つからなかった場合、警告でその個数を表示する
        missing_A_labels = [p for p in self.A_paths
            if not os.path.isfile(self._label_path(p, self.dir_A, self.dir_A_labels))]
        missing_B_labels = [p for p in self.B_paths
            if not os.path.isfile(self._label_path(p, self.dir_B, self.dir_B_labels))]
        
        if missing_A_labels:
            print(f"[warning] ドメインA側で {len(missing_A_labels)}/{len(self.A_paths)} 件のラベルが見つかりません "
                f"(例: {self._label_path(missing_A_labels[0], self.dir_A, self.dir_A_labels)})")

        if missing_B_labels:
            print(f"[warning] ドメインB側で {len(missing_B_labels)}/{len(self.B_paths)} 件のラベルが見つかりません "
                f"(例: {self._label_path(missing_B_labels[0], self.dir_B, self.dir_B_labels)})")

    def __getitem__(self, index):
        """1サンプル分の画像とラベルを返す。

        画像に施した幾何変換（クロップ位置・左右反転）と全く同じものをラベルにも適用する。
        またクロップ位置は、opt.object_aware_crop_p の確率で GT ボックスを含むように選ぶ。

        Parameters:
            index (int) -- データのインデックス

        Returns:
            dict -- A, B (画像テンソル), A_paths, B_paths (パス),
                    labels_A, labels_B ((N, 5) tensor [cls, xc, yc, w, h] 変換後の正規化座標)
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")

        # 以下 data/aligned_dataset.py を参考に実装

        # ラベルを先に読む。object-aware crop sampling がラベルを必要とするため。
        labels_A = load_yolo_txt(self._label_path(A_path, self.dir_A, self.dir_A_labels))
        labels_B = load_yolo_txt(self._label_path(B_path, self.dir_B, self.dir_B_labels))

        # 画像とラベルに適用するクロップ位置・フリップを統一するため取得する
        # A側B側の両方に同じサンプリングポリシーを適用する（片側だけだとクロップ位置の
        # 分布がドメイン間でずれ、大域Dがそれを手がかりにしてしまう）
        transform_params_A = get_params(self.opt, A_img.size, labels=labels_A,
                                        object_aware_p=self.opt.object_aware_crop_p,
                                        object_margin=self.opt.object_crop_margin,
                                        ctx_sizes=self.ctx_sizes, ctx_pad=self.ctx_pad)
        transform_params_B = get_params(self.opt, B_img.size, labels=labels_B,
                                        object_aware_p=self.opt.object_aware_crop_p,
                                        object_margin=self.opt.object_crop_margin,
                                        ctx_sizes=self.ctx_sizes, ctx_pad=self.ctx_pad)

        # トランスフォームを適用する
        transform_A = get_transform(self.opt, params=transform_params_A, grayscale=(self.input_nc == 1))
        transform_B = get_transform(self.opt, params=transform_params_B, grayscale=(self.output_nc == 1))

        A = transform_A(A_img)
        B = transform_B(B_img)

        # ラベルに画像と同じ幾何変換を適用する
        labels_A = transform_yolo_labels(labels_A, transform_params_A, A_img.size,
                                         self.opt, min_visibility=self.opt.min_visibility)
        labels_B = transform_yolo_labels(labels_B, transform_params_B, B_img.size,
                                         self.opt, min_visibility=self.opt.min_visibility)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path,
                "labels_A": torch.from_numpy(labels_A), "labels_B": torch.from_numpy(labels_B)}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def _label_path(self, img_path, img_dir, label_dir):
        """画像パス -> 対応する .txt パス（サブディレクトリ構造を維持）を返す
        
        Parameters:
            img_path    --  ラベルに対応する画像のパス
            img_dir     --  画像が格納されるディレクトリのパス
            label_dir   --  ラベルが格納されるディレクトリのパス
        Returns:
            入力画像に対応するラベルのtxtファイルのパス
        """
        rel = os.path.relpath(img_path, img_dir)          # 'seq01/0001.jpg'
        return os.path.join(label_dir, os.path.splitext(rel)[0] + ".txt")