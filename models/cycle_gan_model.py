import torch
import torch.distributed as dist
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_B", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")
            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
            )
            # ---- 局所 (物体領域) GAN 損失 ----
            # 0 にすると局所Dを一切作らないので、素の CycleGAN と完全に同じ挙動になる（ablation のベースライン）
            parser.add_argument("--lambda_local", type=float, default=0.0, help="物体領域クロップに対するGAN損失の重み。0で無効（=素のCycleGAN）")
            parser.add_argument("--local_ctx_sizes", type=str, default="64,128,192", help="局所コンテキスト窓のサイズ候補(px, カンマ区切り)。各bboxには「長辺+local_ctx_pad 以上で最小の窓」が割り当てられ、最大値を超えるbboxは局所損失から除外される。'64' のみ指定すれば単一スケール")
            parser.add_argument("--local_d_size", type=int, default=64, help="局所Dへの入力サイズ(px)。各窓をこのサイズにリサイズして与える。局所Dはスケールバケットで条件付けされるのでスケール統計は混ざらない")
            parser.add_argument("--local_ctx_pad", type=int, default=16, help="bboxの周囲に最低限確保するコンテキストの幅(px, 上下左右合計)。夜間のハローはbboxの外側に広がるため必要")
            parser.add_argument("--local_max_crops", type=int, default=32, help="1ステップあたりに使うクロップ数の上限（メモリ対策）")
            parser.add_argument("--local_classes", type=str, default="", help="局所損失を適用するクラスIDをカンマ区切りで指定（例 '1'）。空なら全クラス")
            parser.add_argument("--local_n_layers_D", type=int, default=2, help="局所Dのconv層数。ctx_size=64 なら 2（受容野34程度）が目安")
            parser.add_argument("--local_ndf", type=int, default=64, help="局所Dの最初のconvのフィルタ数")
            parser.add_argument("--local_norm", type=str, default="none", help="局所Dの正規化層 [batch | instance | none]。instanceは絶対輝度を消すので自発光物体には不適")
            parser.add_argument("--num_classes", type=int, default=80, help="YOLOラベルのクラス数（class-conditional局所Dの埋め込みサイズ）")
            parser.add_argument("--local_log_freq", type=int, default=100, help="この反復数ごとに、局所クロップの採用/除外内訳をログ出力する。0で無効")
            parser.add_argument("--local_vis_max", type=int, default=32, help="可視化に載せる局所クロップの最大枚数")
            parser.add_argument("--no_local_cond", action="store_true", help="局所Dのクラス/スケール条件付けを両方とも無効化する（ablation用）。個別フラグより優先")
            parser.add_argument("--no_local_cls_cond", action="store_true", help="局所Dのクラス条件付けだけを無効化する（ablation用）")
            parser.add_argument("--no_local_scale_cond", action="store_true", help="局所Dのスケール条件付けだけを無効化する（ablation用）")

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]
        # 局所 (物体領域) GAN 損失を使うか
        self.use_local = self.isTrain and getattr(opt, "lambda_local", 0.0) > 0.0
        self.local_class_ids = None  # 局所損失を適用するクラスID（下で設定）
        # 局所窓サイズの候補（昇順）。小さいものから順に割り当てるため必ずソートする
        self.local_ctx_sizes = sorted(int(v) for v in str(getattr(opt, "local_ctx_sizes", "64")).split(",") if v.strip()) if self.use_local else [64]
        if self.use_local:
            self.loss_names += ["D_A_local", "G_A_local", "D_B_local", "G_B_local"]
            self._init_local_stats()
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_B(A)
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        if self.use_local:
            # 実際に局所Dへ渡っているクロップをそのまま可視化する。
            # 「信号機が窓の中心に、丸ごと入っているか」を目で確認するのが目的。
            self.visual_names += ["local_crops_fake_B", "local_crops_real_B"]
            self.local_crops_fake_B = None
            self.local_crops_real_B = None
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
            if self.use_local:
                self.model_names += ["D_A_local", "D_B_local"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            if self.use_local:
                # D_A_local: 夜(B)ドメインの物体クロップ用 / D_B_local: 昼(A)ドメインの物体クロップ用
                n_scales = len(self.local_ctx_sizes)
                # --no_local_cond は両方を無効化し、個別フラグはそれぞれ片方だけを無効化する
                use_cls_cond = not (opt.no_local_cond or opt.no_local_cls_cond)
                use_scale_cond = not (opt.no_local_cond or opt.no_local_scale_cond)
                if use_scale_cond and n_scales == 1:
                    # バケットが1つだとスケール項は f の線形関数になり、無条件項 psi と冗長で効果がない
                    print("Warning: --local_ctx_sizes が単一スケールのため、局所Dのスケール条件付けは実質的に無効です")
                print("局所D条件付け: class=%s, scale=%s" % (use_cls_cond, use_scale_cond))
                self.netD_A_local = networks.define_local_D(opt.output_nc, opt.local_ndf, opt.local_n_layers_D, opt.local_norm, opt.init_type, opt.init_gain, opt.num_classes, n_scales, use_cls_cond, use_scale_cond)
                self.netD_B_local = networks.define_local_D(opt.input_nc, opt.local_ndf, opt.local_n_layers_D, opt.local_norm, opt.init_type, opt.init_gain, opt.num_classes, n_scales, use_cls_cond, use_scale_cond)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert opt.input_nc == opt.output_nc
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            d_params = [self.netD_A.parameters(), self.netD_B.parameters()]
            if self.use_local:
                d_params += [self.netD_A_local.parameters(), self.netD_B_local.parameters()]
            self.optimizer_D = torch.optim.Adam(itertools.chain(*d_params), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # 局所損失を適用するクラスIDの集合（空なら全クラス）
            self.local_class_ids = None
            if self.use_local and opt.local_classes.strip():
                ids = [int(c) for c in opt.local_classes.split(",") if c.strip()]
                self.local_class_ids = torch.tensor(ids, dtype=torch.long, device=self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]
        # YOLO ラベル: collate 後は (M, 6) = [batch_idx, cls, xc, yc, w, h]（クロップ後画像の正規化座標）
        # real_A / real_B に対応する側を取る（--direction BtoA のとき入れ替わる点に注意）
        key_a, key_b = ("labels_A", "labels_B") if AtoB else ("labels_B", "labels_A")
        # ラベルを返さないデータセットと --lambda_local > 0 の組み合わせは、
        # 例外を出さずに局所損失が常に 0 になるだけなので設定ミスに気づけない。
        # 最初のバッチで落として、無駄な学習を回さないようにする。
        if self.use_local and (key_a not in input or key_b not in input):
            raise ValueError(
                f"--lambda_local {self.opt.lambda_local} が指定されていますが、"
                f"--dataset_mode {self.opt.dataset_mode} は YOLO ラベルを返しません。"
                " ラベル付きデータセット（--dataset_mode unaligned_labeled）を使うか、"
                " --lambda_local 0 で局所損失を無効にしてください。"
            )
        self.labels_A = input[key_a].to(self.device) if key_a in input else None
        self.labels_B = input[key_b].to(self.device) if key_b in input else None

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def _extract_crops(self, images, labels, stats_tag=None, return_geom=False):
        """画像バッチと YOLO ラベルから、局所コンテキスト窓を切り出す。

        各 bbox には「長辺 + local_ctx_pad 以上で最小の窓サイズ」を割り当てる（スケールバケット）。
        bbox にぴったり合わせず余白付きの窓を切るのは、夜間のハロー/ブルームが bbox の外側に
        広がる現象だから。切り出した窓は local_d_size にリサイズして局所Dへ渡すが、
        バケットIDを条件として一緒に与えるので、スケールごとの統計（あるべきボケ量）は混ざらない。

        最大窓サイズでも収まらない bbox は除外する（--local_ctx_sizes で制御）。

        Parameters:
            images (tensor)  -- (B, C, H, W)
            labels (tensor)  -- (M, 6) [batch_idx, cls, xc, yc, w, h]（正規化座標）
            stats_tag (str)  -- "A"/"B" を渡すと採用/除外の内訳を self.local_stats に加算する。
                                None なら統計を取らない（同じラベルを二重計上しないため）
            return_geom      -- True なら4つ目に (K, 2) の「窓に対する bbox の相対サイズ」を返す。
                                可視化で bbox 枠を描くのに使う
        Returns:
            (crops, cls, scale[, geom]) -- crops: (K, C, d, d), cls: (K,), scale: (K,)
                                           該当なしは (None, None, None[, None])
        """
        empty = (None, None, None, None) if return_geom else (None, None, None)
        if labels is None or labels.numel() == 0:
            return empty

        sizes = self.local_ctx_sizes
        d = self.opt.local_d_size
        _, _, H, W = images.shape

        bidx = labels[:, 0].long()
        cls = labels[:, 1].long()
        xc, yc = labels[:, 2] * W, labels[:, 3] * H
        bw, bh = labels[:, 4] * W, labels[:, 5] * H

        # 各 bbox に割り当てる窓サイズ = 「長辺 + pad」以上で最小のもの（無ければ -1 = 除外）
        need = torch.max(bw, bh) + self.opt.local_ctx_pad
        win = torch.full_like(need, -1.0)
        bucket = torch.full_like(need, -1.0)
        for bi in range(len(sizes) - 1, -1, -1):  # 大きい方から上書きしていくと最小のものが残る
            fits = need <= sizes[bi]
            win = torch.where(fits, torch.full_like(win, float(sizes[bi])), win)
            bucket = torch.where(fits, torch.full_like(bucket, float(bi)), bucket)
        # 除外理由を切り分けて数える（class → window → bounds の優先順で排他的に集計）
        class_ok = torch.ones_like(win, dtype=torch.bool) if self.local_class_ids is None else torch.isin(cls, self.local_class_ids)
        has_win = win > 0
        keep = has_win & class_ok

        # 窓が画像内に完全に収まるものだけ使う。はみ出す分を中心をずらして詰めると
        # 「窓の中心に物体がある」という前提が壊れ、Dが学習しにくくなる。
        wi = win.long()
        x0 = torch.round(xc - win / 2).long()
        y0 = torch.round(yc - win / 2).long()
        in_bounds = (x0 >= 0) & (y0 >= 0) & (x0 + wi <= W) & (y0 + wi <= H)
        keep = keep & in_bounds

        idx = torch.nonzero(keep, as_tuple=False).flatten()
        n_cand = int(idx.numel())
        if n_cand > self.opt.local_max_crops:  # メモリ対策のランダムサブサンプル
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[: self.opt.local_max_crops]]

        if stats_tag is not None:
            st = self.local_stats[stats_tag]
            st["total"] += int(labels.shape[0])
            st["class_skip"] += int((~class_ok).sum())
            st["no_window"] += int((class_ok & ~has_win).sum())
            st["out_of_bounds"] += int((class_ok & has_win & ~in_bounds).sum())
            st["subsampled"] += n_cand - int(idx.numel())
            st["used"] += int(idx.numel())
            for bi in bucket[idx].long().tolist():
                self.local_stats[stats_tag]["bucket"][bi] += 1

        if idx.numel() == 0:
            return empty

        bucket = bucket.long()
        bl, xl, yl, wl = bidx.tolist(), x0.tolist(), y0.tolist(), wi.tolist()
        crops, order = [], []
        for bi, s_win in enumerate(sizes):  # バケットごとにまとめて切り出し・リサイズ
            sel = idx[bucket[idx] == bi].tolist()
            if not sel:
                continue
            c = torch.stack([images[bl[k], :, yl[k] : yl[k] + wl[k], xl[k] : xl[k] + wl[k]] for k in sel], dim=0)  # 実際に1枚ずつクロップし、今回のsel分のクロップをまとめる。
            if s_win != d:
                # 縮小時は antialias を入れる。エイリアシング由来の高周波は real/fake の
                # 見分けがつく人工的な手がかりになり、Gがそれを再現しようとして破綻する。
                c = torch.nn.functional.interpolate(c, size=(d, d), mode="bilinear", align_corners=False, antialias=(s_win > d))
            crops.append(c)
            order += sel
        order = torch.tensor(order, dtype=torch.long, device=labels.device)
        out = (torch.cat(crops, dim=0), cls[order], bucket[order])
        if return_geom:
            # 窓に対する bbox の相対サイズ。クロップは中心合わせなので、これだけで枠が描ける
            geom = torch.stack([bw[order] / win[order], bh[order] / win[order]], dim=1)
            return out + (geom,)
        return out

    # ------------------------------------------------------------------
    # 局所クロップの診断ログ / 可視化
    # ------------------------------------------------------------------
    def _init_local_stats(self):
        """採用・除外の内訳カウンタを初期化する。

        window = 直近 --local_log_freq 反復分、total = 学習開始からの累積。
        「信号機がちゃんと切り取れているか」は、この採用率と除外理由の内訳を見れば分かる。

        d_step / d_skip は局所Dの更新回数と、real/fake の片側が0件でスキップした回数。
        used はウィンドウ内の合算なので「採用率100%」に見えても個別の反復では0件が
        混ざりうる。スキップ率が高いなら、局所Dがそもそも十分に更新されていない。
        """
        self._stat_keys = ["total", "class_skip", "no_window", "out_of_bounds", "subsampled", "used", "d_step", "d_skip"]
        n_bucket = len(self.local_ctx_sizes)
        make = lambda: {t: dict({k: 0 for k in self._stat_keys}, bucket=[0] * n_bucket) for t in ("A", "B")}
        self.local_stats = make()          # 直近ウィンドウ
        self.local_stats_total = make()    # 累積
        self.local_log_iter = 0

    def _accumulate_local_stats(self):
        """ウィンドウ分の統計を累積へ足し込み、ウィンドウをゼロに戻す。"""
        for t in ("A", "B"):
            for k in self._stat_keys:
                self.local_stats_total[t][k] += self.local_stats[t][k]
                self.local_stats[t][k] = 0
            for i in range(len(self.local_ctx_sizes)):
                self.local_stats_total[t]["bucket"][i] += self.local_stats[t]["bucket"][i]
                self.local_stats[t]["bucket"][i] = 0

    def _sync_local_stats(self):
        """DDP のとき全ランクの統計を合算する。

        all_reduce は全ランクが呼ばないとデッドロックするので、rank0 判定より前に実行する。
        """
        if not dist.is_initialized():
            return
        flat = []
        for t in ("A", "B"):
            flat += [self.local_stats[t][k] for k in self._stat_keys] + self.local_stats[t]["bucket"]
        buf = torch.tensor(flat, dtype=torch.long, device=self.device)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        vals = buf.tolist()
        n_bucket = len(self.local_ctx_sizes)
        i = 0
        for t in ("A", "B"):
            for k in self._stat_keys:
                self.local_stats[t][k] = vals[i]; i += 1
            self.local_stats[t]["bucket"] = vals[i : i + n_bucket]; i += n_bucket

    def _format_local_stats(self, total_iters):
        """ログ1回分の文字列を組み立てる。"""
        names = {"A": "A→B (昼→夜)", "B": "B→A (夜→昼)"}
        lines = [f"[local-crop] iter {total_iters} (直近 {self.opt.local_log_freq} 反復)"]
        for t in ("A", "B"):
            w, c = self.local_stats[t], self.local_stats_total[t]
            rate = w["used"] / w["total"] * 100 if w["total"] else 0.0
            # 累積は「これまでの累積 + 今回のウィンドウ」。累積への足し込みは出力後に行うため
            c_used, c_total = c["used"] + w["used"], c["total"] + w["total"]
            crate = c_used / c_total * 100 if c_total else 0.0
            lines.append(
                f"  {names[t]}: ラベル {w['total']} → 採用 {w['used']} ({rate:.1f}%)"
                f" | 除外: クラス外 {w['class_skip']}, 窓なし {w['no_window']},"
                f" 画像外 {w['out_of_bounds']}, 間引き {w['subsampled']}"
                f" | 累積採用率 {crate:.1f}%"
            )
            buckets = "  ".join(f"{sz}px={n}" for sz, n in zip(self.local_ctx_sizes, w["bucket"]))
            lines.append(f"      バケット内訳: {buckets}")
            srate = w["d_skip"] / w["d_step"] * 100 if w["d_step"] else 0.0
            lines.append(
                f"      局所D更新: {w['d_step'] - w['d_skip']}/{w['d_step']} 回実行"
                f" (スキップ {w['d_skip']} 回 = {srate:.1f}%, 片側のクロップが0件)"
            )
        return "\n".join(lines)

    def log_local_stats(self, total_iters):
        """局所クロップの採用状況を標準出力と local_crop_log.txt に書き出す。"""
        self._sync_local_stats()  # 全ランクで呼ぶ必要がある
        if not dist.is_initialized() or dist.get_rank() == 0:
            text = self._format_local_stats(total_iters)
            print(text)
            with open(self.save_dir / "local_crop_log.txt", "a", encoding="utf-8") as f:
                f.write(text + "\n")
        self._accumulate_local_stats()

    @staticmethod
    def _draw_box(tile, rel):
        """クロップ 1 枚に bbox の枠を緑で描く（画像は [-1, 1] スケール）。

        クロップは物体中心に合わせて切ってあるので、枠は必ず中央に来る。
        枠が中央からずれていたり、枠の中に信号機が無ければ、ラベルか座標変換がおかしい。
        """
        c, d, _ = tile.shape
        hw = float(rel[0]) * d / 2.0
        hh = float(rel[1]) * d / 2.0
        x1, x2 = max(0, int(round(d / 2 - hw))), min(d - 1, int(round(d / 2 + hw)))
        y1, y2 = max(0, int(round(d / 2 - hh))), min(d - 1, int(round(d / 2 + hh)))
        green = torch.tensor([-1.0, 1.0, -1.0], device=tile.device) if c == 3 else torch.tensor([1.0], device=tile.device)
        g = green.view(c, 1)
        tile[:, y1, x1 : x2 + 1] = g
        tile[:, y2, x1 : x2 + 1] = g
        tile[:, y1 : y2 + 1, x1] = g
        tile[:, y1 : y2 + 1, x2] = g
        return tile

    def _local_crop_grid(self, images, labels):
        """局所Dに渡っているクロップをタイル状に並べた 1 枚の画像 (1, C, H, W) を返す。

        tensor2im はバッチ先頭だけを見るので、グリッドは 1 枚に畳んでから返す。
        """
        c_in = images.shape[1]
        d = self.opt.local_d_size
        with torch.no_grad():
            crops, _, _, geom = self._extract_crops(images.detach(), labels, return_geom=True)
        if crops is None:  # このバッチには使えるクロップが無かった場合
            return torch.full((1, c_in, d, d), -1.0, device=images.device)

        crops = crops[: self.opt.local_vis_max].clone()
        geom = geom[: self.opt.local_vis_max]
        k = crops.shape[0]
        ncol = min(8, k)
        nrow = (k + ncol - 1) // ncol
        pad = 2
        cell = d + pad
        canvas = torch.full((c_in, nrow * cell + pad, ncol * cell + pad), -1.0, device=crops.device, dtype=crops.dtype)
        for i in range(k):
            r, c = divmod(i, ncol)
            y, x = pad + r * cell, pad + c * cell
            canvas[:, y : y + d, x : x + d] = self._draw_box(crops[i], geom[i])
        return canvas.unsqueeze(0)

    def compute_visuals(self):
        """display_freq のタイミングで train.py から呼ばれる。局所クロップの可視化を作る。"""
        if not self.use_local:
            return
        # fake_B の幾何は real_A に従うので labels_A、real_B は labels_B を使う
        self.local_crops_fake_B = self._local_crop_grid(self.fake_B, self.labels_A)
        self.local_crops_real_B = self._local_crop_grid(self.real_B, self.labels_B)

    def _local_gan_loss(self, netD, crops, cls, scale, target_is_real):
        """局所Dの GAN 損失。クロップが 0 個のときも安全に 0 を返す。

        クロップ0個のときにスカラー 0 を返すと、DDP でそのランクだけ netD_*_local の
        パラメータが unused になり "Expected to have finished reduction..." で落ちる。
        そこでダミーのゼロ画像で forward だけ通し、損失を 0 倍して勾配を殺す。
        """
        if crops is None:
            d = self.opt.local_d_size
            in_nc = self.opt.output_nc if netD is self.netD_A_local else self.opt.input_nc
            crops = torch.zeros(1, in_nc, d, d, device=self.device)
            cls = torch.zeros(1, dtype=torch.long, device=self.device)
            scale = torch.zeros(1, dtype=torch.long, device=self.device)
            return self.criterionGAN(netD(crops, cls, scale), target_is_real) * 0.0
        return self.criterionGAN(netD(crops, cls, scale), target_is_real)

    def backward_D_local_basic(self, netD, real_images, real_labels, fake_images, fake_labels, stats_tag=None):
        """局所Dの更新。real は本物画像の物体クロップ、fake は生成画像の物体クロップ。

        生成側のクロップ位置は「生成元画像のラベル」から取る点が重要
        （fake_B = G_A(real_A) なので幾何は real_A に従う）。
        画像プールは使わない（1ステップあたりのクロップ数が多く、実害が小さいため）。

        real / fake のどちらか一方だけクロップが 0 個の反復では、両方をダミー化して
        損失を 0 にする。片側だけで更新すると「常に fake と答えろ」（または逆）方向にしか
        勾配が入らず、Dが偏って G への信号が意味を失うため。物体が写っていない画像は
        データ中に普通に存在するので、これは恒常的に起きる。

        Parameters:
            stats_tag (str) -- "A"/"B" を渡すと更新回数とスキップ回数を self.local_stats に
                               加算する。スキップは損失ログの D_*_local が 0.000 になる以外に
                               痕跡が残らないので、頻度を明示的に数えておく
        """
        real_crops, real_cls, real_scale = self._extract_crops(real_images, real_labels)
        fake_crops, fake_cls, fake_scale = self._extract_crops(fake_images.detach(), fake_labels)
        skipped = real_crops is None or fake_crops is None
        if skipped:
            real_crops = fake_crops = None  # 両方ダミーにして勾配だけ通す（DDP対策）
        if stats_tag is not None:
            self.local_stats[stats_tag]["d_step"] += 1
            self.local_stats[stats_tag]["d_skip"] += int(skipped)
        loss_real = self._local_gan_loss(netD, real_crops, real_cls, real_scale, True)
        loss_fake = self._local_gan_loss(netD, fake_crops, fake_cls, fake_scale, False)
        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        if self.use_local:
            # 本物 = 実夜間画像の信号機クロップ、偽物 = 昼→夜変換した画像の信号機クロップ
            self.loss_D_A_local = self.backward_D_local_basic(self.netD_A_local, self.real_B, self.labels_B, self.fake_B, self.labels_A, stats_tag="A")

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        if self.use_local:
            self.loss_D_B_local = self.backward_D_local_basic(self.netD_B_local, self.real_A, self.labels_A, self.fake_A, self.labels_B, stats_tag="B")

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # 局所 (物体領域) GAN 損失
        # 信号機は 256x256 中の 0.7% 程度の画素しか占めず、大域のGAN損失/cycle損失には
        # ほぼ寄与しない（勾配の希釈）。物体領域だけに立てた損失でこれを直接補償する。
        if self.use_local:
            fake_B_crops, fake_B_cls, fake_B_scale = self._extract_crops(self.fake_B, self.labels_A, stats_tag="A")
            fake_A_crops, fake_A_cls, fake_A_scale = self._extract_crops(self.fake_A, self.labels_B, stats_tag="B")
            self.loss_G_A_local = self._local_gan_loss(self.netD_A_local, fake_B_crops, fake_B_cls, fake_B_scale, True) * self.opt.lambda_local
            self.loss_G_B_local = self._local_gan_loss(self.netD_B_local, fake_A_crops, fake_A_cls, fake_A_scale, True) * self.opt.lambda_local
        else:
            self.loss_G_A_local = 0
            self.loss_G_B_local = 0

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_G_A_local + self.loss_G_B_local
        self.loss_G.backward()

    def _all_Ds(self):
        """大域D + (有効なら) 局所D のリスト。requires_grad の一括切り替え用。"""
        nets = [self.netD_A, self.netD_B]
        if self.use_local:
            nets += [self.netD_A_local, self.netD_B_local]
        return nets

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self._all_Ds(), False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad(self._all_Ds(), True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        if self.use_local and self.opt.local_log_freq > 0:
            self.local_log_iter += 1
            if self.local_log_iter % self.opt.local_log_freq == 0:
                self.log_local_stats(self.local_log_iter)
