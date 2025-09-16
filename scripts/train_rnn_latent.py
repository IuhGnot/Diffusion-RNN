# scripts/train_rnn_latent.py
# -*- coding: utf-8 -*-
import os, sys, time, argparse, random, re
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from diffusers import AutoencoderKL

# ================= 默认配置（直接在此修改，无需命令行） =================
DEFAULTS = dict(
    pairlist = "../data/pairs_all_mixed.txt",
    vallist  = "../data/pairs_all_mixed_val.txt",
    img_size = 256,
    batch    = 8,
    epochs   = 50,
    lr       = 2e-4,
    wd       = 0.0,
    num_workers = 0,
    precision   = "fp32",     # "fp32"|"fp16"|"bf16"
    seed     = 42,
    out_root = "./runs_rnn",

    # 损失权重（均为“越小越好”）
    w_lat = 1.0,    # 潜空间/同噪一致性
    w_dz  = 0.5,    # 相对位移一致性
    w_dir = 0.7,    # 方向一致性（1-cos），强力纠正“方向反”的问题
    w_edge= 0.15,   # 几何/边缘一致性（Sobel L1），抑制重影
    w_l1  = 1.0,    # 图像域 L1
    w_lp  = 0.3,    # LPIPS（参与反传）
    w_reg = 1e-4,   # 简单能量正则

    # 噪声耦合（同噪）训练
    use_noise = True,
    bar_alpha = 0.30,  # \bar{alpha}；建议 0.3~0.95 间尝试
    share_eps = True,  # 预测与目标共享同一 ε

    # VAE 编码（是否用 mode()；False=采样）
    use_det_encode = True,

    # RNN（PredRNN 风格）
    z_channels = 4,
    hidden = 96,
    tdim = 128,
    context_len = 3,   # 相邻 L-1 帧 + 当前帧，L=context_len

    # 稳定化
    grad_clip = 1.0,
    ema_decay = 0.999
)

# ================= 数据集：相邻帧上下文（与采样保持一致） =================
def _parse_frame_id(p):
    m = re.search(r"_(\d+)\.(png|jpg|jpeg|webp|bmp)$", p, re.IGNORECASE)
    return int(m.group(1)) if m else None

def _rebuild_with_id(p, fid):
    return re.sub(r"_(\d+)\.(png|jpg|jpeg|webp|bmp)$",
                  f"_{int(fid):06d}." + p.split(".")[-1],
                  p, flags=re.IGNORECASE)

class PairListAdjacentCtx(Dataset):
    """
    pairlist 行： src|tgt|delta
    - 上下文：src 的相邻前 L-1 帧 + src（若缺帧用 src 回填），严格相邻；
    - 目标：pairlist 指向的 tgt（Δ=1/3/... 均可）。
    """
    def __init__(self, pairlist, img_size=256, context_len=3, delta_scale=None):
        self.items=[]
        with open(pairlist,"r",encoding="utf-8") as f:
            for line in f:
                if "|" not in line: continue
                s,t,d=line.strip().split("|")
                self.items.append((s,t,float(d)))
        if not self.items:
            raise FileNotFoundError(f"Empty pairlist: {pairlist}")

        self.tf=T.Compose([T.Resize((img_size,img_size), antialias=True), T.ToTensor()])
        self.context_len=max(1,int(context_len))

        raw=np.array([it[2] for it in self.items], dtype=float)
        self.delta_scale = float(np.percentile(np.abs(raw),95)) if delta_scale is None else float(delta_scale)

    def _ctx_adjacent(self, s_path: str):
        fid = _parse_frame_id(s_path)
        if fid is None:
            return [s_path]*self.context_len
        ids = list(range(fid-(self.context_len-1), fid+1))
        out=[]
        for k in ids:
            cand=_rebuild_with_id(s_path,k)
            out.append(cand if os.path.exists(cand) else s_path)
        return out

    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        s,t,d=self.items[i]
        ctx_paths=self._ctx_adjacent(s)
        xs_seq=torch.stack([self.tf(Image.open(p).convert("RGB")) for p in ctx_paths], dim=0)
        xt=self.tf(Image.open(t).convert("RGB"))
        dnorm=max(-1.0,min(1.0, float(d)/max(1e-6,self.delta_scale)))
        return xs_seq, xt, torch.tensor(dnorm,dtype=torch.float32), s, t

# ================== 工具 ==================
def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def psnr(a,b):
    return 10*torch.log10(1.0/((a-b).pow(2).mean().clamp(1e-12)))

def bar_to_alpha_sigma(bar_alpha: float):
    bar_alpha = float(bar_alpha)
    bar_alpha = min(max(bar_alpha, 1e-6), 1-1e-6)
    import math
    return math.sqrt(bar_alpha), math.sqrt(1.0-bar_alpha)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

def num_params(m: torch.nn.Module):
    t = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return t, tr

def add_bool_arg(parser, name, default):
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}",     dest=dest, action="store_true")
    group.add_argument(f"--no-{name}",  dest=dest, action="store_false")
    parser.set_defaults(**{dest: bool(default)})

def build_args():
    ap = argparse.ArgumentParser(add_help=False)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            add_bool_arg(ap, k, v)      # 支持 --foo / --no-foo；不传就用 DEFAULTS
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    return ap.parse_args([]) if len(sys.argv) == 1 else ap.parse_args()

# --- 边缘（Sobel） ---
def sobel_xy(x: torch.Tensor):
    """
    x: (B,3,H,W) in [0,1]
    return: (B,2,H,W) -> [gx, gy]
    """
    gray = (0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3])
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    gx = torch.nn.functional.conv2d(gray, kx, padding=1)
    gy = torch.nn.functional.conv2d(gray, ky, padding=1)
    return torch.cat([gx, gy], dim=1)

# ================== 训练主程序 ==================
def main():
    args = build_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # AMP
    if args.precision=="fp16" and device=="cuda":
        cast = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda")
    elif args.precision=="bf16" and device=="cuda":
        cast = lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)
        scaler = None
    else:
        cast = lambda: nullcontext(); scaler = None

    # 输出目录
    run_dir = Path(args.out_root) / time.strftime("%Y%m%d-%H%M%S")
    (run_dir/"images").mkdir(parents=True, exist_ok=True)
    (run_dir/"ckpts").mkdir(parents=True, exist_ok=True)

    # 打印配置与环境
    print("========== TRAIN CONFIG ==========")
    for k in DEFAULTS.keys():
        print(f"{k}: {getattr(args,k)}")
    print("==================================")
    print(f"[env] torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[env] device: {torch.cuda.get_device_name(0)}")
        print(f"[env] capability: {torch.cuda.get_device_capability(0)}")
    print("[run]", run_dir.resolve())

    # 数据
    ds_tr_tmp = PairListAdjacentCtx(args.pairlist, img_size=args.img_size, context_len=args.context_len)
    delta_scale = ds_tr_tmp.delta_scale
    ds_tr = ds_tr_tmp
    ds_va = PairListAdjacentCtx(args.vallist, img_size=args.img_size, delta_scale=delta_scale, context_len=args.context_len)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=min(args.batch, 16), shuffle=False, num_workers=args.num_workers, pin_memory=True)

    steps_per_epoch = int(np.ceil(len(ds_tr) / max(1,args.batch)))
    print(f"[data] train={len(ds_tr)}  val={len(ds_va)}  batch={args.batch}  steps/epoch≈{steps_per_epoch}")
    print(f"[data] img_size={args.img_size}  context_len={args.context_len}  delta_scale(train)≈{delta_scale:.6f}")

    # VAE（统一尺度）
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    sf = float(getattr(getattr(vae,"config",None),"scaling_factor", 1.0))
    print(f"[vae] scaling_factor(sf) = {sf}")

    # 模型（PredRNN）
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.models.diffusion_rnn_predrnn import LatentDynamicsPredRNN
    model = LatentDynamicsPredRNN(
        z_channels=args.z_channels,
        hidden=args.hidden,
        num_layers=2,
        tdim=args.tdim,
        n_res_per_layer=1
    ).to(device)
    tot, tr = num_params(model)
    print(f"[rnn] params total={tot/1e6:.2f}M  trainable={tr/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    lpips = LPIPS(net_type='vgg', normalize=True).to(device).eval()
    for p in lpips.parameters(): p.requires_grad_(False)

    ema = EMA(model, decay=args.ema_decay)

    # 噪声日志
    if args.use_noise:
        a0, s0 = bar_to_alpha_sigma(args.bar_alpha)
        print(f"[noise] enabled: bar_alpha={args.bar_alpha:.3f} → alpha={a0:.3f}, sigma={s0:.3f}  share_eps={args.share_eps}")
    else:
        print(f"[noise] disabled: training on clean latents")

    # 指标说明
    print("""[metrics]（越小越好）
  lat : 潜空间/同噪 MSE（与 bar_alpha 相关）；~0.2–1.0 起步，<0.05–0.20 较好
  dz  : 相对位移 MSE；若长期>lat，说明没学到运动
  dir : 方向（1-cos）；0 最好，≈1（约60°），≈2（反向）
  edge: Sobel 边缘 L1；抑制重影
  l1  : 图像域 L1（0..1）；~0.05–0.15 起步，<0.02–0.05 较好
  lp  : LPIPS；~0.30–0.60 起步，<0.15 较好
""")

    # ============ 训练 ============
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Train e{epoch}", ncols=120)
        for xs_seq, xt, dnorm, *_ in pbar:
            xs_seq = xs_seq.to(device); xt = xt.to(device); dnorm = dnorm.to(device)

            with cast():
                B, L, _, H, W = xs_seq.shape
                # --- 编码相邻上下文 → 潜空间（×sf） ---
                xs_flat = xs_seq.view(B*L, 3, H, W)
                d_ctx = vae.encode(xs_flat*2-1).latent_dist
                z_seq = (d_ctx.mode() if args.use_det_encode else d_ctx.sample()).mul(sf).view(B, L, -1, H//8, W//8)

                # 目标潜空间（×sf）
                d_tar = vae.encode(xt*2-1).latent_dist
                z_t = (d_tar.mode() if args.use_det_encode else d_tar.sample()).mul(sf)
                z_last = z_seq[:, -1]

                # 预测
                z_hat = model(z_seq, dnorm)  # 形状与 z_last 一致（×sf）

                # --- 潜空间/同噪一致性 ---
                if args.use_noise:
                    alpha, sigma = bar_to_alpha_sigma(args.bar_alpha)
                    alpha = torch.tensor(alpha, device=device, dtype=z_t.dtype).view(1,1,1,1)
                    sigma = torch.tensor(sigma, device=device, dtype=z_t.dtype).view(1,1,1,1)
                    eps = torch.randn_like(z_t)
                    if args.share_eps:
                        y_t = alpha*z_t + sigma*eps
                        y_h = alpha*z_hat + sigma*eps
                    else:
                        y_t = alpha*z_t + sigma*torch.randn_like(z_t)
                        y_h = alpha*z_hat + sigma*torch.randn_like(z_hat)
                    loss_lat = F.mse_loss(y_h, y_t)
                else:
                    loss_lat = F.mse_loss(z_hat, z_t)

                # 相对位移一致性
                loss_dz  = F.mse_loss(z_hat - z_last, z_t - z_last)

                # 方向一致性（1-cos）
                v_hat = (z_hat - z_last).flatten(1)
                v_gt  = (z_t   - z_last).flatten(1)
                cos = (v_hat * v_gt).sum(1) / (v_hat.norm(dim=1)*v_gt.norm(dim=1) + 1e-8)
                loss_dir = (1.0 - cos).mean()

                # 图像域（解码 ÷sf）
                x_hat = (vae.decode(z_hat.div(sf)).sample + 1)/2
                x_hat = x_hat.clamp(0,1)
                loss_l1 = (xt - x_hat).abs().mean()
                loss_lp = lpips(xt, x_hat).mean()

                # 边缘（Sobel）
                loss_edge = (sobel_xy(x_hat) - sobel_xy(xt)).abs().mean()

                # 正则
                reg = args.w_reg * ((z_hat.pow(2).mean() - z_t.pow(2).mean()).abs())

                # 总损失
                loss = (args.w_lat*loss_lat + args.w_dz*loss_dz +
                        args.w_dir*loss_dir + args.w_edge*loss_edge +
                        args.w_l1*loss_l1 + args.w_lp*loss_lp + reg)

            opt.zero_grad(set_to_none=True)
            if scaler: scaler.scale(loss).backward()
            else:      loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler:
                scaler.step(opt); scaler.update()
            else:
                opt.step()

            ema.update(model)
            pbar.set_postfix(lat=f"{loss_lat.item():.4f}", dz=f"{loss_dz.item():.4f}",
                             dir=f"{loss_dir.item():.4f}", edge=f"{loss_edge.item():.4f}",
                             l1=f"{loss_l1.item():.4f}", lp=f"{loss_lp.item():.3f}",
                             tot=f"{loss.item():.4f}")

        # ============ 验证（EMA） ============
        bak = {k: v.clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        model.eval()

        psnrs, lpips_vals, lat_mses, dzz_mses, dir_vals = [], [], [], [], []
        with torch.no_grad(), cast():
            first = True
            for xs_seq, xt, dnorm, *_ in tqdm(dl_va, desc="Val", ncols=120):
                xs_seq = xs_seq.to(device); xt = xt.to(device); dnorm = dnorm.to(device)

                B, L, _, H, W = xs_seq.shape
                xs_flat = xs_seq.view(B*L, 3, H, W)
                d_ctx = vae.encode(xs_flat*2-1).latent_dist
                z_seq = d_ctx.mode().mul(sf).view(B, L, -1, H//8, W//8)  # 评估用 mode
                z_last = z_seq[:, -1]

                z_t = vae.encode(xt*2-1).latent_dist.mode().mul(sf)
                z_hat = model(z_seq, dnorm)
                x_hat = (vae.decode(z_hat.div(sf)).sample + 1)/2
                x_hat = x_hat.clamp(0,1)

                # 方向指标
                v_hat = (z_hat - z_last).flatten(1)
                v_gt  = (z_t   - z_last).flatten(1)
                cos = (v_hat * v_gt).sum(1) / (v_hat.norm(dim=1)*v_gt.norm(dim=1) + 1e-8)
                dir_vals.extend((1.0 - cos).detach().cpu().numpy().tolist())

                lp = lpips(xt, x_hat).mean().item()
                lpips_vals.append(lp)
                for i in range(B):
                    psnrs.append(psnr(xt[i:i+1], x_hat[i:i+1]).item())
                    lat_mses.append(F.mse_loss(z_hat[i:i+1], z_t[i:i+1]).item())
                    dzz_mses.append(F.mse_loss((z_hat-z_last)[i:i+1], (z_t-z_last)[i:i+1]).item())

                if first:
                    grid = make_grid(torch.cat([xs_seq[:, -1][:8], xt[:8], x_hat[:8],
                                                (xt[:8]-x_hat[:8]).abs()], dim=0),
                                     nrow=8, padding=2)
                    save_image(grid.float().cpu(), run_dir/"images"/f"e{epoch:03d}_grid.png")
                    first = False

        print(f"[val] e{epoch}  PSNR={np.mean(psnrs):.2f}  LPIPS={np.mean(lpips_vals):.3f}  "
              f"LatMSE={np.mean(lat_mses):.5f}  dZ-MSE={np.mean(dzz_mses):.5f}  Dir={np.mean(dir_vals):.4f}")

        # 复原并保存
        model.load_state_dict(bak, strict=True)

        meta = dict(  # 采样脚本可读取，保证规范一致
            sf=sf,
            context_len=args.context_len,
            delta_scale=delta_scale,
            use_noise=args.use_noise,
            bar_alpha=args.bar_alpha,
            share_eps=args.share_eps,
            use_det_encode=args.use_det_encode,
            z_channels=args.z_channels,
            hidden=args.hidden,
            tdim=args.tdim,
            arch="LatentDynamicsPredRNN",
        )
        torch.save({"model": ema.shadow, "meta": meta}, run_dir/"ckpts"/"latest_ema.ckpt")
        torch.save({"model": ema.shadow, "meta": meta}, run_dir/"ckpts"/f"e{epoch:03d}_ema.ckpt")

if __name__ == "__main__":
    main()
