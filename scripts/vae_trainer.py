# scripts/vae_trainer.py
import os, sys, argparse, random
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext

# TensorBoard (可选)
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_OK = True
except Exception:
    TB_OK = False

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

# 你的 VAE
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.vae import AutoencoderKLCustom

# ----------------------- 默认配置 -----------------------
DEFAULTS = dict(
    data_root="../data/vaetestimages/",
    list_file=None,
    img_size=256,
    batch=8,
    epochs=50,
    lr=1e-4,
    wd=1e-4,
    num_workers=0,          # Windows 建议 0
    val_ratio=0.05,
    lpips_w=0.1,
    beta_max=0.25,
    beta_warmup_epochs=5,
    ema=True,
    ema_decay=0.999,
    val_every=1,
    save_every=5,
    precision="fp32",       # fp16/bf16/fp32
    resume=None,
    outdir=".",             # 运行目录会建在 <outdir>/outputs/run_YYYYmmdd_HHMMSS/
    seed=42,

    # 额外增强（可选）
    use_free_bits=True,
    free_bits_eps=0.03,     # 0.03~0.05 nats 常用
    hf_loss_w=0.10,         # Sobel 高频损失权重；0 代表关闭
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ----------------------- 数据集 -----------------------
class ImageFolderFlat(Dataset):
    """从图片根目录或清单读取；只训练 VAE，不涉及时间信息。"""
    def __init__(self, root, img_size=256, list_file=None, split="train", val_ratio=0.05, seed=42):
        self.root = Path(root)
        self.img_size = img_size

        if list_file:
            with open(list_file, "r", encoding="utf-8") as f:
                files = [line.strip() for line in f if line.strip()]
        else:
            files = [str(p) for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXTS]

        if len(files) == 0:
            raise FileNotFoundError(f"No images found under: {self.root}")

        rng = random.Random(seed)
        rng.shuffle(files)

        n_total = len(files)
        n_val = max(1, int(n_total * val_ratio))
        if n_total - n_val <= 0:
            n_val = max(0, n_total - 1)

        self.files = files[n_val:] if split == "train" else files[:n_val]

        if split == "train":
            self.transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                                    interpolation=IM.BICUBIC, antialias=True),
                T.RandomHorizontalFlip(),
                T.ToTensor(),   # -> [0,1] float32
            ])
        else:
            self.transform = T.Compose([
                T.Resize(int(img_size * 1.15), interpolation=IM.BICUBIC, antialias=True),
                T.CenterCrop(img_size),
                T.ToTensor(),
            ])

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, p

# ----------------------- 工具 -----------------------
def psnr(a, b):
    mse = (a - b).pow(2).mean().clamp(1e-12)
    return 10 * torch.log10(1.0 / mse)

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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

def safe_kl(dist, reduce=None) -> torch.Tensor:
    """
    安全调用 KL：
      - 优先用 dist.kl(reduce)
      - 缺失时用 mean/logvar 回退：对 (C,H,W) 求 mean
      - reduce=None -> (B,), 'mean' -> 标量, 'sum' -> 标量
    """
    try:
        kl = dist.kl(reduce=reduce)
        if kl is not None:
            return kl
    except TypeError:
        try:
            kl_raw = dist.kl()
            if kl_raw is not None:
                if reduce is None:
                    return kl_raw
                elif reduce == "mean":
                    return kl_raw.mean()
                elif reduce == "sum":
                    return kl_raw.sum()
        except Exception:
            pass

    mean = getattr(dist, "mean", None)
    logvar = getattr(dist, "logvar", None)
    if mean is None or logvar is None:
        raise RuntimeError("Cannot compute KL: latent_dist has no mean/logvar and .kl() returned None.")
    kl_map = 0.5 * (torch.exp(logvar) + mean**2 - 1.0 - logvar)  # (B,C,H,W)
    kl_per_sample = kl_map.mean(dim=(1,2,3))                      # (B,)
    if reduce is None:
        return kl_per_sample
    elif reduce == "mean":
        return kl_per_sample.mean()
    elif reduce == "sum":
        return kl_per_sample.sum()
    else:
        raise ValueError("reduce must be one of {None,'mean','sum'}")

def grad_hf_loss(x, y):
    """Sobel 高频一致性：抑制“奶油糊”，x/y ∈ [0,1]."""
    C = x.size(1)
    kx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=x.device, dtype=x.dtype).repeat(C,1,1,1)
    ky = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=x.device, dtype=x.dtype).repeat(C,1,1,1)
    gx_x = torch.conv2d(x, kx, padding=1, groups=C)
    gx_y = torch.conv2d(y, kx, padding=1, groups=C)
    gy_x = torch.conv2d(x, ky, padding=1, groups=C)
    gy_y = torch.conv2d(y, ky, padding=1, groups=C)
    return (gx_x-gx_y).abs().mean() + (gy_x-gy_y).abs().mean()

# ----------------------- 坍缩自检 -----------------------
class CollapseMonitor:
    """
    简单的后验塌缩监控：
      - 条件：|mu| < mu_thresh 且 sigma > sigma_thresh
      - 连续 patience 步满足条件则告警一次
    """
    def __init__(self, mu_thresh=0.02, sigma_thresh=0.95, patience=100):
        self.mu_thresh = mu_thresh
        self.sigma_thresh = sigma_thresh
        self.patience = patience
        self.counter = 0
        self.triggered = False

    def update(self, mu_abs, sigma):
        if mu_abs < self.mu_thresh and sigma > self.sigma_thresh:
            self.counter += 1
        else:
            self.counter = 0
        fire = (not self.triggered) and (self.counter >= self.patience)
        if fire:
            self.triggered = True
        return fire

# ----------------------- 参数解析 -----------------------
def build_args():
    ap = argparse.ArgumentParser(add_help=False)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", action=("store_true" if v else "store_false"))
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    if len(sys.argv) > 1:
        return ap.parse_args()
    else:
        return ap.parse_args([])

# ----------------------- 训练主程序 -----------------------
def main():
    args = build_args()

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # AMP 配置（模型保持 FP32；仅前向混精）
    if args.precision == "fp16" and use_cuda:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
        autocast_ctx = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    elif args.precision == "bf16" and use_cuda:
        amp_dtype = torch.bfloat16
        scaler = None  # BF16 不需要 GradScaler
        autocast_ctx = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        scaler = None
        autocast_ctx = lambda: nullcontext()

    # === 运行目录：每次训练新建独立文件夹 ===
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_out = Path(args.outdir) / "outputs" / run_name
    val_vis_dir  = base_out / "val_vis"
    latents_dir  = base_out / "latents"
    ckpt_dir     = base_out / "checkpoints"
    logs_dir     = base_out / "logs"
    for d in [val_vis_dir, latents_dir, ckpt_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[run] outputs will be saved under: {base_out.resolve()}")

    # 打印配置
    print("========== Config ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================")

    # DataLoader
    nw = 0 if os.name == "nt" else args.num_workers
    train_ds = ImageFolderFlat(args.data_root, args.img_size, args.list_file, "train", args.val_ratio, args.seed)
    val_ds   = ImageFolderFlat(args.data_root, args.img_size, args.list_file, "val",   args.val_ratio, args.seed)
    print(f"[info] train images: {len(train_ds)}  val images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=nw, pin_memory=use_cuda, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=min(args.batch, 16), shuffle=False,
                              num_workers=nw, pin_memory=use_cuda)

    # 模型（保持 FP32 权重！）
    vae = AutoencoderKLCustom().to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=args.wd)
    lpips = LPIPS(net_type='vgg', normalize=True).to(device).eval()

    ema = EMA(vae, args.ema_decay) if args.ema else None
    start_epoch, global_step, best_lpips = 0, 0, 1e9

    # 断点续训（注意：本脚本会把新生成的产物放进新 run 目录）
    if args.resume and Path(args.resume).is_file():
        state = torch.load(args.resume, map_location="cpu")
        vae.load_state_dict(state["state_dict"], strict=False)
        if "opt" in state:
            opt.load_state_dict(state["opt"])
        start_epoch = state.get("epoch", 0)
        global_step = state.get("global_step", 0)
        best_lpips  = state.get("best_lpips", best_lpips)
        print(f"[resume] loaded {args.resume} at epoch {start_epoch}")

    writer = SummaryWriter(log_dir=str(logs_dir)) if TB_OK else None

    # 自检器
    collapse_mon = CollapseMonitor(mu_thresh=0.02, sigma_thresh=0.95, patience=100)

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        vae.train()
        pbar = tqdm(train_loader, desc=f"Train e{epoch}", ncols=120)
        for x01, _ in pbar:
            x01 = x01.to(device, non_blocking=use_cuda)   # FP32 输入
            beta = args.beta_max * min(1.0, (epoch + global_step/len(train_loader)) / max(1, args.beta_warmup_epochs))

            with autocast_ctx():
                enc = vae.encode(x01*2 - 1).latent_dist
                z_scaled = enc.sample() * vae.scaling_factor
                xrec01 = (vae.decode(z_scaled / vae.scaling_factor).sample + 1) / 2

                l1 = (x01 - xrec01).abs().mean()
                xrec01_vis = xrec01.clamp(0, 1)
                with torch.no_grad():
                    lp = lpips(x01, xrec01_vis).mean()

                # KL（支持 Free Bits）
                if args.use_free_bits:
                    kl_per = safe_kl(enc, reduce=None)  # (B,)
                    kl = torch.clamp(kl_per - args.free_bits_eps, min=0).mean()
                else:
                    kl = safe_kl(enc, reduce="mean")

                loss = l1 + args.lpips_w * lp + beta * kl

                # 高频损失（可选）
                if args.hf_loss_w > 0:
                    hf = grad_hf_loss(x01, xrec01_vis)
                    loss = loss + args.hf_loss_w * hf
                else:
                    hf = torch.tensor(0.0, device=x01.device)

            # 自检：mu_abs, sigma
            with torch.no_grad():
                mu_abs = enc.mean.abs().mean().item()
                sigma  = (0.5*enc.logvar).exp().mean().item()

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            if ema: ema.update(vae)

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", l1=f"{l1.item():.3f}",
                             lpips=f"{lp.item():.3f}", kl=f"{kl.item():.3f}",
                             hf=f"{hf.item():.3f}", mu=f"{mu_abs:.3f}",
                             sigma=f"{sigma:.3f}", beta=f"{beta:.3f}")

            # TB 记录
            if writer:
                writer.add_scalar("train/loss",  loss.item(),  global_step)
                writer.add_scalar("train/l1",    l1.item(),    global_step)
                writer.add_scalar("train/lpips", lp.item(),    global_step)
                writer.add_scalar("train/kl",    kl.item(),    global_step)
                writer.add_scalar("train/beta",  beta,         global_step)
                writer.add_scalar("train/mu_abs", mu_abs,      global_step)
                writer.add_scalar("train/sigma",  sigma,       global_step)
                writer.add_scalar("train/hf",     hf.item(),   global_step)

            # 坍缩告警（只触发一次）
            if collapse_mon.update(mu_abs, sigma):
                print("\n[WARN] 可能出现后验塌缩：|mu|≈0 且 sigma≈1 持续过长。"
                      "\n       建议：降低 beta_max（如 0.05）、延长 beta_warmup_epochs（10~20）、"
                      "去掉 weight_decay、或先关闭 mid-attn/降低分辨率。")

        # 验证 & 可视化
        if (epoch % args.val_every) == 0:
            val_psnr, val_lpips = evaluate_and_visualize(
                vae, val_loader, device, lpips, val_vis_dir, latents_dir, epoch, ema, autocast_ctx
            )
            if writer:
                writer.add_scalar("val/psnr",  val_psnr,  epoch)
                writer.add_scalar("val/lpips", val_lpips, epoch)

            # 保存 ckpt（放在本次运行目录）
            if val_lpips < best_lpips:
                best_lpips = val_lpips
                save_ckpt(vae, opt, epoch, global_step, best_lpips, ckpt_dir/"vae_best.ckpt", ema)
            save_ckpt(vae, opt, epoch, global_step, best_lpips, ckpt_dir/"vae_latest.ckpt", ema)

        if (epoch % args.save_every) == 0 and epoch > 0:
            save_ckpt(vae, opt, epoch, global_step, best_lpips, ckpt_dir/f"vae_e{epoch:03d}.ckpt", ema)

    if writer: writer.close()

@torch.no_grad()
def evaluate_and_visualize(vae, val_loader, device, lpips_metric, vis_dir, latents_dir, epoch, ema=None, autocast_ctx=nullcontext):
    # 用 EMA 权重评估更稳
    vanilla_sd = None
    if ema:
        vanilla_sd = {k: v.clone() for k,v in vae.state_dict().items()}
        ema.copy_to(vae)

    vae.eval()
    psnrs, lpips_vals = [], []
    saved_once = False

    for x01, paths in tqdm(val_loader, desc="Validate", ncols=120):
        x01 = x01.to(device)
        with autocast_ctx():
            enc = vae.encode(x01*2 - 1).latent_dist
            # 可视化用均值更稳
            z_scaled = enc.mode() * vae.scaling_factor
            xrec01 = (vae.decode(z_scaled / vae.scaling_factor).sample + 1) / 2
        xrec01_vis = xrec01.clamp(0,1)

        for i in range(x01.size(0)):
            psnrs.append(psnr(x01[i:i+1], xrec01_vis[i:i+1]).item())
        lp = lpips_metric(x01, xrec01_vis).mean().item()
        lpips_vals.append(lp)

        if not saved_once:
            # 可视化网格
            grid = make_grid(torch.cat([
                    x01[:8].clamp(0,1),
                    xrec01_vis[:8],
                    (x01[:8]-xrec01_vis[:8]).abs().clamp(0,1)
                ], dim=0), nrow=8, padding=2)
            out_path = Path(vis_dir)/f"e{epoch:03d}_grid.png"
            Path(vis_dir).mkdir(parents=True, exist_ok=True)
            save_image(grid.float().cpu(), out_path)

            # 保存潜向量样例（均值潜变量）
            Path(latents_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"z": z_scaled[:8].float().cpu(), "paths": paths[:8]},
                       Path(latents_dir) / f"latents_epoch_{epoch:03d}.pt")
            saved_once = True

    if ema and vanilla_sd is not None:
        vae.load_state_dict(vanilla_sd, strict=False)

    mean_psnr = sum(psnrs)/len(psnrs)
    mean_lpips = sum(lpips_vals)/len(lpips_vals)
    print(f"[val] epoch {epoch}  PSNR={mean_psnr:.2f}  LPIPS={mean_lpips:.4f}")
    return mean_psnr, mean_lpips

def save_ckpt(vae, opt, epoch, global_step, best_lpips, path, ema=None):
    state_dict = vae.state_dict() if not ema else {k: v.clone() for k, v in ema.shadow.items()}
    torch.save({
        "state_dict": state_dict,
        "opt": opt.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_lpips": best_lpips,
        "scaling_factor": vae.scaling_factor
    }, path)
    print(f"[save] {path}")

if __name__ == "__main__":
    main()
