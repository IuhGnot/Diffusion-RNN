# scripts/eval_rnn_latent.py  —— 直接运行版（也支持命令行覆盖）
import os, sys, glob, time
from pathlib import Path
import argparse
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from diffusers import AutoencoderKL

# ------------------ 可直接改这里 ------------------
DEFAULTS = dict(
    pairlist = "../data/pairs1s_val.txt",       # 评测用的 pairlist（src|tgt|delta）
    ckpt     = "./runs_rnn/20250910-205653/ckpts/latest_ema.ckpt",                       # 不填就会自动找 runs_rnn/*/ckpts/latest.ckpt
    img_size = 256,
    batch    = 8,
    outdir   = "./eval_out",
    precision= "fp32",                     # "fp32"/"fp16"/"bf16"
    runs_root= "./runs_rnn",               # 自动找 ckpt 的根目录
)
# --------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class PairList(Dataset):
    def __init__(self, pairlist, img_size=256):
        self.items = []
        with open(pairlist, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    s,t,d = line.strip().split("|")
                    self.items.append((s,t,float(d)))
        assert len(self.items) > 0, f"Empty pairlist: {pairlist}"
        self.tf = T.Compose([
            T.Resize((img_size,img_size), antialias=True),
            T.ToTensor()  # [0,1]
        ])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        s,t,d = self.items[i]
        xs = self.tf(Image.open(s).convert("RGB"))
        xt = self.tf(Image.open(t).convert("RGB"))
        return xs, xt, torch.tensor(d, dtype=torch.float32), s, t

def psnr(a,b):  # a,b in [0,1]
    return 10*torch.log10(1.0/((a-b).pow(2).mean().clamp(1e-12)))

def auto_find_latest_ckpt(runs_root="./runs_rnn"):
    """在 runs_rnn/*/ckpts/latest.ckpt 中选 mtime 最新的一个"""
    cands = glob.glob(os.path.join(runs_root, "*", "ckpts", "latest.ckpt"))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def build_config():
    # 有命令行就解析；否则用 DEFAULTS
    ap = argparse.ArgumentParser(add_help=False)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", action=("store_true" if v else "store_false"))
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    cfg = ap.parse_args() if len(sys.argv) > 1 else ap.parse_args([])
    # 自动找 ckpt
    if (cfg.ckpt is None) or (not Path(cfg.ckpt).exists()):
        auto_ckpt = auto_find_latest_ckpt(cfg.runs_root)
        if auto_ckpt is None:
            raise FileNotFoundError(
                f"未找到可用 ckpt。请设置 --ckpt 或确保 {cfg.runs_root}/*/ckpts/latest.ckpt 存在。"
            )
        print(f"[info] 未指定/找不到 ckpt，自动使用：{auto_ckpt}")
        cfg.ckpt = auto_ckpt
    return cfg

def main():
    args = build_config()
    print("========== EVAL CONFIG ==========")
    for k,v in vars(args).items(): print(f"{k}: {v}")
    print("=================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.precision=="fp16" and device=="cuda":
        cast = lambda : torch.amp.autocast("cuda", dtype=torch.float16)
    elif args.precision=="bf16" and device=="cuda":
        cast = lambda : torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        cast = lambda : nullcontext()

    # Data
    ds = PairList(args.pairlist, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.models.diffusion_rnn import LatentDynamicsRNN

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)

    model = LatentDynamicsRNN(z_channels=4, hidden=64, num_res=2, tdim=128).to(device)
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    lpips = LPIPS(net_type='vgg', normalize=True).to(device).eval()

    # Eval
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    psnrs, lpips_vals, lat_mses = [], [], []

    with torch.no_grad(), cast():
        first_batch_saved = False
        for xs, xt, delt, *_ in tqdm(dl, desc="Eval", ncols=120):
            xs = xs.to(device); xt = xt.to(device)
            delt = delt.to(device)
            dnorm = torch.clamp(delt / (delt.max()+1e-6), 0, 1)  # 简单归一（也可自定）

            z_s = vae.encode(xs*2-1).latent_dist.mode()
            z_t = vae.encode(xt*2-1).latent_dist.mode()
            z_hat = model(z_s, dnorm)

            x_hat = (vae.decode(z_hat).sample + 1)/2
            x_hat_vis = x_hat.clamp(0,1)

            # metrics
            lp = lpips(xt, x_hat_vis).mean().item()
            lpips_vals.append(lp)
            for i in range(xs.size(0)):
                psnrs.append(psnr(xt[i:i+1], x_hat_vis[i:i+1]).item())
                lat_mses.append(F.mse_loss(z_hat[i:i+1], z_t[i:i+1]).item())

            # save first batch grid
            if not first_batch_saved:
                err = (xt[:8] - x_hat_vis[:8]).abs()
                grid = make_grid(torch.cat([xs[:8], xt[:8], x_hat_vis[:8], err], dim=0),
                                 nrow=8, padding=2)
                save_image(grid.float().cpu(), outdir/"vis_grid.png")
                first_batch_saved = True

    print(f"[EVAL] pairs={len(ds)}  "
          f"PSNR={sum(psnrs)/len(psnrs):.2f}  "
          f"LPIPS={sum(lpips_vals)/len(lpips_vals):.3f}  "
          f"LatMSE={sum(lat_mses)/len(lat_mses):.5f}")
    print(f"[EVAL] saved vis to {outdir/'vis_grid.png'}")

if __name__ == "__main__":
    # 直接运行即可；若想覆盖参数，可加命令行传入
    main()
