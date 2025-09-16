# scripts/sample_diffusion_latent.py
import os, sys, glob, math, argparse, time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image
from contextlib import nullcontext
from tqdm import tqdm
from diffusers import AutoencoderKL

DEFAULTS = dict(
    pairlist="../data/pairs1s_all_val.txt",
    ckpt=None,
    runs_root="./runs_diff",
    outdir="./samples",
    img_size=256,
    precision="fp32",
    vae_name="stabilityai/sd-vae-ft-mse",
    mode="warmstart",         # warmstart / pure
    steps=50,
    schedule="cosine",
    context_len=None,
    seed=1234,
    warmstart_sigma=0.20,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class PairList(Dataset):
    def __init__(self, pairlist, img_size=256):
        self.items=[]
        with open(pairlist,"r",encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    s,t,d=line.strip().split("|")
                    self.items.append((s,t,float(d)))
        assert len(self.items)>0, f"Empty pairlist: {pairlist}"
        self.tf=T.Compose([T.Resize((img_size,img_size), antialias=True), T.ToTensor()])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        s,t,d=self.items[i]
        xs=self.tf(Image.open(s).convert("RGB"))
        xt=self.tf(Image.open(t).convert("RGB"))
        return xs, xt, torch.tensor(float(d)), s, t

class SimpleNoiseSchedule:
    def __init__(self, num_steps=50):
        self.N=int(num_steps)
        s=0.008
        t=torch.linspace(0,1,self.N+1)
        alphas_bar=torch.cos(((t+s)/(1+s))*math.pi/2)**2
        self.ab = (alphas_bar/alphas_bar[0]).contiguous()
    def ab_t(self, t_idx, device):
        t_idx=torch.clamp(t_idx, 0, self.N)
        return self.ab.to(device).index_select(0, t_idx)

def find_t_from_sigma(sched, target_sigma, device):
    ab = sched.ab.to(device)
    sig = torch.sqrt(1.0 - ab)
    t_idx = int(torch.argmin((sig - target_sigma)**2).item())
    return max(0, min(sched.N, t_idx))

def auto_latest_ckpt(root="./runs_diff"):
    cands=glob.glob(os.path.join(root,"*","ckpts","latest.ckpt"))
    if not cands: return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def build_args():
    ap=argparse.ArgumentParser(add_help=False)
    for k,v in DEFAULTS.items():
        if isinstance(v,bool): ap.add_argument(f"--{k}", action=("store_true" if v else "store_false"))
        else: ap.add_argument(f"--{k}", type=type(v), default=v)
    args = ap.parse_args() if len(sys.argv)>1 else ap.parse_args([])
    if (args.ckpt is None) or (not Path(args.ckpt).exists()):
        auto = auto_latest_ckpt(args.runs_root)
        if auto is None: raise FileNotFoundError("未找到 ckpt，请设置 --ckpt 或先训练。")
        print(f"[auto] ckpt: {auto}")
        args.ckpt = auto
    return args

def main():
    args=build_args()
    print("========== SAMPLE CONFIG ==========")
    for k,v in vars(args).items(): print(f"{k}: {v}")
    print("===================================")

    use_cuda=torch.cuda.is_available()
    device="cuda" if use_cuda else "cpu"
    if args.precision=="fp16" and use_cuda:
        cast=lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    elif args.precision=="bf16" and use_cuda:
        cast=lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        cast=lambda: nullcontext()

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    ds=PairList(args.pairlist, img_size=args.img_size)
    dl=DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=use_cuda)

    vae=AutoencoderKL.from_pretrained(args.vae_name).to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    sf=float(getattr(getattr(vae,"config",None),"scaling_factor", 1.0))
    print(f"[debug] vae.scaling_factor = {sf}")

    ck=torch.load(args.ckpt, map_location="cpu")
    print(f"[load] {args.ckpt}")
    meta=ck.get("meta", {})
    zc    = int(meta.get("z_channels", 4))
    base  = int(meta.get("base", meta.get("hidden", 128)))
    depth = int(meta.get("depth", meta.get("num_res", 4)))
    tdim  = int(meta.get("tdim", 256))
    Ntrain= int(meta.get("noise_steps", 50))

    sched=SimpleNoiseSchedule(num_steps=Ntrain)

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.models.diffusion_unet_old import UNetLatentEps
    # ---- 关键修正：delta_in=1，与旧 ckpt 保持一致 ----
    net = UNetLatentEps(zc=zc, base=base, depth=depth, tdim=tdim, cond=True, delta_in=1).to(device).eval()

    sd = ck.get("ema", None) or ck.get("model", None)
    if sd is None: raise RuntimeError("ckpt 中无 'ema' 或 'model' 权重。")
    # 用 strict=True；如果你后续扩展了键，可调成 False
    net.load_state_dict(sd, strict=True)

    steps=max(1, int(args.steps))
    t_seq=torch.linspace(Ntrain, 0, steps+1, dtype=torch.long)  # N->0
    t_seq=t_seq.clamp(0, Ntrain)

    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), cast():
        for idx, (xs, xt, d, sp, tp) in enumerate(tqdm(dl, ncols=120, desc=f"sample[{args.mode}]")):
            xs=xs.to(device); xt=xt.to(device)
            d =d.to(device).float()

            z_prev = vae.encode(xs*2-1).latent_dist.mode().mul(sf)

            if args.mode.lower()=="warmstart":
                t0 = find_t_from_sigma(sched, float(args.warmstart_sigma), device)
                ab0 = sched.ab_t(torch.tensor([t0], device=device), device=device)[0]
                a0  = ab0.sqrt().view(1,1,1,1)
                s0  = (1 - ab0).sqrt().view(1,1,1,1)
                eps0= torch.randn_like(z_prev)
                x   = a0 * z_prev + s0 * eps0
                start_index = int(torch.argmin((t_seq - t0).abs()).item())
            else:
                x = torch.randn_like(z_prev)
                start_index = 0

            # Δ 归一（与你训练时保持一致：[-1,1]），并且 **(B,1)** 形状喂给 delta_mlp
            dnorm = torch.clamp(d / (d.abs().max() + 1e-6), -1, 1)
            d_emb = dnorm.view(1, 1)  # ---- 关键修正：形状 (B,1)

            for si in range(start_index, len(t_seq)-1):
                t_cur = int(t_seq[si].item())
                t_nxt = int(t_seq[si+1].item())

                ab_t = sched.ab_t(torch.tensor([t_cur], device=device), device=device)[0]
                a_t  = ab_t.sqrt().view(1,1,1,1)
                s_t  = (1 - ab_t).sqrt().view(1,1,1,1)

                # time embedding 输入网络的 time_mlp（网络里做两层线性）
                t_emb = torch.full((1, tdim), float(t_cur)/(Ntrain+1), device=device)

                eps_hat = net(x, z_prev, t_emb, d_emb)
                x0_hat  = (x - s_t * eps_hat) / (a_t + 1e-8)

                ab_n = sched.ab_t(torch.tensor([t_nxt], device=device), device=device)[0]
                a_n  = ab_n.sqrt().view(1,1,1,1)
                s_n  = (1 - ab_n).sqrt().view(1,1,1,1)
                x    = a_n * x0_hat + s_n * eps_hat  # DDIM eta=0

            x_rec = (vae.decode(x.div(sf)).sample + 1)/2
            x_rec = x_rec.clamp(0,1)
            grid = make_grid(torch.cat([xs, xt, x_rec], dim=0), nrow=1, padding=2)
            save_image(grid.float().cpu(), outdir/f"sample_{idx:04d}.png")

    print("[done]", str(outdir))


if __name__ == "__main__":
    main()
