# scripts/encode_images.py
import os, sys, argparse
from pathlib import Path
import torch
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# 让脚本可直接 import 你的 VAE 实现
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.vae import AutoencoderKLCustom

# -------- 默认配置（零参数运行时使用） --------
DEFAULTS = dict(
    ckpt_main="checkpoints/vae_best.ckpt",      # 主 checkpoint
    ckpt_fallback="checkpoints/vae_latest.ckpt",# 备选（主不存在时回退）
    img_root="data/raw/images",                 # 图片根目录
    list_file=None,                             # 可选：每行一个图片路径
    img_size=256,
    out_dir="data/latents",                     # 潜空间分片输出目录
    shard_size=2000,                            # 每个分片保存多少张
    precision="fp16",                           # cpu 会自动回退到 fp32
    save_recon=False,                           # 是否同时保存重建图用于核验
    recon_out="outputs/recon_from_encode",      # 重建图输出目录（当 save_recon=True）
    num_workers=4,
)

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def build_args():
    ap = argparse.ArgumentParser(add_help=False)
    for k,v in DEFAULTS.items():
        if isinstance(v, bool):
            # 对布尔值，用 --save_recon / --no-save_recon 控制更复杂；这里简单用字符串覆盖
            ap.add_argument(f"--{k}", type=str, default=None)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=None)
    if len(sys.argv) > 1:
        raw = ap.parse_args()
        # 覆盖 defaults
        cfg = DEFAULTS.copy()
        for k in DEFAULTS.keys():
            val = getattr(raw, k)
            if val is not None:
                if isinstance(DEFAULTS[k], bool):
                    cfg[k] = (val.lower() in ["1","true","yes","y"])
                else:
                    cfg[k] = val
        return argparse.Namespace(**cfg)
    else:
        return argparse.Namespace(**DEFAULTS)

def gather_images(img_root: str, list_file: str):
    files = []
    if list_file:
        with open(list_file, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()]
    else:
        files = [str(p) for p in Path(img_root).rglob("*") if p.suffix.lower() in IMG_EXTS]
    files.sort()
    if len(files) == 0:
        raise FileNotFoundError(f"No images found. root='{img_root}', list_file='{list_file}'")
    return files

def load_vae(ckpt_main: str, ckpt_fallback: str, device: str, dtype: torch.dtype):
    vae = AutoencoderKLCustom().to(device=device, dtype=dtype).eval()
    ckpt_path = Path(ckpt_main)
    if not ckpt_path.exists() and ckpt_fallback:
        fb = Path(ckpt_fallback)
        if fb.exists():
            ckpt_path = fb
    if not ckpt_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_main} (fallback: {ckpt_fallback})")
    state = torch.load(ckpt_path, map_location="cpu")
    vae.load_state_dict(state["state_dict"], strict=False)
    sf = float(state.get("scaling_factor", vae.scaling_factor))
    print(f"[load] VAE from '{ckpt_path}'  scaling_factor={sf}")
    return vae, sf

def save_shard(buf, out_dir: Path, shard_id: int):
    Z  = torch.stack([b["z0"] for b in buf],  dim=0)
    MU = torch.stack([b["mu"] for b in buf],  dim=0)
    LV = torch.stack([b["logvar"] for b in buf], dim=0)
    paths = [b["path"] for b in buf]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"latents_shard_{shard_id:05d}.pt"
    torch.save({"z0": Z, "mu": MU, "logvar": LV, "paths": paths}, out_path)
    print(f"[save] {out_path}  (N={len(buf)})")

def main():
    args = build_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(args.precision, torch.float16)
    if device == "cpu":
        dtype = torch.float32  # CPU 用 fp32 更稳
        print("[warn] CPU detected, precision set to fp32.")

    files = gather_images(args.img_root, args.list_file)
    print(f"[info] total images: {len(files)}")

    vae, sf = load_vae(args.ckpt_main, args.ckpt_fallback, device, dtype)

    to_tensor = T.Compose([T.Resize((args.img_size,args.img_size)), T.ToTensor()])

    shard_id, buf = 0, []
    recon_dir = Path(args.recon_out)
    if args.save_recon:
        recon_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(files, desc="encode"):
        x01 = to_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            moments = vae.encoder(x01*2-1)                  # (1, 2C, h, w)
            n = moments.shape[1]//2
            mu, logvar = moments[:,:n], moments[:,n:]
            z = (mu + torch.randn_like(mu)*torch.exp(0.5*logvar)) * sf  # 已乘 scaling_factor

            if args.save_recon:
                xrec01 = (vae.decode(z/sf).sample + 1)/2
                name = Path(p).stem + ".png"
                save_image(xrec01.clamp(0,1), recon_dir / name)

        buf.append({
            "z0": z.squeeze(0).cpu().float(),
            "mu": mu.squeeze(0).cpu().float(),
            "logvar": logvar.squeeze(0).cpu().float(),
            "path": p
        })

        if len(buf) >= args.shard_size:
            save_shard(buf, Path(args.out_dir), shard_id)
            buf, shard_id = [], shard_id + 1

    if len(buf) > 0:
        save_shard(buf, Path(args.out_dir), shard_id)

if __name__ == "__main__":
    # 直接运行即可使用 DEFAULTS；也可通过命令行覆盖
    main()
