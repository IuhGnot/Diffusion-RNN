# scripts/decode_latents.py
import os, sys, argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from tqdm import tqdm

# 让脚本能直接 import 你的 VAE 实现
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.vae import AutoencoderKLCustom

# -------- 默认配置（零参数运行时使用） --------
DEFAULTS = dict(
    ckpt_main="checkpoints/vae_best.ckpt",      # 首选权重
    ckpt_fallback="checkpoints/vae_latest.ckpt",# 备选权重（首选不存在时回退）
    latents_path="outputs",                     # 目录（批量解码所有 .pt）或单个 .pt
    outdir="outputs/recon_from_latent",         # 解码后的图片输出目录
    precision="fp32",                           # cpu 自动回退到 fp32
)

def build_args():
    ap = argparse.ArgumentParser(add_help=False)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=None)
    if len(sys.argv) > 1:
        raw = ap.parse_args()
        cfg = DEFAULTS.copy()
        for k in DEFAULTS.keys():
            val = getattr(raw, k)
            if val is not None:
                cfg[k] = val
        return argparse.Namespace(**cfg)
    else:
        return argparse.Namespace(**DEFAULTS)

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

def iter_latent_files(latents_path: str):
    p = Path(latents_path)
    if p.is_file():
        yield p
    else:
        files = sorted(list(p.glob("latents_shard_*.pt")))
        if not files:
            files = sorted(list(p.glob("*.pt")))  # 兼容任意 .pt
        if not files:
            raise FileNotFoundError(f"No latent .pt files found under {latents_path}")
        for f in files:
            yield f

def extract_latents(bundle):
    """
    兼容多种保存格式：
    - {"z0": (N,C,h,w), "paths": [...]}
    - {"z":  (N,C,h,w), "paths": [...]}
    - {"latents": ...} / {"latent": ...} / {"Z": ...}
    - 直接是 Tensor: (N,C,h,w) 或 (C,h,w)
    """
    # 直接 Tensor
    if torch.is_tensor(bundle):
        Z = bundle
        paths = None
    elif isinstance(bundle, dict):
        Z = None
        for key in ["z0", "z", "latents", "latent", "Z"]:
            if key in bundle:
                Z = bundle[key]
                break
        if Z is None:
            raise KeyError(f"No latent key found. Available keys: {list(bundle.keys())}")
        paths = bundle.get("paths", None)
    else:
        raise TypeError(f"Unsupported latent file type: {type(bundle)}")

    if not torch.is_tensor(Z):
        raise TypeError(f"Latents must be a Tensor, got {type(Z)}")

    # (C,h,w) -> (1,C,h,w)
    if Z.ndim == 3:
        Z = Z.unsqueeze(0)
    if Z.ndim != 4:
        raise ValueError(f"Latents must be 4D (N,C,h,w), got {tuple(Z.shape)}")
    return Z, paths

def main():
    args = build_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(args.precision, torch.float16)
    if device == "cpu":
        dtype = torch.float32
        print("[warn] CPU detected, precision set to fp32.")

    vae, sf = load_vae(args.ckpt_main, args.ckpt_fallback, device, dtype)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for lf in iter_latent_files(args.latents_path):
        bundle = torch.load(lf, map_location="cpu")
        Z, paths = extract_latents(bundle)   # <- 关键：统一获取 (N,C,h,w) 和路径
        print(f"[decode] {lf.name}: {Z.shape[0]} items")

        for i in tqdm(range(Z.shape[0]), desc=f"decode {lf.name}"):
            z = Z[i:i+1].to(device=device, dtype=dtype)   # 已乘 scaling_factor
            with torch.no_grad():
                x01 = (vae.decode(z / sf).sample + 1) / 2
            name = (Path(paths[i]).stem if (paths is not None and i < len(paths))
                    else f"{lf.stem}_{i:06d}")
            save_image(x01.clamp(0, 1), outdir / f"{name}.png")

if __name__ == "__main__":
    # 直接运行即可使用 DEFAULTS；也可通过命令行覆盖
    main()
