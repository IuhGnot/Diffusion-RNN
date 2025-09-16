# scripts/train_diffusion_latent.py
# -*- coding: utf-8 -*-
import os, sys, time, argparse, random, math, re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext
from diffusers import AutoencoderKL

# 项目内路径
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.ldm_unet import UNetLatentEps, timestep_embedding

# 你可以换成 PredRNN 版（建议与 RNN 训练脚本保持一致）
from src.models.diffusion_rnn_predrnn import LatentDynamicsPredRNN

# ---------------- Defaults ----------------
DEFAULTS = dict(
    # 数据
    pairlist="../data/pairs_all_mixed.txt",
    vallist ="../data/pairs_all_mixed_val.txt",
    vae_name="stabilityai/sd-vae-ft-mse",
    img_size=256,
    batch=8,
    epochs=15,
    lr=1e-4,
    wd=0.0,
    num_workers=0,
    precision="fp32",
    outroot="./runs_diff",
    seed=42,

    # diffusion 模型结构
    noise_steps=100,       # T
    base=192, depth=3, n_res=2,
    tdim=256, emb_ch=256,

    # 条件开关（训练/验证都生效）
    use_prev=True,        # 是否使用先验（上一帧/RNN）
    use_delta=True,       # 是否使用 delta（pairlist 的 dnorm）

    # 训练技巧
    v_pred=True,           # True=预测 v；False=预测 epsilon
    minsnr_gamma=5.0,      # 0 关闭 Min-SNR；常用 2~5
    ema=True, ema_decay=0.999,

    # 验证/采样
    val_mode="warmstart",  # "pure" | "warmstart"
    warmstart_sigma=0.15,  # 更保守的 warmstart

    # RNN 先验
    rnn_ckpt="./runs_rnn/20250914-173909/ckpts/latest_ema.ckpt",
    rnn_context=3,         # 用相邻 L 帧上下文（应与 RNN 训练一致）
    lambda_prior=0.00,     # 先验对齐权重（建议先 0，稳定后再逐步加 0.02~0.05）

    # 诊断打印的 batch 间隔
    log_every=2,
)

# ---------------- Dataset（相邻帧上下文，与 RNN 保持一致） ----------------
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
    - 上下文：src 的相邻前 L-1 帧 + src（严格相邻，缺帧用 src 回填）
    - 目标：pairlist 的 tgt（可为 Δ=1/3/... 混合）
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

        import numpy as np
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
        dnorm=max(-1.0,min(1.0, float(d)/max(1e-6, self.delta_scale)))
        return xs_seq, xt, torch.tensor(dnorm,dtype=torch.float32), s, t

# ---------------- Utils ----------------
def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay=decay
        self.shadow={k:v.clone().detach() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k,v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# 余弦 alpha_bar
class CosineSchedule:
    def __init__(self, N:int, s:float=0.008):
        self.N = N; self.s = s
    def ab(self, t: torch.Tensor):
        x = (t.float()+0.5)/self.N
        s = self.s
        return (torch.cos((x + s) / (1+s) * torch.pi/2) ** 2).clamp(1e-6, 1.0)

    # DDIM eta=0
    def ddim_step(self, x_t, eps_or_v, ab_t, ab_prev, *, v_pred: bool):
        a_t = ab_t.sqrt().view(-1,1,1,1)
        s_t = (1 - ab_t).sqrt().view(-1,1,1,1)
        if v_pred:
            # x0 = a*x - s*v ; eps = s*x + a*v
            x0_hat = a_t * x_t - s_t * eps_or_v
            eps_hat = s_t * x_t + a_t * eps_or_v
        else:
            eps_hat = eps_or_v
            x0_hat = (x_t - s_t * eps_hat) / (a_t + 1e-8)

        a_prev = ab_prev.sqrt().view(-1,1,1,1)
        s_prev = (1 - ab_prev).sqrt().view(-1,1,1,1)
        x_prev = a_prev * x0_hat + s_prev * eps_hat
        return x_prev, x0_hat

# ---- RNN 加载与 meta/尺度对齐 ----
def _infer_rnn_hyper_from_state_dict(sd):
    hidden = None; tdim = None; num_layers = 2
    # PredRNN: 入口卷积 out_ch
    for k, v in sd.items():
        if k.endswith("in_conv.weight") and v.ndim == 4:
            hidden = int(v.shape[0])
            break
    # FiLM MLP 输入维度
    for k in sd.keys():
        if "film.mlp.0.weight" in k:
            tdim = int(sd[k].shape[1]); break
    if hidden is None: hidden = 96
    if tdim is None:   tdim = 128
    return hidden, tdim, num_layers

def load_rnn_from_ckpt(rnn_ckpt_path, device):
    if not (rnn_ckpt_path and os.path.exists(rnn_ckpt_path)):
        return None, {}
    ck = torch.load(rnn_ckpt_path, map_location="cpu")
    meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
    sd   = ck.get("model", ck)
    hidden, tdim, num_layers = _infer_rnn_hyper_from_state_dict(sd)

    model = LatentDynamicsPredRNN(
        z_channels=4, hidden=hidden, num_layers=num_layers,
        tdim=tdim, n_res_per_layer=1
    ).to(device).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)  # 更稳健
    for p in model.parameters(): p.requires_grad_(False)

    # 读取 sf（RNN 训练脚本已保存），兼容无 meta 的情况
    sf_rnn = float(meta.get("sf", 1.0))
    ctx_rnn = int(meta.get("context_len", 3))
    return model, {"sf": sf_rnn, "context_len": ctx_rnn, "hidden": hidden, "tdim": tdim}

# ---------------- Arg parse ----------------
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
            add_bool_arg(ap, k, v)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    return ap.parse_args([]) if len(sys.argv)==1 else ap.parse_args()

# ---------------- Train ----------------
def main():
    args=build_args()
    set_seed(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"

    # 打印所有配置参数
    print("========== TRAIN CONFIG ==========")
    for k in DEFAULTS.keys(): print(f"{k}: {getattr(args,k)}")
    print("==================================")
    print("[env] torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[env] device:", torch.cuda.get_device_name(0))
        print("[env] capability:", torch.cuda.get_device_capability(0))

    # AMP
    if args.precision=="fp16" and device=="cuda":
        cast=lambda: torch.amp.autocast("cuda", dtype=torch.float16); scaler=torch.amp.GradScaler("cuda")
    elif args.precision=="bf16" and device=="cuda":
        cast=lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16); scaler=None
    else:
        cast=lambda: nullcontext(); scaler=None

    # out dirs
    ts=time.strftime("%Y%m%d-%H%M%S")
    run_dir=Path(args.outroot)/ts
    (run_dir/"ckpts").mkdir(parents=True, exist_ok=True)
    (run_dir/"images").mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}")

    # data（与 RNN 训练保持一致的“相邻上下文”）
    train_ds=PairListAdjacentCtx(args.pairlist, img_size=args.img_size, context_len=max(args.rnn_context, 1))
    val_ds  =PairListAdjacentCtx(args.vallist,  img_size=args.img_size, delta_scale=train_ds.delta_scale, context_len=max(args.rnn_context, 1))
    dl=DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    vl=DataLoader(val_ds,   batch_size=min(args.batch,8), shuffle=False, num_workers=args.num_workers, pin_memory=True)
    steps_per_epoch = math.ceil(len(train_ds)/max(1,args.batch))
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  batch={args.batch}  steps/epoch≈{steps_per_epoch}")
    print(f"[data] img_size={args.img_size}  rnn_context={args.rnn_context}  delta_scale(train)≈{train_ds.delta_scale:.6f}")

    # VAE (frozen)
    vae=AutoencoderKL.from_pretrained(args.vae_name).to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    sf = float(getattr(getattr(vae,"config",None),"scaling_factor", 1.0))
    print(f"[debug] vae.scaling_factor = {sf:.5f}")

    # UNet（注意：use_prev/use_delta 会改变网络第一层/FiLM 尺寸）
    net = UNetLatentEps(
        zc=4, base=args.base, depth=args.depth, n_res=args.n_res,
        tdim=args.tdim, emb_ch=args.emb_ch,
        use_prev=args.use_prev, use_delta=args.use_delta
    ).to(device)
    total_p, train_p = count_params(net)
    print(f"[unet] params total={total_p/1e6:.2f}M  trainable={train_p/1e6:.2f}M")
    named = dict(net.named_modules())
    has_prev = ("prev_enc" in named) and ("prev_proj" in named)
    has_delta = ("delta_mlp" in named)
    print(f"[unet] use_prev={args.use_prev} (has_prev_layers={has_prev})  | use_delta={args.use_delta} (has_delta_layers={has_delta})")
    if args.use_prev:  assert has_prev,  "use_prev=True 但 UNet 缺少 prev_enc/prev_proj"
    if args.use_delta: assert has_delta, "use_delta=True 但 UNet 缺少 delta_mlp"

    # RNN 先验（可选）+ 尺度对齐
    rnn = None; rnn_meta = {}
    if args.use_prev and args.rnn_ckpt and os.path.exists(args.rnn_ckpt):
        rnn, rnn_meta = load_rnn_from_ckpt(args.rnn_ckpt, device=device)
        if rnn is not None:
            print(f"[load rnn] {args.rnn_ckpt}  (hidden={rnn_meta.get('hidden')}  tdim={rnn_meta.get('tdim')})")
            print(f"[rnn] meta: sf={rnn_meta.get('sf',1.0)}  context_len={rnn_meta.get('context_len')}")
    elif args.use_prev:
        print("[warn] use_prev=True 但未加载 RNN；将退化为 z_prev（上一帧）作为条件先验")

    # 调度与优化器
    sched = CosineSchedule(N=args.noise_steps)
    opt=torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    ema = EMA(net, args.ema_decay) if args.ema else None

    print("========== TRAIN SUMMARY =========")
    print(f"device={device}  precision={args.precision}  ema={args.ema}  ema_decay={args.ema_decay}")
    print(f"noise_steps(T)={args.noise_steps}  v_pred={args.v_pred}  minsnr_gamma={args.minsnr_gamma}")
    print(f"val_mode={args.val_mode}  warmstart_sigma={args.warmstart_sigma}")
    print("==================================")

    # 训练
    for epoch in range(args.epochs):
        net.train()
        pbar=tqdm(dl, desc=f"Train e{epoch}", ncols=120)
        for it, (xs_seq, xt, dnorm, *_ ) in enumerate(pbar, 1):
            xs_seq=xs_seq.to(device); xt=xt.to(device); dnorm=dnorm.to(device)
            B,L,_,H,W = xs_seq.shape

            # encode -> scaled latents （×sf）
            xs_flat=xs_seq.view(B*L,3,H,W)
            z_seq = vae.encode(xs_flat*2-1).latent_dist.mode().mul(sf).view(B,L,4,H//8,W//8)
            z_prev = z_seq[:,-1]
            z_tar  = vae.encode(xt*2-1).latent_dist.mode().mul(sf)

            # 组装条件先验 z_cond：优先 RNN，退化为 z_prev
            z_cond = None
            if args.use_prev:
                if rnn is not None:
                    z_ctx = z_seq[:, -args.rnn_context:]                    # (B, Lc, 4,h,w)  ×sf
                    z_cond = rnn(z_ctx, dnorm)                              # 期望也在 ×sf 尺度
                    sf_rnn = float(rnn_meta.get("sf", 1.0))
                    if abs(sf_rnn - sf) > 1e-6:
                        z_cond = z_cond * (sf / sf_rnn)                     # 尺度对齐
                else:
                    z_cond = z_prev

            # 诊断：范数
            if it % max(1,int(args.log_every)) == 0:
                with torch.no_grad():
                    def _n(t): return t.float().pow(2).mean().sqrt().item()
                    if z_cond is not None:
                        print(f"[diag] ||z_prev||={_n(z_prev):.4f}  ||z_tar||={_n(z_tar):.4f}  "
                              f"||z_cond||={_n(z_cond):.4f}  ||z_cond-z_prev||={_n(z_cond-z_prev):.4f}")

            # sample timestep & construct x_t
            t = torch.randint(0, sched.N, (B,), device=device)
            ab_t = sched.ab(t)                        # (B,)
            a = ab_t.sqrt().view(B,1,1,1)
            s = (1 - ab_t).sqrt().view(B,1,1,1)
            eps = torch.randn_like(z_tar)
            x_t = a * z_tar + s * eps

            # targets
            target = a * eps - s * z_tar if args.v_pred else eps

            # MinSNR weighting
            if args.minsnr_gamma and args.minsnr_gamma > 0:
                snr = (ab_t / (1 - ab_t)).clamp(min=1e-6)
                w = torch.minimum(torch.full_like(snr, args.minsnr_gamma), snr) / snr
                w = w.view(B,1,1,1)
            else:
                w = 1.0

            # emb
            t_emb = timestep_embedding(t, args.tdim).to(device)

            # 先验安全门：若 z_cond 与 z_prev 差异过大，衰减其作用
            prior_scale = 1.0
            if z_cond is not None:
                with torch.no_grad():
                    diff = (z_cond - z_prev).float().pow(2).mean().sqrt()
                    base = z_prev.float().pow(2).mean().sqrt() + 1e-8
                    ratio = (diff / base).item()
                    if ratio > 2.5:      # 阈值可调
                        prior_scale = 0.0
                    elif ratio > 1.5:
                        prior_scale = 0.3
            z_cond_eff = (z_cond * prior_scale) if z_cond is not None else None

            with cast():
                pred = net(
                    x_t, t_emb,
                    z_cond_eff if args.use_prev else None,
                    dnorm if args.use_delta else None
                )
                loss = F.mse_loss(pred, target, reduction="none")
                loss = (loss * w).mean()

                # 先验对齐（可选）
                if args.use_prev and args.lambda_prior > 0 and z_cond_eff is not None and prior_scale > 0:
                    if args.v_pred:
                        x0_hat = a * x_t - s * pred
                    else:
                        x0_hat = (x_t - s * pred) / (a + 1e-8)
                    loss_prior = F.mse_loss(x0_hat, z_cond_eff.detach())
                    loss = loss + args.lambda_prior * loss_prior

            opt.zero_grad(set_to_none=True)
            if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
            if ema: ema.update(net)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---------- 验证 ----------
        with torch.no_grad():
            # 用 EMA 做评估
            net_eval = net
            if ema:
                net_eval = UNetLatentEps(
                    zc=4, base=args.base, depth=args.depth, n_res=args.n_res,
                    tdim=args.tdim, emb_ch=args.emb_ch,
                    use_prev=args.use_prev, use_delta=args.use_delta
                ).to(device)
                net_eval.load_state_dict(ema.shadow, strict=True)
                net_eval.eval()

            xs_seq, xt, dnorm, sp, tp = next(iter(vl))
            xs_seq=xs_seq.to(device); xt=xt.to(device); dnorm=dnorm.to(device)
            B,L,_,H,W = xs_seq.shape
            xs_flat = xs_seq.view(B*L,3,H,W)
            z_seq = vae.encode(xs_flat*2-1).latent_dist.mode().mul(sf).view(B,L,4,H//8,W//8)
            z_prev = z_seq[:,-1]
            z_tar  = vae.encode(xt*2-1).latent_dist.mode().mul(sf)

            # 条件先验
            z_cond = None
            if args.use_prev:
                if rnn is not None:
                    z_cond = rnn(z_seq[:, -args.rnn_context:], dnorm)
                    sf_rnn = float(rnn_meta.get("sf", 1.0))
                    if abs(sf_rnn - sf) > 1e-6:
                        z_cond = z_cond * (sf / sf_rnn)
                else:
                    z_cond = z_prev

            # warmstart（保守）或 pure
            if args.val_mode == "warmstart" and z_cond is not None:
                sigma = float(args.warmstart_sigma)
                ab0 = max(1e-6, 1 - sigma**2)
                a0 = ab0**0.5; s0 = (1-ab0)**0.5
                x = a0 * z_cond + s0 * torch.randn_like(z_prev)
            else:
                x = torch.randn_like(z_prev)

            for n in reversed(range(sched.N)):
                n_t = torch.full((B,), n, device=device, dtype=torch.long)
                ab_t = sched.ab(n_t)
                t_emb = timestep_embedding(n_t, args.tdim).to(device)
                pred = net_eval(
                    x, t_emb,
                    z_cond if args.use_prev else None,
                    dnorm if args.use_delta else None
                )
                if n > 0:
                    ab_prev = sched.ab(n_t - 1)
                    x, x0_hat = sched.ddim_step(x, pred, ab_t, ab_prev, v_pred=args.v_pred)
                else:
                    a_t = ab_t.sqrt().view(B,1,1,1)
                    s_t = (1 - ab_t).sqrt().view(B,1,1,1)
                    x0_hat = a_t * x - s_t * pred if args.v_pred else (x - s_t * pred) / (a_t + 1e-8)
                    x = x0_hat

            x_rec = (vae.decode(x.div(sf)).sample + 1)/2
            x_rec = x_rec.clamp(0,1)
            grid = make_grid(torch.cat([xs_seq[:,-1], xt, x_rec], dim=0), nrow=xs_seq.size(0), padding=2)
            save_image(grid.float().cpu(), run_dir/"images"/f"val_e{epoch:03d}.png")

            # ---- Sanity（teacher-forced 单步反推）----
            t = torch.randint(0, sched.N, (B,), device=device)
            ab_t = sched.ab(t); a = ab_t.sqrt().view(B,1,1,1); s = (1 - ab_t).sqrt().view(B,1,1,1)
            eps = torch.randn_like(z_tar)
            x_t = a * z_tar + s * eps
            t_emb = timestep_embedding(t, args.tdim).to(device)
            pred = net_eval(
                x_t, t_emb,
                z_cond if args.use_prev else None,
                dnorm if args.use_delta else None
            )
            x0_hat = a * x_t - s * pred if args.v_pred else (x_t - s * pred) / (a + 1e-8)

            x_tf = (vae.decode(z_tar.div(sf)).sample + 1) / 2
            x_1s = (vae.decode(x0_hat.div(sf)).sample + 1) / 2
            grid = make_grid(torch.cat([xs_seq[:, -1], xt, x_tf.clamp(0, 1), x_1s.clamp(0, 1)], dim=0),
                             nrow=xs_seq.size(0), padding=2)
            save_image(grid.float().cpu(), run_dir / "images" / f"sanity_e{epoch:03d}.png")

        # ---------- 保存 ----------
        meta = dict(
            zc=4, base=args.base, depth=args.depth, n_res=args.n_res,
            tdim=args.tdim, emb_ch=args.emb_ch,
            context_len=args.rnn_context, delta_scale=train_ds.delta_scale, noise_steps=args.noise_steps,
            use_prev=args.use_prev, use_delta=args.use_delta,
            v_pred=args.v_pred, minsnr_gamma=args.minsnr_gamma, sf=sf,
        )
        torch.save({"model": net.state_dict(), "ema": (ema.shadow if ema else None), "meta": meta},
                   run_dir/"ckpts"/"latest.ckpt")
        if ema:
            torch.save({"model": ema.shadow, "ema": ema.shadow, "meta": meta},
                       run_dir/"ckpts"/"latest_ema.ckpt")

    print("[done]", str(run_dir))

if __name__=="__main__":
    main()
