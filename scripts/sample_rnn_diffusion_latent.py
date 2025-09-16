# scripts/sample_rnn_diffusion_latent.py
# 说明：
# - 兼容你现有的 UNetLatentEps / timestep_embedding / LatentDynamicsRNN / CosineSchedule 设定
# - 支持 pairwise 与 rollout 两种推理方式
# - 支持 warmstart（从“轻度加噪的先验”起步）与纯噪声起步
# - 支持 v-预测 或 ε-预测（从 diffusion ckpt 元数据中自动读取）
# - 允许只用 z_prev 做先验（不用 RNN），或加载你训练好的 RNN 权重做先验


import os, sys, glob, argparse
from pathlib import Path

import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image
from contextlib import nullcontext
from tqdm import tqdm
from diffusers import AutoencoderKL

# 项目内模块
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.ldm_unet import UNetLatentEps, timestep_embedding
from src.models.diffusion_rnn import LatentDynamicsRNN

# ---------------- Defaults ----------------
DEFAULTS = dict(
    pairlist="../data/pairs_all_mixed_val.txt",
    outdir="./samples/over1test",
    runs_root="./runs_diff",                      # 自动找 ckpt 用
    ckpt="./runs_diff/20250915-162357\ckpts\latest.ckpt",                                   # 不给就自动找 runs_root/*/ckpts/latest_ema.ckpt 或 latest.ckpt
    use_ema=True,

    vae_name="stabilityai/sd-vae-ft-mse",
    img_size=256,
    precision="fp32",

    # 推理设置
    mode="pairwise",                              # "pairwise" | "rollout"
    steps=100,                                     # DDIM 步数
    noise_steps=100,                              # 训练用的总步数（用于时间索引下采样）
    schedule="cosine",
    warmstart=True,                               # NEW: True=从“轻度加噪的先验”起步；False=纯噪声
    warmstart_sigma=0.1,                          # NEW: warmstart 噪声强度

    # 条件
    use_prev=True,                                # NEW: None=跟随 ckpt；True/False=覆盖
    use_delta=True,                               # NEW: None=跟随 ckpt；True/False=覆盖
    rnn_ckpt="./runs_rnn/20250914-173909/ckpts/e001_ema.ckpt",                                # NEW: 指定 RNN 权重；缺省则不用 RNN，仅用 z_prev

    # 上下文
    context_len=None,                             # None=跟随 ckpt meta；否则覆盖
    roll_n=16,                                    # rollout 预测步数
    roll_delta=1.0,                               # rollout 时每步的 delta（若数据集中默认都是 1.0，可保持 1.0）

    # 评测
    save_grid=True,
    save_each=True,
    compute_psnr=True,
)

# --- 放在文件顶部附近（imports 之后） ---
def blend_guard(z_prev, z_cond, alpha_dir=0.5, thr_cos=0.0, thr_mag=3.0):
    """
    对 RNN 先验做保守门控：
    - 若与 z_prev 的方向余弦 < thr_cos（如小于0，反向），或
    - 若 Δz 的幅度过大（超过 thr_mag × median(|Δz_prev|) 的倍数），
    则把先验向 z_prev 回拉。
    alpha_dir: 回拉强度（0~1），越大越保守。
    """
    B = z_prev.shape[0]
    v = (z_cond - z_prev).flatten(1)
    v0 = torch.zeros_like(v)
    # 以 z_prev 自身的极小扰动做对比“方向”（等价于“靠近原地”）
    # 也可以改成与“RNN 的上一次预测”对齐；无状态时用 v0 更保守
    cos = torch.sum(v * v0, dim=1) / (v.norm(dim=1) * (v0.norm(dim=1)+1e-6) + 1e-6)
    cos = torch.nan_to_num(cos, nan=1.0)  # v0为0时 → cos=1

    # 粗略幅度门：用通道维的 L2 与一个经验阈值
    mag = v.norm(dim=1)  # (B,)
    med = torch.median(mag).clamp_min(1e-6)
    bad = (cos < thr_cos) | (mag > thr_mag * med)

    if bad.any():
        w = bad.float().view(B,1,1,1)
        z_cond = (1 - alpha_dir*w) * z_cond + (alpha_dir*w) * z_prev
    return z_cond

# ==== REPLACE: robust RNN loader (supports PredRNN & old GRU-RNN) ====
def load_rnn_from_ckpt(rnn_ckpt_path: str, device="cpu"):
    """
    Auto-detect RNN architecture and hyperparams from a checkpoint:
    - Supports old LatentDynamicsRNN (ConvGRU) and new LatentDynamicsPredRNN.
    Returns: (rnn_module.eval(), meta_dict)
    """
    if not rnn_ckpt_path or not os.path.exists(rnn_ckpt_path):
        raise FileNotFoundError(f"[rnn] ckpt not found: {rnn_ckpt_path}")

    ck = torch.load(rnn_ckpt_path, map_location="cpu")
    sd = ck.get("model", ck)
    if not isinstance(sd, dict):
        raise RuntimeError(f"[rnn] invalid state dict in {rnn_ckpt_path}")

    # --- heuristics to detect arch and derive hyperparams ---
    # 1) detect predRNN by presence of 'layers.' prefix
    is_predrnn = any(k.startswith("layers.") for k in sd.keys())

    # 2) find a conv weight that looks like "in_conv": shape [hidden, zc(=4), 3, 3]
    hidden = None
    zc = 4
    for k, w in sd.items():
        if isinstance(w, torch.Tensor) and w.ndim == 4:
            oc, ic, kh, kw = w.shape
            if ic == zc and kh == 3 and kw == 3:
                # prefer keys that end with "in_conv.weight"
                if k.endswith("in_conv.weight"):
                    hidden = oc
                    break
                # otherwise, keep the first candidate
                if hidden is None:
                    hidden = oc
    if hidden is None:
        raise RuntimeError(f"[rnn] cannot infer hidden: no conv weight with in_channels=={zc} and k=3 found in {rnn_ckpt_path}")

    # 3) infer tdim from a FiLM MLP weight: '*.film.mlp.0.weight' -> shape [*, tdim]
    tdim = None
    for k in sd.keys():
        if k.endswith("film.mlp.0.weight") and isinstance(sd[k], torch.Tensor) and sd[k].ndim == 2:
            tdim = int(sd[k].shape[1])
            break
    if tdim is None:
        # fallback
        tdim = 128

    # 4) infer num_layers for PredRNN
    num_layers = 1
    if is_predrnn:
        layer_ids = set()
        for k in sd.keys():
            if k.startswith("layers."):
                try:
                    lid = int(k.split(".")[1])
                    layer_ids.add(lid)
                except:
                    pass
        if layer_ids:
            num_layers = max(layer_ids) + 1

    # --- instantiate the right class and load ---
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    if is_predrnn:
        from src.models.diffusion_rnn_predrnn import LatentDynamicsPredRNN
        rnn = LatentDynamicsPredRNN(
            z_channels=zc,
            hidden=hidden,
            num_layers=num_layers,
            tdim=tdim,
            n_res_per_layer=1
        ).to(device)
        rnn.load_state_dict(sd, strict=True)
        arch = "PredRNN"
    else:
        # old ConvGRU RNN
        from src.models.diffusion_rnn import LatentDynamicsRNN
        # try to infer num_res from keys like 'res.N.*'
        res_ids = set()
        for k in sd.keys():
            if k.startswith("res."):
                try:
                    rid = int(k.split(".")[1])
                    res_ids.add(rid)
                except:
                    pass
        num_res = max(res_ids) + 1 if res_ids else 2
        rnn = LatentDynamicsRNN(
            z_channels=zc,
            hidden=hidden,
            num_res=num_res,
            tdim=tdim
        ).to(device)
        rnn.load_state_dict(sd, strict=True)
        arch = "GRU-RNN"

    for p in rnn.parameters():
        p.requires_grad_(False)
    rnn.eval()
    meta = dict(arch=arch, hidden=hidden, tdim=tdim, num_layers=num_layers if is_predrnn else 1, zc=zc)
    print(f"[rnn] loaded {arch}: hidden={hidden}  tdim={tdim}  layers={meta.get('num_layers', 'na')}")
    return rnn, meta

class PairList(Dataset):
    def __init__(self, pairlist, img_size=256, context_len=3, delta_scale=None):
        self.items=[]
        with open(pairlist,"r",encoding="utf-8") as f:
            for line in f:
                if "|" not in line: continue
                s,t,d=line.strip().split("|")
                self.items.append((s,t,float(d)))
        assert len(self.items)>0, f"Empty pairlist: {pairlist}"
        self.tf=T.Compose([T.Resize((img_size,img_size), antialias=True), T.ToTensor()])

        from collections import Counter
        prev={}
        for s,t,_ in self.items:
            prev.setdefault(t, Counter()).update([s])
        self.prev_of={k:c.most_common(1)[0][0] for k,c in prev.items()}
        self.context_len=context_len

        import numpy as np
        raw=np.array([it[2] for it in self.items], dtype=float)
        self.delta_scale = float(np.percentile(abs(raw),95)) if delta_scale is None else float(delta_scale)

    def _ctx(self, s):
        ctx=[s]; cur=s
        for _ in range(self.context_len-1):
            p=self.prev_of.get(cur,None)
            if p is None: break
            ctx.append(p); cur=p
        ctx=list(reversed(ctx))
        while len(ctx)<self.context_len:
            ctx=[ctx[0]]+ctx
        return ctx

    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        s,t,d=self.items[i]
        ctx_paths=self._ctx(s)
        xs_seq=torch.stack([self._load(p) for p in ctx_paths], dim=0)
        xt=self._load(t)
        dnorm=max(-1.0,min(1.0, float(d)/max(1e-6, getattr(self,'delta_scale',1.0))))
        return xs_seq, xt, torch.tensor(dnorm,dtype=torch.float32), s, t
    def _load(self, p): return self.tf(Image.open(p).convert("RGB"))

# ---------------- Sched ----------------
class CosineSchedule:
    def __init__(self, N:int, s:float=0.008):
        self.N = N; self.s = s
    def ab(self, t: torch.Tensor):
        x = (t.float()+0.5)/self.N
        s = self.s
        return (torch.cos((x + s) / (1+s) * torch.pi/2) ** 2).clamp(1e-6, 1.0)
    def step(self, x_t, pred, ab_t, ab_prev, v_pred: bool):  # DDIM eta=0
        a_t = ab_t.sqrt().view(-1,1,1,1)
        s_t = (1 - ab_t).sqrt().view(-1,1,1,1)
        if v_pred:
            x0_hat = a_t * x_t - s_t * pred
            eps_hat = s_t * x_t + a_t * pred
        else:
            eps_hat = pred
            x0_hat = (x_t - s_t * eps_hat) / (a_t + 1e-8)
        a_prev = ab_prev.sqrt().view(-1,1,1,1)
        s_prev = (1 - ab_prev).sqrt().view(-1,1,1,1)
        x_prev = a_prev * x0_hat + s_prev * eps_hat
        return x_prev, x0_hat

# ---------------- Util ----------------
def auto_find_ckpt(root):
    cands = glob.glob(os.path.join(root, "*", "ckpts", "latest_ema.ckpt"))
    if not cands:
        cands = glob.glob(os.path.join(root, "*", "ckpts", "latest.ckpt"))
    if not cands: return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def psnr(a,b):
    return 10*torch.log10(1.0/((a-b).pow(2).mean().clamp(1e-12)))

def add_bool(ap: argparse.ArgumentParser, name: str, default: bool):
    """为布尔开关同时注册 --name / --no-name，并设置默认值"""
    ap.add_argument(f"--{name}", dest=name, action="store_true")
    ap.add_argument(f"--no-{name}", dest=name, action="store_false")
    ap.set_defaults(**{name: default})

def build_args():
    ap = argparse.ArgumentParser(add_help=False)
    for k, v in DEFAULTS.items():
        # 三态支持：如果默认值是 None，就用显式字符串参数，允许 None 透传
        if v is None:
            ap.add_argument(f"--{k}", type=lambda s: None if s.lower()=="none" else
                                          (s.lower()=="true" or s=="1" or s.lower()=="yes"),
                            default=None)
        elif isinstance(v, bool):
            add_bool(ap, k, v)   # ✅ 正确的布尔默认
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    return ap.parse_args([]) if len(sys.argv)==1 else ap.parse_args()


# ---------------- Main ----------------
def main():
    args = build_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # AMP
    if args.precision=="fp16" and device=="cuda":
        cast=lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    elif args.precision=="bf16" and device=="cuda":
        cast=lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        cast=lambda: nullcontext()

    # ckpt
    if (args.ckpt is None) or (not Path(args.ckpt).exists()):          # [CHG] 更健壮的判断
        auto = auto_find_ckpt(args.runs_root)
        if not auto: raise FileNotFoundError("未找到扩散权重：请指定 --ckpt 或确保 runs_root 下有 ckpt")
        print(f"[auto] ckpt: {auto}")
        args.ckpt = auto
    else:
        print(f"[load] ckpt: {args.ckpt}")                              # [CHG] 显示加载路径

    # outdir
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {outdir}")                                           # [CHG] 显示输出目录

    # VAE
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    sf = float(getattr(getattr(vae,"config",None),"scaling_factor",1.0))
    print(f"[debug] vae.scaling_factor = {sf}")

    # load diffusion
    ck = torch.load(args.ckpt, map_location="cpu")
    meta = ck.get("meta", {})
    use_prev = meta.get("use_prev", False) if args.use_prev is None else args.use_prev
    use_delta = meta.get("use_delta", False) if args.use_delta is None else args.use_delta
    v_pred = bool(meta.get("v_pred", True))
    base = int(meta.get("base", 192))
    depth = int(meta.get("depth", 3))
    n_res = int(meta.get("n_res", 2))
    tdim = int(meta.get("tdim", 256))
    emb_ch = int(meta.get("emb_ch", 256))
    context_len = args.context_len if args.context_len is not None else int(meta.get("context_len", 3))
    noise_steps = int(meta.get("noise_steps", args.noise_steps))
    print(f"[meta] use_prev={use_prev}  use_delta={use_delta}  v_pred={v_pred}  depth={depth} n_res={n_res}  context_len={context_len}, T={noise_steps}")

    net = UNetLatentEps(
        zc=4, base=base, depth=depth, n_res=n_res,
        tdim=tdim, emb_ch=emb_ch,
        use_prev=use_prev, use_delta=use_delta
    ).to(device).eval()

    # 加载模型参数（优先 ema）
    state = ck.get("ema", None) if args.use_ema and ck.get("ema", None) else ck["model"]
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:   print("[load][missing]:", sorted(missing))
    if unexpected:print("[load][unexpected]:", sorted(unexpected))

    # RNN（可选）
    rnn = None
    if use_prev:
        if args.rnn_ckpt and Path(args.rnn_ckpt).exists():
            rnn, _rnn_meta = load_rnn_from_ckpt(args.rnn_ckpt, device=device)
            if rnn is None:
                print("[note] use_prev=True 但加载 RNN 失败；退化为 z_prev 作为条件先验")
        else:
            print("[note] use_prev=True 但未指定/找到 rnn_ckpt；退化为 z_prev 作为条件先验")

    # 数据
    ds = PairList(args.pairlist, img_size=args.img_size, context_len=context_len)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 调度
    sched = CosineSchedule(N=noise_steps)
    # 均匀下采样时间步（大到小）
    import math
    idxs = torch.linspace(0, noise_steps-1, steps=args.steps, dtype=torch.long)
    timesteps = idxs.flip(0)

    # 打印配置摘要
    print("========== SAMPLE CONFIG ==========")                          # [CHG] 增加启动日志
    for k,v in vars(args).items(): print(f"{k}: {v}")
    print("===================================")

    # ============ 推理 ============
    with torch.no_grad(), cast():
        if args.mode == "pairwise":
            run_pairwise(args, dl, device, vae, sf, net, rnn, use_prev, use_delta, v_pred,
                         sched, timesteps, outdir, tdim)
        elif args.mode == "rollout":
            run_rollout(args, dl, device, vae, sf, net, rnn, use_prev, use_delta, v_pred,
                        sched, timesteps, outdir, context_len, tdim)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    print("[done]", str(outdir))

# ---------------- Pairwise 推理 ----------------
def run_pairwise(args, dl, device, vae, sf, net, rnn, use_prev, use_delta, v_pred,
                 sched, timesteps, outdir, tdim):
    psnrs = []
    for i, (xs_seq, xt, dnorm, sp, tp) in enumerate(tqdm(dl, ncols=120, desc=f"sample[{args.mode}]")):
        xs_seq = xs_seq.to(device); xt = xt.to(device); dnorm = dnorm.to(device)
        B,L,_,H,W = xs_seq.shape
        xs_flat = xs_seq.view(B*L,3,H,W)

        # 编码上下文与目标
        z_seq = vae.encode(xs_flat*2-1).latent_dist.mode().mul(sf).view(B,L,4,H//8,W//8)
        z_prev = z_seq[:,-1]
        z_tar  = vae.encode(xt*2-1).latent_dist.mode().mul(sf)  # 仅用于评测/可视化

        # 条件先验
        if use_prev:
            if rnn is not None:
                z_cond = rnn(z_seq[:, -z_seq.size(1):], dnorm)  # 若 RNN 需要固定 context，可换成 [:,-context_len:]
                z_cond = blend_guard(z_prev, z_cond, alpha_dir=0.5, thr_cos=0.0, thr_mag=3.0)
            else:
                z_cond = z_prev
                z_cond = blend_guard(z_prev, z_cond, alpha_dir=0.5, thr_cos=0.0, thr_mag=3.0)
        else:
            z_cond = None

        # warmstart or pure noise
        if args.warmstart and z_cond is not None:
            ab0 = 1 - args.warmstart_sigma**2
            a0 = ab0**0.5; s0 = (1-ab0)**0.5
            x = a0 * z_cond + s0 * torch.randn_like(z_prev)
        else:
            x = torch.randn_like(z_prev)

        # DDIM
        for i, n in enumerate(timesteps):  # timesteps: [t_K, t_{K-1}, ..., t_0]
            n_t = torch.full((B,), int(n.item()), device=device, dtype=torch.long)
            ab_t = sched.ab(n_t)
            t_emb = timestep_embedding(n_t, tdim).to(device)

            pred = net(
                x, t_emb,
                z_cond if use_prev else None,
                dnorm if use_delta else None
            )

            if i < len(timesteps) - 1:  # 还有“上一个被选中的时间步”
                n_prev = timesteps[i + 1]  # **注意：不是 n-1**
                n_prev = torch.full((B,), int(n_prev.item()), device=device, dtype=torch.long)
                ab_prev = sched.ab(n_prev)
                x, x0_hat = sched.step(x, pred, ab_t, ab_prev, v_pred=v_pred)
            else:
                # 最后一跳，直接取 x0
                a_t = ab_t.sqrt().view(B, 1, 1, 1)
                s_t = (1 - ab_t).sqrt().view(B, 1, 1, 1)
                x = a_t * x - s_t * pred if v_pred else (x - s_t * pred) / (a_t + 1e-8)

        # 解码
        x_rec = (vae.decode(x.div(sf)).sample + 1)/2
        x_rec = x_rec.clamp(0,1)

        # 保存
        src_stem = Path(sp[0]).stem if isinstance(sp, (list, tuple)) else f"idx{i:06d}"
        if args.save_each:
            save_image(xs_seq[:, -1], outdir / f"{i:06d}_{src_stem}_src.png")
            save_image(xt, outdir / f"{i:06d}_{src_stem}_tgt.png")
            save_image(x_rec, outdir / f"{i:06d}_{src_stem}_pred.png")
        if args.save_grid:
            grid = make_grid(torch.cat([xs_seq[:, -1], xt, x_rec], dim=0), nrow=1, padding=2)
            save_image(grid.float().cpu(), outdir / f"{i:06d}_{src_stem}_grid.png")

    if args.compute_psnr and psnrs:
        print(f"[pairwise] PSNR={sum(psnrs)/len(psnrs):.2f}dB")

# ---------------- Rollout 连推 ----------------
def run_rollout(args, dl, device, vae, sf, net, rnn, use_prev, use_delta, v_pred,
                sched, timesteps, outdir, context_len, tdim):
    roll_n = args.roll_n
    delta_roll = torch.full((1,), float(args.roll_delta), device=device)

    for i, (xs_seq, xt, dnorm, sp, tp) in enumerate(tqdm(dl, ncols=120, desc=f"rollout x{roll_n}")):
        xs_seq = xs_seq.to(device); dnorm = dnorm.to(device)
        B,L,_,H,W = xs_seq.shape
        xs_flat = xs_seq.view(B*L,3,H,W)
        z_seq = vae.encode(xs_flat*2-1).latent_dist.mode().mul(sf).view(B,L,4,H//8,W//8)
        z_queue = [z_seq[:,j] for j in range(L)]
        decoded = []

        for step in range(roll_n):
            z_prev = z_queue[-1].detach()
            if use_prev:
                if rnn is not None:
                    ctx = torch.stack(z_queue[-context_len:], dim=1)  # (B, Ctx, 4, h, w)
                    z_cond = rnn(ctx, delta_roll)                    # 固定 roll_delta
                else:
                    z_cond = z_prev
            else:
                z_cond = None

            if args.warmstart and z_cond is not None:
                ab0 = 1 - args.warmstart_sigma**2
                a0 = ab0**0.5; s0 = (1-ab0)**0.5
                x = a0 * z_cond + s0 * torch.randn_like(z_prev)
            else:
                x = torch.randn_like(z_prev)

            for n in timesteps:
                n_t = torch.full((B,), int(n.item()), device=device, dtype=torch.long)
                ab_t = sched.ab(n_t)
                t_emb = timestep_embedding(n_t, tdim).to(device)
                pred = net(
                    x, t_emb,
                    z_cond if use_prev else None,
                    delta_roll if use_delta else None
                )
                if n.item() > 0:
                    n_prev = torch.full((B,), int(n.item()-1), device=device, dtype=torch.long)
                    ab_prev = sched.ab(n_prev)
                    x, x0_hat = sched.step(x, pred, ab_t, ab_prev, v_pred=v_pred)
                else:
                    a_t = ab_t.sqrt().view(B,1,1,1)
                    s_t = (1 - ab_t).sqrt().view(B,1,1,1)
                    if v_pred:
                        x0_hat = a_t * x - s_t * pred
                    else:
                        x0_hat = (x - s_t * pred) / (a_t + 1e-8)
                    x = x0_hat

            x_rec = (vae.decode(x.div(sf)).sample + 1)/2
            x_rec = x_rec.clamp(0,1)
            decoded.append(x_rec)
            z_queue.append(x.detach())
            z_queue = z_queue[-context_len:]

        sub = outdir / f"rollout_{i:06d}"
        sub.mkdir(parents=True, exist_ok=True)
        vis = [xs_seq[:,-1]] + decoded
        grid = make_grid(torch.cat(vis, dim=0), nrow=len(vis), padding=2)
        save_image(grid.float().cpu(), sub/"rollout_grid.png")
        for j, img in enumerate(decoded):
            save_image(img, sub/f"pred_{j:03d}.png")

# ---------------- 入口 ----------------
if __name__ == "__main__":                                            # [CHG] 加上主入口
    main()
