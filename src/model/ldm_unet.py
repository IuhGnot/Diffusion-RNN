# src/models/ldm_unet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 时间嵌入 ----------
def timestep_embedding(t: torch.Tensor, dim: int, *, max_period: int = 10000):
    if t.dtype not in (torch.float32, torch.float64):
        t = t.float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=t.dtype, device=t.device) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ---------- 基础模块 ----------
def norm_act(ch):
    return nn.GroupNorm(32, ch), nn.SiLU()

class FiLM(nn.Module):
    def __init__(self, in_dim, out_ch):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_ch * 2)
        )
    def forward(self, h, emb_vec):
        gamma, beta = self.mlp(emb_vec).chunk(2, dim=1)
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()
        self.n1, self.a1 = norm_act(in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.n2, self.a2 = norm_act(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb = nn.Linear(emb_ch, out_ch)
        self.film = FiLM(emb_ch, out_ch)

        self.skip = (in_ch == out_ch)
        if not self.skip:
            self.sc = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_vec, cond_vec):
        h = self.a1(self.n1(x))
        h = self.c1(h)
        h = h + self.emb(t_vec)[:, :, None, None]
        h = self.film(h, cond_vec)
        h = self.a2(self.n2(h))
        h = self.c2(h)
        if self.skip:
            h = h + x
        else:
            h = h + self.sc(x)
        return h

class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.pool(x)

class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


# ---------- UNet ----------
class UNetLatentEps(nn.Module):
    def __init__(
        self,
        zc=4, base=192, depth=3, n_res=2, tdim=256, emb_ch=256,
        use_prev=False, use_delta=False
    ):
        super().__init__()
        self.use_prev = use_prev
        self.use_delta = use_delta
        self.emb_ch = emb_ch

        # 条件统一映射到 emb_ch，然后相加聚合
        self.t_mlp = nn.Sequential(nn.Linear(tdim, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))

        if use_prev:
            self.prev_enc = nn.Sequential(
                nn.Conv2d(4, base, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.prev_proj = nn.Linear(base, emb_ch)

        if use_delta:
            self.delta_mlp = nn.Sequential(nn.Linear(1, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))

        # Stem
        self.in_conv = nn.Conv2d(zc, base, 3, padding=1)
        ch = base

        # Down path（恒定通道）
        self.downs = nn.ModuleList()
        for _ in range(depth):
            blocks = nn.ModuleList([ResBlock(ch, ch, emb_ch) for _ in range(n_res)])
            self.downs.append(blocks)
            self.downs.append(Down(ch))

        # Mid
        self.mid1 = ResBlock(ch, ch, emb_ch)
        self.mid2 = ResBlock(ch, ch, emb_ch)

        # Up path（首个 ResBlock 用 2ch->ch，后续用 ch->ch）✅ 修复点
        self.ups = nn.ModuleList()
        for _ in range(depth):
            first = ResBlock(ch * 2, ch, emb_ch)
            rest = [ResBlock(ch, ch, emb_ch) for _ in range(n_res - 1)]
            blocks = nn.ModuleList([first] + rest)
            self.ups.append(Up(ch))
            self.ups.append(blocks)

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, zc, 3, padding=1)

    def _cond_vectors(self, t_emb, z_prev, delta):
        t_vec = self.t_mlp(t_emb)  # (B,emb_ch)
        cond = t_vec.clone()
        if self.use_prev and z_prev is not None:
            pe = self.prev_enc(z_prev).squeeze(-1).squeeze(-1)   # (B, base)
            cond = cond + self.prev_proj(pe)                     # -> emb_ch
        if self.use_delta and delta is not None:
            if delta.dim() == 1:
                delta = delta[:, None]
            cond = cond + self.delta_mlp(delta)                  # -> emb_ch
        return t_vec, cond

    def forward(self, x, t_emb, z_prev=None, delta=None):
        t_only, cond_vec = self._cond_vectors(t_emb, z_prev, delta)
        h = self.in_conv(x)

        # down
        skips = []
        it = iter(self.downs)
        for blocks in it:
            for rb in blocks:
                h = rb(h, t_only, cond_vec)
            skips.append(h)
            down = next(it)
            h = down(h)

        # mid
        h = self.mid1(h, t_only, cond_vec)
        h = self.mid2(h, t_only, cond_vec)

        # up
        it = iter(self.ups)
        for up in it:
            h = up(h)
            blocks = next(it)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)  # (B, 2ch, H, W)
            for rb in blocks:
                h = rb(h, t_only, cond_vec)

        h = self.out_act(self.out_norm(h))
        out = self.out_conv(h)
        return out
