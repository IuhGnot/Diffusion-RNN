# src/models/diffusion_unet_old.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """
    简单 FiLM：ReLU -> Linear(out=2*ch)，拆成 gamma/beta。
    键名保持兼容：film.mlp.1.*
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class ResBlock(nn.Module):
    """
    旧权重键名对齐：
      res.{i}.n1 / c1 / n2 / c2 / film.mlp.1
    """
    def __init__(self, ch, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = FiLM(tdim, 2 * ch)

    def forward(self, h, emb):   # emb: (B, tdim)
        B, C, H, W = h.shape
        gamma_beta = self.film(emb).view(B, 2, C, 1, 1)
        gamma, beta = gamma_beta[:, 0], gamma_beta[:, 1]
        h = self.c1(F.silu(self.n1(h) * (1 + gamma) + beta))
        h = self.c2(F.silu(self.n2(h) * (1 + gamma) + beta))
        return h


class UNetLatentEps(nn.Module):
    """
    旧架构 ε 网络（键名与旧 ckpt 对齐）：
      - 输入：concat([x_t, z_prev]) -> 8 通道
      - time_mlp:  Linear(tdim,tdim)–SiLU–Linear(tdim,tdim)  （键：time_mlp.0/.2）
      - delta_mlp: Linear(delta_in,tdim)–SiLU–Linear(tdim,tdim)（键：delta_mlp.0/.2）
        *注意*：旧权重里 delta_in=1（标量 Δ），所以默认就是 1。
    """
    def __init__(self, zc=4, base=128, depth=4, tdim=256, cond=True, delta_in=1):
        super().__init__()
        in_ch = zc * 2 if cond else zc
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        # ---- 关键修正：delta_mlp 输入维度用 delta_in（默认为 1）----
        self.delta_mlp = nn.Sequential(
            nn.Linear(delta_in, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        self.res = nn.ModuleList([ResBlock(base, tdim) for _ in range(depth)])
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, zc, 3, padding=1)

        self.cfg = dict(zc=zc, base=base, depth=depth, tdim=tdim, cond=cond, delta_in=delta_in)

    def forward(self, x_t, z_prev, t_emb, d_emb):
        """
        x_t   : (B,4,h,w)
        z_prev: (B,4,h,w)
        t_emb : (B,tdim)  —— 原训练里就是 256 维数值/嵌入，交由 time_mlp 处理
        d_emb : (B,delta_in)  —— 旧权重里 delta_in=1（标量 Δ）
        """
        x = torch.cat([x_t, z_prev], dim=1) if z_prev is not None else x_t
        h = self.in_conv(x)

        te = self.time_mlp(t_emb)
        de = self.delta_mlp(d_emb)
        emb = te + de

        for blk in self.res:
            h = h + blk(h, emb)

        h = F.silu(self.out_norm(h))
        eps = self.out_conv(h)
        return eps
