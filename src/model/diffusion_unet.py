# src/models/diffusion_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# ---- time embedding ----
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):  # t: (B,) or (B,1)
        if t.ndim == 2:
            t = t.squeeze(-1)
        device = t.device
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=device) * -(torch.log(torch.tensor(10000.0)) / (half - 1)))
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

def make_mlp(in_dim, out_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim)
    )

# ---- basic blocks ----
class ResBlock(nn.Module):
    def __init__(self, ch, tdim, cond_dim=0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.emb = nn.Linear(tdim, ch)
        self.film = make_mlp(cond_dim, ch*2) if cond_dim>0 else None
        self.act = nn.SiLU()

    def forward(self, x, t_emb, cond=None):
        h = self.conv1(self.act(self.norm1(x)))
        # time
        h = h + self.emb(t_emb)[:, :, None, None]
        # FiLM
        if self.film is not None and cond is not None:
            gamma_beta = self.film(cond)  # (B, 2C)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return x + h

class ConvGRU2d(nn.Module):
    def __init__(self, ch, cond_dim=0):
        super().__init__()
        self.conv_zr = nn.Conv2d(ch*2, ch*2, 3, padding=1)
        self.conv_h  = nn.Conv2d(ch*2, ch, 3, padding=1)
        self.film = make_mlp(cond_dim, ch*2) if cond_dim>0 else None
        self.act = nn.SiLU()

    def forward(self, x, h, cond=None):
        if h is None: h = torch.zeros_like(x)
        inp = torch.cat([x, h], dim=1)
        zr = self.conv_zr(inp)
        z, r = torch.chunk(torch.sigmoid(zr), 2, dim=1)
        h_tilde = self.act(self.conv_h(torch.cat([x, r*h], dim=1)))
        if self.film is not None and cond is not None:
            gb = self.film(cond)
            g, b = gb.chunk(2, dim=1)
            h_tilde = h_tilde * (1 + g[:, :, None, None]) + b[:, :, None, None]
        h_new = (1 - z)*h + z*h_tilde
        return h_new

# ---- UNet for eps prediction ----
class UNetLatentEps(nn.Module):
    """
    统一的构造签名：
      UNetLatentEps(
        z_channels=4,        # latent 通道数
        base=64,             # 基础宽度
        depth=3,             # 若干层 ResBlock
        tdim=128,            # 时间嵌入维度
        cond_prev=True,      # 是否用上一帧 z_{t-1} 作为条件
        cond_rnn=False,      # 是否用 RNN 草稿 z_rnn 作为条件
      )
    forward(x_t, t, z_prev=None, z_rnn=None) -> eps_pred
    """
    def __init__(self, z_channels=4, base=64, depth=3, tdim=128,
                 cond_prev=True, cond_rnn=False):
        super().__init__()
        self.zc = z_channels
        self.base = base
        self.depth = depth
        self.tdim = tdim
        self.cond_prev = cond_prev
        self.cond_rnn = cond_rnn

        self.t_emb = nn.Sequential(
            SinusoidalPosEmb(tdim),
            nn.Linear(tdim, tdim*4), nn.SiLU(),
            nn.Linear(tdim*4, tdim)
        )

        cond_dim = 0
        if cond_prev: cond_dim += z_channels
        if cond_rnn:  cond_dim += z_channels

        self.in_conv = nn.Conv2d(z_channels, base, 3, padding=1)
        self.res = nn.ModuleList([ResBlock(base, tdim, cond_dim) for _ in range(depth)])
        self.gru = ConvGRU2d(base, cond_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, z_channels, 3, padding=1)
        )

    def encode_cond(self, z_prev, z_rnn):
        conds = []
        if self.cond_prev and z_prev is not None:
            # 池化到通道统计
            conds.append(z_prev.mean(dim=(2,3)))
        if self.cond_rnn and z_rnn is not None:
            conds.append(z_rnn.mean(dim=(2,3)))
        if len(conds)==0:
            return None
        return torch.cat(conds, dim=1)

    def forward(self, x_t, t, z_prev=None, z_rnn=None):
        # t: (B,) or (B,1)
        t_emb = self.t_emb(t.float())
        cond = self.encode_cond(z_prev, z_rnn)
        h = self.in_conv(x_t)
        for blk in self.res:
            h = blk(h, t_emb, cond)
        h = self.gru(h, None, cond)
        eps = self.out(h)
        return eps
