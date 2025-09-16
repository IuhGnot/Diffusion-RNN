# src/models/diffusion_rnn.py
import torch
from torch import nn
import torch.nn.functional as F

# --------- 小工具 ---------
def sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) in [0,1]
    return: (B, dim)
    """
    device = t.device
    half = dim // 2
    # 频率范围略宽一些，有助于时间分辨 # [CHG]
    freqs = torch.exp(torch.linspace(0, 9, half, device=device))
    ang = t[:, None] * freqs[None, :] * 2.0 * 3.141592653589793
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=device)], dim=1)
    return emb

class FiLM(nn.Module):
    def __init__(self, hdim: int, tdim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tdim, hdim*2),
            nn.SiLU(),
            nn.Linear(hdim*2, hdim*2),
        )
    def forward(self, h, temb):
        gamma_beta = self.mlp(temb)  # (B,2H)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

class ResBlock(nn.Module):
    def __init__(self, ch: int, tdim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = FiLM(ch, tdim)
        self.act = nn.SiLU()
    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.film(h, temb)
        h = self.conv2(self.act(self.norm2(h)))
        return x + h

class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, h_ch, tdim):
        super().__init__()
        self.conv_zr = nn.Conv2d(in_ch + h_ch, 2*h_ch, 3, padding=1)
        self.conv_h  = nn.Conv2d(in_ch + h_ch, h_ch, 3, padding=1)
        self.film = FiLM(h_ch, tdim)
    def forward(self, x, h, temb):
        if h is None:
            h = torch.zeros(x.shape[0], self.conv_h.out_channels, x.shape[2], x.shape[3],
                            device=x.device, dtype=x.dtype)
        xh = torch.cat([x, h], dim=1)
        zr = self.conv_zr(xh)
        z, r = zr.chunk(2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        xh2 = torch.cat([x, r*h], dim=1)
        n = torch.tanh(self.conv_h(xh2))
        n = self.film(n, temb)
        h_new = (1 - z) * h + z * n
        return h_new

class LatentDynamicsRNN(nn.Module):
    """
    预测潜空间下一时刻：
      - 单帧：      z_hat = z_s + f(z_s, Δ)
      - 多帧序列：  z_hat = z_L + f([z_1..z_L], Δ)   （z_L 为最近一帧）
    """
    def __init__(self, z_channels=4, hidden=96, num_res=2, tdim=128):  # [CHG] hidden 默认 96
        super().__init__()
        self.zc = z_channels
        self.tdim = tdim

        self.t_proj = nn.Sequential(
            nn.Linear(tdim, tdim), nn.SiLU(),
            nn.Linear(tdim, tdim)
        )
        self.in_conv = nn.Conv2d(z_channels, hidden, 3, padding=1)
        self.res = nn.ModuleList([ResBlock(hidden, tdim) for _ in range(num_res)])
        self.gru = ConvGRUCell(hidden, hidden, tdim)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, z_channels, 3, padding=1)
        )

    def _step(self, z, h, temb):
        h_in = self.in_conv(z)
        for rb in self.res:
            h_in = rb(h_in, temb)
        h_out = self.gru(h_in, h, temb)
        return h_out

    def forward(self, z_in, dnorm):
        """
        z_in: (B,C,H,W) 或 (B,L,C,H,W)
        dnorm: (B,)
        """
        temb = sinusoidal_embed(dnorm.clamp(0,1), self.tdim)
        temb = self.t_proj(temb)

        if z_in.dim() == 4:
            # 单帧
            h = self._step(z_in, None, temb)
            delta = self.out_conv(h)
            return z_in + delta
        elif z_in.dim() == 5:  # [CHG] 支持多帧
            B, L, C, H, W = z_in.shape
            h = None
            for t in range(L):
                h = self._step(z_in[:, t], h, temb)
            delta = self.out_conv(h)
            return z_in[:, -1] + delta  # 基于最后一帧做残差
        else:
            raise ValueError(f"z_in must be 4D or 5D, got {z_in.shape}")
