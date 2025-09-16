# src/models/diffusion_rnn_predrnn.py
import math
import torch
from torch import nn
import torch.nn.functional as F

# ---------- 小工具 ----------
def sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) 已经是 [-1,1] 或 [0,1] 的标量条件（你外部会 clamp）
    return: (B, dim)
    """
    device = t.device
    half = dim // 2
    # 比较宽的频率带宽，适合小位移的时间刻度
    freqs = torch.exp(torch.linspace(0., 9., half, device=device))
    ang = t[:, None] * freqs[None, :] * 2.0 * math.pi
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=device, dtype=emb.dtype)], dim=1)
    return emb

class FiLM(nn.Module):
    """ FiLM：把一维条件注入到特征（逐通道仿射） """
    def __init__(self, ch: int, tdim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tdim, ch*2),
            nn.SiLU(),
            nn.Linear(ch*2, ch*2),
        )
    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W), temb: (B,tdim)
        gb = self.mlp(temb)           # (B, 2C)
        gamma, beta = gb.chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

def norm2d(c):     # GroupNorm 稳定一些
    return nn.GroupNorm(8, c)

# ---------- Causal LSTM（PredRNN核心单元，双记忆：C_t / M_t） ----------
class CausalLSTMCell(nn.Module):
    """
    参考 PredRNN/PredRNN++ 的 Causal LSTM 思路做简化实现：
      - 维护两个状态：C_t（cell）和 M_t（st-lstm memory）
      - 门控包含 x/h/c/m 多路交互
      - 用 FiLM 注入标量条件（如 dnorm）
    """
    def __init__(self, in_ch: int, hid_ch: int, tdim: int):
        super().__init__()
        self.in_ch = in_ch
        self.hid_ch = hid_ch

        # 融合多路输入后，再分成多个门
        self.x2h = nn.Conv2d(in_ch,        4*hid_ch, 3, padding=1)
        self.h2h = nn.Conv2d(hid_ch,       4*hid_ch, 3, padding=1)
        self.c2h = nn.Conv2d(hid_ch,       3*hid_ch, 3, padding=1)
        self.m2h = nn.Conv2d(hid_ch,       3*hid_ch, 3, padding=1)

        self.film_x = FiLM(4*hid_ch, tdim)
        self.film_h = FiLM(4*hid_ch, tdim)
        self.film_c = FiLM(3*hid_ch, tdim)
        self.film_m = FiLM(3*hid_ch, tdim)

        self.out_x = nn.Conv2d(in_ch,  hid_ch, 3, padding=1)
        self.out_h = nn.Conv2d(hid_ch, hid_ch, 3, padding=1)
        self.out_m = nn.Conv2d(hid_ch, hid_ch, 3, padding=1)

        self.nx = norm2d(hid_ch)
        self.nh = norm2d(hid_ch)
        self.nm = norm2d(hid_ch)

        self.act = nn.SiLU()

    def forward(self, x, h, c, m, temb):
        """
        x: (B,in_ch,H,W)
        h,c,m: (B,hid,H,W)  若为 None 自动置 0
        temb: (B,tdim)
        return: h_new, c_new, m_new
        """
        B, _, H, W = x.shape
        if h is None:
            z = torch.zeros(B, self.hid_ch, H, W, device=x.device, dtype=x.dtype)
            h = z; c = z; m = z

        xh = self.film_x(self.x2h(x), temb)
        hh = self.film_h(self.h2h(h), temb)
        ch = self.film_c(self.c2h(c), temb)
        mh = self.film_m(self.m2h(m), temb)

        # x/h 产生的门（i,f,g,o）
        xi, xf, xg, xo = xh.chunk(4, dim=1)
        hi, hf, hg, ho = hh.chunk(4, dim=1)

        # c/m 产生的门（i', f', g'）
        ci, cf, cg = ch.chunk(3, dim=1)
        mi, mf, mg = mh.chunk(3, dim=1)

        i = torch.sigmoid(xi + hi + ci + mi)     # input
        f = torch.sigmoid(xf + hf + cf + mf)     # forget
        g = torch.tanh(   xg + hg + cg + mg)     # candidate

        c_new = f * c + i * g

        o = torch.sigmoid(xo + ho + self.out_x(x) + self.out_h(h) + self.out_m(m))
        h_new = o * torch.tanh(self.nx(c_new))

        # 更新 M（spatiotemporal memory）: 这里用一个简洁门控
        m_in  = self.nh(h_new) + self.nm(m)
        m_new = torch.tanh(m_in)

        return h_new, c_new, m_new

# ---------- 一个 PredRNN Block（Conv 编码 + CausalLSTMCell + Conv 解码） ----------
class PredRNNBlock(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, tdim: int, n_res: int = 1):
        super().__init__()
        self.enc = nn.Conv2d(in_ch, hid_ch, 3, padding=1)
        self.cell = CausalLSTMCell(hid_ch, hid_ch, tdim)
        self.res = nn.ModuleList([
            nn.Sequential(norm2d(hid_ch), nn.SiLU(), nn.Conv2d(hid_ch, hid_ch, 3, padding=1))
            for _ in range(n_res)
        ])
        self.dec = nn.Conv2d(hid_ch, hid_ch, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x_seq, temb):
        """
        x_seq: (B,L,C,H,W) 输入序列（已在外部统一到相同通道）
        temb : (B,tdim)
        return: h_T, states  （返回最后时刻隐藏状态，和最终的(c,m)以供堆叠层使用）
        """
        B, L, C, H, W = x_seq.shape
        h = c = m = None
        for t in range(L):
            x = self.act(self.enc(x_seq[:, t]))
            h, c, m = self.cell(x, h, c, m, temb)
            for rb in self.res:
                h = h + rb(h)
        out = self.dec(self.act(h))
        return out, (c, m)

# ---------- 顶层：LatentDynamicsPredRNN（Δz 残差预测） ----------
class LatentDynamicsPredRNN(nn.Module):
    """
    输入:  z_seq: (B,L,zc,h,w) 或单帧 (B,zc,h,w)
           dnorm: (B,)   [-1,1] 或 [0,1] 的标量条件
    输出:  z_last + Δz   （其中 Δz 由网络预测）
    """
    def __init__(self, z_channels=4, hidden=96, num_layers=2, tdim=128, n_res_per_layer=1):
        super().__init__()
        self.zc = z_channels
        self.hid = hidden
        self.tdim = tdim

        self.t_proj = nn.Sequential(
            nn.Linear(tdim, tdim), nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        # 多层堆叠
        self.layers = nn.ModuleList([
            PredRNNBlock(in_ch=(z_channels if i == 0 else hidden),
                         hid_ch=hidden, tdim=tdim, n_res=n_res_per_layer)
            for i in range(num_layers)
        ])

        # 预测 Δz
        self.head = nn.Sequential(
            norm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, z_channels, 3, padding=1)
        )

    @torch.no_grad()
    def params_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, z_in, dnorm):
        """
        z_in: (B,L,C,H,W) or (B,C,H,W)
        dnorm: (B,)
        """
        if z_in.dim() == 4:
            z_seq = z_in.unsqueeze(1)   # (B,1,C,H,W)
        elif z_in.dim() == 5:
            z_seq = z_in
        else:
            raise ValueError(f"z_in must be 4D or 5D, got {z_in.shape}")

        B, L, C, H, W = z_seq.shape
        z_last = z_seq[:, -1]

        t = dnorm.clamp(-1, 1) * 0.5 + 0.5     # 映射到 [0,1]
        temb = sinusoidal_embed(t, self.tdim)
        temb = self.t_proj(temb)

        h = None
        x = z_seq
        for i, layer in enumerate(self.layers):
            x, (c, m) = layer(x, temb)
            # 堆叠：把该层输出作为下一层的输入序列（复用最后一帧的状态）
            # 这里用“把每一时刻都替换成相同的最后隐藏”，让下一层专注于聚合 T 维的高阶特征
            x = x.unsqueeze(1).repeat(1, L, 1, 1, 1)

        delta = self.head(x[:, -1])    # (B,zc,H,W)
        return z_last + delta
