# src/models/vae.py
from typing import Optional  # 放到文件顶部
import torch, torch.nn as nn, torch.nn.functional as F


class DiagonalGaussianDistribution:
    """
    与 diffusers 的接口/语义对齐：
    - parameters: (B, 2*C, H, W)，前 C 为 mean，后 C 为 logvar
    - sample():    z ~ N(mean, exp(logvar))
    - mode():      返回 mean
    - kl(reduce):  KL(q || N(0,1))
        * 默认：对 (C,H,W) 求 mean，返回 (B,) 的每样本 KL
        * reduce='mean'：返回标量（batch 平均）
        * reduce='sum' ：返回标量（batch 求和）
    """
    def __init__(self, parameters):
        assert parameters.dim() == 4, "parameters must be (B, 2*C, H, W)"
        n = parameters.shape[1] // 2
        self.mean   = parameters[:, :n]
        self.logvar = parameters[:, n:]
        # 数值稳定
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)

    def sample(self):
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return self.mean + std * eps

    def mode(self):
        return self.mean

    def kl(self, reduce=None):
        # 对通道与空间做 mean（每像素 KL 标尺）
        kl_map = 0.5 * (torch.exp(self.logvar) + self.mean**2 - 1.0 - self.logvar)  # (B,C,H,W)
        kl_per_sample = kl_map.mean(dim=(1, 2, 3))  # (B,)

        if reduce is None:
            return kl_per_sample
        elif reduce == "mean":
            return kl_per_sample.mean()
        elif reduce == "sum":
            return kl_per_sample.sum()
        else:
            raise ValueError("reduce must be one of {None, 'mean', 'sum'}")


# ---------- Building Blocks ----------
class ResnetBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch=None, groups=32, dropout=0.0):
        super().__init__()
        out_ch = out_ch or in_ch
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act   = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)

class AttnBlock2D(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
    def forward(self, x):
        b, c, h, w = x.shape
        h_ = self.norm(x)
        q = self.q(h_).reshape(b, self.num_heads, c//self.num_heads, h*w)
        k = self.k(h_).reshape(b, self.num_heads, c//self.num_heads, h*w)
        v = self.v(h_).reshape(b, self.num_heads, c//self.num_heads, h*w)
        attn = torch.einsum("bncd,bnce->bnde", q, k) * (c//self.num_heads) ** (-0.5)
        attn = attn.softmax(dim=-1)
        out  = torch.einsum("bnde,bnce->bncd", attn, v).reshape(b, c, h, w)
        return x + self.proj_out(out)

class Downsample2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.deconv(x)

# ---------- Encoder and Decoder ----------
class Encoder(nn.Module):
    def __init__(self, in_ch=3, block_out_channels=(128,256,512), z_ch=4, groups=32, dropout=0.0):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, block_out_channels[0], 3, padding=1)
        ch_in = block_out_channels[0]
        downs = []
        for i, ch in enumerate(block_out_channels):
            downs += [ResnetBlock2D(ch_in, ch, groups, dropout),
                      ResnetBlock2D(ch, ch, groups, dropout)]
            ch_in = ch
            if i != len(block_out_channels)-1:
                downs += [Downsample2D(ch_in)]
        self.downs = nn.ModuleList(downs)
        mid_ch = block_out_channels[-1]
        self.mid1 = ResnetBlock2D(mid_ch, mid_ch, groups, dropout)
        self.mid_attn = AttnBlock2D(mid_ch, num_heads=1)
        self.mid2 = ResnetBlock2D(mid_ch, mid_ch, groups, dropout)
        self.norm_out = nn.GroupNorm(groups, mid_ch)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(mid_ch, z_ch*2, 3, padding=1)

    def forward(self, x11):
        h = self.conv_in(x11)
        for m in self.downs: h = m(h)
        h = self.mid2(self.mid_attn(self.mid1(h)))
        h = self.conv_out(self.act(self.norm_out(h)))
        return h

class Decoder(nn.Module):
    def __init__(self, out_ch=3, block_out_channels=(128,256,512), z_ch=4, groups=32, dropout=0.0):
        super().__init__()
        mid_ch = block_out_channels[-1]
        self.conv_in = nn.Conv2d(z_ch, mid_ch, 3, padding=1)
        self.mid1 = ResnetBlock2D(mid_ch, mid_ch, groups, dropout)
        self.mid_attn = AttnBlock2D(mid_ch, num_heads=1)
        self.mid2 = ResnetBlock2D(mid_ch, mid_ch, groups, dropout)
        ups = []
        ch_in = mid_ch
        for i, ch in list(enumerate(block_out_channels))[::-1]:
            ups += [ResnetBlock2D(ch_in, ch, groups, dropout),
                    ResnetBlock2D(ch, ch, groups, dropout)]
            ch_in = ch
            if i != 0: ups += [Upsample2D(ch_in)]
        self.ups = nn.ModuleList(ups)
        self.norm_out = nn.GroupNorm(groups, ch_in)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(ch_in, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid2(self.mid_attn(self.mid1(h)))
        for m in self.ups: h = m(h)
        x11 = self.conv_out(self.act(self.norm_out(h)))
        return x11

# ---------- AutoencoderKL (Diffusers-compatible) ----------
class AutoencoderKLCustom(nn.Module):
    def __init__(self,
                 in_channels=3, out_channels=3, latent_channels=4,
                 block_out_channels=(128,256,512),
                 norm_num_groups=32, scaling_factor=0.18215, dropout=0.0):
        super().__init__()
        self.encoder = Encoder(in_ch=in_channels,
                               block_out_channels=block_out_channels,
                               z_ch=latent_channels,
                               groups=norm_num_groups,
                               dropout=dropout)
        self.decoder = Decoder(out_ch=out_channels,
                               block_out_channels=block_out_channels,
                               z_ch=latent_channels,
                               groups=norm_num_groups,
                               dropout=dropout)
        self.scaling_factor = float(scaling_factor)

    def encode(self, x11: torch.Tensor):
        moments = self.encoder(x11)
        return type("EncodeOutput", (), {"latent_dist": DiagonalGaussianDistribution(moments)})

    def decode(self, z: torch.Tensor):
        x11 = self.decoder(z)
        return type("DecodeOutput", (), {"sample": x11})
