# src/models/latent_diffusion.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Sinusoidal embedding ----------------
def timestep_embedding(timesteps, dim):
    """
    timesteps: (B,) int/float in [0, N-1] or [0,1]
    returns: (B, dim)
    """
    if not timesteps.dtype.is_floating_point:
        timesteps = timesteps.float()
    device = timesteps.device
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / max(1, half - 1))
    args = timesteps[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

# ---------------- Blocks ----------------
class FiLM(nn.Module):
    def __init__(self, tdim, channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(tdim, channels * 2))

    def forward(self, h, t_emb):
        gb = self.mlp(t_emb)  # (B,2C)
        gamma, beta = gb.chunk(2, dim=1)
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

class ResBlock(nn.Module):
    def __init__(self, ch, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.film = FiLM(tdim, ch)

    def forward(self, x, t_emb):
        h = self.c1(F.silu(self.n1(x)))
        h = self.film(h, t_emb)
        h = self.c2(F.silu(self.n2(h)))
        return x + h

# ---------------- UNet (shallow, single-scale) ----------------
class LatentCondUNet(nn.Module):
    """
    Predict eps for x_t in latent space.
    Inputs:
      x_t:   (B,C,H,W) noisy latent at timestep t
      z_prev:(B,C,H,W) previous-frame latent (scaled latent!)
      t_emb: (B,tdim)  timestep embedding
      dnorm: (B,)      normalized delta
      z_rnn:(B,C,H,W)  optional RNN prior (scaled latent!)
    """
    def __init__(self, z_channels=4, hidden=128, num_res=4, tdim=256, use_rnn_cond=True):
        super().__init__()
        self.use_rnn_cond = use_rnn_cond
        in_ch = z_channels + z_channels + (z_channels if use_rnn_cond else 0)

        self.in_conv = nn.Conv2d(in_ch, hidden, 3, 1, 1)
        self.time_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.delta_mlp = nn.Sequential(nn.Linear(1, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.res = nn.ModuleList([ResBlock(hidden, tdim) for _ in range(num_res)])
        self.out_norm = nn.GroupNorm(8, hidden)
        self.out_conv = nn.Conv2d(hidden, z_channels, 3, 1, 1)

    def forward(self, x_t, z_prev, t_emb, dnorm, z_rnn=None):
        if self.use_rnn_cond and z_rnn is None:
            raise ValueError("use_rnn_cond=True requires z_rnn")
        x = torch.cat([x_t, z_prev] + ([z_rnn] if self.use_rnn_cond else []), dim=1)

        h = self.in_conv(x)
        t_full = self.time_mlp(t_emb) + self.delta_mlp(dnorm.view(-1, 1))
        for blk in self.res:
            h = blk(h, t_full)
        h = F.silu(self.out_norm(h))
        eps_hat = self.out_conv(h)
        return eps_hat

# ---------------- Cosine schedule (DDIM-friendly) ----------------
class SimpleNoiseSchedule:
    """
    Cosine schedule producing cumulative alpha_bar[t] for t=0..N-1.
    Use:
      - TRAIN add-noise: x_t = sqrt(alpha_bar[t]) * z + sqrt(1 - alpha_bar[t]) * eps
      - DDIM sampling (eta=0):
          x0_hat = (x_t - sqrt(1-ab_t)*eps_hat)/sqrt(ab_t)
          x_{t-1} = sqrt(ab_{t-1})*x0_hat + sqrt(1-ab_{t-1})*eps_hat
    """
    def __init__(self, num_steps=50):
        self.N = int(num_steps)
        s = 0.008
        t = torch.linspace(0, 1, self.N + 1)
        alphas_cum = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cum = alphas_cum / alphas_cum[0]
        # use mid-interval value as ab_t
        self.alpha_bar = alphas_cum[1:]  # length N, avoid 1.0 at t=0
        # Guard
        self.alpha_bar = self.alpha_bar.clamp(1e-6, 1.0)

    def ab(self, n, device):
        """Gather alpha_bar[t] by integer timesteps."""
        if isinstance(n, torch.Tensor):
            return self.alpha_bar.to(device)[n.long()]
        return self.alpha_bar[n]

    def t_embed(self, n, dim, device):
        """Embed t in [0,1] with sinusoidal."""
        if isinstance(n, torch.Tensor):
            t01 = n.float() / max(1, self.N - 1)
            return timestep_embedding(t01.to(device), dim)
        else:
            t01 = float(n) / max(1, self.N - 1)
            return timestep_embedding(torch.tensor([t01], device=device), dim)
