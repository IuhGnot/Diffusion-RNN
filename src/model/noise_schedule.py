# src/models/noise_schedule.py
import torch, math

class CosineSchedule:
    """
    经典 cosine ᾱ(t) 调度，离散 N 步
    """
    def __init__(self, num_steps=100, s=0.008):
        self.N = num_steps
        self.s = s

    def ab(self, t, device=None):
        """
        ᾱ_t (cumulative) for integer t in [0, N-1]
        shape: (B,)
        """
        if device is None and hasattr(t, "device"): device = t.device
        t = t.float()
        f = lambda x: torch.cos((x + self.s) / (1 + self.s) * math.pi / 2) ** 2
        ab_t = f(t / (self.N - 1))
        ab_t = torch.clamp(ab_t, 1e-5, 1.0)
        return ab_t

    def t_embed(self, t, dim, device=None):
        from .ldm_unet import timestep_embedding
        return timestep_embedding(t, dim).to(device if device is not None else t.device)

    # ---- helpers for v/eps/x0 convert ----
    @staticmethod
    def eps_from_v(x_t, v, alpha_bar):
        a = alpha_bar.sqrt().view(-1,1,1,1)
        s = (1 - alpha_bar).sqrt().view(-1,1,1,1)
        return a * v + s * x_t

    @staticmethod
    def x0_from_eps(x_t, eps, alpha_bar):
        a = alpha_bar.sqrt().view(-1,1,1,1)
        s = (1 - alpha_bar).sqrt().view(-1,1,1,1)
        return (x_t - s * eps) / (a + 1e-8)

    @staticmethod
    def v_from_eps_x0(eps, x0, alpha_bar):
        a = alpha_bar.sqrt().view(-1,1,1,1)
        s = (1 - alpha_bar).sqrt().view(-1,1,1,1)
        return a * eps - s * x0
