import torch
from src.models.vae import AutoencoderKLCustom

x01 = torch.rand(2,3,256,256)  # [0,1]
vae = AutoencoderKLCustom().eval()
with torch.no_grad():
    dist = vae.encode(x01*2-1).latent_dist
    z = dist.sample() * vae.scaling_factor
    x01_rec = (vae.decode(z/vae.scaling_factor).sample + 1)/2
print("x_rec range:", x01_rec.min().item(), x01_rec.max().item())
print("KL mean:", dist.kl().mean().item())
