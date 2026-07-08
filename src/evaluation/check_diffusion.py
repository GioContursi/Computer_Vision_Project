import torch

from autoencoder import ConvAutoencoder
from diffusion import DiffusionScheduler


device = "cuda" if torch.cuda.is_available() else "cpu"

model = ConvAutoencoder().to(device)
scheduler = DiffusionScheduler(timesteps=1000, device=device)

x = torch.randn(8, 1, 128, 128).to(device)

with torch.no_grad():
    _, z = model(x)

t = torch.randint(0, scheduler.timesteps, (z.shape[0],), device=device)

z_noisy, noise = scheduler.q_sample(z, t)

print("Device:", device)
print("Input image shape:", x.shape)
print("Latent shape:", z.shape)
print("Timesteps shape:", t.shape)
print("Noisy latent shape:", z_noisy.shape)
print("Noise shape:", noise.shape)
print("z min/max:", z.min().item(), z.max().item())
print("z_noisy min/max:", z_noisy.min().item(), z_noisy.max().item())