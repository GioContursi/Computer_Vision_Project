import torch

from model import LatentUNet


device = "cuda" if torch.cuda.is_available() else "cpu"

unet = LatentUNet(
    latent_channels=4,
    base_channels=64,
    time_dim=128
).to(device)

z_noisy = torch.randn(8, 4, 16, 16).to(device)
t = torch.randint(0, 1000, (8,), device=device)

pred_noise = unet(z_noisy, t)

print("Device:", device)
print("Input noisy latent shape:", z_noisy.shape)
print("Timesteps shape:", t.shape)
print("Predicted noise shape:", pred_noise.shape)