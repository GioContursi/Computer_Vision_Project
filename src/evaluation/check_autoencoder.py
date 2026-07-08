import torch
from autoencoder import ConvAutoencoder


device = "cuda" if torch.cuda.is_available() else "cpu"

model = ConvAutoencoder().to(device)

x = torch.randn(8, 1, 128, 128).to(device)

x_rec, z = model(x)

print("Device:", device)
print("Input shape:", x.shape)
print("Latent shape:", z.shape)
print("Output shape:", x_rec.shape)