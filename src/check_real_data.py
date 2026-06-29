import torch
from torch.utils.data import DataLoader

from data import DentalImageDataset
from autoencoder import ConvAutoencoder


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = DentalImageDataset(
    image_dir="data/perio_KPT/0_Baseline/images",
    image_size=128,
    max_images=32
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

images = next(iter(loader))
images = images.to(device)

model = ConvAutoencoder().to(device)

with torch.no_grad():
    reconstructions, latents = model(images)

print("Device:", device)
print("Images shape:", images.shape)
print("Latents shape:", latents.shape)
print("Reconstructions shape:", reconstructions.shape)
print("Image min:", images.min().item())
print("Image max:", images.max().item())