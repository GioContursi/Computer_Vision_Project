import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from autoencoder import ConvAutoencoder
from data import DentalImageDataset


def denormalize(x):
    return (x + 1) / 2


device = "cuda" if torch.cuda.is_available() else "cpu"

image_dir = "data/perio_KPT/0_Baseline/images"

dataset = DentalImageDataset(
    image_dir=image_dir,
    image_size=128,
    max_images=193
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

model = ConvAutoencoder().to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images in loader:
        images = images.to(device)

        reconstructions, latents = model(images)
        loss = criterion(reconstructions, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        images = next(iter(loader)).to(device)
        reconstructions, _ = model(images)

        comparison = torch.cat([
            denormalize(images.cpu()),
            denormalize(reconstructions.cpu())
        ], dim=0)

        save_image(
            comparison,
            f"outputs/reconstructions/epoch_{epoch + 1}.png",
            nrow=8
        )

torch.save(
    model.state_dict(),
    "checkpoints/autoencoder_debug.pth"
)

print("Training completed.")
print("Checkpoint saved in checkpoints/autoencoder_debug.pth")