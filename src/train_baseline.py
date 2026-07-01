import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autoencoder import ConvAutoencoder
from data import DentalImageDataset
from diffusion import DiffusionScheduler
from model import LatentUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline latent diffusion model.")

    parser.add_argument("--image-dir", type=str, nargs="+", default=["data/perio_KPT/0_Baseline/images"])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--autoencoder-checkpoint", type=str, default="checkpoints/autoencoder_debug.pth")
    parser.add_argument("--unet-checkpoint", type=str, default="checkpoints/ldm_unet_debug.pth")

    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--latent-scale", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-path", type=str, default="outputs/logs/ldm_loss.csv")

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(args.unet_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.log_path, "w") as f:
        f.write("epoch,loss\n")

    dataset = DentalImageDataset(
        image_dir=args.image_dir,
        image_size=args.image_size,
        max_images=args.max_images,
        augment=args.augment,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    autoencoder = ConvAutoencoder().to(device)
    autoencoder.load_state_dict(
        torch.load(args.autoencoder_checkpoint, map_location=device)
    )
    autoencoder.eval()

    for param in autoencoder.parameters():
        param.requires_grad = False

    unet = LatentUNet(
        latent_channels=8,
        base_channels=96,
        time_dim=128,
    ).to(device)

    scheduler = DiffusionScheduler(
        timesteps=args.timesteps,
        device=device,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    print("Device:", device)
    print("Dataset size:", len(dataset))
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Timesteps:", args.timesteps)
    print("Latent scale:", args.latent_scale)

    for epoch in range(args.epochs):
        unet.train()
        total_loss = 0

        for images in loader:
            images = images.to(device)

            with torch.no_grad():
                _, z = autoencoder(images)
                z = z * args.latent_scale

            t = torch.randint(
                0,
                scheduler.timesteps,
                (z.shape[0],),
                device=device,
            )

            z_noisy, noise = scheduler.q_sample(z, t)

            predicted_noise = unet(z_noisy, t)

            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch [{epoch + 1}/{args.epochs}] - Loss: {avg_loss:.4f}")

        with open(args.log_path, "a") as f:
            f.write(f"{epoch + 1},{avg_loss}\n")

        if (epoch + 1) % args.save_every == 0:
            intermediate_path = Path(args.unet_checkpoint).with_name(
                f"ldm_unet_epoch_{epoch + 1}.pth"
            )
            torch.save(unet.state_dict(), intermediate_path)

    torch.save(
        unet.state_dict(),
        args.unet_checkpoint,
    )

    print("Baseline LDM training completed.")
    print(f"Checkpoint saved in {args.unet_checkpoint}")


if __name__ == "__main__":
    main()