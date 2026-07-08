import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from network.autoencoder import ConvAutoencoder
from data import DentalImageDataset
from network.diffusion import DiffusionScheduler
from network.model import LatentUNet
from globals import (
    DEVICE, IMAGE_SIZE, BATCH_SIZE, LR_BASELINE, TIMESTEPS, LATENT_SCALE,
    SAVE_EVERY, BASE_CHANNELS_BASELINE, TIME_DIM, LATENT_CHANNELS_BASELINE,
    DATA_DIR_BASELINE, AE_CHECKPOINT, BASELINE_UNET_CHECKPOINT, LOG_DIR,
)

import time


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline latent diffusion model.")

    parser.add_argument("--image-dir", type=str, nargs="+", default=[DATA_DIR_BASELINE])
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--max-images", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=LR_BASELINE)

    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--autoencoder-checkpoint", type=str, default=AE_CHECKPOINT)
    parser.add_argument("--unet-checkpoint", type=str, default=BASELINE_UNET_CHECKPOINT)

    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--latent-scale", type=float, default=LATENT_SCALE)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--log-path", type=str, default=f"{LOG_DIR}/ldm_loss.csv")

    parser.add_argument("--resume-unet-checkpoint", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    device = DEVICE

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
        num_workers=8,
        pin_memory=True,
        persistent_workers=True if args.epochs > 1 else False
    )

    autoencoder = ConvAutoencoder(latent_channels=LATENT_CHANNELS_BASELINE).to(device)
    autoencoder.load_state_dict(
        torch.load(args.autoencoder_checkpoint, map_location=device)
    )
    autoencoder.eval()

    for param in autoencoder.parameters():
        param.requires_grad = False

    unet = LatentUNet(
        latent_channels=LATENT_CHANNELS_BASELINE,
        base_channels=BASE_CHANNELS_BASELINE,
        time_dim=TIME_DIM,
    ).to(device)

    if args.resume_unet_checkpoint is not None:
        unet.load_state_dict(torch.load(args.resume_unet_checkpoint, map_location=device))
        print(f"Resumed UNet from {args.resume_unet_checkpoint}")

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
        start_time = time.time()
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

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")        

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