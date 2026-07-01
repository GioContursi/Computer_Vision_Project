import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from autoencoder import ConvAutoencoder
from data import DentalImageDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train convolutional autoencoder.")

    parser.add_argument("--image-dir", type=str, nargs="+", default=["data/perio_KPT/0_Baseline/images"])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--log-path", type=str, default="outputs/logs/autoencoder_loss.csv")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/autoencoder_debug.pth")
    parser.add_argument("--output-dir", type=str, default="outputs/reconstructions")

    return parser.parse_args()


def denormalize(x):
    return (x + 1) / 2


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_path, "w") as f:
        f.write("epoch,loss\n")

    dataset = DentalImageDataset(
        image_dir=args.image_dir,
        image_size=args.image_size,
        max_images=args.max_images,
        augment=args.augment
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = ConvAutoencoder().to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Device:", device)
    print("Dataset size:", len(dataset))
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for images in loader:
            images = images.to(device)

            reconstructions, _ = model(images)
            loss = criterion(reconstructions, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch [{epoch + 1}/{args.epochs}] - Loss: {avg_loss:.4f}")
        with open(args.log_path, "a") as f:
            f.write(f"{epoch + 1},{avg_loss}\n")

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
                f"{args.output_dir}/epoch_{epoch + 1}.png",
                nrow=args.batch_size
            )
        if (epoch + 1) % args.save_every == 0:
            intermediate_path = Path(args.checkpoint_path).with_name(
                f"autoencoder_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), intermediate_path)

    torch.save(
        model.state_dict(),
        args.checkpoint_path
    )

    print("Training completed.")
    print(f"Checkpoint saved in {args.checkpoint_path}")


if __name__ == "__main__":
    main()