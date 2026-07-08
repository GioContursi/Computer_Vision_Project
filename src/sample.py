import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from autoencoder import ConvAutoencoder
from diffusion import DiffusionScheduler
from model import LatentUNet
from globals import (
    DEVICE, LATENT_CHANNELS_BASELINE, LATENT_SIZE, LATENT_SCALE, TIMESTEPS,
    BASE_CHANNELS_BASELINE, TIME_DIM, AE_CHECKPOINT, BASELINE_UNET_CHECKPOINT,
    OUTPUT_DIR,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images from baseline LDM.")

    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=LATENT_CHANNELS_BASELINE)
    parser.add_argument("--latent-size", type=int, default=LATENT_SIZE)
    parser.add_argument("--latent-scale", type=float, default=LATENT_SCALE)
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)

    parser.add_argument("--autoencoder-checkpoint", type=str, default=AE_CHECKPOINT)
    parser.add_argument("--unet-checkpoint", type=str, default=BASELINE_UNET_CHECKPOINT)

    parser.add_argument("--output-path", type=str, default=f"{OUTPUT_DIR}/generated_debug.png")
    parser.add_argument("--nrow", type=int, default=4)

    return parser.parse_args()


def denormalize(x):
    return (x + 1) / 2


def main():
    args = parse_args()

    device = DEVICE

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    latent_shape = (
        args.num_samples,
        args.latent_channels,
        args.latent_size,
        args.latent_size
    )

    autoencoder = ConvAutoencoder().to(device)
    autoencoder.load_state_dict(
        torch.load(args.autoencoder_checkpoint, map_location=device)
    )
    autoencoder.eval()

    unet = LatentUNet(
        latent_channels=LATENT_CHANNELS_BASELINE,
        base_channels=BASE_CHANNELS_BASELINE,
        time_dim=TIME_DIM
    ).to(device)

    unet.load_state_dict(
        torch.load(args.unet_checkpoint, map_location=device)
    )
    unet.eval()

    scheduler = DiffusionScheduler(
        timesteps=args.timesteps,
        device=device
    )

    z = torch.randn(latent_shape).to(device)

    print("Device:", device)
    print("Num samples:", args.num_samples)
    print("Timesteps:", args.timesteps)
    print("Latent shape:", latent_shape)

    with torch.no_grad():
        for timestep in tqdm(reversed(range(scheduler.timesteps)), total=scheduler.timesteps):
            t = torch.full(
                (args.num_samples,),
                timestep,
                device=device,
                dtype=torch.long
            )

            predicted_noise = unet(z, t)
            z = scheduler.p_sample(z, t, predicted_noise)

        generated_images = autoencoder.decode(z / args.latent_scale)

    save_image(
        denormalize(generated_images.cpu()),
        args.output_path,
        nrow=args.nrow
    )

    print("Sampling completed.")
    print(f"Generated images saved in {args.output_path}")


if __name__ == "__main__":
    main()