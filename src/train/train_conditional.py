"""
train_conditional.py
--------------------
Addestra il Latent Diffusion Model condizionato anatomicamente.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# IMPORT CORRETTI CON 'network.'
from network.autoencoder import ConvAutoencoder
from data import PerioKPTDataset, collate_fn
from network.diffusion import DiffusionScheduler
from network.model import LatentUNet
from network.anatomy import (
    AnatomyCondition,
    AnatomyEncoder,
    ConditioningStrength,
    collate_to_anatomy,
    IN_CH_COND,
)
from globals import (
    DEVICE, IMAGE_SIZE, LATENT_SIZE, BATCH_SIZE, LATENT_CHANNELS_COND,
    BASE_CHANNELS_COND, TIME_DIM, TIMESTEPS, COND_WEIGHT_DEFAULT, KPT_SIGMA,
    EPOCHS, LR_CONDITIONAL, SAVE_EVERY, AE_CHECKPOINT, CHECKPOINT_DIR,
    DATA_DIR_EXPERIMENT,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train conditional LDM.")

    # Dati
    p.add_argument("--data-dir",    type=str, default=DATA_DIR_EXPERIMENT)
    p.add_argument("--box-type",    type=str, default="standard_box",
                   choices=["standard_box", "rotating_box"])
    p.add_argument("--fold",        type=int, default=0)
    p.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=0)

    # Modello
    p.add_argument("--latent-channels", type=int, default=LATENT_CHANNELS_COND)
    p.add_argument("--base-channels",   type=int, default=BASE_CHANNELS_COND)
    p.add_argument("--time-dim",        type=int, default=TIME_DIM)
    p.add_argument("--timesteps",       type=int, default=TIMESTEPS)

    # Condizionamento
    p.add_argument("--cond-weight", type=float, default=COND_WEIGHT_DEFAULT,
                   help="Forza del condizionamento in [0,1]. 0=baseline non condizionato.")
    p.add_argument("--kpt-sigma",   type=float, default=KPT_SIGMA)

    # Training
    p.add_argument("--epochs",        type=int,   default=EPOCHS)
    p.add_argument("--lr",            type=float, default=LR_CONDITIONAL)
    p.add_argument("--save-every",    type=int,   default=SAVE_EVERY)

    # Checkpoint
    p.add_argument("--ae-checkpoint",   type=str, default=f"../{AE_CHECKPOINT}")
    p.add_argument("--output-dir",      type=str, default=f"../{CHECKPOINT_DIR}")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = DEVICE
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset (CON FILTRO PER CLASSI RARE) ─────────────────────────────
    train_ds = PerioKPTDataset(args.data_dir, split='train',
                               box_type=args.box_type, fold=args.fold, target_classes=[3])
    val_ds   = PerioKPTDataset(args.data_dir, split='test',
                               box_type=args.box_type, fold=args.fold, target_classes=[3])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              collate_fn=collate_fn)

    print(f"Train: {len(train_ds)} campioni | Val: {len(val_ds)} campioni")

    # ── Anatomy ──────────────────────────────────────────────────────────
    anatomy_cond = AnatomyCondition(image_size=(IMAGE_SIZE, IMAGE_SIZE),
                                    latent_size=(LATENT_SIZE, LATENT_SIZE),
                                    kpt_sigma=args.kpt_sigma)
    cond_strength = ConditioningStrength(weight=args.cond_weight)
    anat_enc = AnatomyEncoder(in_channels=2, out_channels=2).to(device)

    # ── Autoencoder (frozen) ─────────────────────────────────────────────
    ae = ConvAutoencoder(in_channels=1, latent_channels=args.latent_channels).to(device)
    ae_path = Path(args.ae_checkpoint)
    if ae_path.exists():
        ae.load_state_dict(torch.load(ae_path, map_location=device))
        print(f"Autoencoder caricato da {ae_path}")
    else:
        print(f"ATTENZIONE: checkpoint AE non trovato in {ae_path} — pesi random!")
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    # ── UNet condizionale ─────────────────────────────────────────────────
    unet = LatentUNet(
        latent_channels=IN_CH_COND,  
        base_channels=args.base_channels,
        time_dim=args.time_dim,
    ).to(device)
    
    unet.output_conv = nn.Conv2d(args.base_channels, args.latent_channels,
                                 kernel_size=3, padding=1).to(device)

    scheduler = DiffusionScheduler(timesteps=args.timesteps, device=device)

    params = list(unet.parameters()) + list(anat_enc.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    # ── Loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        unet.train(); anat_enc.train()
        total_loss = 0.0

        for images, annotations in tqdm(train_loader,
                                        desc=f"Epoch {epoch}/{args.epochs} [train]",
                                        leave=False):
            images = images.to(device)

            anat = collate_to_anatomy(annotations, anatomy_cond, cond_strength, device)
            anat = anat_enc(anat)

            with torch.no_grad():
                z = ae.encode(images)                  

            t = torch.randint(0, scheduler.timesteps, (z.shape[0],), device=device)
            z_noisy, noise = scheduler.q_sample(z, t)  

            z_cond = torch.cat([z_noisy, anat], dim=1) 
            pred_noise = unet(z_cond, t)               

            loss = criterion(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()

        lr_scheduler.step()
        avg_train = total_loss / len(train_loader)

        # ── Validation ───────────────────────────────────────────────────
        unet.eval(); anat_enc.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, annotations in val_loader:
                images = images.to(device)
                anat = collate_to_anatomy(annotations, anatomy_cond, cond_strength, device)
                anat = anat_enc(anat)
                z = ae.encode(images)
                t = torch.randint(0, scheduler.timesteps, (z.shape[0],), device=device)
                z_noisy, noise = scheduler.q_sample(z, t)
                z_cond = torch.cat([z_noisy, anat], dim=1)
                pred_noise = unet(z_cond, t)
                val_loss += criterion(pred_noise, noise).item()
        avg_val = val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"cond_weight={args.cond_weight}")

        # ── Checkpoint ───────────────────────────────────────────────────
        ckpt = {
            'epoch': epoch,
            'unet_state': unet.state_dict(),
            'anat_enc_state': anat_enc.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': avg_val,
            'args': vars(args),
        }
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(ckpt, out_dir / 'ldm_conditional_best.pth')
            print(f"  → Salvato best (val={best_val_loss:.4f})")

        if epoch % args.save_every == 0:
            torch.save(ckpt, out_dir / f'ldm_conditional_ep{epoch:04d}.pth')

    print("Training completato.")


if __name__ == "__main__":
    main()