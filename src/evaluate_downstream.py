"""
evaluate_downstream.py
----------------------
Due task di valutazione:

1. ABLATION STUDY
   Confronta la qualità delle immagini generate al variare del condizionamento:
   - baseline      : cond_weight=0.0  (nessun condizionamento)
   - cond_0.5      : cond_weight=0.5
   - cond_full     : cond_weight=1.0
   - box_only      : solo canale 0 (box mask)
   - kpt_only      : solo canale 1 (keypoint heatmap)

   Metriche:
   - PSNR  (più alto = meglio)
   - SSIM  (più alto = meglio)
   - Anatomy Alignment Score (sovrapposizione generato/anatomy)

2. DOWNSTREAM TASK
   Addestra un classificatore leggero su:
   a) solo dati reali
   b) reali + sintetici baseline
   c) reali + sintetici condizionati
   Confronta accuracy/F1.

Uso
---
    # Ablation
    cd src
    python evaluate_downstream.py ablation \
        --data-dir ../data/perio_KPT/1_Experiment \
        --ae-checkpoint ../checkpoints/autoencoder.pth \
        --cond-checkpoint ../checkpoints/ldm_conditional_best.pth \
        --output-dir ../results/ablation

    # Downstream
    python evaluate_downstream.py downstream \
        --data-dir ../data/perio_KPT/1_Experiment \
        --ae-checkpoint ../checkpoints/autoencoder.pth \
        --baseline-checkpoint ../checkpoints/ldm_unet_debug.pth \
        --cond-checkpoint ../checkpoints/ldm_conditional_best.pth \
        --output-dir ../results/downstream
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from autoencoder import ConvAutoencoder
from data import PerioKPTDataset, collate_fn
from diffusion import DiffusionScheduler
from model import LatentUNet
from anatomy import (
    AnatomyCondition,
    AnatomyEncoder,
    ConditioningStrength,
    collate_to_anatomy,
    IN_CH_COND,
    LATENT_CH,
    COND_CHANNELS,
)
from utils import calculate_psnr, get_ssim_metric


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------

def anatomy_alignment_score(generated: torch.Tensor,
                             anatomy_maps: torch.Tensor,
                             threshold: float = 0.3) -> float:
    """
    Fraction di pixel anatomy-attivi che risultano anche attivi nel generato.
    generated    : (B, 1, H, W)   ∈ [-1, 1]
    anatomy_maps : (B, 2, lH, lW) ∈ [0, 1]
    """
    B, _, H, W = generated.shape
    anat = anatomy_maps.max(dim=1, keepdim=True).values   # (B,1,lH,lW)
    anat_up = F.interpolate(anat, size=(H, W), mode='nearest')
    anat_bin = (anat_up > 0.1).float()
    gen_bin  = (((generated + 1) / 2) > threshold).float()
    inter = (anat_bin * gen_bin).sum()
    union = anat_bin.sum().clamp(min=1.0)
    return (inter / union).item()


# ---------------------------------------------------------------------------
# Sampling condizionale
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_conditional(
    unet: LatentUNet,
    scheduler: DiffusionScheduler,
    ae: ConvAutoencoder,
    anat_enc: AnatomyEncoder,
    anatomy_maps: torch.Tensor,   # (B, 2, 16, 16)
    device: torch.device,
    latent_channels: int = LATENT_CH,
    latent_size: int = 16,
) -> torch.Tensor:
    """Reverse diffusion condizionato → immagini decodificate (B,1,128,128)."""
    B = anatomy_maps.shape[0]
    z = torch.randn(B, latent_channels, latent_size, latent_size, device=device)
    anat_proj = anat_enc(anatomy_maps.to(device))

    for t_idx in tqdm(reversed(range(scheduler.timesteps)),
                      total=scheduler.timesteps, leave=False):
        t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)
        z_in = torch.cat([z, anat_proj], dim=1)   # (B, 6, 16, 16)
        pred_noise = unet(z_in, t_batch)
        z = scheduler.p_sample(z, t_batch, pred_noise)

    return ae.decode(z)


@torch.no_grad()
def sample_baseline(
    unet: LatentUNet,
    scheduler: DiffusionScheduler,
    ae: ConvAutoencoder,
    B: int,
    device: torch.device,
    latent_channels: int = LATENT_CH,
    latent_size: int = 16,
) -> torch.Tensor:
    """Reverse diffusion non condizionato → (B,1,128,128)."""
    z = torch.randn(B, latent_channels, latent_size, latent_size, device=device)
    for t_idx in tqdm(reversed(range(scheduler.timesteps)),
                      total=scheduler.timesteps, leave=False):
        t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)
        pred_noise = unet(z, t_batch)
        z = scheduler.p_sample(z, t_batch, pred_noise)
    return ae.decode(z)


# ---------------------------------------------------------------------------
# Caricamento modelli
# ---------------------------------------------------------------------------

def load_ae(path: str, device: torch.device) -> ConvAutoencoder:
    ae = ConvAutoencoder(in_channels=1, latent_channels=LATENT_CH).to(device)
    if Path(path).exists():
        ae.load_state_dict(torch.load(path, map_location=device))
    else:
        print(f"ATTENZIONE: AE checkpoint non trovato in {path}")
    ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)
    return ae


def load_conditional_ldm(ckpt_path: str, device: torch.device
                         ) -> Tuple[LatentUNet, AnatomyEncoder]:
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get('args', {})
    unet = LatentUNet(latent_channels=IN_CH_COND,
                      base_channels=saved_args.get('base_channels', 64),
                      time_dim=saved_args.get('time_dim', 128)).to(device)
    unet.output_conv = nn.Conv2d(
        saved_args.get('base_channels', 64), LATENT_CH, 3, padding=1).to(device)
    unet.load_state_dict(ckpt['unet_state'])
    unet.eval()

    anat_enc = AnatomyEncoder(COND_CHANNELS, COND_CHANNELS).to(device)
    anat_enc.load_state_dict(ckpt['anat_enc_state'])
    anat_enc.eval()
    return unet, anat_enc


def load_baseline_ldm(ckpt_path: str, device: torch.device) -> LatentUNet:
    unet = LatentUNet(latent_channels=LATENT_CH, base_channels=64, time_dim=128).to(device)
    unet.load_state_dict(torch.load(ckpt_path, map_location=device))
    unet.eval()
    return unet


# ---------------------------------------------------------------------------
# 1. ABLATION STUDY
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    {'name': 'baseline',   'weight': 0.0, 'channels': None},
    {'name': 'cond_0.5',   'weight': 0.5, 'channels': None},
    {'name': 'cond_full',  'weight': 1.0, 'channels': None},
    {'name': 'box_only',   'weight': 1.0, 'channels': [0]},
    {'name': 'kpt_only',   'weight': 1.0, 'channels': [1]},
]


def _apply_channel_mask(anat: torch.Tensor, active: List[int] | None) -> torch.Tensor:
    if active is None:
        return anat
    mask = torch.zeros_like(anat)
    for c in active:
        mask[:, c] = anat[:, c]
    return mask


def run_ablation(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ae = load_ae(args.ae_checkpoint, device)
    unet, anat_enc = load_conditional_ldm(args.cond_checkpoint, device)
    scheduler = DiffusionScheduler(timesteps=1000, device=device)

    ds = PerioKPTDataset(args.data_dir, split='test',
                         box_type=args.box_type, fold=args.fold)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    anatomy_cond  = AnatomyCondition()
    ssim_fn = get_ssim_metric(device)
    results = {}

    for cfg in ABLATION_CONFIGS:
        print(f"\nAblation: {cfg['name']}")
        strength = ConditioningStrength(weight=cfg['weight'])
        psnr_vals, ssim_vals, align_vals = [], [], []

        for real_imgs, annotations in tqdm(loader, desc=cfg['name']):
            real_imgs = real_imgs.to(device)
            anat = collate_to_anatomy(annotations, anatomy_cond, strength, device)
            anat = _apply_channel_mask(anat, cfg['channels'])

            gen = sample_conditional(unet, scheduler, ae, anat_enc, anat, device)

            psnr_vals.append(calculate_psnr(real_imgs, gen))
            ssim_vals.append(ssim_fn(gen, real_imgs).item())
            align_vals.append(anatomy_alignment_score(gen, anat.cpu()))

        results[cfg['name']] = {
            'psnr':  float(np.mean(psnr_vals)),
            'ssim':  float(np.mean(ssim_vals)),
            'anatomy_align': float(np.mean(align_vals)),
        }

    with open(out_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    _print_table(results, title="ABLATION STUDY",
                 cols=['psnr', 'ssim', 'anatomy_align'])
    print(f"\nRisultati salvati in {out_dir / 'ablation_results.json'}")


# ---------------------------------------------------------------------------
# 2. DOWNSTREAM TASK
# ---------------------------------------------------------------------------

class _SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x): return self.head(self.net(x).flatten(1))


def _train_eval_classifier(train_imgs, train_labels, val_imgs, val_labels,
                            device, epochs=20, batch_size=32) -> Dict:
    model = _SimpleClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ld = DataLoader(TensorDataset(train_imgs, train_labels),
                    batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for imgs, lbls in ld:
            loss = F.cross_entropy(model(imgs.to(device)), lbls.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(val_imgs.to(device)).argmax(1).cpu().numpy()
    lbls_np = val_labels.numpy()
    acc = (preds == lbls_np).mean()
    tp = ((preds==1)&(lbls_np==1)).sum()
    fp = ((preds==1)&(lbls_np==0)).sum()
    fn = ((preds==0)&(lbls_np==1)).sum()
    f1 = 2*tp / (2*tp + fp + fn + 1e-8)
    return {'accuracy': float(acc), 'f1': float(f1)}


def run_downstream(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ae = load_ae(args.ae_checkpoint, device)
    cond_unet, anat_enc = load_conditional_ldm(args.cond_checkpoint, device)
    base_unet = load_baseline_ldm(args.baseline_checkpoint, device)
    scheduler = DiffusionScheduler(timesteps=1000, device=device)

    ds = PerioKPTDataset(args.data_dir, split='train',
                         box_type=args.box_type, fold=args.fold)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    anatomy_cond = AnatomyCondition()
    strength_full = ConditioningStrength(1.0)

    all_imgs, all_labels, all_anat = [], [], []
    for imgs, annotations in tqdm(loader, desc="Raccolta dati reali"):
        anat = collate_to_anatomy(annotations, anatomy_cond, strength_full, device)
        # Pseudo-label: 1 se almeno un keypoint visibile, 0 altrimenti
        has_kpt = (anat[:, 1].max(dim=-1).values.max(dim=-1).values > 0.05).long()
        all_imgs.append(imgs)
        all_labels.append(has_kpt.cpu())
        all_anat.append(anat.cpu())

    all_imgs   = torch.cat(all_imgs)
    all_labels = torch.cat(all_labels)
    all_anat   = torch.cat(all_anat)

    N = len(all_imgs)
    n_val = max(1, int(0.2 * N))
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    real_tr_imgs, real_tr_lbl = all_imgs[train_idx], all_labels[train_idx]
    val_imgs, val_lbl         = all_imgs[val_idx],   all_labels[val_idx]
    syn_anat                  = all_anat[train_idx].to(device)

    print("Generazione immagini sintetiche baseline...")
    base_syn = sample_baseline(base_unet, scheduler, ae, len(real_tr_imgs), device).cpu()

    print("Generazione immagini sintetiche condizionate...")
    cond_syn = sample_conditional(cond_unet, scheduler, ae, anat_enc,
                                  syn_anat, device).cpu()

    syn_lbl = (syn_anat.cpu()[:, 1].max(-1).values.max(-1).values > 0.05).long()

    configs = {
        'real_only':             (real_tr_imgs, real_tr_lbl),
        'real_+_baseline_syn':   (torch.cat([real_tr_imgs, base_syn]),
                                  torch.cat([real_tr_lbl, syn_lbl])),
        'real_+_conditional_syn':(torch.cat([real_tr_imgs, cond_syn]),
                                  torch.cat([real_tr_lbl, syn_lbl])),
    }

    results = {}
    for name, (tr_imgs, tr_lbl) in configs.items():
        print(f"\nClassificatore: {name}  ({len(tr_imgs)} campioni)")
        results[name] = _train_eval_classifier(
            tr_imgs, tr_lbl, val_imgs, val_lbl, device)
        print(f"  Accuracy={results[name]['accuracy']:.4f}  F1={results[name]['f1']:.4f}")

    with open(out_dir / 'downstream_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    _print_table(results, title="DOWNSTREAM TASK", cols=['accuracy', 'f1'])
    print(f"\nRisultati salvati in {out_dir / 'downstream_results.json'}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _print_table(results: Dict, title: str, cols: List[str]) -> None:
    header = f"{'Config':<30} " + " ".join(f"{c:>12}" for c in cols)
    sep = "=" * len(header)
    print(f"\n{sep}\n{title}\n{sep}\n{header}\n{'-'*len(header)}")
    for name, m in results.items():
        row = f"{name:<30} " + " ".join(f"{m.get(c,0):>12.4f}" for c in cols)
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--data-dir',      type=str, required=True)
    shared.add_argument('--box-type',      type=str, default='standard_box')
    shared.add_argument('--fold',          type=int, default=0)
    shared.add_argument('--ae-checkpoint', type=str, required=True)
    shared.add_argument('--output-dir',    type=str, default='../results')
    shared.add_argument('--batch-size',    type=int, default=8)

    abl = sub.add_parser('ablation',   parents=[shared])
    abl.add_argument('--cond-checkpoint', type=str, required=True)

    ds = sub.add_parser('downstream', parents=[shared])
    ds.add_argument('--cond-checkpoint',     type=str, required=True)
    ds.add_argument('--baseline-checkpoint', type=str, required=True)
    ds.add_argument('--clf-epochs',          type=int, default=20)

    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    if args.command == 'ablation':
        run_ablation(args)
    elif args.command == 'downstream':
        run_downstream(args)