"""
anatomy.py
----------
Costruisce il condizionamento anatomico a partire dalle annotazioni
del dataset perio_KPT e lo converte in mappe compatibili con la
risoluzione latente dell'autoencoder (4×16×16).

Formato annotazioni in ingresso (da PerioKPTDataset / collate_fn):
    annotations = {
        'classes':   list of int
        'boxes':     list of [cx, cy, w, h]        normalizzati in [0,1]
        'keypoints': list of [[x, y, vis], ...]    normalizzati in [0,1]
        'angles':    list of float                 radianti
    }

Output: tensore float32 (C_cond, latent_H, latent_W)
    - C_cond = 2  (box mask + keypoint heatmap)
    - compatibile con generate_anatomy_mask di utils.py

Strategia di integrazione nella UNet:
    z_cond = cat([z_noisy, anatomy_map_downsampled], dim=1)
    → shape (B, 4+2, 16, 16) = (B, 6, 16, 16)
    → LatentUNet(latent_channels=6) predice rumore su 4 canali originali
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
COND_CHANNELS = 2   # canale 0: box mask  |  canale 1: keypoint heatmap
LATENT_CH     = 4   # canali latenti di ConvAutoencoder
IN_CH_COND    = LATENT_CH + COND_CHANNELS   # = 6, input del conditional UNet


# ---------------------------------------------------------------------------
# Costruzione mappe in image-space (numpy)
# ---------------------------------------------------------------------------

def _box_mask(boxes: List, H: int, W: int) -> np.ndarray:
    """Riempie i bounding box normalizzati su una mappa H×W."""
    canvas = np.zeros((H, W), dtype=np.float32)
    for box in boxes:
        if isinstance(box, torch.Tensor):
            cx, cy, w, h = box.tolist()
        else:
            cx, cy, w, h = box
        x1 = max(0, int((cx - w / 2) * W))
        y1 = max(0, int((cy - h / 2) * H))
        x2 = min(W, int((cx + w / 2) * W))
        y2 = min(H, int((cy + h / 2) * H))
        canvas[y1:y2, x1:x2] = 1.0
    return canvas


def _keypoint_heatmap(keypoints: List, H: int, W: int, sigma: float = 3.0) -> np.ndarray:
    """Gaussian heatmap per ogni keypoint visibile."""
    canvas = np.zeros((H, W), dtype=np.float32)
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)
    for tooth_kpts in keypoints:
        for kp in tooth_kpts:
            if isinstance(kp, torch.Tensor):
                x_n, y_n, vis = kp.tolist()
            else:
                x_n, y_n, vis = kp
            if vis > 0.0:
                px, py = x_n * W, y_n * H
                blob = np.exp(-((gx - px)**2 + (gy - py)**2) / (2 * sigma**2))
                canvas = np.maximum(canvas, blob)
    return canvas


# ---------------------------------------------------------------------------
# AnatomyCondition: da dict annotazioni → tensore latente
# ---------------------------------------------------------------------------

class AnatomyCondition:
    """
    Converte le annotazioni di un singolo campione in un tensore
    (COND_CHANNELS, latent_H, latent_W) pronto per la concatenazione
    con il latente rumoroso nella UNet condizionale.

    Parametri
    ----------
    image_size  : (H, W) della patch in input all'autoencoder (default 128×128)
    latent_size : (H, W) del latente prodotto da ConvAutoencoder  (default 16×16)
    kpt_sigma   : spread in pixel image-space per i Gaussian keypoint
    """

    N_CHANNELS = COND_CHANNELS

    def __init__(
        self,
        image_size:  tuple = (128, 128),
        latent_size: tuple = (16, 16),
        kpt_sigma:   float = 3.0,
    ):
        self.image_size  = image_size
        self.latent_size = latent_size
        self.kpt_sigma   = kpt_sigma

    def build(self, annotations: Dict) -> torch.Tensor:
        """
        Parametri
        ----------
        annotations : dict con chiavi boxes, keypoints (dal collate_fn di data.py)

        Ritorna
        -------
        torch.Tensor  shape (2, latent_H, latent_W), dtype float32
        """
        H, W = self.image_size

        ch0 = _box_mask(annotations.get('boxes', []), H, W)
        ch1 = _keypoint_heatmap(annotations.get('keypoints', []), H, W, self.kpt_sigma)

        cond = np.stack([ch0, ch1], axis=0)              # (2, H, W)
        t = torch.from_numpy(cond).unsqueeze(0)          # (1, 2, H, W)
        t = F.adaptive_avg_pool2d(t, self.latent_size)   # (1, 2, lH, lW)
        return t.squeeze(0)                              # (2, lH, lW)

    def build_batch(self, annotations_list: List[Dict]) -> torch.Tensor:
        """
        Versione batch: lista di dict → (B, 2, lH, lW)
        Usata nel training loop per convertire l'output di collate_fn.
        """
        maps = [self.build(ann) for ann in annotations_list]
        return torch.stack(maps, dim=0)


# ---------------------------------------------------------------------------
# ConditioningStrength: scala per ablation study
# ---------------------------------------------------------------------------

class ConditioningStrength:
    """
    Moltiplica il tensore di condizionamento per weight ∈ [0, 1].

    weight = 0.0 → baseline non condizionato
    weight = 1.0 → condizionamento pieno

    Unico parametro da variare nell'ablation study.
    """
    def __init__(self, weight: float = 1.0):
        assert 0.0 <= weight <= 1.0
        self.weight = weight

    def __call__(self, anatomy: torch.Tensor) -> torch.Tensor:
        return anatomy * self.weight


# ---------------------------------------------------------------------------
# AnatomyEncoder: proiezione learnable opzionale (1×1 conv)
# ---------------------------------------------------------------------------

class AnatomyEncoder(nn.Module):
    """
    Proietta i 2 canali anatomy in `out_channels` con una conv 1×1.
    Lasciato a out_channels=2 è un'identità appresa.
    Aumentando out_channels si dà più capacità al condizionamento.
    """
    def __init__(self, in_channels: int = COND_CHANNELS, out_channels: int = COND_CHANNELS):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(2, out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Helpers per il training loop
# ---------------------------------------------------------------------------

def collate_to_anatomy(annotations_batch: Dict, anatomy_cond: AnatomyCondition,
                       strength: ConditioningStrength, device: torch.device) -> torch.Tensor:
    """
    Converte l'output di collate_fn in un batch di mappe anatomy.

    annotations_batch : dict con liste di lunghezza B
                        (output diretto del DataLoader con collate_fn)
    Ritorna: (B, 2, lH, lW) su device
    """
    B = len(annotations_batch['boxes'])
    maps = []
    for i in range(B):
        ann = {
            'boxes':     [b.tolist() if isinstance(b, torch.Tensor) else b
                          for b in annotations_batch['boxes'][i]],
            'keypoints': [[kp.tolist() if isinstance(kp, torch.Tensor) else kp
                           for kp in tooth]
                          for tooth in annotations_batch['keypoints'][i]],
        }
        maps.append(anatomy_cond.build(ann))
    batch = torch.stack(maps, dim=0)      # (B, 2, lH, lW)
    batch = strength(batch)
    return batch.to(device)