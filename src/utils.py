import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def denormalize(tensor):
    """
    Riporta un tensore normalizzato [-1, 1] nel range originale [0, 1]
    per poterlo visualizzare o salvare come immagine standard.
    """
    return (tensor * 0.5) + 0.5


def plot_batch_images(batch_tensor, num_images=16, title="Batch di Esempio (128x128)"):
    """
    Visualizza un grid di immagini (batch). Utile per fare sanity check
    prima di passare i dati alla rete neurale.
    """
    batch_tensor = denormalize(batch_tensor)

    imgs_to_plot = min(len(batch_tensor), num_images)
    grid_size = int(np.ceil(np.sqrt(imgs_to_plot)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < imgs_to_plot:
            img = batch_tensor[i].squeeze(0).cpu().numpy()
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_sample_with_annotations(image_tensor, annotations):
    """
    Visualizza una patch 128x128 con i box e i keypoint YOLO.
    """
    img = denormalize(image_tensor).squeeze(0).cpu().numpy()
    img_h, img_w = img.shape

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img, cmap='gray')

    # 1. Disegna i Bounding Box
    if len(annotations.get('boxes', [])) > 0:
        for i, box in enumerate(annotations['boxes']):
            cx, cy, w, h = box

            pix_w = w * img_w
            pix_h = h * img_h
            pix_x_min = (cx - w / 2) * img_w
            pix_y_min = (cy - h / 2) * img_h

            edge_color = 'magenta' if len(annotations.get('angles', [])) > 0 else 'cyan'

            rect = patches.Rectangle(
                (pix_x_min, pix_y_min), pix_w, pix_h,
                linewidth=1.5, edgecolor=edge_color, facecolor='none'
            )
            ax.add_patch(rect)

            if len(annotations.get('angles', [])) > i:
                angle = annotations['angles'][i]
                ax.text(pix_x_min, pix_y_min - 2, f"{angle:.1f}°", color='magenta', fontsize=8)

    # 2. Disegna i Keypoint
    if len(annotations.get('keypoints', [])) > 0:
        for tooth_kpts in annotations['keypoints']:
            for kp in tooth_kpts:
                x_norm, y_norm, visibility = kp
                if visibility > 0.0:
                    ax.plot(x_norm * img_w, y_norm * img_h, 'go', markersize=4)

    ax.axis('off')
    plt.title(f"Patch Annotata (Denti: {len(annotations.get('classes', []))})")
    plt.show()


def generate_anatomy_mask(annotations, img_size=128):
    """
    Genera una maschera anatomica a 2 canali:
      - Canale 0: maschera binaria dei bounding box (dove sono i denti)
      - Canale 1: gaussian heatmap dei keypoint (dove sono i punti CEJ)

    Output: tensore [2, img_size, img_size] float32 in [0, 1]
    Pronto per essere usato dalla Persona 3 in anatomy.py come condizionamento.
    """
    box_mask = np.zeros((img_size, img_size), dtype=np.float32)
    kpt_mask = np.zeros((img_size, img_size), dtype=np.float32)

    # --- Canale 0: Box mask ---
    for box in annotations.get('boxes', []):
        if isinstance(box, torch.Tensor):
            cx, cy, w, h = box.tolist()
        else:
            cx, cy, w, h = box

        x_min = max(0, int((cx - w / 2) * img_size))
        y_min = max(0, int((cy - h / 2) * img_size))
        x_max = min(img_size, int((cx + w / 2) * img_size))
        y_max = min(img_size, int((cy + h / 2) * img_size))

        box_mask[y_min:y_max, x_min:x_max] = 1.0

    # --- Canale 1: Gaussian heatmap keypoint ---
    sigma = 3.0
    # Griglia precalcolata per efficienza
    gy_grid, gx_grid = np.mgrid[0:img_size, 0:img_size]

    for tooth_kpts in annotations.get('keypoints', []):
        for kp in tooth_kpts:
            if isinstance(kp, torch.Tensor):
                x_norm, y_norm, vis = kp.tolist()
            else:
                x_norm, y_norm, vis = kp

            if vis > 0.0:
                px = int(x_norm * img_size)
                py = int(y_norm * img_size)
                blob = np.exp(-((gx_grid - px)**2 + (gy_grid - py)**2) / (2 * sigma**2))
                kpt_mask = np.maximum(kpt_mask, blob)

    mask = np.stack([box_mask, kpt_mask], axis=0)  # [2, H, W]
    return torch.tensor(mask, dtype=torch.float32)


def plot_sample_with_mask(image_tensor, annotations, img_size=128):
    """
    Visualizza 4 pannelli affiancati:
      1. Patch originale
      2. Patch + bounding box + keypoint
      3. Canale 0: box mask
      4. Canale 1: keypoint heatmap

    Restituisce il tensore maschera [2, img_size, img_size] per uso futuro.
    """
    mask = generate_anatomy_mask(annotations, img_size=img_size)
    img  = denormalize(image_tensor).squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Pannello 1: immagine grezza
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Patch originale')
    axes[0].axis('off')

    # Pannello 2: immagine + box + keypoint
    axes[1].imshow(img, cmap='gray')
    for box in annotations.get('boxes', []):
        if isinstance(box, torch.Tensor):
            cx, cy, w, h = box.tolist()
        else:
            cx, cy, w, h = box
        rect = patches.Rectangle(
            ((cx - w / 2) * img_size, (cy - h / 2) * img_size),
            w * img_size, h * img_size,
            linewidth=1.5, edgecolor='cyan', facecolor='none'
        )
        axes[1].add_patch(rect)
    for tooth_kpts in annotations.get('keypoints', []):
        for kp in tooth_kpts:
            if isinstance(kp, torch.Tensor):
                x_norm, y_norm, vis = kp.tolist()
            else:
                x_norm, y_norm, vis = kp
            if vis > 0.0:
                axes[1].plot(x_norm * img_size, y_norm * img_size, 'go', markersize=4)
    axes[1].set_title('Box + Keypoint')
    axes[1].axis('off')

    # Pannello 3: box mask
    axes[2].imshow(mask[0].numpy(), cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Maschera Box (ch. 0)')
    axes[2].axis('off')

    # Pannello 4: keypoint heatmap
    axes[3].imshow(mask[1].numpy(), cmap='hot', vmin=0, vmax=1)
    axes[3].set_title('Heatmap Keypoint (ch. 1)')
    axes[3].axis('off')

    plt.suptitle(f"Anatomical Mask — Denti: {len(annotations.get('classes', []))}", fontsize=14)
    plt.tight_layout()
    plt.show()

    return mask


def calculate_psnr(img_real, img_recon, data_range=2.0):
    """
    Calcola il PSNR tra l'immagine originale e quella ricostruita dall'Autoencoder.
    data_range è 2.0 perché i tensori sono normalizzati in [-1, 1].
    """
    mse = F.mse_loss(img_real, img_recon)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()


def get_ssim_metric(device='cpu'):
    """
    Inizializza l'oggetto SSIM di TorchMetrics.
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    return ssim