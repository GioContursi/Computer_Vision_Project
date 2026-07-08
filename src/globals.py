"""
globals.py
----------
Variabili globali condivise da tutto il progetto: device, dimensioni
immagine/latente, canali, iperparametri di default e path standard.

Vengono raccolte qui le costanti che oggi sono ripetute (o passate come
default negli argparse) in più script (train_autoencoder.py,
train_baseline.py, train_conditional.py, sample.py, evaluate_downstream.py),
in modo da avere un unico punto di verità.

Uso:
    from globals import DEVICE, IMAGE_SIZE, LATENT_SIZE, ...
"""

import torch


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dimensioni immagine / latente
# ---------------------------------------------------------------------------
IMAGE_SIZE       = 128            # H, W della patch in input all'autoencoder
LATENT_SIZE      = 16             # H, W del latente prodotto da ConvAutoencoder

# ATTENZIONE: nel progetto esistono due pipeline con canali latenti diversi:
#   - baseline (train_baseline.py, sample.py): ConvAutoencoder/LatentUNet a 8 canali
#   - conditional (train_conditional.py, anatomy.py, evaluate_downstream.py): 4 canali
LATENT_CHANNELS_BASELINE = 8
LATENT_CHANNELS_COND     = 4
COND_CHANNELS            = 2       # canali anatomy (box mask + keypoint heatmap)
IN_CH_COND = LATENT_CHANNELS_COND + COND_CHANNELS   # input UNet condizionale (6)


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------
TIMESTEPS   = 1000
BETA_START  = 1e-4
BETA_END    = 0.02
LATENT_SCALE = 5.0   # fattore di scala applicato al latente prima della diffusione


# ---------------------------------------------------------------------------
# Modello (UNet)
# ---------------------------------------------------------------------------
# Stessa cosa dei latent_channels: baseline e conditional usano larghezze diverse.
BASE_CHANNELS_BASELINE = 96
BASE_CHANNELS_COND     = 64
TIME_DIM               = 128


# ---------------------------------------------------------------------------
# Training - iperparametri di default
# ---------------------------------------------------------------------------
BATCH_SIZE    = 16
EPOCHS        = 50
LR_AUTOENCODER = 1e-4
LR_BASELINE    = 5e-5
LR_CONDITIONAL = 1e-4
SAVE_EVERY     = 10
SEED           = 42

# Ablation study (anatomy conditioning)
COND_WEIGHT_DEFAULT = 1.0   # 0.0 = baseline non condizionato, 1.0 = condizionamento pieno
KPT_SIGMA           = 3.0   # spread Gaussian keypoint (pixel, image-space)


# ---------------------------------------------------------------------------
# Path di default (relativi a src/)
# ---------------------------------------------------------------------------
DATA_DIR_BASELINE    = "data/perio_KPT/0_Baseline/images"
DATA_DIR_EXPERIMENT  = "../data/perio_KPT/1_Experiment"

CHECKPOINT_DIR       = "checkpoints"
AE_CHECKPOINT        = f"{CHECKPOINT_DIR}/autoencoder.pth"
BASELINE_UNET_CHECKPOINT    = f"{CHECKPOINT_DIR}/ldm_unet_debug.pth"
CONDITIONAL_UNET_CHECKPOINT = f"{CHECKPOINT_DIR}/ldm_conditional_best.pth"

OUTPUT_DIR   = "outputs"
RESULTS_DIR  = "../results"
LOG_DIR      = f"{OUTPUT_DIR}/logs"


if __name__ == "__main__":
    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE, "| Latent size:", LATENT_SIZE)
    print("Latent channels baseline:", LATENT_CHANNELS_BASELINE,
          "| conditional:", LATENT_CHANNELS_COND,
          "| Cond channels:", COND_CHANNELS,
          "| In-ch conditional UNet:", IN_CH_COND)
    print("Timesteps:", TIMESTEPS)