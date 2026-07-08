"""
globals.py
----------
Variabili globali condivise da tutto il progetto: device, dimensioni
immagine/latente, canali, iperparametri di default e path standard.
"""

import torch

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Dimensioni immagine / latente
# ---------------------------------------------------------------------------
IMAGE_SIZE       = 128            
LATENT_SIZE      = 16             

# ATTENZIONE: Uniformato a 8 canali per usare l'Autoencoder pre-addestrato
LATENT_CHANNELS_BASELINE = 8
LATENT_CHANNELS_COND     = 8       # AGGIORNATO DA 4 A 8
COND_CHANNELS            = 2       
IN_CH_COND = LATENT_CHANNELS_COND + COND_CHANNELS   # 10

# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------
TIMESTEPS   = 1000
BETA_START  = 1e-4
BETA_END    = 0.02
LATENT_SCALE = 5.0   

# ---------------------------------------------------------------------------
# Modello (UNet)
# ---------------------------------------------------------------------------
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

COND_WEIGHT_DEFAULT = 1.0   
KPT_SIGMA           = 3.0   

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