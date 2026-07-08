# Anatomically-Constrained LDMs for Rare Case Augmentation

Computer Vision project (Prof. Amerini, Sapienza Università di Roma) — synthetic dental radiographic patch generation via a Latent Diffusion Model in PyTorch, constrained by anatomical annotations to augment rare periodontal disease cases.

## Overview

This project studies the generation of synthetic dental radiographic patches through a Latent Diffusion Model implemented in PyTorch. The work is based on the **perio-KPT** dataset, which contains radiographic images and anatomical annotations such as bounding boxes, rotating boxes, and keypoints.

The goal is to compare a baseline generative model, which produces images without explicit anatomical constraints, against a conditioned model that receives structural information about the tooth during generation, and to measure whether that conditioning helps when the synthetic images are used to augment a severely underrepresented disease class.

## Dataset

**Periodontal Keypoint and Object Detection Dataset (perio-KPT)**
Guerrero, M. E., Banks, R., García-Madueño, N. M., Thengane, V., Li, Y., Tang, H. L., Chaurasia, A. — Zenodo.

- Latest version (v2.0, 192 images): <https://doi.org/10.5281/zenodo.17272200>
- Original version (v1.0, 193 images): <https://doi.org/10.5281/zenodo.14711842>
- License: Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) — **non-commercial research use only**.
- Access: the record is public, but the files themselves are **restricted** — you must request access on the Zenodo page (approved to university-affiliated researchers).
- Companion paper: R. Banks et al., *"Periodontal bone loss analysis via keypoint detection with heuristic post-processing"*, Computers in Biology and Medicine, 204:111515, 2026. [doi:10.1016/j.compbiomed.2026.111515](https://doi.org/10.1016/j.compbiomed.2026.111515)
- Reference code released with the dataset: <https://github.com/Banksylel/Bone-Loss-Keypoint-Detection-Code>

If you use this dataset, cite both the dataset record and the paper above (see the Zenodo page for the exact BibTeX entry).

**Why this dataset fits the project brief.** Bounding box class `3` in the annotation schema is **ARR — Alveolar Ridge Resorption**, a rare, severe periodontal finding. `train/train_conditional.py` filters training samples to `target_classes=[3]` specifically to focus the conditional LDM on this underrepresented class, which is the "rare case" the project title refers to.

**Folder layout expected by the code** (unpack the dataset so it matches the paths used by `globals.py` / the `--data-dir` and `--image-dir` CLI flags):

```
data/perio_KPT/
├── 0_Baseline/                      # all radiographs + YOLO-pose .txt annotations
│   └── images/
├── 1_Experiment/
│   ├── standard_box/                # 5-fold CV split, axis-aligned boxes + keypoints
│   ├── rotating_box/                # same split, rotated boxes only (no keypoints)
│   └── holdout_test_standard_box/   # held-out test set, axis-aligned boxes + keypoints
```

Annotations are in YOLO-pose format (`class x_center y_center w h [kpt_x kpt_y visibility] * 11`), normalized to `[0, 1]`. See the Zenodo page for the full class list (11 keypoint classes: CEJ, bone level and root level points, mesial/distal, furcation points, ARR) and the visibility convention (`0`=not trained, `1`=partially visible, `2`=visible).

## Repository structure

```
src/
├── globals.py            # Globals: device, dimensions, hyperparameters and shared paths
├── utils.py               # Utils: plotting, anatomical masks, metrics (PSNR/SSIM)
├── data.py                 # Data: Dataset and DataLoader (PerioKPTDataset, DentalImageDataset, collate_fn)
├── network/                # Network: model architectures
│   ├── autoencoder.py      #   ConvAutoencoder
│   ├── model.py            #   LatentUNet
│   ├── diffusion.py        #   DiffusionScheduler
│   └── anatomy.py          #   Anatomical conditioning (box mask + keypoint heatmap)
├── train/                  # Train: training scripts
│   ├── train_autoencoder.py
│   ├── train_baseline.py
│   └── train_conditional.py
└── evaluation/             # Evaluation: sampling, evaluation, sanity checks
    ├── sample.py
    ├── evaluate_downstream.py
    ├── check_dataset.py
    ├── check_setup.py
    ├── check_autoencoder.py
    ├── check_diffusion.py
    ├── check_unet.py
    └── check_real_data.py
```

Each `.py` file imports its own dependencies at the top; since this is not a notebook project, there is no single shared import cell. All scripts under `train/` and `evaluation/` are meant to be run as modules (`python -m train.xxx` / `python -m evaluation.xxx`) from inside `src/`, so that the `network.xxx` imports resolve correctly.

## Project structure

The project is split into three connected parts. Each part produces an output used by the next stage of the pipeline.

| Part | Responsibility | Main files |
|---|---|---|
| Part 1 | Dataset, preprocessing and evaluation | `data.py`, `utils.py`, `evaluation/check_dataset.py` |
| Part 2 | Autoencoder and baseline LDM | `network/autoencoder.py`, `network/model.py`, `network/diffusion.py`, `train/train_autoencoder.py`, `train/train_baseline.py`, `evaluation/sample.py` |
| Part 3 | Anatomical guidance, ablation and downstream task | `network/anatomy.py`, `train/train_conditional.py`, `evaluation/evaluate_downstream.py` |

Constants and hyperparameters shared across all three parts (device, image/latent dimensions, timesteps, default paths) are centralized in `globals.py`.

## General pipeline

```
perio-KPT dataset
  -> reading images and YOLO-pose labels
  -> extracting 128x128 or 512x512 dental patches
  -> PyTorch DataLoader
  -> autoencoder: image -> latent -> reconstruction
  -> baseline LDM in latent space
  -> generation of unconditioned synthetic images

anatomical annotations: box + keypoints
  -> filter to the rare/target class(es)
  -> box mask + keypoint heatmap
  -> maps compatible with the latent
  -> conditional LDM
  -> ablation study + downstream evaluation (PSNR / SSIM / anatomy alignment / accuracy / F1 / mAP)
```

---

## Part 1 — Dataset, preprocessing and evaluation

The first part prepares the data that feeds the rest of the pipeline. The dataset is parsed to separate images, labels, bounding boxes, rotating boxes and keypoints. Bounding boxes are used to localize the tooth and extract a radiographic patch centered on the relevant region. Keypoints instead describe relevant anatomical points.

The code builds PyTorch Datasets and DataLoaders, so training receives already-normalized, consistent batches. Sanity-check visualizations are also implemented to verify that images, boxes, keypoints and masks are correctly aligned.

**Box type.** `--box-type` selects which annotation subfolder is loaded:
- `standard_box` — axis-aligned bounding boxes + keypoints (used for training/evaluating the LDMs).
- `rotating_box` — boxes with an added rotation angle, no keypoints; only used downstream to compute bone-loss percentage relative to the tooth's true orientation, not for LDM training.

**Fold.** `--fold` selects which of the dataset's 5 cross-validation folds is used as train/test split.

*Main files*
- `src/data.py`
- `src/utils.py`

*Dataset check*
```bash
cd src
python -m evaluation.check_dataset --data-dir ../data/perio_KPT/1_Experiment --box-type standard_box --fold 0
```

*Output of part 1*
- Radiographic patches ready for the autoencoder and generative models.
- Annotations organized in a format readable by conditional training.
- Sanity-check visualizations for dataset, boxes, keypoints and masks.
- Support metrics to compare real, reconstructed, and generated images.

---

## Part 2 — Autoencoder and baseline LDM

The second part implements the baseline generative model. First, a convolutional autoencoder is trained on the dental patches. The encoder compresses each image into a smaller latent tensor, while the decoder reconstructs the patch from this latent.

After pretraining, the autoencoder is frozen. The Latent Diffusion Model then operates in latent space: during training, noise is added to the latent and a UNet learns to predict the noise to remove. At sampling time, the model starts from random noise and progressively generates a new latent, which is then decoded into a synthetic image.

*Main files*
- `src/network/autoencoder.py`
- `src/network/model.py`
- `src/network/diffusion.py`
- `src/train/train_autoencoder.py`
- `src/train/train_baseline.py`
- `src/evaluation/sample.py`

*Training the autoencoder*
```bash
cd src
python -m train.train_autoencoder --image-dir ../data/perio_KPT/0_Baseline/images ../data/perio_KPT/1_Experiment/standard_box --image-size 512 --epochs 250 --batch-size 8 --lr 1e-4 --augment --save-every 50 --checkpoint-path ../checkpoints/autoencoder_8ch_512_best.pth --output-dir ../outputs/reconstructions_8ch_512_best --log-path ../outputs/logs/autoencoder_8ch_512_loss.csv
```

*Training the baseline LDM*
```bash
python -m train.train_baseline --image-dir ../data/perio_KPT/0_Baseline/images ../data/perio_KPT/1_Experiment/standard_box --image-size 512 --epochs 500 --batch-size 8 --lr 5e-5 --timesteps 1000 --augment --latent-scale 5.0 --save-every 50 --autoencoder-checkpoint ../checkpoints/autoencoder_8ch_512_best.pth --unet-checkpoint ../checkpoints/ldm_unet_8ch_512_best.pth --log-path ../outputs/logs/ldm_8ch_512_loss.csv
```

*Sampling from the baseline LDM*
```bash
python -m evaluation.sample --num-samples 16 --latent-size 64 --timesteps 1000 --latent-scale 5.0 --autoencoder-checkpoint ../checkpoints/autoencoder_8ch_512_best.pth --unet-checkpoint ../checkpoints/ldm_unet_8ch_512_best.pth --output-path ../outputs/generated_8ch_512_best.png --nrow 4
```

*Output of part 2*
- Checkpoint of the trained autoencoder.
- Checkpoint of the baseline Latent Diffusion Model.
- Synthetic images generated without anatomical conditioning.
- A baseline useful for comparing the limits and improvements of the conditional model.

---

## Part 3 — Anatomical guidance, ablation and downstream task

The third part builds on the work of the first two. From part 1 it takes boxes and keypoints; from part 2 it takes the autoencoder, UNet and diffusion scheduler. The anatomical annotations are turned into maps: a box mask and a keypoint heatmap.

These maps are resized to the latent resolution and concatenated to the noisy latent. This way, the conditional UNet receives both the signal to denoise and guidance on the expected anatomical structure. The expected result is a generation more consistent with the position and shape of the tooth compared to the baseline.

**Filtering (rare-class training).** `train_conditional.py` restricts both the train and the test `PerioKPTDataset` to `target_classes=[3]` (the ARR / Alveolar Ridge Resorption class) — this is what actually implements "rare case augmentation": the conditional model only ever sees, and is only meant to generate, the rare pathology the project targets, rather than the full mixed-class dataset.

**Conditioning strength.** `--cond-weight` (0.0–1.0) scales the anatomical maps before concatenation and is the single parameter varied in the ablation study: `0.0` reproduces the unconditioned baseline, `1.0` is full conditioning.

*Main files*
- `src/network/anatomy.py`
- `src/train/train_conditional.py`
- `src/evaluation/evaluate_downstream.py`

*Training the conditional LDM*
```bash
cd src
python -m train.train_conditional --data-dir ../data/perio_KPT/1_Experiment --box-type standard_box --fold 0 --ae-checkpoint ../checkpoints/autoencoder.pth --output-dir ../checkpoints --cond-weight 1.0 --epochs 50
```

*Ablation study (baseline vs. cond_0.5 vs. cond_full vs. box-only vs. keypoint-only, evaluated with PSNR / SSIM / anatomy alignment score)*
```bash
python -m evaluation.evaluate_downstream ablation --data-dir ../data/perio_KPT/1_Experiment --box-type standard_box --fold 0 --ae-checkpoint ../checkpoints/autoencoder.pth --cond-checkpoint ../checkpoints/ldm_conditional_best.pth --output-dir ../results/ablation
```

*Downstream classification task (real-only vs. real+baseline-synthetic vs. real+conditional-synthetic, evaluated with accuracy / F1 / mAP)*
```bash
python -m evaluation.evaluate_downstream downstream --data-dir ../data/perio_KPT/1_Experiment --box-type standard_box --fold 0 --ae-checkpoint ../checkpoints/autoencoder.pth --baseline-checkpoint ../checkpoints/ldm_unet_debug.pth --cond-checkpoint ../checkpoints/ldm_conditional_best.pth --output-dir ../results/downstream --clf-epochs 20
```

The `downstream` command trains a lightweight CNN classifier three times (real data only; real + baseline-generated synthetic data; real + conditionally-generated synthetic data) on a pseudo-label derived from keypoint visibility, and reports **accuracy, F1, and mean Average Precision (mAP)**, computed with `sklearn.metrics.average_precision_score`, on a fixed held-out validation split — this is the metric the project brief asks for to quantify whether anatomically-conditioned synthetic data helps a downstream model more than unconditioned synthetic data.

*Output of part 3*
- Anatomical maps compatible with the latent resolution.
- Conditional Latent Diffusion Model.
- Qualitative and quantitative comparison between baseline and conditional LDM.
- Ablation study on the type and strength of conditioning (`--cond-weight` from 0.0 to 1.0, plus box-only / keypoint-only variants).
- Downstream evaluation (accuracy / F1 / mAP) using real and synthetic data.

---

## Quick sanity checks

Fast verification scripts for the environment and individual components, in `src/evaluation/`:

```bash
cd src
python -m evaluation.check_setup          # verify installed libraries (torch, cv2, etc.)
python -m evaluation.check_autoencoder    # ConvAutoencoder forward pass
python -m evaluation.check_diffusion      # diffusion scheduler forward pass
python -m evaluation.check_unet           # LatentUNet forward pass
python -m evaluation.check_real_data      # DataLoader on real data + autoencoder
```

## Expected outputs

- `checkpoints/`: weights of the autoencoder, baseline LDM and conditional LDM.
- `outputs/`: reconstructions, generated images and visual checks.
- `results/`: ablation and downstream task metrics (`ablation_results.json`, `downstream_results.json`).
- `presentation/`: final project slides.

## Requirements

- Python 3.10+, PyTorch, torchvision
- `opencv-python`, `numpy`, `matplotlib`
- `tqdm`
- `scikit-learn` (for `average_precision_score` in the downstream mAP evaluation)

## Authors

- Ciro Ronca
- Giovanni Contursi
- Vincenzo Nazzaro

Computer Vision, Prof. Irene Amerini — Sapienza Università di Roma, 2025/2026.

