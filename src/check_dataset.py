"""
check_dataset.py
-----------------
Equivalente da terminale del notebook 01_dataset_exploration.ipynb.

Verifica che il modulo dataset (data.py) funzioni correttamente:
1. Costruisce i DataLoader (train/test/holdout) con get_dataloaders.
2. Stampa il numero di batch per split.
3. Visualizza un campione per split con box + keypoint (plot_sample_with_annotations).
4. Visualizza la maschera anatomica a 2 canali per un campione di train (plot_sample_with_mask).

Uso:
    python check_dataset.py --data-dir ../data/perio_KPT/1_Experiment \
                             --box-type standard_box --fold 0
"""

import argparse

from data import get_dataloaders
from utils import plot_sample_with_annotations, plot_sample_with_mask
from globals import DATA_DIR_EXPERIMENT


def parse_args():
    p = argparse.ArgumentParser(description="Dataset exploration / sanity check.")
    p.add_argument("--data-dir",   type=str, default=DATA_DIR_EXPERIMENT)
    p.add_argument("--box-type",   type=str, default="standard_box",
                   choices=["standard_box", "rotating_box"])
    p.add_argument("--fold",       type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def _first_sample(loader):
    imgs, labels = next(iter(loader))
    return imgs[0], {
        'classes':   labels.get('classes',   [[]])[0],
        'boxes':     labels.get('boxes',     [[]])[0],
        'keypoints': labels.get('keypoints', [[]])[0],
        'angles':    labels.get('angles',    [[]])[0],
    }


def main():
    args = parse_args()

    print("Configuration")
    print(f"  • Data dir  : {args.data_dir}")
    print(f"  • Box type  : {args.box_type}")
    print(f"  • Fold      : {args.fold}")
    print(f"  • Batch size: {args.batch_size}")

    dls = get_dataloaders(
        data_dir=args.data_dir,
        box_type=args.box_type,
        batch_size=args.batch_size,
        fold=args.fold,
    )

    print(f"Train batches   : {len(dls['train'])}")
    print(f"Test batches    : {len(dls['test'])}")
    print(f"Holdout batches : {len(dls['holdout'])}")

    # --- Visual check per split -------------------------------------------
    for split in ("train", "test", "holdout"):
        img, ann = _first_sample(dls[split])
        print(f"\n[{split}] shape immagine: {img.shape}")
        plot_sample_with_annotations(img, ann)

    # --- Maschera anatomica (train) ----------------------------------------
    img, ann = _first_sample(dls['train'])
    mask = plot_sample_with_mask(img, ann)
    print(f"\nMaschera shape : {mask.shape}")
    print(f"Box mask range : [{mask[0].min():.2f}, {mask[0].max():.2f}]")
    print(f"KPT mask range : [{mask[1].min():.2f}, {mask[1].max():.2f}]")


if __name__ == "__main__":
    main()