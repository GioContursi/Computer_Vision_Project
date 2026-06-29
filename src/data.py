import os
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


# =========================================================================
# 1. DATASET BASE (Solo immagini, utile per pre-training iniziale Autoencoder)
# =========================================================================
class DentalImageDataset(Dataset):
    def __init__(self, image_dir, image_size=128, max_images=None):
        self.image_dir = Path(image_dir)

        self.image_paths = sorted(
            list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.jpeg"))
        )

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image


# =========================================================================
# 2. DATASET AVANZATO (Patching condizionato, Keypoints, Folds, YOLO)
# =========================================================================
class PerioKPTDataset(Dataset):
    def __init__(self, data_dir, split='train', box_type='standard_box', fold=0):
        self.data_dir = data_dir
        self.split = split
        self.box_type = box_type  # 'standard_box' o 'rotating_box'
        self.fold = fold

        # Gestione del set di holdout (Test finale)
        if split == 'holdout':
            base_dir = os.path.join(data_dir, f'holdout_test_{box_type}')
            if os.path.exists(os.path.join(base_dir, 'images')):
                self.images_dir = os.path.join(base_dir, 'images')
                self.labels_dir = os.path.join(base_dir, 'labels')
            else:
                self.images_dir = base_dir
                self.labels_dir = base_dir
        else:
            actual_split = 'val' if split == 'test' else split
            base_dir = os.path.join(data_dir, box_type, f'f{self.fold}', actual_split)
            if os.path.exists(os.path.join(base_dir, 'images')):
                self.images_dir = os.path.join(base_dir, 'images')
                self.labels_dir = os.path.join(base_dir, 'labels')
            else:
                self.images_dir = base_dir
                self.labels_dir = base_dir

        # Creiamo un elenco di campioni (patch) invece di un elenco di immagini
        image_filenames = sorted([
            f for f in os.listdir(self.images_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        self.samples = []
        for img_name in image_filenames:
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_name)
            
            # Parsiamo le annotazioni per capire quanti denti ci sono
            annotations = self._parse_yolo_txt(label_path)
            
            if len(annotations['boxes']) > 0:
                # Se ci sono N denti, creiamo N campioni per questa immagine
                for i in range(len(annotations['boxes'])):
                    self.samples.append({'img_name': img_name, 'target_box_idx': i})
            else:
                # Se l'immagine non ha denti annotati, la teniamo per fare un crop centrale
                self.samples.append({'img_name': img_name, 'target_box_idx': -1})

    def _parse_yolo_txt(self, txt_path):
        """Analizza il file .txt secondo le specifiche del paper."""
        annotations = {'classes': [], 'boxes': [], 'keypoints': [], 'angles': []}

        if not os.path.exists(txt_path):
            return annotations

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = [float(x) for x in line.strip().split()]
            if len(parts) == 0:
                continue

            annotations['classes'].append(int(parts[0]))
            annotations['boxes'].append(parts[1:5])  # [cx, cy, w, h] normalizzati

            if self.box_type == 'standard_box' and len(parts) >= 38:
                kpts_flat = parts[5:]
                kpts_grouped = []
                for i in range(0, 33, 3):
                    kpts_grouped.append([kpts_flat[i], kpts_flat[i+1], kpts_flat[i+2]])
                annotations['keypoints'].append(kpts_grouped)

            elif self.box_type == 'rotating_box' and len(parts) == 6:
                annotations['angles'].append(parts[5])

        return annotations

    def __len__(self):
        # Ora la lunghezza del dataset è il numero totale di DENTI, non di immagini
        return len(self.samples)

    def __getitem__(self, idx):
        # Recuperiamo quale immagine e quale specifico dente dobbiamo estrarre
        sample_info = self.samples[idx]
        img_name = sample_info['img_name']
        target_box_idx = sample_info['target_box_idx']
        
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert('L')
        img_w, img_h = image.size

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)
        
        # Ri-parsiamo le annotazioni "fresche" per modificarle senza effetti collaterali
        annotations = self._parse_yolo_txt(label_path)

        # --- PATCH EXTRACTION basata sul bounding box reale ---
        patch_size = 128
        padding = 1.3  # 30% di margine attorno al dente

        if target_box_idx != -1:
            # Estraiamo esattamente il dente richiesto da questo indice! (Niente più random)
            target_box = annotations['boxes'][target_box_idx]
            cx_norm, cy_norm, w_norm, h_norm = target_box

            # Centro del dente in pixel
            cx_pixel = int(cx_norm * img_w)
            cy_pixel = int(cy_norm * img_h)

            # Dimensione del crop basata sul box reale + padding
            crop_w = int(w_norm * img_w * padding)
            crop_h = int(h_norm * img_h * padding)
            crop_dim = max(crop_w, crop_h)  # crop quadrato

            # Assicurati che il crop sia almeno patch_size
            crop_dim = max(crop_dim, patch_size)

            # Coordinate top-left centrate sul dente
            top  = max(0, cy_pixel - crop_dim // 2)
            left = max(0, cx_pixel - crop_dim // 2)

            # Aggiusta se esce dai bordi
            if top  + crop_dim > img_h: top  = img_h - crop_dim
            if left + crop_dim > img_w: left = img_w - crop_dim
            top  = max(0, top)
            left = max(0, left)

            # Ritaglia e ridimensiona a 128x128
            image_patch = TF.crop(image, top, left, crop_dim, crop_dim)
            image_patch = TF.resize(image_patch, [patch_size, patch_size])
        else:
            # Nessuna annotazione: center crop di default
            image_patch = TF.center_crop(image, [patch_size, patch_size])

        # --- NORMALIZZAZIONE in [-1, 1] ---
        image_tensor = TF.to_tensor(image_patch)
        image_tensor = TF.normalize(image_tensor, mean=[0.5], std=[0.5])

        return image_tensor, annotations


# =========================================================================
# 3. HELPER FUNCTIONS
# =========================================================================
def collate_fn(batch):
    """Gestisce batch con annotazioni di lunghezza variabile."""
    images = torch.stack([item[0] for item in batch])

    classes_list, boxes_list, keypoints_list, angles_list = [], [], [], []

    for _, annotations in batch:
        classes_list.append([torch.tensor(c) for c in annotations['classes']])
        boxes_list.append([torch.tensor(b, dtype=torch.float32) for b in annotations['boxes']])

        kpts = []
        for kpt_group in annotations['keypoints']:
            kpts.append(torch.tensor(kpt_group, dtype=torch.float32))  # [11, 3]
        keypoints_list.append(kpts)

        angles_list.append([torch.tensor(a, dtype=torch.float32) for a in annotations['angles']])

    return images, {
        'classes':   classes_list,
        'boxes':     boxes_list,
        'keypoints': keypoints_list,
        'angles':    angles_list
    }


def get_dataloaders(data_dir, box_type='standard_box', batch_size=16, num_workers=0, fold=0):
    print(f"[*] Inizializzazione DataLoader - Type: {box_type}, Fold: {fold}, Batch Size: {batch_size}")

    datasets = {
        'train':   PerioKPTDataset(data_dir, split='train',   box_type=box_type, fold=fold),
        'test':    PerioKPTDataset(data_dir, split='test',    box_type=box_type, fold=fold),
        'holdout': PerioKPTDataset(data_dir, split='holdout', box_type=box_type)
    }

    dataloaders = {
        'train':   DataLoader(datasets['train'],   batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True,  collate_fn=collate_fn),
        'test':    DataLoader(datasets['test'],    batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn),
        'holdout': DataLoader(datasets['holdout'], batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    }

    return dataloaders