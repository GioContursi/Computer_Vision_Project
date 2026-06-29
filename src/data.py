from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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