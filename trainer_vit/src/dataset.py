"""ViT dataset for screen detector.

Three-class classification: natural, screenshot, screen_photo.
"""

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import get_train_transforms, get_val_transforms

# Label mapping
LABEL_MAP = {
    "natural_photo": 0,  # natural
    "screenshot": 1,     # screenshot
    "screen_photo": 2,   # screen_photo
}

LABEL_NAMES = ["natural", "screenshot", "screen_photo"]

# Valid image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ScreenDetectorDataset(Dataset):
    """Screen detector dataset for ViT training.

    Args:
        data_dir: Root directory containing class folders
        transform: Albumentations transform
        split: Dataset split ('train' or 'val')
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: A.Compose | None = None,
        split: str = "train",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split

        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load all samples from data directory."""
        for class_name, label in LABEL_MAP.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"{class_dir} does not exist, skipping")
                continue

            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, label))

        logger.info(f"[{self.split}] Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image tensor, label)
        """
        img_path, label = self.samples[idx]

        # Load image as RGB
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    image_size: int = 224,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Validation split ratio
        image_size: Image size for ViT
        seed: Random seed for split

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create train dataset with augmentation
    train_dataset = ScreenDetectorDataset(
        data_dir=data_dir,
        transform=get_train_transforms(image_size),
        split="train",
    )

    # Create val dataset without augmentation
    val_dataset = ScreenDetectorDataset(
        data_dir=data_dir,
        transform=get_val_transforms(image_size),
        split="val",
    )

    # Split dataset
    total_size = len(train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_subset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_subset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader
