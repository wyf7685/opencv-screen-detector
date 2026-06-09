"""Dataset loader for screen detector V3 training.

Supports:
- Two-input dataset (RGB + FFT)
- Recursive subdirectory scanning (for natural_photo/ subfolders)
- Data map configuration for single-stage three-class training
- WeightedRandomSampler for oversampling minority classes
"""

# pyright: reportPrivateImportUsage=none
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
    random_split,
)

from shared.fft_transform import compute_fft_spectrum as _compute_fft_shared

from . import config

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_fft_spectrum(image_np: np.ndarray, size: int = 224) -> np.ndarray:
    """将图像转换为 FFT 频谱图

    Args:
        image_np: 输入图像 (RGB numpy array)
        size: 输出尺寸

    Returns:
        FFT 频谱图，形状 (1, H, W)
    """
    # shared 版本返回 (1, 1, H, W)，squeeze 掉 batch 维度
    return _compute_fft_shared(image_np, size, color_space="rgb").squeeze(0)


class TwoInputDataset(Dataset):
    """返回 (rgb_image, fft_spectrum, label) 的数据集

    支持:
    - data_map 配置，从多个目录加载并映射到统一标签
    - rglob("*") 递归扫描子目录
    """

    def __init__(
        self,
        data_map: dict[str, list[str]],
        data_dir: Path,
        transform=None,
        image_size: int = config.IMAGE_SIZE,
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size

        self.samples: list[tuple[str, int]] = []  # (image_path, label_idx)

        self._load_samples(data_map)

    def _load_samples(self, data_map: dict[str, list[str]]) -> None:
        """Load all image paths and labels from data map."""
        for class_idx, (_class_name, source_dirs) in enumerate(data_map.items()):
            for source_dir in source_dirs:
                dir_path = self.data_dir / source_dir
                if not dir_path.exists():
                    continue

                # 递归扫描，支持子目录
                for img_path in dir_path.rglob("*"):
                    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    self.samples.append((str(img_path), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image using PIL (RGB)
        image = np.array(Image.open(img_path).convert("RGBA").convert("RGB"))

        # RGB 分支
        if self.transform:
            augmented = self.transform(image=image)
            rgb_tensor = augmented["image"]
        else:
            # Default: resize and normalize
            image_resized = cv2.resize(image, (self.image_size, self.image_size))
            rgb_tensor = (
                torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            )

        # FFT 分支
        fft_spectrum = compute_fft_spectrum(image, self.image_size)
        fft_tensor = torch.from_numpy(fft_spectrum).float()

        return rgb_tensor, fft_tensor, label


class TransformSubset(Dataset):
    """Subset of a dataset with applied transform."""

    def __init__(
        self,
        subset: Subset[tuple[torch.Tensor, torch.Tensor, int]],
        transform: A.Compose | None = None,
    ) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        rgb_tensor, fft_tensor, label = self.subset[idx]

        # Convert tensor back to numpy for albumentations
        if isinstance(rgb_tensor, torch.Tensor):
            image_np = rgb_tensor.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = rgb_tensor

        if self.transform:
            augmented = self.transform(image=image_np)
            rgb_tensor = augmented["image"]

        return rgb_tensor, fft_tensor, label


def create_data_loaders(
    data_map: dict[str, list[str]],
    data_dir: Path,
    transform_train: A.Compose | None = None,
    transform_val: A.Compose | None = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    train_ratio: float = config.TRAIN_VAL_SPLIT,
    use_weighted_sampler: bool = False,
):
    """Create train and validation data loaders for two-input dataset.

    Args:
        data_map: Data mapping {class_name: [source_dirs]}
        data_dir: Data directory
        transform_train: Training transforms
        transform_val: Validation transforms
        batch_size: Batch size
        num_workers: Number of workers
        train_ratio: Train/val split ratio
        use_weighted_sampler: Whether to use WeightedRandomSampler for oversampling

    Returns:
        Tuple of (train_loader, val_loader, full_dataset)
    """

    # Create full dataset
    full_dataset = TwoInputDataset(
        data_map=data_map,
        data_dir=data_dir,
        transform=None,
    )

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # Apply transforms by wrapping datasets
    train_dataset = TransformSubset(train_dataset, transform_train)
    val_dataset = TransformSubset(val_dataset, transform_val)

    # Create sampler for oversampling if requested
    train_sampler = None
    shuffle = True

    if use_weighted_sampler:
        # Get labels for training subset
        if hasattr(train_dataset, "subset"):
            train_indices = train_dataset.subset.indices
        else:
            train_indices = list(range(len(train_dataset)))
        train_labels = [full_dataset.samples[i][1] for i in train_indices]

        # Calculate class counts and weights
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        class_weights = {
            cls: total_samples / count
            for cls, count in class_counts.items()
        }

        # Assign weight to each sample
        sample_weights = [class_weights[label] for label in train_labels]
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

        train_sampler = WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True,
        )
        shuffle = False  # Sampler handles shuffling

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset
