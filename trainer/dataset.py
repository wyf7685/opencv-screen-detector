"""Dataset loader for screen detector V3 training.

Supports:
- Two-input dataset (RGB + FFT)
- Recursive subdirectory scanning (for natural_photo/ subfolders)
- Data map configuration for two-stage training
"""
# pyright: reportPrivateImportUsage=none

import hashlib
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from . import config

# Image index path for tracking trained images
IMAGE_INDEX_PATH = config.PROJECT_ROOT / "data" / "image_index.json"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_image_index() -> dict:
    """Load image index from JSON file."""
    if IMAGE_INDEX_PATH.exists():
        with IMAGE_INDEX_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_image_index(index: dict) -> None:
    """Save image index to JSON file."""
    IMAGE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with IMAGE_INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def _get_file_hash(file_path: str) -> str:
    """Compute MD5 hash of file content."""
    h = hashlib.md5()  # noqa: S324
    with Path(file_path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_fft_spectrum(image_np: np.ndarray, size: int = 224) -> np.ndarray:
    """将图像转换为 FFT 频谱图

    Args:
        image_np: 输入图像 (RGB numpy array)
        size: 输出尺寸

    Returns:
        FFT 频谱图，形状 (1, H, W)
    """
    # 转灰度
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # 先 resize 到目标尺寸，避免大图片导致内存不足
    gray = cv2.resize(gray, (size, size))

    # FFT
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # 频谱图 (magnitude)
    magnitude = np.log(np.abs(fshift) + 1)

    # 归一化到 [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # pyright: ignore[reportCallIssue, reportArgumentType]

    # 归一化到 [0, 1] 然后 ImageNet 灰度归一化
    magnitude = magnitude.astype(np.float32) / 255.0
    mean, std = 0.449, 0.226
    magnitude = (magnitude - mean) / std

    return magnitude.reshape(1, size, size)


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
        self.image_ids: list[str] = []

        self._load_samples(data_map)

    def _load_samples(self, data_map: dict[str, list[str]]) -> None:
        """Load all image paths and labels from data map."""
        index = _load_image_index()
        hash_to_id = {info["hash"]: img_id for img_id, info in index.items()}
        index_modified = False

        for class_idx, (_class_name, source_dirs) in enumerate(data_map.items()):
            for source_dir in source_dirs:
                dir_path = self.data_dir / source_dir
                if not dir_path.exists():
                    continue

                # 递归扫描，支持子目录
                for img_path in dir_path.rglob("*"):
                    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue

                    file_hash = _get_file_hash(str(img_path))
                    img_id = hash_to_id.get(file_hash)

                    # If image not in index, add it
                    if img_id is None:
                        img_id = str(uuid.uuid4())
                        index[img_id] = {
                            "hash": file_hash,
                            "filename": img_path.name,
                            "class": source_dir,
                            "path": str(img_path),
                            "trained": False,
                        }
                        hash_to_id[file_hash] = img_id
                        index_modified = True

                    self.image_ids.append(img_id)
                    self.samples.append((str(img_path), class_idx))

        # Save index if modified
        if index_modified:
            _save_image_index(index)

    def __len__(self) -> int:
        return len(self.samples)

    def get_image_ids(self) -> list[str]:
        """Get list of image IDs for all samples."""
        return self.image_ids

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image using PIL (RGB)
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # RGB 分支
        if self.transform:
            augmented = self.transform(image=image_np)
            rgb_tensor = augmented["image"]
        else:
            # Default: resize and normalize
            image_resized = cv2.resize(image_np, (self.image_size, self.image_size))
            rgb_tensor = (
                torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            )

        # FFT 分支
        fft_spectrum = compute_fft_spectrum(image_np, self.image_size)
        fft_tensor = torch.from_numpy(fft_spectrum).float()

        return rgb_tensor, fft_tensor, label


class TransformSubset(Dataset):
    """Subset of a dataset with applied transform."""

    def __init__(self, subset, transform=None) -> None:
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
    transform_train=None,
    transform_val=None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    train_ratio: float = config.TRAIN_VAL_SPLIT,
):
    """Create train and validation data loaders for two-input dataset."""

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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
