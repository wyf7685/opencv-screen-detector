"""Tests for dataset module."""

import pytest
import torch


def test_dataset_loads_images(data_dir):
    """Test TwoInputDataset loads images correctly."""
    from trainer.dataset import TwoInputDataset

    data_map = {
        "natural": ["natural_photo"],
        "screen_like": ["screen_like", "screenshot"],
    }

    dataset = TwoInputDataset(data_map=data_map, data_dir=data_dir)

    assert len(dataset) > 0, "Dataset should have at least one sample"


def test_dataset_returns_rgb_fft_label(data_dir):
    """Test dataset returns (rgb, fft, label) tuple."""
    from trainer.dataset import TwoInputDataset

    data_map = {
        "natural": ["natural_photo"],
        "screen_like": ["screen_like"],
    }

    dataset = TwoInputDataset(data_map=data_map, data_dir=data_dir)

    if len(dataset) == 0:
        pytest.skip("No samples found")

    rgb, fft, label = dataset[0]

    assert isinstance(rgb, torch.Tensor)
    assert isinstance(fft, torch.Tensor)
    assert isinstance(label, int)

    assert rgb.shape == (3, 224, 224)
    assert fft.shape == (1, 224, 224)


def test_dataset_natural_photo_recursive(data_dir):
    """Test natural_photo subdirectories are scanned recursively."""
    from trainer.dataset import TwoInputDataset

    data_map = {
        "natural": ["natural_photo"],
    }

    dataset = TwoInputDataset(data_map=data_map, data_dir=data_dir)

    # Should find images in subdirectories
    assert len(dataset) > 10, "Should find images in natural_photo subdirectories"


def test_dataset_stage1_label_mapping(data_dir):
    """Test Stage 1 label mapping."""
    from trainer.dataset import TwoInputDataset

    data_map = {
        "natural": ["natural_photo"],
        "screen_like": ["screen_like", "screenshot"],
    }

    dataset = TwoInputDataset(data_map=data_map, data_dir=data_dir)

    labels = set()
    for _, _, label in dataset:
        labels.add(label)

    assert labels == {0, 1}, f"Expected labels {{0, 1}}, got {labels}"


def test_dataset_stage2_label_mapping(data_dir):
    """Test Stage 2 label mapping."""
    from trainer.dataset import TwoInputDataset

    data_map = {
        "screenshot": ["screenshot"],
        "screen_photo": ["screen_photo"],
    }

    dataset = TwoInputDataset(data_map=data_map, data_dir=data_dir)

    labels = set()
    for _, _, label in dataset:
        labels.add(label)

    assert labels == {0, 1}, f"Expected labels {{0, 1}}, got {labels}"
