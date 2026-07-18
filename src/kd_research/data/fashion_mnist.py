"""Minimal Fashion-MNIST data loading utilities.

This module only prepares data access for sanity checks and future experiments.
It does not define models, losses, training loops, or Knowledge Distillation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class FashionMNISTDataConfig:
    """Small configuration object for Fashion-MNIST data loading.

    The default batch size is only for quick inspection, not a final experiment
    hyperparameter.
    """

    data_dir: Path = Path("data")
    batch_size: int = 8
    download: bool = False
    num_workers: int = 0


def get_fashion_mnist_transform() -> transforms.Compose:
    """Return the basic Fashion-MNIST transform."""

    return transforms.Compose([transforms.ToTensor()])


def get_fashion_mnist_datasets(
    data_dir: str | Path = "data",
    *,
    download: bool = False,
) -> tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """Create Fashion-MNIST train and test datasets."""

    transform = get_fashion_mnist_transform()
    root = Path(data_dir)

    train_dataset = datasets.FashionMNIST(
        root=str(root),
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=str(root),
        train=False,
        download=download,
        transform=transform,
    )
    return train_dataset, test_dataset


def create_fashion_mnist_loaders(
    data_dir: str | Path = "data",
    *,
    batch_size: int = 8,
    download: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create Fashion-MNIST train and test DataLoaders."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be zero or a positive integer.")

    train_dataset, test_dataset = get_fashion_mnist_datasets(
        data_dir=data_dir,
        download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
