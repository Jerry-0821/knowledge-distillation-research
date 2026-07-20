"""Minimal CIFAR-10 data loading utilities.

This module only prepares data access for sanity checks and future experiments.
"""

#define CIFAR-10 transform
#create train/test CIFAR-10 datasets
#create train/test DataLoaders
#validate simple loader arguments like batch_size > 0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# actually there is another expression method . config ={......}
@dataclass(frozen=True)
class CIFAR10DataConfig:
    """Small configuration object for CIFAR-10 data loading.

    The default batch size is only for quick inspection, not a final experiment
    hyperparameter.
    """
    data_dir: Path = Path("data")
    batch_size: int = 8
    download: bool = False
    num_workers: int = 0

def get_cifar10_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


def get_cifar10_datasets(
    data_dir: str | Path = "data",
    *,
    download: bool = False,
) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Create CIFAR-10 train and test datasets."""

    root = Path(data_dir)
    transform = get_cifar10_transform()

    # dataset
    train_dataset = datasets.CIFAR10(
        root=str(root), train=True, transform=transform, download=download
    )
    test_dataset = datasets.CIFAR10(
        root=str(root), train=False, transform=transform, download=download)
    return train_dataset, test_dataset

def create_cifar10_loaders(
    data_dir: str | Path = "data",
    *,
    batch_size: int = 8,
    download: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test DataLoaders."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be zero or a positive integer.")

    train_dataset, test_dataset = get_cifar10_datasets(data_dir=data_dir, download=download)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def create_cifar10_train_val_test_loaders(
    data_dir: str | Path = "data",
    *,
    batch_size: int = 8,
    val_size: int = 5000,
    split_seed: int = 0,
    download: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create CIFAR-10 train, validation, and test DataLoaders."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be zero or a positive integer.")
    if val_size <= 0:
        raise ValueError("val_size must be a positive integer.")

    train_dataset, test_dataset = get_cifar10_datasets(data_dir=data_dir, download=download)
    train_size = len(train_dataset) - val_size

    if train_size <= 0:
        raise ValueError("val_size must be smaller than the CIFAR-10 train dataset size.")

    generator = torch.Generator().manual_seed(split_seed)
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
