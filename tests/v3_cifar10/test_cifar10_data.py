# This file is a quick go-through / safety check for the CIFAR-10 data-loader code.

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

import kd_research.data.cifar10 as cifar10_module
from kd_research.data import (
    CIFAR10DataConfig,
    create_cifar10_loaders,
    create_cifar10_train_val_test_loaders,
    get_cifar10_transform,
)


def test_cifar10_data_config_defaults() -> None:
    config = CIFAR10DataConfig()

    assert config.data_dir == Path("data")
    assert config.batch_size == 8
    assert config.download is False
    assert config.num_workers == 0

def test_create_cifar10_loaders_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        create_cifar10_loaders(batch_size=0)

def test_create_cifar10_loaders_rejects_negative_num_workers() -> None:
    with pytest.raises(ValueError, match="num_workers must be zero or a positive integer"):
        create_cifar10_loaders(num_workers=-1)

def test_create_cifar10_train_val_test_loaders_splits_train_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCIFAR10Dataset(Dataset):
        def __init__(self, size: int) -> None:
            self.size = size

        def __len__(self) -> int:
            return self.size

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(3, 32, 32), index % 10

    def fake_get_cifar10_datasets(
        data_dir: str = "data",
        *,
        download: bool = False,
    ) -> tuple[FakeCIFAR10Dataset, FakeCIFAR10Dataset]:
        return FakeCIFAR10Dataset(50), FakeCIFAR10Dataset(10)

    monkeypatch.setattr(cifar10_module, "get_cifar10_datasets", fake_get_cifar10_datasets)

    train_loader, val_loader, test_loader = create_cifar10_train_val_test_loaders(
        batch_size=5,
        val_size=5,
        split_seed=0,
    )

    assert len(train_loader.dataset) == 45
    assert len(val_loader.dataset) == 5
    assert len(test_loader.dataset) == 10

def test_cifar10_transform_returns_rgb_tensor() -> None:
    transform = get_cifar10_transform()
    image = Image.new("RGB", (32, 32), color=(255, 128, 0))

    tensor = transform(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0
