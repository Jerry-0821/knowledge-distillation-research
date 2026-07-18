from pathlib import Path

import pytest
import torch
from PIL import Image

from kd_research.data.fashion_mnist import (
    FashionMNISTDataConfig,
    create_fashion_mnist_loaders,
    get_fashion_mnist_transform,
)


def test_fashion_mnist_config_defaults_are_for_inspection_only() -> None:
    config = FashionMNISTDataConfig()

    assert config.data_dir == Path("data")
    assert config.batch_size == 8
    assert config.download is False
    assert config.num_workers == 0


def test_fashion_mnist_transform_converts_grayscale_image_to_tensor() -> None:
    image = Image.new("L", (28, 28), color=0)

    tensor = get_fashion_mnist_transform()(image)

    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (1, 28, 28)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() == 0.0
    assert tensor.max().item() == 0.0


def test_create_fashion_mnist_loaders_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        create_fashion_mnist_loaders(batch_size=0, download=False)
