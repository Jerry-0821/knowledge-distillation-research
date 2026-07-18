"""Download/check Fashion-MNIST without training any model.

This script is a small data smoke test:
- It can download Fashion-MNIST through TorchVision when ``--download`` is used.
- It prints dataset sizes and one batch shape.
- It does not train, evaluate, or save checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kd_research.data.fashion_mnist import create_fashion_mnist_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Fashion-MNIST data loading.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Project-local data directory. Dataset files should not be committed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Small inspection batch size only; not a final experiment setting.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow TorchVision to download Fashion-MNIST if it is not present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_loader, test_loader = create_fashion_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        download=args.download,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print("Fashion-MNIST data check")
    print(f"data directory: {args.data_dir}")
    print(f"train dataset size: {len(train_loader.dataset)}")
    print(f"test dataset size: {len(test_loader.dataset)}")
    print(f"image batch shape: {tuple(images.shape)}")
    print(f"label batch shape: {tuple(labels.shape)}")
    print(f"image dtype: {images.dtype}")
    print(f"label dtype: {labels.dtype}")
    print(f"image value range: [{images.min().item():.3f}, {images.max().item():.3f}]")
    print(f"sample labels: {labels.tolist()}")
    print("No training was run.")


if __name__ == "__main__":
    main()
