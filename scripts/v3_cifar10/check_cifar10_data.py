"""Download/check CIFAR-10 without training any model.

This script is a small data smoke test:
- It can download CIFAR-10 through TorchVision when --download is used.
- It prints dataset sizes and one batch shape.
- It does not train, evaluate, or save checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kd_research.data import create_cifar10_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CIFAR-10 data loading.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_loader, test_loader = create_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        download=args.download,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print("CIFAR-10 data check")
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