"""Train the MNIST teacher model with hard labels.

This script trains SimpleCNNTeacherModel with true MNIST labels only. The
trained teacher will later be used as the source of soft targets, but this
script does not run Knowledge Distillation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from kd_research.data import create_mnist_loaders
from kd_research.models import SimpleCNNTeacherModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST hard-label teacher.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def evaluate_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total_examples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    return correct / total_examples


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    train_loader, test_loader = create_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        download=args.download,
        num_workers=0,
    )

    model = SimpleCNNTeacherModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    print("MNIST hard-label teacher")
    print(f"device: {device}")
    print(f"batch size: {args.batch_size}")
    print(f"epochs: {args.epochs}")
    print(f"learning rate: {args.learning_rate}")
    print(f"seed: {args.seed}")
    print("label type: hard labels only")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_accuracy = evaluate_accuracy(
            model=model,
            loader=test_loader,
            device=device,
        )

        print(
            f"epoch {epoch}: "
            f"train loss = {train_loss:.4f}, "
            f"test accuracy = {test_accuracy:.4f}"
        )

    if args.checkpoint_path is not None:
        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": "SimpleCNNTeacherModel",
                "model_state_dict": model.state_dict(),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
            args.checkpoint_path,
        )
        print(f"Saved teacher checkpoint to {args.checkpoint_path}")

    print("No soft targets or distillation loss was used.")


if __name__ == "__main__":
    main()
