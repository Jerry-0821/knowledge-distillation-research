"""Train the CIFAR-10 hard-label student baseline"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from kd_research.data import create_cifar10_train_val_test_loaders
from kd_research.models import CIFAR10CNNStudentModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 hard-label student baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()

def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested_device)

def validate_batch_limit(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided.")



def train_one_epoch(
        model: nn.Module, loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module, optimizer: Adam, device: torch.device,
        max_batches: int | None,
) -> tuple[float, float]:

    model.train()
    total_loss = 0.0
    correct = 0
    total_examples = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)
        predictions = logits.argmax(dim=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, correct / total_examples

def evaluate_loss_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_examples = 0

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    return total_loss / total_examples, correct / total_examples


def main() -> None:
    args = parse_args()
    validate_batch_limit("--max-train-batches", args.max_train_batches)
    validate_batch_limit("--max-eval-batches", args.max_eval_batches)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, test_loader = create_cifar10_train_val_test_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size, download=args.download,
        num_workers=0, val_size=args.val_size, split_seed=args.split_seed
    )

    model = CIFAR10CNNStudentModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    print("CIFAR-10 hard-label student baseline")
    print(f"device: {device}")
    print(f"batch size: {args.batch_size}")
    print(f"epochs: {args.epochs}")
    print(f"learning rate: {args.learning_rate}")
    print(f"seed: {args.seed}")
    print(f"train dataset size: {len(train_loader.dataset)}")
    print(f"validation dataset size: {len(val_loader.dataset)}")
    print(f"test dataset size: {len(test_loader.dataset)}")
    print(f"validation split seed: {args.split_seed}")
    print(f"max train batches: {args.max_train_batches}")
    print(f"max eval batches: {args.max_eval_batches}")
    print(f"evaluate test: {args.evaluate_test}")
    print("label type: hard labels only")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model, loader=train_loader, loss_fn=loss_fn, optimizer=optimizer,
            device=device, max_batches=args.max_train_batches
        )

        val_loss, val_accuracy = evaluate_loss_accuracy(
            model=model, loader=val_loader, loss_fn=loss_fn, device=device, max_batches=args.max_eval_batches
        )

        print(
            f"epoch {epoch}: "
            f"train loss = {train_loss:.4f}, "
            f"train accuracy = {train_accuracy:.4f}, "
            f"validation loss = {val_loss:.4f}, "
            f"validation accuracy = {val_accuracy:.4f}"
        )

    if args.evaluate_test:
        test_loss, test_accuracy = evaluate_loss_accuracy(
            model=model, loader=test_loader, loss_fn=loss_fn, device=device, max_batches=args.max_eval_batches
        )
        print(f"final test loss = {test_loss:.4f}, final test accuracy = {test_accuracy:.4f}")

    if args.checkpoint_path is not None:
        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": "CIFAR10CNNStudentModel",
                "dataset": "CIFAR-10",
                "device": str(device),
                "model_state_dict": model.state_dict(),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "val_size": args.val_size,
                "split_seed": args.split_seed,
                "evaluate_test": args.evaluate_test,
                "max_train_batches": args.max_train_batches,
                "max_eval_batches": args.max_eval_batches,
            },
            args.checkpoint_path,
        )
        print(f"Saved student checkpoint to {args.checkpoint_path}")

    print("No soft targets or distillation loss was used.")


if __name__ == "__main__":
    main()
