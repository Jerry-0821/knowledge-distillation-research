"""Train the Fashion-MNIST student with Knowledge Distillation.

This script loads a saved SimpleCNNTeacherModel checkpoint, keeps the teacher
fixed, and trains SimpleCNNStudentModel with a combined hard-label and
soft-target distillation loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from kd_research.data import create_fashion_mnist_loaders
from kd_research.models import SimpleCNNStudentModel, SimpleCNNTeacherModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Fashion-MNIST student with KD."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def validate_batch_limit(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided.")


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    true_labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    soft_loss = (
        F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        * (temperature**2)
    )
    hard_loss = F.cross_entropy(student_logits, true_labels)
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

    return hard_loss, soft_loss, total_loss


def load_teacher(
    checkpoint_path: Path,
    device: torch.device,
) -> SimpleCNNTeacherModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    with torch.random.fork_rng(devices=[]):
        teacher = SimpleCNNTeacherModel().to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    return teacher


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: Adam,
    device: torch.device,
    temperature: float,
    alpha: float,
    max_batches: int | None,
) -> tuple[float, float, float]:
    student.train()
    teacher.eval()

    total_hard_loss = 0.0
    total_soft_loss = 0.0
    total_combined_loss = 0.0
    total_examples = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            teacher_logits = teacher(images)

        student_logits = student(images)
        hard_loss, soft_loss, combined_loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            true_labels=labels,
            temperature=temperature,
            alpha=alpha,
        )

        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_hard_loss += hard_loss.item() * batch_size
        total_soft_loss += soft_loss.item() * batch_size
        total_combined_loss += combined_loss.item() * batch_size
        total_examples += batch_size

    return (
        total_hard_loss / total_examples,
        total_soft_loss / total_examples,
        total_combined_loss / total_examples,
    )


def evaluate_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    correct = 0
    total_examples = 0

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    return correct / total_examples


def main() -> None:
    args = parse_args()
    validate_batch_limit("--max-train-batches", args.max_train_batches)
    validate_batch_limit("--max-eval-batches", args.max_eval_batches)

    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    train_loader, test_loader = create_fashion_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        download=args.download,
        num_workers=0,
    )

    student = SimpleCNNStudentModel().to(device)
    teacher = load_teacher(
        checkpoint_path=args.teacher_checkpoint,
        device=device,
    )
    optimizer = Adam(student.parameters(), lr=args.learning_rate)

    print("Fashion-MNIST distillation student")
    print(f"device: {device}")
    print(f"teacher checkpoint: {args.teacher_checkpoint}")
    print(f"batch size: {args.batch_size}")
    print(f"epochs: {args.epochs}")
    print(f"learning rate: {args.learning_rate}")
    print(f"temperature: {args.temperature}")
    print(f"alpha: {args.alpha}")
    print(f"seed: {args.seed}")
    print(f"max train batches: {args.max_train_batches}")
    print(f"max eval batches: {args.max_eval_batches}")
    print("label type: hard labels + teacher soft targets")

    for epoch in range(1, args.epochs + 1):
        hard_loss, soft_loss, combined_loss = train_one_epoch(
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature,
            alpha=args.alpha,
            max_batches=args.max_train_batches,
        )
        test_accuracy = evaluate_accuracy(
            model=student,
            loader=test_loader,
            device=device,
            max_batches=args.max_eval_batches,
        )

        print(
            f"epoch {epoch}: "
            f"hard loss = {hard_loss:.4f}, "
            f"soft loss = {soft_loss:.4f}, "
            f"combined loss = {combined_loss:.4f}, "
            f"test accuracy = {test_accuracy:.4f}"
        )

    print("Teacher checkpoint was fixed; only the student was trained.")


if __name__ == "__main__":
    main()
