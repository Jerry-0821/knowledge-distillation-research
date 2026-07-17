"""Run one forward pass through SimpleCNNTeacherModel without training."""

from __future__ import annotations

import torch

from kd_research.models import SimpleCNNTeacherModel


def main() -> None:
    model = SimpleCNNTeacherModel()
    images = torch.zeros((8, 1, 28, 28), dtype=torch.float32)

    with torch.inference_mode():
        logits = model(images)

    print("Teacher model forward-pass check")
    print(f"input image batch shape: {tuple(images.shape)}")
    print(f"output logits shape: {tuple(logits.shape)}")
    print("No training was run.")


if __name__ == "__main__":
    main()
