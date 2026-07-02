"""Run a tiny PyTorch forward-pass smoke test without training."""

from __future__ import annotations

import sys


def main() -> int:
    """Check tensor operations and one tiny forward pass."""

    try:
        import torch
        from torch import nn
    except (ImportError, OSError) as error:
        print(
            f"PyTorch import failed: {type(error).__name__}. "
            "Run scripts/setup.ps1 after choosing a PyTorch build.",
            file=sys.stderr,
        )
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)
    doubled = dummy * 2
    layer = nn.Linear(4, 2).to(device)

    with torch.inference_mode():
        output = layer(doubled)

    print("=== PyTorch Smoke Test ===")
    print(f"Device: {device}")
    print(f"Input shape: {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
