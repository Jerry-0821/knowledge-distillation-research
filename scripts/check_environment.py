"""Print a privacy-conscious summary of the local research environment."""

from __future__ import annotations

import platform
from pathlib import Path
import shutil
import subprocess


def print_system_gpu() -> None:
    """Report NVIDIA hardware through nvidia-smi when it is available."""

    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi available: False")
        print("System NVIDIA GPU: Not detected")
        print("NVIDIA driver: Not detected")
        return

    print("nvidia-smi available: True")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except OSError as error:
        print(f"System NVIDIA GPU: Check failed ({type(error).__name__})")
        print("NVIDIA driver: Unknown")
        return

    if result.returncode != 0 or not result.stdout.strip():
        print("System NVIDIA GPU: nvidia-smi returned no usable data")
        print("NVIDIA driver: Unknown")
        return

    fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
    print(f"System NVIDIA GPU: {fields[0]}")
    print(f"NVIDIA driver: {fields[1] if len(fields) > 1 else 'Unknown'}")
    print(f"GPU memory: {fields[2] if len(fields) > 2 else 'Unknown'}")


def print_pytorch() -> None:
    """Report PyTorch and TorchVision without crashing when unavailable."""

    try:
        import torch
    except (ImportError, OSError) as error:
        print(f"PyTorch version: Not available ({type(error).__name__})")
        print("TorchVision version: Not checked")
        print("CUDA available: False")
        print("PyTorch CUDA version: Not available")
        print("PyTorch GPU name: Not available")
        return

    print(f"PyTorch version: {torch.__version__}")

    try:
        import torchvision
    except (ImportError, OSError) as error:
        print(f"TorchVision version: Not available ({type(error).__name__})")
    else:
        print(f"TorchVision version: {torchvision.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    print(f"PyTorch CUDA version: {torch.version.cuda or 'None (CPU-only build)'}")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None via PyTorch"
    print(f"PyTorch GPU name: {gpu_name}")


def main() -> None:
    """Print environment fields without exposing a full private path."""

    print("=== Research Environment Check ===")
    print(
        "Operating system: "
        f"{platform.system()} {platform.release()} "
        f"(build {platform.version()}, {platform.machine()})"
    )
    print(f"Python version: {platform.python_version()}")
    print(f"Working directory: {Path.cwd().name} (full path omitted)")
    print_system_gpu()
    print_pytorch()


if __name__ == "__main__":
    main()
