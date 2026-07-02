from pathlib import Path
import subprocess
import sys

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_project_script(script_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_dummy_tensor_operation_uses_available_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([1.0, 2.0, 3.0], device=device)

    result = tensor * 2

    assert result.device.type == device.type
    assert result.tolist() == [2.0, 4.0, 6.0]


def test_tiny_forward_pass_works_on_cpu_or_cuda() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = nn.Linear(4, 2).to(device)
    inputs = torch.ones((1, 4), device=device)

    with torch.inference_mode():
        outputs = layer(inputs)

    assert outputs.shape == (1, 2)
    assert outputs.device.type == device.type


def test_check_environment_script_reports_core_fields() -> None:
    result = run_project_script("check_environment.py")

    assert result.returncode == 0, result.stderr
    assert "Python version:" in result.stdout
    assert "PyTorch version:" in result.stdout
    assert "TorchVision version:" in result.stdout
    assert "CUDA available:" in result.stdout
    assert "Working directory:" in result.stdout


def test_smoke_test_script_completes_without_training() -> None:
    result = run_project_script("smoke_test.py")

    assert result.returncode == 0, result.stderr
    assert "Output shape: (2, 2)" in result.stdout
    assert "Smoke test passed." in result.stdout
