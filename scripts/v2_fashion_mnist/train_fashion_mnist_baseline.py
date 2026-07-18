"""Compatibility wrapper for the Fashion-MNIST student baseline script.

The clearer V2 script name is:

    train_fashion_mnist_student_baseline.py

This wrapper is kept so earlier logged commands remain reproducible.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    script_path = Path(__file__).with_name("train_fashion_mnist_student_baseline.py")
    runpy.run_path(str(script_path), run_name="__main__")
