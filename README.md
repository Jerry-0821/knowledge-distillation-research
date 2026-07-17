# Knowledge Distillation Research

This repository is a first paper-reproduction project for knowledge distillation, based on the core idea from Hinton, Vinyals, and Dean's *Distilling the Knowledge in a Neural Network*.

The goal of Version 1 is not to reproduce every experiment from the paper. Instead, it implements a small, controlled teacher-student pipeline and tests whether a student CNN trained with teacher soft targets performs differently from the same student CNN trained only with hard labels.

## Current Status

MNIST Version 1 is complete.

The project includes:

- MNIST dataset loading and training utilities.
- A larger CNN teacher model.
- A smaller CNN student model.
- A hard-label student baseline.
- A knowledge distillation student training script.
- Controlled comparisons using fixed seeds.
- Result summaries and a final project write-up.

The strongest Version 1 result was exploratory: the KD student with temperature `6.0` and alpha `0.65` reached `0.9609` test accuracy on seed `0`, compared with the hard-label seed-0 baseline of `0.9547`. This should be described carefully because it was tested on one seed and the improvement is small.

## Key Result Summary

| Experiment | Seed | Temperature | Alpha | Test Accuracy | Notes |
|---|---:|---:|---:|---:|---|
| Hard-label student | 0 | N/A | N/A | 0.9547 | baseline |
| KD student | 0 | 2.0 | 0.7 | 0.9537 | did not improve |
| KD student | 0 | 4.0 | 0.7 | 0.9599 | improved over baseline |
| Hard-label student | 1 | N/A | N/A | 0.9517 | baseline |
| KD student | 1 | 4.0 | 0.7 | 0.9526 | small improvement |
| KD student exploratory | 0 | 6.0 | 0.65 | 0.9609 | best observed seed-0 result |

More details:

- [Final MNIST V1 project summary](docs/MNIST_V1_PROJECT_SUMMARY_FINAL.md)
- [MNIST V1 results summary](docs/MNIST_V1_RESULTS_SUMMARY.md)
- [Result summary CSV](results/tables/mnist_v1_result_summary.csv)
- [Controlled experiment log](docs/CONTROLLED_EXPERIMENT_LOG.md)

## Main Finding

Knowledge distillation did not automatically improve every setting. On MNIST, the hard-label student already performed strongly, so the improvement from distillation was small. Temperature mattered: higher temperature settings produced better KD results in the tested runs.

The Version 1 conclusion is cautious: KD can help the student model under suitable settings, but this MNIST setup does not prove that KD always improves student performance.

## Repository Structure

- `src/kd_research/`: reusable Python package code for data loading and model definitions.
- `scripts/`: environment checks, MNIST data checks, and training scripts.
- `docs/`: paper notes, reproduction scope, experiment logs, and final summaries.
- `results/tables/`: small tracked result tables.
- `results/raw/`: local raw experiment records, ignored by Git except `.gitkeep`.
- `data/`: local datasets, ignored by Git except documentation.
- `checkpoints/`: local model checkpoints, ignored by Git except `.gitkeep`.
- `notebooks/`: learning walkthrough notebooks.

## Setup

Create the local environment and install the project:

```powershell
Set-Location knowledge-distillation-research
.\scripts\setup.ps1 -InstallTorchCpu
.\.venv\Scripts\Activate.ps1
```

The project was run on CPU for Version 1.

## Reproduce MNIST V1

Download/check MNIST data:

```powershell
.\.venv\Scripts\python.exe scripts\check_mnist_data.py --download
```

Train the teacher checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\train_mnist_teacher.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0 --checkpoint-path checkpoints\mnist_teacher_baseline_001.pt --download
```

Run the hard-label student baseline:

```powershell
.\.venv\Scripts\python.exe scripts\train_mnist_baseline.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0
```

Run the KD student:

```powershell
.\.venv\Scripts\python.exe scripts\train_mnist_distillation.py --teacher-checkpoint checkpoints\mnist_teacher_baseline_001.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 4.0 --alpha 0.7 --seed 0
```

Run the best observed exploratory setting:

```powershell
.\.venv\Scripts\python.exe scripts\train_mnist_distillation.py --teacher-checkpoint checkpoints\mnist_teacher_baseline_001.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 6.0 --alpha 0.65 --seed 0
```

## Validation

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## What Is Not Included

This repository does not claim to reproduce the full original paper.

Version 1 does not include:

- Speech recognition experiments.
- Specialist models.
- Mixture of Experts.
- Large-scale training.
- A full hyperparameter grid search.
- Committed datasets or model checkpoints.

Datasets and checkpoints are generated locally and are intentionally ignored by Git.

## Future Work

Version 2 will test Fashion-MNIST as a short bridge: it keeps the same image classification structure as MNIST but is harder, so the KD effect may be easier to observe.

Version 3 can extend the same pipeline to CIFAR-10 as a stronger portfolio stretch. CIFAR-10 is more realistic, but it will require more model and training adjustments than Fashion-MNIST.
