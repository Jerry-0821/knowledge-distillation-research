# Knowledge Distillation Research

This repository is a first paper-reproduction project for knowledge distillation, based on the core idea from Hinton, Vinyals, and Dean's *Distilling the Knowledge in a Neural Network*.

The goal is not to reproduce every experiment from the paper. Instead, the project builds small, controlled teacher-student experiments and tests whether a student CNN trained with teacher soft targets performs differently from the same student CNN trained only with hard labels.

## Current Status

MNIST Version 1 and Fashion-MNIST Version 2 are complete.

The project includes:

- MNIST dataset loading and training utilities.
- Fashion-MNIST dataset loading and V2 training scripts.
- A larger CNN teacher model.
- A smaller CNN student model.
- A hard-label student baseline.
- A knowledge distillation student training script.
- Controlled comparisons using fixed seeds.
- Result summaries and a final project write-up.

The strongest Version 1 result was exploratory: the KD student with temperature `6.0` and alpha `0.65` reached `0.9609` test accuracy on seed `0`, compared with the hard-label seed-0 baseline of `0.9547`. This should be described carefully because it was tested on one seed and the improvement is small.

## MNIST V1 Key Result Summary

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

## Fashion-MNIST V2 Key Result Summary

| Experiment | Seed | Temperature | Alpha | Test Accuracy | Notes |
|---|---:|---:|---:|---:|---|
| Teacher reference | 0 | N/A | N/A | 0.8760 | fixed teacher checkpoint |
| Hard-label student | 0 | N/A | N/A | 0.8597 | baseline |
| KD student | 0 | 2.0 | 0.7 | 0.8604 | first approved KD run |
| KD student exploratory | 0 | 7.0 | 0.7 | 0.8646 | best visible seed-0 exploratory run |
| Hard-label student | 1 | N/A | N/A | 0.8579 | reliability baseline |
| KD student | 1 | 7.0 | 0.7 | 0.8570 | did not repeat seed-0 improvement |

More details:

- [Fashion-MNIST V2 scope](docs/v2_fashion_mnist/FASHION_MNIST_V2_SCOPE.md)
- [Fashion-MNIST V2 results summary](docs/v2_fashion_mnist/FASHION_MNIST_V2_RESULTS_SUMMARY.md)
- [Fashion-MNIST V2 result summary CSV](results/tables/v2_fashion_mnist/fashion_mnist_v2_result_summary.csv)
- [Controlled experiment log](docs/CONTROLLED_EXPERIMENT_LOG.md)

## Main Findings

Knowledge distillation did not automatically improve every setting. On MNIST, the hard-label student already performed strongly, so the improvement from distillation was small. Temperature mattered: higher temperature settings produced better KD results in the tested runs.

The Version 1 conclusion is cautious: KD can help the student model under suitable settings, but this MNIST setup does not prove that KD always improves student performance.

The Fashion-MNIST V2 conclusion is also cautious. Seed 0 showed a small KD improvement over the hard-label student, especially with higher temperature, but seed 1 did not repeat the improvement. This suggests that temperature affects KD performance, but the tested V2 runs do not show enough evidence to claim that KD consistently beats hard-label training on Fashion-MNIST.

## Repository Structure

- `src/kd_research/`: reusable Python package code for data loading and model definitions.
- `scripts/`: environment checks, MNIST training scripts, and V2 Fashion-MNIST scripts.
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

The project was run on CPU for Version 1 and Version 2.

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

## Reproduce Fashion-MNIST V2

Check Fashion-MNIST data:

```powershell
.\.venv\Scripts\python.exe scripts\v2_fashion_mnist\check_fashion_mnist_data.py --download
```

Train the V2 teacher checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_teacher.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0 --checkpoint-path checkpoints\v2_fashion_mnist_teacher_seed0.pt --download
```

Run the hard-label student baseline:

```powershell
.\.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_student_baseline.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0
```

Run the KD student:

```powershell
.\.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_distillation.py --teacher-checkpoint checkpoints\v2_fashion_mnist_teacher_seed0.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 7.0 --alpha 0.7 --seed 0
```

## Validation

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## What Is Not Included

This repository does not claim to reproduce the full original paper.

This repository does not include:

- Speech recognition experiments.
- Specialist models.
- Mixture of Experts.
- Large-scale training.
- A full hyperparameter grid search.
- Committed datasets or model checkpoints.

Datasets and checkpoints are generated locally and are intentionally ignored by Git.

## Future Work

Version 3 can extend the same pipeline to CIFAR-10 as a stronger portfolio stretch. CIFAR-10 is more realistic, but it will require more model and training adjustments than Fashion-MNIST.
