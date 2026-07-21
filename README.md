# Knowledge Distillation Research

This repository is a first paper-reproduction project for knowledge distillation, based on the core idea from Hinton, Vinyals, and Dean's *Distilling the Knowledge in a Neural Network*.

The goal is not to reproduce every experiment from the paper. Instead, the project builds small, controlled teacher-student experiments and tests whether a student CNN trained with teacher soft targets performs differently from the same student CNN trained only with hard labels.

## Current Status

MNIST Version 1, Fashion-MNIST Version 2, and CIFAR-10 Version 3 are complete.

The project includes:

- MNIST dataset loading and training utilities.
- Fashion-MNIST dataset loading and V2 training scripts.
- CIFAR-10 dataset loading, validation split support, and V3 training scripts.
- A larger CNN teacher model.
- A smaller CNN student model.
- CIFAR-10 CNN teacher and student models.
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

## CIFAR-10 V3 Key Result Summary

CIFAR-10 V3 tested the same teacher-student KD idea on a harder RGB image
dataset. Unlike V2, this required a new CIFAR-10 data loader, a train/validation/test
split, and CIFAR-10-specific CNN models.

The selected KD setting was `T=6.0`, `alpha=0.5`, batch size `64`, and learning
rate `0.001`.

The CIFAR-10 teacher has `289,194` trainable parameters, while the student has
`36,122`. This gives about an `8.01x` teacher/student parameter ratio, or an
`87.51%` reduction from teacher to student. This is architecture context only,
not proof that KD succeeded.

| Stage | Seed(s) | Hard-label Student | KD Student | Difference | Notes |
|---|---:|---:|---:|---:|---|
| 10-epoch validation mean | 0, 1, 2 | 0.6853 | 0.6994 | +0.0141 | KD helped early training |
| 20-epoch validation mean | 0, 1, 2 | 0.7387 | 0.7165 | -0.0222 | KD advantage did not persist |
| 70-epoch validation | 0 | 0.7912 | 0.7744 | -0.0168 | hard-label baseline stronger |
| 20-epoch final test | 0 | 0.7385 | 0.7144 | -0.0241 | official test set, no further tuning |
| 70-epoch final test | 0 | 0.7860 | 0.7622 | -0.0238 | official test set, no further tuning |

For the 10-epoch and 20-epoch validation means, sample standard deviation over
seeds 0, 1, and 2 was:

| Stage | Hard-label SD | KD SD | Paired Difference SD |
|---|---:|---:|---:|
| 10-epoch validation | 0.0090 | 0.0040 | 0.0053 |
| 20-epoch validation | 0.0030 | 0.0297 | 0.0267 |

The 20-epoch KD result was less stable across seeds, especially because seed 1
was much weaker. No statistical significance test is claimed from only three
seeds.

More details:

- [CIFAR-10 V3 scope](docs/v3_cifar10/CIFAR10_V3_SCOPE.md)
- [CIFAR-10 V3 results summary](docs/v3_cifar10/CIFAR10_V3_RESULTS_SUMMARY.md)
- [CIFAR-10 V3 result summary CSV](results/tables/v3_cifar10/cifar10_v3_result_summary.csv)
- [Controlled experiment log](docs/CONTROLLED_EXPERIMENT_LOG.md)
- V3 figures are stored under `results/figures/v3_cifar10/`.

## Main Findings

Knowledge distillation did not automatically improve every setting. On MNIST, the hard-label student already performed strongly, so the improvement from distillation was small. Temperature mattered: higher temperature settings produced better KD results in the tested runs.

The Version 1 conclusion is cautious: KD can help the student model under suitable settings, but this MNIST setup does not prove that KD always improves student performance.

The Fashion-MNIST V2 conclusion is also cautious. Seed 0 showed a small KD improvement over the hard-label student, especially with higher temperature, but seed 1 did not repeat the improvement. This suggests that temperature affects KD performance, but the tested V2 runs do not show enough evidence to claim that KD consistently beats hard-label training on Fashion-MNIST.

CIFAR-10 V3 showed a clearer early-training KD improvement at 10 epochs across
seeds 0, 1, and 2. However, the advantage did not persist when training was
extended to 20 and 70 epochs. The final official test evaluation on seed 0 also
favored the hard-label student. V3 therefore suggests that KD may help early
training under this setup, but it does not support a strong claim that KD
improves final CIFAR-10 performance overall.

## Repository Structure

- `src/kd_research/`: reusable Python package code for data loading and model definitions.
- `scripts/`: environment checks, MNIST scripts, V2 Fashion-MNIST scripts, and V3 CIFAR-10 scripts.
- `docs/`: paper notes, reproduction scope, experiment logs, and final summaries.
- `results/tables/`: small tracked result tables.
- `results/figures/`: tracked figures for experiment visualization.
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

The CPU setup is sufficient to run the project, but CIFAR-10 training will be
slower on CPU. Version 3 experiments were run locally with a CUDA-enabled
PyTorch environment because CIFAR-10 training is larger.

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

## Reproduce CIFAR-10 V3

Check CIFAR-10 data:

```powershell
.\.venv\Scripts\python.exe scripts\v3_cifar10\check_cifar10_data.py --download
```

Train the V3 CIFAR-10 teacher checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 10 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --checkpoint-path checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt
```

Run the hard-label student baseline:

```powershell
.\.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_student_baseline.py --epochs 10 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto
```

Run the selected KD student setting:

```powershell
.\.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt --epochs 10 --batch-size 64 --learning-rate 0.001 --temperature 6.0 --alpha 0.5 --seed 0 --device auto
```

The longer 20-epoch and 70-epoch final evaluations are recorded in
`docs/v3_cifar10/CIFAR10_V3_RESULTS_SUMMARY.md`.

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
- A claim that KD always beats hard-label training.

Datasets and checkpoints are generated locally and are intentionally ignored by Git.

## Future Work

- Add CIFAR-10 normalization and data augmentation.
- Train a stronger teacher model.
- Try longer teacher training and learning-rate scheduling.
- Repeat final CIFAR-10 test evaluation across more seeds.
- Explore a wider but still controlled KD hyperparameter search.
