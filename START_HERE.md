# Start Here

This file is a quick orientation guide for returning to the project.

## Current Project Stage

MNIST Version 1, Fashion-MNIST Version 2, and CIFAR-10 Version 3 are complete.

The project now contains a working teacher-student knowledge distillation pipeline:

1. Train a larger CNN teacher with hard labels.
2. Train a smaller CNN student with hard labels only.
3. Train the same student with knowledge distillation.
4. Compare the hard-label student and the distilled student under controlled settings.
5. Record results and write a cautious summary.

## Main Files to Read

Start with these files:

- `README.md`: public GitHub overview.
- `docs/MNIST_V1_PROJECT_SUMMARY_FINAL.md`: final MNIST V1 write-up.
- `docs/MNIST_V1_RESULTS_SUMMARY.md`: short result interpretation.
- `results/tables/mnist_v1_result_summary.csv`: result table.
- `docs/v2_fashion_mnist/FASHION_MNIST_V2_RESULTS_SUMMARY.md`: Fashion-MNIST V2 result interpretation.
- `results/tables/v2_fashion_mnist/fashion_mnist_v2_result_summary.csv`: V2 result table.
- `docs/v3_cifar10/CIFAR10_V3_RESULTS_SUMMARY.md`: CIFAR-10 V3 result interpretation.
- `results/tables/v3_cifar10/cifar10_v3_result_summary.csv`: V3 result table.
- `docs/CONTROLLED_EXPERIMENT_LOG.md`: detailed experiment record.

## Main Code Files

- `src/kd_research/data/mnist.py`: MNIST dataset and dataloader helpers.
- `src/kd_research/data/fashion_mnist.py`: Fashion-MNIST dataset and dataloader helpers.
- `src/kd_research/data/cifar10.py`: CIFAR-10 dataset, validation split, and dataloader helpers.
- `src/kd_research/models/simple_cnn_teacher_model.py`: teacher CNN.
- `src/kd_research/models/simple_cnn_student_model.py`: student CNN.
- `src/kd_research/models/cifar_cnn_models.py`: CIFAR-10 teacher and student CNN models.
- `scripts/train_mnist_teacher.py`: hard-label teacher training.
- `scripts/train_mnist_baseline.py`: hard-label student baseline.
- `scripts/train_mnist_distillation.py`: KD student training.
- `scripts/v2_fashion_mnist/`: Fashion-MNIST V2 data check and training scripts.
- `scripts/v3_cifar10/`: CIFAR-10 V3 data check and training scripts.

## Reproduce the Main Runs

Use the local virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts\train_mnist_teacher.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0 --checkpoint-path checkpoints\mnist_teacher_baseline_001.pt --download
```

```powershell
.\.venv\Scripts\python.exe scripts\train_mnist_baseline.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0
```

```powershell
.\.venv\Scripts\python.exe scripts\train_mnist_distillation.py --teacher-checkpoint checkpoints\mnist_teacher_baseline_001.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 4.0 --alpha 0.7 --seed 0
```

For CIFAR-10 V3, start with the V3 summary first because the full run plan is
longer:

```powershell
.\.venv\Scripts\python.exe scripts\v3_cifar10\check_cifar10_data.py --download
```

```powershell
.\.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 10 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --checkpoint-path checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt
```

```powershell
.\.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt --epochs 10 --batch-size 64 --learning-rate 0.001 --temperature 6.0 --alpha 0.5 --seed 0 --device auto
```

## Current Conclusion

Knowledge distillation did not automatically improve every setting. In this MNIST V1 setup, temperature mattered, and some KD settings produced a small improvement over the same student trained only with hard labels. The result should be described carefully because MNIST is simple and the improvement is small.

Fashion-MNIST V2 reached a similar cautious conclusion. Seed 0 showed a small KD improvement over the hard-label student, but seed 1 did not repeat the improvement. Temperature affected the tested KD accuracy, but the current V2 runs do not prove that KD consistently beats hard-label training on Fashion-MNIST.

CIFAR-10 V3 found that KD helped in the 10-epoch validation comparison across
seeds 0, 1, and 2. However, this early advantage did not persist when training
was extended to 20 and 70 epochs. The final official test evaluation on seed 0
also favored the hard-label student. The careful V3 conclusion is that KD may
help early training under this setup, but V3 does not prove that KD improves
final CIFAR-10 performance overall.

## Next Direction

The next direction is packaging and presentation: keep V1, V2, and V3 results
separate, publish the tracked summaries/tables/figures, and avoid committing
datasets, checkpoints, or raw local run logs unless there is a specific reason.
