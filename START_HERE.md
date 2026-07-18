# Start Here

This file is a quick orientation guide for returning to the project.

## Current Project Stage

MNIST Version 1 and Fashion-MNIST Version 2 are complete.

The project now contains a working teacher-student knowledge distillation pipeline:

1. Train a larger CNN teacher on MNIST with hard labels.
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
- `docs/CONTROLLED_EXPERIMENT_LOG.md`: detailed experiment record.

## Main Code Files

- `src/kd_research/data/mnist.py`: MNIST dataset and dataloader helpers.
- `src/kd_research/data/fashion_mnist.py`: Fashion-MNIST dataset and dataloader helpers.
- `src/kd_research/models/simple_cnn_teacher_model.py`: teacher CNN.
- `src/kd_research/models/simple_cnn_student_model.py`: student CNN.
- `scripts/train_mnist_teacher.py`: hard-label teacher training.
- `scripts/train_mnist_baseline.py`: hard-label student baseline.
- `scripts/train_mnist_distillation.py`: KD student training.
- `scripts/v2_fashion_mnist/`: Fashion-MNIST V2 data check and training scripts.

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

## Current Conclusion

Knowledge distillation did not automatically improve every setting. In this MNIST V1 setup, temperature mattered, and some KD settings produced a small improvement over the same student trained only with hard labels. The result should be described carefully because MNIST is simple and the improvement is small.

Fashion-MNIST V2 reached a similar cautious conclusion. Seed 0 showed a small KD improvement over the hard-label student, but seed 1 did not repeat the improvement. Temperature affected the tested KD accuracy, but the current V2 runs do not prove that KD consistently beats hard-label training on Fashion-MNIST.

## Next Direction

Version 3 can test CIFAR-10 as a stronger portfolio stretch.
