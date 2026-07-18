# Fashion-MNIST V2 Results Summary

## Main Comparison

| Setting | Seed | Accuracy | Difference vs same-seed hard-label student |
|---|---:|---:|---:|
| Teacher reference | 0 | 0.8760 | not the student baseline |
| Hard-label student | 0 | 0.8597 | reference |
| KD, T=2.0, alpha=0.7 | 0 | 0.8604 | +0.0007 |
| KD, T=7.0, alpha=0.7 | 0 | 0.8646 | +0.0049 |
| Hard-label student | 1 | 0.8579 | reference |
| KD, T=7.0, alpha=0.7 | 1 | 0.8570 | -0.0009 |

## Exploratory Seed-0 Runs

| Setting | Seed | Accuracy | Difference vs hard-label student seed 0 |
|---|---:|---:|---:|
| KD, T=6.0, alpha=0.7 | 0 | 0.8642 | +0.0045 |
| KD, T=7.0, alpha=0.7 | 0 | 0.8646 | +0.0049 |

## Student Interpretation

The conclusion is similar to V1. We still cannot conclude that the distillation model can beat or is better than the hard-label model when using the same student model. In V2, temperature is still an important factor that affects the accuracy score and model performance. For seed 0, the hard-label student reached 0.8597, while the best observed KD run reached 0.8646. However, when the seed changed from 0 to 1, the KD student became slightly worse than the hard-label student: 0.8570 for KD compared with 0.8579 for hard-label training.

Therefore, the result is promising for seed 0 but not robust across seeds yet.

## Careful Interpretation

The Fashion-MNIST V2 results do not show a clear, stable KD improvement over hard-label student training. Temperature mattered in the tested runs, and higher-temperature exploratory settings improved the seed-0 KD result. However, the seed-1 check did not repeat the seed-0 improvement.

This means the V2 result should be reported cautiously: KD may help under some temperature settings, but this reproduction does not show enough evidence to claim that KD consistently beats hard-label training on Fashion-MNIST.

## Limitations

- The best seed-0 result came from exploratory temperature checking.
- The seed-1 reliability check did not repeat the seed-0 improvement.
- Only one training epoch was used in the recorded V2 runs.
- Extra alpha and temperature checks were reported by the student, but only the visible best result was recorded exactly.
- These results should not be used to claim that KD always improves Fashion-MNIST performance.

## Result Files

- `results/tables/v2_fashion_mnist/fashion_mnist_v2_result_summary.csv`
- `docs/CONTROLLED_EXPERIMENT_LOG.md`

Local raw CSV files were also written under `results/raw/`, but that folder is
ignored by Git so raw run records stay local unless explicitly published.
