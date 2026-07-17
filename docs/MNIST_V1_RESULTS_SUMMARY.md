# MNIST V1 Results Summary

## Main Comparison

| Setting | Seed | Accuracy | Difference vs same-seed hard-label |
|---|---:|---:|---:|
| Hard-label student | 0 | 0.9547 | reference |
| KD, T=2.0, alpha=0.7 | 0 | 0.9537 | -0.0010 |
| KD, T=4.0, alpha=0.7 | 0 | 0.9599 | +0.0052 |
| Hard-label student | 1 | 0.9517 | reference |
| KD, T=4.0, alpha=0.7 | 1 | 0.9526 | +0.0009 |

## Exploratory Best Seed-0 Runs

| Setting | Seed | Accuracy | Difference vs hard-label seed 0 |
|---|---:|---:|---:|
| KD, T=6.0, alpha=0.65 | 0 | 0.9609 | +0.0062 |
| KD, T=6.0, alpha=0.6 | 0 | 0.9607 | +0.0060 |

## Careful Interpretation

KD did not automatically improve every setting. Temperature mattered: T=2.0 did not clearly beat hard-label training, while T=4.0 improved over hard-label in seeds 0 and 1. The best observed seed-0 result was T=6.0 and alpha=0.65, but that setting is exploratory and has not been repeated across more seeds.

## Future Work

Version 2 can test Fashion-MNIST to see whether KD helps more on a slightly harder dataset. After that, CIFAR-10 can be a stronger Version 3 stretch because it is closer to a real computer vision task but needs more model and training adjustments.
