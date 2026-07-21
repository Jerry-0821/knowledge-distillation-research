# CIFAR-10 V3 Results Summary

## Main Question

V3 moved from MNIST/Fashion-MNIST to CIFAR-10 to test knowledge distillation
on a harder image dataset. CIFAR-10 uses RGB images with shape `3 x 32 x 32`,
so V3 required a new data loader, a CIFAR-10 CNN model, and a train/validation/test
experiment design.

The main comparison stayed the same as V1/V2:

- hard-label student baseline
- same student architecture trained with KD
- teacher checkpoint fixed during KD
- validation set used for model selection
- official test set used only at the final evaluation stage

## Selected V3 Setup

| Item | Choice |
|---|---|
| Dataset | CIFAR-10 |
| Train/validation/test split | 45,000 / 5,000 / 10,000 |
| Batch size | 64 |
| Teacher learning rate | 0.001 |
| Student learning rate | 0.001 |
| Selected KD temperature | 6.0 |
| Selected KD alpha | 0.5 |
| Teacher checkpoint | `checkpoints/v3_cifar10_teacher_lr0001_10epoch_metrics.pt` |

## Architecture / Compression Context

The CIFAR-10 teacher and student are both small CNNs, but the teacher has more
capacity than the student. These parameter counts are calculated directly from
the current model definitions.

| Model | Trainable Parameters |
|---|---:|
| CIFAR-10 teacher CNN | 289,194 |
| CIFAR-10 student CNN | 36,122 |

The teacher has about `8.01x` as many trainable parameters as the student. The
student has about `87.51%` fewer trainable parameters than the teacher.

This is architecture/compression context only. It does not prove that KD
succeeded; the actual hard-label versus KD comparison is reported below.

## Teacher Learning Rate Check

The teacher learning-rate check compared `0.001` and `0.0005` for 10 epochs.

| Teacher LR | Seed | Epochs | Train Accuracy | Validation Accuracy | Decision |
|---:|---:|---:|---:|---:|---|
| 0.001 | 0 | 10 | 0.8280 | 0.8104 | Selected |
| 0.0005 | 0 | 10 | 0.8093 | 0.8138 | Slightly higher validation accuracy |

The validation difference was only `0.0034` absolute accuracy, or 0.34
percentage points. The two learning rates were treated as approximately tied
for this small V3 check. LR `0.001` was retained as the default setting for
consistency across the teacher and student experiments, not because it was
clearly superior on validation accuracy.

## Main 10-Epoch Validation Comparison

At 10 epochs, the selected KD setting (`T=6.0`, `alpha=0.5`) improved over the
same-seed hard-label baseline across seeds 0, 1, and 2.

| Seed | Hard-label Student Validation Accuracy | KD Validation Accuracy | Difference |
|---:|---:|---:|---:|
| 0 | 0.6776 | 0.6952 | +0.0176 |
| 1 | 0.6952 | 0.7032 | +0.0080 |
| 2 | 0.6832 | 0.6998 | +0.0166 |

Mean validation accuracy:

| Setting | Mean Validation Accuracy |
|---|---:|
| Hard-label student | 0.6853 |
| KD student | 0.6994 |

Using sample standard deviation across seeds 0, 1, and 2:

| Setting | Mean Validation Accuracy | Sample SD |
|---|---:|---:|
| Hard-label student | 0.6853 | 0.0090 |
| KD student | 0.6994 | 0.0040 |
| Paired KD - hard-label difference | +0.0141 | 0.0053 |

This was the clearest positive V3 result. It suggests that KD can help early
training under this setup. With only three seeds, this should be treated as a
small robustness check, not a formal statistical significance test.

## 20-Epoch Validation Check

When the same selected KD setting was extended to 20 epochs, the advantage did
not persist.

| Seed | Hard-label Student Validation Accuracy | KD Validation Accuracy | Difference |
|---:|---:|---:|---:|
| 0 | 0.7400 | 0.7244 | -0.0156 |
| 1 | 0.7352 | 0.6836 | -0.0516 |
| 2 | 0.7408 | 0.7414 | +0.0006 |

Mean validation accuracy:

| Setting | Mean Validation Accuracy |
|---|---:|
| Hard-label student | 0.7387 |
| KD student | 0.7165 |

Using sample standard deviation across seeds 0, 1, and 2:

| Setting | Mean Validation Accuracy | Sample SD |
|---|---:|---:|
| Hard-label student | 0.7387 | 0.0030 |
| KD student | 0.7165 | 0.0297 |
| Paired KD - hard-label difference | -0.0222 | 0.0267 |

By 20 epochs, the hard-label baseline mostly caught up or surpassed the KD
student. Seed 2 was effectively a tie. The 20-epoch KD results were also much
less stable across seeds than the hard-label baseline, mainly because seed 1
was much weaker.

## 70-Epoch Validation Check

For the longer 70-epoch check, V3 used seed 0.

| Setting | Seed | Final Validation Accuracy | Best Validation Accuracy |
|---|---:|---:|---:|
| Hard-label student | 0 | 0.7912 | 0.7940 at epoch 69 |
| KD, T=6.0, alpha=0.5 | 0 | 0.7744 | 0.7780 at epoch 67 |

At 70 epochs, the hard-label student remained stronger than the selected KD
student on validation accuracy.

## Final Official Test Evaluation

The official CIFAR-10 test set was used only after the validation-based
experiments were complete. These results should not be used for further
hyperparameter tuning.

Important reproducibility note: these final-test commands retrained the 20-epoch
and 70-epoch models from scratch with `--evaluate-test`. They were not simple
test-set evaluations of the earlier validation-only checkpoints. Because the
current CUDA training setup is not strictly deterministic, the validation
accuracies in the final-test runs can differ slightly from the earlier
validation-only runs even when the seed and hyperparameters match.

| Epochs | Setting | Seed | Final Validation Accuracy | Official Test Accuracy | Difference vs Hard-label Test |
|---:|---|---:|---:|---:|---:|
| 20 | Hard-label student | 0 | 0.7422 | 0.7385 | reference |
| 20 | KD, T=6.0, alpha=0.5 | 0 | 0.7198 | 0.7144 | -0.0241 |
| 70 | Hard-label student | 0 | 0.7812 | 0.7860 | reference |
| 70 | KD, T=6.0, alpha=0.5 | 0 | 0.7656 | 0.7622 | -0.0238 |

The final test set is consistent with the longer-training validation story for
seed 0: the selected KD student did not beat the hard-label student at 20 or
70 epochs.

## Student Interpretation

V3 shows that KD can help in the early training stage. In the 10-epoch
validation comparison, KD with `T=6.0` and `alpha=0.5` was better than the
hard-label student across seeds 0, 1, and 2.

However, this improvement was not stable after longer training. At 20 epochs,
the hard-label baseline caught up or became better for most seeds. At 70
epochs on seed 0, the hard-label baseline was still stronger. The final
official test results also favored the hard-label baseline at both 20 and 70
epochs.

Therefore, V3 cannot claim that KD is better overall under this setup. A more
careful conclusion is that KD was promising for early training, but the chosen
teacher, student, temperature, alpha, and training setup did not produce a
strong final-performance improvement over hard-label training.

## Careful Interpretation

The negative longer-training result does not mean the project failed. It shows
that KD is sensitive to experimental design. Temperature, alpha, model
capacity, teacher strength, training length, and data preprocessing can all
change whether KD helps.

The V3 result is useful because it avoids overclaiming. It records both the
positive early result and the weaker longer-training result. This is closer to
a controlled reproduction than only reporting the best-looking number.

## Limitations

- The CIFAR-10 model is still a simple CNN, not a large modern architecture.
- V3 used `ToTensor()` only; normalization and data augmentation were left as
  future enhancements.
- The teacher was trained for only 10 epochs and may not be strong enough for
  robust distillation.
- The final official test evaluation was run for seed 0 only.
- The KD search was limited to a small set of temperature and alpha values.
- No learning-rate scheduler or longer teacher training was used.
- The pipeline improved during V3, so early smoke tests and later full runs
  should not be mixed as equal evidence.

## Result Files

- `results/tables/v3_cifar10/cifar10_v3_result_summary.csv`
- `docs/CONTROLLED_EXPERIMENT_LOG.md`
- `results/raw/v3_cifar10/`
- `results/figures/v3_cifar10/`

Local raw CSV files and run logs are kept under `results/raw/`. The raw folder
is ignored by Git unless explicitly changed.

## Figure Reproducibility Note

The public repository tracks the V3 PNG figures under
`results/figures/v3_cifar10/` and the compact summary tables under
`results/tables/v3_cifar10/`.

However, many V3 figures were generated from local per-epoch CSV files and run
logs under `results/raw/v3_cifar10/`, which are intentionally ignored by Git.
No public plotting script is currently tracked. This means the figures document
the observed results, but not every figure can be regenerated from only the
public tracked files.

A clean future reproducibility improvement would be to publish minimal
non-sensitive per-epoch CSVs under a tracked folder such as
`results/records/v3_cifar10/` and add a plotting script. Datasets, checkpoints,
and large raw logs should stay untracked unless there is a specific reason to
publish them.
