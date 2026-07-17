# Controlled Experiment Log

This file has two jobs:

1. Record the experiment blueprint: what we are trying to do and why.
2. Record the actual data: what command was run and what result came out.

## Part 0: Recording Rules

- Every run with numbers must be recorded here.
- Real experiment and controlled comparison runs also need a CSV file in `results/raw/`.
- Smoke-test accuracy is not a real result, but the smoke test still needs to be recorded.
- Bad or confusing results must stay in the log.
- If we fix code and that fix changes experiment fairness, update the blueprint immediately.

## Part 1: Blueprint Steps

### Step 0: Prepare fixed teacher

Train the teacher with hard labels and save a checkpoint.

Purpose: KD needs a fixed trained teacher. During student distillation, the teacher should not keep changing.

### Step 1: KD smoke test

Run the KD script with:

```text
--max-train-batches 1 --max-eval-batches 1
```

Purpose: Only check that the script can run: teacher loads, loss computes, backward works, and student updates.

Important: Accuracy from this step is not meaningful.

### Step 2: First full KD run

Remove the smoke-test limits:

```text
--max-train-batches 1 --max-eval-batches 1
```

Purpose: Train the distilled student for one full epoch and evaluate on the full test set.

### Step 3: Same-seed controlled comparison

Run both hard-label student and distilled student with the same seed:

```text
--seed 0
```

Purpose: Make the comparison fairer by controlling random initialization and data shuffle.

### Step 4: Fairness fix after review

Problem found: the KD script loaded the teacher before creating the student. Teacher initialization may consume random numbers, so the KD student might not start from the same initial weights as the hard-label student, even with `--seed 0`.

Fix made in:

```text
scripts/train_mnist_distillation.py
```

What changed:

```text
student is created first
teacher loading is isolated from the random number generator
```

Purpose: Make the corrected same-seed comparison cleaner.

### Step 5: Corrected same-seed comparison

Rerun hard-label student and distilled student with:

```text
--seed 0
```

Purpose: Compare hard-label vs KD after the RNG fairness fix.

### Future Step: Limited hyperparameter checks

Possible variables:

```text
alpha
temperature
epochs
seed
```

Purpose: Test planned settings only. Do not randomly search until the best test accuracy appears.

### Step 6: Corrected alpha 0.9 check

Rerun the alpha 0.9 KD setting after the RNG fairness fix:

```text
--alpha 0.9 --temperature 2.0 --seed 0
```

Purpose: The old alpha 0.9 run had the RNG fairness caveat. This corrected run checks whether giving more weight to hard labels still helps when the comparison is cleaner.

## Part 2: Data Log

### Quick Result Table

| Run | Type | Key setting | Test accuracy | Meaning | Raw file |
|---|---|---:|---:|---|---|
| Student baseline 001 | Reference | hard labels, no seed recorded | 0.9590 | Early hard-label student baseline | `results/raw/mnist_student_baseline_001.csv` |
| Teacher baseline 001 | Reference | hard labels, no checkpoint | 0.9806 | Earlier teacher run, not the saved teacher checkpoint | `results/raw/mnist_teacher_baseline_001.csv` |
| Teacher checkpoint 001 | Teacher prep | saved checkpoint | 0.9788 | Fixed teacher used for KD | `results/raw/mnist_teacher_checkpoint_001.csv` |
| KD smoke test 1 | Smoke test | one train batch, one eval batch | 0.0469 | Script ran; accuracy not meaningful | not saved as CSV |
| KD smoke test 2 | Smoke test | one train batch, one eval batch | 0.1094 | Script ran; accuracy not meaningful | not saved as CSV |
| KD full run 001 | Full KD run | alpha 0.7, temperature 2.0, no seed recorded | 0.9594 | Similar to baseline, tiny increase over 0.9590 | `results/raw/mnist_distillation_001.csv` |
| Seed 0 pair, before fix | Controlled pair with caveat | alpha 0.7, temperature 2.0, seed 0 | KD 0.9358 vs hard 0.9547 | KD worse, but later found RNG fairness caveat | `results/raw/mnist_seed0_controlled_pair.csv` |
| Alpha 0.9, before fix | Limited alpha check with caveat | alpha 0.9, temperature 2.0, seed 0 | 0.9404 | Better than alpha 0.7 before fix, still below hard-label; has RNG caveat | `results/raw/mnist_distillation_alpha09_seed0.csv` |
| Corrected seed 0 pair | Corrected controlled pair | alpha 0.7, temperature 2.0, seed 0 | KD 0.9537 vs hard 0.9547 | After RNG fix, KD is almost same as hard-label but slightly lower | `results/raw/mnist_seed0_corrected_pair.csv` |
| Corrected alpha 0.9 | Corrected limited alpha check | alpha 0.9, temperature 2.0, seed 0 | 0.9535 | Similar to corrected alpha 0.7, still slightly below hard-label | `results/raw/mnist_distillation_alpha09_seed0_corrected.csv` |
| Corrected temperature 4.0 | Corrected limited temperature check | alpha 0.7, temperature 4.0, seed 0 | 0.9599 | First corrected seed-0 KD run above hard-label seed 0; promising but only one seed | `results/raw/mnist_distillation_temp40_alpha07_seed0_corrected.csv` |
| Temperature 4.0 seed 1 pair | Reliability check | alpha 0.7, temperature 4.0, seed 1 | KD 0.9526 vs hard 0.9517 | KD is slightly higher for seed 1 | `results/raw/mnist_seed1_temp40_controlled_pair.csv` |
| Exploratory temperature 6.0 | Exploratory run | alpha 0.65, temperature 6.0, seed 0 | 0.9609 | Best observed seed-0 KD accuracy so far; not yet replicated | `results/raw/mnist_exploratory_temp6_seed0.csv` |
| Exploratory temperature 6.0 | Exploratory run | alpha 0.6, temperature 6.0, seed 0 | 0.9607 | Similar to alpha 0.65, also above hard-label seed 0 | `results/raw/mnist_exploratory_temp6_seed0.csv` |

### Details: Fixed Teacher Checkpoint

Command:

```text
.venv\Scripts\python.exe scripts\train_mnist_teacher.py --epochs 1 --batch-size 64 --learning-rate 0.001 --checkpoint-path checkpoints\mnist_teacher_baseline_001.pt
```

Result:

```text
train loss = 0.1828
test accuracy = 0.9788
checkpoint = checkpoints\mnist_teacher_baseline_001.pt
```

Meaning: This saved teacher is the teacher used by later KD runs.

### Details: Smoke Tests

Smoke-test command included:

```text
--max-train-batches 1 --max-eval-batches 1
```

Observed accuracies:

```text
0.0469
0.1094
```

Meaning: These low accuracies are expected because only one train batch and one eval batch were used. These runs only checked whether the code path worked.

### Details: First Full KD Run

Command:

```text
.venv\Scripts\python.exe scripts\train_mnist_distillation.py --teacher-checkpoint checkpoints\mnist_teacher_baseline_001.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 2.0 --alpha 0.7
```

Result:

```text
hard loss = 0.3099
soft loss = 0.8477
combined loss = 0.4713
test accuracy = 0.9594
```

Meaning: First full KD run was close to the early hard-label baseline `0.9590`.

### Details: Same-Seed Pair Before RNG Fix

Hard-label result:

```text
train loss = 0.3163
test accuracy = 0.9547
```

KD result:

```text
hard loss = 0.3599
soft loss = 0.9897
combined loss = 0.5489
test accuracy = 0.9358
```

Meaning: KD looked much worse, but later we found a fairness issue: teacher initialization may have changed the RNG sequence before the KD student was created. Keep this result, but do not treat it as the final fair comparison.

### Details: Alpha 0.9 Before RNG Fix

Command changed:

```text
alpha 0.7 -> alpha 0.9
```

Result:

```text
hard loss = 0.3438
soft loss = 1.0822
combined loss = 0.4176
test accuracy = 0.9404
```

Meaning: Alpha 0.9 was better than the previous alpha 0.7 KD run before the fix, but still worse than hard-label. This run also has the RNG fairness caveat.

### Details: Corrected Same-Seed Pair After RNG Fix

Hard-label result:

```text
train loss = 0.3163
test accuracy = 0.9547
```

KD result:

```text
hard loss = 0.3208
soft loss = 0.9129
combined loss = 0.4984
test accuracy = 0.9537
```

Comparison:

```text
0.9537 - 0.9547 = -0.0010
```

Meaning: After the RNG fairness fix, KD and hard-label training were almost the same under seed 0. KD was slightly lower by 0.10 percentage points.

### Details: Corrected Alpha 0.9 After RNG Fix

Command changed:

```text
alpha 0.7 -> alpha 0.9
```

Result:

```text
hard loss = 0.3166
soft loss = 1.0332
combined loss = 0.3883
test accuracy = 0.9535
```

Comparison:

```text
corrected alpha 0.9 KD - corrected alpha 0.7 KD = 0.9535 - 0.9537 = -0.0002
corrected alpha 0.9 KD - hard-label seed 0 = 0.9535 - 0.9547 = -0.0012
```

Meaning: After the RNG fairness fix, alpha 0.9 did not improve over alpha 0.7. Both KD settings were very close to the hard-label student but slightly lower.

### Details: Corrected Temperature 4.0 Check After RNG Fix

Command changed:

```text
temperature 2.0 -> temperature 4.0
alpha stays 0.7
seed stays 0
```

Result:

```text
hard loss = 0.3312
soft loss = 2.2702
combined loss = 0.9129
test accuracy = 0.9599
```

Comparison:

```text
hard-label seed 0: 0.9547
KD alpha 0.7 temperature 2.0 seed 0: 0.9537
KD alpha 0.7 temperature 4.0 seed 0: 0.9599
```

Meaning: With seed 0, increasing temperature from 2.0 to 4.0 improved the corrected KD result and made it higher than the hard-label seed-0 baseline. This is promising, but it is only one seed and should not be treated as final proof yet.

### Details: Temperature 4.0 Seed 1 Pair

Command changed:

```text
seed 0 -> seed 1
temperature stays 4.0
alpha stays 0.7
```

Result:

```text
hard-label train loss = 0.3340
hard-label test accuracy = 0.9517

hard loss = 0.3464
soft loss = 2.3503
combined loss = 0.9476
KD test accuracy = 0.9526
```

Comparison:

```text
0.9526 - 0.9517 = +0.0009
```

Meaning: Under seed 1, the temperature 4.0 KD setting was slightly higher than the matching hard-label baseline. The improvement is small, but it goes in the same direction as seed 0.

### Details: Exploratory Temperature 6.0 Seed 0 Runs

These runs were exploratory follow-ups after temperature 4.0 looked promising.

Results:

```text
temperature = 6.0, alpha = 0.65, seed = 0
hard loss = 0.3303
soft loss = 2.8282
combined loss = 1.2045
test accuracy = 0.9609

temperature = 6.0, alpha = 0.6, seed = 0
hard loss = 0.3330
soft loss = 2.7853
combined loss = 1.3139
test accuracy = 0.9607
```

Comparison:

```text
hard-label seed 0: 0.9547
KD temperature 4.0 alpha 0.7 seed 0: 0.9599
KD temperature 6.0 alpha 0.65 seed 0: 0.9609
KD temperature 6.0 alpha 0.6 seed 0: 0.9607
```

Meaning: Temperature 6.0 produced the best observed seed-0 KD result so far. Because these were exploratory runs on seed 0, they should be reported as exploratory unless repeated on another seed.

## Part 3: Current Careful Summary

The corrected temperature 4.0 setting outperformed the matching hard-label baseline for seed 0 and seed 1, but the seed 1 improvement was very small. Exploratory temperature 6.0 runs gave the best seed-0 KD results so far, but they have not been replicated on another seed.

Best careful statement:

```text
In the corrected runs so far, KD with temperature 4.0 and alpha 0.7 outperformed the matching hard-label baseline for seed 0 and seed 1. Exploratory temperature 6.0 settings improved the seed-0 result further, with the best observed run reaching 0.9609. The result is promising but should be described cautiously because the best exploratory setting has not been repeated across seeds.
```

Do not claim KD is generally better yet.

## Part 4: Remaining Blueprint

### Finish Version 1

Recommended next action:

```text
Stop running new experiments and write the MNIST Version 1 analysis.
```

Purpose: Avoid endless tuning and turn the current evidence into a clear reproduction story.

### Analysis step

Make a small comparison table:

```text
hard-label seed 0
KD alpha 0.7 seed 0
KD alpha 0.9 seed 0
KD temperature 4.0 seed 0
KD temperature 4.0 seed 1
hard-label seed 1
exploratory KD temperature 6.0 seed 0
```

Purpose: Summarize what has actually been observed so far.
