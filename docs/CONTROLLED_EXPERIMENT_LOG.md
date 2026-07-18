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

## Part 5: Fashion-MNIST V2 Blueprint

### V2 Step 0: Scope approval

Create and approve the Fashion-MNIST V2 scope before implementation.

Purpose: Keep Version 2 focused on the research question:

```text
Does knowledge distillation show a clearer improvement over hard-label training when the dataset is more difficult than MNIST?
```

### V2 Step 1: Fashion-MNIST data loader and smoke test

Add a Fashion-MNIST data loader without changing the MNIST V1 loader.

Purpose: Check that Fashion-MNIST can be loaded as grayscale `1 x 28 x 28` images with 10 classes before any model training begins.

Important: This is only a data pipeline smoke test. It does not train a model, evaluate accuracy, or say anything about KD performance.

### V2 Step 2: Training script one-batch smoke tests

Add Fashion-MNIST V2 training scripts for:

```text
teacher hard-label training
student hard-label baseline
student KD training
```

Purpose: Check that the three training code paths run without crashing before full training begins.

Important: These smoke tests use one train batch and one eval batch only. The numbers are not meaningful performance results and should not be used to claim that any model or method is better.

Student review boundary: Before full training, the student should review the scripts and confirm the experiment design, including comparison, seed, temperature, alpha, epochs, teacher checkpoint, and success criteria.

### V2 Step 3: First approved seed-0 controlled run

Run the first real Fashion-MNIST V2 pass with:

```text
seed = 0
epochs = 1
batch size = 64
learning rate = 0.001
temperature = 2.0
alpha = 0.7
```

Purpose: Train a real teacher checkpoint first, then compare the same student architecture trained with hard labels against the same student architecture trained with KD.

Success criterion: KD should be compared against the hard-label student baseline with the same seed. A better KD result is only preliminary until repeated with another seed; a weaker KD result must still be recorded.

## Part 6: Fashion-MNIST V2 Data Log

### Quick Result Table

| Run | Type | Key setting | Output numbers | Meaning | Raw file |
|---|---|---|---|---|---|
| Fashion-MNIST data smoke test 001 | Data smoke test | batch size 8, download enabled | train size 60000; test size 10000; image batch shape `(8, 1, 28, 28)`; label batch shape `(8,)`; image range `[0.000, 1.000]` | Fashion-MNIST downloaded locally and the loader produced the expected grayscale image tensor shape. No training was run. | not saved as CSV |
| Fashion-MNIST data smoke test 002 | Data smoke test | batch size 8, existing local data | train size 60000; test size 10000; image batch shape `(8, 1, 28, 28)`; label batch shape `(8,)`; image range `[0.000, 1.000]` | Student reran the Fashion-MNIST data check after the notebook/test walkthrough. The loader produced the expected grayscale image tensor shape. No training was run. | not saved as CSV |
| Fashion-MNIST teacher smoke test 001 | One-batch smoke test | teacher, seed 0, 1 train batch, 1 eval batch | train loss 2.2878; test accuracy 0.2500 | Teacher script ran and saved a smoke-test checkpoint. Accuracy is not meaningful because this was one batch only. | not saved as CSV |
| Fashion-MNIST teacher smoke test 002 | One-batch smoke test | student rerun of teacher smoke, seed 0, 1 train batch, 1 eval batch | train loss 2.2878; test accuracy 0.2500 | Student reran the teacher smoke test from terminal. The script ran and saved the smoke-test checkpoint again. Accuracy is not meaningful because this was one batch only. | not saved as CSV |
| Fashion-MNIST baseline smoke test 001 | One-batch smoke test | student hard-label baseline, seed 0, 1 train batch, 1 eval batch | train loss 2.3172; test accuracy 0.1562 | Student baseline script ran. Accuracy is not meaningful because this was one batch only. | not saved as CSV |
| Fashion-MNIST baseline smoke test 002 | One-batch smoke test | student rerun of hard-label baseline, seed 0, 1 train batch, 1 eval batch | train loss 2.3172; test accuracy 0.1562 | Student reran the baseline smoke test from terminal. Accuracy is not meaningful because this was one train batch and one eval batch only. | not saved as CSV |
| Fashion-MNIST KD smoke test 001 | One-batch smoke test | KD student, smoke teacher checkpoint, seed 0, temperature 2.0, alpha 0.7, 1 train batch, 1 eval batch | hard loss 2.3172; soft loss 0.4226; combined loss 1.7489; test accuracy 0.1562 | KD script loaded the fixed smoke teacher and trained only the student. Accuracy is not meaningful because this was one batch only. | not saved as CSV |
| Fashion-MNIST KD smoke test 002 | One-batch smoke test | student rerun of KD smoke, smoke teacher checkpoint, seed 0, temperature 2.0, alpha 0.7, 1 train batch, 1 eval batch | hard loss 2.3172; soft loss 0.4226; combined loss 1.7489; test accuracy 0.1562 | Student reran the KD smoke test from terminal. The script loaded the fixed smoke teacher and trained only the student. Accuracy is not meaningful because this was one batch only. | not saved as CSV |
| Fashion-MNIST teacher seed 0 | Real teacher run | teacher, seed 0, 1 full epoch, batch size 64, learning rate 0.001 | train loss 0.4167; test accuracy 0.8760 | First real V2 teacher checkpoint for seed-0 comparison. This is a reference teacher run, not a KD result. | `results/raw/fashion_mnist_v2_teacher_seed0.csv` |
| Fashion-MNIST student baseline seed 0 | Real hard-label student baseline | student, seed 0, 1 full epoch, batch size 64, learning rate 0.001 | train loss 0.4762; test accuracy 0.8597 | First real V2 hard-label student baseline. KD seed 0 should be compared against this same student setup. | `results/raw/fashion_mnist_v2_seed0_controlled_pair.csv` |
| Fashion-MNIST KD temp 2.0 alpha 0.7 seed 0 | Real KD run | student, fixed teacher seed 0, temperature 2.0, alpha 0.7, 1 full epoch | hard loss 0.4580; soft loss 0.3830; combined loss 0.4355; test accuracy 0.8604 | First approved V2 KD run. Compare against hard-label student baseline 0.8597, not teacher 0.8760. | `results/raw/fashion_mnist_v2_seed0_controlled_pair.csv` |
| Fashion-MNIST KD temp 6.0 alpha 0.7 seed 0 | Exploratory KD run | student, fixed teacher seed 0, temperature 6.0, alpha 0.7, 1 full epoch | hard loss 0.4530; soft loss 0.7430; combined loss 0.5400; test accuracy 0.8642 | Unplanned exploratory temperature check. Compare against hard-label student baseline 0.8597, not teacher 0.8760. Needs repeat before stronger claim. | `results/raw/fashion_mnist_v2_seed0_controlled_pair.csv` |
| Fashion-MNIST KD temp 7.0 alpha 0.7 seed 0 | Exploratory KD run | student, fixed teacher seed 0, temperature 7.0, alpha 0.7, 1 full epoch | hard loss 0.4529; soft loss 0.7524; combined loss 0.5428; test accuracy 0.8646 | Best visible exploratory V2 seed-0 KD result so far. Compare against hard-label student baseline 0.8597, not teacher 0.8760. Needs repeat before stronger claim. | `results/raw/fashion_mnist_v2_seed0_controlled_pair.csv` |
| Fashion-MNIST student baseline seed 1 | Real hard-label student baseline | student, seed 1, 1 full epoch, batch size 64, learning rate 0.001 | train loss 0.4969; test accuracy 0.8579 | Seed-1 hard-label student baseline for reliability check. | `results/raw/fashion_mnist_v2_seed1_temp7_alpha07_pair.csv` |
| Fashion-MNIST KD temp 7.0 alpha 0.7 seed 1 | Reliability check | student, fixed teacher seed 0, seed 1, temperature 7.0, alpha 0.7, 1 full epoch | hard loss 0.4785; soft loss 0.8910; combined loss 0.6022; test accuracy 0.8570 | Seed-1 check for the best visible seed-0 exploratory setting. KD was slightly below the matching hard-label student baseline 0.8579. | `results/raw/fashion_mnist_v2_seed1_temp7_alpha07_pair.csv` |

### Details: Fashion-MNIST Data Smoke Test 001

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\check_fashion_mnist_data.py --download
```

Result:

```text
train dataset size = 60000
test dataset size = 10000
image batch shape = (8, 1, 28, 28)
label batch shape = (8,)
image dtype = torch.float32
label dtype = torch.int64
image value range = [0.000, 1.000]
sample labels = [4, 2, 4, 0, 2, 5, 2, 7]
```

Meaning: The Fashion-MNIST data pipeline works for the first V2 data check. The image shape matches the MNIST model input shape, so the V1 simple CNN teacher and student can be reused for the first Fashion-MNIST pass. These numbers are not model performance results.

### Details: Fashion-MNIST Data Smoke Test 002

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\check_fashion_mnist_data.py
```

Result:

```text
train dataset size = 60000
test dataset size = 10000
image batch shape = (8, 1, 28, 28)
label batch shape = (8,)
image dtype = torch.float32
label dtype = torch.int64
image value range = [0.000, 1.000]
sample labels = [7, 6, 6, 5, 1, 2, 9, 0]
```

Meaning: The student reran the Fashion-MNIST data smoke test using the existing local dataset. The data loader still produced the expected dataset sizes, tensor shape, dtype, and value range. No training was run, so these numbers are not model performance results.

### Details: Fashion-MNIST Teacher Smoke Test 001

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_teacher.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0 --max-train-batches 1 --max-eval-batches 1 --checkpoint-path checkpoints\v2_fashion_mnist_teacher_smoke.pt
```

Result:

```text
train loss = 2.2878
test accuracy = 0.2500
checkpoint = checkpoints\v2_fashion_mnist_teacher_smoke.pt
```

Meaning: The teacher training script can run one train batch, evaluate one batch, and save a checkpoint. This checkpoint is only for KD smoke testing and should not be used as the real V2 teacher.

### Details: Fashion-MNIST Teacher Smoke Test 002

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_teacher.py --epochs 1 --batch-size 64 --seed 0 --max-train-batches 1 --max-eval-batches 1 --checkpoint-path checkpoints\v2_fashion_mnist_teacher_smoke.pt
```

Result:

```text
train loss = 2.2878
test accuracy = 0.2500
checkpoint = checkpoints\v2_fashion_mnist_teacher_smoke.pt
```

Meaning: The student reran the teacher one-batch smoke test from terminal. The script executed, evaluated, and saved the smoke checkpoint again. This remains a smoke test only, not a real teacher performance result.

### Details: Fashion-MNIST Baseline Smoke Test 001

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_baseline.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0 --max-train-batches 1 --max-eval-batches 1
```

Result:

```text
train loss = 2.3172
test accuracy = 0.1562
```

Meaning: The hard-label student baseline script can run one train batch and evaluate one batch. This is not a real baseline result.

### Details: Fashion-MNIST Baseline Smoke Test 002

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_baseline.py --epochs 1 --batch-size 64 --seed 0 --max-train-batches 1 --max-eval-batches 1
```

Result:

```text
train loss = 2.3172
test accuracy = 0.1562
```

Meaning: The student reran the hard-label student baseline smoke test from terminal. The low accuracy is expected because this was limited to one train batch and one eval batch. It is a code-path check, not a real baseline result.

### Details: Fashion-MNIST KD Smoke Test 001

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_distillation.py --teacher-checkpoint checkpoints\v2_fashion_mnist_teacher_smoke.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 2.0 --alpha 0.7 --seed 0 --max-train-batches 1 --max-eval-batches 1
```

Result:

```text
hard loss = 2.3172
soft loss = 0.4226
combined loss = 1.7489
test accuracy = 0.1562
```

Meaning: The KD script can load the fixed smoke teacher checkpoint, compute hard loss, soft loss, and combined loss, and update only the student. This is not a real KD result.

### Details: Fashion-MNIST KD Smoke Test 002

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_distillation.py --teacher-checkpoint checkpoints\v2_fashion_mnist_teacher_smoke.pt --epochs 1 --batch-size 64 --seed 0 --temperature 2.0 --alpha 0.7 --max-train-batches 1 --max-eval-batches 1
```

Result:

```text
hard loss = 2.3172
soft loss = 0.4226
combined loss = 1.7489
test accuracy = 0.1562
```

Meaning: The student reran the KD smoke test from terminal. The script loaded the fixed smoke teacher checkpoint, computed hard loss, soft loss, and combined loss, and updated only the student. This is a code-path check, not a real KD result.

### Details: Fashion-MNIST Teacher Seed 0

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_teacher.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0 --checkpoint-path checkpoints\v2_fashion_mnist_teacher_seed0.pt
```

Result:

```text
train loss = 0.4167
test accuracy = 0.8760
checkpoint = checkpoints\v2_fashion_mnist_teacher_seed0.pt
```

Meaning: This is the first real Fashion-MNIST V2 teacher checkpoint for the seed-0 controlled comparison. It was trained with hard labels only. No KD claim can be made from this run because the hard-label student baseline and KD student have not been run yet.

### Details: Fashion-MNIST Student Baseline Seed 0

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_baseline.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0
```

Result:

```text
train loss = 0.4762
test accuracy = 0.8597
```

Meaning: This is the first real Fashion-MNIST V2 hard-label student baseline. The run used the student model with true labels only. After this run, the V2 script was renamed more clearly to `train_fashion_mnist_student_baseline.py`; the old filename remains as a compatibility wrapper for reproducibility.

### Details: Fashion-MNIST KD Temperature 2.0 Alpha 0.7 Seed 0

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_distillation.py --teacher-checkpoint checkpoints\v2_fashion_mnist_teacher_seed0.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 2.0 --alpha 0.7 --seed 0
```

Result:

```text
hard loss = 0.4580
soft loss = 0.3830
combined loss = 0.4355
test accuracy = 0.8604
```

Meaning: This is the first approved V2 seed-0 KD run. It should be compared against the hard-label student baseline accuracy `0.8597`, not the teacher accuracy `0.8760`.

### Details: Fashion-MNIST KD Temperature 6.0 Alpha 0.7 Seed 0

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_distillation.py --teacher-checkpoint checkpoints\v2_fashion_mnist_teacher_seed0.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 6.0 --alpha 0.7 --seed 0
```

Result:

```text
hard loss = 0.4530
soft loss = 0.7430
combined loss = 0.5400
test accuracy = 0.8642
```

Meaning: This is an unplanned exploratory temperature check after the approved temperature 2.0 run. It should still be recorded. The result should be treated cautiously because it was not part of the first approved controlled design and has not been repeated on another seed.

### Details: Fashion-MNIST KD Temperature 7.0 Alpha 0.7 Seed 0

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_distillation.py --teacher-checkpoint checkpoints\v2_fashion_mnist_teacher_seed0.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 7 --alpha 0.7 --seed 0
```

Result:

```text
hard loss = 0.4529
soft loss = 0.7524
combined loss = 0.5428
test accuracy = 0.8646
```

Meaning: This is the best visible exploratory Fashion-MNIST V2 seed-0 KD result so far. It is above the hard-label student baseline `0.8597` by `0.0049`, but it came from exploratory alpha/temperature checking and has not been repeated on another seed.

Note: The student reported trying multiple temperature and alpha settings, including temperatures `2, 4, 6, 8, 10` and alpha values `0.65, 0.7, 0.75`. The exact outputs for those other exploratory runs were not provided in this chat, so only the visible best run above is recorded with exact numeric details here. If the other terminal outputs are available, they should also be recorded rather than only keeping the best result.

### Details: Fashion-MNIST Student Baseline Seed 1

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_student_baseline.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 1
```

Result:

```text
train loss = 0.4969
test accuracy = 0.8579
```

Meaning: This is the matching hard-label student baseline for the seed-1 reliability check.

### Details: Fashion-MNIST KD Temperature 7.0 Alpha 0.7 Seed 1

Command:

```text
.venv\Scripts\python.exe scripts\v2_fashion_mnist\train_fashion_mnist_distillation.py --teacher-checkpoint checkpoints\v2_fashion_mnist_teacher_seed0.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 7.0 --alpha 0.7 --seed 1
```

Result:

```text
hard loss = 0.4785
soft loss = 0.8910
combined loss = 0.6022
test accuracy = 0.8570
```

Meaning: This seed-1 reliability check did not repeat the seed-0 improvement. KD was slightly below the matching hard-label student baseline by `0.0009`.
