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

Remove both `--max-train-batches` and `--max-eval-batches` from the command.

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

## Part 7: CIFAR-10 V3 Blueprint

### V3 Step 0: Scope approval

Create and approve the CIFAR-10 V3 scope before implementation.

Purpose: Keep Version 3 focused on the research question:

```text
Can the V1/V2 knowledge distillation pipeline be extended to CIFAR-10, and under controlled settings does KD improve the same student model compared with hard-label training?
```

### V3 Step 1: CIFAR-10 data loader and smoke test

Add a CIFAR-10 data loader without changing the MNIST V1 or Fashion-MNIST V2 loaders.

Purpose: Check that CIFAR-10 can be loaded as RGB `3 x 32 x 32` images with 10 classes before any model training begins.

Important: This is only a data pipeline smoke test. It does not train a model, evaluate model accuracy, or say anything about KD performance.

### V3 Step 2: CIFAR-10 model strategy

Create a modest CIFAR-10 CNN teacher/student pair using BatchNorm and Dropout.

Purpose: CIFAR-10 changes the input shape from grayscale `1 x 28 x 28` to RGB `3 x 32 x 32`, so the V1/V2 simple CNN models cannot be reused directly.

Fairness rule:

```text
The hard-label student and KD student must use the exact same CIFAR-10 student architecture.
```

### V3 Step 3: CIFAR-10 teacher one-batch smoke test

Add the CIFAR-10 hard-label teacher training script and run it with:

```text
--max-train-batches 1 --max-eval-batches 1
```

Purpose: Check that the teacher script can load CIFAR-10 data, create the CIFAR-10 teacher model, run one train batch, evaluate one test batch, and save a checkpoint.

Important: The accuracy from this step is not meaningful model performance because only one train batch and one eval batch are used.

### V3 Step 4: CIFAR-10 hard-label student one-batch smoke test

Add the CIFAR-10 hard-label student baseline script and run it with:

```text
--max-train-batches 1 --max-eval-batches 1
```

Purpose: Check that the student baseline script can load CIFAR-10 data, create the CIFAR-10 student model, run one train batch, and evaluate one test batch.

Important: The accuracy from this step is not meaningful model performance because only one train batch and one eval batch are used.

### V3 Step 5: CIFAR-10 KD one-batch smoke test

Run the CIFAR-10 KD script using the smoke teacher checkpoint with:

```text
--max-train-batches 1 --max-eval-batches 1
```

Purpose: Check that the KD script can load the fixed teacher checkpoint, compute hard loss, soft loss, combined KD loss, update only the student, and evaluate one test batch.

Important: The accuracy from this step is not meaningful KD performance because only one train batch and one eval batch are used. The smoke teacher checkpoint is not a real trained teacher.

## Part 8: CIFAR-10 V3 Data Log

### Quick Result Table

| Run | Type | Key setting | Output numbers | Meaning | Raw file |
|---|---|---|---|---|---|
| CIFAR-10 data smoke test 001 | Data smoke test | batch size 8, download enabled | train size 50000; test size 10000; image batch shape `(8, 3, 32, 32)`; label batch shape `(8,)`; image range `[0.000, 1.000]` | CIFAR-10 downloaded locally and the loader produced the expected RGB image tensor shape. No training was run. | not saved as CSV |
| CIFAR-10 data smoke test 002 | Data smoke test | batch size 8, existing local data | train size 50000; test size 10000; image batch shape `(8, 3, 32, 32)`; label batch shape `(8,)`; image range `[0.000, 1.000]` | CIFAR-10 loaded from existing local files without `--download`. No training was run. | not saved as CSV |

### Details: CIFAR-10 Data Smoke Test 001

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\check_cifar10_data.py --download
```

Result:

```text
train dataset size = 50000
test dataset size = 10000
image batch shape = (8, 3, 32, 32)
label batch shape = (8,)
image dtype = torch.float32
label dtype = torch.int64
image value range = [0.000, 1.000]
sample labels = [3, 0, 3, 8, 4, 7, 7, 5]
```

Warning observed:

```text
TorchVision emitted a NumPy VisibleDeprecationWarning while loading CIFAR-10.
```

Meaning: The CIFAR-10 data pipeline works for the first V3 data check. The image shape matches the expected CIFAR-10 RGB format: batch size 8, 3 color channels, 32 by 32 pixels. No training was run, so these numbers are not model performance results.

### Details: CIFAR-10 Data Smoke Test 002

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\check_cifar10_data.py
```

Result:

```text
train dataset size = 50000
test dataset size = 10000
image batch shape = (8, 3, 32, 32)
label batch shape = (8,)
image dtype = torch.float32
label dtype = torch.int64
image value range = [0.000, 1.000]
sample labels = [8, 5, 6, 7, 7, 6, 7, 5]
```

Warning observed:

```text
TorchVision emitted a NumPy VisibleDeprecationWarning while loading CIFAR-10.
```

Meaning: The student reran the CIFAR-10 data smoke test using the existing local dataset. The data loader still produced the expected dataset sizes, RGB tensor shape, dtype, and value range. No training was run, so these numbers are not model performance results.

## Part 9: CIFAR-10 V3 Training Script Smoke Log

### Quick Result Table

| Run | Type | Key setting | Output numbers | Meaning | Raw file |
|---|---|---|---|---|---|
| CIFAR-10 teacher smoke test 001 | One-batch smoke test | teacher, seed 0, batch size 64, 1 train batch, 1 eval batch, device auto resolved to CPU | train loss 2.4169; test accuracy 0.1406 | Teacher script ran, evaluated one batch, and saved a smoke checkpoint. Accuracy is not meaningful because this was one batch only. | not saved as CSV |
| CIFAR-10 student baseline smoke test 001 | One-batch smoke test | student, seed 0, batch size 64, 1 train batch, 1 eval batch, device auto resolved to CPU | train loss 2.2955; test accuracy 0.1719 | Student baseline script ran and evaluated one batch. Accuracy is not meaningful because this was one batch only. | not saved as CSV |
| CIFAR-10 KD smoke test 001 | One-batch smoke test | KD student, smoke teacher checkpoint, seed 0, temperature 2.0, alpha 0.7, batch size 64, 1 train batch, 1 eval batch, device auto resolved to CPU | hard loss 2.2955; soft loss 0.0503; combined loss 1.6219; test accuracy 0.1719 | KD script loaded the fixed smoke teacher checkpoint, computed the KD losses, and updated only the student. Accuracy is not meaningful because this was one batch only. | not saved as CSV |
| CIFAR-10 teacher GPU timing attempt 001 | Failed GPU timing smoke | teacher, seed 0, batch size 128, 20 train batches, 5 eval batches, device auto resolved to CUDA | failed before epoch output with CUDA device busy/unavailable | PyTorch detected the RTX 3060 but could not allocate CUDA tensors. No training result was produced. | not saved as CSV |
| CIFAR-10 CUDA environment check 002 | Environment check | after NVIDIA driver update and restart | torch 2.12.1+cu126; CUDA 12.6; CUDA available True; RTX 3060 Laptop GPU; tensor allocation succeeded | CUDA allocation is now working, so V3 timing/full training can use GPU. This is not a model performance result. | not saved as CSV |
| CIFAR-10 teacher GPU timing smoke 002 | GPU timing smoke | teacher, seed 0, batch size 128, 20 train batches, 5 eval batches, device auto resolved to CUDA | train loss 1.9882; test accuracy 0.1484 | Teacher script can now train and evaluate on CUDA. Accuracy is not meaningful because this used only a small batch subset. | not saved as CSV |
| CIFAR-10 teacher GPU timing smoke 003 | GPU timing smoke | teacher, seed 0, batch size 64, 100 train batches, 20 eval batches, device auto resolved to CUDA | train loss 1.7830; test accuracy 0.3937 | Batch size 64 can train/evaluate on CUDA. Accuracy is not a final result because this used only a subset. | not saved as CSV |
| CIFAR-10 teacher GPU timing smoke 004 | GPU timing smoke | teacher, seed 0, batch size 32, 100 train batches, 20 eval batches, device auto resolved to CUDA | train loss 1.8954; test accuracy 0.3187 | Batch size 32 can train/evaluate on CUDA. Accuracy is not a final result because this used only a subset. | not saved as CSV |
| CIFAR-10 validation split test 001 | Data split unit test | train/validation/test loader support, fake dataset split check | 5 pytest tests passed in 3.92s | The V3 data loader now supports deterministic train/validation/test splitting without touching the official test set. This is not a model performance result. | not saved as CSV |
| CIFAR-10 teacher validation smoke 001 | Validation split smoke | teacher, seed 0, batch size 64, 1 train batch, 1 validation eval batch, device auto resolved to CUDA | train loss 2.4384; validation accuracy 0.1250 | Teacher script uses 45,000 train, 5,000 validation, and keeps 10,000 test separate. No test accuracy was evaluated. | not saved as CSV |
| CIFAR-10 student validation smoke 001 | Validation split smoke | hard-label student baseline, seed 0, batch size 64, 1 train batch, 1 validation eval batch, device auto resolved to CUDA | train loss 2.3547; validation accuracy 0.1094 | Student baseline script uses 45,000 train, 5,000 validation, and keeps 10,000 test separate. No test accuracy was evaluated. | not saved as CSV |
| CIFAR-10 KD validation smoke 001 | Validation split smoke | KD student, smoke teacher checkpoint, seed 0, batch size 64, T=2.0, alpha=0.7, 1 train batch, 1 validation eval batch, device auto resolved to CUDA | hard loss 2.3547; soft loss 0.0470; combined loss 1.6624; validation accuracy 0.1094 | KD script uses 45,000 train, 5,000 validation, and keeps 10,000 test separate. Teacher was fixed and only student trained. | not saved as CSV |
| CIFAR-10 teacher LR mini-check 001 | Teacher LR mini-check | teacher, seed 0, batch size 64, LR 0.001, 3 epochs, device auto resolved to CUDA | epoch 1 val 0.5540; epoch 2 val 0.5490; epoch 3 val 0.6890 | LR 0.001 is a candidate teacher learning rate based on validation accuracy. No test accuracy was evaluated. | not saved as CSV |
| CIFAR-10 teacher LR mini-check 002 | Teacher LR mini-check | teacher, seed 0, batch size 64, LR 0.0005, 3 epochs, device auto resolved to CUDA | epoch 1 val 0.5616; epoch 2 val 0.5674; epoch 3 val 0.6968 | LR 0.0005 slightly exceeded LR 0.001 by validation accuracy at epoch 3. No test accuracy was evaluated. | not saved as CSV |
| CIFAR-10 teacher LR mini-check metrics rerun 003 | Teacher LR mini-check rerun | teacher, seed 0, batch size 64, LR 0.001, 3 epochs, four-metric output, device auto resolved to CUDA | epoch 3 train loss 0.8589; train acc 0.6980; val loss 1.0545; val acc 0.6362 | Rerun after adding train/validation loss and accuracy. No test accuracy was evaluated. | not saved as CSV |
| CIFAR-10 teacher LR mini-check metrics rerun 004 | Teacher LR mini-check rerun | teacher, seed 0, batch size 64, LR 0.0005, 3 epochs, four-metric output, device auto resolved to CUDA | epoch 3 train loss 0.9066; train acc 0.6789; val loss 0.8837; val acc 0.6874 | LR 0.0005 had better epoch-3 validation loss and validation accuracy in the four-metric rerun. No test accuracy was evaluated. | not saved as CSV |
| CIFAR-10 teacher LR 10-epoch comparison 005 | Teacher LR comparison | teacher, seed 0, batch size 64, LR 0.001, 10 epochs, device auto resolved to CUDA | epoch 10 train loss 0.5029; train acc 0.8280; val loss 0.5502; val acc 0.8104 | LR 0.001 trained faster and reached slightly lower train loss/higher train accuracy. No test accuracy was evaluated. | results/tables/v3_cifar10/teacher_lr_10epoch_metrics.csv |
| CIFAR-10 teacher LR 10-epoch comparison 006 | Teacher LR comparison | teacher, seed 0, batch size 64, LR 0.0005, 10 epochs, device auto resolved to CUDA | epoch 10 train loss 0.5531; train acc 0.8093; val loss 0.5543; val acc 0.8138 | LR 0.0005 reached slightly higher validation accuracy with nearly tied validation loss. No test accuracy was evaluated. | results/tables/v3_cifar10/teacher_lr_10epoch_metrics.csv |
| CIFAR-10 teacher LR comparison plots 001 | Visualization artifact | compares LR 0.001 vs 0.0005 over 10 epochs | two plot files created | Training plot compares train loss/accuracy; validation plot compares validation loss/accuracy. | results/figures/v3_cifar10/teacher_lr_training_metrics.png; results/figures/v3_cifar10/teacher_lr_validation_metrics.png |

### Details: CIFAR-10 Teacher Smoke Test 001

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --max-train-batches 1 --max-eval-batches 1 --checkpoint-path checkpoints\v3_cifar10_teacher_smoke.pt
```

Result:

```text
device = cpu
batch size = 64
epochs = 1
learning rate = 0.001
seed = 0
max train batches = 1
max eval batches = 1
train loss = 2.4169
test accuracy = 0.1406
checkpoint = checkpoints\v3_cifar10_teacher_smoke.pt
```

Warning observed:

```text
TorchVision emitted a NumPy VisibleDeprecationWarning while loading CIFAR-10.
```

Meaning: The CIFAR-10 teacher script can run one train batch, evaluate one batch, and save a checkpoint. This checkpoint is only for smoke testing and should not be used as the real V3 teacher.

### Details: CIFAR-10 Student Baseline Smoke Test 001

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_student_baseline.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --max-train-batches 1 --max-eval-batches 1
```

Result:

```text
device = cpu
batch size = 64
epochs = 1
learning rate = 0.001
seed = 0
max train batches = 1
max eval batches = 1
train loss = 2.2955
test accuracy = 0.1719
```

Warning observed:

```text
TorchVision emitted a NumPy VisibleDeprecationWarning while loading CIFAR-10.
```

Meaning: The CIFAR-10 hard-label student baseline script can run one train batch and evaluate one batch. This is not a real baseline result.

### Details: CIFAR-10 KD Smoke Test 001

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_smoke.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 2.0 --alpha 0.7 --seed 0 --device auto --max-train-batches 1 --max-eval-batches 1
```

Result:

```text
device = cpu
teacher checkpoint = checkpoints\v3_cifar10_teacher_smoke.pt
batch size = 64
epochs = 1
learning rate = 0.001
temperature = 2.0
alpha = 0.7
seed = 0
max train batches = 1
max eval batches = 1
hard loss = 2.2955
soft loss = 0.0503
combined loss = 1.6219
test accuracy = 0.1719
```

Warning observed:

```text
TorchVision emitted a NumPy VisibleDeprecationWarning while loading CIFAR-10.
```

Meaning: The CIFAR-10 KD script loaded the fixed smoke teacher checkpoint, computed hard loss, soft loss, and combined loss, and updated only the student. This is a code-path check, not a real KD result.

### Details: CIFAR-10 Teacher GPU Timing Attempt 001

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 1 --batch-size 128 --learning-rate 0.001 --seed 0 --device auto --max-train-batches 20 --max-eval-batches 5 --checkpoint-path checkpoints\v3_cifar10_teacher_gpu_timing.pt
```

Result:

```text
The command failed before producing an epoch result.
Error: CUDA-capable device(s) is/are busy or unavailable.
```

Follow-up check:

```text
torch = 2.12.1+cu126
torch.cuda.is_available() = True
device name = NVIDIA GeForce RTX 3060 Laptop GPU
torch.ones(1, device="cuda") failed with the same busy/unavailable error
nvidia-smi driver version = 528.49
nvidia-smi CUDA version display = 12.0
RazerAxon.Player.exe was shown as a GPU process. After closing it once, CUDA
allocation still failed and Razer Axon restarted with a new process id.
```

Meaning: This is an environment/GPU availability problem, not a CIFAR-10 training result. No teacher accuracy or loss should be recorded from this failed attempt.

### Details: CIFAR-10 CUDA Environment Check 002

Command:

```text
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x=torch.ones(1, device='cuda'); print(x); print('CUDA OK')"
```

Result:

```text
torch = 2.12.1+cu126
torch.version.cuda = 12.6
torch.cuda.is_available() = True
device name = NVIDIA GeForce RTX 3060 Laptop GPU
torch.ones(1, device="cuda") = tensor([1.], device="cuda:0")
CUDA OK
nvidia-smi driver version = 610.62
```

Meaning: The NVIDIA driver update and restart fixed CUDA tensor allocation. V3 training can now use CUDA, subject to a small timing check before full training.

### Details: CIFAR-10 Teacher GPU Timing Smoke 002

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 1 --batch-size 128 --learning-rate 0.001 --seed 0 --device auto --max-train-batches 20 --max-eval-batches 5 --checkpoint-path checkpoints\v3_cifar10_teacher_gpu_timing.pt
```

Result:

```text
device = cuda
batch size = 128
epochs = 1
learning rate = 0.001
seed = 0
max train batches = 20
max eval batches = 5
train loss = 1.9882
test accuracy = 0.1484
checkpoint = checkpoints\v3_cifar10_teacher_gpu_timing.pt
```

Warning observed:

```text
TorchVision emitted a NumPy VisibleDeprecationWarning while loading CIFAR-10.
```

Meaning: The teacher training script now runs on CUDA. This is still a timing/smoke subset, not a real teacher result.

### Details: CIFAR-10 Teacher GPU Timing Smoke 003

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --max-train-batches 100 --max-eval-batches 20 --checkpoint-path checkpoints\v3_cifar10_teacher_gpu_timing_bs64.pt
```

Result:

```text
device = cuda
batch size = 64
epochs = 1
learning rate = 0.001
seed = 0
max train batches = 100
max eval batches = 20
train loss = 1.7830
test accuracy = 0.3937
checkpoint = checkpoints\v3_cifar10_teacher_gpu_timing_bs64.pt
```

Meaning: The teacher script ran on CUDA with batch size 64. This is a subset timing/planning run, not a final teacher result.

### Details: CIFAR-10 Teacher GPU Timing Smoke 004

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 1 --batch-size 32 --learning-rate 0.001 --seed 0 --device auto --max-train-batches 100 --max-eval-batches 20 --checkpoint-path checkpoints\v3_cifar10_teacher_gpu_timing_bs32.pt
```

Result:

```text
device = cuda
batch size = 32
epochs = 1
learning rate = 0.001
seed = 0
max train batches = 100
max eval batches = 20
train loss = 1.8954
test accuracy = 0.3187
checkpoint = checkpoints\v3_cifar10_teacher_gpu_timing_bs32.pt
```

Meaning: The teacher script ran on CUDA with batch size 32. This is a subset timing/planning run, not a final teacher result.

### Details: CIFAR-10 Validation Split Test 001

Command:

```text
.venv\Scripts\python.exe -m pytest tests\v3_cifar10\test_cifar10_data.py
```

Result:

```text
5 passed in 3.92s
```

Meaning: The CIFAR-10 data tests pass after adding deterministic train/validation/test loader support. The intended real V3 split is 45,000 train images, 5,000 validation images, and the official 10,000 test images kept for final selected configurations.

### Details: CIFAR-10 Teacher Validation Smoke 001

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --max-train-batches 1 --max-eval-batches 1 --checkpoint-path checkpoints\v3_cifar10_teacher_val_smoke.pt
```

Result:

```text
device = cuda
batch size = 64
epochs = 1
learning rate = 0.001
seed = 0
train dataset size = 45000
validation dataset size = 5000
test dataset size = 10000
validation split seed = 0
max train batches = 1
max eval batches = 1
evaluate test = False
train loss = 2.4384
validation accuracy = 0.1250
checkpoint = checkpoints\v3_cifar10_teacher_val_smoke.pt
```

Meaning: The teacher script now uses the train/validation/test split correctly and does not evaluate the official test set during tuning smoke tests.

### Details: CIFAR-10 Student Validation Smoke 001

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_student_baseline.py --epochs 1 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --max-train-batches 1 --max-eval-batches 1
```

Result:

```text
device = cuda
batch size = 64
epochs = 1
learning rate = 0.001
seed = 0
train dataset size = 45000
validation dataset size = 5000
test dataset size = 10000
validation split seed = 0
max train batches = 1
max eval batches = 1
evaluate test = False
train loss = 2.3547
validation accuracy = 0.1094
```

Meaning: The hard-label student baseline script now uses the train/validation/test split correctly and does not evaluate the official test set during tuning smoke tests.

### Details: CIFAR-10 KD Validation Smoke 001

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_val_smoke.pt --epochs 1 --batch-size 64 --learning-rate 0.001 --temperature 2.0 --alpha 0.7 --seed 0 --device auto --max-train-batches 1 --max-eval-batches 1
```

Result:

```text
device = cuda
teacher checkpoint = checkpoints\v3_cifar10_teacher_val_smoke.pt
batch size = 64
epochs = 1
learning rate = 0.001
temperature = 2.0
alpha = 0.7
seed = 0
train dataset size = 45000
validation dataset size = 5000
test dataset size = 10000
validation split seed = 0
max train batches = 1
max eval batches = 1
evaluate test = False
hard loss = 2.3547
soft loss = 0.0470
combined loss = 1.6624
validation accuracy = 0.1094
```

Meaning: The KD script now uses the train/validation/test split correctly, keeps the teacher checkpoint fixed, trains only the student, and does not evaluate the official test set during tuning smoke tests.

### Details: CIFAR-10 Teacher LR Mini-Check 001

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 3 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --checkpoint-path checkpoints\v3_cifar10_teacher_lr0001_3epoch.pt
```

Result:

```text
device = cuda
batch size = 64
epochs = 3
learning rate = 0.001
seed = 0
train dataset size = 45000
validation dataset size = 5000
test dataset size = 10000
validation split seed = 0
evaluate test = False
epoch 1: train loss = 1.3566, validation accuracy = 0.5540
epoch 2: train loss = 0.9940, validation accuracy = 0.5490
epoch 3: train loss = 0.8612, validation accuracy = 0.6890
checkpoint = checkpoints\v3_cifar10_teacher_lr0001_3epoch.pt
```

Meaning: This is a teacher LR mini-check using the validation set only. The official test set was not evaluated.

### Details: CIFAR-10 Teacher LR 10-Epoch Comparison 005

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 10 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --checkpoint-path checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt
```

Result:

```text
device = cuda
batch size = 64
epochs = 10
learning rate = 0.001
seed = 0
train dataset size = 45000
validation dataset size = 5000
test dataset size = 10000
validation split seed = 0
evaluate test = False
epoch 1: train loss = 1.3554, train accuracy = 0.5065, validation loss = 1.1505, validation accuracy = 0.5826
epoch 2: train loss = 0.9893, train accuracy = 0.6448, validation loss = 1.4000, validation accuracy = 0.5422
epoch 3: train loss = 0.8569, train accuracy = 0.6994, validation loss = 1.0136, validation accuracy = 0.6470
epoch 4: train loss = 0.7625, train accuracy = 0.7340, validation loss = 0.8300, validation accuracy = 0.7094
epoch 5: train loss = 0.6927, train accuracy = 0.7596, validation loss = 0.6372, validation accuracy = 0.7772
epoch 6: train loss = 0.6403, train accuracy = 0.7782, validation loss = 0.6408, validation accuracy = 0.7788
epoch 7: train loss = 0.6020, train accuracy = 0.7911, validation loss = 0.6621, validation accuracy = 0.7792
epoch 8: train loss = 0.5620, train accuracy = 0.8081, validation loss = 0.6658, validation accuracy = 0.7730
epoch 9: train loss = 0.5334, train accuracy = 0.8106, validation loss = 0.6055, validation accuracy = 0.7894
epoch 10: train loss = 0.5029, train accuracy = 0.8280, validation loss = 0.5502, validation accuracy = 0.8104
checkpoint = checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt
```

Meaning: LR 0.001 trained faster and achieved better training metrics, but its final validation accuracy was slightly below LR 0.0005. The official test set was not evaluated.

### Details: CIFAR-10 Teacher LR 10-Epoch Comparison 006

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 10 --batch-size 64 --learning-rate 0.0005 --seed 0 --device auto --checkpoint-path checkpoints\v3_cifar10_teacher_lr00005_10epoch_metrics.pt
```

Result:

```text
device = cuda
batch size = 64
epochs = 10
learning rate = 0.0005
seed = 0
train dataset size = 45000
validation dataset size = 5000
test dataset size = 10000
validation split seed = 0
evaluate test = False
epoch 1: train loss = 1.4072, train accuracy = 0.4915, validation loss = 1.2916, validation accuracy = 0.5490
epoch 2: train loss = 1.0350, train accuracy = 0.6311, validation loss = 1.3241, validation accuracy = 0.5582
epoch 3: train loss = 0.9083, train accuracy = 0.6780, validation loss = 0.9154, validation accuracy = 0.6750
epoch 4: train loss = 0.8147, train accuracy = 0.7165, validation loss = 0.8118, validation accuracy = 0.7134
epoch 5: train loss = 0.7474, train accuracy = 0.7402, validation loss = 0.7632, validation accuracy = 0.7378
epoch 6: train loss = 0.6893, train accuracy = 0.7614, validation loss = 0.7666, validation accuracy = 0.7330
epoch 7: train loss = 0.6496, train accuracy = 0.7747, validation loss = 0.6866, validation accuracy = 0.7686
epoch 8: train loss = 0.6122, train accuracy = 0.7890, validation loss = 0.6224, validation accuracy = 0.7850
epoch 9: train loss = 0.5790, train accuracy = 0.8016, validation loss = 0.5868, validation accuracy = 0.7972
epoch 10: train loss = 0.5531, train accuracy = 0.8093, validation loss = 0.5543, validation accuracy = 0.8138
checkpoint = checkpoints\v3_cifar10_teacher_lr00005_10epoch_metrics.pt
```

Meaning: LR 0.0005 trained slightly more slowly but achieved the highest final validation accuracy in this 10-epoch comparison. The official test set was not evaluated.

### Details: CIFAR-10 Teacher LR Comparison Plots 001

Artifacts:

```text
results/tables/v3_cifar10/teacher_lr_10epoch_metrics.csv
results/tables/v3_cifar10/cifar10_v3_result_summary.csv
results/raw/v3_cifar10/teacher_lr_10epoch_metrics.csv
results/figures/v3_cifar10/teacher_lr_training_metrics.png
results/figures/v3_cifar10/teacher_lr_validation_metrics.png
```

Meaning: These files summarize and visualize the teacher LR comparison using train/validation metrics only. The official test set was not evaluated.

### Decision Path: CIFAR-10 Teacher Learning Rate

Path:

```text
The first 1-epoch teacher run was only a smoke test to check whether the script
could run.

The first 3-epoch LR mini-check compared LR 0.001 and LR 0.0005, but 3 epochs
was too short to confidently decide the teacher LR.

The comparison was extended to 10 epochs. Even after 10 epochs, the margin was
small: LR 0.0005 had slightly higher final validation accuracy, while LR 0.001
had stronger training loss/accuracy and caught up closely on validation.
```

Selected setting:

```text
teacher learning rate = 0.001
```

Important limitation:

```text
This decision used train and validation metrics only. The official CIFAR-10 test
set was not evaluated.
```

### Decision: CIFAR-10 Teacher LR Selection 001

Decision:

```text
Selected teacher learning rate = 0.001
```

Reason:

```text
Although LR 0.0005 ended with slightly higher validation accuracy at epoch 10
in the 10-epoch comparison, the validation margin was small enough that the two
learning rates were treated as approximately tied for this V3 check. LR 0.001
was retained as the default/consistent setting for the teacher and student
experiments, not because it was clearly superior on validation accuracy:

LR 0.001 epoch 10: validation loss = 0.5502, validation accuracy = 0.8104
LR 0.0005 epoch 10: validation loss = 0.5543, validation accuracy = 0.8138
```

Meaning: V3 will use the existing `checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt` checkpoint as the selected seed-0 teacher checkpoint for the first student baseline and KD runs. This decision was made using validation metrics only; the official test set remains untouched.

## Part 10: CIFAR-10 V3 Student Baseline and KD Sweep Log

### Quick Result Table

| Run | Type | Key setting | Epoch-10 output numbers | Meaning | Raw file |
|---|---|---|---|---|---|
| CIFAR-10 student baseline seed 0 | Hard-label baseline | student, seed 0, batch size 64, LR 0.001, 10 epochs | train loss 0.9304; train acc 0.6683; val loss 0.8993; val acc 0.6776 | Same student architecture trained with hard labels only. This is the baseline for KD comparison. Test set was not evaluated. | results/raw/v3_cifar10/student_vs_kd_alpha07_seed0_10epoch_metrics.csv |
| CIFAR-10 KD T=2 alpha=0.7 seed 0 | KD sweep | student, fixed teacher, seed 0, T=2.0, alpha=0.7, 10 epochs | train loss 1.0074; train acc 0.6781; val loss 0.9193; val acc 0.6854 | Best final validation accuracy among the first alpha=0.7 KD temperature sweep; slightly above baseline. Test set was not evaluated. | results/raw/v3_cifar10/student_vs_kd_alpha07_seed0_10epoch_metrics.csv |
| CIFAR-10 KD T=4 alpha=0.7 seed 0 | KD sweep | student, fixed teacher, seed 0, T=4.0, alpha=0.7, 10 epochs | train loss 1.2630; train acc 0.6670; val loss 1.2252; val acc 0.6548 | Below hard-label baseline at epoch 10. Test set was not evaluated. | results/raw/v3_cifar10/student_vs_kd_alpha07_seed0_10epoch_metrics.csv |
| CIFAR-10 KD T=6 alpha=0.7 seed 0 | KD sweep | student, fixed teacher, seed 0, T=6.0, alpha=0.7, 10 epochs | train loss 1.3132; train acc 0.6663; val loss 1.1077; val acc 0.6742 | Slightly below hard-label baseline at epoch 10. Test set was not evaluated. | results/raw/v3_cifar10/student_vs_kd_alpha07_seed0_10epoch_metrics.csv |
| CIFAR-10 student baseline LR 0.0005 seed 0 | Student LR comparison | student, seed 0, batch size 64, LR 0.0005, 10 epochs | train loss 0.9723; train acc 0.6538; val loss 0.9196; val acc 0.6752 | Slightly below LR 0.001 hard-label student baseline at epoch 10. Test set was not evaluated. | results/raw/v3_cifar10/student_lr_and_kd_alpha05_seed0_10epoch_metrics.csv |
| CIFAR-10 KD T=2 alpha=0.5 seed 0 | KD sweep | student, fixed teacher, seed 0, T=2.0, alpha=0.5, 10 epochs | train loss 1.0289; train acc 0.6804; val loss 1.0399; val acc 0.6508 | Peaked earlier but ended below baseline at epoch 10. Test set was not evaluated. | results/raw/v3_cifar10/student_lr_and_kd_alpha05_seed0_10epoch_metrics.csv |
| CIFAR-10 KD T=4 alpha=0.5 seed 0 | KD sweep | student, fixed teacher, seed 0, T=4.0, alpha=0.5, 10 epochs | train loss 1.3817; train acc 0.6702; val loss 1.1307; val acc 0.6710 | Below hard-label baseline at epoch 10. Test set was not evaluated. | results/raw/v3_cifar10/student_lr_and_kd_alpha05_seed0_10epoch_metrics.csv |
| CIFAR-10 KD T=6 alpha=0.5 seed 0 | KD sweep | student, fixed teacher, seed 0, T=6.0, alpha=0.5, 10 epochs | train loss 1.4920; train acc 0.6652; val loss 1.0708; val acc 0.6952 | Best epoch-10 validation accuracy among current KD sweep results. Test set was not evaluated. | results/raw/v3_cifar10/student_lr_and_kd_alpha05_seed0_10epoch_metrics.csv |

### Details: Student vs KD Alpha 0.7 Sweep

Artifacts:

```text
results/tables/v3_cifar10/cifar10_v3_result_summary.csv
results/raw/v3_cifar10/student_vs_kd_alpha07_seed0_10epoch_metrics.csv
results/figures/v3_cifar10/student_vs_kd_alpha07_training_metrics.png
results/figures/v3_cifar10/student_vs_kd_alpha07_validation_metrics.png
results/figures/v3_cifar10/student_vs_kd_alpha07_epoch10_summary.png
results/figures/v3_cifar10/student_lr_training_metrics.png
results/figures/v3_cifar10/student_lr_validation_metrics.png
results/figures/v3_cifar10/kd_sweep_validation_metrics_seed0.png
results/figures/v3_cifar10/kd_sweep_epoch10_and_peak_summary_seed0.png
```

Interpretation:

```text
At epoch 10, KD with T=2.0 and alpha=0.7 reached validation accuracy 0.6854,
which is slightly above the hard-label student baseline validation accuracy
0.6776. The difference is small, so this should be treated as promising but not
robust yet.

T=4.0 and T=6.0 with alpha=0.7 did not beat the hard-label baseline at epoch 10.

After adding alpha=0.5 runs, KD T=6.0 alpha=0.5 reached the highest epoch-10
validation accuracy so far: 0.6952. However, KD T=2.0 alpha=0.7 still had the
highest best-within-10-epochs validation accuracy: 0.6956 at epoch 9. The V3
decision should explicitly choose whether selection uses final epoch or best
validation epoch / early stopping.

The student LR check showed LR 0.001 remains the preferred student learning
rate for the hard-label baseline, because LR 0.0005 ended slightly lower:
0.6752 vs 0.6776 validation accuracy at epoch 10.
```

Important limitation:

```text
These are validation results only. The official CIFAR-10 test set was not
evaluated.
```

### Details: CIFAR-10 Seed-1 Student Repeat

Commands:

```text
powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_student_baseline.py --epochs 10 --batch-size 64 --learning-rate 0.001 --seed 1 --device auto --checkpoint-path checkpoints\v3_cifar10_student_baseline_seed1_lr0001_10epoch.pt 2>&1 | Tee-Object -FilePath results\raw\v3_cifar10\run_logs\seed1_student_baseline_lr0001_10epoch.txt"

.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt --epochs 10 --batch-size 64 --learning-rate 0.001 --temperature 6 --alpha 0.5 --seed 1 --device auto
```

Run logs:

```text
results/raw/v3_cifar10/run_logs/seed1_student_baseline_lr0001_10epoch.txt
results/raw/v3_cifar10/run_logs/seed1_kd_t6_alpha05_lr0001_10epoch.txt
```

Artifacts:

```text
results/raw/v3_cifar10/seed1_repeat_student_vs_kd_t6_alpha05_10epoch_metrics.csv
results/figures/v3_cifar10/seed1_student_vs_kd_t6_alpha05_training_metrics.png
results/figures/v3_cifar10/seed1_student_vs_kd_t6_alpha05_validation_metrics.png
results/figures/v3_cifar10/selected_kd_seed_repeat_validation_summary.png
```

Result:

| Run | Type | Changed variables | Epoch-10 output | Meaning | Raw CSV |
|---|---|---|---|---|---|
| CIFAR-10 student baseline seed 1 | Seed repeat | student, seed 1, batch size 64, LR 0.001, 10 epochs | train loss 0.9126; train acc 0.6768; val loss 0.8650; val acc 0.6952 | Same hard-label student baseline repeated with seed 1. Test set was not evaluated. | results/raw/v3_cifar10/seed1_repeat_student_vs_kd_t6_alpha05_10epoch_metrics.csv |
| CIFAR-10 KD T=6 alpha=0.5 seed 1 | Seed repeat | student, fixed teacher checkpoint, seed 1, T=6.0, alpha=0.5, 10 epochs | train loss 1.5048; train acc 0.6624; val loss 1.0809; val acc 0.7032 | Selected KD setting repeated with seed 1. It is +0.0080 above the seed-1 hard-label baseline by final validation accuracy. Test set was not evaluated. | results/raw/v3_cifar10/seed1_repeat_student_vs_kd_t6_alpha05_10epoch_metrics.csv |

Interpretation:

```text
The selected KD setting, T=6.0 and alpha=0.5, improved over the hard-label
student baseline in seed 1 as well as seed 0 when using final epoch-10
validation accuracy:

seed 0: hard-label 0.6776 vs KD 0.6952, difference +0.0176
seed 1: hard-label 0.6952 vs KD 0.7032, difference +0.0080

This is a stronger V3 validation signal than V1/V2 so far, but it is still not
a final test-set conclusion. The teacher checkpoint is fixed from seed 0, and
the official test set has not been evaluated.
```

### Details: CIFAR-10 Seed-2 Student Repeat

Commands:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_student_baseline.py --epochs 10 --batch-size 64 --learning-rate 0.001 --seed 2 --device auto --checkpoint-path checkpoints\v3_cifar10_student_baseline_seed2_lr0001_10epoch.pt

.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt --epochs 10 --batch-size 64 --learning-rate 0.001 --temperature 6 --alpha 0.5 --seed 2 --device auto
```

Run logs:

```text
results/raw/v3_cifar10/run_logs/seed2_student_baseline_lr0001_10epoch.txt
results/raw/v3_cifar10/run_logs/seed2_kd_t6_alpha05_lr0001_10epoch.txt
```

Artifacts:

```text
results/raw/v3_cifar10/seed2_repeat_student_vs_kd_t6_alpha05_10epoch_metrics.csv
results/figures/v3_cifar10/seed2_student_vs_kd_t6_alpha05_training_metrics.png
results/figures/v3_cifar10/seed2_student_vs_kd_t6_alpha05_validation_metrics.png
```

Result:

| Run | Type | Changed variables | Epoch-10 output | Meaning | Raw CSV |
|---|---|---|---|---|---|
| CIFAR-10 student baseline seed 2 | Seed repeat | student, seed 2, batch size 64, LR 0.001, 10 epochs | train loss 0.8960; train acc 0.6810; val loss 0.8816; val acc 0.6832 | Same hard-label student baseline repeated with seed 2. Test set was not evaluated. | results/raw/v3_cifar10/seed2_repeat_student_vs_kd_t6_alpha05_10epoch_metrics.csv |
| CIFAR-10 KD T=6 alpha=0.5 seed 2 | Seed repeat | student, fixed teacher checkpoint, seed 2, T=6.0, alpha=0.5, 10 epochs | train loss 1.4688; train acc 0.6711; val loss 1.2990; val acc 0.6998 | Selected KD setting repeated with seed 2. It is +0.0166 above the seed-2 hard-label baseline by final validation accuracy. Test set was not evaluated. | results/raw/v3_cifar10/seed2_repeat_student_vs_kd_t6_alpha05_10epoch_metrics.csv |

Interpretation:

```text
The selected KD setting, T=6.0 and alpha=0.5, also improved over the hard-label
student baseline for seed 2 by final epoch-10 validation accuracy:

seed 2: hard-label 0.6832 vs KD 0.6998, difference +0.0166

Together with seed 0 and seed 1, this gives a consistent validation-level
signal across three student seeds. This is still not a final test-set claim.
The teacher checkpoint is fixed from seed 0, and the official test set has not
been evaluated.
```

### Details: CIFAR-10 Selected Seed 0/1/2 Combined Visualization

Artifacts:

```text
results/figures/v3_cifar10/selected_kd_seed012_final_validation_comparison.png
results/figures/v3_cifar10/selected_kd_seed012_validation_curves_by_seed.png
```

Meaning:

```text
These plots combine only the selected comparison lines:
hard-label student baseline vs KD T=6.0 alpha=0.5. They do not include the
full seed-0 KD sweep, and they do not use the official test set.
```

### Details: CIFAR-10 Stage 1 Seed-0 20-Epoch Check

Commands:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_student_baseline.py --epochs 20 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --checkpoint-path checkpoints\v3_cifar10_student_baseline_seed0_lr0001_20epoch.pt

.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt --epochs 20 --batch-size 64 --learning-rate 0.001 --temperature 6 --alpha 0.5 --seed 0 --device auto
```

Run logs:

```text
results/raw/v3_cifar10/run_logs/stage1_seed0_student_baseline_lr0001_20epoch.txt
results/raw/v3_cifar10/run_logs/stage1_seed0_kd_t6_alpha05_lr0001_20epoch.txt
```

Artifact:

```text
results/raw/v3_cifar10/stage1_seed0_student_vs_kd_t6_alpha05_20epoch_metrics.csv
results/figures/v3_cifar10/stage1_seed0_20epoch_training_metrics.png
results/figures/v3_cifar10/stage1_seed0_20epoch_validation_metrics.png
```

Result:

| Run | Type | Changed variables | Epoch-20 output | Meaning | Raw CSV |
|---|---|---|---|---|---|
| CIFAR-10 student baseline seed 0, 20 epochs | Stage 1 longer check | student, seed 0, batch size 64, LR 0.001, 20 epochs | train loss 0.8222; train acc 0.7088; val loss 0.7443; val acc 0.7400 | Hard-label baseline improved substantially when training was extended from 10 to 20 epochs. Test set was not evaluated. | results/raw/v3_cifar10/stage1_seed0_student_vs_kd_t6_alpha05_20epoch_metrics.csv |
| CIFAR-10 KD T=6 alpha=0.5 seed 0, 20 epochs | Stage 1 longer check | student, fixed teacher checkpoint, seed 0, T=6.0, alpha=0.5, 20 epochs | train loss 1.3242; train acc 0.7032; val loss 0.9556; val acc 0.7244 | KD improved from the 10-epoch run but ended below the 20-epoch hard-label baseline by -0.0156. Test set was not evaluated. | results/raw/v3_cifar10/stage1_seed0_student_vs_kd_t6_alpha05_20epoch_metrics.csv |

Interpretation:

```text
At 10 epochs, the selected KD setting beat the seed-0 hard-label student:
0.6952 vs 0.6776 validation accuracy.

At 20 epochs, the hard-label student surpassed the selected KD setting:
0.7400 vs 0.7244 validation accuracy.

This means the 10-epoch KD advantage may be partly an early-training advantage.
The longer-run result is more nuanced and should be recorded rather than hidden.
The official test set has still not been evaluated.
```

### Details: CIFAR-10 Stage 1 Seed-1/2 20-Epoch Check

Commands:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_student_baseline.py --epochs 20 --batch-size 64 --learning-rate 0.001 --seed 1 --device auto --checkpoint-path checkpoints\v3_cifar10_student_baseline_seed1_lr0001_20epoch.pt

.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt --epochs 20 --batch-size 64 --learning-rate 0.001 --temperature 6 --alpha 0.5 --seed 1 --device auto

.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_student_baseline.py --epochs 20 --batch-size 64 --learning-rate 0.001 --seed 2 --device auto --checkpoint-path checkpoints\v3_cifar10_student_baseline_seed2_lr0001_20epoch.pt

.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt --epochs 20 --batch-size 64 --learning-rate 0.001 --temperature 6 --alpha 0.5 --seed 2 --device auto
```

Run logs:

```text
results/raw/v3_cifar10/run_logs/stage1_seed1_student_baseline_lr0001_20epoch.txt
results/raw/v3_cifar10/run_logs/stage1_seed1_kd_t6_alpha05_lr0001_20epoch.txt
results/raw/v3_cifar10/run_logs/stage1_seed2_student_baseline_lr0001_20epoch.txt
results/raw/v3_cifar10/run_logs/stage1_seed2_kd_t6_alpha05_lr0001_20epoch.txt
```

Artifacts:

```text
results/raw/v3_cifar10/stage1_seed12_student_vs_kd_t6_alpha05_20epoch_metrics.csv
results/figures/v3_cifar10/stage1_seed1_20epoch_training_metrics.png
results/figures/v3_cifar10/stage1_seed1_20epoch_validation_metrics.png
results/figures/v3_cifar10/stage1_seed2_20epoch_training_metrics.png
results/figures/v3_cifar10/stage1_seed2_20epoch_validation_metrics.png
```

Result:

| Seed | Hard-label student val acc | KD T=6 alpha=0.5 val acc | KD - baseline | Meaning |
|---|---:|---:|---:|---|
| 1 | 0.7352 | 0.6836 | -0.0516 | Hard-label baseline clearly surpassed KD by epoch 20. KD peaked earlier at 0.7344 on epoch 16 but did not end well. |
| 2 | 0.7408 | 0.7414 | +0.0006 | KD and hard-label baseline were essentially tied by epoch 20. |

Interpretation:

```text
The seed-1 and seed-2 20-epoch checks do not support the stronger claim that
the selected KD configuration remains better with longer training.

Across 20-epoch seed checks:
seed 0: hard-label 0.7400 vs KD 0.7244, KD -0.0156
seed 1: hard-label 0.7352 vs KD 0.6836, KD -0.0516
seed 2: hard-label 0.7408 vs KD 0.7414, KD +0.0006

This supports a cautious interpretation: KD gave consistent validation gains
at 10 epochs, but the advantage mostly disappeared or reversed by 20 epochs.
The official test set has still not been evaluated.
```

### Details: CIFAR-10 Stage 2 Seed-0 70-Epoch Check

Code change before this run:

```text
scripts/v3_cifar10/train_cifar10_distillation.py now supports --checkpoint-path.
This saves the trained KD student checkpoint with model state, teacher
checkpoint, temperature, alpha, seed, epoch count, batch size, learning rate,
and validation split metadata.
```

Commands:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_student_baseline.py --epochs 70 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --checkpoint-path checkpoints\v3_cifar10_student_baseline_seed0_lr0001_70epoch.pt

.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt --epochs 70 --batch-size 64 --learning-rate 0.001 --temperature 6 --alpha 0.5 --seed 0 --device auto --checkpoint-path checkpoints\v3_cifar10_kd_t6_alpha05_seed0_lr0001_70epoch.pt
```

Run logs:

```text
results/raw/v3_cifar10/run_logs/stage2_seed0_student_baseline_lr0001_70epoch.txt
results/raw/v3_cifar10/run_logs/stage2_seed0_kd_t6_alpha05_lr0001_70epoch.txt
```

Artifacts:

```text
results/raw/v3_cifar10/stage2_seed0_student_vs_kd_t6_alpha05_70epoch_metrics.csv
results/figures/v3_cifar10/stage2_seed0_70epoch_training_metrics.png
results/figures/v3_cifar10/stage2_seed0_70epoch_validation_metrics.png
results/figures/v3_cifar10/stage2_seed0_70epoch_validation_accuracy.png
checkpoints/v3_cifar10_student_baseline_seed0_lr0001_70epoch.pt
checkpoints/v3_cifar10_kd_t6_alpha05_seed0_lr0001_70epoch.pt
```

Result:

| Run | Type | Changed variables | Final epoch output | Best validation accuracy | Meaning |
|---|---|---|---|---|---|
| CIFAR-10 student baseline seed 0, 70 epochs | Stage 2 longer check | student, seed 0, batch size 64, LR 0.001, 70 epochs | train loss 0.6545; train acc 0.7721; val loss 0.6123; val acc 0.7912 | 0.7940 at epoch 69 | Hard-label training remained stronger under long training. Test set was not evaluated. |
| CIFAR-10 KD T=6 alpha=0.5 seed 0, 70 epochs | Stage 2 longer check | student, fixed teacher checkpoint, seed 0, T=6.0, alpha=0.5, 70 epochs | train loss 1.1312; train acc 0.7437; val loss 0.7505; val acc 0.7744 | 0.7780 at epoch 67 | KD improved compared with earlier epochs but stayed below the matching hard-label baseline. Test set was not evaluated. |

Interpretation:

```text
At 70 epochs on seed 0, the hard-label baseline is still better than the
selected KD configuration:

final epoch: hard-label 0.7912 vs KD 0.7744, KD -0.0168
best epoch:  hard-label 0.7940 vs KD 0.7780, KD -0.0160

This strengthens the longer-training interpretation from the 20-epoch checks:
KD helped at 10 epochs, but the advantage did not persist when training was
extended. The official test set has still not been evaluated.
```

### Details: CIFAR-10 Final Test Set Evaluation, Seed 0

Important policy:

```text
This is the first V3 use of the official CIFAR-10 test set. These results
should be treated as final evaluation outputs, not as hyperparameter-tuning
feedback. No further tuning should be based on the test set.
```

Reproducibility note:

```text
These final-test commands retrained the 20-epoch and 70-epoch models from
scratch with --evaluate-test. They were not simple test-set evaluations of the
earlier validation-only checkpoints. Because the current CUDA training setup is
not strictly deterministic, the final-test runs can have slightly different
validation accuracies from the earlier validation-only runs even when the seed
and hyperparameters match.
```

Commands:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_student_baseline.py --epochs 20 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --evaluate-test --checkpoint-path checkpoints\v3_cifar10_student_baseline_seed0_lr0001_20epoch_finaltest.pt

.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt --epochs 20 --batch-size 64 --learning-rate 0.001 --temperature 6 --alpha 0.5 --seed 0 --device auto --evaluate-test --checkpoint-path checkpoints\v3_cifar10_kd_t6_alpha05_seed0_lr0001_20epoch_finaltest.pt

.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_student_baseline.py --epochs 70 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --evaluate-test --checkpoint-path checkpoints\v3_cifar10_student_baseline_seed0_lr0001_70epoch_finaltest.pt

.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_distillation.py --teacher-checkpoint checkpoints\v3_cifar10_teacher_lr0001_10epoch_metrics.pt --epochs 70 --batch-size 64 --learning-rate 0.001 --temperature 6 --alpha 0.5 --seed 0 --device auto --evaluate-test --checkpoint-path checkpoints\v3_cifar10_kd_t6_alpha05_seed0_lr0001_70epoch_finaltest.pt
```

Run logs:

```text
results/raw/v3_cifar10/run_logs/finaltest_seed0_student_baseline_lr0001_20epoch.txt
results/raw/v3_cifar10/run_logs/finaltest_seed0_kd_t6_alpha05_lr0001_20epoch.txt
results/raw/v3_cifar10/run_logs/finaltest_seed0_student_baseline_lr0001_70epoch.txt
results/raw/v3_cifar10/run_logs/finaltest_seed0_kd_t6_alpha05_lr0001_70epoch.txt
```

Artifacts:

```text
results/raw/v3_cifar10/finaltest_seed0_student_vs_kd_t6_alpha05_20_70epoch_metrics.csv
results/raw/v3_cifar10/finaltest_seed0_student_vs_kd_t6_alpha05_20_70epoch_summary.csv
results/figures/v3_cifar10/finaltest_seed0_test_accuracy_comparison.png
results/figures/v3_cifar10/finaltest_seed0_validation_vs_test_accuracy.png
checkpoints/v3_cifar10_student_baseline_seed0_lr0001_20epoch_finaltest.pt
checkpoints/v3_cifar10_kd_t6_alpha05_seed0_lr0001_20epoch_finaltest.pt
checkpoints/v3_cifar10_student_baseline_seed0_lr0001_70epoch_finaltest.pt
checkpoints/v3_cifar10_kd_t6_alpha05_seed0_lr0001_70epoch_finaltest.pt
```

Result:

| Epochs | Run | Final validation accuracy | Best validation accuracy | Official test accuracy | Meaning |
|---:|---|---:|---:|---:|---|
| 20 | Hard-label student | 0.7422 | 0.7438 at epoch 19 | 0.7385 | Hard-label student outperformed KD on the official test set. |
| 20 | KD T=6 alpha=0.5 | 0.7198 | 0.7214 at epoch 17 | 0.7144 | KD was -0.0241 below the matching hard-label student on test accuracy. |
| 70 | Hard-label student | 0.7812 | 0.7888 at epoch 69 | 0.7860 | Hard-label student remained stronger under longer training. |
| 70 | KD T=6 alpha=0.5 | 0.7656 | 0.7796 at epoch 57 | 0.7622 | KD was -0.0238 below the matching hard-label student on test accuracy. |

Interpretation:

```text
The official test set is consistent with the longer-training validation story
for seed 0.
For both 20 and 70 epochs, the hard-label student outperformed the selected KD
student on test accuracy:

20 epochs: hard-label 0.7385 vs KD 0.7144, KD -0.0241
70 epochs: hard-label 0.7860 vs KD 0.7622, KD -0.0238

This means the final V3 conclusion should not claim that KD beats hard-label
training overall. A cautious conclusion is that KD showed promising early
validation gains at 10 epochs, but those gains did not persist in longer
training or in the official seed-0 test evaluation.
```

### Details: CIFAR-10 Teacher LR Mini-Check Metrics Rerun 003

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 3 --batch-size 64 --learning-rate 0.001 --seed 0 --device auto --checkpoint-path checkpoints\v3_cifar10_teacher_lr0001_3epoch_metrics.pt
```

Result:

```text
device = cuda
batch size = 64
epochs = 3
learning rate = 0.001
seed = 0
train dataset size = 45000
validation dataset size = 5000
test dataset size = 10000
validation split seed = 0
evaluate test = False
epoch 1: train loss = 1.3590, train accuracy = 0.5050, validation loss = 1.1308, validation accuracy = 0.5904
epoch 2: train loss = 0.9939, train accuracy = 0.6441, validation loss = 1.3985, validation accuracy = 0.5378
epoch 3: train loss = 0.8589, train accuracy = 0.6980, validation loss = 1.0545, validation accuracy = 0.6362
checkpoint = checkpoints\v3_cifar10_teacher_lr0001_3epoch_metrics.pt
```

Meaning: This four-metric rerun shows LR 0.001 continued improving train metrics, but validation loss/accuracy were weaker than LR 0.0005 at epoch 3. The official test set was not evaluated.

### Details: CIFAR-10 Teacher LR Mini-Check Metrics Rerun 004

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 3 --batch-size 64 --learning-rate 0.0005 --seed 0 --device auto --checkpoint-path checkpoints\v3_cifar10_teacher_lr00005_3epoch_metrics.pt
```

Result:

```text
device = cuda
batch size = 64
epochs = 3
learning rate = 0.0005
seed = 0
train dataset size = 45000
validation dataset size = 5000
test dataset size = 10000
validation split seed = 0
evaluate test = False
epoch 1: train loss = 1.4063, train accuracy = 0.4916, validation loss = 1.2716, validation accuracy = 0.5524
epoch 2: train loss = 1.0339, train accuracy = 0.6304, validation loss = 1.3393, validation accuracy = 0.5582
epoch 3: train loss = 0.9066, train accuracy = 0.6789, validation loss = 0.8837, validation accuracy = 0.6874
checkpoint = checkpoints\v3_cifar10_teacher_lr00005_3epoch_metrics.pt
```

Meaning: This four-metric rerun supports selecting LR 0.0005 for teacher training because it had better epoch-3 validation loss and validation accuracy. The official test set was not evaluated.

### Details: CIFAR-10 Teacher LR Mini-Check 002

Command:

```text
.venv\Scripts\python.exe scripts\v3_cifar10\train_cifar10_teacher.py --epochs 3 --batch-size 64 --learning-rate 0.0005 --seed 0 --device auto --checkpoint-path checkpoints\v3_cifar10_teacher_lr00005_3epoch.pt
```

Result:

```text
device = cuda
batch size = 64
epochs = 3
learning rate = 0.0005
seed = 0
train dataset size = 45000
validation dataset size = 5000
test dataset size = 10000
validation split seed = 0
evaluate test = False
epoch 1: train loss = 1.4072, validation accuracy = 0.5616
epoch 2: train loss = 1.0368, validation accuracy = 0.5674
epoch 3: train loss = 0.9090, validation accuracy = 0.6968
checkpoint = checkpoints\v3_cifar10_teacher_lr00005_3epoch.pt
```

Meaning: This is a teacher LR mini-check using the validation set only. The official test set was not evaluated.
