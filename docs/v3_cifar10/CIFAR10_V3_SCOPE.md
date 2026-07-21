# CIFAR-10 Knowledge Distillation: Version 3 Scope

This file defines the planned scope for Version 3 of the knowledge distillation reproduction project.

Historical planning document: this scope was written before implementation.
V3 implementation, evaluation, results summary, and GitHub packaging are now
complete. The original planning language below is preserved so the project
history remains clear.

Original stage at the time of writing:

```text
V3 scope draft created.
CIFAR-10 implementation had not started.
```

Version 3 should begin only after this scope is reviewed and approved.

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-19
- Verification Status: UNVERIFIED planning document
- Version Label: v3_cifar10_scope_v1

## 1. Learning and Portfolio Context

This project is the student's first paper reproduction project. Version 1 showed a complete MNIST teacher-student KD pipeline. Version 2 extended the same pipeline to Fashion-MNIST and produced a cautious result: seed 0 was promising, but seed 1 did not repeat the KD improvement.

Version 3 should be a stronger portfolio extension, but it should still be taught step by step. The student should understand:

- why CIFAR-10 is a larger jump than Fashion-MNIST
- why CIFAR-10 requires new data and model decisions
- why the same student model must still be compared against itself
- why teacher strength matters for distillation
- why smoke tests are not final results
- why weak or negative results still belong in the research story

Codex should help with scaffolding, code edits after approval, verification, exact experiment logging, and writing polish. The student should keep ownership of the research question, experiment approval, result interpretation, and portfolio-facing claims.

Before each V3 step, use this responsibility checkpoint:

```text
Student should do:
- understand the concept or research reason behind the step
- decide whether the comparison is fair
- approve model/data/training choices before full runs
- explain important observations in their own words
- choose what belongs in the portfolio-facing story

Codex can help:
- create or edit files after explaining the purpose
- run tests and smoke checks after approval
- record outputs exactly as produced
- keep bad or weak results visible
- polish wording without changing the student's meaning
```

Important V3 learning boundary:

- The student should review important code, especially the CIFAR-10 loader, model definitions, training scripts, KD loss, and result-writing logic.
- Codex may scaffold files and run tests/smoke tests, but full training requires student approval first.
- Before full training, confirm the comparison, seed, teacher checkpoint, model architecture, epochs, batch size, learning rate, temperature, alpha, and success criteria.
- After results are produced, Codex should show the numbers and ask the student to interpret them before writing final claims.
- README and final summary claims should start from the student's draft. Codex can polish but should not add stronger claims silently.

## 2. Why CIFAR-10 After Fashion-MNIST

MNIST V1 and Fashion-MNIST V2 used small grayscale images:

```text
MNIST/Fashion-MNIST input shape = 1 x 28 x 28
number of classes = 10
task type = image classification
```

Fashion-MNIST was a useful bridge because it changed dataset difficulty while keeping the same image shape. That allowed V2 to reuse the simple CNN teacher and student models.

CIFAR-10 is a stronger Version 3 target because it is more realistic:

```text
CIFAR-10 input shape = 3 x 32 x 32
number of classes = 10
task type = color image classification
```

This makes CIFAR-10 a better portfolio stretch, but it also means V3 is not just a dataset swap. The project must handle color channels, larger images, more complex visual patterns, and likely longer training time.

The main Version 3 research question is:

```text
Can the V1/V2 knowledge distillation pipeline be extended to CIFAR-10, and under controlled settings does KD improve the same student model compared with hard-label training?
```

The safer wording is important. Version 3 should test whether KD helps in this reproduction setting; it should not assume KD will beat hard-label training.

## 3. V3 File Organization

To avoid mixing V1, V2, and V3, V3-facing files should live in V3-specific folders:

```text
docs/v3_cifar10/
scripts/v3_cifar10/
tests/v3_cifar10/
results/tables/v3_cifar10/
```

Reusable package code should stay in the normal package folders:

```text
src/kd_research/data/cifar10.py
src/kd_research/models/
```

Reason: documentation, scripts, tests, and final result tables are easier to find when they are grouped by project version. Shared Python package code should stay organized by function so scripts can import it cleanly.

Raw real-run CSV files should still go under:

```text
results/raw/
```

Those raw files are local and ignored by Git unless the student explicitly decides to publish them.

Do not move MNIST V1 or Fashion-MNIST V2 result files while starting V3. Earlier versions should remain reproducible.

## 4. What Stays the Same From V1 and V2

Version 3 should preserve the main controlled comparison:

- Train a teacher model with hard labels.
- Save and freeze a fixed teacher checkpoint.
- Train the student model with hard labels only.
- Train the same student model with knowledge distillation.
- Compare the hard-label student and KD student under matched settings.
- Use the official training split for training and official test split for evaluation.
- Use fixed seeds for controlled comparisons.
- Record every numeric run in `docs/CONTROLLED_EXPERIMENT_LOG.md`.
- Store real raw run records under `results/raw/`.
- Keep datasets and checkpoints local and ignored by Git.
- Report negative, weak, failed, or confusing results honestly.
- Avoid unlimited test-set tuning.

The most important fairness rule remains:

```text
hard-label student and KD student must use the same student architecture.
```

If the student architecture changes later, both hard-label and KD runs must use the changed architecture before making a comparison.

## 5. What Can Be Reused

The V3 research workflow can reuse:

- teacher hard-label training
- student hard-label baseline training
- KD student training
- same-student comparison logic
- seed 0 and seed 1 controlled comparison
- experiment log plus raw CSV plus summary CSV habit
- smoke test first, full training later
- cautious conclusion style

The V3 code can partially reuse:

- `train_one_epoch`
- `evaluate_accuracy`
- `distillation_loss`
- `load_teacher`
- argparse command style
- logging and print style
- batch-limit smoke-test arguments

The V3 implementation should not blindly copy:

- MNIST/Fashion-MNIST model shapes
- 1-channel convolution inputs
- `28 x 28` flatten dimensions
- the assumption that one epoch is enough for an informative run

## 6. What Must Change From V1 and V2

The required dataset change is:

```text
Fashion-MNIST -> CIFAR-10
```

This changes the data shape:

```text
Fashion-MNIST: 1 x 28 x 28 grayscale
CIFAR-10:      3 x 32 x 32 RGB/color
```

Because of this, V3 needs:

- a new CIFAR-10 data loader
- tests for `3 x 32 x 32` image batches
- channel-aware transforms and normalization decisions
- a model strategy designed for color images
- model smoke tests for teacher and student output shape
- careful runtime planning before full training

The first V3 data loader should be conservative and understandable. It should prepare CIFAR-10 tensors correctly before any training starts. Random augmentation can be considered later, but only after the student approves whether it belongs in the controlled design.

## 7. Model Strategy

The V1/V2 simple CNN models should not be reused directly for CIFAR-10 because they expect grayscale `1 x 28 x 28` input.

Version 3 should use a modest CIFAR-10 teacher/student pair:

- teacher: larger than the student, but still small enough to train locally
- student: smaller model used for both hard-label baseline and KD
- both models: accept `3 x 32 x 32` images and output 10 class logits

The first model design should be simple enough for the student to understand. A reasonable direction is a small convolutional network with pooling, not a large ResNet-style model. A larger pretrained model or advanced architecture should not be introduced unless the student approves a later scope change.

Teacher accuracy matters because KD depends on useful soft targets. If the teacher is too weak, a weak KD result may reflect the teacher setup, not a general failure of distillation.

Model decision checkpoint:

```text
Before writing V3 model code, the student should approve whether to:
1. adapt the simple CNN style into a CIFAR-specific CNN, or
2. create a modest new CIFAR CNN teacher/student pair.
```

The recommended first choice is option 2: create a modest CIFAR CNN teacher/student pair.

## 8. Main Comparison

The main Version 3 comparison is:

```text
same CIFAR-10 student trained with hard labels
vs
same CIFAR-10 student trained with teacher soft targets / KD
```

The comparison should use matched settings as much as practical:

- same dataset
- same train/test split
- same student architecture
- same batch size
- same learning rate
- same number of epochs
- same seed for each controlled pair
- same evaluation code
- same device type when practical

The teacher is a reference model and source of soft targets. The main claim should not be "student beats teacher." The main comparison is student hard-label vs student KD.

The first controlled pair should answer a narrow question:

```text
Under the same CIFAR-10 student setup and same seed, does KD perform differently from hard-label training?
```

## 9. Initial Experiment Design

Version 3 should not start with a large hyperparameter search.

The initial controlled design should likely use:

- dataset: CIFAR-10
- device: CPU first, unless the student confirms a GPU setup
- seed: 0 for first controlled pair
- repeat seed: 1 for reliability check
- primary metric: test accuracy
- baseline: hard-label student with the same architecture
- KD setting: one planned temperature/alpha pair approved before training
- epochs: more than V1/V2 may be needed, but final number should be approved after runtime estimate

V1 and V2 suggest that temperature matters. V3 can start from a planned KD setting inspired by earlier runs, but any temperature/alpha choice should be documented before the run.

Possible first KD setting for discussion:

```text
temperature = 4.0 or 6.0
alpha = 0.7
```

This is not approved yet. It is a discussion candidate because V1 and V2 showed that higher temperature sometimes helped, but CIFAR-10 may behave differently.

## 10. Success Criteria

Version 3 is successful if it produces a clear, honest controlled comparison on CIFAR-10.

Positive KD improvement is not required for success.

Minimum success criteria:

- CIFAR-10 data loader works and is tested.
- Data smoke test confirms expected dataset size, image shape, label shape, dtype, and value range.
- CIFAR-10 teacher and student models pass shape smoke tests.
- Teacher training script runs and saves a fixed checkpoint.
- Hard-label student baseline trains and evaluates.
- KD student trains against the fixed teacher and evaluates.
- Same-seed hard-label vs KD comparisons are recorded.
- At least one important comparison is repeated with seed 1.
- Raw result CSV files are saved for real controlled runs.
- Results are summarized without exaggeration.

Research success means the project can answer:

```text
What happened when the V1/V2 KD pipeline was extended to CIFAR-10 under controlled conditions?
```

Possible valid outcomes:

- KD clearly improves over hard-label training.
- KD is similar to hard-label training.
- KD performs worse than hard-label training.
- KD only helps under some temperature/alpha settings.
- Results are inconclusive because the teacher or training budget is too weak.

All five outcomes are useful if they are recorded carefully.

## 11. Limitations / What Not To Claim

Version 3 should not claim:

- Knowledge distillation always improves student performance.
- CIFAR-10 proves the full Hinton et al. paper.
- One seed proves a general effect.
- The best test accuracy from exploratory checking is a confirmed result.
- A weak teacher is enough evidence that KD itself fails.
- A small local CNN is comparable to modern CIFAR-10 state-of-the-art systems.

Important limitations:

- CIFAR-10 is harder than MNIST and Fashion-MNIST.
- The first V3 teacher may not be strong enough.
- CPU training may limit epochs and model size.
- Temperature and alpha can strongly affect KD results.
- Repeated seeds are needed before making stronger claims.
- Test-set tuning must be limited and described as exploratory when it happens.

The careful Version 3 writing style should be:

```text
In this controlled CIFAR-10 reproduction setting, KD showed / did not show a stable improvement over the same student trained with hard labels.
```

Not:

```text
KD is always better for CIFAR-10.
```

## 12. Planned Experiment Order

Do not jump directly to full KD training. The planned order is:

1. Approve this V3 scope document.
2. Add a CIFAR-10 data loader.
3. Add data loader tests.
4. Run a data smoke test and record it if it produces numeric output.
5. Decide the model strategy: adapted simple CNN style or modest new CIFAR CNN.
6. Add teacher and student model definitions.
7. Add model shape tests.
8. Add CIFAR-10 teacher, hard-label student, and KD scripts, or carefully generalize existing training code.
9. Run one-batch smoke tests and record all numeric outputs.
10. Train the CIFAR-10 teacher and save a fixed checkpoint.
11. Run the hard-label student baseline.
12. Run the KD student controlled comparison.
13. Repeat important comparisons with at least one additional seed.
14. Record all results in `docs/CONTROLLED_EXPERIMENT_LOG.md` and `results/raw/`.
15. Write `docs/v3_cifar10/CIFAR10_V3_RESULTS_SUMMARY.md`.
16. Update `README.md` only after Version 3 is complete.

## 13. Suggested 12-13 Day Time Plan

The user has about 12-13 days. This is realistic if V3 stays focused and avoids endless tuning.

| Day | Focus | Notes |
|---|---|---|
| Day 1 | Scope approval | Decide what V3 is allowed to claim and not claim. |
| Day 2 | CIFAR-10 data loader | Add loader only after scope approval. |
| Day 3 | Data tests and data smoke test | Confirm `3 x 32 x 32` shape before model work. |
| Day 4 | Model strategy | Decide teacher/student architecture. |
| Day 5 | Model files and shape tests | No full training yet. |
| Day 6 | Training scripts | Teacher, hard-label student, and KD script paths. |
| Day 7 | One-batch smoke tests | Check code paths, not performance. |
| Day 8 | Teacher seed 0 training | Save fixed checkpoint. |
| Day 9 | Hard-label student seed 0 | First real baseline. |
| Day 10 | KD student seed 0 | First real KD comparison. |
| Day 11 | Seed 1 reliability check | Repeat important pair. |
| Day 12 | Result table and V3 summary | Student interprets first, Codex polishes. |
| Day 13 | README/GitHub packaging buffer | Only after V3 result story is approved. |

Practical estimate:

```text
2-3 focused hours per day -> possible in 12-13 days
1 hour per day -> risky, especially if CPU training is slow
large model or hyperparameter search -> likely too much for 12-13 days
```

This schedule is a planning estimate, not a research result. Debugging, dataset download time, CPU speed, model size, and careful logging may change the actual time.

## 14. Approval Gate

Before implementation, the student should confirm:

- CIFAR-10 is the correct V3 dataset.
- The V3 research question is correct.
- V3 should use separate `v3_cifar10` folders.
- The first model strategy should be a modest CIFAR CNN teacher/student pair.
- The main comparison is same student hard-label vs same student KD.
- Positive KD improvement is not required for V3 success.
- Full training will not start until the student approves the exact run settings.

Codex should not begin CIFAR-10 implementation or training until this scope is approved.
