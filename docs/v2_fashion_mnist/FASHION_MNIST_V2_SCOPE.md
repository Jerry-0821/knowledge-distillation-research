# Fashion-MNIST Knowledge Distillation: Version 2 Scope

This file defines the planned scope for Version 2 of the knowledge distillation reproduction project.

Current stage:

```text
V2 scope approved.
Fashion-MNIST V2 implementation and results summary are complete.
See docs/v2_fashion_mnist/FASHION_MNIST_V2_RESULTS_SUMMARY.md for results.
```

Version 2 began only after this scope was reviewed and approved.

## 1. Learning and Portfolio Context

This project is the student's first paper reproduction project. The purpose is not only to produce code, but also to show research skills for professors and future resume/portfolio use.

Version 2 should therefore be taught step by step. The student should understand:

- why Fashion-MNIST is being used after MNIST
- what makes a controlled comparison fair
- why the same student model must be compared against itself under different training objectives
- how teacher training, hard-label baseline training, and KD training fit together
- why smoke tests are not final results
- how to record negative, weak, or confusing outcomes honestly

Codex should help with scaffolding, code edits, verification, experiment logging, and writing polish. The student should keep ownership of the research judgment, result interpretation, and portfolio-facing story.

Before each V2 step, use this responsibility checkpoint:

```text
Student should do:
- understand the concept or research reason behind the step
- decide whether the comparison is fair
- explain important observations in their own words
- choose what belongs in the portfolio-facing story

Codex can help:
- create or edit files after explaining the purpose
- run tests and smoke checks
- record outputs exactly as produced
- keep bad or weak results visible
- polish wording without changing the student's meaning
```

Important V2 learning boundary:

- The student should review important code, especially loaders, training scripts,
  losses, and result-writing logic.
- Codex may scaffold files and run tests/smoke tests, but full training requires
  student approval first.
- Before full training, confirm the comparison, seed, temperature, alpha, epochs,
  teacher checkpoint, and success criteria.
- After results are produced, Codex should show the numbers and ask the student
  to interpret them before writing final claims.
- README and final summary claims should start from the student's draft. Codex
  can polish but should not add stronger claims silently.

## 2. Why Fashion-MNIST After MNIST

MNIST Version 1 completed the full teacher-student pipeline:

```text
teacher model -> soft targets -> student model
```

The main Version 1 result was careful and limited. Knowledge distillation did not automatically improve every setting. The hard-label student already performed strongly on MNIST, so the improvement from KD was small and depended on temperature.

Fashion-MNIST is a good next dataset because it is harder than MNIST but keeps the same basic input and output structure:

```text
input shape = 1 x 28 x 28
number of classes = 10
task type = image classification
```

This makes Fashion-MNIST a useful bridge. It changes the dataset difficulty without immediately changing the whole computer vision setup. That helps isolate the research question:

```text
Does knowledge distillation show a clearer improvement over hard-label training when the dataset is more difficult than MNIST?
```

## 3. V2 File Organization

To avoid mixing MNIST V1 and Fashion-MNIST V2, V2-facing files should live in V2-specific folders:

```text
docs/v2_fashion_mnist/
scripts/v2_fashion_mnist/
tests/v2_fashion_mnist/
results/tables/v2_fashion_mnist/
```

Reusable package code should stay in the normal package folders:

```text
src/kd_research/data/fashion_mnist.py
```

Reason: documentation, scripts, tests, and final tables are easier to find when they are grouped by project version. But shared Python package code should stay organized by function, so future scripts can import it cleanly.

Do not move MNIST V1 result files while starting V2. V1 should remain reproducible.

## 4. What Stays the Same From V1

Version 2 should preserve the main controlled comparison from Version 1:

- Train a teacher model with hard labels.
- Freeze the trained teacher checkpoint.
- Train the student model with hard labels only.
- Train the same student model with knowledge distillation.
- Compare the hard-label student and KD student under matched settings.
- Keep the same dataset split type: official training split for training and official test split for evaluation.
- Use fixed seeds for controlled comparisons.
- Record every numeric run in `docs/CONTROLLED_EXPERIMENT_LOG.md`.
- Store real raw run records under `results/raw/`.
- Keep datasets and checkpoints local and ignored by Git.
- Report negative or weak results honestly.
- Avoid unlimited test-set tuning.

The most important fairness rule remains:

```text
hard-label student and KD student must use the same student architecture.
```

If the student architecture changes later, both hard-label and KD runs must use the changed architecture before making a comparison.

## 5. What Changes From V1

The required change is the dataset:

```text
MNIST -> Fashion-MNIST
```

Fashion-MNIST images are still grayscale 28x28 images, but the classes are clothing items instead of handwritten digits. The task is harder because the class boundaries are less visually simple.

The first implementation should be conservative:

- Add a Fashion-MNIST data loader instead of replacing the MNIST loader.
- Add tests for Fashion-MNIST data shape and loader validation.
- Reuse the V1 simple CNN teacher and student models first, because the image shape and class count are the same.
- Only consider changing the models after the first Fashion-MNIST teacher/baseline/KD path has been tested and logged.
- Keep V1 MNIST files intact so Version 1 remains reproducible.

Possible later changes, only after the first V2 pass:

- More epochs if one epoch is too weak for Fashion-MNIST.
- A modest architecture adjustment if the teacher is not strong enough.
- A small planned temperature/alpha check, based on V1 evidence and not random test-set searching.
- Repeated seeds for any result that looks important.

## 6. Main Comparison

The main Version 2 comparison is:

```text
same student trained with hard labels
vs
same student trained with teacher soft targets / KD
```

The comparison should use matched settings as much as possible:

- same dataset
- same train/test split
- same student architecture
- same batch size
- same learning rate
- same number of epochs
- same seed for each controlled pair
- same evaluation code
- same device type when practical

The first controlled pair should answer a narrow question:

```text
Under the same seed and training setup, does the KD student perform differently from the hard-label student on Fashion-MNIST?
```

## 7. Success Criteria

Version 2 is successful if it produces a clear, honest controlled comparison on Fashion-MNIST.

Positive KD improvement is not required for success.

Minimum success criteria:

- Fashion-MNIST data loader works and is tested.
- The teacher model trains and is evaluated.
- The hard-label student baseline trains and is evaluated.
- The KD student trains and is evaluated against the fixed teacher.
- Same-seed hard-label vs KD comparisons are recorded.
- Raw result CSV files are saved for real controlled runs.
- Results are summarized without exaggeration.

Research success means the project can answer:

```text
Does the Fashion-MNIST setting make the KD effect clearer than the MNIST setting?
```

Possible valid outcomes:

- KD clearly improves over hard-label training.
- KD is similar to hard-label training.
- KD performs worse than hard-label training.
- KD only helps under some temperature/alpha settings.

All four outcomes are useful if they are recorded carefully.

## 8. Limitations / What Not To Claim

Version 2 should not claim:

- Knowledge distillation always improves student performance.
- Fashion-MNIST proves the full Hinton et al. paper.
- One seed proves a general effect.
- The best test accuracy from a small exploratory search is a confirmed result.
- A weak teacher is enough evidence that KD itself fails.
- A result from Fashion-MNIST automatically transfers to CIFAR-10 or real-world vision tasks.

Important limitations:

- Fashion-MNIST is harder than MNIST, but it is still a small benchmark.
- The first V2 model setup may be too simple.
- One epoch may be too short for a stable conclusion.
- Temperature and alpha can strongly affect KD results.
- Repeated seeds are needed before making stronger claims.
- Test-set tuning must be limited and described as exploratory when it happens.

The careful Version 2 writing style should be:

```text
In this reproduction setting, KD showed / did not show a clearer improvement on Fashion-MNIST under the tested conditions.
```

Not:

```text
KD is always better.
```

## 9. Planned Experiment Order

Do not jump directly to full KD training. The planned order is:

1. Approve this V2 scope document.
2. Add a Fashion-MNIST data loader.
3. Add data loader tests.
4. Run a data smoke test and record it if it produces numeric output.
5. Decide whether to reuse the V1 CNN teacher and student models for the first pass.
6. Add Fashion-MNIST teacher, hard-label student, and KD scripts, or carefully generalize the existing MNIST scripts.
7. Run one-batch smoke tests and record all numeric outputs.
8. Train the Fashion-MNIST teacher and save a fixed checkpoint.
9. Run the hard-label student baseline.
10. Run the KD student controlled comparison.
11. Repeat important comparisons with at least one additional seed if the first result looks meaningful.
12. Record all results in `docs/CONTROLLED_EXPERIMENT_LOG.md` and `results/raw/`.
13. Write `docs/v2_fashion_mnist/FASHION_MNIST_V2_RESULTS_SUMMARY.md`.
14. Update `README.md` only after Version 2 is complete.

## 10. Suggested Time Plan

The expected working timeline is:

| Day | Focus | Approximate time | Notes |
|---|---|---:|---|
| Day 1 | Scope, Fashion-MNIST data loader, data test | 2 hours | Start with the scope document, then implement the dataset loader and smoke test only after approval. |
| Day 2 | Training scripts and smoke tests | 2 hours | Adapt or carefully generalize MNIST scripts, then run one-batch smoke tests without treating the numbers as final results. |
| Day 3 | Teacher, hard-label baseline, KD controlled run | 2-3 hours | Train the teacher, then run seed 0 hard-label baseline and KD comparison. |
| Day 4 | Seed 1, result summary, README update | 2 hours | Run seed 1, write result CSV and V2 summary, then update README only after V2 is complete. |

Practical estimate:

```text
2 hours per day -> about 4 days
3-4 hours per day -> about 2.5 to 3 days
1 hour per day -> about 5 to 7 days
```

This schedule is a planning estimate, not a research result. Debugging, dataset download time, CPU speed, and careful logging may change the actual time.

## 11. Approval Gate

Before implementation, the student should confirm:

- The V2 research question is correct.
- Fashion-MNIST is the right next dataset.
- The first comparison should reuse the same student model before trying larger changes.
- The planned order is small enough to understand and verify step by step.

Codex should not begin Fashion-MNIST implementation or training until this scope is approved.
