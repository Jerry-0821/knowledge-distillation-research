# Rules for Future Codex Sessions

This repository is a learning-oriented paper reproduction project. The student owns the research judgment. Codex helps with engineering, verification, logging, and polishing.

## Required Experiment Recording

Before any training, evaluation, smoke test, controlled comparison, or hyperparameter run, read and follow:

```text
docs/EXPERIMENT_RECORDING_RULES.md
```

Every run with numeric outputs must be recorded in:

```text
docs/CONTROLLED_EXPERIMENT_LOG.md
```

Real experiment/control runs should also be recorded under:

```text
results/raw/
```

Negative, failed, or disappointing results must still be recorded.

## Safety Rules

1. Do not fake, infer, or invent experiment results.
2. Do not hide failing tests, warnings, failed commands, or weak results.
3. Do not delete experiment logs, failure logs, or raw result files to make the project look better.
4. Do not change the research question, main hypothesis, or evaluation standard without the student confirming it.
5. Do not describe exploratory hyperparameter tuning as a confirmed general result.
6. Do not upload secrets, datasets, or checkpoints.
7. Do not run long training jobs or download datasets without confirmation.
8. Keep paper claims separate from this project's observed results.

## Student vs Codex Responsibility

Before each learning, experiment, code, or writing step, Codex should pause and
separate responsibilities:

```text
Student should do:
- understand the research reason for the step
- make or confirm the research judgment
- interpret important results in their own words

Codex can help:
- scaffold files and scripts
- run verification after explaining what is being checked
- record outputs exactly
- polish writing without changing the student's claim
```

This split is part of the teaching method. Do not treat the project as only a
coding task.

## Responsibility Rule

The student is using this project to learn the paper reproduction process. Codex
must not silently make all research decisions.

Student should do:

- Decide the research question. For example, the student should confirm whether
  the V2 question is: "Does KD help more on Fashion-MNIST than MNIST?"
- Explain concepts in their own words, such as temperature, alpha, hard loss,
  soft loss, and why Fashion-MNIST is harder.
- Approve experiment design before full training. Before running full training,
  ask the student to confirm the comparison, seed, temperature, alpha, epochs,
  and success criteria.
- Interpret results first. After a run finishes, show the numbers and ask what
  the student thinks they mean before writing the final conclusion.
- Write the first draft of portfolio-facing text. For README, final summary,
  limitations, and main findings, the student drafts first. Codex can polish
  grammar and clarity but should not add new claims unless it clearly labels
  them.

Codex can do:

- Inspect repo files and explain the current structure.
- Create scaffolding files after explaining what they are for.
- Edit code after the student approves the plan.
- Run tests and smoke tests.
- Record experiment outputs exactly into logs or CSV files.
- Fix mechanical bugs.
- Polish grammar without changing the student's meaning.
- Prepare GitHub packaging after the student approves the final story.

Important boundary:

- Do not run full training or hyperparameter search without student
  confirmation.
- Do not write final research claims before the student gives an interpretation.
- Do not claim KD is better just because one run is better.
- Do not hide weak or failed results.

Student should do or explicitly confirm:

- Research questions and claims.
- First-time conceptual explanations.
- Model/loss/training-loop understanding.
- Experiment interpretation.
- Portfolio-facing writing.
- Decisions about what to publish on GitHub.

Codex can help with:

- Mechanical edits and formatting.
- Verification commands.
- Small scripts and scaffolding.
- Result logging exactly as produced.
- Grammar polishing without changing the student's meaning.
- GitHub packaging after the student approves the story.

## Learning Pipeline

Use this pipeline unless the student explicitly changes the plan:

1. Paper first reading.
2. Reproduction scope.
3. Data pipeline.
4. Model smoke test.
5. Hard-label baseline.
6. Teacher model.
7. Distillation experiment.
8. Controlled comparison and analysis.
9. Portfolio/GitHub packaging.
10. Version 2 extension.

Current Version 1 status:

```text
MNIST V1 complete.
```

Current Version 2 status:

```text
Fashion-MNIST V2 complete.
```

Current Version 3 status:

```text
CIFAR-10 V3 complete. README/GitHub packaging should preserve the cautious
conclusion: KD helped at 10 epochs but did not beat the hard-label baseline at
20/70 epochs or on the final seed-0 official test evaluation.
```

Recommended future route:

```text
Package the completed V1/V2/V3 project and keep future extensions separate.
```

## Response Header

For this project, Codex replies should start with a short pipeline/progress header so the student does not lose the main thread.

During active Fashion-MNIST V2 work, use this V2-only shape with emoji status markers:

```markdown
## V2 Pipeline / Progress

1. V2 scope — ✅ done
2. Fashion-MNIST data loader — ✅ done
3. Data loader tests — ✅ done
4. Data smoke test — ✅ done
5. Training scripts — 🟡 current
6. One-batch smoke tests — ⬜ not yet
7. Teacher training — ⬜ not yet
8. Hard-label baseline — ⬜ not yet
9. KD controlled run — ⬜ not yet
10. V2 results summary — ⬜ not yet
11. README update — ⬜ not yet

Current position: ...
Now doing: ...
Not doing yet: ...
```

Use ✅ for done, 🟡 for current/next, and ⬜ for not yet. Keep the header concise.

After V2 packaging is complete, do not restart V2 unless the student explicitly
asks. Future work should normally move toward V3 CIFAR-10 planning.

## Public Repo Guidance

For GitHub, prefer polished project-facing files:

- `README.md`
- `START_HERE.md`
- `docs/MNIST_V1_PROJECT_SUMMARY_FINAL.md`
- `docs/MNIST_V1_RESULTS_SUMMARY.md`
- `results/tables/mnist_v1_result_summary.csv`
- `docs/v2_fashion_mnist/FASHION_MNIST_V2_SCOPE.md`
- `docs/v2_fashion_mnist/FASHION_MNIST_V2_RESULTS_SUMMARY.md`
- `results/tables/v2_fashion_mnist/fashion_mnist_v2_result_summary.csv`
- `docs/v3_cifar10/CIFAR10_V3_SCOPE.md`
- `docs/v3_cifar10/CIFAR10_V3_RESULTS_SUMMARY.md`
- `results/tables/v3_cifar10/cifar10_v3_result_summary.csv`
- `results/figures/v3_cifar10/`

Keep local scratch work private unless the student explicitly wants it published:

- `00_STUDENT_WORK/`
- `homework/`

Datasets, checkpoints, and raw run files should remain local unless there is a specific publishing reason and the student confirms it.
