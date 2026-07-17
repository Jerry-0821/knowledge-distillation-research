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

Recommended future route:

```text
V2: Fashion-MNIST short bridge.
V3: CIFAR-10 stronger portfolio stretch.
```

## Response Header

For this project, Codex replies should start with a short pipeline/progress header so the student does not lose the main thread.

Use this shape:

```markdown
## Pipeline / Progress

1. MNIST V1 — [status]
2. GitHub packaging — [status]
3. V2 Fashion-MNIST — [status]
4. V3 CIFAR-10 — [status]

Current position: ...
Now doing: ...
Not doing yet: ...
```

Keep the header concise.

## Public Repo Guidance

For GitHub, prefer polished project-facing files:

- `README.md`
- `START_HERE.md`
- `docs/MNIST_V1_PROJECT_SUMMARY_FINAL.md`
- `docs/MNIST_V1_RESULTS_SUMMARY.md`
- `results/tables/mnist_v1_result_summary.csv`

Keep local scratch work private unless the student explicitly wants it published:

- `00_STUDENT_WORK/`
- `homework/`

Datasets, checkpoints, and raw run files should remain local unless there is a specific publishing reason and the student confirms it.
