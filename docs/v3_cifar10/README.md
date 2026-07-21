# CIFAR-10 V3 Folder

This folder keeps CIFAR-10 Version 3 documentation separate from MNIST Version 1 and Fashion-MNIST Version 2.

Current V3 status: CIFAR-10 implementation, tests, training scripts,
controlled runs, final test evaluation, results summary, README updates, and
GitHub packaging are complete.

Use this folder for:

- V3 scope and planning
- V3 result summary, after experiments are complete
- V3 interpretation notes
- V3 limitations and portfolio-facing wording
- V3 collaboration rules and approval checkpoints

Main files:

- `CIFAR10_V3_SCOPE.md`
- `CIFAR10_V3_ENHANCEMENTS.md`
- `CIFAR10_V3_RESULTS_SUMMARY.md`
- `V3_RESEARCH_REFLECTION.md`

Related V3 locations:

```text
scripts/v3_cifar10/
tests/v3_cifar10/
results/tables/v3_cifar10/
src/kd_research/data/cifar10.py
src/kd_research/models/
```

The `src/` files are shared package code, so they should stay outside this docs folder. The V3 docs, scripts, tests, and final tables should stay grouped by version to avoid mixing V1, V2, and V3 work.

## V3 Collaboration Rules

At the start of each V3 step, separate responsibilities clearly:

```text
Student should do:
- understand the research reason for the step
- review or type new V3-specific code when it introduces a new idea
- approve new dataset/model/training design choices
- interpret important results before final claims are written

Codex can help:
- explain the purpose of each file
- provide code for the student to type and understand
- directly handle repeated V1/V2-style boilerplate when it does not introduce a new research idea
- check the saved version before moving to the next step
- run tests or smoke checks after approval
- record numeric outputs exactly when experiments begin
```

New V3-specific code should not be silently completed first. Codex should show the proposed code and explain it so the student can type or review it.

For experiment-like commands, prefer this workflow:

```text
1. Codex gives the exact command and explains why it is being run.
2. Student runs the command locally to experience the project workflow.
3. Student sends the important terminal output back to Codex.
4. Codex checks, records, compares, and explains the result.
```

After Codex checks a saved file and it is correct, Codex can immediately give the next file or command. The student does not need to ask "next" every time.

For large new learning pieces, prefer a notebook walkthrough first. For small, mechanical, or repeated project files, a `.py` file is acceptable.

Every time the student edits, types, or saves an important file, Codex should check the saved version before giving the next step.
