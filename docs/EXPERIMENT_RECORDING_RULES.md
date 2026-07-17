# Experiment Recording Rules

Codex must follow this file for every future experiment step in this repository.

## Mandatory Rule

Every training, evaluation, smoke test, controlled comparison, or hyperparameter run that produces numeric outputs must be recorded before moving to the next step.

The user should not need to remind Codex.

## What To Record

For each run, record:

- command
- purpose
- changed variables
- output numbers
- interpretation or meaning
- whether the run is smoke, full experiment, controlled comparison, failed run, or reference run

## Where To Record

Human-readable record:

```text
docs/CONTROLLED_EXPERIMENT_LOG.md
```

Machine-readable record:

```text
results/raw/
```

## Recording Policy

- Real experiment runs and controlled comparisons must get a CSV record under `results/raw/`.
- Smoke tests must be written in `docs/CONTROLLED_EXPERIMENT_LOG.md`; if they become important for debugging, also create a CSV record.
- Negative, worse, failed, or confusing results must be recorded.
- Do not delete or hide bad results.
- If a run changes seed, alpha, temperature, epochs, learning rate, batch size, teacher checkpoint, model architecture, dataset split, or run scope, record that change explicitly.
- If a code fix changes experiment fairness or interpretation, update the blueprint section immediately, not only the data log.
- Keep the data log readable: use a quick result table plus short details instead of long repeated blocks when many runs exist.
- Before interpreting results, verify that the latest numeric output has been recorded.

## Preferred Structure

Use this structure in `docs/CONTROLLED_EXPERIMENT_LOG.md`:

```text
Part 1: Blueprint Steps
Part 2: Data Log
Part 3: Analysis Notes, if needed
```
