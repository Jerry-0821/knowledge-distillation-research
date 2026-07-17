# Reproduction Scope

This file defines the first version of this Knowledge Distillation reproduction project.

Current stage:

```text
Stage 2: define reproduction scope
```

No dataset has been downloaded yet. No model has been trained yet. No result has been produced yet.

## Version 1 Goal

The first version will reproduce the core idea of Knowledge Distillation:

```text
teacher model -> soft targets -> student model
```

The main comparison will be:

```text
student trained with hard labels
vs
student trained with distillation
```

## Required Scope

Version 1 should include:

- Use MNIST as the first sanity-check dataset.
- Train a teacher model.
- Train a smaller student model with hard labels.
- Train the same student model with teacher soft targets.
- Compare hard-label student and distilled student.
- Record accuracy, parameter count, model size, and simple inference latency.
- Save configs, commands, logs, and notes clearly.

## Optional Stretch Scope

If MNIST is complete and time remains, extend the same pipeline to CIFAR-10.

CIFAR-10 is optional. It is for portfolio value, not required for the first successful reproduction.

## Out of Scope for Version 1

Version 1 will not reproduce:

- Speech recognition experiments.
- Specialist ensemble models.
- Mixture of Experts.
- Commercial-scale systems.
- All experiments from the original paper.
- Large hyperparameter search.
- Long training runs.

## Success Criteria

Version 1 is successful if the project can clearly answer:

```text
Does a student trained with distillation perform differently from the same student trained only with hard labels?
```

The result does not have to be positive. If distillation does not improve performance, the project should explain possible reasons.

## First Implementation Order

Do not start with Knowledge Distillation loss.

The implementation order should be:

1. MNIST dataset loading.
2. Simple CNN forward pass.
3. Hard-label student baseline.
4. Evaluation script.
5. Teacher model training.
6. Distillation training.
7. Controlled comparison.
8. Results table and failure analysis.

## Current Decision

Start with MNIST because it is small, fast, and good for learning the full research workflow.

Consider CIFAR-10 only after the MNIST pipeline is clean and reproducible.
