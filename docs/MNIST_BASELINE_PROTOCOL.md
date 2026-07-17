# MNIST Baseline Protocol

This file plans the first baseline experiment before writing or running training code.

## Experiment ID

mnist_student_baseline_001

## Goal

Train a small student CNN on MNIST using hard labels only. The goal is to confirm that dataset loading, model forward pass, training loop, and evaluation pipeline work before starting Knowledge Distillation.

## Why this experiment comes first

This experiment comes before Knowledge Distillation because we need a normal supervised baseline first.

If the student model cannot learn from hard labels, then it does not make sense to compare it with a distilled student.

This baseline will later be compared with the same student model trained using teacher soft targets.

## Hypothesis

The student model should learn better than random guessing on MNIST.

The training loss should decrease, and the test accuracy should become clearly higher than 10%.

This experiment does not try to get the best possible MNIST accuracy. It only checks whether the basic pipeline works.

## Dataset

- Dataset: MNIST
- Train split: MNIST training set
- Test split: MNIST test set
- Notes: Dataset should be downloaded only when implementation begins. Dataset files should not be committed to Git.

## Model

- Model name: small student CNN
- Model type: simple convolutional neural network
- Why this model is small enough for a student baseline: It should be smaller than the future teacher model and simple enough to understand.

## Training setup

- Label type: hard labels only
- Loss function: cross entropy
- Optimizer: to be decided before implementation
- Epochs: to be decided before implementation
- Batch size: to be decided before implementation
- Device: CPU first; GPU only if confirmed safe and available

## Metrics

- Training loss
- Test accuracy
- Parameter count
- Model size
- Simple inference latency

## Controlled variables

When comparing this baseline with distillation later, these should stay the same as much as possible:

- Same dataset
- Same train/test split
- Same student model architecture
- Same evaluation code
- Same metrics
- Same device type when measuring latency
- Same random seed if possible

## Success criteria

This baseline is working if:

- The code runs without crashing.
- The model output shape is correct.
- The training loss decreases.
- Test accuracy is clearly higher than random guessing.
- Results can be recorded in the experiment log.

## Failure cases

Possible failures:

- Dataset does not load correctly.
- Tensor shapes are wrong.
- Model output shape is not `[batch_size, 10]`.
- Loss does not decrease.
- Accuracy stays near random guessing.
- Training or evaluation script crashes.
- Evaluation accidentally runs in training mode.
- Results are not recorded clearly.

## Stopping rule

For the first run, stop after a small number of epochs.

The purpose is to check that the pipeline works, not to optimize performance.

Do not run long training until the baseline code is verified.

## Result

Leave blank until the experiment has actually been run.

## Interpretation

Leave blank until the experiment has actually been run.
