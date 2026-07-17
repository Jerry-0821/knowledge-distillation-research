# MNIST Knowledge Distillation Reproduction: Version 1 Final Summary

Key result table:

[MNIST V1 result summary CSV](../results/tables/mnist_v1_result_summary.csv)

Supporting result summary:

[MNIST V1 results summary](MNIST_V1_RESULTS_SUMMARY.md)

## 1. Goal

My goal in Version 1 was to reproduce the core idea of knowledge distillation by comparing a student CNN trained with teacher soft labels against the same student CNN trained with normal hard labels. I did not expect a large improvement, but I wanted to see whether distillation could produce at least a small difference.

I chose MNIST first because this is my first time reproducing a paper, and MNIST is easier to use and well supported in PyTorch. It allowed me to experience the full teacher-student pipeline without spending too much time debugging data processing. However, MNIST is also simple, and many tools are already available for it. This also means that the hard-label baseline can already achieve strong results, making the difference from distillation harder to see.

The main question I wanted to test was whether a student CNN trained with teacher soft targets could perform better than the same student trained with hard labels, while also using a smaller model with lower computational cost and memory than the teacher.

## 2. Method

The project used a teacher-student setup.

The teacher model used more convolution channels and one extra convolution layer. This made it stronger than the student model, so I used it as the teacher model.

The student model was a smaller CNN with one convolution layer and fewer channels.

The hard-label baseline trained the student using Adam and cross-entropy loss, because I wanted to know whether the student could perform well without distillation.

The distillation experiment trained the same student using a combined loss. The loss included cross-entropy with the true labels and a distillation loss that compared the softened teacher output distribution with the softened student output distribution.

The combined loss was:

```text
total_loss = alpha * hard_label_loss + (1 - alpha) * distillation_loss
```

In this setup, temperature controls how soft the probability distribution is. A higher temperature makes the teacher probabilities less sharp, so the student can learn more information about relationships between classes.

Alpha controls the balance between hard-label loss and distillation loss. A higher alpha gives more weight to the true labels, while a lower alpha gives more weight to the teacher soft targets.

## 3. Experimental Setup

Dataset:

MNIST

Model setup:

Teacher model: a larger CNN with two convolution layers and more channels.

Student model: a smaller CNN with one convolution layer and fewer channels.

The hard-label baseline and the distillation experiment used the same student model.

Teacher checkpoint:

`checkpoints/mnist_teacher_baseline_001.pt`

This checkpoint was trained with hard labels only and then fixed during student distillation.

Training settings:

```text
epochs = 1
batch size = 64
learning rate = 0.001
device = CPU
```

Controlled comparison settings:

```text
seed = 0 and 1
main temperature = 4.0
main alpha = 0.7
exploratory best setting = temperature 6.0, alpha 0.65, seed 0
```

I used the same seed because I wanted the hard-label and distillation runs to start from comparable random initialization and data shuffle, so the comparison focused more on the training objective.

## 4. Key Results Table

Use the CSV table here:

[MNIST V1 result summary CSV](../results/tables/mnist_v1_result_summary.csv)

| Experiment | Seed | Temperature | Alpha | Test Accuracy | Notes |
|---|---:|---:|---:|---:|---|
| Hard-label student | 0 | N/A | N/A | 0.9547 | baseline |
| KD student | 0 | 2.0 | 0.7 | 0.9537 | did not improve |
| KD student | 0 | 4.0 | 0.7 | 0.9599 | improved over baseline |
| Hard-label student | 1 | N/A | N/A | 0.9517 | baseline |
| KD student | 1 | 4.0 | 0.7 | 0.9526 | small improvement |
| KD student exploratory | 0 | 6.0 | 0.65 | 0.9609 | best observed seed-0 result |

## 5. Main Finding

The main finding was that knowledge distillation can help the student model, but only under suitable settings and hyperparameters.

KD did not automatically improve every setting. When I used temperature 2.0, the model trained with soft targets performed worse than the same student model trained with hard labels.

Temperature seemed important because it had a bigger impact on the result. Higher temperature settings produced better KD results.

The best observed result was the exploratory setting with temperature 6.0 and alpha 0.65 on seed 0. It reached 0.9609 test accuracy, compared with the hard-label seed-0 baseline of 0.9547.

However, I should describe this carefully because the best setting was exploratory and was only tested on seed 0. The improvement is promising, but it is still small and should not be claimed as a strong general result yet.

## 6. Limitations

This experiment has several limitations.

First, MNIST is a simple dataset, so the hard-label student baseline already achieved high accuracy. That is why it is difficult to see a large improvement from knowledge distillation.

Second, both the teacher and student models are still simple CNN models. A stronger teacher or a different student architecture might change the distillation effect.

Third, I only tested a small number of hyperparameter settings. A more systematic hyperparameter search could test more values of temperature and alpha, but it should be planned carefully and not just tuned on the test set.

The biggest limitation is that MNIST may be too simple for showing a large distillation improvement. This is why testing a harder dataset may make the effect easier to observe.

I should not claim that knowledge distillation always improves student performance.

## 7. Future Work: Fashion-MNIST and CIFAR-10

For Version 2, I want to test Fashion-MNIST because it is more difficult than MNIST. MNIST may be too simple, so the hard-label student already performs strongly and leaves only a small room for improvement.

Fashion-MNIST is useful as a next step because it keeps the same image classification structure as MNIST, but the classes are harder to separate. This may make the effect of knowledge distillation easier to observe.

In Version 2, I would keep the same teacher-student comparison: one student trained with hard labels only, and the same student trained with knowledge distillation.

I would change the dataset from MNIST to Fashion-MNIST. I would also run a more controlled comparison with more seeds and a clearer temperature/alpha search.

The question for Version 2 would be: Does knowledge distillation show a clearer improvement over hard-label training when the dataset is more difficult?

After Fashion-MNIST, a stronger Version 3 extension could test CIFAR-10. This would be closer to a real computer vision task, but it would require more model and training adjustments than Fashion-MNIST.
