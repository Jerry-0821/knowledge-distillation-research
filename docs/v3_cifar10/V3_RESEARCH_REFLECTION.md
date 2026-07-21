# CIFAR-10 V3 Research Reflection

## Purpose

This reflection summarizes what I learned from extending my knowledge
distillation reproduction project from MNIST and Fashion-MNIST to CIFAR-10.
V3 was not just a coding exercise. It was my first time experiencing how a
paper idea becomes a controlled experiment with scope decisions, model choices,
validation rules, repeated seeds, and cautious conclusions.

I used Codex as an AI coding and organization assistant, but the project was
not passive. Codex helped scaffold code, debug errors, organize logs, generate
figures, and polish writing. My role was to question the design, approve the
research direction, run and inspect important early commands, interpret the
results, and decide what the project should and should not claim.

## What V3 Was Trying To Test

The main V3 question was:

```text
Can the V1/V2 knowledge distillation pipeline be extended to CIFAR-10, and
under controlled settings does KD improve the same student model compared with
hard-label training?
```

This question mattered because CIFAR-10 is a bigger jump than Fashion-MNIST.
MNIST and Fashion-MNIST use `1 x 28 x 28` grayscale images, while CIFAR-10 uses
`3 x 32 x 32` RGB images with more complex visual patterns. Because of this, I
could reuse the research workflow, but not simply reuse every model or data
assumption.

The parts I reused were the controlled comparison structure: train a teacher
with hard labels, train a student with hard labels, train the same student with
KD, and compare the two student results under matched settings. The parts that
needed redesign were the CIFAR-10 data loader, the train/validation/test split,
the CIFAR-specific CNN teacher and student, GPU/runtime planning, and longer
training checks.

## Important Research Decisions

One important decision was separating validation from the official test set.
At first, the CIFAR-10 pipeline followed the simpler V1/V2 style more closely.
During the design discussion, I realized that using the official test set
repeatedly for learning-rate, temperature, alpha, or epoch decisions would be
unfair. It would be like preparing for an exam by repeatedly looking at the
final exam paper. We then used 45,000 CIFAR-10 training images for training,
5,000 for validation, and kept the official 10,000-image test set for final
evaluation only.

Another decision was keeping the model simple but CIFAR-specific. I questioned
whether the MNIST/Fashion-MNIST CNN was enough for RGB images, and approved a
new CNN design using convolution layers, BatchNorm, ReLU, MaxPool, Dropout, and
adaptive average pooling. I wanted the model to stay readable in a mostly
`nn.Sequential` style so that I could understand the architecture instead of
hiding the project inside a large model.

I also questioned preprocessing and runtime. `ToTensor()` converts images to
the `[0, 1]` range, but it does not do channel-wise normalization. We kept the
first V3 setup simple and recorded normalization as a future enhancement. For
runtime, I pushed to use GPU training because CIFAR-10 was much heavier than
MNIST/Fashion-MNIST. Batch size was kept fixed at 64 for controlled
comparisons.

The teacher learning-rate check also taught me that a small difference should
not be overinterpreted. LR `0.001` and `0.0005` were close after 10 epochs:
`0.001` reached validation accuracy `0.8104`, while `0.0005` reached `0.8138`.
I selected `0.001` as the default setting for consistency, not because it was
clearly superior.

## What The Results Taught Me

The selected KD setting was `temperature = 6.0` and `alpha = 0.5`. At 10
epochs, this setting looked promising. Across seeds 0, 1, and 2, KD improved
over the same hard-label student baseline:

```text
seed 0: +0.0176
seed 1: +0.0080
seed 2: +0.0166
```

This was the strongest positive V3 result because it was not only one seed.
However, I learned that an early result is not enough for a final claim. When
the same selected setting was extended to 20 epochs, the story changed. The
hard-label baseline caught up or became stronger for most seeds. At 70 epochs
on seed 0, the hard-label baseline was also stronger. The final official test
evaluation on seed 0 favored hard-label training at both 20 and 70 epochs.

This changed my conclusion. V3 does not show that KD is better overall on
CIFAR-10 under this setup. A more careful statement is that KD helped early
training in the 10-epoch validation comparison, but the advantage did not
persist under longer training or final test evaluation.

This negative/weak final result did not make the project a failure. It showed
that KD is sensitive to teacher strength, student capacity, temperature, alpha,
training length, preprocessing, and optimization choices. It also showed why a
reproduction project should record bad or mixed results instead of only keeping
the best-looking number.

## From Exploration To Verification

After choosing `T=6.0` and `alpha=0.5`, the work became more mechanical:
repeat the same comparison, change the seed or epoch count, record the result,
and compare it fairly. At first this felt less exciting than choosing the
model or tuning hyperparameters, but I realized this is part of research. Once
the design is fixed, the job becomes verification.

The seed repeats helped me see that one seed can be misleading. The longer
20/70-epoch runs helped me see that epoch budget can change the conclusion.
The final test-set rule helped me understand research hygiene. The plots also
helped because curves made it easier to see whether KD was consistently better
or only temporarily ahead.

## What I Learned

V3 helped me understand that paper reproduction is not only about implementing
the method. It is about asking whether the comparison is fair, deciding what
can be reused, changing what must be redesigned, and being honest when the
final result is weaker than the early result.

My main contributions were research judgment and learning ownership: deciding
that CIFAR-10 should be treated as a larger V3 step, requiring V1/V2/V3 results
to stay separate, questioning normalization, approving train/validation/test
separation, pushing for a CIFAR-specific CNN, using GPU for practical runtime,
asking for better metrics, extending the learning-rate check, limiting the KD
sweep instead of endlessly tuning, asking for seed repeats, using plots to
understand curves, and accepting the 20/70-epoch reversal without overclaiming.

The honest final summary is:

```text
KD with T=6.0 and alpha=0.5 helped the student in the 10-epoch validation
comparison, but the hard-label baseline caught up or became stronger when
training was extended to 20 and 70 epochs. The final seed-0 official test
evaluation also favored the hard-label baseline.
```

For a first reproduction project, this was useful because I learned how to move
from a paper idea to a controlled experiment, and why cautious conclusions are
important in research.
