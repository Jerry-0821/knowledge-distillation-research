# CIFAR-10 V3 Enhancement Notes

This file records useful V3 ideas that should not be silently added before the student understands or approves them.

## 1. CIFAR-10 Normalization

Current first-loader plan:

```text
Use transforms.ToTensor() only.
```

Reason:

```text
The first data smoke test should be simple. ToTensor converts CIFAR-10 images to
float tensors with shape [3, 32, 32] and values in the range [0, 1].
```

Important distinction:

```text
ToTensor() scales raw image values into [0, 1].
Normalize(mean, std) standardizes each channel after ToTensor().
```

Normalization is not automatic. It should be considered before real CIFAR-10 training because CIFAR-10 is an RGB dataset and channel-wise normalization can make CNN training more stable.

Future decision checkpoint:

```text
Before full CIFAR-10 training, decide whether to use:
1. ToTensor() only, for maximum simplicity and closer V1/V2 style, or
2. ToTensor() + CIFAR-10 channel normalization, for more standard CIFAR-10 training.
```

Do not add normalization silently. If it is added, record the transform choice in the V3 scope/results notes and keep hard-label student and KD student using the same transform.

## 2. Train / Validation / Test Split

Current first-loader plan:

```text
Use the official CIFAR-10 train split for training and the official CIFAR-10 test split for final evaluation.
```

Reason:

```text
This matches the simpler V1/V2 setup and keeps the first V3 data loader easy to understand.
```

Implemented loader support:

```text
Split the official CIFAR-10 training set into:
- train subset: 45,000 images
- validation subset: 5,000 images

Keep the official CIFAR-10 test set untouched for final reporting: 10,000 images.
The split is deterministic with split_seed = 0.
```

Why this may matter:

- A validation split is better for choosing temperature, alpha, epochs, or model settings.
- It helps avoid tuning directly on the test set.
- It makes the V3 experiment design more standard if we do more than one planned setting.

Current decision checkpoint:

```text
Before hyperparameter choices or repeated model selection, switch the V3 training
scripts to use validation accuracy for tuning and keep test accuracy for final
selected configurations only.
```

Do not tune on the official test set. The hard-label student and KD student must use the same train/validation split.

## 3. GPU Runtime and Batch Size Plan

Current environment check:

```text
NVIDIA GPU is present with about 6GB VRAM.
Student reports the GPU is likely an RTX 3060 6GB.
CUDA-enabled PyTorch was installed on 2026-07-20.
torch = 2.12.1+cu126
torchvision = 0.27.1+cu126
torch.version.cuda = 12.6
torch.cuda.is_available() = True
device name = NVIDIA GeForce RTX 3060 Laptop GPU
nvidia-smi driver version = 528.49
nvidia-smi CUDA version display = 12.0
```

Meaning:

```text
This first CUDA check detected the GPU but could not allocate CUDA tensors.
torch.ones(1, device="cuda") failed with:
CUDA-capable device(s) is/are busy or unavailable.
```

Resolved environment check:

```text
After updating the NVIDIA driver and restarting, CUDA allocation worked.
torch = 2.12.1+cu126
torch.version.cuda = 12.6
torch.cuda.is_available() = True
device name = NVIDIA GeForce RTX 3060 Laptop GPU
nvidia-smi driver version = 610.62
torch.ones(1, device="cuda") returned tensor([1.], device="cuda:0")
CUDA OK
```

Future improvement:

```text
Before full CIFAR-10 training, run a small GPU smoke/timing check and then choose
the final batch size and epoch count.
```

Why this may matter:

- CIFAR-10 training is much heavier than MNIST/Fashion-MNIST.
- GPU training may reduce runtime enough to allow more epochs or reliability checks.
- GPU setup can introduce dependency issues, so it should be a deliberate checkpoint.

Batch size note:

```text
The loader default batch size of 8 is only for data inspection.
The real training batch size is not decided yet.
```

Future batch-size decision:

```text
Before full teacher/student/KD training, choose one fixed training batch size after
checking device support and smoke-test runtime.
```

Candidate values for later discussion:

```text
CPU: 64 may be safer.
GPU: 64 or 128 may be possible, depending on memory and CUDA setup.
```

Fairness rule:

```text
Do not change batch size between hard-label student and KD student unless it is a
planned experiment variable. Keep batch size fixed for controlled comparisons.
```

Do not add GPU setup, dynamic batch-size schedules, or changing batch size during training silently. If GPU training is enabled, record the PyTorch build, CUDA availability, device name, batch size, and any runtime estimate.

Planned script direction:

```text
Use --device auto in V3 training scripts.
If CUDA-enabled PyTorch is installed later, auto can select cuda.
If the environment stays CPU-only, auto stays on cpu.
```

Important:

```text
V3 training scripts already use --device auto. With the CUDA-enabled environment,
auto should select cuda.
```

## 4. Runtime vs Model Performance Balance

Current goal:

```text
Balance CIFAR-10 model performance with realistic local training time.
```

Why this matters:

```text
CIFAR-10 has 50,000 training images. With batch size 64, one epoch has about
782 training batches. With batch size 128, one epoch has about 391 training batches.
```

Initial estimate before timing:

```text
CPU-only PyTorch:
- teacher epoch may take several minutes or longer
- student epoch should be faster than teacher
- KD epoch may be slower because both teacher and student forward passes are used

CUDA-enabled GPU PyTorch:
- full-set training may become much more practical
- setup must be verified before depending on it
```

Future decision checkpoint:

```text
Before full-set training, run a small timing smoke test and use it to estimate:
- teacher time per epoch
- student baseline time per epoch
- KD time per epoch
```

Planned conservative approach:

```text
Start with a small number of epochs, likely 3-5, then decide whether the teacher
is strong enough and whether runtime allows more.
```

Fairness rule:

```text
For controlled comparisons, hard-label student and KD student should use the same
epoch count, batch size, learning rate, seed, data transform, and device type.
```
