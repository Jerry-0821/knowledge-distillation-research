# Environment Report

- **Recorded:** 2026-07-02
- **Scope:** local setup for `knowledge-distillation-research`
- **Privacy:** full user and repository paths are intentionally omitted

## Operating system and tools

| Item | Detected value |
|---|---|
| Operating system | Windows 11, build 10.0.26200, AMD64 |
| Registry product label | Windows 10 Home Single Language, 25H2, build 26200 |
| PowerShell | 5.1.26100.8737 |
| Git | 2.53.0.windows.2 |
| VS Code CLI | Available, version 1.126.0, x64 |
| Working directory | Repository folder `knowledge-distillation-research`; full path omitted |
| Disk | C: approximately 462.6 GB total and 120.2 GB free at inspection time |

The Windows platform API identifies this build as Windows 11, while a legacy registry label still says Windows 10. The build number is recorded so the discrepancy is visible rather than silently resolved.

## Project Python environment

| Item | Detected value |
|---|---|
| Environment | Project-local `.venv`; global packages are not used |
| Python | 3.13.14 |
| pip | 26.1.2 |
| PyTorch | 2.12.1+cpu |
| TorchVision | 0.27.1+cpu |
| NumPy | 2.4.4 |
| pandas | 3.0.3 |
| matplotlib | 3.11.0 |
| PyYAML | 6.0.3 |
| tqdm | 4.68.3 |
| pytest | 9.1.1 |
| `torch.cuda.is_available()` | `False` |
| CUDA reported by PyTorch | `None` because this is a CPU-only build |

The pre-existing global Python environment had PyTorch 2.12.0+cpu and TorchVision 0.27.0+cpu. Those packages were not reused because `.venv` is intentionally isolated.

## GPU and NVIDIA driver

| Item | Detected value |
|---|---|
| `nvidia-smi` | Available |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| GPU memory | 6144 MiB |
| NVIDIA driver | 528.49 |
| CUDA version shown by `nvidia-smi` | 12.0 |
| GPU usable by current PyTorch build | No; current project build is CPU-only |

The CUDA value shown by `nvidia-smi` describes driver capability. It does not prove that a CUDA toolkit is installed, and it is different from `torch.version.cuda`. A future GPU-enabled PyTorch installation must be selected only after checking the current driver against the official PyTorch installation options.

## Verification

| Check | Result |
|---|---|
| Environment script | Passed |
| PyTorch import | Passed |
| TorchVision import | Passed |
| Dummy tensor operation | Passed on CPU |
| Tiny `nn.Linear` forward pass | Passed on CPU |
| Config loading tests | Passed |
| Full pytest suite | 9 passed in 11.90 seconds |
| Warnings in final verification | None |

## Safety observations

- The workspace was empty before this project folder was created, so no existing project files were overwritten.
- An unrelated higher-level Git repository was detected above the workspace. This project is kept in its own repository folder so no parent files are staged or committed.
- No dataset was downloaded.
- No teacher model, student model, Knowledge Distillation loss, training loop or experiment result was created.
- No CUDA build, CUDA toolkit, API key, token, checkpoint or GitHub remote was added.
