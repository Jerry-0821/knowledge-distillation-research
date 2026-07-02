# Knowledge Distillation Research

## Purpose

这是一个面向首次论文复现的科研工程仓库。当前阶段只准备干净、可学习、可复现的本地开发环境，为之后阅读并复现 *Distilling the Knowledge in a Neural Network* 做准备。

## Current status

**Environment preparation only.** 仓库目前只包含项目结构、环境脚本、配置示例、测试和研究记录模板。

## Intended paper and direction

- Intended paper: *Distilling the Knowledge in a Neural Network*
- Future direction: 研究 teacher–student knowledge transfer，以及 accuracy、parameter count、model size 和 inference latency 的 trade-off。

## Repository structure

- `src/kd_research/`: 未来的可复用 Python package；目前不包含论文算法。
- `configs/`: 实验配置示例；示例值不是最终 hyperparameters。
- `scripts/`: 环境检查、安装和 smoke test。
- `tests/`: 只验证项目 setup、imports、config 和极小 tensor operation。
- `docs/`: 论文笔记与实验记录模板。
- `data/`, `checkpoints/`, `logs/`, `results/`: 本地数据和生成物的预留位置。

## Setup

```powershell
Set-Location knowledge-distillation-research
.\scripts\setup.ps1 -InstallTorchCpu
.\.venv\Scripts\Activate.ps1
```

`-InstallTorchCpu` 是明确选择 CPU-only PyTorch 的开关。以后若改用 GPU build，应先核对 NVIDIA driver 与 PyTorch 官方支持组合。

## Verification

```powershell
.\scripts\verify_setup.ps1
```

也可以分开运行：

```powershell
python scripts/check_environment.py
python scripts/smoke_test.py
python -m pytest
```

> **Important limitation:** No paper reproduction or experimental result has been completed yet.
