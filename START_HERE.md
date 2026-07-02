# Start Here

这是 Knowledge Distillation 论文复现的本地科研仓库。现在已经准备好目录结构、项目专用 `.venv`、VS Code 配置、环境检查、基础测试和研究记录模板。

当前**没有**下载 dataset，没有实现 teacher/student model、Knowledge Distillation loss 或 training loop，也没有任何实验结果。

## 第一次打开项目

```powershell
code .
```

在 VS Code integrated terminal 中启用项目环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止 activation script，可以不激活，直接使用 `.\.venv\Scripts\python.exe` 执行后续命令。

## 检查与测试

```powershell
python scripts/check_environment.py
python scripts/smoke_test.py
python -m pytest
```

或一次运行：

```powershell
.\scripts\verify_setup.ps1
```

正式开始读论文后，先打开 `docs/PAPER_NOTES_TEMPLATE.md`，阅读论文的 title、abstract、introduction 和实验问题，再由你自己填写模板；不要从结论倒推答案。
