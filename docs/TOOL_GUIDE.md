# Tool Guide

| Tool | 它是什么？ | 为什么需要？何时使用？ | 主要环境？ |
|---|---|---|---|
| VS Code | 代码编辑器与调试界面。 | 打开整个 repository、阅读和修改文件、使用 integrated terminal、运行测试与 debug。 | **是**，这是主要工作界面。 |
| Python | 本项目使用的编程语言和运行时。 | 执行 scripts、tests，以及未来由你确认后实现的研究代码。 | **是**，这是主要运行环境。 |
| `.venv` | repository 内的独立 Python virtual environment。 | 隔离本项目 packages，避免其他项目升级依赖后破坏实验。每次进入项目终端时使用。 | **是**，所有本地 Python 命令应优先使用它。 |
| pip | Python package installer。 | 在 `.venv` 内安装 `pyproject.toml` 定义的依赖；不要用它修改全局环境。 | **是**，但只负责依赖安装。 |
| PyTorch | tensor 与 neural network 计算框架。 | 现在只用于 installation check 和极小 smoke test；以后才用于经确认的研究实现。 | **是**，未来的主要 ML framework。 |
| TorchVision | PyTorch 的 computer-vision 数据与模型辅助库。 | 现在只检查能否 import；以后可能用于 dataset/transforms，但具体用途尚未决定。 | **辅助**，不是独立工作环境。 |
| Git | 本地版本控制工具。 | 在重要修改与实验之间保存可追踪 snapshot，并把结果对应到 commit。 | **是**，负责本地历史。 |
| GitHub | 远程 repository 托管与协作平台。 | 以后经你确认 visibility 和 remote 后用于备份、分享与协作。 | **否**，目前尚未连接。 |
| pytest | Python testing framework。 | 修改 setup 或代码后运行，及时发现 import、config 与基础环境错误。 | **辅助但必要**，不是训练工具。 |
| YAML config | 人类可读的结构化配置文件。 | 以后记录每次实验设置，避免把研究选择散落在代码中；example 不能当最终配置。 | **辅助**，由 Python pipeline 读取。 |
| Jupyter Notebook | 交互式 Python 文档。 | 只用于少量数据查看、临时想法与 exploratory plots，不承载主要 training pipeline。 | **否**，仅辅助探索。 |
| Google Colab | 云端 notebook 与可选计算环境。 | 本机算力不足时才考虑；核心 source、configs 和结果定义仍由 repository 管理。 | **否**，是备用计算环境。 |
| Codex | repository 内的 AI engineering assistant。 | 可协助解释、编辑、测试和记录，但 research question、方法选择与结论必须由你判断。 | **辅助**，不是研究负责人。 |

GitLens 可作为可选 VS Code extension；它能更方便查看 Git history，但当前 setup 不强制安装。
