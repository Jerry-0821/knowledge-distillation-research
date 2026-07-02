# Research Workflow

```text
论文
  → research question
  → baseline
  → implementation
  → sanity check
  → controlled experiment
  → ablation
  → repeated seeds
  → results
  → failure analysis
  → conclusion
  → README
```

1. **论文：** 先区分作者实际写了什么、你如何理解，以及尚未理解的问题。
2. **Research question：** 把“想试试看”改写成可检验的问题。
3. **Baseline：** 先建立公平、可重复的比较起点。
4. **Implementation：** 只实现当前问题需要的最小代码，并保存 config 与 commit。
5. **Sanity check：** 用极小输入检查 shape、loss direction、device 和 data flow，不把能运行误当成方法正确。
6. **Controlled experiment：** 一次改变一个主要因素，其余条件保持一致。
7. **Ablation：** 检查某个组成部分是否真的贡献了观察结果。
8. **Repeated seeds：** 对重要结果重复随机种子，减少偶然波动造成的误判。
9. **Results：** 保存 raw results，再从 raw results 生成 table 和 figure。
10. **Failure analysis：** 记录失败、排查过程和限制，不删除不漂亮的实验。
11. **Conclusion：** 只总结证据支持的范围，并区分论文 claim 与本项目 observation。
12. **README：** 最后才更新对外说明，避免提前宣称完成复现。

Codex 可以帮助执行命令、检查代码、整理记录和发现不一致；你必须负责 research question、实验公平性、证据强度和最终研究判断。
