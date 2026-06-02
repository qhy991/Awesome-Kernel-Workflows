# KEET 内核性能解释工作流

[English](README.md) · **简体中文**

用于解释 GPU 内核“为什么快/慢”的工作流，实现 [KEET](https://arxiv.org/abs/2605.04467) 方法。重点是**解释与诊断**，而不是直接生成更快代码。

## 核心流程

1. 先读代码生成性能假设（不看 profile）
2. 再用 NCU 数据验证/否定假设
3. 输出可追溯、可审计的解释报告与改进建议

## 入口文件

- 工作流脚本：`keet-kernel-explanation.js`
- 英文完整文档：`README.md`

## 论文

- [KEET: Explaining Performance of GPU Kernels Using LLM Agents](https://arxiv.org/abs/2605.04467)
