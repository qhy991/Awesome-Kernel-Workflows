# ARGUS 工作流

[English](README.md) · **简体中文**

基于“数据流不变量 + ICRL（上下文强化学习）”的 GPU 内核优化流程，实现 [ARGUS](https://arxiv.org/abs/2604.18616) 方法。

## 核心特点

- 用不变量违规反例提供密集反馈，而不只是 pass/fail
- 外层 ICRL 策略学习，内层逐步优化执行
- 更适合复杂、全局耦合的优化问题（如流水线、调度、融合）

## 入口文件

- 工作流脚本：`argus-kernel-optimization.js`
- 英文完整文档：`README.md`

## 论文

- [ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants](https://arxiv.org/abs/2604.18616)
