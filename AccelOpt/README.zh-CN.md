# AccelOpt 内核优化工作流

[English](README.md) · **简体中文**

基于 Nsight Compute（NCU）的自改进 CUDA 内核优化工作流，实现 [AccelOpt](https://arxiv.org/abs/2511.15915) 方法。

## 核心循环

`Plan → Execute → Profile → Summarize → Accumulate Experience → Repeat`

## 适用场景

- 需要基于真实 profiling 数据做优化（避免“拍脑袋”）
- 希望保留候选者池（beam），而不是只保留一个最优解
- 希望从慢/快样本对中沉淀可复用经验模式

## 入口文件

- 工作流脚本：`accelopt-kernel-optimization.js`
- 英文完整文档：`README.md`

## 论文

- [AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization](https://arxiv.org/abs/2511.15915)
