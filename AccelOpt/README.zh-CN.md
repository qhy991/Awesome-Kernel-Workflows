# AccelOpt 内核优化工作流

[English](README.md) · **简体中文**

基于 Nsight Compute（NCU）的自改进 CUDA 内核优化工作流，实现 [AccelOpt](https://arxiv.org/abs/2511.15915) 方法。

## CuTe DSL SOL 适配

设置 `language: cute-dsl` 和 `integration_pattern: sol_execbench_solution` 时，工作流保留规划、实现、选择和经验积累循环，但反馈来自官方 Host 的完整工作负载正确性与延迟，而非 NCU 计数器。这是**保留方法思想的适配**，不是 AccelOpt 的 profiler 高保真复现。只有 Host 验收并绑定源码的候选才能进入候选池；模型的静态估计不能晋升。报告与经验规则应标明证据仅为 Host 延迟和源码结构。CuTe 候选需为包含 `run(...)` 和已编译 CuTe 内核的完整 Python 模块；继承的 CUDA 实现保留为独立性能底线。

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
