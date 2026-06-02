# CUDA Agent 工作流

[English](README.md) · **简体中文**

多轮技能化 CUDA 内核优化流程，实现 [CUDA Agent](https://arxiv.org/abs/2602.24286) 的推理时 agent 循环：profile → implement → verify → refine。

## 核心特点

- 多文件工作空间（`kernel.cu` / `kernel_binding.cpp` / `model_new.py`）
- 迭代式修复与优化，直到达到相对 `torch.compile` 的加速目标
- 采用离散奖励里程碑，降低 reward hacking 风险

## 入口文件

- 工作流脚本：`cuda-agent-kernel-optimization.js`
- 英文完整文档：`README.md`

## 论文

- [CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation](https://arxiv.org/abs/2602.24286)
