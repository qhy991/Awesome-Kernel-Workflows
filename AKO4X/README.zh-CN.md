# AKO4X 工作流

[English](README.md) · **简体中文**

多轮闭环的 Agentic GPU 内核优化流程，实现 AKO4X 方法（round × iteration 双层迭代），强调经验沉淀与噪声感知评测。

## 核心特点

- 外层轮次选策略，内层迭代做实现细化
- 先写假设，再跑 benchmark（避免事后解释）
- smoke test 失败则不进入 full bench
- 归档前门控（silent-skip / delegation 检查）

## 入口文件

- 工作流脚本：`ako4x-kernel-optimizer.js`
- 英文完整文档：`README.md`

## 来源

- [AKO4X Project](https://tongminglaic.github.io/AKO)
