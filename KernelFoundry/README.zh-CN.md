# KernelFoundry 工作流

[English](README.md) · **简体中文**

基于 MAP-Elites 的质量-多样性进化搜索工作流，实现 [KernelFoundry](https://arxiv.org/abs/2603.12440) 方法。

## 核心机制

- 行为描述符网格维护多样化高质量候选（防止模式坍缩）
- 元提示词与候选内核共同进化（缓解上下文退化）
- 模板化参数搜索（算法搜索与参数调优解耦）

## 入口文件

- 工作流脚本：`kernelfoundry-kernel-optimization.js`
- 英文完整文档：`README.md`

## 论文

- [KernelFoundry: Hardware-aware evolutionary GPU kernel optimization](https://arxiv.org/abs/2603.12440)
