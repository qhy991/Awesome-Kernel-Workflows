# AdaExplore 工作流

[English](README.md) · **简体中文**

基于 MCTS 的 GPU 内核搜索流程，实现 [AdaExplore](https://arxiv.org/abs/2604.16625) 方法：失败驱动适应 + 多样性保持探索。

## 两阶段机制

- **Adapt**：从失败中提取 “You cannot ...” 规则，写入技能记忆
- **Explore**：用 MCTS 在大步探索与小步细化间平衡，并维持候选多样性

## 入口文件

- 工作流脚本：`adaexplore-kernel-optimization.js`
- 英文完整文档：`README.md`

## 论文

- [AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation](https://arxiv.org/abs/2604.16625)
