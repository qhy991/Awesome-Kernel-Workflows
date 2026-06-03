# KernelBlaster 工作流

面向 CUDA 内核优化的记忆增强上下文内强化学习（MAIC-RL）工作流，改编自 KernelBlaster 论文：

- 论文：[KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning](https://arxiv.org/abs/2602.14293)
- 代码：[github.com/NVlabs/KernelBlaster](https://github.com/NVlabs/KernelBlaster)（Apache-2.0）
- 工作流文件：[`kernelblaster-kernel-optimization.js`](kernelblaster-kernel-optimization.js)

[English](README.md) · **简体中文**

---

## 它做什么

KernelBlaster 把 CUDA 内核优化重构为**记忆增强的上下文内强化学习（MAIC-RL）**。它不靠猜测，也不会"忘记"在之前内核上奏效的优化，而是维护一个**持久化的、以性能状态为键的知识库**，并在不同内核与不同运行之间复用经验。

每次 **rollout** 是一条多步轨迹：

1. **剖析 + 分类**：用 NCU 把当前内核归入某个硬件性能*状态*（内存受限 / 计算受限 / 延迟受限）。
2. **检索**：从知识库中取出该状态下最优的已知优化（按 置信度 × 预测加速 打分，并惩罚已用过的技术）。
3. **应用**：以策略引导的提示词施加该优化。
4. **评估**：编译、验证正确性、NCU 剖析得到 **Elapsed Cycles**。
5. **奖励 + 更新**：计算奖励，追加一步轨迹，并更新数据库条目的实测收益与置信度。

知识库会写回磁盘，因此经验可以**跨调用累积**——这是论文的核心创新点。

---

## 性能状态分类

| 状态 | 主要瓶颈 | 代表性技术 |
|------|----------|-----------|
| `memory_bandwidth_limited` | 内存受限 | 向量化访存（float4/half2）、合并访问、共享内存分块、AoS→SoA |
| `compute_throughput_limited` | 计算受限 | Tensor Core、指令混合平衡、功能单元利用、算法替换 |
| `latency_occupancy_limited` | 延迟受限 | 提升占用率、增加并行度、减少同步、线程粗化 |

每条技术条目都跟踪 `confidence_score`、`usage_count`、`predicted_improvement`、`actual_speedup`，工作流会随实测结果更新这些值。

---

## 奖励函数

与论文 `calculate_reward` 一致：

```
实测提升% = (prev_cycles − new_cycles) / prev_cycles × 100
reward = 实测提升/100
       + 预测准确度奖励   # 0.8 ≤ 实测/预测 ≤ 1.2 时 +0.2，否则 −0.1·|accuracy−1|
       + 惩罚            # 若变体未变快则 −0.5
```

当出现严重劣化（实测提升 < −20%）或该状态下没有未用过的优化时，rollout 提前停止。

---

## 示例调用

```javascript
Workflow({
  name: 'kernelblaster-kernel-optimization',
  args: {
    kernel_path: '/path/to/kernelbench-cuda/level1/001_Square_matrix_multiplication/init.cu',
    op_description: 'Square matrix multiplication',
    optimization_db_path: '/path/to/optimization_database.json',
    harness_build_cmd: 'nvcc -O3 -lineinfo -arch=sm_89 -o bench driver.cpp init.cu',
    kernel_name_regex: 'matmul_kernel',
    ncu_binary: 'ncu',
    rl_iterations: 3,
    rollout_steps: 4,
    gpu_type: 'L40S',
  },
})
```

用**同一个** `optimization_db_path` 连续优化多个内核，即可累积跨任务经验。

---

## 输出

`baseline_cycles`、`best_cycles`、`overall_speedup`、`rollouts_completed`、`trajectory_count`、`optimization_db`（最终知识库，同时写回磁盘）、`best_kernel_code`、`report`。

---

## 引用

```bibtex
@article{dong2026kernelblaster,
  title={KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning},
  author={Dong, Kris Shengjun and Modi, Sahil and Nikiforov, Dima and Damani, Sana and Lin, Edward and Hari, Siva Kumar Sastry and Kozyrakis, Christos},
  journal={arXiv preprint arXiv:2602.14293},
  year={2026},
  url={https://arxiv.org/abs/2602.14293}
}
```
