# KernelBand: Steering LLM-based Kernel Optimization via Hardware-Aware Multi-Armed Bandits

**English** · [简体中文](#kernelband基于硬件感知多臂老虎机的-llm-内核优化导航)

A Multi-Armed Bandit framework that formulates kernel optimization as a structured exploration-exploitation problem, using hardware-aware pruning and trace-driven clustering to efficiently navigate the optimization space. Implements the [KernelBand](https://arxiv.org/abs/2511.18868) methodology (Ran, Xie et al., PKU / Tongming Lake / ECNU / HKUST, 2026).

## Overview

KernelBand addresses a fundamental mismatch: code LLMs excel at generating functionally correct kernels, but lack the hardware-specific intuition to navigate toward performance-optimal regions. A naive LLM optimizer performs what amounts to a random walk, wasting iterations on transformations that yield negligible or negative speedups.

### Key Insight: Bandit Policy > LLM Intuition

The ablation study (Table 4) shows that replacing KernelBand's bandit policy with LLM-based strategy selection causes a catastrophic drop to 0.97× speedup (below baseline). This confirms that **learned execution statistics outperform LLM intuition** for strategy selection.

### Results (from paper)

- **1.91×** geometric mean speedup on A100 (TritonBench-G, T=20 iterations)
- **79.8%** correctness rate across 183 Triton kernels
- **35–50%** higher speedup per dollar compared to unguided methods
- Consistent superiority across RTX 4090, H20, and A100

## Architecture

```
                    ┌─────────────────────────────┐
                    │         KernelBand           │
                    └─────────────┬───────────────┘
                                  │
┌──────────────┐    ┌─────────────▼───────────────┐    ┌──────────────┐
│ Candidate    │    │  Hardware-Constrained Bandit │    │  Hardware    │
│ Kernel Pool  │◄──►│  (Masked UCB)               │◄──►│  Evaluation  │
│ (P)          │    │                             │    │              │
└──────┬───────┘    │  Kernel Selection (k_t)     │    │ Compile &    │
       │            │  Strategy Selection (s_t)   │    │ Run (k'_{t+1})│
       │            └─────────────┬───────────────┘    │              │
       │                          │                    │ Correctness  │
┌──────▼───────┐    ┌─────────────▼───────────────┐    │ Check        │
│ Profiling &  │    │  LLM Code Generation        │    │              │
│ Clustering   │    │  LLM(k_t, s_t, H) → k'_t   │    │ Measure Time │
│              │    └─────────────────────────────┘    └──────┬───────┘
│ φ(k): behavioral                                            │
│ feature vector                                              │
│              │                                    Reward r_t │
│ K-Means on   │◄─────────────────────────────────────────────┘
│ φ-space      │         Update Cluster Stats (μ̂, N)
└──────────────┘
```

### Three Key Mechanisms

| Mechanism | Purpose | Implementation |
|-----------|---------|----------------|
| **Runtime Behavior Characterization** | Enable knowledge sharing between similar kernels | φ(k) = [T̄, n_reg, n_smem, d_block, η_occ] — 5D behavioral feature vector from NCU |
| **Dynamic Clustering** | Manage expanding action space efficiently | K-Means (K=3) on φ(k), re-cluster every τ=10 iterations, profile only centroids |
| **Hardware-Constrained Bandit** | Prune invalid strategies, balance explore/exploit | Masked UCB: only select strategies targeting non-saturated resources (θ_sat=75%) |

### Optimization Strategies (|S| = 6)

| Strategy | Target Resource | Risk/Reward Profile |
|----------|----------------|---------------------|
| Tiling | SM (compute) | High risk, high reward (14.4% success, 61.5% best contribution) |
| Vectorization | DRAM (memory) | Low risk, low reward (57.1% success, 17.1% best) |
| Fusion | DRAM (memory) | Balanced (75% success, 55% best) |
| Pipeline | SM (compute) | Medium (64.4% success, 26.3% best) |
| Reordering | L2 (cache) | Medium (48.7% success, 25.3% best) |
| Access & Layout | DRAM (memory) | Medium (29.5% success, 19.2% best) |

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| **Claude Code** | Workflow runtime |
| **NVIDIA GPU + CUDA** | Kernel compilation and execution |
| **NCU (Nsight Compute)** | Profiling for behavioral features and hardware signatures |
| **Python + Triton** | Kernel JIT (for Triton kernels) |

### Claude Code Skills (Recommended)

| Skill | Purpose |
|-------|---------|
| `ncu-report-skill` | Automated NCU profiling and metric extraction |
| `kernel-auto-tester` | Correctness verification across input shapes |
| `KernelWiki` | Architecture-specific optimization strategies |

## Usage

```javascript
Workflow({name: 'kernelband-kernel-optimization', args: {
  kernel_path: '/path/to/kernel.py',
  op_description: 'Fused attention forward (batch=32, seq=2048, heads=32)',
  harness_path: '/path/to/benchmark.py',
  compile_command: '<user-provided compile/import command with {kernel_path}/{result_path}>',
  benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
  ncu_command: '<user-provided profiler command with {kernel_path}/{result_path}>',
  target_gpu: 'A100',
  iterations: 20,
  num_clusters: 3,
  recluster_period: 10,
  strategies: ['tiling', 'vectorization', 'fusion', 'pipeline', 'reordering', 'access_layout'],
  ucb_exploration: 2.0,
  saturation_threshold: 0.75,
}})
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_path` | `''` | Path to the kernel source file |
| `op_description` | `'GPU kernel'` | Description of the operation being optimized |
| `harness_path` | `''` | Benchmark harness file |
| `compile_command` | `''` | Compilation/import command |
| `benchmark_command` | `''` | Performance measurement command |
| `ncu_command` | `''` | NCU profiling command for feature extraction |
| `target_gpu` | `'A100'` | Target GPU architecture |
| `iterations` | `20` | Total optimization iterations T |
| `num_clusters` | `3` | Number of clusters K (paper recommends K=3) |
| `recluster_period` | `10` | Re-cluster every τ iterations |
| `strategies` | `[tiling, vectorization, ...]` | Available optimization strategies |
| `ucb_exploration` | `2.0` | UCB exploration constant c |
| `saturation_threshold` | `0.75` | θ_sat: prune strategies above this utilization |
| `exp_dir` | `'/tmp/kernelband_exp'` | Experiment output directory |

## How It Differs from Other Workflows

| Aspect | KernelBand | AccelOpt / GEAK-style |
|--------|-----------|----------------------|
| Strategy selection | Bandit policy (data-driven, UCB) | LLM decides (intuition-based) |
| Exploration | Principled (regret bounds) | Ad hoc (random/self-reflection) |
| Hardware awareness | Profiling-based pruning (mandatory) | Optional NCU analysis |
| Knowledge transfer | Clustering enables cross-kernel learning | Per-kernel experience only |
| Budget efficiency | 35-50% more speedup per dollar | Higher cost per gain |

## References

- [KernelBand: Steering LLM-based Kernel Optimization via Hardware-Aware Multi-Armed Bandits](https://arxiv.org/abs/2511.18868) — Ran, Xie, Ji, Liu, Wu, Cao, Guo, Yu, Li, Hu, Yang, Xie (PKU et al., 2026)

---

# KernelBand：基于硬件感知多臂老虎机的 LLM 内核优化导航

**English** · 简体中文

多臂老虎机框架，将内核优化建模为结构化的探索-利用问题，使用硬件感知剪枝和轨迹驱动聚类高效导航优化空间。实现了 [KernelBand](https://arxiv.org/abs/2511.18868) 方法论（Ran, Xie 等人，北大 / 通明湖 / 华师 / 港科大，2026）。

## 概述

KernelBand 解决了一个根本性不匹配：代码 LLM 擅长生成功能正确的内核，但缺乏硬件特定直觉来导航至性能最优区域。朴素的 LLM 优化器本质上是随机游走，浪费大量预算在收益为零或负面的变换上。

### 核心发现

消融实验表明：用 LLM 语义推理替代 bandit 策略选择会导致性能灾难性下降至 0.97×（低于基线），证实了**学习到的执行统计优于 LLM 直觉**。

### 论文结果

- A100 上 **1.91×** 几何平均加速（TritonBench-G，T=20 迭代）
- **79.8%** 正确率
- 相比无引导方法，每美元 **35–50%** 更高加速

## 三大机制

| 机制 | 目的 | 实现 |
|------|------|------|
| **运行时行为表征** | 相似内核间知识共享 | φ(k) = [T̄, n_reg, n_smem, d_block, η_occ] |
| **动态聚类** | 管理扩展动作空间 | K-Means (K=3)，每 τ=10 轮重聚类，仅对质心 profiling |
| **硬件约束 Bandit** | 剪枝无效策略 + 平衡探索利用 | Masked UCB：仅选择目标资源未饱和的策略 (θ_sat=75%) |

## 前置依赖

| 依赖 | 用途 |
|------|------|
| **Claude Code** | Workflow 运行时 |
| **NVIDIA GPU + CUDA** | 内核编译执行 |
| **NCU** | 行为特征提取和硬件签名 |
| **Python + Triton** | Triton 内核 JIT 编译 |

## 使用方法

```javascript
Workflow({name: 'kernelband-kernel-optimization', args: {
  kernel_path: '/path/to/kernel.py',
  op_description: 'Fused attention forward',
  target_gpu: 'A100',
  iterations: 20,
  num_clusters: 3,
}})
```

## 与其他 Workflow 的区别

| 方面 | KernelBand | AccelOpt 类方法 |
|------|-----------|----------------|
| 策略选择 | Bandit 策略（数据驱动，UCB） | LLM 决定（基于直觉） |
| 探索方式 | 有原则的（有遗憾界） | 随机/自反思 |
| 硬件感知 | 强制性 profiling 剪枝 | 可选 NCU 分析 |
| 知识迁移 | 聚类实现跨内核学习 | 仅单内核经验 |

## 参考文献

- [KernelBand](https://arxiv.org/abs/2511.18868) — Ran, Xie 等 (北大等, 2026)
