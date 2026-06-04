# AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search

**English** · [简体中文](README.zh-CN.md)

MCTS-based tree search for GPU kernel generation with failure-driven skill memory. Implements the [AdaExplore](https://arxiv.org/abs/2604.16625) methodology (Du et al., CMU / NVIDIA / OctoAI / UW, 2026).

## Overview

AdaExplore addresses two key challenges in LLM-based kernel generation:
1. **High failure rate**: LLM-generated Triton kernels often fail to compile or produce wrong results
2. **Mode collapse**: Repeated generation tends to converge on similar (often suboptimal) solutions

### Two-Stage Solution

**Stage 1 — Adapt** (Skill Memory): Collect failures → extract "You cannot..." constraint rules → inject into future prompts to prevent repeat errors.

**Stage 2 — Explore** (MCTS): Organize candidates as a tree, balance generating wholly new kernels (large iterations) vs refining existing ones (small iterations), using diversity-preserving context pools.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Setup                                    │
│  Read operator spec → PyTorch baseline → load skill memory    │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                 MCTS Search Loop (×STEPS)                      │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ SELECT: UCB1 tree traversal → promising node          │     │
│  └───────────────────────┬──────────────────────────────┘     │
│                           │                                    │
│                    ┌──────┴──────┐                             │
│                    │   DECIDE    │                             │
│                    └──┬──────┬──┘                             │
│          Large step   │      │   Small step                   │
│         (explore)     │      │   (exploit)                    │
│                       ▼      ▼                                │
│  ┌────────────────────┐  ┌──────────────────────────────┐    │
│  │ PROPOSER            │  │ REVISER → TUNER              │    │
│  │ Diverse pool +      │  │ Suggestions → surgical edits │    │
│  │ skill memory →      │  │ on parent kernel             │    │
│  │ new kernel from     │  └──────────────┬───────────────┘    │
│  │ scratch             │                  │                    │
│  └─────────┬──────────┘                  │                    │
│             └──────────────┬──────────────┘                    │
│                            ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ EVALUATE: compile → correctness → speedup measurement   │  │
│  └───────────────────────┬─────────────────────────────────┘  │
│                           │                                    │
│  ┌───────────────────────▼─────────────────────────────────┐  │
│  │ BACKPROPAGATE                                            │  │
│  │ • Update UCB1 rewards up the tree                       │  │
│  │ • On failure: extract "You cannot..." → skill memory    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                    Report                                      │
│  Best kernel + MCTS stats + skill memory                      │
└──────────────────────────────────────────────────────────────┘
```

## Key Mechanisms

### Skill Memory (Adapt Stage)

Failure-driven constraint accumulation:
```
Compile error: "SyntaxError: invalid syntax at line 42"
  → LLM extracts: "You cannot have an else: without matching if at this indentation level."
  → Rule added to skill memory with score 1.0
  → Same rule triggered again → score incremented
  → Rules above threshold injected into all future prompts
```

### Diverse Pool (Explore Stage)

For **large iterations** (proposer), context uses at most 1 kernel per MCTS branch:
- Prevents mode collapse (seeing many similar kernels → generating more of the same)
- Ensures structural diversity in the proposer's context
- Pool selection uses softmax-weighted sampling

### UCB1 with Expand Policy

Two separate scores:
- **UCB1** (node selection): balance exploit (high-reward subtrees) vs explore (under-visited subtrees)
- **Expand policy**: decide whether to add a new sibling (large step) or deepen (small step)
  - Force large step if `small_step_children >= small_step_limit`
  - Random large step with probability `p_large`

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| **Claude Code** | Workflow runtime |
| **NVIDIA GPU + CUDA** | Triton kernel execution |
| **Python + PyTorch + Triton** | Kernel generation and evaluation |

### Claude Code Skills (Recommended)

| Skill | Purpose |
|-------|---------|
| `ncu-report-skill` | Post-optimization profiling |
| `kernel-auto-tester` | Correctness validation infrastructure |

## Usage

```javascript
Workflow({name: 'adaexplore-kernel-optimization', args: {
  problem_definition: 'class Model(nn.Module):\n  def forward(self, x):\n    return F.layer_norm(x, ...) * F.gelu(x)',
  op_description: 'Fused LayerNorm + GELU',
  baseline_command: '<user-provided baseline command with {baseline_path}>',
  benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
  skill_memory_path: '/path/to/general_memory_v1_200.txt',
  iterations: 50,
  small_step_limit: 2,
  p_large: 0.2,
}})
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `problem_definition` | `''` | PyTorch operator source code |
| `op_description` | `'PyTorch operator'` | Human description |
| `baseline_command` | `''` | Command to measure PyTorch baseline |
| `benchmark_command` | `''` | Command to evaluate generated kernel |
| `skill_memory_path` | `''` | Pre-built skill memory file |
| `iterations` | `30` | MCTS search iterations (total expansions) |
| `small_step_limit` | `2` | Max small iterations before forcing large |
| `p_large` | `0.2` | Random large step probability |
| `ucb_c` | `1.41` | UCB1 exploration constant |
| `diversity_pool_size` | `5` | Max kernels in proposer context |
| `correctness_atol` | `0.05` | Numerical tolerance (absolute) |
| `correctness_rtol` | `0.05` | Numerical tolerance (relative) |

## Output

```javascript
{
  best_speedup: number,
  best_kernel_code: string,
  tree_size: number,
  compiled_count: number,
  correct_count: number,
  large_steps: number,
  small_steps: number,
  skill_memory: string[],
  skill_memory_size: number,
  report: string,
}
```

## References

- [AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation](https://arxiv.org/abs/2604.16625) — Du, Zhuo, Dong, He, Sun, Zheng, Karunaratne, Fox, Dettmers, Chen, Yang, Welleck (2026)
- [Project Page](https://stiglidu.github.io/AdaExplore/)

---

# AdaExplore：失败驱动的自适应 + 多样性保持搜索

基于 MCTS 树搜索的 GPU 内核生成，带失败驱动技能记忆。实现了 [AdaExplore](https://arxiv.org/abs/2604.16625) 方法论（Du 等人，CMU / NVIDIA / OctoAI / UW，2026）。

## 概述

AdaExplore 解决 LLM 内核生成的两个关键挑战：
1. **高失败率**：LLM 生成的 Triton 内核经常编译失败或产出错误结果
2. **模式坍缩**：重复生成趋向收敛到相似（通常次优）的解

### 两阶段方案

**阶段 1 — 适应**（技能记忆）：收集失败 → 提取 "You cannot..." 约束规则 → 注入未来提示词防止重复错误

**阶段 2 — 探索**（MCTS）：将候选者组织为树，平衡生成全新内核（大步）与细化现有内核（小步），使用多样性保持的上下文池

## 核心机制

### 技能记忆

```
编译错误: "SyntaxError: invalid syntax at line 42"
  → LLM 提取: "You cannot have an else: without matching if..."
  → 规则加入技能记忆，得分 1.0
  → 同一规则再次触发 → 得分递增
  → 得分超过阈值的规则注入所有未来提示词
```

### 多样性池

大步（proposer）的上下文每个 MCTS 分支最多用 1 个内核，防止模式坍缩。

### UCB1 + 扩展策略

- **UCB1**：平衡高奖励子树（利用）和低访问子树（探索）
- **扩展策略**：小步子节点 ≥ `small_step_limit` 时强制大步

## 使用方法

```javascript
Workflow({name: 'adaexplore-kernel-optimization', args: {
  problem_definition: 'class Model(nn.Module): ...',
  op_description: '融合 LayerNorm + GELU',
  benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
  iterations: 50,
}})
```

## 参考文献

- [AdaExplore](https://arxiv.org/abs/2604.16625) — Du, Zhuo, Dong 等 (2026)
- [项目主页](https://stiglidu.github.io/AdaExplore/)
