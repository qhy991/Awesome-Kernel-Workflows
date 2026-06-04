# CUDA Agent: Large-Scale Agentic RL for CUDA Kernel Generation

**English** · [简体中文](README.zh-CN.md)

Skill-integrated multi-turn kernel optimization with profiling-driven iterative refinement. Implements the inference-time agent loop from [CUDA Agent](https://arxiv.org/abs/2602.24286) (Dai et al., ByteDance Seed / Tsinghua AIR, 2026).

## Overview

CUDA Agent defines a structured **skill-based workflow** (SKILL.md) for CUDA kernel optimization that iterates through profile → implement → verify → refine cycles until a speedup target over `torch.compile` is achieved.

### Results (from paper)

On KernelBench (250 tasks), CUDA Agent achieves:
- **100%** faster rate on Level 1 & Level 2 over `torch.compile`
- **92%** faster rate on Level 3
- **2.11x** geometric mean speedup over `torch.compile`
- Outperforms Claude Opus 4.5, Gemini 3 Pro by ~40% on Level 3

### Key Design Principles

1. **Skill-based workflow** (SKILL.md): Structured 4-step process formalized as agent instructions
2. **Multi-file workspace**: `kernel.cu` + `kernel_binding.cpp` + `model_new.py` — separates concerns
3. **Robust reward schedule**: Discrete milestones (r ∈ {-1, 1, 2, 3}) instead of raw speedup — prevents reward hacking
4. **Anti-reward-hacking**: File permission isolation, no `torch.nn.functional` fallbacks, multi-input correctness testing
5. **Iterative refinement**: Up to 200 interaction turns with compilation/runtime/profiling feedback

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    Setup                                 │
│  Read model.py → create workspace → analyze operators  │
└───────────────────────┬────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│                   Profile                                │
│  Run profile.py → eager time, compile time              │
│  Identify bottlenecks (excessive launches, memory)      │
│  Plan optimization strategy (fusion, tiling, etc.)      │
└───────────────────────┬────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│           Iterative Loop (×MAX_TURNS)                    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Implement                                         │   │
│  │ kernel.cu + kernel_binding.cpp + model_new.py     │   │
│  └──────────────────────┬───────────────────────────┘   │
│                          │                               │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │ Verify                                            │   │
│  │ Compile → Correctness (5 random inputs) →         │   │
│  │ Benchmark (100 iters) → Compute reward            │   │
│  └──────────────────────┬───────────────────────────┘   │
│                          │                               │
│            ┌─────────────┴─────────────┐                 │
│            │ Target met?               │                 │
│            │ (speedup ≥ target)        │                 │
│            └──┬──────────────────┬─────┘                 │
│         Yes   │                  │  No                   │
│               ▼                  ▼                       │
│            [Done]         ┌──────────────┐              │
│                           │ Refine       │              │
│                           │ Fix errors   │              │
│                           │ based on     │              │
│                           │ feedback     │              │
│                           └──────┬───────┘              │
│                                  │ (next turn)           │
└──────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│                    Report                                │
│  Best kernel + speedup + history + optimization notes   │
└────────────────────────────────────────────────────────┘
```

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| **Claude Code** | Workflow runtime |
| **NVIDIA GPU + CUDA** | Kernel compilation and execution |
| **Python + PyTorch** | Reference model execution and benchmarking |
| **nvcc + pybind11** | Kernel compilation and Python bindings |

## Usage

```javascript
Workflow({name: 'cuda-agent-kernel-optimization', args: {
  kernel_path: '/path/to/model.py',
  op_description: 'Fused SwiGLU + Linear projection',
  test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
  profile_command: '<user-provided profiling command with {kernel_path}/{result_path}>',
  target_speedup: 1.05,    // 5% faster than torch.compile
  max_turns: 20,
}})
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_path` | (required) | Path to PyTorch model.py |
| `op_description` | `'PyTorch model'` | Operation description |
| `test_command` | `''` | Correctness verification command |
| `profile_command` | `''` | Performance profiling command |
| `compile_command` | `''` | Kernel compilation command |
| `target_speedup` | `1.05` | Required speedup over torch.compile |
| `max_turns` | `15` | Max refinement iterations |
| `exp_dir` | `'/tmp/cuda_agent_exp'` | Experiment directory |

## Reward Schedule

| Condition | Reward | Meaning |
|-----------|--------|---------|
| Correctness fails | -1 | Penalize incorrect kernels |
| Correct, faster than both eager AND compile (>5%) | 3 | Best outcome |
| Correct, faster than eager only (>5%) | 2 | Good, but compile still wins |
| Correct, not faster | 1 | Functional but no speedup |

## References

- [CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation](https://arxiv.org/abs/2602.24286) — Dai, Wu, Yu et al. (ByteDance Seed / Tsinghua AIR, 2026)
- [Project Page](https://cuda-agent.github.io/)
- [CUDA-Agent-Ops-6K Dataset](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K)

---

# CUDA Agent：大规模 Agent RL 驱动的 CUDA 内核生成

基于技能集成的多轮内核优化，通过性能分析驱动的迭代改进。实现了 [CUDA Agent](https://arxiv.org/abs/2602.24286) 的推理时 agent 循环（Dai 等人，字节跳动 Seed / 清华 AIR，2026）。

## 概述

CUDA Agent 定义了一个结构化的**技能工作流**（SKILL.md），通过 分析 → 实现 → 验证 → 改进 循环迭代，直到超过 `torch.compile` 的加速目标。

### 论文结果

在 KernelBench（250 个任务）上：
- Level 1 & 2：**100%** 快于 `torch.compile`
- Level 3：**92%** 快于 `torch.compile`
- 几何平均加速比：**2.11x**

### 核心设计

1. **技能工作流**：4 步结构化过程（profile → implement → verify → refine）
2. **多文件工作空间**：`kernel.cu` + `binding.cpp` + `model_new.py`
3. **鲁棒奖励调度**：离散里程碑（r ∈ {-1, 1, 2, 3}），防止奖励作弊
4. **迭代改进**：最多 200 轮交互，基于编译/运行/性能反馈

## 使用方法

```javascript
Workflow({name: 'cuda-agent-kernel-optimization', args: {
  kernel_path: '/path/to/model.py',
  op_description: '融合 SwiGLU + Linear',
  test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
  profile_command: '<user-provided profiling command with {kernel_path}/{result_path}>',
  target_speedup: 1.05,
  max_turns: 20,
}})
```

## 参考文献

- [CUDA Agent](https://arxiv.org/abs/2602.24286) — Dai, Wu, Yu 等 (ByteDance Seed / Tsinghua AIR, 2026)
- [项目主页](https://cuda-agent.github.io/)
