# AKO4X: Agentic Kernel Optimization — Advanced & Extensible

**English** · [简体中文](README.zh-CN.md)

Multi-round closed-loop GPU kernel optimization with experience accumulation. Implements the [AKO4X](https://tongminglaic.github.io/AKO) methodology — a two-level iteration protocol (iterations × iterations) with structured lesson archival and noise-aware benchmarking.

## Overview

AKO4X goes beyond simple "try and benchmark" optimization through:

- **Two-level iteration**: Outer iterations (strategy selection) × inner iterations (variant refinement)
- **Pre-commit Expected**: Write hypothesis BEFORE benchmarking to catch retrofitted explanations
- **Smoke test → Full bench**: Don't waste time on full benchmark if quick check fails
- **Pre-archive gates**: Silent-skip detection + library-delegation check before promoting
- **Lessons-in-header**: Optimization knowledge colocated in the kernel file as structured 5-section headers
- **Two-layer WHEN**: Narrow (this kernel) + Broad (general GPU) applicability for each lesson
- **Dead-ends as priors**: Failed approaches stored with WHY, not as prohibition rules
- **TRAPS.md**: Cross-variant silent-bug patterns accumulated across iterations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Setup                                       │
│  Read kernel → detect language → create workspace → baseline │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│               Round Loop (×ROUNDS)                            │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Round-Init                                              │  │
│  │ Cross-round reflection → select parent variant          │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                            │                                  │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │ Iterate (inner loop ×ITERS_PER_ROUND)                   │  │
│  │ Hypothesize → Implement → Smoke test → Bench → Commit  │  │
│  │ (plateau detection stops early)                          │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                            │                                  │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │ Archive                                                 │  │
│  │ Pre-archive gates → 5-section header → TRAPS.md update  │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                            │                                  │
│  ┌────────────────────────▼───────────────────────────────┐  │
│  │ Retrospect (Mode 3 only)                                │  │
│  │ Phase-2 harness retrospective → extract lessons          │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                            ▼ (next round)                     │
└──────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Report                                     │
│  Best variant + lessons + dead-ends + open directions         │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| **Claude Code** | Workflow runtime |
| **GPU + Toolchain** | NVIDIA (CUDA/nvcc/ncu) or target platform |
| **Benchmark harness** | User-provided `benchmark_command` |
| **Python** | For Triton/TileLang kernels |

### Claude Code Skills (Recommended)

| Skill | Purpose |
|-------|---------|
| `ncu-report-skill` | NCU profiling and bottleneck diagnosis |
| `KernelWiki` | Hopper/Blackwell optimization patterns |
| `kernel-auto-tester` | Test infrastructure generation |

## Usage

```javascript
Workflow({name: 'ako4x-kernel-optimizer', args: {
  kernel_path: '/path/to/kernel.py',
  op_description: 'Multi-head Latent Attention paged decode',
  language: 'triton',
  benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
  smoke_test_command: '<user-provided quick compile+correctness command>',
  iterations: 5,
  iters_per_round: 5,
  breadth: 3,
  samples_per_plan: 2,
  target_gpu: 'b200',
  mode: 2,                   // 2 = static harness, 3 = harness co-evolution
}})
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_path` | (required) | Path to target kernel file |
| `op_description` | `'GPU kernel'` | Operation description |
| `language` | `'auto'` | triton / cuda / cute-dsl / tilelang / cpp / pytorch |
| `benchmark_command` | `''` | Full benchmark command |
| `smoke_test_command` | `''` | Quick compile + correctness check |
| `ncu_binary` | `'ncu'` | Path to NCU CLI |
| `iterations` | `5` | Max optimization iterations |
| `iters_per_round` | `5` | Max iterations per round |
| `breadth` | `3` | Hypotheses per round |
| `samples_per_plan` | `2` | Variants per hypothesis |
| `target_gpu` | `'b200'` | Target GPU architecture |
| `mode` | `2` | 2 = static harness, 3 = harness co-evolution |

## Source

- [AKO4X Project Page](https://tongminglaic.github.io/AKO)
- [AKO4ALL GitHub](https://github.com/TongmingLAIC/AKO4ALL)

---

# AKO4X：Agent 驱动的内核优化 — 进阶 & 可扩展

多轮闭环 GPU 内核优化，带经验累积。实现了 [AKO4X](https://tongminglaic.github.io/AKO) 方法论 — 双层迭代协议（轮次 × 迭代）+ 结构化经验归档 + 噪声感知基准测试。

## 核心特性

- **双层迭代**：外层轮次（策略选择）× 内层迭代（变体细化）
- **预提交假设**：先写假设再跑 benchmark，防止事后解释偏差
- **烟雾测试 → 完整 bench**：快速检查失败就不浪费完整 bench 时间
- **归档前门控**：静默跳过检测 + 库委托检查
- **经验嵌入头部**：优化知识以 5 段结构化 header 形式嵌入内核文件
- **双层 WHEN**：每条经验标注窄（本内核）+ 宽（通用 GPU）适用范围
- **TRAPS.md**：跨变体静默 bug 模式累积

## 使用方法

```javascript
Workflow({name: 'ako4x-kernel-optimizer', args: {
  kernel_path: '/path/to/kernel.py',
  op_description: '多头潜注意力分页解码',
  language: 'triton',
  benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
  iterations: 5,
  iters_per_round: 5,
  target_gpu: 'b200',
}})
```

## 来源

- [AKO4X 项目主页](https://tongminglaic.github.io/AKO)
- [AKO4ALL GitHub](https://github.com/TongmingLAIC/AKO4ALL)
