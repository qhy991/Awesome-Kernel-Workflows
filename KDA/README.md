# KDA: Kernel Design Agents

**English** · [简体中文](README.zh-CN.md)

Evidence-driven kernel optimization with draft-plan-implement-validate-record cycles. Implements the [Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents) methodology — a repeatable loop for agent-driven implementation work with explicit promotion criteria and evidence records.

## Overview

KDA is designed for **benchmark-contest-style** kernel optimization where:
- You have a clear task contract (objective, correctness requirements, performance target)
- You need auditable evidence for every decision (promote / revise / reject)
- You want systematic exploration without losing track of what was tried

### Key Principles

1. **Task contract first** — Define objective, constraints, validation command, and promotion criteria before writing any code
2. **Draft before plan** — Write `docs/draft.md` (exploration) before committing to `docs/plan.md` (execution)
3. **One candidate at a time** — Implement, validate, measure, decide — then next
4. **Evidence-based promotion** — Only promote when task contract is satisfied AND target metric improves
5. **Record everything** — Rejected candidates get recorded with reasons, not silently discarded

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    Inspect                               │
│  Read workspace → understand baseline → identify        │
│  validation path → extract task contract                │
└───────────────────────┬────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│                     Plan                                 │
│  Write docs/draft.md (explore approaches)               │
│  Convert to docs/plan.md (executable iterations)             │
└───────────────────────┬────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│              Candidate Loop (×MAX_CANDIDATES)            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Implement                                         │   │
│  │ One candidate from the plan                       │   │
│  └──────────────────────┬───────────────────────────┘   │
│                          │                               │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │ Validate                                          │   │
│  │ Run test_command → correctness check        │   │
│  │ Run benchmark_command → measure target metric    │   │
│  └──────────────────────┬───────────────────────────┘   │
│                          │                               │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │ Decide                                            │   │
│  │ Record evidence → promote / revise / reject       │   │
│  │ Update candidates.jsonl                           │   │
│  └──────────────────────┬───────────────────────────┘   │
│                          │                               │
│                          ▼ (next candidate or stop)      │
└──────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│                    Report                                │
│  Final report + candidates.jsonl + best variant          │
└────────────────────────────────────────────────────────┘
```

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| **Claude Code** | Workflow runtime |
| **GPU + CUDA** | Kernel compilation and execution |
| **Validation script** | User-provided correctness check |
| **Benchmark script** | User-provided performance measurement |

## Usage

```javascript
Workflow({name: 'kda-kernel-workflow', args: {
  kernel_path: '/path/to/kernel.cu',
  task_name: 'quantized-gemm-q4_0',
  objective: 'Optimize Q4_0 GEMM kernel for H100',
  correctness_requirements: 'Output must match baseline within 1e-5 relative error',
  performance_target: 'Achieve < 0.5ms on M=4096, N=4096, K=4096',
  allowed_approaches: 'CUDA C++, shared memory tiling, warp-level primitives',
  test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
  benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
  promotion_criteria: 'Speedup >= 1.2x over baseline AND passes validation',
  max_candidates: 5,
}})
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_path` | (required) | Path to target kernel file |
| `task_name` | `'unnamed-task'` | Identifier for this optimization task |
| `objective` | `'Optimize the target kernel'` | What you're trying to achieve |
| `correctness_requirements` | `'Must produce correct output'` | Correctness criteria |
| `performance_target` | `'Improve over baseline'` | Performance goal |
| `allowed_approaches` | `'Any approach'` | Constraints on implementation |
| `test_command` | (required) | Command to check correctness |
| `benchmark_command` | same as validation | Command to measure performance |
| `promotion_criteria` | `'Passes validation and improves metric'` | When to promote a candidate |
| `max_candidates` | `5` | Max candidates to try |

## Output

```javascript
{
  task_name: string,
  objective: string,
  baseline_metrics: object,
  best_candidate_id: string,
  best_metrics: object,
  speedup: number,
  candidates_tried: number,
  promoted_count: number,
  rejected_count: number,
  candidates: [...],      // Full candidates.jsonl equivalent
  best_kernel_code: string,
  report: string,
}
```

## Source

- [Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents) — MIT HAN Lab
- MLSys 2026 Kernel Contest solution (#1-3 on tracks)

---

# KDA：内核设计 Agents

基于证据驱动的内核优化，采用 草案-计划-实现-验证-记录 循环。实现了 [Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents) 方法论 — 用于 agent 驱动实现工作的可重复循环，带显式晋升标准和证据记录。

## 核心原则

1. **任务契约优先** — 编码前先定义目标、约束、验证命令和晋升标准
2. **先草案后计划** — 先写 `docs/draft.md`（探索），再写 `docs/plan.md`（执行）
3. **逐个候选** — 实现、验证、测量、决策，然后下一个
4. **基于证据晋升** — 仅在任务契约满足 + 目标指标改善时才晋升
5. **记录一切** — 被拒绝的候选也带原因记录，不静默丢弃

## 使用方法

```javascript
Workflow({name: 'kda-kernel-workflow', args: {
  kernel_path: '/path/to/kernel.cu',
  task_name: 'quantized-gemm-q4_0',
  objective: '为 H100 优化 Q4_0 GEMM 内核',
  test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
  benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
  promotion_criteria: '加速比 >= 1.2x 且通过验证',
  max_candidates: 5,
}})
```

## 来源

- [Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents) — MIT HAN Lab
- MLSys 2026 Kernel Contest 解决方案（赛道 #1-3 名）
