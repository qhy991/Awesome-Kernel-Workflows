# ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants

**English** · [简体中文](README.zh-CN.md)

An iterative GPU kernel optimization workflow using **data-flow invariants** for dense structured feedback and **in-context reinforcement learning (ICRL)** for adaptive planning. Implements the [ARGUS](https://arxiv.org/abs/2604.18616) methodology (Mai et al., CausalFlow / HKUST / Stanford, 2026).

## Overview

ARGUS bridges the gap between LLM-generated kernels and hand-optimized assembly libraries (HipBLASLt, AITER). While existing agents achieve correctness through unit tests, they lack the **dense feedback** needed to guide complex, globally-coupled optimizations like software pipelining and instruction scheduling.

### Key Innovation: Data-Flow Invariants as Dense Feedback

```
Tag Functions → Symbolic Annotations → Tag Assertions → Violation Counterexamples
```

Instead of sparse pass/fail from tests:
- **Tag functions** attach symbolic coordinates to tensor elements
- **Tag assertions** enforce relational constraints at use sites (e.g., "the MFMA operands must come from matching K-slices")
- **Violations** produce concrete counterexamples: (thread, element, program point)
- This gives the agent **precise, structured feedback** for targeted fixes

### Results (from paper)

On AMD MI300X, ARGUS-generated kernels achieve:
- **99-104%** of hand-optimized assembly throughput (HipBLASLt, HipKittens, AITER)
- **2-1543x** faster than existing agentic systems (KernelBench, KSearch, CUDAForge, KernelFalcon)
- **100%** Pass@1 on KernelBench Level 1, **90%** on Level 2

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Setup                                         │
│  Read kernel spec → Analyze → Baseline profiling                 │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────┐
│                   ICRL Outer Loop                                  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Plan (ICRL Planner)                                         │  │
│  │ Knowledge Base + History + Policy → Ranked proposals        │  │
│  │ Each proposal: (optimization, context, invariant, score)    │  │
│  └─────────────────────────┬──────────────────────────────────┘  │
│                             │                                      │
│  ┌─────────────────────────▼──────────────────────────────────┐  │
│  │ Select (Optimization Selector)                              │  │
│  │ Sample from distribution, resolve dependencies              │  │
│  └─────────────────────────┬──────────────────────────────────┘  │
│                             │                                      │
│  ┌─────────────────────────▼──────────────────────────────────┐  │
│  │ Lower (Lowering Agent) — inner loop per step                │  │
│  │ Implement transformations + insert tag functions/assertions │  │
│  └─────────────────────────┬──────────────────────────────────┘  │
│                             │                                      │
│  ┌─────────────────────────▼──────────────────────────────────┐  │
│  │ Validate                                                    │  │
│  │ • Compile-time invariant checking (→ counterexamples)       │  │
│  │ • Unit tests (functional correctness)                       │  │
│  │ • Runtime profiling (throughput measurement)                │  │
│  │ • Reward = f(invariants, correctness, performance)          │  │
│  └─────────────────────────┬──────────────────────────────────┘  │
│                             │                                      │
│  ┌─────────────────────────▼──────────────────────────────────┐  │
│  │ Learn (ICRL Policy Update)                                  │  │
│  │ PolicyEval → Analyze (text gradients) → ParameterUpdate     │  │
│  │ Dense reward from invariant violations guides learning      │  │
│  └─────────────────────────┬──────────────────────────────────┘  │
│                             │                                      │
│                             ▼ (next ICRL iteration)                │
└────────────────────────────────────────────────────────────────────┘
```

### Knowledge Base Categories (ARGUS Table 1)

| Category | Optimizations | Scope |
|----------|--------------|-------|
| **Global Intrusive** | Software pipelining, Split K, MFMA Matmul, Stagger K, Async memcpy | Restructure entire kernel |
| **Local Source** | Bank conflict mitigation, Vectorized loads, Loop unrolling, Workgroup swizzling | Small localized patches |
| **ISA-Specific** | HW OOB-guarded loads, AGPR usage, Instruction scheduling | Hardware intrinsics |

## Prerequisites

### Required

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **Claude Code** | Workflow runtime | [claude.ai/claude-code](https://claude.ai/claude-code) |
| **GPU Hardware** | AMD MI300X / NVIDIA H100 | Target execution platform |
| **Compiler Toolchain** | ROCm (AMD) or CUDA (NVIDIA) | Kernel compilation and profiling |
| **Python 3.10+** | Kernel execution, test harness | For DSL front-end and benchmarking |

### Claude Code Skills (Recommended)

| Skill | Purpose |
|-------|---------|
| `ncu-report-skill` | NCU profiling for NVIDIA targets |
| `KernelWiki` | Hopper/Blackwell optimization knowledge |
| `kernel-auto-tester` | Test infrastructure generation |

## Usage

```javascript
Workflow({name: 'argus-kernel-optimization', args: {
  kernel_path: '/path/to/kernel.py',
  kernel_spec: 'Flash attention GQA, bf16, d=128, Br=256, Bc=64, 512 threads',
  hardware_target: 'AMD MI300X',
  test_command: 'python test_kernel.py',
  benchmark_command: 'python bench_kernel.py --warmup 100 --iters 1000',
  iterations: 10,
  inner_steps: 5,
  optimization_categories: ['global_intrusive', 'local_source', 'isa_specific'],
}})
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_path` | (required) | Path to kernel source file |
| `kernel_spec` | `''` | Natural language specification of the kernel |
| `hardware_target` | `'NVIDIA H100'` | Target GPU architecture |
| `test_command` | `''` | Correctness test command |
| `benchmark_command` | `''` | Performance benchmark command |
| `iterations` | `5` | ICRL outer loop iterations |
| `inner_steps` | `3` | Max optimization steps per iteration |
| `exp_dir` | `'/tmp/argus_exp'` | Experiment artifacts directory |
| `optimization_categories` | `['global_intrusive', 'local_source', 'isa_specific']` | Which knowledge base categories to explore |
| `knowledge_base_path` | `''` | Custom knowledge base file |

## Output

```javascript
{
  computation_type: string,
  hardware_target: string,
  baseline_throughput_tflops: number,
  best_throughput_tflops: number,
  overall_speedup: number,
  iterations_completed: number,
  optimization_steps: number,
  optimization_history: [...],
  invariant_violations_total: number,
  planner_policy_final: string,
  candidate_beam: [{label, throughput}],
  best_kernel_code: string,
  report: string,
}
```

## Comparison with AccelOpt and KEET

| Aspect | ARGUS | AccelOpt | KEET |
|--------|-------|----------|------|
| Goal | Near-library throughput | Iterative speedup | Understand performance |
| Feedback | Data-flow invariant violations (dense) | NCU metrics + slow-fast pairs | NCU → explanation report |
| Learning | ICRL with text gradients | Experience memory accumulation | No learning (single-pass) |
| Target | GEMM/Attention/MoE (peak perf) | General CUDA kernels | Any profiled kernel |
| Abstraction | Tile-based DSL | Raw CUDA | Raw CUDA |
| Hardware | AMD MI300X (+ generalizable) | NVIDIA (NCU-based) | NVIDIA (NCU-based) |

## References

- [ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants](https://arxiv.org/abs/2604.18616) — Mai, Guo, Ding, Li, Yu, Guo, Wang, Zhao, Kozyrakis, Yuan (2026)
- [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496) — ICRL gradient computation method used by ARGUS

---

# ARGUS：基于数据流不变量的 GPU 内核优化 Agent 框架

使用**数据流不变量**作为密集结构化反馈、结合**上下文强化学习 (ICRL)** 进行自适应规划的迭代式 GPU 内核优化工作流。实现了 [ARGUS](https://arxiv.org/abs/2604.18616) 方法论（Mai 等人，CausalFlow / 港科大 / 斯坦福，2026）。

## 概述

ARGUS 弥补了 LLM 生成内核与手工优化汇编库（HipBLASLt、AITER）之间的性能差距。现有 agent 通过单元测试实现正确性，但缺乏引导复杂全局耦合优化（如软件流水线、指令调度）所需的**密集反馈**。

### 核心创新：数据流不变量 = 密集反馈

```
Tag 函数 → 符号标注 → Tag 断言 → 违规反例
```

取代稀疏的通过/失败测试：
- **Tag 函数**：将符号坐标附加到张量元素上
- **Tag 断言**：在使用点强制执行关系约束（如"MFMA 操作数必须来自匹配的 K 切片"）
- **违规**：生成具体反例——（线程, 元素, 程序点）
- 为 agent 提供**精确的结构化反馈**以进行定向修复

### 论文结果

在 AMD MI300X 上，ARGUS 生成的内核达到：
- 手工优化汇编吞吐量的 **99-104%**
- 比现有 agent 系统快 **2-1543 倍**
- KernelBench Level 1 **100%** Pass@1，Level 2 **90%**

## 前置依赖

### 必需

| 依赖 | 用途 | 安装方式 |
|------|------|----------|
| **Claude Code** | 工作流运行时 | [claude.ai/claude-code](https://claude.ai/claude-code) |
| **GPU 硬件** | AMD MI300X / NVIDIA H100 | 目标执行平台 |
| **编译工具链** | ROCm (AMD) 或 CUDA (NVIDIA) | 内核编译和性能分析 |
| **Python 3.10+** | 内核执行、测试框架 | DSL 前端和基准测试 |

### Claude Code Skills（推荐）

| Skill | 用途 |
|-------|------|
| `ncu-report-skill` | NVIDIA 目标的 NCU 性能分析 |
| `KernelWiki` | Hopper/Blackwell 优化知识 |
| `kernel-auto-tester` | 测试基础设施生成 |

## 使用方法

```javascript
Workflow({name: 'argus-kernel-optimization', args: {
  kernel_path: '/path/to/kernel.py',
  kernel_spec: 'Flash attention GQA, bf16, d=128, Br=256, Bc=64',
  hardware_target: 'AMD MI300X',
  test_command: 'python test_kernel.py',
  benchmark_command: 'python bench_kernel.py',
  iterations: 10,
  inner_steps: 5,
}})
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kernel_path` | (必需) | 内核源文件路径 |
| `kernel_spec` | `''` | 内核的自然语言规格描述 |
| `hardware_target` | `'NVIDIA H100'` | 目标 GPU 架构 |
| `test_command` | `''` | 正确性测试命令 |
| `benchmark_command` | `''` | 性能基准命令 |
| `iterations` | `5` | ICRL 外层循环次数 |
| `inner_steps` | `3` | 每轮最大优化步数 |
| `optimization_categories` | 全部三类 | 探索哪些知识库类别 |

## 与 AccelOpt / KEET 的对比

| 方面 | ARGUS | AccelOpt | KEET |
|------|-------|----------|------|
| 目标 | 达到库级吞吐 | 迭代加速 | 理解性能 |
| 反馈 | 不变量违规（密集） | NCU 指标 + 慢快对 | NCU → 解释报告 |
| 学习 | ICRL + 文本梯度 | 经验记忆累积 | 无学习（单遍） |
| 目标场景 | GEMM/Attention/MoE | 通用 CUDA 内核 | 任意已分析内核 |
| 抽象层次 | Tile-based DSL | 原始 CUDA | 原始 CUDA |

## 参考文献

- [ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants](https://arxiv.org/abs/2604.18616) — Mai, Guo, Ding, Li, Yu, Guo, Wang, Zhao, Kozyrakis, Yuan (2026)
- [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496) — ARGUS 使用的 ICRL 梯度计算方法
