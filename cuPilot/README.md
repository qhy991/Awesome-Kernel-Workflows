# cuPilot: Strategy-Coordinated Multi-Agent Framework for CUDA Kernel Evolution

**English** · [简体中文](README.zh-CN.md)

Evolutionary kernel optimization using **strategy as an intermediate semantic representation** for crossover, with roofline-guided prompting and RAG-based population initialization. Implements the [cuPilot](https://arxiv.org/abs/2512.16465) methodology (Chen, Wu et al., Southeast University / Tsinghua / Tsing Micro, 2025).

## Overview

cuPilot identifies three mismatches in existing LLM-based kernel evolution frameworks and addresses each:

| Mismatch | Problem | cuPilot Solution |
|----------|---------|------------------|
| **Crossover representation** | Code-level crossover forces LLM to traverse strategy identification → combination → synthesis in one shot; degrades as complexity grows | Strategy-level crossover: evolve optimization ideas separately from code |
| **Fitness representation** | Raw speedup has weak semantic correlation with bottlenecks | Roofline-guided prompting: classify as compute/memory-bound, focus on relevant metrics |
| **Population initialization** | Sparse code-level init leads to premature convergence | Strategy-level RAG: retrieve historical (kernel, optimized_kernel, strategy) triples |

### Results (from paper)

- **3.09x** average speedup over PyTorch on 100 KernelBench kernels
- **4.06x** on GEMM kernels with sophisticated optimizations (tensor cores, swizzling, multi-stage pipeline)
- Outperforms AI CUDA Engineer on hardware utilization metrics

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Setup                                    │
│  Kernel Generator → initial kernel                             │
│  Roofline Prophet → compute/memory/middle-zone classification  │
│  Strategy Pool → RAG from historical optimizations             │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│            Evolutionary Loop (Epochs × Generations)            │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ SCE Manager (Strategize)                              │     │
│  │ Tournament selection → strategy-level crossover       │     │
│  │ (combine optimization IDEAS, not code)                │     │
│  └───────────────────────┬──────────────────────────────┘     │
│                           │                                    │
│  ┌───────────────────────▼──────────────────────────────┐     │
│  │ Strategy Translator (Translate)                       │     │
│  │ Apply each strategy to kernel code (strategy→code)    │     │
│  └───────────────────────┬──────────────────────────────┘     │
│                           │                                    │
│  ┌───────────────────────▼──────────────────────────────┐     │
│  │ Kernel Revisor (Revise)                               │     │
│  │ NVCC check → Function check → NCU profile → Fix      │     │
│  │ (loop until correct or max attempts)                   │     │
│  └───────────────────────┬──────────────────────────────┘     │
│                           │                                    │
│  ┌───────────────────────▼──────────────────────────────┐     │
│  │ Evolve                                                │     │
│  │ Tournament + elitism → strategy alignment → next gen  │     │
│  └───────────────────────┬──────────────────────────────┘     │
│                           ▼ (next generation)                  │
└────────────────────────────────────────────────────────────────┘
```

### Multi-Agent Roles

| Agent | Level | Role |
|-------|-------|------|
| Roofline Prophet | High | Classify kernel on roofline → guide metric focus |
| SCE Manager | High | Population management, strategy crossover, tournament selection |
| Strategy Translator | Mid | Bridge: apply strategy description to kernel code |
| Kernel Revisor | Low | Compile check → function check → NCU profiling → fix loop |
| Kernel Generator | Low | Produce initial vanilla kernel |

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| **Claude Code** | Workflow runtime |
| **NVIDIA GPU + CUDA** | Kernel compilation (nvcc) and execution |
| **NCU** | Performance profiling for the Kernel Revisor |

### Claude Code Skills (Recommended)

| Skill | Purpose |
|-------|---------|
| `ncu-report-skill` | NCU profiling and bottleneck analysis |
| `KernelWiki` | Architecture-specific optimization knowledge |

## Usage

```javascript
Workflow({name: 'cupilot-kernel-optimization', args: {
  kernel_spec: 'class Model(nn.Module):\n  def forward(self, A, B):\n    return torch.matmul(A, B)',
  op_description: 'Standard GEMM (M×N×K, bf16)',
  target_gpu: 'A100',
  compile_command: '<user-provided compile command with {kernel_path}/{result_path}>',
  test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
  ncu_command: '<user-provided profiler command with {kernel_path}/{result_path}>',
  epochs: 3,
  generations_per_epoch: 4,
  population_size: 50,
}})
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_spec` | `''` | PyTorch operator code or description |
| `op_description` | `'CUDA kernel'` | Operation description |
| `target_gpu` | `'A100'` | Target GPU architecture |
| `compile_command` | `''` | nvcc compilation command |
| `test_command` | `''` | Correctness test command |
| `ncu_command` | `''` | NCU profiling command |
| `epochs` | `2` | Number of evolutionary epochs |
| `generations_per_epoch` | `4` | Generations per epoch |
| `population_size` | `30` | Max population size |
| `strategy_pool_path` | `''` | Historical strategy pool for RAG |
| `max_revise_loops` | `3` | Max revision iterations per kernel |

## References

- [cuPilot: A Strategy-Coordinated Multi-agent Framework for CUDA Kernel Evolution](https://arxiv.org/abs/2512.16465) — Chen, Wu, Li, Ma, Si, Hu, Yin, Yang (2025)
- [cuPilot Kernels (open-source)](https://github.com/champloo2878/cuPilot-Kernels)

---

# cuPilot：策略协调的多 Agent CUDA 内核进化框架

**English** · 简体中文

使用**策略作为中间语义表示**进行进化交叉的内核优化，结合 roofline 引导提示和基于 RAG 的种群初始化。实现了 [cuPilot](https://arxiv.org/abs/2512.16465) 方法论（Chen, Wu 等人，东南大学 / 清华 / 清微智能，2025）。

## 概述

cuPilot 识别了现有 LLM 内核进化框架的三个不匹配问题：

| 不匹配 | 问题 | cuPilot 方案 |
|--------|------|-------------|
| **交叉表示** | 代码级交叉要求 LLM 一步完成策略识别→组合→合成 | 策略级交叉：将优化思想与代码解耦 |
| **适应度表示** | 原始加速比与瓶颈语义关联弱 | Roofline 引导：分类为计算/内存受限，聚焦相关指标 |
| **种群初始化** | 稀疏的代码级初始化导致过早收敛 | 策略级 RAG：检索历史（内核,优化内核,策略）三元组 |

### 论文结果

- 100 个 KernelBench 内核平均加速 **3.09x**
- GEMM 内核 **4.06x**（使用 tensor core、swizzling、多级流水线）

## 使用方法

```javascript
Workflow({name: 'cupilot-kernel-optimization', args: {
  kernel_spec: 'class Model(nn.Module): ...',
  op_description: 'Standard GEMM',
  target_gpu: 'A100',
  epochs: 3,
  generations_per_epoch: 4,
}})
```

## 参考文献

- [cuPilot](https://arxiv.org/abs/2512.16465) — Chen, Wu 等 (东南大学/清华/清微智能, 2025)
- [cuPilot Kernels (开源)](https://github.com/champloo2878/cuPilot-Kernels)
