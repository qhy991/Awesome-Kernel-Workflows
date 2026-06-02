# KernelFoundry: Hardware-Aware Evolutionary GPU Kernel Optimization

**English** · [简体中文](README.zh-CN.md)

MAP-Elites quality-diversity search with meta-prompt evolution and templated parameter tuning for GPU kernel generation. Implements the [KernelFoundry](https://arxiv.org/abs/2603.12440) methodology (Wiedemann et al., Intel Corporation, 2026).

## Overview

KernelFoundry prevents **mode collapse** and **context degradation** — two failure modes of iterative LLM kernel generation — through evolutionary quality-diversity search. Instead of converging on one solution, it maintains a diverse archive of high-performing kernels across different optimization strategies.

### Three Key Mechanisms

1. **MAP-Elites with Kernel-Specific Behavioral Descriptors**
   - 3D grid (4×4×4 = 64 cells) indexed by optimization characteristics
   - Each cell holds the best kernel for that specific strategy combination
   - Diversity is maintained by construction — cells evolve independently

2. **Meta-Prompt Evolution**
   - 4 evolvable prompt sections co-evolve WITH the kernels
   - A separate meta-prompter LLM analyzes outcomes and edits guidance
   - Prevents context degradation by pruning failed advice

3. **Templated Parameter Optimization**
   - Kernels can declare configurable parameters (tile sizes, work-group dims)
   - Each configuration evaluated independently
   - Separates algorithmic search from parameter tuning

### Behavioral Descriptors

| Dimension | Level 0 | Level 1 | Level 2 | Level 3 |
|-----------|---------|---------|---------|---------|
| **d_mem** (Memory) | Scalar/strided | Coalesced/vectorized | Shared memory + tiling | Multi-level hierarchy |
| **d_algo** (Algorithm) | Direct translation | Fused operations | Reformulated (flash pattern) | Novel algorithm |
| **d_sync** (Parallelism) | No sync | Work-group barriers | Sub-group primitives | Global coordination |

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| **Claude Code** | Workflow runtime |
| **GPU + Compiler** | SYCL (Intel DPC++), CUDA (nvcc), or Triton |
| **Python + PyTorch** | Reference implementation and benchmarking |

### Claude Code Skills (Recommended)

| Skill | Purpose |
|-------|---------|
| `ncu-report-skill` | Profiling for NVIDIA targets |
| `kernel-auto-tester` | Correctness validation |
| `KernelWiki` | Architecture-specific optimization knowledge |

## Usage

```javascript
Workflow({name: 'kernelfoundry-kernel-optimization', args: {
  task_spec: 'class Model(nn.Module):\n  def forward(self, x):\n    return F.softmax(x, dim=-1)',
  op_description: 'Softmax over last dimension',
  target_language: 'sycl',              // 'sycl', 'cuda', 'triton'
  target_hardware: 'Intel Arc B580',
  test_command: 'python test_kernel.py',
  benchmark_command: 'python bench_kernel.py',
  generations: 40,
  meta_prompt_interval: 10,
  speedup_target: 2.0,
}})
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_spec` | `''` | PyTorch operator source code |
| `op_description` | `'GPU kernel'` | Operation description |
| `target_language` | `'cuda'` | Target: sycl, cuda, or triton |
| `target_hardware` | `'NVIDIA GPU'` | Target GPU platform |
| `test_command` | `''` | Correctness test command |
| `benchmark_command` | `''` | Performance benchmark command |
| `generations` | `30` | Number of evolutionary generations |
| `meta_prompt_interval` | `10` | Generations between prompt evolution |
| `speedup_target` | `2.0` | Target speedup for fitness normalization |
| `selection_strategy` | `'mixed'` | Parent selection: mixed/uniform/fitness/curiosity |

## Output

```javascript
{
  best_speedup: number,
  best_cell: string,               // e.g. "2,1,2"
  best_kernel_code: string,
  archive_coverage: number,        // occupied cells out of 64
  archive_summary: [...],          // top cells with strategies
  improvements: number,
  discoveries: number,
  final_meta_prompt: {...},        // evolved prompt sections
  report: string,
}
```

## References

- [KernelFoundry: Hardware-aware evolutionary GPU kernel optimization](https://arxiv.org/abs/2603.12440) — Wiedemann, Leboutet, Paulitsch, Wofk, Ummenhofer (Intel, 2026)
- [MAP-Elites](https://arxiv.org/abs/1504.04909) — Mouret & Clune (2015) — foundational QD algorithm

---

# KernelFoundry：硬件感知的进化式 GPU 内核优化

MAP-Elites 质量-多样性搜索 + 元提示词进化 + 模板化参数调优的 GPU 内核生成框架。实现了 [KernelFoundry](https://arxiv.org/abs/2603.12440) 方法论（Wiedemann 等人，Intel，2026）。

## 概述

KernelFoundry 通过进化式质量-多样性搜索防止**模式坍缩**和**上下文退化**。它维护一个多样化的高性能内核档案，覆盖不同优化策略，而非收敛到单一解。

### 三个核心机制

1. **MAP-Elites + 内核专用行为描述符**：4×4×4 = 64 个行为单元，各自独立进化
2. **元提示词进化**：4 个可进化提示词段落与内核共同进化，防止上下文退化
3. **模板化参数优化**：将算法搜索与参数调优分离

### 行为描述符

| 维度 | Level 0 | Level 1 | Level 2 | Level 3 |
|------|---------|---------|---------|---------|
| **d_mem** | 标量/跨步 | 合并/向量化 | 共享内存 + 分块 | 多级存储层次 |
| **d_algo** | 直接翻译 | 融合操作 | 重构算法 | 新颖算法 |
| **d_sync** | 无同步 | 工作组 barrier | 子组原语 | 全局协调 |

## 使用方法

```javascript
Workflow({name: 'kernelfoundry-kernel-optimization', args: {
  task_spec: 'class Model(nn.Module): ...',
  op_description: 'Softmax',
  target_language: 'sycl',
  target_hardware: 'Intel Arc B580',
  generations: 40,
}})
```

## 参考文献

- [KernelFoundry](https://arxiv.org/abs/2603.12440) — Wiedemann 等 (Intel, 2026)
- [MAP-Elites](https://arxiv.org/abs/1504.04909) — Mouret & Clune (2015)
