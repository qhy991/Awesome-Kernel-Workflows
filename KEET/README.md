# KEET: Kernel Execution Explanation Toolkit

**English** · [简体中文](README.zh-CN.md)

An LLM-agent pipeline for interpreting Nsight Compute (NCU) profiles into actionable, data-grounded natural language performance explanations. Implements the [KEET](https://arxiv.org/abs/2605.04467) methodology (Davis et al., UMD / NVIDIA / LLNL, 2026).

## Overview

KEET answers the question: **"WHY does this kernel perform the way it does?"**

Unlike optimization workflows that try to make kernels faster, KEET focuses on *understanding* — producing interpretable explanations that identify bottlenecks, cite specific metrics, and suggest targeted code changes. The output can be used standalone or as context for downstream optimization tasks.

### Key Innovation: Hypothesis-First Analysis

```
Source Code → Performance Hypotheses → NCU Profile Data → Confirm/Refute → Explanation
```

KEET generates predictions from code alone BEFORE seeing profiling data, then verifies against hardware measurements. This:
- Prevents confirmation bias
- Produces explanations with clear provenance ("I predicted X because of code Y, and metric Z confirms/refutes it")
- Makes the analysis process interpretable and auditable

## Architecture

```
                    ┌────────────────────────────────────────────────────┐
                    │                     Setup                           │
                    │  Read source files + Extract NCU profile data       │
                    └───────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────▼────────────────────────────┐
                    │           Source Code Inspection                    │
                    │  • Describe source files                           │
                    │  • Summarize algorithm                             │
                    │  • Generate performance hypotheses (NO data yet!)  │
                    └───────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────▼────────────────────────────┐
                    │            Profile Inspection                       │
                    │  • Select relevant metrics (hypothesis-informed)   │
                    │  • Produce grounded analysis (cite all claims)     │
                    │  • [Optional] DrGPU rule-based evaluation          │
                    └───────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────▼────────────────────────────┐
                    │              Aggregation                            │
                    │  Combine all analyses → unified explanation report  │
                    └───────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────▼────────────────────────────┐
                    │                Review                               │
                    │  Cross-check hypotheses: Confirmed / Refuted /     │
                    │  Inconclusive with evidence                         │
                    └────────────────────────────────────────────────────┘
```

### Agent Roles (from KEET paper Section III)

| Agent | Stage | Role |
|-------|-------|------|
| Source File Describer | Source Inspection | Catalog source files and their structure |
| Algorithm Summarizer | Source Inspection | Summarize the parallelization strategy |
| Performance Hypothesizer | Source Inspection | Predict bottlenecks from code alone |
| Metric Selector | Profile Inspection | Choose most relevant NCU metrics |
| Profile Analyzer | Profile Inspection | Produce grounded performance analysis |
| DrGPU Evaluator | Profile Inspection | (Optional) Rule-based stall tree analysis |
| Analysis Aggregator | Aggregation | Combine analyses into final report |
| Explanation Reviewer | Review | Confirm/refute hypotheses against data |

## Prerequisites

### Required

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **Claude Code** | Workflow runtime | [claude.ai/claude-code](https://claude.ai/claude-code) or `npm install -g @anthropic-ai/claude-code` |
| **NVIDIA CUDA Toolkit** | `nvcc` compiler | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| **Nsight Compute (ncu)** | Kernel profiling | Bundled with CUDA Toolkit (≥11.0) |
| **NVIDIA GPU** | H100 / A100 / V100 etc. | Physical or cloud |

### Claude Code Skills (Recommended)

| Skill | Purpose |
|-------|---------|
| `ncu-report-skill` | Structured NCU report parsing, metric extraction, analysis dimensions |
| `kernel-auto-tester` | Post-analysis validation of suggested optimizations |
| `KernelWiki` | Architecture-specific optimization knowledge for H100/B200 |

### Environment

```bash
nvcc --version        # CUDA ≥ 11.0
ncu --version         # NCU ≥ 2022.1
nvidia-smi            # GPU accessible
claude --version      # Claude Code installed
```

## Usage

### Basic: Explain a single kernel profile

```javascript
Workflow({name: 'keet-kernel-explanation', args: {
  kernel_path: '/path/to/kernel.cu',
  ncu_report_path: '/path/to/report.ncu-rep',
  op_description: 'Fan2 kernel from gaussian application',
}})
```

### With multiple source files + DrGPU integration

```javascript
Workflow({name: 'keet-kernel-explanation', args: {
  kernel_path: '/path/to/main_kernel.cu',
  source_paths: ['/path/to/utils.cuh', '/path/to/types.h'],
  ncu_report_path: '/path/to/report.ncu-rep',
  op_description: 'LULESH hydrodynamics kernel',
  run_metadata: 'Block: 128, Grid: (256,1,1), Regs: 64/thread',
  include_drgpu: true,
}})
```

### Multi-profile analysis (tuning knob sweep)

```javascript
Workflow({name: 'keet-kernel-explanation', args: {
  kernel_path: '/path/to/kernel.cu',
  ncu_report_path: '/path/to/default_config.ncu-rep',
  additional_profiles: [
    '/path/to/block64.ncu-rep',
    '/path/to/block256.ncu-rep',
    '/path/to/block512.ncu-rep',
  ],
  op_description: 'XSBench cross-section lookup',
  run_metadata: 'Varying block size: 64, 128, 256, 512',
}})
```

### Generate profile on-the-fly

```javascript
Workflow({name: 'keet-kernel-explanation', args: {
  kernel_path: '/path/to/kernel.cu',
  harness_build_cmd: 'nvcc -O3 -lineinfo -arch=sm_90 -o bench harness.cu',
  harness_run_args: '--size 4096',
  kernel_name_regex: 'my_kernel',
  op_description: 'Custom reduction kernel',
  exp_dir: '/tmp/keet_analysis',
}})
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_path` | (required) | Path to the CUDA kernel `.cu` file |
| `ncu_report_path` | `''` | Existing `.ncu-rep` file (generates one if empty) |
| `op_description` | `'CUDA kernel'` | Human description of the operation |
| `source_paths` | `[]` | Additional source files to analyze |
| `ncu_binary` | `'ncu'` | Path to NCU CLI |
| `exp_dir` | `'/tmp/keet_exp'` | Directory for artifacts |
| `kernel_name_regex` | `''` | Regex for `ncu -k` kernel selection |
| `run_metadata` | `''` | Launch configuration description |
| `include_drgpu` | `false` | Enable DrGPU rule-based analysis |
| `additional_profiles` | `[]` | Extra `.ncu-rep` paths for multi-profile mode |
| `analysis_guidelines_path` | `''` | Custom performance analysis guidelines |
| `harness_build_cmd` | `''` | Build command (for generating profiles) |
| `harness_run_args` | `''` | Runtime args for harness |

## Output

```javascript
{
  operation: string,              // What was analyzed
  algorithm_summary: string,      // How the kernel works
  performance_hypotheses: [...],  // Predictions from code alone
  hypothesis_verdicts: [...],     // Confirmed/Refuted/Inconclusive
  primary_bottleneck: string,     // Main performance limiter
  bottleneck_list: [...],         // All bottlenecks with citations
  optimization_suggestions: [...],// Prioritized suggestions
  full_report: string,            // Complete explanation report
  review_summary: string,         // Reviewer assessment
  review_quality: string,         // Overall quality rating
  drgpu_suggestions: [...] | null,// DrGPU output (if enabled)
  profiles_analyzed: number,      // How many profiles were used
}
```

## Example Output (from paper)

For the gaussian Fan2 kernel:
```
## Summary of main bottlenecks

1. **Memory-latency bound, not bandwidth bound**
   - Long Scoreboard stalls dominate ('21.39' per issue).
   - L2 hit rate is high; 'dram__throughput' is only 1.77%

2. **Very poor global memory coalescing**
   - 'smsp__...bytes_per_sector_mem_global_op_ld = 8.34 B/sector' (ideal is 32 B/sector).
   - 'derived__...sectors_global_excessive = 374,514' sectors.

3. **Low warp and thread-level utilization**
   - Blocks have 16 threads (half-warp).
   - Average active threads per issued inst ~= 15.15 (~50%)
   - 'sm__warps_active' only 10.4%

4. **Algorithmic tail and wasted threads**
   - Fixed grid size independent of pivot index 't'
   leads to many threads/blocks doing no useful work for larger 't'.
```

## Comparison with AccelOpt

| Aspect | KEET | AccelOpt |
|--------|------|----------|
| Goal | **Understand** performance | **Optimize** performance |
| Output | Explanation report | Faster kernel code |
| Loop | Single-pass pipeline | Iterative self-improving |
| NCU usage | Interpret existing profiles | Drive optimization decisions |
| Downstream | Context for optimization LLM | Direct code generation |

**Complementary usage**: Run KEET first to understand the kernel, then feed its report into AccelOpt's planner as additional context.

## References

- [KEET: Explaining Performance of GPU Kernels Using LLM Agents](https://arxiv.org/abs/2605.04467) — Davis, Rydzy, Ramesh, Nilay, Nichols, Raj, Jain, Bhatele (2026)
- [DrGPU: A Top-Down Profiler for GPU Programs](https://arxiv.org/abs/2404.02095) — Related tool for rule-based stall analysis

---

# KEET：GPU 内核性能解释工具

基于 LLM Agent 流水线的 Nsight Compute (NCU) 性能分析报告解释工具，实现了 [KEET](https://arxiv.org/abs/2605.04467) 方法论（Davis 等人，马里兰大学 / NVIDIA / LLNL，2026）。

## 概述

KEET 回答的核心问题是：**"这个内核为什么会有这样的性能表现？"**

与优化工作流不同，KEET 专注于*理解*——生成可解释的性能分析报告，识别瓶颈，引用具体指标，并建议针对性的代码修改。输出可独立使用，也可作为下游优化任务的上下文。

### 核心创新：假设优先分析

```
源代码 → 性能假设 → NCU 分析数据 → 验证/否定 → 性能解释
```

KEET 在看到分析数据之前先从代码生成预测，然后用硬件测量值验证。这样做：
- 防止确认偏误
- 生成有明确来源的解释（"我基于代码 Y 预测了 X，指标 Z 证实/否定了它"）
- 使分析过程可解释、可审计

## 架构

```
                    ┌────────────────────────────────────────────────────┐
                    │                     初始化                          │
                    │  读取源文件 + 提取 NCU 分析数据                      │
                    └───────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────▼────────────────────────────┐
                    │              源代码检查                              │
                    │  • 描述源文件结构                                    │
                    │  • 总结算法                                          │
                    │  • 生成性能假设（此时还未看数据！）                    │
                    └───────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────▼────────────────────────────┐
                    │              Profile 检查                            │
                    │  • 选择相关指标（基于假设引导）                       │
                    │  • 生成有据可查的性能分析                             │
                    │  • [可选] DrGPU 规则分析                             │
                    └───────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────▼────────────────────────────┐
                    │                聚合                                  │
                    │  合并所有分析 → 统一的性能解释报告                     │
                    └───────────────────────┬────────────────────────────┘
                                            │
                    ┌───────────────────────▼────────────────────────────┐
                    │                审查                                  │
                    │  对照假设：已确认 / 已否定 / 不确定（附证据）           │
                    └────────────────────────────────────────────────────┘
```

### Agent 角色（来自 KEET 论文第 III 节）

| Agent | 阶段 | 角色 |
|-------|------|------|
| 源文件描述器 | 源代码检查 | 编目源文件及其结构 |
| 算法总结器 | 源代码检查 | 总结并行化策略 |
| 性能假设器 | 源代码检查 | 仅从代码预测瓶颈 |
| 指标选择器 | Profile 检查 | 选择最相关的 NCU 指标 |
| Profile 分析器 | Profile 检查 | 生成有据可查的性能分析 |
| DrGPU 评估器 | Profile 检查 | （可选）基于规则的停顿树分析 |
| 分析聚合器 | 聚合 | 合并分析为最终报告 |
| 解释审查器 | 审查 | 用数据验证假设 |

## 前置依赖

### 必需

| 依赖 | 用途 | 安装方式 |
|------|------|----------|
| **Claude Code** | 工作流运行时 | [claude.ai/claude-code](https://claude.ai/claude-code) 或 `npm install -g @anthropic-ai/claude-code` |
| **NVIDIA CUDA Toolkit** | `nvcc` 编译器 | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| **Nsight Compute (ncu)** | 内核性能分析 | 随 CUDA Toolkit (≥11.0) 捆绑 |
| **NVIDIA GPU** | H100 / A100 / V100 等 | 物理 GPU 或云实例 |

### Claude Code Skills（推荐）

| Skill | 用途 |
|-------|------|
| `ncu-report-skill` | 结构化 NCU 报告解析、指标提取、分析维度 |
| `kernel-auto-tester` | 分析后验证建议的优化 |
| `KernelWiki` | H100/B200 架构特定优化知识 |

## 使用方法

### 基本：解释单个内核 profile

```javascript
Workflow({name: 'keet-kernel-explanation', args: {
  kernel_path: '/path/to/kernel.cu',
  ncu_report_path: '/path/to/report.ncu-rep',
  op_description: 'gaussian 应用的 Fan2 内核',
}})
```

### 多 profile 分析（调优参数扫描）

```javascript
Workflow({name: 'keet-kernel-explanation', args: {
  kernel_path: '/path/to/kernel.cu',
  ncu_report_path: '/path/to/default_config.ncu-rep',
  additional_profiles: [
    '/path/to/block64.ncu-rep',
    '/path/to/block256.ncu-rep',
  ],
  op_description: 'XSBench 截面查找',
  run_metadata: '变化的 block size: 64, 128, 256',
}})
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kernel_path` | (必需) | CUDA 内核 `.cu` 文件路径 |
| `ncu_report_path` | `''` | 已有的 `.ncu-rep` 文件（为空则自动生成） |
| `op_description` | `'CUDA kernel'` | 算子的可读描述 |
| `source_paths` | `[]` | 主内核之外的额外源文件 |
| `run_metadata` | `''` | 启动配置描述 |
| `include_drgpu` | `false` | 启用 DrGPU 规则分析 |
| `additional_profiles` | `[]` | 多 profile 模式的额外 `.ncu-rep` 路径 |

## 与 AccelOpt 的对比

| 方面 | KEET | AccelOpt |
|------|------|----------|
| 目标 | **理解**性能 | **优化**性能 |
| 输出 | 解释报告 | 更快的内核代码 |
| 循环 | 单遍流水线 | 迭代自改进 |
| NCU 用途 | 解释已有 profile | 驱动优化决策 |
| 下游 | 作为优化 LLM 的上下文 | 直接生成代码 |

**互补使用**：先运行 KEET 理解内核，再将其报告送入 AccelOpt 的规划器作为额外上下文。

## 参考文献

- [KEET: Explaining Performance of GPU Kernels Using LLM Agents](https://arxiv.org/abs/2605.04467) — Davis, Rydzy, Ramesh, Nilay, Nichols, Raj, Jain, Bhatele (2026)
- [DrGPU: A Top-Down Profiler for GPU Programs](https://arxiv.org/abs/2404.02095) — 相关的基于规则的停顿分析工具
