# AccelOpt Kernel Optimization Workflow

**English** · [简体中文](README.zh-CN.md)

A self-improving CUDA kernel optimization workflow powered by Nsight Compute (NCU) profiling, implementing the [AccelOpt](https://arxiv.org/abs/2511.15915) methodology (MLSys 2026).

## Overview

This workflow automates iterative GPU kernel optimization through evidence-based profiling rather than guesswork. It implements AccelOpt's core loop:

```
Plan → Execute → Profile → Summarize → Accumulate Experience → Repeat
```

Each iteration:
1. **Plans** multiple optimization strategies informed by real NCU metrics (stall reasons, memory patterns, occupancy)
2. **Executes** multiple kernel implementations per plan in parallel
3. **Evaluates** all variants with NCU profiling, deduplicates per-plan, updates the candidate beam
4. **Learns** reusable optimization patterns from threshold-filtered slow-fast kernel pairs
5. **Iterates** with growing experience memory and an evolving candidate population

## Key Features

- **NCU-Driven Planning**: Planners receive real profiling data (stall reasons, throughput, occupancy) — never guess bottlenecks
- **Candidate Beam Pool**: Maintains top-K kernels across iterations (not just a single best), enabling cross-pollination of ideas
- **Experience Memory with Sampling**: Accumulated optimization patterns are randomly sampled into prompts, with recent discoveries prioritized
- **Per-Branch Deduplication**: Same plan, multiple samples → only the best survives, preventing homogeneous solutions from filling the beam
- **Threshold-Filtered Learning**: Only sufficiently impactful pairs (positive or negative) are summarized, matching AccelOpt's selection heuristics

## Prerequisites

### Required

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **Claude Code** | Workflow runtime — executes the `.js` workflow script, spawns subagents | [claude.ai/claude-code](https://claude.ai/claude-code) or `npm install -g @anthropic-ai/claude-code` |
| **NVIDIA CUDA Toolkit** | `nvcc` compiler for building kernel variants | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| **Nsight Compute (ncu)** | NCU CLI for hardware-level kernel profiling | Bundled with CUDA Toolkit (≥11.0), or standalone from [developer.nvidia.com/nsight-compute](https://developer.nvidia.com/nsight-compute) |
| **NVIDIA GPU** | Target hardware (H100/A100/RTX 4090 etc.) | Physical GPU or cloud instance (e.g., AWS p5.48xlarge) |

### Claude Code Skills (Recommended)

These skills enhance the workflow's profiling and kernel development capabilities:

| Skill | Purpose | Usage |
|-------|---------|-------|
| `ncu-report-skill` | Structured NCU report parsing, 6-dimension analysis, bottleneck diagnosis playbook | Informs the Setup and Evaluate phases with expert NCU interpretation |
| `kernel-auto-tester` | Automatic test infrastructure generation for CUDA kernels | Can be used post-workflow to validate correctness of the optimized kernel |
| `KernelWiki` | Hopper/Blackwell GPU optimization knowledge (warp specialization, TMEM, tensor cores) | Enhances Plan/Execute phases for H100/B200 targets |

### Environment Setup

```bash
# Verify CUDA toolkit
nvcc --version          # Requires CUDA ≥ 11.0
ncu --version           # Requires NCU ≥ 2022.1

# Verify GPU access
nvidia-smi              # Should show available GPU(s)

# Verify Claude Code
claude --version        # Should show Claude Code version

# (Optional) Verify ncu profiling permissions
# On Linux, ncu requires either root or the following:
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
# Or run ncu with --target-processes all as root
```

### Kernel Requirements

The target `.cu` file must:
- Contain at least one `__global__` kernel function
- Include a `forward()` wrapper function (called by the harness)
- Export via `PYBIND11_MODULE` (or equivalent entry point)
- Be compilable with `nvcc -O3 -lineinfo`

### File Structure (After Running)

```
exp_dir/
├── baseline/
│   ├── harness/bench          # Compiled profiling binary
│   ├── reports/               # .ncu-rep files
│   └── analysis/              # Parsed NCU text output
├── iter_0/
│   ├── plan_0_sample_0/       # Each variant's directory
│   │   └── kernel.cu
│   ├── plan_0_sample_1/
│   └── ...
├── iter_1/
│   └── ...
└── ...
```

## Usage

```javascript
Workflow({name: 'accelopt-kernel-optimization', args: {
  // Required
  kernel_path: '/path/to/kernel.cu',

  // Recommended
  op_description: 'Quantized GEMM Q4_0 weight * FP32 activation',
  harness_path: '/path/to/harness.cu',
  harness_build_cmd: '<user-provided harness build command>',
  harness_run_args: '',
  kernel_name_regex: 'forward_kernel',
  exp_dir: '/path/to/experiment/output',

  // Loop control
  iterations: 3,          // Number of optimization iterations
  breadth: 3,             // Parallel plans per iteration
  samples_per_plan: 2,    // Implementations per plan

  // AccelOpt-aligned parameters
  topk_candidates: 3,           // Candidate beam size
  max_experience_in_prompt: 8,  // Max experience entries injected into planner
  max_threshold: 1.05,          // Min speedup to qualify as positive example
  min_threshold: 1.05,          // Min slowdown to qualify as negative example (1/threshold)
  topk_learn: 5,                // Total pairs to summarize per iteration

  // Optional
  ncu_binary: '<user-provided ncu binary path>',
  test_command: '',
  benchmark_command: '',
}})
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_path` | (required) | Path to the CUDA kernel `.cu` file |
| `op_description` | `'CUDA kernel'` | Human-readable description of the operation |
| `harness_path` | `''` | Pre-built profiling harness binary |
| `harness_build_cmd` | `''` | Build command (use `KERNEL_PATH` as placeholder) |
| `kernel_name_regex` | `''` | Regex for `ncu -k` to select the target kernel |
| `exp_dir` | `'/tmp/accelopt_exp'` | Directory for experiment artifacts |
| `iterations` | `2` | Number of Plan-Execute-Profile-Learn cycles |
| `breadth` | `3` | Number of parallel optimization plans |
| `samples_per_plan` | `2` | Independent implementations per plan |
| `topk_candidates` | `3` | Beam pool size (top-K kernels retained) |
| `max_experience_in_prompt` | `8` | Max patterns sampled into planner context |
| `max_threshold` | `1.05` | Speedup threshold for positive learning pairs |
| `min_threshold` | `1.05` | Slowdown threshold for negative learning pairs |
| `topk_learn` | `5` | Total pairs sent to summarizer per iteration |
| `ncu_binary` | `'ncu'` | Path to Nsight Compute CLI |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Setup Phase                                  │
│  Read kernel → Build harness → NCU --set full → Baseline profile   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────────┐
              │        Iteration Loop              │
              │                                    │
              │  ┌──────────────────────────────┐  │
              │  │ Plan (×BREADTH parallel)      │  │
              │  │ NCU data + beam + experience  │  │
              │  └─────────────┬────────────────┘  │
              │                │                    │
              │  ┌─────────────▼────────────────┐  │
              │  │ Execute (×SAMPLES parallel)   │  │
              │  │ Full .cu implementations      │  │
              │  └─────────────┬────────────────┘  │
              │                │                    │
              │  ┌─────────────▼────────────────┐  │
              │  │ Evaluate + Dedup + Beam       │  │
              │  │ NCU profile → per-plan best   │  │
              │  │ → update candidate beam       │  │
              │  └─────────────┬────────────────┘  │
              │                │                    │
              │  ┌─────────────▼────────────────┐  │
              │  │ Learn (threshold-filtered)    │  │
              │  │ slow-fast pairs → patterns    │  │
              │  │ → experience pool grows       │  │
              │  └─────────────┬────────────────┘  │
              │                │                    │
              │                ▼ (next iteration)   │
              └────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────────┐
              │         Final Report               │
              │  Journey + patterns + beam state   │
              └───────────────────────────────────┘
```

## Output

The workflow returns:

```javascript
{
  baseline_latency_ms: number,
  best_latency_ms: number,
  overall_speedup: number,
  iterations_completed: number,
  candidate_beam: [{plan_title, latency_ms, speedup}],
  experience_patterns_count: number,
  experience_patterns: string[],
  best_kernel_code: string,
  ncu_baseline_profile: string,
  report: string,
}
```

## References

- [AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization](https://arxiv.org/abs/2511.15915) (MLSys 2026)
- [Adaptive Self-improvement LLM Agentic System for ML Library Development](https://proceedings.mlr.press/v267/zhang25at.html) (ICML 2025)

---

# AccelOpt 内核优化工作流

基于 Nsight Compute (NCU) 性能分析的自改进 CUDA 内核优化工作流，实现了 [AccelOpt](https://arxiv.org/abs/2511.15915) 方法论（MLSys 2026）。

## 概述

本工作流通过基于证据的性能分析（而非猜测）来自动化迭代式 GPU 内核优化。它实现了 AccelOpt 的核心循环：

```
规划 → 执行 → 性能分析 → 总结 → 累积经验 → 重复
```

每次迭代：
1. **规划**：基于真实 NCU 指标（停顿原因、内存模式、占用率）生成多个优化策略
2. **执行**：为每个方案并行生成多个内核实现
3. **评估**：用 NCU 分析所有变体，按方案去重，更新候选者 beam 池
4. **学习**：从阈值过滤的慢-快内核对中提取可复用的优化模式
5. **迭代**：将增长的经验记忆和候选群体带入下一轮

## 核心特性

- **NCU 驱动规划**：规划器接收真实的性能分析数据（停顿原因、吞吐量、占用率）—— 绝不猜测瓶颈
- **候选者 Beam 池**：跨迭代维护 top-K 内核（而非仅保留单一最优），促进不同方案间的思想交叉
- **经验记忆采样**：累积的优化模式随机采样注入提示词，最近发现的模式优先
- **逐方案去重**：同一方案的多个采样只保留最优，防止同质方案占满 beam
- **阈值过滤学习**：仅对影响足够大的对（正面或负面）进行总结，与 AccelOpt 的选择启发式一致

## 前置依赖

### 必需

| 依赖 | 用途 | 安装方式 |
|------|------|----------|
| **Claude Code** | 工作流运行时 — 执行 `.js` 工作流脚本，调度子 agent | [claude.ai/claude-code](https://claude.ai/claude-code) 或 `npm install -g @anthropic-ai/claude-code` |
| **NVIDIA CUDA Toolkit** | `nvcc` 编译器，用于构建内核变体 | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| **Nsight Compute (ncu)** | NCU 命令行工具，用于硬件级内核性能分析 | 随 CUDA Toolkit (≥11.0) 捆绑，或从 [developer.nvidia.com/nsight-compute](https://developer.nvidia.com/nsight-compute) 独立安装 |
| **NVIDIA GPU** | 目标硬件（H100/A100/RTX 4090 等） | 物理 GPU 或云实例（如 AWS p5.48xlarge） |

### Claude Code Skills（推荐）

以下 Skills 增强工作流的性能分析和内核开发能力：

| Skill | 用途 | 使用场景 |
|-------|------|----------|
| `ncu-report-skill` | 结构化 NCU 报告解析、6 维分析、瓶颈诊断决策树 | 为 Setup 和 Evaluate 阶段提供专业的 NCU 解读 |
| `kernel-auto-tester` | 为 CUDA 内核自动生成测试基础设施 | 可在工作流完成后验证优化内核的正确性 |
| `KernelWiki` | Hopper/Blackwell GPU 优化知识（warp specialization、TMEM、tensor core） | 为 H100/B200 目标增强 Plan/Execute 阶段 |

### 环境配置

```bash
# 验证 CUDA 工具链
nvcc --version          # 需要 CUDA ≥ 11.0
ncu --version           # 需要 NCU ≥ 2022.1

# 验证 GPU 访问
nvidia-smi              # 应显示可用的 GPU

# 验证 Claude Code
claude --version        # 应显示 Claude Code 版本

# （可选）验证 ncu 性能分析权限
# 在 Linux 上，ncu 需要 root 权限或以下设置：
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
```

### 内核要求

目标 `.cu` 文件必须：
- 包含至少一个 `__global__` 内核函数
- 包含 `forward()` 包装函数（由 harness 调用）
- 通过 `PYBIND11_MODULE`（或等效入口点）导出
- 可用 `nvcc -O3 -lineinfo` 编译

### 文件结构（运行后）

```
exp_dir/
├── baseline/
│   ├── harness/bench          # 编译后的分析二进制
│   ├── reports/               # .ncu-rep 文件
│   └── analysis/              # 解析后的 NCU 文本输出
├── iter_0/
│   ├── plan_0_sample_0/       # 每个变体的目录
│   │   └── kernel.cu
│   ├── plan_0_sample_1/
│   └── ...
├── iter_1/
│   └── ...
└── ...
```

## 使用方法

```javascript
Workflow({name: 'accelopt-kernel-optimization', args: {
  // 必需
  kernel_path: '/path/to/kernel.cu',

  // 推荐
  op_description: '量化 GEMM Q4_0 权重 * FP32 激活',
  harness_path: '/path/to/harness.cu',
  harness_build_cmd: '<user-provided harness build command>',
  harness_run_args: '',
  kernel_name_regex: 'forward_kernel',
  exp_dir: '/path/to/experiment/output',

  // 循环控制
  iterations: 3,          // 优化迭代次数
  breadth: 3,             // 每轮并行方案数
  samples_per_plan: 2,    // 每方案实现数

  // AccelOpt 对齐参数
  topk_candidates: 3,           // 候选者 beam 大小
  max_experience_in_prompt: 8,  // 注入规划器的最大经验条目数
  max_threshold: 1.05,          // 正样本加速比下限
  min_threshold: 1.05,          // 负样本减速比下限（取倒数）
  topk_learn: 5,                // 每轮总结的对数上限

  // 可选
  ncu_binary: '<user-provided ncu binary path>',
}})
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kernel_path` | (必需) | CUDA 内核 `.cu` 文件路径 |
| `op_description` | `'CUDA kernel'` | 算子的可读描述 |
| `harness_path` | `''` | 预构建的性能分析 harness 二进制 |
| `harness_build_cmd` | `''` | 构建命令（用 `KERNEL_PATH` 作占位符） |
| `kernel_name_regex` | `''` | `ncu -k` 选择目标内核的正则表达式 |
| `exp_dir` | `'/tmp/accelopt_exp'` | 实验产物目录 |
| `iterations` | `2` | 规划-执行-分析-学习的循环次数 |
| `breadth` | `3` | 并行优化方案数 |
| `samples_per_plan` | `2` | 每方案独立实现数 |
| `topk_candidates` | `3` | Beam 池大小（保留的 top-K 内核数） |
| `max_experience_in_prompt` | `8` | 采样到规划器上下文的最大模式数 |
| `max_threshold` | `1.05` | 正向学习对的加速比阈值 |
| `min_threshold` | `1.05` | 负向学习对的减速比阈值 |
| `topk_learn` | `5` | 每轮送给总结器的总对数 |
| `ncu_binary` | `'ncu'` | Nsight Compute CLI 路径 |

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         初始化阶段                                    │
│  读取内核 → 构建 harness → NCU --set full → 基线性能画像             │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────────┐
              │          迭代循环                   │
              │                                    │
              │  ┌──────────────────────────────┐  │
              │  │ 规划（×BREADTH 并行）         │  │
              │  │ NCU 数据 + beam + 经验        │  │
              │  └─────────────┬────────────────┘  │
              │                │                    │
              │  ┌─────────────▼────────────────┐  │
              │  │ 执行（×SAMPLES 并行）         │  │
              │  │ 完整 .cu 实现                  │  │
              │  └─────────────┬────────────────┘  │
              │                │                    │
              │  ┌─────────────▼────────────────┐  │
              │  │ 评估 + 去重 + Beam 更新       │  │
              │  │ NCU 分析 → 逐方案最优         │  │
              │  │ → 更新候选者 beam              │  │
              │  └─────────────┬────────────────┘  │
              │                │                    │
              │  ┌─────────────▼────────────────┐  │
              │  │ 学习（阈值过滤）              │  │
              │  │ 慢-快对 → 优化模式             │  │
              │  │ → 经验池增长                   │  │
              │  └─────────────┬────────────────┘  │
              │                │                    │
              │                ▼（下一迭代）        │
              └────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────────┐
              │          最终报告                   │
              │  优化历程 + 模式 + beam 状态        │
              └───────────────────────────────────┘
```

## 输出

工作流返回：

```javascript
{
  baseline_latency_ms: number,    // 基线延迟
  best_latency_ms: number,        // 最优延迟
  overall_speedup: number,        // 总体加速比
  iterations_completed: number,   // 完成的迭代次数
  candidate_beam: [{plan_title, latency_ms, speedup}],  // 最终候选者 beam
  experience_patterns_count: number,   // 累积的经验模式数
  experience_patterns: string[],       // 全部经验模式
  best_kernel_code: string,            // 最优内核代码
  ncu_baseline_profile: string,        // NCU 基线画像
  report: string,                      // 技术优化报告
}
```

## 参考文献

- [AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization](https://arxiv.org/abs/2511.15915) (MLSys 2026)
- [Adaptive Self-improvement LLM Agentic System for ML Library Development](https://proceedings.mlr.press/v267/zhang25at.html) (ICML 2025)
