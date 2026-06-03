# StitchCUDA: Three-Agent Orchestration for CUDA Kernel Synthesis

**Paper**: [StitchCUDA: Three-agent orchestration for CUDA kernel synthesis](https://arxiv.org/abs/2603.02637)  
**Venue**: arXiv preprint (2024)

## Overview

StitchCUDA implements a **three-agent orchestration framework** for CUDA kernel synthesis: **Planner → Coder → Verifier** with adaptive replanning. The Planner generates strategic optimization plans, the Coder implements them as CUDA kernels, and the Verifier checks correctness and performance. Failure patterns trigger replanning with alternative strategies.

## Core Insight

**Separation of concerns with adaptive replanning**: By splitting optimization into strategic planning (Planner), implementation (Coder), and verification (Verifier), each agent specializes in its domain. Adaptive replanning learns from failure patterns: consecutive compile failures → replan for compilation issues; consecutive correctness failures → replan for algorithmic correctness; performance stagnation → replan with alternative optimization approach. This structure avoids getting stuck in local optima.

## Loop Topology

```
Sequential three-agent pipeline with adaptive replanning:

PLANNER:
  Generate optimization plan
  (strategy, steps, threading config)
  → Plan (JSON)

CODER:
  Implement plan as CUDA kernel
  → CUDA kernel code

VERIFIER:
  Compile check (nvcc)
  Correctness check (numerical)
  Performance check (benchmark)
  → Verification result

ADAPTIVE REPLANNING (conditional):
  If 2 consecutive compile failures
    → Replan with focus on compilation
  If 2 consecutive correctness failures
    → Replan with focus on algorithmic correctness
  If 3 iterations of stagnation (CV < 5%)
    → Replan with alternative optimization approach

Repeat for max_attempts (default: 20)
Track best kernel across all attempts
```

## Components

### Agent 1: Planner

**Role**: Strategic optimization planning

**Responsibilities**:
- High-level optimization approach (memory-bound, compute-bound, balanced)
- Decompose into implementation steps
- Specify threading/block configuration
- Identify key optimizations

**Output**: JSON plan with:
- `strategy`: "memory-bound" | "compute-bound" | "balanced"
- `steps`: Array of implementation steps
- `threading_config`: `{block_size, grid_size, threads_per_block}`
- `key_optimizations`: Array of optimization names (e.g., "coalescing", "shared_memory", "register_blocking")

### Agent 2: Coder

**Role**: CUDA kernel code generation

**Responsibilities**:
- Generate complete CUDA kernel following plan
- Implement all steps from plan
- Apply key optimizations (coalescing, shared memory, register blocking, loop unrolling, etc.)
- PyTorch `load_inline` compatible format

**Output**: Complete CUDA kernel code + host launch code

### Agent 3: Verifier

**Role**: Correctness and performance verification

**Responsibilities**:
- **Compile check**: nvcc compilation, syntax validation, resource usage
- **Correctness check**: Numerical accuracy vs reference implementation
- **Performance check**: Benchmark execution time, profile with nsys/ncu (optional)
- **KernelBench evaluation**: Standardized evaluation framework

**Output**: Verification result:
- `compilation_success`: boolean
- `correctness_passed`: boolean
- `speedup`: float (vs baseline)
- `performance_gflops`: float
- `errors`: Array of error messages

### Adaptive Replanning

**Triggers**:

1. **N consecutive compile failures** (N=2)
   - **Action**: Replan with focus on compilation issues
   - **Strategy**: Simplify code, avoid complex templates, check resource limits

2. **N consecutive correctness failures** (N=2)
   - **Action**: Replan with focus on algorithmic correctness
   - **Strategy**: Review algorithm, add validation checks, simplify logic

3. **Performance stagnation** (M=3 iterations, CV < 5%)
   - **Action**: Replan with alternative optimization approach
   - **Strategy**: Switch from memory-bound to compute-bound (or vice versa), explore different threading configs

**Replanning process**:
1. Diagnose root cause of failures
2. Generate alternative approach
3. Avoid previous failure modes
4. Reset failure counters after replan

## Hardware Target

- **NVIDIA GPUs**: CUDA-capable (sm_80, sm_89, sm_90)
- **Architectures**: NVIDIA A100, L40S, H100
- **Features**: CUDA Compute Capability 8.0+, Shared memory, Registers, Tensor Cores (optional)

## Feedback Signals

- **compilation_success** (gate): Kernel must compile without errors
- **correctness_passed** (gate): Numerical correctness vs reference
- **speedup** (higher better): Speedup vs baseline (threshold: 1.0x)
- **performance_gflops** (higher better): Sustained throughput

## Typical Results

- **Speedup**: 1.5x - 5x vs baseline
- **Attempts**: 10-20 (depends on max_attempts)
- **Runtime**: 20-60 minutes (depends on max_attempts)

## Example Usage

```javascript
// In Claude Code:
/workflow stitchcuda-kernel-optimization

// The workflow will:
// 1. Initialize CUDA + KernelBench environment
// 2. For each attempt (up to max_attempts):
//    a. Planner generates optimization plan
//    b. Coder generates CUDA kernel
//    c. Verifier checks compilation, correctness, performance
//    d. If failures trigger replan: adapt strategy
// 3. Track best kernel across all attempts
// 4. Return best kernel with report
```

## Key Parameters

- **max_attempts**: Maximum synthesis attempts (default: 20)
- **compile_failure_threshold**: Consecutive compile failures to trigger replan (default: 2)
- **correctness_failure_threshold**: Consecutive correctness failures to trigger replan (default: 2)
- **stagnation_threshold**: Iterations for performance stagnation (default: 3)
- **stagnation_cv_threshold**: Coefficient of variation for stagnation detection (default: 5%)
- **early_termination_speedup**: Speedup to trigger early termination (default: 2.0x)

## Notes

- **Three-agent specialization**: Strategic planning, code generation, verification
- **Adaptive replanning**: Learns from failure patterns
- **Fast iteration**: PyTorch `load_inline` for rapid compile-test cycles
- **KernelBench integration**: Standardized evaluation framework
- **Failure counters**: Track consecutive compile/correctness failures
- **Stagnation detection**: Coefficient of variation < 5%
- **Early termination**: Stop when excellent performance achieved (≥2x speedup)

## Related Workflows

- **FACT**: Compositional kernel synthesis with pattern discovery
- **KernelBlaster**: MAIC-RL approach with knowledge base
- **GPU Forecasters**: PUCT search with learned speedup forecasting
