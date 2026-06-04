# CutlassGEMM: CUTLASS-Based GEMM Optimization with NCU Profiling & Multi-Config Dispatch Tuning

**English** · [简体中文](README.zh-CN.md)

An iterative GEMM kernel optimization workflow using **NVIDIA CUTLASS** device-level APIs with **Nsight Compute (NCU) profiling** for hardware-level bottleneck analysis, **multi-configuration runtime dispatch**, and **StreamK load balancing**. Targets SOL-ExecBench problems with variable batch dimension (M).

## Overview

CutlassGEMM bridges the gap between static CUTLASS template instantiations and cuBLAS's per-problem autotuning. While cuBLAS selects tile configurations dynamically at runtime for each (M, N, K) combination, a single CUTLASS configuration cannot be optimal across all M values. This workflow solves that by:

1. **Analyzing** the problem's M-range distribution to identify distinct performance regimes
2. **Generating** multiple CUTLASS `GemmUniversal` instantiations with different tile configs
3. **Profiling** per-workload performance via SOL-ExecBench
4. **Iteratively tuning** dispatch thresholds and tile parameters based on speedup feedback

### Key Innovation: M-Adaptive Multi-Config Dispatch

```
M >= 512:  Data-parallel GemmUniversal, large tiles (256x128x32), swizzle=8
           → Full SM occupancy, minimal scheduler overhead

M = 128-511: StreamK GemmUniversal, same large tiles
           → Balances load when ceil(M/tile_M) × ceil(N/tile_N) ≠ k×num_SMs

M < 128:  StreamK GemmUniversal, smaller tiles (128x128x32), more stages
           → Reduces wasted edge compute, deeper pipeline hides latency
```

### Results (A800, problem 011_gemm_n28672_k4096, Llama 3.1 8B MLP)

| M range | Speedup vs cuBLAS | Strategy |
|---------|-------------------|----------|
| M=128-256 | **1.05-1.09x** | StreamK + large tiles |
| M=512-2053 | 0.95-1.06x | Data-parallel |
| M=8192 | 0.97x | Data-parallel |
| M<128 | 0.83-1.00x | StreamK + small tiles |
| **All 43 workloads** | **PASS correctness** | |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Analyze                                   │
│  Read definition.json → Parse shapes/dtypes → Layout mapping │
│  Determine: RowMajor/ColumnMajor, alignment, instruction ISA │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                     Baseline                                  │
│  Generate 3-config solution.json → Compile → Benchmark       │
│  Establish per-workload speedup baseline                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                 Iterative Tuning Loop                         │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ NCU Profile (NEW)                                     │   │
│  │ User-provided profile cmd over representative M values │   │
│  │ Collect: SM%, MemBW%, Occupancy%, TensorCore%, L2Hit% │   │
│  │ Diagnose: memory-bound? occupancy-limited? underoccup?│   │
│  └────────────────────────┬─────────────────────────────┘   │
│                            │                                  │
│  ┌────────────────────────▼─────────────────────────────┐   │
│  │ Tune (NCU-guided)                                     │   │
│  │ Apply fixes based on hardware evidence:               │   │
│  │ • Low occupancy → reduce tile/stages                  │   │
│  │ • Memory bound → increase tile for reuse              │   │
│  │ • Grid underoccupied → StreamK / split-K              │   │
│  │ • Low L2 hit → increase swizzle factor                │   │
│  └────────────────────────┬─────────────────────────────┘   │
│                            │                                  │
│  ┌────────────────────────▼─────────────────────────────┐   │
│  │ Validate                                              │   │
│  │ Run SOL-ExecBench → Check correctness → Compare perf  │   │
│  │ Accept if improved, revert otherwise                  │   │
│  └────────────────────────┬─────────────────────────────┘   │
│                            ▼ (next iteration)                 │
└───────────────────────────────────────────────────────────────┘
```

### NCU Metrics Collected (per representative M value)

| Metric | What it tells | Tuning action |
|--------|--------------|---------------|
| `sm__throughput` | Compute utilization | Low → memory bound, needs more reuse |
| `gpu__compute_memory_throughput` | Memory BW utilization | High → memory bound |
| `sm__warps_active` | Achieved occupancy | Low → too many regs/smem, reduce tile |
| `sm__pipe_tensor_cycles_active` | Tensor core usage | Low → alignment issue or stalled mainloop |
| `dram__bytes_read.pct_of_peak` | HBM read BW | Near peak → bandwidth ceiling |
| `lts__t_sector_hit_rate` | L2 cache hit rate | Low → swizzle not effective |
| `launch__waves_per_multiprocessor` | Grid waves | <1.5 → grid underoccupied, need StreamK |
| `launch__registers_per_thread` | Register pressure | >128 → occupancy limited |

### Optimization Levers

| Lever | Options | Impact |
|-------|---------|--------|
| ThreadblockShape | 256x128x32, 128x256x32, 128x128x32, 64x64x32 | Determines CTAs/SM, register pressure |
| WarpShape | 64x64x32, 32x64x32, 64x32x32, 32x32x32 | Warp-level parallelism |
| Pipeline Stages | 3-10 | Latency hiding vs shared memory usage |
| Swizzle | Identity(1,2,4,8), StreamK | L2 locality vs load balance |
| Split-K | 1, 2, 4, 8 | Parallelism for small M × large K |
| Accumulator | fp32 (required), fp16 (incorrect) | Correctness constraint |
| Alignment | 8 (fp16), 4 (fp32) | Vectorized memory access width |

## Prerequisites

### Required

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **Claude Code** | Workflow runtime | [claude.ai/claude-code](https://claude.ai/claude-code) |
| **NVIDIA GPU** | sm_80+ (A100/A800/H100) | Target execution |
| **CUTLASS 3.x/4.x** | Template library headers | `git clone https://github.com/NVIDIA/cutlass` |
| **SOL-ExecBench** | Evaluation framework | [SOL-ExecBench repo](https://github.com/NVIDIA/SOL-ExecBench) |
| **Nsight Compute (ncu)** | Hardware profiling | Ships with CUDA Toolkit 12.x |
| **PyTorch + CUDA** | Compilation via `cpp_extension` | `pip install torch` |
| **uv** | Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

> **Note on NCU permissions:** On many systems, NCU requires elevated privileges to access hardware counters. If `ncu` needs sudo, ensure passwordless sudo is configured: `echo "$USER ALL=(ALL) NOPASSWD: /usr/local/cuda-12.4/bin/ncu" | sudo tee /etc/sudoers.d/ncu`

### Claude Code Skills (Recommended)

| Skill | Purpose |
|-------|---------|
| `ncu-report-skill` | Deep NCU profiling for bottleneck identification |
| `KernelWiki` | Hopper/Blackwell-specific optimization knowledge |
| `kernel-auto-tester` | Standalone test generation |

## Usage

```javascript
Workflow({name: 'cutlass-gemm-optimization', args: {
  problem_dir: '/path/to/SOL-ExecBench/data/benchmark/FlashInfer-Bench/011_gemm_n28672_k4096',
  cutlass_dir: '/path/to/cutlass',
  sol_execbench_dir: '/path/to/SOL-ExecBench',
  output_dir: '/tmp/cutlass_gemm_opt',
  iterations: 3,
  target_gpu: 'sm_80',
}})
```

### Direct Evaluation (without workflow)

```bash
CUTLASS_DIR=/path/to/cutlass uv run sol-execbench \
  /path/to/problem_dir \
  --solution solution_example.json \
  --no-json -v
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `problem_dir` | (required) | Path to SOL-ExecBench problem directory |
| `cutlass_dir` | `/usr/local/cutlass` | Path to CUTLASS source tree (needs `include/`) |
| `sol_execbench_dir` | `/home/.../SOL-ExecBench` | SOL-ExecBench installation root |
| `output_dir` | `/tmp/cutlass_gemm_opt` | Where to write solution artifacts |
| `iterations` | `2` | Tuning loop iterations (with cost-benefit early stop) |
| `target_gpu` | `sm_80` | Target GPU compute capability |
| `peak_tflops` | `312` (sm_80) / `989` (sm_90) | Peak fp16 tensor core TFLOPS for MFU calculation |
| `ncu_binary` | `sudo /usr/local/cuda-12.4/bin/ncu` | NCU binary path (with sudo if needed) |
| `ncu_profile_m_values` | `[8, 64, 256, 2048]` | Representative M values to profile with NCU |
| `enable_hybrid_fallback` | `true` | Enable cuBLAS fallback for overhead-dominated tiny M |
| `cublas_fallback_threshold` | `32` | M below this uses cuBLAS instead of CUTLASS |

## Output

```javascript
{
  problem: string,
  operation: string,
  hardware: string,
  peak_tflops: number,            // Hardware peak (312 for A800, 989 for H100)
  fixed_N: number,
  fixed_K: number,
  variable_range: {min, max},
  iterations_completed: number,
  best_avg_speedup: number,       // Average speedup vs torch.matmul
  baseline_avg_speedup: number,
  improvement: number,
  ceiling_detected: boolean,      // True if overhead-dominated M range found
  ceiling_threshold: number,      // M below this is overhead-bound
  hybrid_enabled: boolean,        // cuBLAS fallback active?
  mfu: {
    peak_tflops: number,
    compute_bound_avg_mfu_pct: number,  // Avg MFU for compute-bound workloads
    compute_bound_max_mfu_pct: number,  // Best MFU achieved
    compute_bound_count: number,
    memory_bound_count: number,
    per_workload: [{
      m: number,
      mfu_pct: number,            // Our MFU
      ref_mfu_pct: number,        // cuBLAS MFU (reference)
      achieved_tflops: number,
      arithmetic_intensity: number,
      regime: 'compute' | 'memory',
    }],
  },
  tuning_history: [...],
  output_dir: string,
  solution_path: string,
}
```

### MFU Interpretation Guide

| MFU Range | Meaning | Action |
|-----------|---------|--------|
| > 70% | Near hardware peak | Kernel is well-optimized, diminishing returns |
| 50-70% | Good utilization | Check occupancy, pipeline stalls |
| 30-50% | Moderate | Likely memory-bound or occupancy-limited |
| < 30% | Low | Likely overhead-dominated or grid-starved |

**Note:** MFU is only meaningful for **compute-bound** workloads (arithmetic intensity > ridge point). For memory-bound workloads, HBM bandwidth utilization is the correct efficiency metric.

## CUTLASS Layout Mapping (Critical)

PyTorch stores tensors in **row-major** order. CUTLASS's `RowMajor` and `ColumnMajor` must be set based on how the operation maps:

| Operation | A shape | B shape | CUTLASS LayoutA | CUTLASS LayoutB |
|-----------|---------|---------|-----------------|-----------------|
| C = A @ B | [M, K] | [K, N] | RowMajor | RowMajor |
| C = A @ B.T | [M, K] | [N, K] | RowMajor | **ColumnMajor** |
| C = A.T @ B | [K, M] | [K, N] | ColumnMajor | RowMajor |

The leading dimension (stride) for row-major A `[M, K]` is K. For B `[N, K]` treated as ColumnMajor, the leading dimension is also K.

## Comparison with Other Approaches

| Aspect | CutlassGEMM | cuBLAS (torch.matmul) | Triton autotuning |
|--------|-------------|----------------------|-------------------|
| Tuning | Multi-config + dispatch thresholds | Per-call heuristic selection | JIT grid search |
| Overhead | Zero runtime overhead (static dispatch) | ~5us per cublasGemm call | First-call JIT compilation |
| Flexibility | Full control over tile/swizzle/pipeline | Black-box | Limited to Triton IR |
| Best for | Known workload distribution | General use | Quick prototyping |
| Peak perf | 0.95-1.09x cuBLAS | 1.0x (baseline) | 0.8-1.0x cuBLAS |

## References

- [CUTLASS: Fast Linear Algebra in CUDA C++](https://github.com/NVIDIA/cutlass)
- [StreamK: Stream-K Parallel Decomposition for GEMM](https://arxiv.org/abs/2301.03598) — Osama et al., 2023
- [SOL-ExecBench: GPU Kernel Evaluation Framework](https://github.com/NVIDIA/SOL-ExecBench)
- [Efficient GEMM in CUDA](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/) — NVIDIA Performance Guide
