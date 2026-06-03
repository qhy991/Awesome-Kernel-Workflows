# Xe-Forge: Multi-Stage CoVeR Optimization for Intel XPU

**Paper**: [Xe-Forge: LLM-driven Triton kernel optimization for Intel XPUs](https://github.com/intel/Xe-Forge)  
**Venue**: GitHub Project (2024)

## Overview

Xe-Forge implements a **multi-stage CoVeR (Code-Verify-Refine) optimization pipeline** for Triton kernels targeting Intel XPUs (Arc Pro, Ponte Vecchio). The workflow progresses through 11 hard-ordered optimization stages, each running a nested CoVeR loop with best-of tracking and a ≥2% improvement gate.

## Core Insight

**Multi-stage progression with hard dependencies**: Early stages (ALGORITHMIC, DTYPE_FIX) must complete before later stages (DEVICE_SPECIFIC, AUTOTUNING). Each stage runs a CoVeR loop: generate candidates → verify correctness & performance → track best-in-stage. Only improvements ≥2% advance. This structure prevents premature optimization and ensures a solid foundation before device-specific tuning.

## Loop Topology

```
11-stage sequential pipeline with nested CoVeR loops:

STAGE 1 (ALGORITHMIC):
  CoVeR loop (max 10 iterations):
    Generate → Verify → Refine
    Track best (≥2% gate)
  → Promote best to Stage 2 baseline

STAGE 2 (DISCOVERY):
  CoVeR loop...
  → Promote best to Stage 3 baseline

...

STAGE 11 (VECTORIZATION):
  CoVeR loop...
  → Final best kernel
```

**Stage ordering** (hard dependencies):
1. ALGORITHMIC
2. DISCOVERY
3. DTYPE_FIX
4. FUSION
5. MEMORY_ACCESS
6. BLOCK_POINTERS
7. PERSISTENT_KERNEL
8. DEVICE_SPECIFIC
9. AUTOTUNING
10. POLISHING
11. VECTORIZATION

## Components

### Optimization Stages

1. **ALGORITHMIC**: High-level algorithm selection, data flow optimization
2. **DISCOVERY**: Broad exploration of optimization space
3. **DTYPE_FIX**: Data type selection (fp32/fp16/bf16), mixed precision
4. **FUSION**: Operation fusion, epilogue fusion
5. **MEMORY_ACCESS**: Memory coalescing, bank conflict reduction
6. **BLOCK_POINTERS**: Triton block pointer optimization
7. **PERSISTENT_KERNEL**: Persistent kernel patterns for streaming
8. **DEVICE_SPECIFIC**: Intel XPU-specific (XMX, EU optimization)
9. **AUTOTUNING**: Tile sizes, block dimensions, launch parameters
10. **POLISHING**: Final code cleanup, minor tweaks
11. **VECTORIZATION**: SIMD optimization, vector instruction usage

### Verification

- **Compilation**: Intel Triton compiler for XPU
- **Correctness**: `torch.allclose` numerical comparison
- **Performance**: Wall-clock execution time
- **Profiling**: Intel VTune GPU profiling
- **Improvement gate**: ≥2% speedup required to accept candidate

### Feedback Signals

- **speedup** (higher better): Wall-clock speedup vs baseline
- **correctness** (gate): Numerical error threshold
- **improvement_gate** (≥2%): Minimum improvement to accept
- **eu_occupancy**: Execution Unit utilization
- **xmx_utilization**: Xe Matrix Extensions usage

## Hardware Target

- **Intel Arc Pro**: A40, A50, A60
- **Intel Data Center GPU Max**: Ponte Vecchio
- **Features**: XMX (Xe Matrix Extensions), EU (Execution Units), HBM

## Typical Results

- **Speedup**: 1.5x - 5x vs naive Triton baseline
- **Iterations per stage**: 5-10
- **Total runtime**: 30-90 minutes (depends on kernel complexity)

## Example Usage

```javascript
// In Claude Code:
/workflow xe-forge-kernel-optimization

// The workflow will:
// 1. Initialize Intel XPU + Triton environment
// 2. Load baseline Triton kernel
// 3. For each stage (ALGORITHMIC → VECTORIZATION):
//    a. Run CoVeR loop: generate → verify → refine
//    b. Track best-in-stage (≥2% improvement gate)
//    c. Promote best to next stage baseline
// 4. Return optimized kernel with stage-by-stage breakdown
```

## Key Parameters

- **max_iterations_per_stage**: Maximum CoVeR loop iterations per stage (default: 10)
- **improvement_threshold**: Minimum improvement to accept candidate (default: 2.0%)
- **correctness_tolerance**: Numerical error threshold (default: 1e-5 absolute, 1e-3 relative)
- **enable_vtune**: Enable VTune profiling (default: false, expensive)

## Notes

- **Hard stage ordering**: Early stages set the foundation for later optimizations
- **CoVeR loop**: Generate → Verify → Refine (not just generate → accept)
- **Best-of tracking**: Keep best across all iterations, not just last
- **Intel XPU specific**: Leverages XMX, EU architecture, HBM bandwidth
- **Triton as IR**: Higher-level than CUDA, compiles to SPIR-V for Intel GPUs
- **VTune integration**: Optional profiling for detailed bottleneck analysis

## Related Workflows

- **KernelBlaster**: MAIC-RL approach with knowledge base
- **KernelFoundryDx**: Diagnostic-driven multi-island evolution for Triton
- **GPU Forecasters**: PUCT search with learned speedup forecasting
