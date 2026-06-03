# FACT: Compositional Kernel Synthesis Framework

**Paper**: [FACT: Compositional kernel synthesis framework](https://github.com/Project-FACT/FACT)  
**Venue**: GitHub Project (2024)

## Overview

FACT implements a **three-stage compositional synthesis pipeline** for CUTLASS kernels: Pattern Discovery → Pattern Realization → Pattern Composition. The workflow learns optimization patterns from exemplar kernels, realizes them as CUTLASS code transformations indexed in a pattern registry, and composes them into optimized kernels with ablation validation.

## Core Insight

**Pattern registry with compositional synthesis**: Instead of monolithic optimization, FACT decomposes the process into discovering reusable patterns (tiling, memory, compute, fusion, layout), indexing them as `T(rule_type, dtype, architecture)`, and composing them systematically. Ablation studies validate each pattern's contribution. This enables **transfer learning** (patterns discovered on one kernel apply to others) and **interpretability** (understand which patterns matter).

## Loop Topology

```
Three-stage sequential pipeline:

DISCOVERY:
  Analyze exemplar kernels
  Extract optimization patterns
  → Pattern candidates (REUSE, ADAPT, NEW)

REALIZATION:
  For each pattern:
    Generate CUTLASS code template
    Define transformation parameters
    Specify applicability constraints
    Create dependency graph
  → Realized pattern registry

COMPOSITION:
  Select composition strategy (greedy, dependency-aware, search-based)
  Generate pattern combinations
  Filter by constraints
  Tune parameters
  Compile and evaluate
  → Composed kernel candidates

ABLATION:
  Select top-K kernels
  Create ablation variants (leave-one-out, leave-group-out)
  Measure performance impact
  → Pattern contribution analysis

REPORT:
  Best kernel + pattern breakdown
```

## Components

### Stage 1: Pattern Discovery

**Goal**: Extract optimization patterns from exemplar kernels

**Strategies**:
- **REUSE**: Existing pattern applies directly
- **ADAPT**: Existing pattern needs modification
- **NEW**: Create new pattern from scratch

**Pattern types**:
- **Tiling**: Block/thread tiling strategies
- **Memory**: Memory hierarchy optimization (shared memory, registers)
- **Compute**: Computation patterns (Tensor Core usage, instruction scheduling)
- **Fusion**: Operation fusion (epilogue, multi-stage)
- **Layout**: Data layout transformations

### Stage 2: Pattern Realization

**Goal**: Realize patterns as concrete CUTLASS code transformations

**Output**: Pattern registry `T(rule_type, dtype, architecture)`

**CUTLASS layers**:
- **Tile**: Block-level tiling strategy
- **Kernel**: Warp-level computation pattern
- **Grid**: Thread block configuration

**Validation**:
- Syntactic correctness
- Type safety
- Resource constraints (shared memory, registers)

### Stage 3: Pattern Composition

**Goal**: Compose patterns into optimized kernels

**Composition strategies**:
1. **Greedy**: Iteratively add high-impact patterns
2. **Dependency-aware**: Respect pattern dependencies
3. **Search-based**: Explore pattern space systematically

**Constraints**:
- Resource limits (shared memory, registers)
- Pattern compatibility (dependencies, conflicts)
- Architecture-specific constraints

### Stage 4: Ablation

**Goal**: Validate pattern contributions

**Ablation configurations**: 2^N + 1 (N patterns + baseline)

**Methodology**:
- **Leave-one-out**: Remove each pattern individually
- **Leave-group-out**: Remove groups of related patterns
- **Performance impact**: Measure speedup loss

## Hardware Target

- **NVIDIA A100, H100**
- **Architecture**: sm_80, sm_89, sm_90
- **Features**: Tensor Cores, HBM2e/HBM3, CUDA Compute Capability 8.0+

## Feedback Signals

- **speedup** (higher better): Speedup vs PyTorch baseline
- **gflops** (higher better): Sustained throughput
- **correctness** (gate): Numerical correctness
- **tensor_core_utilization** (higher better): Tensor Core usage %
- **memory_bandwidth_utilization** (higher better): HBM bandwidth usage %

## Typical Results

- **Speedup**: 2x - 10x vs PyTorch baseline
- **Patterns discovered**: 10-20
- **Kernels composed**: 20-50
- **Runtime**: 60-120 minutes (depends on pattern count)

## Example Usage

```javascript
// In Claude Code:
/workflow fact-kernel-optimization

// The workflow will:
// 1. Analyze exemplar kernels (e.g., CUTLASS examples, PyTorch implementations)
// 2. Discover optimization patterns (tiling, memory, compute, fusion, layout)
// 3. Realize patterns as CUTLASS code transformations
// 4. Compose patterns into kernel candidates
// 5. Run ablation studies to validate contributions
// 6. Return best kernel with pattern breakdown
```

## Key Parameters

- **exemplar_kernels**: Path to exemplar implementations (CUTLASS examples, PyTorch)
- **composition_strategy**: "greedy" | "dependency-aware" | "search-based"
- **max_patterns**: Maximum number of patterns to compose (default: 10)
- **ablation_top_k**: Number of top kernels to ablate (default: 5)
- **cutlass_version**: CUTLASS version (default: "3.x")

## Notes

- **Pattern registry is key**: `T(rule_type, dtype, architecture)` indexes transformations
- **Exemplar-driven**: Learns from high-performance references
- **Compositional**: Builds complex optimizations from simpler patterns
- **Ablation validation**: Ensures patterns contribute meaningfully
- **CUTLASS-specific**: Leverages CUTLASS 3.x API (Tile/Kernel/Grid)
- **Supervisor checkpoints**: Optional human-in-the-loop approval gates
- **Transfer learning**: Patterns discovered on one kernel apply to others

## Related Workflows

- **StitchCUDA**: Three-agent orchestration for CUDA kernel synthesis
- **KernelFoundryDx**: Diagnostic-driven multi-island evolution for Triton
- **Xe-Forge**: Multi-stage CoVeR optimization for Intel XPU
