# ArchAgent: Evolutionary Cache Replacement Policy Optimization

**Paper**: [ArchAgent: Evolutionary cache replacement policy optimization](https://arxiv.org/abs/2602.22425)  
**Venue**: arXiv preprint (2024)

## Overview

ArchAgent implements **AlphaEvolve: evolutionary search with short-to-long evaluation cascade** for LLC (Last-Level Cache) replacement policy optimization. The workflow uses ChampSim cycle-accurate simulation to evolve C++ cache replacement policies through population-based search with fitness-based selection.

## Core Insight

**Short → Long evaluation cascade**: Evaluating policies on long traces (100M instructions) is expensive. ArchAgent filters the population with fast short evaluations (10M instructions) first, then runs thorough long evaluations only on survivors. This saves 10x compute while maintaining search quality. Evolutionary search explores the policy space systematically via mutation (algorithmic changes, parameter tuning) and crossover (combine elements from parents).

## Loop Topology

```
Population-based evolutionary search:

SEED POPULATION:
  Sample from known policies (LRU, RRIP, SHiP, Hawkeye)
  LLM-generated novel policies
  Hybrid: combine elements from multiple policies
  → Initial population (N=20-50)

GENERATION LOOP (10-30 generations):

  SHORT EVALUATION:
    Evaluate all candidates on 10M instruction traces
    → Quick fitness screening

  SELECTION:
    Select top-K candidates (tournament or top-K)
    → Survivors for long evaluation

  LONG EVALUATION:
    Evaluate survivors on 100M instruction traces
    → Accurate fitness

  EVOLUTION:
    Select parents (fitness-proportional or tournament)
    Mutate: algorithmic changes, parameter tuning, pattern injection
    Crossover: combine elements from two parents
    Diversity check: reject duplicates
    → Next generation

  CONVERGENCE CHECK:
    If no IPC improvement for N generations → STOP
    If max generations reached → STOP
    If population diversity < threshold → STOP

REPORT:
  Best policy + evolution trajectory
```

## Components

### Seed Population Strategies

1. **Sample from known policies**: LRU, RRIP, SHiP, Hawkeye
2. **LLM-generated novel policies**: Creative policy designs
3. **Hybrid**: Combine prediction logic from one policy with update logic from another

### Short Evaluation (10M instructions)

- **Purpose**: Quick fitness screening to filter population
- **Speedup**: ~10x faster than long evaluation
- **Metrics**: IPC (Instructions Per Cycle), MPKI (Misses Per Kilo Instructions)

### Long Evaluation (100M instructions)

- **Purpose**: Accurate fitness for final ranking
- **Metrics**: IPC, MPKI, speedup vs baseline (LRU or RRIP)

### Evolutionary Operators

1. **Mutation**:
   - **Algorithmic**: Change prediction logic, update mechanism
   - **Parameter**: Adjust thresholds, counters, table sizes
   - **Pattern**: Inject known good patterns (e.g., bypass heuristics, reuse distance estimation)

2. **Crossover**:
   - Combine prediction logic from parent A with update logic from parent B
   - Merge parameter sets from two parents

3. **Diversity Maintenance**:
   - Reject duplicates or near-duplicates (code similarity > threshold)
   - Maintain population diversity via distance metrics (edit distance, AST similarity)

### Selection Strategies

- **Top-K**: Select K best candidates
- **Tournament**: Random pairs, winner advances
- **Fitness-proportional**: Probability proportional to fitness

## Hardware Target

- **CPU cache hierarchy (LLC focus)**
- **Architectures**: x86-64 (Intel, AMD), ARM (optional)
- **Cache levels**: LLC (Last-Level Cache) - primary target; L1/L2 - future extensions

## Feedback Signals

- **ipc** (higher better): Instructions Per Cycle (primary fitness)
- **mpki** (lower better): Misses Per Kilo Instructions (secondary)
- **speedup** (higher better): Speedup vs baseline (LRU or RRIP)

## Typical Results

- **IPC improvement**: 1.05x - 1.20x vs LRU baseline
- **Generations**: 10-30
- **Population size**: 20-50
- **Runtime**: 2-8 hours (depends on trace length, population size)

## Example Usage

```javascript
// In Claude Code:
/workflow archagent-cache-policy-optimization

// The workflow will:
// 1. Initialize ChampSim environment
// 2. Create seed population (LLM + known policies)
// 3. For each generation:
//    a. Short evaluation (10M instructions) → filter population
//    b. Long evaluation (100M instructions) → accurate fitness
//    c. Selection → top performers
//    d. Evolution → mutation + crossover → next generation
// 4. Return best policy with analysis
```

## Key Parameters

- **population_size**: Number of candidates per generation (default: 30)
- **max_generations**: Maximum number of generations (default: 20)
- **short_trace_length**: Short evaluation trace length (default: 10M instructions)
- **long_trace_length**: Long evaluation trace length (default: 100M instructions)
- **selection_top_k**: Number of survivors for long evaluation (default: 10)
- **mutation_rate**: Probability of mutation (default: 0.3)
- **crossover_rate**: Probability of crossover (default: 0.5)
- **diversity_threshold**: Minimum code similarity for duplicate rejection (default: 0.9)

## Notes

- **Short → Long cascade**: Saves compute by filtering bad candidates early
- **AlphaEvolve**: Population-based, not single-trajectory
- **ChampSim integration**: Cycle-accurate simulation for cache behavior
- **Diversity maintenance**: Prevents premature convergence to local optima
- **C++ code generation**: Policies are executable ChampSim modules
- **Evaluation cost**: Long traces can take minutes per policy
- **Traces**: SPEC CPU 2017, memory-intensive workloads

## Related Workflows

- **GPU Forecasters**: PUCT search with learned speedup forecasting
- **KernelFoundryDx**: Diagnostic-driven multi-island evolution for GPU kernels
- **Xe-Forge**: Multi-stage CoVeR optimization for Intel XPU
