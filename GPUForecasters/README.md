# GPU Forecasters: Selective Surrogate Optimization with Learned Speedup Forecasting

**Paper**: [GPU Forecasters: Selective surrogate optimization with learned speedup forecasting](https://arxiv.org/abs/2605.31464)  
**Venue**: arXiv preprint (2024)

## Overview

GPU Forecasters implements **PUCT search with learned speedup forecaster and native abstain mechanism**. The workflow trains a language model to predict kernel speedup in 8 ordinal bins, then uses PUCT tree search to explore the optimization space. The forecaster can abstain when uncertain, deferring to actual GPU execution. This balances exploration (high uncertainty) and exploitation (high predicted speedup) under a fixed GPU iterations.

## Core Insight

**Selective surrogate with abstain mechanism**: Training a surrogate model to predict speedup saves GPU evaluations, but surrogate errors can mislead search. GPU Forecasters' key innovation is the **native abstain mechanism**: the forecaster defers to GPU execution when uncertain. PUCT search leverages this: explore high-uncertainty regions (abstain → GPU eval), exploit high-confidence high-speedup regions (predict → no GPU cost). This achieves better GPU iterations efficiency than either pure surrogate or pure search.

## Loop Topology

```
Two-phase workflow: Train Forecaster → PUCT Search

PHASE 1: TRAIN FORECASTER (Curriculum Phase)
  Generate initial kernel candidates (N=50-100)
  Execute on GPU (ground truth speedup)
  Train forecaster: kernel code + context → 8-bin ordinal classification
  Validate forecaster: accuracy, abstain rate
  → Trained forecaster model

PHASE 2: PUCT SEARCH (Fixed GPU Budget)
  Initialize search tree (root = baseline kernel)
  
  For each search step (until GPU iterations M exhausted):
    1. SELECT: Choose node with highest PUCT score
       PUCT(node) = Q(node) + C * sqrt(log(N_parent) / N_node)
    
    2. GENERATE: Create child candidates (mutations, variations)
    
    3. FORECAST: LM predicts speedup bin or abstains
       - If ABSTAIN or high uncertainty: execute on GPU (use iterations)
       - If PREDICT: use forecasted speedup (no GPU cost)
    
    4. UPDATE: Update tree statistics (Q values, visit counts)
  
  → Search tree with best kernel path

EVALUATE:
  Evaluate top-K candidates from search tree on GPU
  → Best kernel + search trajectory

REPORT:
  Best kernel + forecaster analysis + GPU iterations usage
```

## Components

### Speedup Forecaster

**Architecture**: Transformer LM (e.g., CodeLlama, GPT-based)

**Input**:
- Kernel code (CUDA source)
- Context (operation name, tensor shapes, dtypes)

**Output**:
- 8-bin ordinal classification: [0.5x, 0.7x, 0.9x, 1.0x, 1.2x, 1.5x, 2.0x, 3.0x+]
- Abstain option: defer to GPU when uncertain

**Training**:
- Supervised learning on (kernel, actual_speedup) pairs
- Ordinal regression loss (respects bin ordering)
- Calibration: GRPO (Group Relative Policy Optimization) for abstain threshold

**Abstain mechanism**:
- Uncertainty threshold: if prediction confidence < threshold → abstain
- Native abstain: built into model output (not post-hoc filter)

### PUCT Search

**Formula**: `PUCT(node) = Q(node) + C * sqrt(log(N_parent) / N_node)`

**Components**:
- **Q(node)**: Average speedup of subtree (exploitation)
- **C * sqrt(...)**: Exploration bonus (higher for less-visited nodes)
- **C**: Exploration constant (tunable, default: 2.0)

**Budget management**:
- Fixed GPU iterations: M evaluations total (e.g., M=100-500)
- Abstain → consumes 1 GPU eval
- Predict → no GPU cost
- Search terminates when iterations exhausted

**Selection strategy**:
- Greedy: always select highest PUCT score
- Balances exploitation (high Q) and exploration (low visit count)

### Candidate Generation

**Mutation strategies**:
- Algorithmic: change tiling, fusion, memory access patterns
- Parameter: adjust block sizes, thread counts, unroll factors
- Pattern injection: apply known optimization patterns

**Diversity**:
- Reject duplicates (code similarity > threshold)
- Encourage exploration of diverse optimization strategies

## Hardware Target

- **NVIDIA GPUs**: CUDA-capable (sm_80, sm_89, sm_90)
- **Architectures**: NVIDIA A100, L40S, H100
- **Features**: CUDA Compute Capability 8.0+, Tensor Cores (optional)

## Feedback Signals

- **forecasted_speedup_bin** (ordinal): LM forecasts speedup in 8 bins
- **actual_speedup** (higher better): Ground truth speedup from GPU execution
- **abstain_decision**: Forecaster abstains when uncertain
- **puct_score** (higher better): PUCT = exploitation + exploration

## Typical Results

- **Speedup**: 1.5x - 4x vs baseline
- **GPU iterations**: 100-500 evaluations
- **Forecaster accuracy**: 60-80% (ordinal bins)
- **Abstain rate**: 20-40% (depends on calibration)
- **GPU iterations efficiency**: best_speedup / num_gpu_evals (higher is better)
- **Runtime**: 30-120 minutes (depends on GPU iterations)

## Example Usage

```javascript
Workflow({
  name: 'gpuforecasters-kernel-optimization',
  args: {
    kernel_path: '/path/to/baseline.cu',
    problem_definition: 'Optimize a CUDA row-wise reduction kernel',
    language: 'cuda',
    problem_type: 'cuda-kernel-optimization',
    target_gpu: 'H100',
    note: 'Use the provided baseline and validation tolerance exactly.',
    test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
    benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
    baseline_latency_ms: 0.42,
    curriculum_size: 100,
    gpu_budget: 200,
    exp_dir: '/tmp/gpuforecasters_exp',
  },
})
```

## Key Parameters

- **kernel_path**: Existing baseline kernel to optimize. If omitted, provide `problem_definition`, `problem_path`, or `note` so the workflow can generate initial candidates.
- **problem_definition** / **problem_path**: Task specification for generation or optimization.
- **note**: User-supplied authoritative context. Validation commands, baseline details, tolerances, and constraints written here are threaded into every workflow phase.
- **test_command**: Correctness command. It should support `{kernel_path}` and `{result_path}` substitutions.
- **benchmark_command**: Performance command. It should support `{kernel_path}` and `{result_path}` substitutions and write measured latency/speedup JSON.
- **baseline_latency_ms**: Optional measured baseline latency used as the speedup denominator.
- **exp_dir**: Directory for generated candidates and evaluator artifacts (default: `/tmp/gpuforecasters_exp`).
- **curriculum_size**: Number of initial kernels for forecaster training (default: 100)
- **gpu_budget**: Fixed GPU evaluation iterations (default: 200)
- **puct_c**: Exploration constant (default: 2.0)
- **abstain_threshold**: Uncertainty threshold for abstain (default: 0.3)
- **speedup_bins**: Ordinal bins for speedup classification (default: [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0])
- **tree_depth**: Maximum search tree depth (default: 10)

## Notes

- **Selective surrogate**: Forecaster defers to GPU when uncertain (abstain mechanism)
- **PUCT search**: Balances exploration (uncertainty) and exploitation (predicted speedup)
- **Ordinal bins**: Speedup is discretized into 8 categories for classification
- **Fixed GPU iterations**: M evaluations across entire search (not per-generation)
- **Evidence contract**: Provided `note`, `test_command`, `benchmark_command`, and baseline fields are injected into setup, training, PUCT search, refinement, validation, and reporting prompts.
- **Curriculum training**: Train forecaster on initial data, apply in later search
- **Abstain rate**: Key metric for forecaster quality
  - Too high → wastes iterations (frequent GPU evals)
  - Too low → poor guidance (low-confidence predictions mislead search)
- **GPU iterations efficiency**: best_speedup / num_gpu_evals (higher is better)
- **GRPO calibration**: Tunes abstain threshold to optimize iterations efficiency

## Related Workflows

- **StitchCUDA**: Three-agent orchestration with adaptive replanning
- **KernelBlaster**: MAIC-RL approach with knowledge base
