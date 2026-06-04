# Generalist Kernel Optimization

A best-of-breed **single solver** built on the [KerSor Solver SDK substrate](../_substrate/SOLVER-SDK.md).
It internalizes the universal best-of-breed components (the 7 composable axes from
[`SOLVER-AUDIT.md`](../_substrate/SOLVER-AUDIT.md)) and uses **one** search
topology — a beam controller. It is directly callable like any workflow, and is
also the strongest default member / hardest single-solver baseline for KerSor
(see [`ARCHITECTURE.md`](../_substrate/ARCHITECTURE.md)).

> Topologies are mutually exclusive — this solver does **not** stack MCTS /
> evolutionary / bandit. Cross-topology coverage is KerSor's job, not this file's.

## How it's best-of-breed

Every "smart" decision is delegated to a **deterministic substrate script** (run
by agent Bash iterations) — the scripts are the fidelity anchor, not the prompts:

| Phase | Substrate (Layer) | Borrowed from |
|---|---|---|
| Diagnose | `diagnose.py` (C) → `bottleneck_class` | AccelOpt / KernelBand / cuPilot |
| Retrieve | `memory_store.py` (D) → ranked techniques + dead-ends, **persistent across runs** | KernelBlaster + AKO4X |
| Gate | `method_gate.py` (E) → `allowed_methods` (LLM may only pick within) | KernelSkill |
| Plan | grounded anchors `<<<IMPROVE BEGINS/ENDS>>>` | STARK |
| Evaluate | `anti_cheat.py` (B) → robust reward (beat eager AND compile) + cheat block; `evidence_schema.py` (A) | CUDAAgent / CUDALLM / TritorX |
| Learn | `memory_store.py update` with measured outcome | KernelBlaster |
| Continue | beam top-K + ceiling/stagnation/early-term (inline) | AccelOpt / Xe-Forge / KSearch |

Conformance: **L3** (uses Layers A–F). Output is the Layer A evidence envelope, so
KerSor consumes it natively.

## Invocation

```javascript
Workflow({ name: 'generalist-kernel-optimization', args: {
  kernel_path: '/path/to/kernel.cu',
  op_description: 'Quantized GEMM Q4_0 weight x FP32 activation',
  benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
  ncu_command: '<user-provided profiler command with {kernel_path}/{result_path}>',           // optional
  substrate_dir: '/path/to/Awesome-Kernel-Workflows/_substrate',
  exp_dir: '/path/to/experiment/output',
  memory_db: '/path/to/experiment/memory.json', // persistent; defaults to exp_dir/memory.json
  iterations: 3, breadth: 3, topk: 3, target_speedup: 1.5,
  // P0 — optional cost/robustness controls:
  model_mechanical: 'haiku',   // script-running / parsing iterations
  model_profile: 'sonnet',     // profiling + metric normalization
  model_judgment: 'opus',      // planning / implementation / report
  est_tokens_per_candidate: 20000,  // iterations-scaling estimate
}})
```

**P0 applied** (see [`../_substrate/IMPROVEMENTS.md`](../_substrate/IMPROVEMENTS.md)):
token-iterations wiring (scales `breadth` to `iterations.remaining()`, hard-stops when a
round won't fit), **worktree isolation** on parallel candidate implementation
(no file-edit conflicts), and **model routing** (mechanical iterations run on Haiku/Sonnet,
only planning/codegen on Opus — most tokens are mechanical, so this is the cheapest
large saving).

`benchmark_command` must write JSON: `{compiled, correct, candidate_latency_ms,
eager_latency_ms, compile_latency_ms, speedup, metrics:{dram_pct, sm_pct,
occupancy, latency_ms}}`. See the file header for the full contract.

## Status

Reference implementation on the `dev/solver-substrate` branch. The deterministic
substrate scripts are unit-tested; the workflow orchestration is validated for
syntax. End-to-end runs require a real GPU + `benchmark_command`.
