# AscendC Kernel Optimization

**English** · [简体中文](README.zh-CN.md)

The canonical **Ascend / AscendC** kernel-optimization workflow for Ascend 910B
NPUs. Ascend-native (not a CUDA workflow with swapped labels): it targets AscendC,
profiles with **msprof** (CANN), and delegates compile + correctness + NPU-event
timing to the MultiKernelBench `ascendc_direct_launch` runner via the substrate
[ascend backend](../_substrate/backends/ascend/).

> Resolves [issue #16](https://github.com/qhy991/Awesome-Kernel-Workflows/issues/16):
> previously the catalog had no usable Ascend workflow, so every Ascend task
> **STALLed at workflow selection (round 1)** and had to evolve a session-local
> variant that did not carry across sessions. This is that proven variant,
> promoted upstream and generalized.

## Source / fidelity

Derived from the session-local variant `generalist-ascend-optimization`, evolved
and validated across 910b-exp sessions `20260622-160108` /
`20260623-{194914,190425,191018}` (see `AKW-Exp/910b-exp/KEY-ISSUES.md §B`). The
task-specific (batched matmul) hints were removed; the canonical
`ascendc_direct_launch` submission contract and the substrate ascend backend
driver were retained. Fidelity boundary: **substrate_reference** — the real
measurement signal is owned by `eval_single_runner.py`, not the prompt.

## How it works

```
Setup      → read reference op + ascend backend manifest; assert language=ascendc
Generate   → BREADTH AscendC candidates as ascendc_direct_launch JSON
             (kernel.cpp + pybind11.cpp + ModelNew.py), written ONE FILE AT A TIME
Evaluate   → substrate ascend run.sh: bisheng+cmake compile + correctness + NPU timing
Optimize   → for each round:
               msprof the current best → pick ONE AscendC improvement
               (Cube/MTE pipelining, FRACTAL_NZ, tiling to L1/L0, dispatch folding)
               implement → eval ONCE → RETURN (turn boundary)
Report     → best kernel + convergence status
```

**Loop control:** stop on `target_speedup` reached (`converged`), stagnation
(2 consecutive rounds < 2% improvement → `stalled`), or token budget exhausted
(`budget_exhausted`).

## Evidence contract

Compile + correctness + timing are coupled inside MultiKernelBench's
`eval_single_runner.py` (invoked by `_substrate/backends/ascend/run.sh`). The
result JSON carries `{compiled, correct, candidate_latency_ms, eager_latency_ms,
compile_latency_ms, claimed_speedup}`. `correct:false` floors the speedup at 1.0.
Metrics are **never fabricated** — a failed eval reports `compiled=false` /
`correct=false` with the real error.

## Robustness scaffolding (issue #17)

- **`agentRetry` + null-guards**: every `agent()` call is wrapped in a bounded
  (5×) retry and every dereferenced result is null-guarded, so a transient API 429
  or agent skip no longer crashes the run (the single highest-leverage fix from
  session `20260622-161357`).
- **Turn boundary**: each implement turn writes files → runs eval **once** → reads
  the result → **returns** (no 130-minute runaway turns).
- **Per-file Bash write**: AscendC sources are written one file at a time, not as
  one JSON blob (the blob path truncates / times out — KEY-ISSUES §3 row 9).
- **NO HARNESS MANIPULATION**: allocator / free-pool scrubbing to make the
  reference match is forbidden (KEY-ISSUES §3 row 3).

## Invocation

```javascript
Workflow({ name: 'ascendc-kernel-optimization', args: {
  kernel_path: '/path/to/reference_kernel.cpp',   // or use problem_definition for generation
  op_description: 'Flash attention on Ascend 910B2',
  substrate_dir: '/path/to/Awesome-Kernel-Workflows/_substrate',
  backend_dir:   '/path/to/Awesome-Kernel-Workflows/_substrate/backends/ascend',
  mkb_root:      process.env.MULTIKERNELBENCH_ROOT,
  op:            'flash_attention',
  exp_dir:       '/path/to/experiment/output',
  iterations: 3, breadth: 2, target_speedup: 1.5,
}})
```

Requires: bisheng (AscendC compiler), `torch_npu`, the MultiKernelBench eval
harness, and an Ascend NPU (msprof/CANN). On a host without an NPU, the substrate
ascend scripts degrade honestly (exit 4 / `ok:false`) rather than fabricate.

## What it is NOT

Not suitable for CUDA / Triton / ROCm / Metal targets — those have their own
profilers (ncu / rocprof / metal-capture) and compile chains. For those, pick the
matching catalog workflow.
