# Atrex Kernel Optimization Adapter

**English** · [简体中文](README.zh-CN.md)

A strict adapter to [Atrex Kernel Agent](https://github.com/alibaba/atrex-kernel-agent), released with [Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent](https://arxiv.org/abs/2607.14541).

## Why a Strict Adapter

The current Atrex repository exposes exactly one supported optimization entry point: `orchestrator/optimize.py`. Its supervisor already owns the hard parts:

- isolated Git workspace per framework and target;
- correctness-passing V0 and optional framework-native V1;
- Long Horizon clean-session episodes over the complete workload set;
- profile/research/plan/edit/repair cycles and optimization dropout;
- layered GPU wiki, reference kernels, and upstream source lookup;
- explicit budgets and recovery state;
- terminal validation, same-allocation ABBA verification, and squash promotion;
- canonical `memory/v<N>.json`, journals, profiler artifacts, and aggregation provenance.

Duplicating that control plane in AKW would introduce a second authority. This workflow therefore performs only four actions: doctor the checkout and command, launch the official orchestrator once, audit canonical evidence, and report the official promoted result.

## Authority Boundary

| Evidence / decision | Owner |
| --- | --- |
| Public problem derivation and hidden workload isolation | Atrex orchestrator |
| Profiling, optimization dropout, knowledge retrieval | Atrex episode engine |
| Complete-workload correctness | Atrex terminal validator |
| Strict performance improvement | Atrex same-allocation ABBA verifier |
| Promotion | Atrex supervisor and squash promotion |
| Adapter readiness and final evidence audit | This AKW workflow |

`memory/live.json` is observability state, not promotion evidence. Only canonical `memory/v<N>.json` and the official terminal/ABBA records may justify success.

## Core Flow

```text
Doctor existing Atrex checkout and command
→ launch orchestrator/optimize.py exactly once
→ official Atrex campaign owns all inner loops
→ audit canonical memory + journals + terminal validation + ABBA + promotion
→ return exact promoted artifact and evidence paths
```

## Usage

```javascript
Workflow({name: 'atrex-kernel-optimization', args: {
  atrex_root: '/abs/path/atrex-kernel-agent',
  operator_input: 'atrex-bench/attention_forward',
  atrex_command: 'python3 {atrex_root}/orchestrator/optimize.py --operator {operator_input} --workspace {exp_dir} --platform {platform} --framework {framework} --mode {mode} --max-iters {max_iters}',
  platform: 'H20',
  framework: 'cuda',
  mode: 'production',
  max_iters: 300,
  exp_dir: '/tmp/atrex-akw',
  min_speedup: 1.01,
  turn_timeout_min: 720,
}})
```

The workflow never builds a command from assumptions. `atrex_command` is caller-owned, and the doctor rejects it unless it invokes the checkout's official `orchestrator/optimize.py` and keeps campaign state below `exp_dir`.

## Supported Scope

Use for:

- SOL-ExecBench operator optimization;
- native Atrex-Bench operators;
- NVIDIA CUDA or AMD ROCm campaigns configured by the official repositories;
- Triton, Gluon, FlyDSL, or CuteDSL candidates supported by Atrex-Bench.

Do not use when the Atrex checkout, GPU gateway/platform configuration, or official command is missing. For an ordinary project-owned harness without Atrex, use [HarnessEngineering](../HarnessEngineering/).

## Fidelity

`strict_high_fidelity`: the official Atrex checkout executes the method. AKW does not reproduce the Long Horizon engine, profile-driven optimizer, optimization dropout, hidden evaluator, or promotion policy.

## References

- [Atrex paper](https://arxiv.org/abs/2607.14541)
- [alibaba/atrex-kernel-agent](https://github.com/alibaba/atrex-kernel-agent)
- [alibaba/atrex-bench](https://github.com/alibaba/atrex-bench)
