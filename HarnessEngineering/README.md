# Harness Engineering Kernel Optimization

**English** · [简体中文](README.zh-CN.md)

An evidence-first workflow adaptation of [Harness Engineering for LLM-Driven GPU Kernel Generation](https://arxiv.org/abs/2607.17979), with its public [MLSys 2026 FlashInfer contest artifact](https://github.com/syhya/mlsys26-flashinfer-contest).

## What It Preserves

This workflow separates two responsibilities:

| Responsibility | Owner |
| --- | --- |
| Profile-backed hypothesis selection | Workflow controller |
| Candidate implementation | Workflow implementer, under `exp_dir` only |
| Compile and correctness verdict | Caller-supplied `test_command` |
| Deeper safety/portability verdict | Optional caller-supplied `verification_command` |
| Timing and speedup | Caller-supplied `benchmark_command` |
| Candidate retention and promotion | Deterministic workflow gate |

The evaluation contract is frozen before optimization. A candidate is benchmarked only after it compiles, passes correctness, and—when configured—passes the selected deeper verification profile. It replaces the incumbent only when the measured result is strictly better.

This is a **workflow adaptation**, not a reproduction of the contest infrastructure, hidden tests, or the paper's model configuration.

## Core Loop

```text
Freeze harness contract
→ correctness-gated baseline
→ optional profile + one bounded hypothesis
→ candidate under exp_dir
→ official correctness / verification / timing gate
→ keep strictly better candidate or reject
→ artifact/evidence audit
```

## Evaluation Profiles

The profile names select an evidence expectation; they never replace an executable verifier:

| Profile | Intended emphasis |
| --- | --- |
| `contract-grade` | Contract-derived compile, numerical, safety, determinism, and portability obligations inspired by [Kernel Contracts](https://arxiv.org/abs/2608.12700) |
| `kernelbench-verified` | Hidden-distribution correctness, memory efficiency, and realistic TF32 baselines inspired by [KernelBench-Verified](https://arxiv.org/abs/2607.16241) |
| `kernelgenbench` | Multi-source, multi-hardware, and cost/quality provenance inspired by [KernelGenBench](https://arxiv.org/abs/2607.27231) |
| `custom` | Project-owned evidence contract |

See [`_substrate/verification/README.md`](../_substrate/verification/README.md). If `verification_command` is supplied, its `verified=true` verdict is mandatory for promotion.

## Usage

```javascript
Workflow({name: 'harness-engineering-kernel-optimization', args: {
  harness_root: '/abs/path/mlsys26-flashinfer-contest',
  kernel_path: '/abs/path/solution.py',
  problem_path: '/abs/path/config.toml',
  backend: 'cuda',
  test_command: './project-test --candidate {candidate_path} --json {result_path}',
  benchmark_command: './project-bench --candidate {candidate_path} --json {result_path}',
  profile_command: './project-profile --candidate {candidate_path} --out {artifact_path}',
  verification_command: './project-verify --candidate {candidate_path} --profile kernelbench-verified --json {result_path}',
  verification_profile: 'kernelbench-verified',
  iterations: 6,
  min_speedup: 1.05,
  exp_dir: '/tmp/harness-engineering-run',
}})
```

Commands are examples of the contract shape, not built-in tools. Supply commands owned by the target harness.

## Required Command Results

The workflow asks the harness adapter to normalize evidence into these fields:

```json
{
  "candidate_path": "/abs/path/to/candidate",
  "compiled": true,
  "correct": true,
  "verified": true,
  "latency_ms": 0.42,
  "speedup": 1.17,
  "evidence_path": "/abs/path/to/evidence.json",
  "artifact_paths": [],
  "notes": []
}
```

Absent or failed evidence must remain `false`/`null`; prose is never upgraded into a pass.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `harness_root` | required | Authoritative harness checkout. |
| `kernel_path` | required | Existing baseline solution. |
| `test_command` | required | Compile and correctness command template. |
| `benchmark_command` | required | Correctness-gated timing command template. |
| `profile_command` | empty | Optional profile command template. |
| `verification_command` | empty | Optional deeper verifier. |
| `verification_profile` | `contract-grade` | One of the shared evaluation profiles. |
| `iterations` | `4` | Candidate budget. |
| `min_speedup` | `1.01` | Early-success threshold. |
| `exp_dir` | `/tmp/harness-engineering` | Sole workflow write boundary. |

## References

- [Harness Engineering for LLM-Driven GPU Kernel Generation](https://arxiv.org/abs/2607.17979)
- [syhya/mlsys26-flashinfer-contest](https://github.com/syhya/mlsys26-flashinfer-contest)
- [Kernel Contracts](https://arxiv.org/abs/2608.12700)
- [KernelBench-Verified](https://arxiv.org/abs/2607.16241)
- [KernelGenBench](https://arxiv.org/abs/2607.27231)
