# GemmPTX: GEMM Optimization from PTX/SASS Evidence

**English** · [简体中文](README.zh-CN.md)

GemmPTX is a GEMM-specific workflow for optimizing an **existing CUDA/CuTe/CUTLASS GEMM kernel** from the instruction layer upward. It follows a human expert loop:

```
Hardware Census -> GEMM Signature -> Baseline Evidence
-> Instruction Plan -> Implement -> Disassemble Verify
-> Profile -> Decide -> Report
```

The key contract is simple: a candidate cannot claim `mma.sync`, `wgmma.mma_async`, TMA, or `tcgen05` success unless PTX/SASS evidence proves that the expected instruction path appeared. A candidate that compiles and is fast but does not lower to the planned instruction is recorded as `hypothesis_not_realized`.

## Scope

Use GemmPTX when:

- You already have a CUDA, CuTe, CUTLASS, or C++ GEMM-like kernel.
- You can provide compile, correctness, benchmark, and disassembly commands.
- You want to optimize instruction selection and lowering behavior: `mma.sync`, `wgmma.mma_async`, `cp.async.bulk.tensor` / TMA, mbarrier pipelines, or `tcgen05` / TMEM.

Avoid GemmPTX when:

- You need a generic compute-bound optimizer. This is **not a generic compute-bound optimizer**.
- The task is softmax, reduction, layernorm, stencil, elementwise, FFT, or sampling rather than GEMM/matmul.
- You cannot provide a `disassemble_command`; without disassembly the workflow cannot verify the PTX/SASS path.
- You want CUTLASS runtime dispatch threshold tuning for SOL-ExecBench; use [CutlassGEMM](../CutlassGEMM/) for that.

## Required Evidence

| Argument | Required | Contract |
|---|---:|---|
| `kernel_path` | yes | Existing GEMM kernel source. The workflow writes candidates under `exp_dir`, not over the original. |
| `compile_command` | yes | Command using `{candidate_path}` or `{kernel_path}` and `{result_path}`; should report `compiled` and an artifact path. |
| `test_command` | yes | Correctness command; should report `correct`. |
| `benchmark_command` | yes | Performance command; should report `latency_ms`, `throughput`, or `speedup`. |
| `disassemble_command` | yes | PTX/SASS evidence command; should report artifact paths, observed instructions, registers/thread, SMEM, local memory, and spills. |
| `hardware_probe_command` | no | Runtime GPU facts. If absent, the workflow can only use static/fallback facts and marks them as unmeasured. |
| `profile_command` / `ncu_command` | no | Optional NCU/native profile metrics for mechanism diagnosis. |

Commands may use these placeholders: `{kernel_path}`, `{candidate_path}`, `{artifact_path}`, `{result_path}`, `{exp_dir}`, `{target_gpu}`.

## Workflow-Local Skill

GemmPTX ships a local expert skill at `GemmPTX/skills/gemmptx-instruction-evidence/SKILL.md`. The workflow treats it as an optional skill binding named `gemmptx-instruction-evidence`; planning, implementation, and disassembly verification prompts ask agents to read it when available.

The skill contains the compact rule pack for architecture-to-instruction mapping, GEMM triage, PTX/SASS regex evidence gates, and common WGMMA/TMA/`tcgen05` failure modes. KerSor or another runner may inject this skill, but the canonical source lives with the workflow in AKW.

## Example

```javascript
Workflow({name: 'gemmptx-gemm-optimization', args: {
  kernel_path: '/abs/project/src/gemm.cu',
  problem_definition: 'bf16 GEMM: C[M,N] = A[M,K] @ B[K,N], fp32 accumulation',
  language: 'cuda',
  target_gpu: 'H100',
  exp_dir: '/tmp/gemmptx-run',
  iterations: 3,

  hardware_probe_command: '/abs/tools/probe_gpu.py --json > {result_path}',
  compile_command: '/abs/project/tools/build_candidate.sh {candidate_path} {artifact_path} > {result_path}',
  test_command: '/abs/project/tools/test_candidate.sh {candidate_path} > {result_path}',
  benchmark_command: '/abs/project/tools/bench_candidate.sh {candidate_path} > {result_path}',
  disassemble_command: '/abs/project/tools/disasm_candidate.sh {artifact_path} --ptx --sass > {result_path}',
  profile_command: '/abs/project/tools/profile_candidate.sh {artifact_path} > {result_path}',
}})
```

## Candidate Statuses

| Status | Meaning |
|---|---|
| `compile_error` | Candidate did not compile. |
| `incorrect` | Candidate compiled but failed correctness. |
| `hypothesis_not_realized` | Candidate compiled and was correct, but PTX/SASS did not contain the expected instruction regex. |
| `rejected` | Candidate was correct and instruction-verified, but did not improve enough. |
| `accepted` | Candidate was correct, instruction-verified, and improved over the best measured result. |

## Fidelity Boundary

This is an original AKW engineering workflow, not a strict reproduction of a single paper. Its load-bearing mechanism is the evidence loop: hardware facts and GEMM signature propose an instruction hypothesis, compile/test/disassembly verify the hypothesis, and benchmark/profile decide whether to accept it.

The workflow is strongest for tensor-core GEMM work. For broader compute-bound tasks, use the same evidence schema but a different operator-specific rule pack.
