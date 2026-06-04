# KernelSkill Workflow

Multi-agent CUDA kernel optimization with a **dual-level memory**, adapted from the KernelSkill paper:

- Paper: [KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization](https://arxiv.org/abs/2603.10085)
- Reference implementation: [github.com/0satan0/KernelMem](https://github.com/0satan0/KernelMem) (CC-BY-4.0)
- Workflow file: [`kernelskill-kernel-optimization.js`](kernelskill-kernel-optimization.js)

**English** · [简体中文](README.zh-CN.md)

---

## What it does

KernelSkill generates a custom CUDA kernel from a PyTorch reference and then refines it over multiple iterations. Its central idea is to **replace the opaque, implicitly-learned heuristics inside the LLM with explicit, externalized "expert optimization skills"** — so the choice of *what* optimization to apply is knowledge-driven and auditable.

This is realized as a dual-level memory:

- **Long-term memory** — a curated, cross-task **skill library** with two stores:
  - a **Deterministic Decision Policy** (`field_mapping`, `derived_fields`, headroom tiers, bottleneck priority rules, and a decision table → `allowed_methods`) that *screens* feasible methods from profiling + code evidence;
  - **Method Knowledge** (rationale + implementation cues) used only *after* gating, to make the selected action interpretable.
- **Short-term memory** — per-task **optimize-history** (methods tried + measured outcomes) and **repair-chain** memory (to avoid cyclic repair).

---

## The loop

1. **Setup** — read the PyTorch reference, measure the Torch eager baseline.
2. **Seed** — Generator produces several correctness-first candidate kernels; the Reviewer evaluates them; the best valid one becomes the working kernel.
3. **Refinement iterations** (paper uses 15) — each round is a **two-branch** control flow:
   - **Repair branch** (kernel invalid): `Diagnoser → Repairer`, conditioned on chained repair memory so the model does not oscillate between the same faulty variants.
   - **Optimize branch** (kernel valid): `Feature Extractor → deterministic Gate (allowed_methods) → Retrieve method knowledge → Planner → Optimizer`, with the Planner conditioned on short-term optimization memory.
4. **Report** — best kernel, speedup trajectory, skills applied, repair history.

```
                        ┌──────── Reviewer (Compiler+Verifier+Profiler ncu/nsys) ────────┐
 Seed kernels ──▶ best ─┤                                                                 │
                        ▼                                                                 │
                   valid? ──no──▶ Diagnoser ──▶ Repairer ──┐                              │
                     │ yes                                  │                             │
                     ▼                                      │                             │
   Feature Extractor ▶ Gate(allowed_methods) ▶ Retrieve ▶ Planner ▶ Optimizer ──┐         │
                                                                                 ▼         │
                                                              update best + short-term mem ┘
                                                                       │ (next round)
```

The selection metric is **speedup = reference (Torch eager) latency / test latency**; `ncu` memory/SM/occupancy metrics and `nsys` kernel-launch counts drive *which* method is selected.

---

## Recommended use cases

- KernelBench-style operator/model tasks (Level 1–3), PyTorch → custom CUDA.
- When you want method selection to be **explicit and auditable** (gate-screened `allowed_methods`) rather than relying on the model's implicit judgement.
- Multi-round refinement where stabilizing repair (no cyclic fixes) matters.

---

## Prerequisites

- A PyTorch reference task file (`Model` + `get_inputs` / `get_init_inputs`, KernelBench format).
- NVIDIA GPU + CUDA toolchain; PyTorch with GPU support.
- Recommended: NVIDIA Nsight Compute (`ncu`) and Nsight Systems (`nsys`) for profiling-driven method selection. Without them the workflow falls back to static reasoning.

---

## Arguments

| Arg | Required | Default | Description |
|-----|----------|---------|-------------|
| `reference_path` | ✅ | — | Path to the PyTorch reference task (`.py`) |
| `op_description` | | `'PyTorch operator/model'` | Human-readable description |
| `target_gpu` | | `'A100-80GB'` | Target GPU for prompts/profiling |
| `iterations` | | `15` | Number of refinement iterations |
| `seed_candidates` | | `3` | Correctness-first seed kernels generated up front |
| `skill_library_path` | | `''` | Path to a long-term skill library YAML/JSON. **Overrides** the embedded library |
| `benchmark_command` | | `''` | Compile + benchmark command |
| `ncu_binary` | | `'ncu'` | Nsight Compute binary |
| `nsys_binary` | | `'nsys'` | Nsight Systems binary |
| `tol` | | `0.01` | Max abs error tolerated vs reference |
| `warmup` / `repeat` | | `25` / `100` | Benchmark iterations |
| `exp_dir` | | `'/tmp/kernelskill_exp'` | Output dir for kernels/profiles/memory |
| `run_timestamp_iso` | | `'unknown'` | ISO timestamp for report metadata |

The embedded **skill library** (long-term memory) is faithful to `memorybank/bottleneck_headroom_kernelstructure.yaml` from the upstream repo (NCU `field_mapping`, `derived_fields`, headroom tiers, decision table → `allowed_methods`). Point `skill_library_path` at your own file to extend or replace it across tasks.

---

## Example invocation

```javascript
Workflow({
  name: 'kernelskill-kernel-optimization',
  args: {
    reference_path: '/path/to/KernelBench/level1/19_ReLU.py',
    op_description: 'ReLU activation over a large tensor',
    target_gpu: 'A100-80GB',
    iterations: 15,
    seed_candidates: 3,
    benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
    ncu_binary: '<user-provided ncu binary path>',
    nsys_binary: '<user-provided nsys binary path>',
    tol: 0.01,
    warmup: 25,
    repeat: 100,
    exp_dir: '/tmp/kernelskill_exp',
    run_timestamp_iso: '2026-06-02T15:00:00+08:00',
  },
})
```

---

## Outputs

The workflow returns:

- `baseline_latency_ms` — Torch eager baseline
- `best_speedup` — best speedup vs eager
- `best_kernel_code` — source of the best valid kernel
- `rounds_completed`
- `skills_applied` — per-round methods + resulting speedups
- `speedup_trajectory` — best speedup after each round
- `opt_memory` — short-term optimization memory
- `report` — final technical report

---

## Differences from the reference implementation

This workflow is a faithful **decision-process** adaptation, not a line-for-line port:

- The deterministic `machine_check` gate is expressed as an in-prompt policy (embedded skill library) rather than a separate Python module; the Planner is instructed to honour `allowed_methods` as a **hard constraint**, matching the upstream `judger_optimization_memory_latest.py` contract.
- The seed/repair/optimize agent roles map to the upstream Generator / Reviewer (Compiler+Verifier+Profiler) / Feature Extractor / Planner / Optimizer / Diagnoser / Repairer.
- Short-term optimize-history and repair-chain memory are carried in workflow state; long-term skills can be persisted via `skill_library_path`.

---

## Citation

```bibtex
@article{sun2026kernelskill,
  title={KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization},
  author={Sun, Qitong and Han, Jun and Li, Tianlin and Tang, Zhe and Chen, Sheng and Yang, Fei and Liu, Aishan and Liu, Xianglong and Liu, Yang},
  journal={arXiv preprint arXiv:2603.10085},
  year={2026},
  url={https://arxiv.org/abs/2603.10085}
}
```
