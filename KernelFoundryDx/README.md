# Kernel Foundry (Diagnosis-driven, Multi-Experts) Workflow

Island-model evolutionary Triton-kernel optimizer with expert-guided RAG initialization and a self-refining diagnosis-driven hint library, adapted from:

- Paper: [Kernel Foundry: A Diagnosis-driven Evolutionary Kernel Optimizer with Multi-Experts](https://arxiv.org/abs/2605.30359) (arXiv:2605.30359)
- Workflow file: [`kernelfoundrydx-kernel-optimization.js`](kernelfoundrydx-kernel-optimization.js)

**English** · [简体中文](README.zh-CN.md)

---

> **Naming note.** This is a **different paper** from the Intel [`KernelFoundry/`](../KernelFoundry/)
> (arXiv:2603.12440, SYCL, MAP-Elites) already in this repo. They share the "Kernel Foundry" name
> but are distinct works (different authors, arXiv IDs, methods, and target backend). This one
> (CUHK + Huawei Noah's Ark Lab, **Triton**) lives under `KernelFoundryDx/` — *Dx* for
> **D**iagnosis-driven — to avoid colliding with the existing directory.

---

## What it does

`kernelfoundrydx-kernel-optimization` treats kernel synthesis as an **iterative evolutionary
optimization** problem rather than one-shot generation. It runs several **role-specialized
islands** in parallel, each evolving its own population of Triton kernels, and a centralized
**experience/hint library** that learns which natural-language optimization hints actually
correlate with measured speedups.

Loop:

1. **Setup** — read the PyTorch reference, benchmark the eager baseline, seed the hint library.
2. **Init** — expert-guided RAG initialization: retrieve similar verified PyTorch→Triton pairs,
   a domain expert model generates correct seeds, anti-cheating validates each.
3. For each iteration, per island:
   - **Evolve** — LLM mutates a parent, conditioned on the island's role, retrieved hints, and history.
   - **Evaluate** — compile + run on real hardware, measure correctness and speedup (lightweight
     signals only — no ncu/nsys), anti-cheating check.
   - **Diagnose** — Result Analyzer classifies each candidate: failure mode (incorrect) or coarse
     limiter `memory-bound / latency-bound / instruction-bound` (correct), then emits hints.
   - **Evolve-Pop** — update populations + elite archives, reinforce/down-weight hints by observed
     speedup, and probabilistically migrate elites between islands on stagnation.
4. **Report** — best valid kernel across all islands, hint-library evolution, island trajectories.

---

## Key ideas (faithful to the paper)

- **Diagnosis-driven hints.** A three-stage Result Analyzer (Signal Extraction → Diagnosis Engine →
  Hint Generation) turns each evaluation into reusable, tagged natural-language hints. Hints are
  reinforced when their use correlates with speedups and down-weighted otherwise.
- **Multi-experts.** Two senses: (1) a small domain-specialized **expert model** produces correct
  RAG-seeded initial kernels; (2) **role-specialized islands** each get a different system prompt and
  hint subset (operator fusion / memory access / parameter tuning / instruction optimization).
- **Island-model evolution.** Independent persistent populations with local elite archives;
  coordination via probabilistic elite migration, not voting or merging.
- **Anti-cheating.** Two-level: prompt constraints (valid `@triton.jit`, all compute in Triton) plus
  an LLM validator that scores cheating likelihood; candidates above `cheating_threshold` are discarded.
- **No profiler dependency.** Bottleneck class is inferred from runtime stats + launch config — the
  method deliberately avoids heavy profiling.

---

## Arguments

| Arg | Default | Meaning |
|-----|---------|---------|
| `ref_path` (required) | — | PyTorch reference module (.py) to port to Triton |
| `op_description` | `'PyTorch operator'` | Human-readable op description |
| `target_gpu` | `'RTX5090'` | Target GPU (paper: RTX 5090, A800-80G for memory-heavy) |
| `num_islands` | `4` | Number of role-specialized islands |
| `iterations` | `5` | Evolution iterations (paper used 30) |
| `population_size` | `3` | Population per island |
| `migration_stagnation` | `2` | Non-improving iterations before elite migration |
| `bench_command` | `''` | Compile + benchmark command |
| `rtol` / `atol` | `0.01` | Correctness tolerance |
| `retrieval_corpus_path` | `''` | Verified PyTorch→Triton pairs for RAG init |
| `hint_library_path` | `''` | Persisted hint library JSON (cross-task reuse) |
| `exp_dir` | `/tmp/kernelfoundrydx_exp` | Output directory |
| `cheating_threshold` | `0.5` | Discard candidates above this cheating likelihood |

---

## Example invocation

```javascript
Workflow({
  name: 'kernelfoundrydx-kernel-optimization',
  args: {
    ref_path: '/path/to/KernelBench/level2/95_Matmul_Add_Swish.py',
    op_description: 'Matmul + Add + Swish + Tanh + GELU + Hardtanh fusion',
    target_gpu: 'RTX5090',
    num_islands: 4,
    iterations: 30,
    population_size: 4,
    bench_command: 'python eval_triton.py --kernel',
    retrieval_corpus_path: '/path/to/verified_triton_pairs.jsonl',
    hint_library_path: '/path/to/kernelfoundrydx_hints.json',
    exp_dir: '/tmp/kernelfoundrydx_exp',
  },
})
```

Run inside a container/venv with a Triton toolchain — agents compile and run kernels on the GPU.

---

## Outputs

- `best_speedup`, `best_kernel_code`
- `iterations_completed`, `kernels_evaluated`
- `islands_summary` (per-island role + best speedup)
- `hint_library` (final hints with confidence stats — persist via `hint_library_path`)
- `report`

---

## Notes on fidelity

No public code repository was available for this paper as of the preprint, so this workflow is a
faithful adaptation of the method as described in the paper text and appendices (the Evolver system
prompt, the expert-recommendation hint taxonomy grouped by Correctness / Memory-bound /
Instruction-bound / Latency-bound, and the worked evolution examples). The retrieval corpus and hint
library are exposed as paths so you can supply your own KernelBook-style data and accumulate hints
across runs.

---

## Citation

```bibtex
@article{huang2026kernelfoundry,
  title={Kernel Foundry: A Diagnosis-driven Evolutionary Kernel Optimizer with Multi-Experts},
  author={Huang, Zixuan and Chen, Da and Huang, Kecheng and Yin, Lihao and Li, Xing and Zhen, Huiling and Yuan, Mingxuan and Shao, Zili},
  journal={arXiv preprint arXiv:2605.30359},
  year={2026},
  url={https://arxiv.org/abs/2605.30359}
}
```
