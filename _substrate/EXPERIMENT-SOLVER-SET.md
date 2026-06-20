# Experiment Solver Set — topology de-duplication

**Status: CONFIRMED — NCU available; 5 solvers (one per topology family).**
Derived from [`SOLVER-AUDIT.md`](./SOLVER-AUDIT.md). This is the operative pool in
`paper/kernelnav-experiment-design.md`. Lock it at invocation with `--workflows`
(KerSor's default stays "all 26"; the 5-pool is an experiment override, not a
behavior change). Canonical reference list also stored in KerSor
`config/default_config.json` → `experiment_workflows`.

**Final pool (5):** `AKO4X · AdaExplore · KernelBand · cuPilot · KernelAgent`
— iterative / MCTS / bandit / evolutionary / generation, five non-overlapping
topologies, all executable. Per-session budget stays 3 (each session picks ≤3 from
this pool).

`--workflows` string:
`ako4x-kernel-optimizer,adaexplore-kernel-optimization,kernelband-kernel-optimization,cupilot-kernel-optimization,kernelagent-triton-synthesis`

## Why de-duplicate by topology (not just "use fewer")

The previously-planned set — **AKO4X / KDA / AdaExplore / KSearch** — is
**2 iterative + 2 tree**: topology-redundant, with no bandit / evolutionary /
generation solver. That *weakens* the Phase-1 complementarity argument, whose
whole point is "different kernel families favor different topologies." Shrinking
should therefore keep **one strongest, most-executable representative per topology
family**, maximizing topology spread so no-single-dominator is most likely to show.

## Selection criteria

1. Highest **fidelity** within the family (executable evidence owner > prompt-mediated).
2. Strongest distinctive mechanism.
3. **Backend-comparable** on the same NVIDIA benchmark (CUDA/Triton, runs on SOL-ExecBench-style tasks).
4. **Distinct topology** from the others (diversity is the point).

## Clustering and per-family pick

| Family | Candidates | Pick | Why strongest in family | Runner-up |
|---|---|---|---|---|
| Iterative self-improving | AKO4X, KDA, AccelOpt, CUDAAgent, CUDALLM, Astra, StitchCUDA, KernelSkill | **AKO4X** | strict high-fidelity; two-layer loop + pre-commit Expected + TRAPS + dead-ends(WHY) + archive gates | KDA (lighter; promotion gate + lineage) |
| Tree / MCTS / graph | KSearch, AdaExplore, STARK, ReGraphT, GPUForecasters | **AdaExplore** | high-fidelity MCTS, executable evaluator, failure-driven skill memory + diversity | KSearch (world-model tree) |
| Population (evolutionary/QD/bandit) | KernelBand, cuPilot, KernelFoundryDx, KernelFoundry | **KernelBand** | all signals executable (φ features + NCU masking + measured reward), hardware-aware pruning | cuPilot (NCU + strategy crossover) |
| Generation (spec→kernel) | KernelAgent, TritorX, FACT, CutlassGEMM | **KernelAgent** | sandboxed executable verification, parallel seeds, auto-routing; covers the spec-to-kernel task mode | TritorX (operator-gen, niche) |

## Final set (5 solvers, 5 distinct topologies, all executable)

> **AKO4X (iterative) · AdaExplore (MCTS) · KernelBand (bandit) · cuPilot (evolutionary) · KernelAgent (parallel synthesis / generation)**

Spans CUDA + Triton, optimize-existing + generate-from-spec. The five topologies do
not overlap — the configuration most likely to surface complementarity.

**The 5th = cuPilot, not KSearch.** Rationale: KSearch would add a *second tree*
(overlapping AdaExplore's MCTS); cuPilot adds a genuinely new **evolutionary**
topology (strategy-level crossover + roofline), and is NCU-driven — consistent with
the confirmed NCU-available environment. Maximizing topology spread beats within-tree
contrast for the complementarity argument.

## Excluded from the main experiment (with reasons)

| Solver | Reason |
|---|---|
| KEET | diagnostic-only, not an optimizer (KerSor's role gate already filters it) |
| Xe-Forge, KernelFoundry | Intel XPU / SYCL backend — not comparable on an NVIDIA benchmark |
| CutlassGEMM, FACT | CUTLASS/GEMM-only — operator coverage too narrow (keep as optional GEMM case study) |
| KernelBlaster | value is cross-run persistent-memory compounding; a single-session experiment can't show it — keep as a separate persistent-memory ablation/case study |
| AccelOpt, CUDALLM, Astra, CUDAAgent, KDA, KernelSkill, StitchCUDA, STARK, ReGraphT, GPUForecasters, cuPilot, KernelFoundryDx, TritorX | redundant within their family vs the picked best (listed as runner-ups above) |

## Decisions (resolved)

1. **NCU availability** → **NCU available.** Population pick = **KernelBand** (NCU
   masking + measured reward); cuPilot (also NCU) admitted as the 5th. No
   KernelFoundryDx fallback needed.
2. **4 or 5** → **5.** The 5th is **cuPilot** (new evolutionary topology), not
   KSearch (would duplicate the tree family).

## Caveat on conclusions

Because the main experiment runs a topology-deduplicated 4-solver set, any
"KerSor approximates the oracle portfolio" claim is over **this** portfolio. The
excluded solvers (esp. the GEMM-specific and cross-run-memory ones) are not
absent because they are weak but because they are non-comparable or single-session
-invisible; state this explicitly so reviewers don't read the 4-set as the whole field.
