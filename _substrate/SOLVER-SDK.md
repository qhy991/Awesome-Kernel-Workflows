# KerSor Solver SDK — the shared substrate

A **solver** is a kernel-optimization workflow. This SDK is the part every solver
should share, so that "集各家所长" happens by *composition* (a common substrate ×
heterogeneous search topologies, orchestrated by KerSor) rather than by building
one monolith. It is grounded in a 26-solver audit
([`SOLVER-AUDIT.md`](./SOLVER-AUDIT.md)): the components below are the
best-of-breed pick on each orthogonal axis, and each is only included **with its
executable evidence owner** (no prompt-only mechanisms).

> How the substrate, a directly-callable **generalist solver**, and **KerSor**
> relate — and how each is invoked — is in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## What is shared vs what stays per-solver

| Shared substrate (this SDK — every solver inherits) | Per-solver (the differentiator) |
|---|---|
| Evidence contract, anti-cheat gate, diagnosis, persistent memory, method gate, cost-control gates | **Search topology / controller** (beam / MCTS / evolutionary / bandit / graph) and generation prompts |

Topologies are *mutually exclusive alternatives* — they must not be stacked in one
solver. Heterogeneity across topologies is exactly what KerSor's portfolio +
transfer object exist to exploit. The substrate makes every topology stronger and
mutually comparable; it does not replace them.

`_meta/templates/{iterative-loop,search-based,tree-exploration,single-pass}.js`
already encode the topology layer. The SDK is the contract those templates import.

---

## Layer A — Evidence contract  *(keystone; borrowed from Astra/CUDAAgent/CUDALLM)*

Every attempt emits one record. **This is byte-for-byte the schema KerSor's
`result-analyzer` and `select-workflow.sh` already consume** — so a conformant
solver feeds KerSor natively, with no prose reverse-engineering.

```json
{
  "attempt_id": "a3", "parent_id": "a1",
  "compiled": true, "correct": true, "speedup": 1.23,
  "metrics": { "latency_ms": 0.41, "dram_pct": 62, "sm_pct": 48, "occupancy": 0.5 },
  "best_kernel": "candidate_v3.cu",
  "convergence_status": "budget_exhausted",
  "insights": [ { "kind": "...", "directive": "...", "evidence": "...", "confidence": "..." } ],
  "failed_strategies": [ { "...": "..." } ]
}
```

`insights`/`failed_strategies` are typed transfer items
(`KernelNav/docs/transfer-object.md`). A solver that emits these natively makes
cross-solver transfer free.

## Layer B — Anti-cheat gate  *(deterministic; CUDAAgent + CUDALLM + TritorX + AKO4X)*

A shared validator that runs before any speedup is recorded. Returns
`valid: false` + reason if it detects:
- hardcoded shapes / skipped compute / PyTorch-or-library fallback / silent-skip;
- a speedup that does **not beat both** eager and `torch.compile` (robust reward:
  `r=3` beats both, `r=2` beats eager only, `r=-1` fail);
- (AKO4X) a result whose pre-committed `Expected` hypothesis was not written
  *before* benchmarking — guards against retrofitted explanations.

An invalid attempt yields `speedup=0`, never enters memory. This is a single
reusable module, not re-implemented per solver.

## Layer C — Diagnosis  *(AccelOpt + KernelBand φ + KernelBlaster + cuPilot/KEET)*

Profile → a **shared bottleneck taxonomy** so memory keys line up across solvers:

```
bottleneck_class ∈ { memory_bound | compute_bound | latency_occupancy | overhead_bound }
```

- One-shot NCU/roofline → class + dense metrics (real `ncu`/profiler required;
  degrade to `unknown` if absent — never fabricate).
- (KEET) hypothesis-first: form falsifiable predictions from source *before*
  reading the profile, to block confirmation bias.
- (ARGUS, optional) if an executable invariant checker is provided, emit
  tag-assertion counterexamples as dense `evidence: "invariant"` items.

## Layer D — Persistent memory  *(KernelBlaster + AKO4X + AdaExplore)*

The biggest compounding advantage most solvers lack. A **cross-run, cross-kernel**
store keyed by `bottleneck_class`:

```json
{ "memory_bound": [
    { "technique": "vectorize_128b", "confidence": 0.7,
      "usage_count": 9, "avg_actual_speedup": 1.18 } ],
  "dead_ends": [ { "claim": "split-K atomics", "why": "atomic contention at this N", "revalidate_if": "N<512" } ],
  "traps": [ { "pattern": "silent shape assumption", "symptom": "..." } ] }
```

- `confidence` blends prior with **measured** outcome (decays on failure, grows on
  payoff) — KernelBlaster.
- Dead-ends carry **WHY + revalidate_if**, not blind prohibition — AKO4X (so a
  toolchain change can re-open them).
- Persists at a configurable path; this is what enables **cross-session** transfer
  in KerSor, beyond per-round.

## Layer E — Method gate  *(deterministic; KernelSkill)*

Normalized metrics → decision table → `allowed_methods`. The LLM may only pick
*within* the gated set; a non-binding `llm_assist` rationale comes **after**
gating, never drives it. Same "evidence-guided, not LLM-discretion" principle as
KerSor's own routing — applied inside the solver.

## Layer F — Cost-control gates  *(CutlassGEMM + KernelBand + StitchCUDA + Xe-Forge)*

Composable stop/skip gates (all stack):
- **ceiling detection** → switch to library fallback when overhead-bound (CutlassGEMM);
- **hardware masking** → skip strategies whose target resource is already saturated >75% (KernelBand);
- **replanning trigger** → 2 consecutive failures or CV<5% stagnation (StitchCUDA);
- **early-term** → improvement <5% (Xe-Forge); dual-stagnation (KSearch).

---

## Conformance levels (checked by `_meta/tools/validate-workflow.js`)

- **L0 — KerSor-compatible:** emits Layer A evidence (incl. typed insights). *Required* for any solver in the catalog.
- **L1 — Honest:** + Layer B anti-cheat gate wired to a real harness.
- **L2 — Diagnostic:** + Layer C shared bottleneck taxonomy.
- **L3 — Compounding:** + Layer D persistent memory + Layer E method gate + Layer F gates.

Validator adds checks: does the return envelope match Layer A? are insights typed?
is there an anti-cheat hook? is the bottleneck taxonomy the canonical enum?

## KerSor integration (why this is also a paper contribution)

- Layer A **is** the transfer-object / `analysis.json` interface → makes C2
  ("unified evidence interface") literally an SDK every solver implements.
- Layer C bottleneck taxonomy = the routing features `select-workflow.sh` scores on.
- Layer D persistent memory = cross-session transfer, a strict superset of the
  per-round transfer we already built.

## Build order (recommended)

1. **Layer A schema module + validator L0 check** — the keystone; unlocks native KerSor feed.
2. **Layer B anti-cheat module** (`_substrate/anti_cheat.*`) + **Layer C diagnosis module** (`_substrate/diagnose.*`) — deterministic, unit-testable, reused by all.
3. **Retrofit one strict solver** (AKO4X or KDA) to L1/L2 as proof-of-concept.
4. **Layer D persistent memory store** + Layer E gate; retrofit to L3.
5. Backfill remaining solvers to L0 (cheap) over time.
