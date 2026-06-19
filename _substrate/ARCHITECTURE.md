# Architecture & invocation: substrate · generalist solver · KerSor

How the three pieces relate, and how each is actually called. Companion to
[`SOLVER-SDK.md`](./SOLVER-SDK.md) (the substrate contract) and
[`SOLVER-AUDIT.md`](./SOLVER-AUDIT.md) (the 7+1 axis decomposition this builds on).

## The one-line relationship

The audit splits the design space into **7 composable component axes + 1 exclusive
topology axis**. That split *is* the relationship:

> **Generalist solver = max out the 7 composable axes** (the most a single solver
> can internalize). **KerSor = handle the 1 exclusive topology axis** (what no
> single solver can — route across topologies + transfer evidence).

They are orthogonal and complementary, not competitors. KerSor exists *because*
topology cannot be stacked into one solver.

## You never "call" the substrate

The substrate (`_substrate/`) is a **build-time dependency, not a call-time
entity.** Nothing invokes it directly. You invoke a *solver* that uses it, or
KerSor. "Directly callable" is therefore **not** the dividing line between a
monolith and the substrate approach — both produce directly-callable solvers.

## Three callable forms

| | Monolith mega-workflow | Substrate → generalist solver | Substrate → KerSor |
|---|---|---|---|
| How to call | `Workflow({name:'apex-...', args})` | `Workflow({name:'generalist-...', args})` | `/kersor:optimize kernel.cu` |
| Directly callable? | yes | **yes (identical call shape)** | yes (call the orchestrator) |
| Internals | all components hardcoded in one file | same file; agent steps call `_substrate/` shared scripts | routes substrate-conformant solvers + transfer |
| Topology | all stacked (category error) | one controller + substrate | heterogeneous, routed per round |
| Component reuse | none (26 solvers each copy) | shared (fix once, all benefit) | shared |

The generalist's call shape is **identical to the monolith's** — the caller never
sees the substrate. The only difference is internal: its agent steps run
`_substrate/anti_cheat.*`, `diagnose.*`, etc. instead of inlining them.

## How the substrate is pulled in (technical reality)

The Claude Code `Workflow` runtime has **no filesystem / no module import** — a
workflow `.js` body cannot `require` modules or read disk. So the substrate is
**not** JS imports. It is realized exactly like KerSor's existing scripts/hooks:

1. **Deterministic components = standalone scripts** (`_substrate/anti_cheat.py`,
   `diagnose.py`, `memory_store.py`, …), invoked by a solver's **agent steps via
   Bash** (agents have Bash; the workflow body does not) — same pattern as KerSor's
   `select-workflow.sh` / `render-handoff.sh`.
2. **Contract / prompt conventions = baked by the template.**
   `_meta/generate-workflow.js` injects the Layer A evidence schema, the anti-cheat
   hook call, and the diagnosis enum into each generated solver. Solvers stay
   self-contained and directly callable; they "inherit" the substrate at generation
   time, not at runtime.

## Layering

```
        KerSor (orchestration)        ← the TOPOLOGY axis: per-round routing + cross-solver transfer
        ├── Generalist solver  (substrate + beam/MCTS)   ← strongest default member
        ├── AKO4X              (substrate + iterative)
        ├── AdaExplore         (substrate + MCTS)
        ├── KernelBand         (substrate + bandit)
        └── cuPilot / KernelAgent ...
              ↑ all stand on the SAME shared substrate (the 7 component axes)
```

## Backend Driver axis -- a cross-cutting data axis

The Backend Driver axis is the third design axis (alongside the universal substrate and
the exploration topology). It extracts the `(source language) x (hardware/profiler
vendor)` product that was previously welded into each workflow's body -- `nvcc`, `ncu`,
`__global__`, `.cu` in prompt strings -- into a pluggable, per-backend directory under
`_substrate/backends/<backend_id>/`. Each driver contains six files (`manifest.json`,
`build.sh`, `run.sh`, `profile.sh`, `to_evidence.py`, `idioms.json`) that adapt native
backend tooling to the universal substrate vocabulary. A workflow switches backends by
setting `args.backend`; the method body never names a vendor profiler or vendor metric
directly.

The Backend Driver axis is **owned by neither the solver nor the generator**. It is a
cross-cutting data layer that sits between the workflow body (which builds string paths
from `args.backend`) and the substrate scripts (which consume canonical metrics). The
driver translates native profiler output into the canonical four-key metric dict
(`latency_ms`, `dram_pct`, `sm_pct`, `occupancy`) via `to_evidence.py`, and maps
abstract `method_gate.TABLE` names to concrete backend idioms via `idioms.json`. This
separation means any clean optimization method can run on any backend as a matrix:
`method x backend`, without re-authoring prompts or shell strings.

See [`BACKEND-DRIVER-SDK.md`](./BACKEND-DRIVER-SDK.md) for the full driver contract and
[`backends/REGISTRY.md`](./backends/REGISTRY.md) for the per-driver index.

## Three concrete relationships between generalist and KerSor

1. **Portfolio member.** The generalist is one solver in KerSor's pool — its
   strongest default member. KerSor can route *to* it like any other solver.
2. **Toughest single-solver baseline.** In the paper, the generalist is the hardest
   `Best Single Solver` baseline: it answers "does KerSor's topology orchestration
   beat even a single solver whose *components* are all maxed out?" — isolating the
   value of topology diversity from the value of good components.
3. **Shared foundation.** Generalist = "substrate + one topology"; every KerSor
   solver = "substrate + its topology." The substrate is the common floor, owned by
   neither layer.

## Does a strong generalist undercut KerSor's thesis?

No. A generalist still has **one** topology, so it still loses on kernel families
favoring a different one. It **raises the floor** (every component strong); the
**topology ceiling stays per-family** — that ceiling is exactly KerSor's job. A
substrate-strengthened portfolio also makes the comparison **fairer**: with
components equalized, the remaining difference between solvers is purely topology,
which is what the complementarity experiment wants to measure.

## When to call which

| Goal | Call |
|---|---|
| One kernel, simplest path, no orchestration overhead | **Generalist** (`Workflow({name:'generalist-...'})`) |
| Cross-family coverage / the portfolio story | **KerSor** (`/kersor:optimize`) |
| Measure how much orchestration beats the best single solver | run both; generalist as the baseline |

## "集大成" exists at two levels

- **Component-level 集大成 = the generalist solver** (directly callable, internalizes
  the 7 composable axes).
- **Topology-level 集大成 = KerSor** (orchestrates across the 1 exclusive axis).

The generalist is also a member of, and the hardest baseline for, KerSor.
