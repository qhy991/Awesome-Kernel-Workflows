# KerSor Solver SDK — `_substrate/`

The shared substrate: deterministic, unit-tested components every solver can
reuse, plus the design docs. "集大成" by composition (substrate × topologies),
not a monolith.

## Design docs

| Doc | What |
|---|---|
| [`SOLVER-AUDIT.md`](./SOLVER-AUDIT.md) | 26-solver best-of-breed decomposition (7 composable axes + 1 exclusive topology axis) |
| [`SOLVER-SDK.md`](./SOLVER-SDK.md) | The substrate contract: Layers A–F, conformance levels L0–L3 |
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | substrate ↔ generalist solver ↔ KerSor relationship + invocation |
| [`EXPERIMENT-SOLVER-SET.md`](./EXPERIMENT-SOLVER-SET.md) | topology-deduplicated 5-solver experiment pool |
| [`IMPROVEMENTS.md`](./IMPROVEMENTS.md) | prioritized pattern improvements (token budget, model routing, worktree isolation, adversarial verify, …) |

## Deterministic scripts (stdlib Python; run by agent Bash steps)

These are the fidelity anchor — real, deterministic, unit-tested. The Claude Code
`Workflow` runtime has no fs/import, so solvers invoke these from agent steps
(same pattern as KerSor's `select-workflow.sh` / `render-handoff.sh`).

| Script | Layer | One-line usage |
|---|---|---|
| `evidence_schema.py` | A | `evidence_schema.py validate result.json` — canonical attempt evidence (KerSor-native) |
| `anti_cheat.py` | B | `anti_cheat.py --source k.cu --metrics m.json` — validity gate + robust reward |
| `diagnose.py` | C | `diagnose.py --metrics m.json` — metrics → `bottleneck_class` |
| `memory_store.py` | D | `memory_store.py --db DB retrieve|update|add-deadend` — persistent state-keyed memory |
| `method_gate.py` | E | `method_gate.py --class memory_bound` — `bottleneck_class` → `allowed_methods` |
| `verify_insight.py` | B+ | `verify_insight.py --insight i.json --refuted 1` — deterministic confidence downgrade (pairs with an LLM refuter) |

Layer F (cost-control: ceiling / stagnation / early-term) is inline JS in the solver.

The first solver built on this substrate: [`../Generalist/`](../Generalist/).

## Status

`dev/solver-substrate` branch. Scripts unit-tested (see commit). Not yet merged to
main — design + reference implementation, not production-wired into every solver.
