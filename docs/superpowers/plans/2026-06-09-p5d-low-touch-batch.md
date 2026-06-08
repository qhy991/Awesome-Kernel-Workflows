# P5d — Low-Touch Batch Plan

Date: 2026-06-09
Branch: `dev/p5d-plan`
Predecessors: P5a (foundation), P5b (high-complexity), P5c (mid-complexity)
Sibling references on `main`: AdaExplore, KernelAgent, CUDALLM, KDA, StitchCUDA, Astra, STARK

---

## §1 Scope

P5d covers the remaining seven workflows in the awesome-kernel-workflows substrate that have been pre-screened as "low-touch" — meaning the banned-API audit returned **clean** (no host-side fast-paths, no oracle imports, no NumPy/PyTorch shortcuts in the kernel call chain), so byte-identity gates are immediately viable without a stripping pre-pass.

### In scope (7 workflows)

1. **KSearch** — multi-language search-tree kernel synthesis (triton default, cuda+python supported)
2. **ReGraphT** — graph-rewrite topology, currently cuda-only, widening to cuda+triton
3. **KernelFoundry** — sycl/cuda/triton matrix with `args.language` switch (already multi-language)
4. **KernelFoundryDx** — triton-only diagnostic variant of KernelFoundry
5. **KernelSkill** — clean-tier skill-based synthesizer (triton+cuda assumed, verify in A1)
6. **AKO4X** — 6 backends × 6 skills matrix governed by a `MODE` arg
7. **KernelBand** — clean-tier with a `SATURATION_THRESHOLD` φ-gate that must be driver-resolved

### Out of scope

- Any tier-2 (mid-complexity) or tier-3 (high-complexity) workflows already addressed by P5a/b/c.
- Cross-substrate driver work beyond the φ-gate plumbing required by KernelBand.
- New language drivers; we re-use the existing `cuda`, `triton`, `sycl`, `python` drivers registered by P3/P4.

### Gate

- **Byte-identity gate** for all seven workflows (manifest-declared `gate.kind: byte_identity` on every (workflow × language) cell where applicable).
- **Baseline:** 162 tests currently green on `main` (post-P5c).
- **Expected delta:** +~85 tests (7 workflows × ~12 assertions averaging across A1/A2/B1/B2/C1/C2 clusters; AKO4X contributes more due to its 6×6 matrix dimension assertions).
- **Stop condition:** all 7 manifests parse, all language cells return `viable=True` from the L0 driver dry-run, and the per-workflow agent registry matches sibling implementation parity within ±1 agent.

---

## §2 Pre-conditions

### Landed work

- **P5a** — substrate foundations: driver registration for `cuda`, `triton`, `sycl`, `python`; `manifest.yaml` schema v1.2 with `supported_languages` and `matrix_eligible` keys; envelope assertion helpers in `_meta/tools/lib/envelope.py`.
- **P5b** — high-complexity workflows shipped (AdaExplore-derived, KernelAgent-derived, CUDALLM-derived clusters) including the cross-workflow contention guard in `_meta/tools/lib/registry.py`.
- **P5c** — mid-complexity workflows shipped (KDA-derived, StitchCUDA-derived, Astra-derived, STARK-derived clusters) including the `vendor_collapse` evidence merge used by triton paths.

### Sibling references available on `main`

P5d agents may consult, but must not import from, the sibling implementations:

- `AdaExplore/` — search-tree precedent for KSearch
- `KernelAgent/` — agent-pool precedent for KernelFoundry / KernelFoundryDx
- `CUDALLM/` — CUDA-only L0 driver precedent for ReGraphT widening
- `KDA/` — skill-registry precedent for KernelSkill
- `StitchCUDA/` — backend-matrix precedent for AKO4X
- `Astra/` — φ-gate precedent for KernelBand `SATURATION_THRESHOLD`
- `STARK/` — manifest-creation precedent for the 6 workflows that lack `manifest.yaml`

### Required tools

- `_meta/tools/probe_banned_apis.py` — re-run as smoke check; all 7 must report 0 hits.
- `_meta/tools/manifest_lint.py` — must pass on every new/edited `manifest.yaml`.
- `_meta/tools/driver_dryrun.py` — must return `viable=True` for every declared `(workflow, language)` cell.

---

## §3 Per-workflow spike findings

All findings below are **pre-captured**; do NOT re-spike. Cluster agents start from these.

### §3.1 KSearch

- **Banned-API audit:** clean.
- **Manifest status:** missing — needs creation.
- **Languages:** `supported_languages = ['triton', 'cuda', 'python']`, `default_language = 'triton'`.
- **Topology:** search-tree (sibling: AdaExplore). Branch-and-bound over candidate kernel rewrites; each node is one compile-run-profile cycle.
- **Matrix eligibility:** `full` (every language is a first-class cell).
- **Estimated agent count:** 3 (search-coordinator, candidate-emitter, prune-evaluator).
- **Estimated fixture key count:** ~7 (3 language cells × 2 trivial kernels + 1 cross-language regression fixture).
- **Special hazards:** `python` cell uses the `python` reference driver — byte-identity is over the python-emitted host code, not over device bytes.

### §3.2 ReGraphT

- **Banned-API audit:** clean.
- **Manifest status:** present — needs edit only.
- **Languages:** currently `supported_languages = ['cuda']`. Widen to `['cuda', 'triton']`. `default_language` remains `cuda`.
- **Topology:** graph-rewrite (sibling: CUDALLM for cuda cell, generic triton driver for triton cell). Clean — no host fast-path.
- **Matrix eligibility:** `partial` (triton cell is opt-in; cuda is the canonical reference).
- **Estimated agent count:** 2 (graph-rewriter, gate-emitter).
- **Estimated fixture key count:** ~5 (2 cuda + 2 triton + 1 cross-cell parity check).
- **Special hazards:** the existing cuda manifest entry hard-codes `gate.kind: byte_identity` — preserve that and ensure the new triton entry uses the same gate kind for parity.

### §3.3 KernelFoundry

- **Banned-API audit:** clean.
- **Manifest status:** missing — needs creation.
- **Languages:** `supported_languages = ['sycl', 'cuda', 'triton']`, `default_language = 'cuda'`. The workflow already routes via `args.language` so the L0 driver dispatch is trivial.
- **Topology:** agent-pool (sibling: KernelAgent). Multiple specialist agents propose candidates; a foreman merges.
- **Matrix eligibility:** `full`.
- **Estimated agent count:** 4 (foreman + 3 language specialists).
- **Estimated fixture key count:** ~9 (3 language cells × 2 kernels + 3 args.language switch fixtures).
- **Special hazards:** `sycl` driver is GPU-deferred per P4; expect `viable=True` with `deferred=true` in the dry-run envelope.

### §3.4 KernelFoundryDx

- **Banned-API audit:** clean.
- **Manifest status:** missing — needs creation.
- **Languages:** `supported_languages = ['triton']`, `default_language = 'triton'`. Triton-only by design (the "Dx" variant strips the multi-language dispatcher).
- **Topology:** agent-pool, triton-restricted.
- **Matrix eligibility:** `partial` (one language only; matrix is a 1×N degenerate row).
- **Estimated agent count:** 2 (triton-specialist + diagnostic-emitter).
- **Estimated fixture key count:** ~5 (2 triton kernels + 3 diagnostic-output fixtures).
- **Special hazards:** the diagnostic envelope keys (`dx.compile_log`, `dx.ptx_summary`) are workflow-specific — register them in the per-workflow allowlist, not in the global envelope schema.

### §3.5 KernelSkill

- **Banned-API audit:** clean.
- **Manifest status:** missing — needs creation.
- **Languages:** assumed `supported_languages = ['triton', 'cuda']`, `default_language = 'triton'`. **A1 must verify** by reading the workflow's skill registry; if `python` skills exist, widen accordingly and revise the fixture count.
- **Topology:** skill-registry (sibling: KDA). Skills are reusable transformation primitives composed into pipelines.
- **Matrix eligibility:** `full`.
- **Estimated agent count:** 3 (skill-registrar, pipeline-composer, gate-runner).
- **Estimated fixture key count:** ~6 (2 cuda + 2 triton + 2 skill-registry round-trip checks).
- **Special hazards:** the assumed language list is unverified; flag A1 to confirm before A2 begins.

### §3.6 AKO4X

- **Banned-API audit:** clean.
- **Manifest status:** missing — needs creation.
- **Topology:** 6 backends × 6 skills matrix gated by a `MODE` arg (sibling: StitchCUDA backend-matrix precedent).
- **Languages:** the backend dimension maps to languages: `{cuda, triton, sycl, metal, python, cpu_ref}`. `default_language = 'cuda'`. `supported_languages = ['cuda', 'triton', 'sycl', 'metal', 'python', 'cpu_ref']`.
- **Matrix eligibility:** `full` for the (backend × skill) product. A `MODE` arg selects a subset of cells per invocation.
- **Estimated agent count:** 5 (mode-router + 4 backend-family coordinators).
- **Estimated fixture key count:** ~12 (one fixture per (backend × skill) sample cell + 6 MODE-routing fixtures).
- **Special hazards:**
  - **Intersectional guard required:** the `MODE` arg and the backend driver must agree. If `MODE=triton_only` selects the cuda backend, the L0 driver must reject with a typed error, not silently degrade.
  - **Cross-agent contention:** 5 agents × 6 backends produces the most fan-out in P5d; sequential dispatch only.

### §3.7 KernelBand

- **Banned-API audit:** clean.
- **Manifest status:** missing — needs creation.
- **Languages:** `supported_languages = ['triton', 'cuda']` (per master-plan note), `default_language = 'triton'`.
- **Topology:** clean-tier band-search (sibling: Astra for the φ-gate precedent).
- **Matrix eligibility:** `full`.
- **Estimated agent count:** 3 (band-explorer, threshold-evaluator, gate-emitter).
- **Estimated fixture key count:** ~7 (2 cuda + 2 triton + 3 saturation-threshold sweep fixtures).
- **Special hazards:** the `SATURATION_THRESHOLD` φ-gate must be **driver-resolved**, not workflow-resolved. The driver returns the achieved saturation in the L0 envelope; the gate then compares against the workflow-declared threshold. Wiring this requires touching `_meta/tools/lib/gates.py` to register a new `gate.kind: phi_threshold` variant.

---

## §4 Batch order recommendation

P5d is split into three sub-batches. The split is driven by **risk concentration**, not by alphabetical or topological order: workflows that introduce new gate plumbing or intersectional guards stand alone so a single batch failure does not block the rest.

### P5d.1 — Pure clean-tier widening (3 workflows, parallel-safe)

1. **KSearch** — fresh manifest, multi-language, no new gates.
2. **ReGraphT** — manifest edit only, widen cuda → cuda+triton.
3. **KernelFoundryDx** — fresh manifest, triton-only, smallest surface.

Rationale: all three reuse existing driver plumbing and existing gate kinds; the only novelty is manifest authoring. Suitable for back-to-back execution by a single cluster operator.

### P5d.2 — Multi-language agent-pool (2 workflows, parallel-safe)

4. **KernelFoundry** — sycl+cuda+triton matrix, agent-pool topology.
5. **KernelSkill** — skill-registry topology, languages verified in A1.

Rationale: both introduce per-workflow agent registries of similar shape; KernelSkill's language list is unverified, so it goes second to absorb any spike adjustments without holding up KernelFoundry.

### P5d.3.a — AKO4X (standalone)

6. **AKO4X** — 6 backends × 6 skills with `MODE` arg.

Rationale: the intersectional `MODE`-vs-backend guard is the single highest-risk piece of new logic in P5d. Standalone execution means a failed guard implementation does not contaminate the other sub-batches' commit history, and the larger fixture count (~12) and agent count (5) consume disproportionate review bandwidth.

### P5d.3.b — KernelBand (standalone)

7. **KernelBand** — `SATURATION_THRESHOLD` φ-gate.

Rationale: registering a new `gate.kind: phi_threshold` variant in `_meta/tools/lib/gates.py` is a substrate-level change. Standalone execution means the substrate diff is reviewed in isolation, and any roll-back is one revert away from clean.

### Execution order

```
P5d.1 (3 workflows, ~24 commits)
   |
   v
P5d.2 (2 workflows, ~16 commits)
   |
   v
P5d.3.a AKO4X (~12 commits)
   |
   v
P5d.3.b KernelBand (~8 commits)
```

Total expected commit count: ~60 (see §8 for the cluster-derived estimate).

---

## §5 Cluster decomposition

Every P5d workflow follows the same six-cluster shape (no A0 cluster — all workflows are pre-screened clean, so the strip-fast-path cluster is skipped). The shape mirrors P5c §6.

### A1 — Manifest authoring / edit

- Author or edit `manifest.yaml`.
- Declare `supported_languages`, `default_language`, `matrix_eligible`.
- Declare every `gate.kind` cell.
- Pass `_meta/tools/manifest_lint.py`.

### A2 — Driver dry-run wiring

- For every declared `(workflow, language)` cell, invoke the registered L0 driver with a no-op kernel input.
- Assert `viable=True` (or `viable=True, deferred=true` for GPU-deferred drivers).
- Persist the dry-run envelope under `_substrate/dryruns/<workflow>/<language>.json`.

### B1 — Agent registry

- Register the per-workflow agents (counts per §3) in `_meta/tools/lib/registry.py`.
- Each agent gets a one-line docstring and a typed input/output schema.
- Sequential dispatch only; document cross-agent dependencies in the registry comment block.

### B2 — Fixture authoring

- Author the per-cell fixtures (counts per §3).
- Each fixture is a triple: `(input_kernel_src, expected_envelope_keys, byte_identity_anchor)`.
- Place under `_substrate/fixtures/<workflow>/`.

### C1 — Gate assertions

- For every fixture, run the declared gate and assert pass.
- For byte-identity gates, the anchor is the SHA-256 of the canonicalized device bytes (or, for `python` cells, of the emitted host source).
- For KernelBand's φ-gate, assert the achieved saturation crosses the workflow-declared threshold.

### C2 — Integration / regression test

- One end-to-end test per workflow that exercises the default language cell from manifest-parse through gate-pass.
- Add to the project's pytest collection; expected to land in the +~85 test delta.

---

## §6 Per-workflow task tables

Each table lists the file paths touched per cluster, the count of new assertion labels added, and the count of fixture keys authored. Counts are estimates; A1 may revise within ±1.

### §6.1 KSearch

| Cluster | Files | Labels | Fixtures |
|---|---|---|---|
| A1 | `KSearch/manifest.yaml` (new) | 6 | — |
| A2 | `_substrate/dryruns/ksearch/{triton,cuda,python}.json` | 3 | — |
| B1 | `_meta/tools/lib/registry.py` (edit) | 3 | — |
| B2 | `_substrate/fixtures/ksearch/*` | — | 7 |
| C1 | `tests/ksearch/test_gates.py` | 7 | — |
| C2 | `tests/ksearch/test_e2e.py` | 1 | — |

### §6.2 ReGraphT

| Cluster | Files | Labels | Fixtures |
|---|---|---|---|
| A1 | `ReGraphT/manifest.yaml` (edit) | 4 | — |
| A2 | `_substrate/dryruns/regrapht/{cuda,triton}.json` | 2 | — |
| B1 | `_meta/tools/lib/registry.py` (edit) | 2 | — |
| B2 | `_substrate/fixtures/regrapht/*` | — | 5 |
| C1 | `tests/regrapht/test_gates.py` | 5 | — |
| C2 | `tests/regrapht/test_e2e.py` | 1 | — |

### §6.3 KernelFoundry

| Cluster | Files | Labels | Fixtures |
|---|---|---|---|
| A1 | `KernelFoundry/manifest.yaml` (new) | 8 | — |
| A2 | `_substrate/dryruns/kernelfoundry/{sycl,cuda,triton}.json` | 3 | — |
| B1 | `_meta/tools/lib/registry.py` (edit) | 4 | — |
| B2 | `_substrate/fixtures/kernelfoundry/*` | — | 9 |
| C1 | `tests/kernelfoundry/test_gates.py` | 9 | — |
| C2 | `tests/kernelfoundry/test_e2e.py` | 1 | — |

### §6.4 KernelFoundryDx

| Cluster | Files | Labels | Fixtures |
|---|---|---|---|
| A1 | `KernelFoundryDx/manifest.yaml` (new) | 4 | — |
| A2 | `_substrate/dryruns/kernelfoundrydx/triton.json` | 1 | — |
| B1 | `_meta/tools/lib/registry.py` (edit) | 2 | — |
| B2 | `_substrate/fixtures/kernelfoundrydx/*` | — | 5 |
| C1 | `tests/kernelfoundrydx/test_gates.py` | 5 | — |
| C2 | `tests/kernelfoundrydx/test_e2e.py` | 1 | — |

### §6.5 KernelSkill

| Cluster | Files | Labels | Fixtures |
|---|---|---|---|
| A1 | `KernelSkill/manifest.yaml` (new) | 6 | — |
| A2 | `_substrate/dryruns/kernelskill/{triton,cuda}.json` | 2 | — |
| B1 | `_meta/tools/lib/registry.py` (edit) | 3 | — |
| B2 | `_substrate/fixtures/kernelskill/*` | — | 6 |
| C1 | `tests/kernelskill/test_gates.py` | 6 | — |
| C2 | `tests/kernelskill/test_e2e.py` | 1 | — |

### §6.6 AKO4X

| Cluster | Files | Labels | Fixtures |
|---|---|---|---|
| A1 | `AKO4X/manifest.yaml` (new) | 12 | — |
| A2 | `_substrate/dryruns/ako4x/{cuda,triton,sycl,metal,python,cpu_ref}.json` | 6 | — |
| B1 | `_meta/tools/lib/registry.py` (edit) + `_meta/tools/lib/mode_guard.py` (new) | 5 | — |
| B2 | `_substrate/fixtures/ako4x/*` | — | 12 |
| C1 | `tests/ako4x/test_gates.py` + `tests/ako4x/test_mode_guard.py` | 14 | — |
| C2 | `tests/ako4x/test_e2e.py` | 1 | — |

### §6.7 KernelBand

| Cluster | Files | Labels | Fixtures |
|---|---|---|---|
| A1 | `KernelBand/manifest.yaml` (new) | 5 | — |
| A2 | `_substrate/dryruns/kernelband/{triton,cuda}.json` | 2 | — |
| B1 | `_meta/tools/lib/registry.py` (edit) + `_meta/tools/lib/gates.py` (edit: register `phi_threshold`) | 3 | — |
| B2 | `_substrate/fixtures/kernelband/*` | — | 7 |
| C1 | `tests/kernelband/test_gates.py` + `tests/kernelband/test_phi_threshold.py` | 9 | — |
| C2 | `tests/kernelband/test_e2e.py` | 1 | — |

---

## §7 Risk matrix

| Risk | Workflow(s) | Severity | Mitigation |
|---|---|---|---|
| Cross-agent contention on shared registry | All 7 (esp. AKO4X with 5 agents) | Medium | Sequential dispatch — only one cluster operator edits `_meta/tools/lib/registry.py` at a time. Each B1 cluster commits independently. |
| Intersectional MODE-vs-backend mismatch | AKO4X | High | New `_meta/tools/lib/mode_guard.py` module; typed error on mismatch (not silent degrade); dedicated `test_mode_guard.py` suite in C1. |
| φ-gate driver wiring breaks existing gates | KernelBand | High | Register `phi_threshold` as a new `gate.kind` variant — do not modify existing `byte_identity` path. Add a regression assertion that the baseline 162 tests still pass after the gates.py edit. |
| Unverified language list | KernelSkill | Low | A1 task explicitly verifies the skill registry; if assumption wrong, A1 revises the manifest and B2 fixture count before A2 starts. |
| `sycl` GPU-deferred driver returns unexpected envelope shape | KernelFoundry | Low | A2 asserts both `viable=True` and `deferred=true`; reuse the P4-defined deferred envelope schema. |
| `python` cell byte-identity ambiguity (device vs host bytes) | KSearch | Low | Document in the manifest that the python cell's byte-identity anchor is the emitted host source SHA-256, not device bytes. |
| Manifest authoring drift across 6 new manifests | All except ReGraphT | Medium | Use STARK's manifest as a copy-paste template; lint every new file with `_meta/tools/manifest_lint.py` in A1. |

### Resolved by sequential dispatch

The cross-agent contention risk is the most pervasive but also the cheapest to mitigate: we never run two cluster operators concurrently on `_meta/tools/lib/registry.py`. Per §4 the sub-batch boundaries align with this constraint.

### Standalone justifications (recap from §4)

- **AKO4X stands alone** because the intersectional guard is a substrate-level addition (`mode_guard.py`) that warrants isolated review.
- **KernelBand stands alone** because registering a new `gate.kind` variant in `gates.py` is a substrate-level addition that warrants isolated review.

---

## §8 Estimated effort

### Per-workflow commit count

Each workflow yields one commit per cluster (A1, A2, B1, B2, C1, C2) plus, for AKO4X and KernelBand, one additional substrate-level commit for the new module/gate-variant.

| Workflow | Cluster commits | Extra | Total |
|---|---|---|---|
| KSearch | 6 | 0 | 6 |
| ReGraphT | 6 | 0 | 6 |
| KernelFoundry | 6 | 0 | 6 |
| KernelFoundryDx | 6 | 0 | 6 |
| KernelSkill | 6 | 0 | 6 |
| AKO4X | 6 | 2 (mode_guard.py + test_mode_guard.py) | 8 |
| KernelBand | 6 | 2 (gates.py edit + test_phi_threshold.py) | 8 |

**Grand total: 46 commits** for the per-workflow work, plus ~10 cross-cutting commits (sub-batch fixups, registry consolidations, final regression-baseline updates), yielding **~56 commits** end-to-end, matching the master-plan target of "~7 workflows × 8 tasks ≈ 56 commits".

### Test delta

- A2 dryrun assertions: 19 across all 7 workflows.
- C1 gate assertions: 55 across all 7 workflows.
- C2 e2e assertions: 7 (one per workflow).
- **Expected new tests:** ~81; rounding for incidental helper-test additions yields the +~85 target.

### Wall-clock estimate (sequential dispatch)

Assuming each cluster commit takes one cluster-operator session of bounded budget:

- P5d.1: 3 workflows × 6 clusters = 18 sessions.
- P5d.2: 2 workflows × 6 clusters = 12 sessions.
- P5d.3.a: 1 workflow × 8 clusters = 8 sessions.
- P5d.3.b: 1 workflow × 8 clusters = 8 sessions.
- **Total: 46 cluster sessions**, plus review/fixup overhead.

### Stop conditions

P5d is considered done when:

1. All 7 `manifest.yaml` files parse and lint clean.
2. All 19 dry-run envelopes exist under `_substrate/dryruns/` and report `viable=True`.
3. All ~85 new tests pass alongside the baseline 162, yielding ~247 green.
4. `mode_guard.py` rejects every constructed mismatched (MODE, backend) pair in its test suite.
5. `gate.kind: phi_threshold` is registered and exercised by KernelBand's `test_phi_threshold.py`.
6. The banned-API probe re-run still reports 0 hits across all 7 workflows.
