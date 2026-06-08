# P5 — Clean-Workflow Migration: Master Plan

> **Status:** master/navigation plan. Decomposes Appendix A row "P5+" of the
> Backend Driver Axis spec into named, dependency-ordered sub-plans
> (P5a..P5f). Each sub-plan's detailed implementation plan is to be written
> separately when that sub-plan is "up" — this document is scoping +
> ordering + exit criteria, not file-line implementation detail.
>
> **Spec of record:** `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md`
> (focus §9.1 Phase 2 ordering, §9.2 generator/template/schema changes,
> §9.3 CI/conformance/matrix-smoke, §9.4 docs, Appendix A "P5+").
>
> **Pattern of record:** `docs/superpowers/plans/2026-06-08-accelopt-driver-pilot.md`
> — the executed P4 pilot. P5b/c/d generalize its retrofit checklist; P5e+P5f
> close out CI + docs.
>
> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:writing-plans`
> for each sub-plan; `superpowers:subagent-driven-development` to execute
> them. Each sub-plan ends in its own commit cluster.

---

## 1. Goal

Migrate all 13 clean-tier workflows (§7.2: AdaExplore, KernelAgent, KDA,
CUDALLM, Astra, StitchCUDA, STARK, KSearch, ReGraphT, KernelFoundry,
KernelFoundryDx, KernelSkill, AKO4X, KernelBand — and Generalist as the
substrate reference, migrated last) onto the backend-driver contract proved
out by P4 on AccelOpt. Land the §9.2 generator/template/schema changes that
make new clean workflows emit driver-shaped code by default. Land the §9.3
CI tiers (substrate diff-guard, driver conformance L0–L3, matrix smoke).
Land the §9.4 documentation pass. No GPU dependency in any exit criterion;
GPU verification is the opt-in deferred tier (carried over from P4).

> **Workflow count:** §7.2 row 1 (13) + KernelBand (`clean*`) +
> Generalist (`clean`) = **15 clean / matrix-eligible total**, same as
> §9.1 Phase 2's enumeration. This plan adopts that ordering verbatim.

## 2. Non-goals (out of scope for P5)

- Vendor_locked workflows (AccelOpt is done in P4; CUDAAgent, cuPilot, KEET,
  KernelBlaster, GPUForecasters, TritorX, Xe-Forge) — spec §9.1 Phase 3.
- Method_intrinsic workflows (ArchAgent, CutlassGEMM, FACT, ARGUS) —
  spec §9.1 Phase 4.
- Metal driver + Apple calibration + MPS fallback patterns — spec P6.
- Deduping `_tools/`↔`_meta/tools/` and `_templates/`↔`_meta/templates/`
  (pre-existing duplication; flagged but not in scope — spec §2.3).
- Real-hardware (GPU) verification — opt-in self-hosted-runner tier; the
  `DEFERRED-GPU-VERIFICATION.md` checklist style from P4 carries forward.

## 3. Hard preconditions (must be true before P5a starts)

1. P1 (driver contract scaffolding) merged: `_substrate/backends/_schema/`
   or its Python equivalent (`validate_backend.py` per the SDK doc's
   "Deviations" section), `REGISTRY.md`, `BACKEND-DRIVER-SDK.md` present.
2. P2 (the three §5.3 substrate edits) merged: `diagnose.py` vendor
   threshold profile + null rule; `anti_cheat.py --vendor-patterns-file`
   covering both `[fallback]` and `[skip]`. Substrate diff-guard green —
   only those three hunks differ from baseline.
3. P3 (reference cuda + triton drivers) merged: both drivers L0 conformant;
   `to_evidence.py` shared nvidia mapping verified on macOS via fake-tool
   PATH stubs. (Per `_substrate/backends/REGISTRY.md` note, scripts are
   GPU-untested; that's accepted.)
4. P4 (AccelOpt driver pilot) merged: `_meta/tools/print-workflow-prompts.js`
   harness + `schema-stub.js` + `run-workflow.js` lib present, working,
   tested (`_meta/tools/test/`); AccelOpt byte-identity gate green; triton
   dry-run green; deferred-GPU checklist exists at
   `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md`.

If any precondition is unmet, P5 cannot start — escalate to the spec-owner
to land the missing P1–P4 deliverable first.

## 4. Dependency graph

```
   P5a (schema + generator + template + validate-backend)
    │
    ├─► P5b (AdaExplore + KernelAgent)
    │       │
    │       └─► P5c (KDA, CUDALLM, Astra, StitchCUDA, STARK)
    │               │
    │               └─► P5d (KSearch, ReGraphT, KernelFoundry,
    │                        KernelFoundryDx, KernelSkill, AKO4X,
    │                        KernelBand)
    │                        │
    │                        ▼
    └─────────────────► P5e (Generalist + matrix-smoke CI;
                              needs ≥1 matrix_eligible workflow —
                              guaranteed by P5b)
                              │
                              ▼
                             P5f (§9.4 docs pass)
```

**Edges in plain English:**
- `P5a → all` — manifest `backend:` block + validate-backend contract is
  the schema everything conforms to.
- `P5b → P5c` — AdaExplore + KernelAgent are the proving ground; their
  spikes (canonical `kernel_path`, determinism) inform P5c.
- `P5c → P5d` — sequencing for risk management only; bug discoveries in
  P5c adjust the P5d checklist. Not a hard contract edge.
- `P5b → P5e` (matrix smoke) — needs ≥1 matrix_eligible workflow
  retrofitted; AdaExplore guarantees that.
- `Generalist last` — substrate reference (§9.1 Phase 2); earlier
  batches still have a clean reference to compare against until P5e.
- `P5e → P5f` — docs reference the new CI tiers as live.

## 5. Sub-plan catalogue

### P5a — Schema + generator/template prep + validate-backend.js

**Scope (precise, bounded):** Land the §9.2 schema and generator changes so
new clean workflows emit driver-shaped code by default and the manifest
declares its backend posture. Add the deterministic L0 backend validator
called out in §9.3. **No workflow body edits in this sub-plan** — this is
contract-only.

**Spec sections driving it:** §9.2 (entire); §9.3 "Driver conformance L0–L3"
+ "Substrate diff-guard"; §4.4 (manifest fields the schema mirrors); §4.9
(L0–L3 contract).

**Files touched:**
- `_meta/manifests/schema.yaml` — add top-level `backend:` block
  (`supported[]`, `default` (may be `null` → force explicit `--backend`),
  `matrix_eligible: true|partial|false`, `portability:
  clean|vendor_locked|method_intrinsic`, `intrinsic_to` (required when
  not clean)). Extend `args.optional[]` examples: neutral `backend`,
  `backend_dir`, `driver_shell_prefix`, `substrate_dir`, neutral
  `profile_command`; keep `ncu_command` as a documented deprecated alias
  (deliberate contract change, not a "tightening" — it is first-class
  today per `generate-workflow.js:282`). Bump header to v1.1.
- `_meta/tools/generate-workflow.js` — edit two **agent-prompt** regions
  (not a substitution table):
  - model-args prompt ~L263–282: teach the agent to extract
    `native_backend` + `portability_class` from manifest, emit
    `backend`/`backend_dir` as standard optional args, mark
    `ncu_command` deprecated.
  - generate prompt ~L492–524 (the "Generation Rules"; rule #2 today
    emits `WORKFLOW_SUITABILITY` with `supported_languages`): teach the
    agent to emit the §6.4 split (`method_supported_backends` /
    `default_backend` / `requires_capability`), the §6.1 path-helper
    block, and the §6.2 Setup `load-driver` agent gated on `USE_DRIVER`.
    Tighten rule #10 to forbid naming `nvcc`/`ncu`/`@triton.jit`/
    `__global__`/`PYBIND11_MODULE` in workflow body prompts.
- `_meta/tools/validate-workflow.js` — add **prompt checklist items**
  only (LLM-driven; per spec §9.2 NOT a hard gate). Warn on
  `supported_languages:` and on CUDA literal tokens in any prompt of a
  `portability: clean` workflow.
- `_meta/tools/validate-backend.js` (**new**) OR
  `_substrate/backends/validate_backend.py` — per SDK doc's "Deviations"
  section, Python won; settle in P5a's implementation plan, do not ship
  both. Deterministic L0: manifest validates, `backend_id ==
  basename(dir)`, `idioms.json` references only real `method_gate.TABLE`
  names, `capabilities.metrics ⊆` 4 canonical keys,
  `bottleneck_classes ⊆` 4 meaningful ∪ `{unknown}`, substrate scripts
  byte-identical to baseline except the three §5.3 hunks. Separate from
  LLM-driven `validate-workflow.js`.
- `_templates/{iterative-loop,search-based,single-pass,tree-exploration}.js`
  AND `_meta/templates/` mirrors — update `{{TOKEN}}`/`[BLOCK]` guidance
  text + input-policy comment (currently `_templates/iterative-loop.js:62-66`)
  to teach "body never names a vendor profiler or vendor metric; tokens
  come from `idioms.json`". Edit `_meta/templates/` authoritatively per
  §2.3 convention; flag duplication, do not deepen.
- `_meta/tools/test/validate-backend.test.js` (**new**, or Python pytest)
  — L0 against P3 cuda + triton drivers as positives + ≥3 crafted negatives.
- `_meta/tools/test/generator-prompt-schema.test.js` (**new**) — the
  regression test: pin generate-workflow agent's structured-output
  `schema:` (the object after L520) — backend-axis prompt edits must
  not alter the keys downstream consumes. Use P4 `capturePrompts` in
  prompt-introspection mode (the generator is itself a workflow .js).

**Depends on:** P1, P2, P3, P4. Hard precondition.

**Exit criteria (testable, no GPU):**
1. `_meta/manifests/schema.yaml` v1.1 in place; every existing manifest
   under `_meta/manifests/` (or `_manifests/` if that is where they live —
   verify in implementation plan) continues to parse against the v1.1
   schema with the `backend:` block defaulted/absent for backward
   compatibility.
2. `validate-backend(.js|.py)` returns L0 PASS on the cuda + triton P3
   drivers; returns the right L0 FAIL on each of ≥3 crafted negative
   fixtures.
3. Generator-prompt-schema regression test green: clean manifest input
   produces a generate-workflow schema deep-equal to today's.
4. Substrate diff-guard from P2 still green (P5a touches no `_substrate/`
   .py beyond `validate_backend.py` if Python wins).
5. Template `[BLOCK]` guidance updates do not break any existing workflow
   (templates are read by humans + the generator agent; structural
   `{{TOKEN}}` markers preserved).
6. **No workflow `.js` file changed.** Iron rule for this sub-plan.

**Estimated commits:** 5–7
1. schema.yaml v1.1 + header
2. validator (new file + tests)
3. generator model-args prompt edit + regression test
4. generator generate-prompt edit
5. template guidance edits (one commit per template, or one batched)
6. validate-workflow.js prompt-checklist additions
7. final integration: full `node --test _meta/tools/test/*.test.js` green

**Open questions for P5a's implementation plan (spike first):**
- **Manifest tree location.** Spec §9.2 names `_meta/manifests/schema.yaml`
  but a top-level `_manifests/` also exists. Confirm canonical tree
  before editing; do not deepen any `_manifests/`↔`_meta/manifests/`
  divergence (mirrors §2.3 convention).
- **Python vs Node validator.** SDK doc says Python in
  `_substrate/backends/validate_backend.py`; spec §9.2/§9.3 name
  `_meta/tools/validate-backend.js`. Resolve once; document in SDK
  "Deviations"; do not ship both.
- **`ncu_command` deprecation window length.** Spec §10 lists as open;
  pick a concrete number in P5a (1 or 2 minor versions).

---

### P5b — AdaExplore + KernelAgent retrofit (Phase 2 first wave)

**Scope:** Generalize the AccelOpt-pilot pattern (P4) into a reusable
checklist and apply it to AdaExplore + KernelAgent. These are §9.1's
"proven generic topologies — MCTS and routing+parallel-seeds —
retrofittable by swapping only the evaluation harness." Each workflow is
its own commit cluster following the §6 checklist. AdaExplore is exercised
in P5e's matrix smoke as the first matrix_eligible retrofitted workflow.

**Spec sections driving it:** §9.1 Phase 2 ordering; §6.1/§6.2/§6.3/§6.4
(the driver-consumption pattern); §8.2 (the seam-swap template from
AccelOpt that generalizes); §7.2 row 1 (clean tier; `'any'`
method-supported-backends).

**Generalized AccelOpt-pattern checklist** (apply per workflow; per-workflow
detail lives in each implementation plan):
1. **Golden capture** — `_meta/tools/print-workflow-prompts.js` with fixed
   args + deterministic `agentReturns` map that unlocks the full loop.
   Commit pre-retrofit goldens + `<workflow>-GOLDEN-BASELINE-SHA.txt` per
   P4 Stage A Task 4.
2. **Harness fixtures** — args.json + agent-returns.json per mode
   (optimize vs generate, when both apply).
3. **`USE_DRIVER` gate** — §6.1 path helpers + §6.4 guard split; wrap
   each backend-laden seam in `USE_DRIVER ? <driver> : <legacy>`. Legacy
   literal extracted to column-0 named const before gating (P4 col-0
   rule — load-bearing).
4. **Driver-path guards** — Setup `load-driver` agent (§6.2) only when
   `USE_DRIVER`; `JSON_PASSTHROUGH` schema; `diagnose.py` + Layer-A
   evidence assembly only on the driver path.
5. **Triton dry-run** — non-CUDA driver-path render verified; assert no
   CUDA tokens (`nvcc`/`ncu`/`__global__`/`PYBIND11_MODULE`/
   `cuda_runtime.h`/`NCU Profile Results`/`cuda` fence) leak.
6. **Byte-identity gate** — Stage-C-style test diffing no-`backend_dir`
   capture against pre-retrofit golden; green is the merge gate.
7. **Deferred-GPU checklist update** — append workflow to
   `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md`.

**Backend posture per workflow:**

| Workflow | Triton-capable? | Notes |
|---|---|---|
| AdaExplore | Yes (clean) | CUDA `.cu` today (MCTS). Spike #1: confirm `args.kernel_path` is the only source-input arg. |
| KernelAgent | Triton-flavored today (per filename `kernelagent-triton-synthesis.js`); clean per §7.2 | Verify: is the *method* triton-only, or is the topology language-agnostic and only the prompts triton-flavored? If method-intrinsic, reclassify. |

**Files touched:**
- `AdaExplore/adaexplore-kernel-optimization.js` — Stage B retrofit
  (additive, `backend_dir`-gated).
- `KernelAgent/kernelagent-triton-synthesis.js` — Stage B retrofit.
- `_meta/tools/fixtures/adaexplore-*.json` (args / agent-returns /
  golden / SHA file).
- `_meta/tools/fixtures/kernelagent-*.json` (same set).
- `_meta/tools/test/adaexplore-cuda-byte-identity.test.js`,
  `adaexplore-guard.test.js`, `adaexplore-triton-dryrun.test.js`.
- `_meta/tools/test/kernelagent-*.test.js` (same trio).
- `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md` (append).

**Depends on:** P5a (schema + validator contract MUST be in place; the
retrofit emits manifests that conform to v1.1).

**Exit criteria (testable, no GPU):**
1. Both workflows render byte-identical prompts on the legacy path (no
   `backend_dir`) versus pre-retrofit goldens; tests green under
   `node --test _meta/tools/test/*.test.js`.
2. Triton dry-run green for both: rendering with the P3 triton driver
   produces prompts with zero CUDA tokens across the full unlocked
   prompt set.
3. Guard tests green: §6.4 resolution + conflict-throw behavior matches
   spec.
4. Both workflows' manifests declare `backend:` block per P5a schema:
   `portability: clean`, `matrix_eligible: true`, `supported: [cuda, triton]`
   (or `[any]` per resolution), `default: cuda`.
5. Substrate diff-guard still green (no `.py` changes in this sub-plan).

**Estimated commits:** 14–20 (per workflow: 7-stage cluster mirroring P4
Stages A/B/C — Stage-A harness fixture + golden + SHA; Stage-B retrofit in
3–5 task commits; Stage-C test trio — so 7×2 = 14 minimum).

**Open questions surfaced for P5b's implementation plan:**
- **Canonical `args.kernel_path` discipline.** AccelOpt uses
  `args.kernel_path` as the single source-input arg. Do AdaExplore and
  KernelAgent use the same canonical name, or do they have method-specific
  variants (e.g. `args.source_path`, `args.seed_kernel`)? The
  driver-dispatch `driverSh('build.sh', --source ${...})` assumes a
  single resolvable string. **Spike before writing the Stage-B edits** —
  read the first ~150 lines of each workflow .js, grep for argument
  destructuring, list every source-input arg.
- **MCTS vs beam state machinery.** P4's golden capture relies on
  AccelOpt's deterministic `sampleWithoutReplacement` index-shuffle. Verify
  AdaExplore's MCTS rollout is similarly deterministic (no `Math.random`,
  no `Date.now`) — if not, the byte-identity gate cannot be the merge
  criterion; introduce a seeded-RNG arg first or fall back to a
  per-label-shape gate.

---

### P5c — Mid-complexity batch (KDA, CUDALLM, Astra, StitchCUDA, STARK)

**Scope:** Apply the (now-proven-on-2-workflows) checklist from P5b to the
mid-complexity clean tier. These workflows are larger but share the
iterative-loop topology; the P5b spike outcomes (canonical kernel_path,
determinism verification) reduce per-workflow uncertainty. May ship as 2–3
sub-units (e.g. P5c.1 = KDA + CUDALLM; P5c.2 = Astra + StitchCUDA;
P5c.3 = STARK) to keep PR sizes review-friendly.

**Spec sections driving it:** §9.1 Phase 2 (workflow ordering); §7.2 row 1.

**Backend posture per workflow:**

| Workflow | Notes |
|---|---|
| KDA | CUDA-flavored prompts today; clean per §7.2. Has its own `manifest.yaml` already (`KernelAgent/manifest.yaml` exists; check KDA). |
| CUDALLM | CUDA-named per workflow name; clean tier — verify the *method* (likely an LLM-judge over kernels) is backend-agnostic. |
| Astra | Check directory contents in spike. |
| StitchCUDA | Name says CUDA; clean tier means the *stitch* topology generalizes. |
| STARK | Check directory contents in spike. |

**Files touched:** the 5 workflow .js files, 5 sets of fixtures + tests
in `_meta/tools/{fixtures,test}/`, append to deferred-GPU checklist.

**Depends on:** P5b (the pattern is hardened; canonical kernel_path
question is resolved; determinism question is resolved). Soft dependency
— could in principle run after P5a alone, but mistakes are cheaper to
fix after P5b.

**Exit criteria (testable, no GPU):** byte-identity gate green for all 5;
triton dry-run green for all 5; guard tests green; manifests conform to
v1.1; substrate diff-guard still green.

**Estimated commits:** 35–50 (5 workflows × ~7-stage cluster).

**Open questions for P5c spike:**
- Per-workflow, is the topology iterative-loop, search-based, or something
  bespoke? Spec §9.1 batches them as "decreasing complexity" but does not
  pin topology. The retrofit checklist is iterative-loop-shaped; bespoke
  topologies may need adapted Stage-A fixtures.
- Do any of these workflows already declare `WORKFLOW_SUITABILITY` with
  `supported_languages` other than `['cuda']`? If so, the §6.4 guard flip
  must preserve their exact prior throw behavior.

---

### P5d — Lower-complexity batch (KSearch, ReGraphT, KernelFoundry, KernelFoundryDx, KernelSkill, AKO4X, KernelBand)

**Scope:** Apply the checklist to the remaining clean workflows except
Generalist. 7 workflows. KernelBand is `clean*` (§7.2 footnote: hardware
masking φ-gate leans on NVIDIA utilization; threshold becomes
driver-resolved). May split as P5d.1 = KSearch + ReGraphT + KernelFoundry +
KernelFoundryDx + KernelSkill, P5d.2 = AKO4X + KernelBand.

**Spec sections driving it:** §9.1 Phase 2 (ordering); §7.2 row 1 + the
`clean*` KernelBand footnote ("threshold becomes driver-resolved, effort M").

**Backend posture per workflow:**

| Workflow | Notes |
|---|---|
| KSearch | **Verify Triton-intrinsic vs clean.** §7.2 lists as clean, but implementation may be triton-flavored such that method IS the backend. If triton-intrinsic, reclassify to `vendor_locked ['triton']` + skip matrix smoke. |
| ReGraphT | "ReGraph"+"T" — possibly Triton-flavored; verify in spike. |
| KernelFoundry, KernelFoundryDx | Likely paired; check if Dx is a strict superset / experimental fork. May share fixtures. |
| KernelSkill | Check topology in spike. |
| AKO4X | Largest .js in batch (46.8K); extra review attention but same pattern. |
| KernelBand | `clean*`. φ-gate threshold becomes driver-resolved (from `idioms.json:read_metric_guide` or new manifest field). Substantive extra task on top of the checklist. |

**Files touched:** 7 workflow .js files; 7 sets of fixtures + tests;
KernelBand additionally edits a φ-gate threshold lookup; deferred-GPU
checklist append.

**Depends on:** P5c (cumulative confidence in the pattern; if P5c finds a
common bug, fix it once before P5d).

**Exit criteria (testable, no GPU):** byte-identity, triton dry-run, guard,
manifest conformance, substrate diff-guard — all green for 7 workflows.
KernelBand additionally: φ-gate threshold resolves from driver, not
hardcoded literal; legacy path unchanged.

**Estimated commits:** 50–65 (7 workflows × ~7-stage cluster; KernelBand
+1–2 extra commits for the φ-gate driver lookup).

**Open questions for P5d spike:**
- **KSearch tier reclassification** (user-surfaced). Spike first; if
  triton-intrinsic, file an amendment to spec §7.2 and reclassify before
  writing the implementation plan.
- KernelBand's φ-gate threshold: which manifest field carries the value
  (a new `capabilities.utilization_threshold`?), or does it live in
  `idioms.json`? Settle in implementation plan.

---

### P5e — Generalist retrofit + matrix-smoke-test CI

**Scope:** Two strands sharing one sub-plan because they depend on each other:
1. **Generalist retrofit** — substrate reference (§9.1 Phase 2:
   "Generalist last"); apply the same checklist.
2. **Matrix smoke + CI tiers** (§9.3):
   - **Substrate diff-guard** (every PR) — universal scripts byte-identical
     to baseline except the three §5.3 hunks.
   - **Driver conformance L0–L3** (every PR) — L0 deterministic via P5a
     validator; L1/L2/L3 use synthetic fixtures under
     `_substrate/backends/_fixtures/`.
   - **Matrix smoke** (nightly) — for each `matrix_eligible:true` workflow
     × matrix driver (cuda, triton), capture prompts via P4 harness with
     mock harness fixtures (canned `build.sh`/`run.sh`/`profile.sh` JSON,
     no GPU). Assert structurally: guard passes, right driver dispatched,
     Layer-A envelope conforms. Negative cells assert exact error
     substring/code. Structural-only is sufficient; performance
     comparability is a deferred GPU concern (§10).

**Spec sections driving it:** §9.1 Phase 2 ("Generalist last"); §9.3
(entire — substrate diff-guard, driver conformance, matrix smoke,
anti_cheat per-backend patterns, tiering); §10 (the
"performance-comparable cross-backend scoring" risk — call it out in
matrix-smoke docs).

**Files touched:**
- `Generalist/generalist-kernel-optimization.js` — Stage B retrofit.
- `_meta/tools/fixtures/generalist-*.json` (full set).
- `_meta/tools/test/generalist-*.test.js` (full trio).
- `_meta/tools/test/matrix-smoke.test.js` (**new**) — the structural
  matrix runner; iterates over each `matrix_eligible:true` manifest +
  each matrix driver.
- `_substrate/backends/_fixtures/` (**new dir**) — canned
  `build.sh`/`run.sh`/`profile.sh` JSON outputs per driver, per fixture
  kernel. Per §9.3: "fixtures in `_substrate/backends/_fixtures/`".
- `.github/workflows/` or whichever CI definition the repo uses — three
  tiers wired up (every-PR, nightly, opt-in self-hosted-GPU). **Verify
  CI definition path in spike; this repo may not use GitHub Actions.**
- `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md` — append the
  Generalist row + document the opt-in GPU CI tier.

**Depends on:** P5b minimum (matrix smoke needs ≥1 matrix_eligible
workflow retrofitted; AdaExplore guarantees that). In practice waits for
P5d so the matrix smoke exercises the full clean tier and any common
bug found in P5c/d is fixed before CI starts gating PRs.

**Exit criteria (testable, no GPU):**
1. Generalist byte-identity + triton dry-run + guard tests green.
2. `matrix-smoke.test.js` green on the local checkout — exercises every
   `matrix_eligible:true` workflow (post-P5b/c/d that's all 15 clean
   workflows including Generalist) × {cuda, triton}.
3. CI definitions parse + run locally (or via `act` / equivalent dry-run);
   substrate diff-guard fires when a contrived `_substrate/diagnose.py`
   edit is staged.
4. Negative-cell coverage: assert vendor_locked workflow (e.g. AccelOpt
   from P4) throws on `backend:'metal'` with the intrinsic-reason message;
   assert method_intrinsic workflow throws on any off-list backend.
5. anti_cheat per-backend negative fixture: feed a Metal-shaped
   library-fallback kernel + a skipped-compute stub through the
   `[fallback]`/`[skip]` mechanism (P2 deliverable) — both yield
   `valid:false`. This proves the §5.3.3 wiring even before P6 ships the
   Metal driver (uses synthetic Metal manifest just for the assertion).

**Estimated commits:** 12–18 (Generalist 7 + matrix-smoke 3–5 + CI 2–4 +
fixtures 2–3).

**Open questions for P5e spike:**
- **CI runner.** What CI does this repo actually use? If none, P5e ships
  test files but not CI wiring; document the gap.
- **Matrix smoke determinism.** Can P4 harness iterate N workflows × M
  drivers in one process, or do we need a `matrix-runner.js` orchestrator?
  Suspect the latter; size in implementation plan.
- **Mock-harness contract.** §9.3 says "documented interface injecting
  canned JSON"; specify exactly (fixture URL? `$PATH` redirection to
  fake scripts?). Pick one.

---

### P5f — §9.4 documentation pass

**Scope:** Close out the documentation deliverables enumerated in spec
§9.4. Pure docs sub-plan; no code, no tests beyond markdown link/lint.

**Spec sections driving it:** §9.4 (entire), with cross-refs to §4.4
(manifest schema), §5.3 (substrate edits), §6 (consumption pattern),
§7 (portability tiers), §9.3 (CI tiers documented in registry README).

**Files touched:**
- `_substrate/BACKEND-DRIVER-SDK.md` — refresh to reflect P5a's manifest
  v1.1 schema additions, the validator location decision (Python or
  Node), and the matrix-smoke contract. Update the "Deviations from the
  spec" section if the §9.3 mock-harness interface adds any.
- `_substrate/backends/REGISTRY.md` — per-driver README sections (or
  separate per-driver READMEs) for cuda, triton: profiler, emitted
  metric names, fallback patterns, threshold-profile deviations from
  `nvidia` and why. Currently a 2-row table; expand to a real registry
  index.
- `_meta/manifests/schema.yaml` — header version bump + a clear
  "v1.1 changelog" comment explaining the `backend:` block + the
  `profile_command`/`ncu_command` migration.
- `_meta/README.md` (if it exists; verify in spike) — header refresh for
  the `backend:` block.
- `_substrate/ARCHITECTURE.md` — one new paragraph placing the Backend
  Driver axis as "a cross-cutting data axis owned by neither solver nor
  generator" (verbatim §9.4 item 4 language).
- `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md` —
  consider a "P5 outcomes" appendix noting which spec open questions
  P5 resolved (e.g. KSearch tier per P5d spike; mock-harness contract
  per P5e). Optional; can be a separate spec-amendment commit.

**Depends on:** P5e (the docs reference the CI tiers as live;
prematurely shipping docs that claim a tier exists before P5e merges
breaks the trust contract).

**Exit criteria (testable):**
1. Every file in §9.4 deliverables list updated.
2. SDK doc's "Deviations from the spec" section reflects the
   validator-language and mock-harness decisions made in P5a + P5e.
3. REGISTRY.md per-driver section exists for cuda + triton; each names
   profiler, emitted metric names, anti-cheat patterns, threshold-profile
   deviations.
4. Markdown lint clean (`markdownlint` or equivalent if the repo lints).
5. No broken cross-references to spec sections.

**Estimated commits:** 3–5 (SDK + REGISTRY + schema header + ARCHITECTURE +
optional spec amendment).

**Open questions for P5f:**
- Does the repo lint markdown? Spike — if not, just human-review.
- Do per-driver READMEs live under `_substrate/backends/<id>/README.md`
  or as sections inside `REGISTRY.md`? Spec §9.4 says "REGISTRY.md +
  per-driver README sections" — ambiguous. Pick one in implementation plan.

---

## 6. Risks (P5-level — beyond per-sub-plan risks already inline)

- **Generator-prompt drift.** P5a edits agent prompts; per spec §9.2 the
  generator is an LLM-agent pipeline, not a substitution engine. The P5a
  regression test pins structured-output schema but cannot pin body
  content — accept that the next generated workflow may need a manual
  conformance check; carry as a known limitation in the SDK doc.
- **Determinism assumption.** P5b/c/d/e all rely on workflows being
  deterministic (no `Date.now`/`Math.random`/`performance.now`) so the
  byte-identity gate works. Per-workflow spike: grep for these tokens; if
  found, the gate weakens to label-set + schema check.
- **Per-workflow tier reclassification.** §7.2 is one author's
  classification. Spikes (KSearch in P5d, KernelAgent in P5b) may reveal
  a workflow is actually `vendor_locked` or `method_intrinsic`. Each
  reclassification is a spec amendment via a §7.2-table edit committed
  before that workflow's implementation plan proceeds.
- **CI runner unknown.** P5e assumes a CI exists. If not, P5e ships
  tests but does not gate PRs; matrix smoke becomes a manual procedure.
- **Manifest tree duplication** (`_manifests/` vs `_meta/manifests/`) —
  mirrors the §2.3 `_tools/`↔`_meta/tools/` issue. P5a edits `_meta/`
  authoritatively and flags duplication; does not dedup (out of scope).

## 7. Recommended sub-plan authoring order

1. **P5a first** — schema; everything depends on it; spikes are bounded.
2. **P5b** after P5a merges — its spikes (canonical `kernel_path`,
   per-workflow determinism) inform P5c/d scope.
3. **P5c, P5d** — mechanical applications of P4+P5b pattern; one session
   per batch.
4. **P5e** — can be drafted in parallel with P5c/d (matrix-smoke design
   only depends on contract stability), but merges after P5d.
5. **P5f** — docs cleanup, last.

## 8. What this master plan deliberately does NOT decide

Per-workflow seam inventories; per-workflow agent-returns map contents;
exact CI runner / YAML; spec amendments triggered by spikes (each spike
commits its amendment separately); GPU-tier verification (opt-in,
inherited from P4's `DEFERRED-GPU-VERIFICATION.md`).

---

## Appendix A — Cross-reference index

| Spec section | Sub-plan(s) it drives |
|---|---|
| §4.4 manifest schema | P5a, P5f |
| §4.9 L0–L3 conformance | P5a (validator), P5e (CI tiers) |
| §5.3 substrate edits | precondition (P2); P5e (CI diff-guard verifies) |
| §6.1/§6.2/§6.3/§6.4 driver consumption | P5b, P5c, P5d, P5e (Generalist) |
| §7.2 portability classification | P5b/c/d/e (per-workflow application; spikes amend the table) |
| §9.1 Phase 2 ordering | P5b → P5c → P5d → P5e |
| §9.2 generator/template/schema | P5a |
| §9.3 CI / conformance / matrix smoke | P5a (validator), P5e (CI wiring) |
| §9.4 documentation | P5f |
| §10 risks — performance-comparable cross-backend | documented as a known limitation in P5e + P5f |
| Appendix A "P5+" row | this entire master plan |

**Workflow → sub-plan summary:** P5b={AdaExplore, KernelAgent};
P5c={KDA, CUDALLM, Astra, StitchCUDA, STARK}; P5d={KSearch, ReGraphT,
KernelFoundry, KernelFoundryDx, KernelSkill, AKO4X, KernelBand};
P5e={Generalist + matrix-smoke CI}. 15 workflows total. All exit P5 with
byte-identical legacy paths plus a working driver path validated against
the cuda + triton P3 drivers.
