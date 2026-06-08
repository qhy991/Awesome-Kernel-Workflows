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

> **Workflow count reconciliation:** §7.2 names 13 clean workflows in the
> first row, plus KernelBand as `clean*`, plus Generalist as `clean` — 15
> clean / matrix-eligible total. Phase 2 of §9.1 enumerates the same 13 +
> KernelBand + Generalist. This plan adopts §9.1 Phase 2's ordering verbatim:
> AdaExplore + KernelAgent first (P5b), then mid-complexity (P5c), then
> lower-complexity + KernelBand (P5d), then Generalist last (P5e).

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
                    P5a (schema + generator + template + validate-backend.js)
                     │   schema is contract; everything depends on it
        ┌────────────┼────────────┬──────────────┐
        │            │            │              │
        ▼            ▼            ▼              ▼
       P5b          P5b          (waits)        (waits)
   AdaExplore + KernelAgent          │              │
        │                            │              │
        └─────────────► P5c ◄────────┘              │
                  (KDA, CUDALLM, Astra,             │
                   StitchCUDA, STARK)               │
                        │                           │
                        └──────► P5d ◄──────────────┘
                          (KSearch, ReGraphT, KernelFoundry,
                           KernelFoundryDx, KernelSkill,
                           AKO4X, KernelBand)
                                  │
                                  ▼
                                 P5e
                       (Generalist + matrix-smoke CI;
                        needs ≥1 matrix_eligible workflow
                        retrofitted — guaranteed by P5b)
                                  │
                                  ▼
                                 P5f
                          (§9.4 docs pass)
```

**Edges in plain English:**
- `P5a → all` — the manifest `backend:` block and the validate-backend
  contract are the schema everything else conforms to.
- `P5b → P5c` — AdaExplore + KernelAgent are the proving ground; they
  surface generalizing-the-AccelOpt-pattern unknowns (e.g. canonical
  `kernel_path` discipline) that P5c batch-applies. Do not start P5c until
  the P5b sub-plans are merged.
- `P5c → P5d` — only sequencing for risk management (mid-complexity first;
  bug discoveries here adjust the P5d checklist). Not a hard contract edge.
- `P5b → P5e` (matrix smoke) — the matrix smoke needs ≥1 matrix_eligible
  workflow to actually exercise; P5b's AdaExplore guarantees that.
- `Generalist last` — it is the substrate reference (§9.1 Phase 2); any
  bug found in earlier batches still has a clean reference to compare
  against until P5e edits it.
- `P5e → P5f` — docs pass references the new CI tiers + the matrix smoke
  outcome; finalize after the CI lands.

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
  (`supported[]`, `default` (clean methods may set `null` → force explicit
  `--backend`), `matrix_eligible: true|partial|false`, `portability:
  clean|vendor_locked|method_intrinsic`, `intrinsic_to` (required when
  not clean)). Extend `args.optional[]` examples to include neutral
  `backend`, `backend_dir`, `driver_shell_prefix`, `substrate_dir` and a
  neutral `profile_command`; keep `ncu_command` as a documented alias with
  a deprecation-window comment (`ncu_command` is currently first-class per
  `generate-workflow.js:282`, so this is a deliberate contract change, not
  a "tightening"). Bump header to "Manifest Schema v1.1".
- `_meta/tools/generate-workflow.js` — edit the **agent prompts**, not a
  substitution table. Two regions:
  - model-args prompt at ~L263–282 (the standard-args list + the
    "Do not emit concrete default commands…" paragraph): teach the agent to
    extract `native_backend` + `portability_class` from the manifest, to
    emit `backend`/`backend_dir` as standard optional args, and to mark
    `ncu_command` as a deprecated alias for `profile_command`.
  - generate prompt at ~L492–524 (the "Generation Rules" list, currently
    rule #2 emits `WORKFLOW_SUITABILITY` with `supported_languages`): teach
    the agent to emit the §6.4 `method_supported_backends` /
    `default_backend` / `requires_capability` split, the §6.1 path-helper
    block (`SUBSTRATE` / `PY` / `SH` / `DRIVER_DIR` / `USE_DRIVER` /
    `substrateInstruction` / `driverPy` / `driverSh`), and the §6.2 Setup
    `load-driver` agent gated on `USE_DRIVER`. Rule #10 ("Never hardcode
    an evaluator/compiler/profiler command") tightened to forbid naming
    `nvcc`/`ncu`/`@triton.jit`/`__global__`/`PYBIND11_MODULE` in workflow
    body prompts — those come from `idioms.json:impl_requirements` /
    `lang_fence` injected at Setup.
- `_meta/tools/validate-workflow.js` — add **prompt checklist items only**
  (LLM-driven validator; see spec §9.2: any backend checks here are prompt
  hints, *not* hard gates). Items: warn on `supported_languages:` (legacy
  key); warn on `nvcc`/`ncu`/literal CUDA tokens in any agent prompt of
  a workflow whose manifest declares `portability: clean`.
- `_meta/tools/validate-backend.js` (**new**) OR
  `_substrate/backends/validate_backend.py` (per the SDK doc's
  "Deviations" section — Python won; settle that here, not in P5a's
  implementation plan). Deterministic L0 checks: `manifest.json` validates,
  `backend_id == basename(dir)`, `idioms.json` references only real
  `method_gate.TABLE` names, `capabilities.metrics ⊆` the 4 canonical
  keys, `bottleneck_classes ⊆` the 4 meaningful ∪ `{unknown}`, substrate
  scripts byte-identical to baseline EXCEPT the three §5.3 hunks. This is
  separate from the LLM-driven `validate-workflow.js`.
- `_templates/iterative-loop.js` and the other three templates
  (`search-based.js`, `single-pass.js`, `tree-exploration.js`): update the
  `[BLOCK]`/`{{TOKEN}}` **guidance text** + the input-policy comment
  (currently `_templates/iterative-loop.js:62-66`) to teach the LLM "the
  body never names a vendor profiler or vendor metric; backend tokens come
  from `idioms.json`". Documentation edits — not a substitution API.
  **Mirror the same edits to `_meta/templates/`** so the canonical tree
  is consistent; do NOT deepen `_tools/`↔`_meta/tools/` divergence (spec
  §2.3) — flag the duplication, edit `_meta/templates/` authoritatively.
- `_meta/tools/test/validate-backend.test.js` (**new**) OR Python pytest
  equivalent — exercises L0 against the cuda + triton drivers from P3 as
  positive cases plus crafted negatives (bad `backend_id`, unknown
  method_gate name, missing manifest field).
- `_meta/tools/test/generator-prompt-schema.test.js` (**new**) — the
  regression test the user called out: assert generator-prompt OUTPUT
  SCHEMA is unchanged for vendor-omitted (clean) manifest inputs.
  Concretely: pin the generate-workflow agent's `schema:` (the existing
  object after the L520 region) — backend-axis prompt edits must not
  alter the structured output keys the rest of the pipeline reads. Use
  the P4 `capturePrompts` harness in **prompt-introspection mode** (the
  generator IS itself a workflow .js) — same fixture style as P4.

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

**Open questions for P5a's implementation plan to resolve (spike first):**
- **Manifest tree location.** Spec §9.2 names `_meta/manifests/schema.yaml`,
  but there is also a top-level `_manifests/` dir. Confirm which is the
  authoritative tree before editing. (Spec §2.3 calls out `_tools/`↔
  `_meta/tools/` duplication; if a parallel `_manifests/`↔`_meta/manifests/`
  divergence exists, do NOT deepen it.)
- **Python vs Node validator.** The SDK doc declared the L0 validator is
  Python in `_substrate/backends/validate_backend.py`. The spec §9.2 + §9.3
  name `_meta/tools/validate-backend.js`. Resolve once and document the
  choice in the BACKEND-DRIVER-SDK "Deviations" log; do not ship both.
- **`ncu_command` deprecation window length.** Spec §10 lists this as an
  "open question". Pick a number (1 minor version? 2?) in P5a so the
  schema comment is concrete; do not paper over.

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

**Generalized AccelOpt-pattern checklist** (apply per workflow; the
per-workflow detail lives in each implementation plan):
1. **Golden capture** — capture today's prompts via
   `_meta/tools/print-workflow-prompts.js` with a fixed args fixture +
   deterministic `agentReturns` map that unlocks the full loop (the P4
   AccelOpt fixture is the reference shape). Commit pre-retrofit goldens
   under `_meta/tools/fixtures/<workflow>-today.golden.json` PLUS a
   `<workflow>-GOLDEN-BASELINE-SHA.txt` per P4 Stage A Task 4.
2. **Harness fixtures** — args.json + agent-returns.json per mode
   (optimize-existing vs generate-then-optimize, when both apply).
3. **`USE_DRIVER` gate** — add §6.1 path helpers + §6.4 guard split;
   wrap every backend-laden seam in `USE_DRIVER ? <driver> : <legacy>`.
   Legacy literal extracted to a column-0 named const before gating (P4
   "col-0 extraction" rule — load-bearing).
4. **Driver-path guards** — Setup `load-driver` agent (§6.2) only when
   `USE_DRIVER`; `JSON_PASSTHROUGH` schema; `diagnose.py` + Layer-A
   evidence assembly only on the driver path.
5. **Triton dry-run** — at least one non-CUDA driver-path render verified;
   assert no CUDA tokens (`nvcc`/`ncu`/`__global__`/`PYBIND11_MODULE`/
   `cuda_runtime.h`/`NCU Profile Results`/`cuda` fence) leak across the
   FULL rendered set.
6. **Byte-identity gate** — Stage-C-style test diffing
   no-`backend_dir` capture against the pre-retrofit golden; green is the
   merge gate.
7. **Deferred-GPU checklist update** — append the workflow to
   `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md` (created in P4).

**Backend posture per workflow:**

| Workflow | Source language(s) today | Triton-capable? | CUDA-only? | Notes |
|---|---|---|---|---|
| AdaExplore | CUDA `.cu` (MCTS over kernel space) | Yes (clean) | No | Method is backend-agnostic per §7.2; the *current* prompts name CUDA. Spike #1: confirm `args.kernel_path` is the only source-input arg before applying the checklist. |
| KernelAgent | Triton (per `KernelAgent/kernelagent-triton-synthesis.js` filename) | Triton-intrinsic *today*; clean tier per §7.2 means the topology generalizes | No | Verify: is KernelAgent's *method* triton-only, or is the topology language-agnostic and only the prompts triton-flavored? If method-intrinsic, demote to P5e/P3 phase classification. |

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
| KSearch | **Verify Triton-intrinsic vs clean.** User explicitly flagged: "KSearch may be Triton-intrinsic — verify." Spec §7.2 lists it as clean, but the implementation may be triton-flavored such that the method IS the backend. If triton-intrinsic, demote to vendor_locked `['triton']` and skip matrix smoke. |
| ReGraphT | "ReGraph" + "T" — possibly Triton-flavored; verify in spike. |
| KernelFoundry, KernelFoundryDx | Likely paired; check if Dx is a strict superset / experimental fork. May share fixtures. |
| KernelSkill | Check topology in spike. |
| AKO4X | Largest .js in the lower batch (46.8K per `ls` output); may need extra reviewer attention but pattern is the same. |
| KernelBand | `clean*`. NVIDIA-utilization φ-gate threshold becomes driver-resolved (read from `idioms.json:read_metric_guide` or a new manifest field). This is a *substantive* extra task on top of the checklist — call it out explicitly in the per-workflow plan. |

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

**Scope:** Two work strands sharing one sub-plan because they depend on
each other:
1. **Generalist retrofit** — Generalist is the substrate reference (§9.1
   Phase 2: "Generalist last"); retrofitting it earlier would remove the
   stable reference earlier workflows are sanity-checked against.
   Apply the same checklist.
2. **Matrix smoke test + CI tiers** (§9.3) — wire up the CI tiers
   guarded by P5a's validator + the deferred-GPU checklist:
   - **Substrate diff-guard** — runs every PR; asserts the 6 universal
     scripts byte-identical to baseline except the three §5.3 hunks.
     Each PR must keep the diff green or explicitly update the golden
     with a recorded rationale.
   - **Driver conformance (L0–L3)** — runs every PR for each driver
     under `_substrate/backends/`. L0 deterministic via P5a validator;
     L1/L2/L3 use synthetic fixtures under
     `_substrate/backends/_fixtures/` (per spec §9.3 + §4.9).
   - **Matrix smoke test** — runs nightly. For each `matrix_eligible:true`
     workflow × matrix driver (cuda, triton), capture prompts via the P4
     harness with **mock harness fixtures** (canned `build.sh`/`run.sh`/
     `profile.sh` JSON injected — no GPU). Assert structurally: guard
     passes, right driver dispatched, Layer-A envelope conforms via
     `evidence_schema.py validate`. **Negative cells:** assert exact
     error substring/code (`matrix_eligible:false` throws the
     intrinsic-reason; vendor_locked throws for illegal backends).
     Structural-only assertion is sufficient — performance comparability
     is a deferred GPU-tier concern.

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
  the test files but not the CI wiring; document the gap.
- **Matrix smoke determinism without P4 harness extensions.** The P4
  harness was designed for a single workflow at a time; can it iterate
  cleanly over N workflows × M drivers in one process, or do we need a
  `matrix-runner.js` orchestrator? Suspect the latter; size it in the
  implementation plan.
- **Mock-harness contract.** §9.3 says "mock harness (a documented
  interface injecting canned `build.sh`/`run.sh`/`profile.sh` JSON
  without a GPU)". The interface needs to be specified — does
  `build.sh` get replaced by a `fixture://` URL, or does the test set
  `$PATH` to point at fake scripts? Pick one in the implementation
  plan.

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

- **Generator-prompt drift.** P5a edits agent prompts in
  `generate-workflow.js`. Per spec §9.2: "the generator and validator
  are LLM-agent pipelines, not substitution engines." A prompt edit
  that *says* "emit the §6.1 helpers" may produce non-conforming code
  for novel manifests. **Mitigation:** P5a ships a regression test
  pinning the structured-output schema, but cannot pin the body content
  — accept that the next workflow generated post-P5a may need a manual
  conformance check against the P4 AccelOpt pattern; carry this as a
  known limitation in the SDK doc.
- **Determinism assumption.** P5b/c/d/e all rely on workflows being
  deterministic (no `Date.now` / `Math.random` / `performance.now`) so
  the byte-identity gate works. AccelOpt satisfies this. Per workflow
  spike: grep for those tokens. If a workflow uses them, the gate
  weakens to a label-set + schema check, and the merge criterion gets
  laxer — flag in that workflow's implementation plan.
- **Per-workflow tier reclassification.** Spec §7.2 is one author's
  classification. Spikes (KSearch in P5d, KernelAgent in P5b) may
  reveal a workflow is actually `vendor_locked` or `method_intrinsic`.
  Each reclassification is a spec amendment, not a hidden
  implementation-plan choice. **Mitigation:** every spike output gets a
  one-paragraph note appended to the spec §7.2 table; the
  implementation plan does not proceed until the table is updated.
- **CI runner unknown.** P5e assumes a CI exists. If not, P5e ships
  the test files but does not gate PRs; the matrix smoke becomes a
  manual-run procedure. Surfaces in P5e spike.
- **Manifest tree duplication (`_manifests/` vs `_meta/manifests/`).**
  Mirrors the §2.3 `_tools/`↔`_meta/tools/` issue. If both trees exist,
  P5a edits the canonical one (`_meta/` per §2.3 convention) and
  flags the duplication — does not dedup it (out of scope, spec §2.3).
- **Workflow count drift.** §7.2 says 15 clean / matrix-eligible
  total. §9.1 Phase 2 names 13 + Generalist + KernelBand = 15. P5b/c/d/e
  collectively cover all 15. If a new clean workflow lands between
  P5b and P5d, slot it into the appropriate batch — do not silently skip.

## 7. Recommended sub-plan ordering for the author

1. **Write P5a's implementation plan first.** It is the schema; everything
   depends on it; the spikes (manifest tree location; Python vs Node
   validator; `ncu_command` deprecation window) are well-bounded and
   resolvable in a single session.
2. After P5a merges, write P5b's implementation plan. P5b's spikes
   (canonical `kernel_path`; per-workflow determinism) inform P5c/d's
   scope.
3. P5c/d implementation plans are mechanical applications of the P4+P5b
   pattern; each can be written in one session per batch.
4. P5e's implementation plan should be written after P5b is merged but
   can start before P5c/d merge (the matrix-smoke design does not depend
   on P5c/d, only on the contract being stable).
5. P5f's implementation plan can be written last; it is a docs cleanup.

## 8. What this master plan deliberately does NOT decide

- Per-workflow seam inventories (each implementation plan does its own,
  patterned on P4's "Seam inventory — 21 CUDA/NCU couplings" table).
- Per-workflow agent-returns map contents (each implementation plan
  derives these from reading the workflow source, like P4 did).
- The exact CI runner / GitHub Actions YAML (P5e spike-then-decide).
- Spec amendments triggered by spikes (each spike commits its amendment
  separately; the master plan only flags that this can happen).
- GPU-tier verification — opt-in self-hosted-runner; out of scope per
  §10 + carried over from P4's `DEFERRED-GPU-VERIFICATION.md`.

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

## Appendix B — Workflow → sub-plan mapping

| Workflow | Sub-plan | Tier (per spec §7.2) |
|---|---|---|
| AdaExplore | P5b | clean / any |
| KernelAgent | P5b | clean / any (verify in spike) |
| KDA | P5c | clean / any |
| CUDALLM | P5c | clean / any |
| Astra | P5c | clean / any |
| StitchCUDA | P5c | clean / any |
| STARK | P5c | clean / any |
| KSearch | P5d | clean / any (**verify Triton-intrinsic**) |
| ReGraphT | P5d | clean / any |
| KernelFoundry | P5d | clean / any |
| KernelFoundryDx | P5d | clean / any |
| KernelSkill | P5d | clean / any |
| AKO4X | P5d | clean / any |
| KernelBand | P5d | clean* (φ-gate driver-resolved) |
| Generalist | P5e | clean / any (substrate reference; migrate last) |

15 workflows total. All exit P5 with byte-identical legacy paths plus a
working driver path validated against the cuda + triton P3 drivers.
