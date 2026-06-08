# P5c — Mid-complexity batch retrofit (KDA, CUDALLM, Astra, StitchCUDA, STARK)

> **Branch:** `dev/p5c-plan` for the plan; implementation lands on dev branches
> (`dev/p5c-kda-*`, `dev/p5c-cudallm-*`, …) and merges into
> `dev/solver-substrate`.
>
> **Parent:** `docs/superpowers/plans/2026-06-08-p5-clean-workflow-migration.md`
> §P5c. Pattern template (proven on 2 workflows):
> `docs/superpowers/plans/2026-06-08-p5b-adaexplore-kernelagent-retrofit.md`.
> Pattern source (executed pilot):
> `docs/superpowers/plans/2026-06-08-accelopt-driver-pilot.md`.
> Spec of record:
> `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md` §9.1
> Phase 2 (KDA, CUDALLM, Astra, StitchCUDA, STARK listed in this order).
>
> **Scope summary:** Apply the (now-proven-twice) P5b 7-stage retrofit
> checklist to **five mid-complexity clean-tier workflows**. Each is its
> own commit cluster following the §6 driver-consumption checklist. No
> CI work (deferred to P5e); no docs pass (deferred to P5f); no GPU
> verification (deferred via `DEFERRED-GPU-VERIFICATION.md`).
>
> **Pre-retrofit spike findings (30 spikes — 6 per workflow):** completed
> on `dev/p5c-plan` before this plan was written. See §2.
>
> **Critical finding up front:** **STARK has 5 `Math.random()` call sites**
> in `selectNode()` (lines 212, 213, 217, 228, 231). This breaks the
> byte-identity merge gate. STARK is therefore **flagged as a P5c blocker**
> requiring a redesign decision before it can be retrofitted (introduce a
> seeded-RNG arg, or downgrade to a per-label-shape gate, or stub
> `selectNode()` in fixtures). See §2.5 Spike c-5-3 and §8 Risk R1.
> KDA / CUDALLM / Astra / StitchCUDA have **no banned-API hits** and can
> proceed as scheduled.

---

## 1. Goal, scope, non-goals

### 1.1 Goal

Retrofit 5 mid-complexity clean-tier workflows onto the backend-driver
contract proved out by P4 on AccelOpt and hardened by P5b on AdaExplore +
KernelAgent. After P5c merges, the legacy path for each workflow is
byte-identical to today; an opt-in `args.backend_dir` path consumes the
cuda or triton driver via the §6 pattern. P5c clears the §9.1 Phase 2
mid-complexity batch in spec ordering.

### 1.2 Scope

5 workflows × 9 task slots (A1–A4, B1–B3, C1–C2) = 45 tasks total. Each
workflow ships its own commit cluster (~7–9 commits depending on seam
count). Per-workflow files touched:

- `<workflow>/manifest.yaml` (create for KDA; extend for the other 4)
- `<workflow>/<workflow-entry>.js` (additive, `backend_dir`-gated)
- `_meta/tools/fixtures/<workflow>-{args,agent-returns,golden,args-<other-backend>}.json`
- `_meta/tools/fixtures/<WORKFLOW>-GOLDEN-BASELINE-SHA.txt`
- `_meta/tools/test/<workflow>-{byte-identity,guard,<other-backend>-dryrun}.test.js`
- `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md` (append rows)

### 1.3 Non-goals (explicit)

P5c is **5 workflows, no CI work, no docs pass**. Specifically out of
scope:

- CI runner wiring, matrix-smoke harness, `_substrate/backends/_fixtures/`
  authoring — all deferred to **P5e**.
- §9.4 documentation pass (SDK, REGISTRY, ARCHITECTURE) — deferred to **P5f**.
- Any edits under `_substrate/**` (substrate diff-guard must stay green).
- Any edits to `_meta/tools/{schema-stub,run-workflow,print-workflow-prompts,validate-workflow,validate-backend,generate-workflow}.js`
  (those are P5a-owned and considered frozen for P5c's purposes).
- Triton-tier or vendor-locked workflows (CUDAAgent, cuPilot, KEET,
  KernelBlaster, GPUForecasters, TritorX, Xe-Forge) — Phase 3.
- Method-intrinsic workflows (ArchAgent, CutlassGEMM, FACT, ARGUS) —
  Phase 4.
- KSearch, ReGraphT, KernelFoundry(Dx), KernelSkill, AKO4X, KernelBand,
  Generalist — deferred to **P5d / P5e**.
- L3 method-coverage tightening (`idioms.methods.<name>` per-method
  guidance enrichment) — explicit deferred item in P5b §B2 step 3,
  carried forward unchanged.
- GPU verification of any driver path — appended to
  `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md`.

### 1.4 Hard preconditions

- **P5a merged** (manifest v1.1 schema + `validate-workflow.js` warns on
  `supported_languages:` + `validate-backend.{js,py}` L0).
- **P5b merged** (the 7-stage checklist proven; `capturePrompts` harness +
  `print-workflow-prompts.js` + `schema-stub.js` + `run-workflow.js`
  present at `_meta/tools/`; AdaExplore + KernelAgent fixtures + tests
  green; new column-0 helper `workspaceKernelPath(args, driver)` exists
  in SDK doc per P5b Risk-1 mitigation).
- P3 cuda + triton drivers L0 conformant (precondition of P5b).
- P4 AccelOpt pilot merged.

If any precondition is missing, P5c cannot start. The plan assumes a
clean `dev/solver-substrate` with P5b fully landed.

---

## 2. Pre-flight spike resolutions (30 spikes, 6 per workflow)

Each workflow resolves the six P5c-mandated spikes:

- **c-N-1** — `agent()` site inventory (line, label, phase, schema,
  vendor-token presence).
- **c-N-2** — loop unlock + fixture key count.
- **c-N-3** — banned-API audit (`Math.random()`, `Date.now()`,
  `performance.now()`, `new Date()`).
- **c-N-4** — manifest status (exists / schema version / `backend:` block /
  `supported_languages`).
- **c-N-5** — tier classification (clean confirmed vs demoted).
- **c-N-6** — kernel-path convention (AccelOpt-style, KernelAgent-style,
  or new third pattern).

All file:line citations are against today's tree (commit ec680f8 +
P5c-plan branch state). Re-verify before Stage-A capture begins.

### 2.1 KDA — `KDA/kda-kernel-workflow.js` (593 lines, 23.5K)

**c-1-1 — `agent()` site inventory.** 7 call sites; all use schemed
`JSON_PASSTHROUGH`-style return objects.

| # | Line | Label/phase | Schema shape (top keys) | Vendor tokens in prompt |
|---|---|---|---|---|
| 1 | 177 | Setup / Generation fallback (no `kernel_path`) | `{generated_kernel_path, initial_candidates[], initial_generation_result}` | `cuda`, `kernel_path`/`result_path` braces |
| 2 | 214 | Inspect | `{baseline_summary, validation_command, eval_command, …}` | `cuda` (in suitability fallback) |
| 3 | 271 | Plan / draft | `{draft_md, plan_md, candidates[]}` | `cuda` (objective text) |
| 4 | 339 | Plan / executable | `{candidates[{title, changes, expected_effect}], notes}` | `cuda`, `ncu-report-skill` |
| 5 | 403 | Implement (per-candidate loop) | `{kernel_code, files[], rationale}` | `cuda`, `kernel_path`/`result_path` |
| 6 | 451 | Validate | `{validation_result, perf_metrics, evidence_path}` | `cuda`, `ncu-report-skill`, `test_command`, `benchmark_command` |
| 7 | 556 | Report | `{report_md, candidates_jsonl}` | `cuda` (campaign summary) |

Net: **7 `agent()` calls; ~6 driver-gated seams** (#1 fallback is gated
only in the `args.kernel_path === ''` branch; #7 report has no
backend-specific tokens beyond a literal `cuda` reference and **needs no
edit** in P5c — same call as P5b AdaExplore S7).

**c-1-2 — loop unlock + fixture key count.** Single iterative loop
(L394 `for (iteration = 0; iteration < Math.min(planCandidates.length,
MAX_CANDIDATES) && !promotionMet; iteration++)`). Branches:
`promotionMet` short-circuit (controlled via `validate.<i>` return value
`promotion_met: boolean`); `args.kernel_path === ''` Setup fork (true =
exercises #1, false = skips). For full-render coverage at
`max_candidates: 2`: `setup.fallback?`, `inspect`, `draft`, `plan`,
`implement.0`, `validate.0`, `implement.1`, `validate.1`, `report` =
**9 keys** (or 10 with the no-kernel-path Setup fork). Fixture should pin
`max_candidates: 2` to keep this minimal.

**c-1-3 — banned-API audit.** **CLEAN.** `grep -nE
'Math\.random|Date\.now|performance\.now|new Date\(\)'` returns zero
hits. Byte-identity gate **viable** as the merge criterion.

**c-1-4 — manifest status.** **MISSING.** `KDA/manifest.yaml` does
**not** exist (verified). P5c must **create** it from scratch under v1.1
schema (P5a). Compare KernelAgent/CUDALLM/Astra/STARK all have one;
StitchCUDA has one. KDA is the **lone create-from-scratch** of the
batch.

**c-1-5 — tier classification.** **CLEAN confirmed.** Per §7.2 row 1.
The README ("Kernel Design Agents workflow: evidence-driven kernel
optimization with draft-plan-implement-validate-record cycles") and the
phase list (Inspect → Plan → Implement → Validate → Decide → Report) are
language-agnostic; the CUDA-ness lives in (i) `WORKFLOW_SUITABILITY`
`supported_languages: ['cuda']` literal at L30 (will be widened to
`[cuda, triton]` per §6.4 guard split), (ii) prompt text inside the 7
`agent()` calls, and (iii) optional `cuda-kernel-development` skill
binding at L14 (skill remains optional; the workflow body does not import
it). **No vendor-locked control flow.** `supported: [cuda, triton]`,
`default: cuda`, `matrix_eligible: true`, `method_supported_backends:
any`.

**c-1-6 — kernel-path convention.** **AccelOpt-style.** L120
`let KERNEL_PATH = args.kernel_path || ''` plus L138 hard-throw if
none of `kernel_path | problem_definition | problem_path` is supplied.
KDA exposes **three** input modes:

- `args.kernel_path` (canonical AccelOpt pattern — file on disk).
- `args.problem_definition` (inline spec text — triggers the L177
  "generation fallback" agent that **produces** a kernel and assigns to
  `KERNEL_PATH`).
- `args.problem_path` (file on disk containing a problem spec).

For driver dispatch, KDA is **AccelOpt-pattern** for the first mode and
**KernelAgent-workspace-local-pattern** for the latter two (the generated
kernel lands at `${EXP_DIR}/generated/<filename>`, then `KERNEL_PATH` is
reassigned). **Both P5b helpers apply unchanged** — no third canonical
helper needed. KDA's Stage-B B1 selects the right helper based on
whether `KERNEL_PATH` was user-supplied vs generated.

### 2.2 CUDALLM — `CUDALLM/cudallm-fsr-kernel-generation.js` (583 lines, 19.8K)

**c-2-1 — `agent()` site inventory.** 8 call sites.

| # | Line | Label/phase | Schema shape (top keys) | Vendor tokens in prompt |
|---|---|---|---|---|
| 1 | 182 | Setup | `{task_summary, evaluator_contract, …}` | `cuda` (~6×), `kernel_path`/`result_path` |
| 2 | 226 | Catalog (CUDA feature search space) | `{features[{name, category, hint}]}` | `cuda` (heavy — feature catalog explicitly names "shared memory tiling", "warp shuffle", "tensor core mma", "ldmatrix") |
| 3 | 278 | Test plan | `{tests[{name, shapes, dtype, expected}]}` | `cuda` |
| 4 | 324 | Selection (per-iteration) | `{selected_features[], rationale}` | `cuda` |
| 5 | 364 | Generation (per-iteration × per-sample) | `{kernel_code, feature_use_log}` | `cuda`, `.cu` extension |
| 6 | 403 | Evaluation (per-iteration × per-sample) | `{compiled, correct, latency_ms, log}` | `cuda`, `nvcc` (in eval contract), `ncu-report-skill` |
| 7 | 464 | Reinforce (per-iteration) | `{updated_scores{}, kept_features[], dropped_features[]}` | `cuda` |
| 8 | 512 | Final report | `{report_md, best_features[]}` | `cuda` |

Net: **8 `agent()` calls; ~7 driver-gated seams** (#8 is summary-only;
no command tokens; **no edit needed** per P5b convention).

**c-2-2 — loop unlock + fixture key count.** Nested loop: `for (iter
= 0; iter < ITERATIONS; iter++) { for (sample = 0; sample <
SAMPLES_PER_FEATURE_SET; sample++) { ... } }`. Branches: convergence
short-circuit at the end of each iteration (controlled by `reinforce`
return value `converged: boolean`); `compiled && correct` per-sample
gate that skips reinforce contribution (handled inside the agent's
returned `evaluation` schema, not in JS control flow). For minimal
full-render fixture pin `iterations: 2`, `samples_per_feature_set: 2`:
`setup`, `catalog`, `test_plan`, `selection.0`, `generation.0.0`,
`generation.0.1`, `evaluation.0.0`, `evaluation.0.1`, `reinforce.0`,
`selection.1`, `generation.1.0`, `generation.1.1`, `evaluation.1.0`,
`evaluation.1.1`, `reinforce.1`, `final_report` = **16 keys**.
Largest fixture in the P5c batch.

**c-2-3 — banned-API audit.** **CLEAN.** Zero hits. Byte-identity gate
**viable**.

**c-2-4 — manifest status.** **EXISTS, v1.0** (no `schema_version` key,
no `backend:` block — confirmed via grep). Has rich `source`, `workflow`,
`method`, `topology` sections per existing manifest convention. P5c
**extends** with `schema_version: "1.1"` header + `backend:` block.

**c-2-5 — tier classification.** **CLEAN confirmed** despite the
workflow name. The README + the manifest `method.category:
iterative_self_improving` describe the *method* (Feature Search +
Reinforcement) as backend-agnostic — features happen to be CUDA-named
today, but the algorithm is "select features → generate candidate →
evaluate → update scores" which works for any backend with a feature
catalog. The CUDA-flavored feature **catalog text** (Spike #1 row 2) is
prompt content the driver swaps for `${driverIdioms().feature_catalog
|| LEGACY_CUDA_FEATURE_CATALOG}` — same pattern as P5b AdaExplore S4
reviser perf hints. **Out-of-scope deviation flag:** the triton driver
today has no `feature_catalog` idiom; the driver path falls back to a
generic "explore the standard Triton optimization idioms" placeholder.
File as a deferred L3-tightening item alongside P5b's reviser hints.
`supported: [cuda, triton]`, `default: cuda`, `matrix_eligible: true`.

**c-2-6 — kernel-path convention.** **KernelAgent-workspace-local
pattern.** No `args.kernel_path` at all (verified — only `EXP_DIR`,
`PROBLEM_DEFINITION`, `TASK_SPEC_PATH`, `REFERENCE_CODE_PATH`). The
kernel is **generated** by the L364 agent and lands at
`${EXP_DIR}/cudallm_iter_${iteration}_sample_${sample}.cu` (literal at
L414). For driver dispatch, the workspace path is computed per-sample:
`workspaceKernelPath(args, driver) = ${EXP_DIR}/cudallm_iter_${iter}_sample_${sample}${driver.source_ext}`.
**No third pattern needed** — P5b KernelAgent helper covers this; the
only deviation is the path template (per-iteration-per-sample vs
KernelAgent's per-seed). Captured as a column-0 helper
`cudallmCandidatePath(iter, sample, driver)` local to the workflow.

### 2.3 Astra — `Astra/astra-kernel-optimization.js` (631 lines, 22.0K)

**c-3-1 — `agent()` site inventory.** 10 call sites.

| # | Line | Label/phase | Schema shape (top keys) | Vendor tokens in prompt |
|---|---|---|---|---|
| 1 | 179 | Setup / Generation fallback | `{generated_kernel_path, initial_candidates[], initial_generation_result}` | `cuda`, `.cu`, `kernel_path` |
| 2 | 218 | Setup (Astra setup agent) | `{kernel_summary, integration_plan, baseline_module, …}` | `cuda`, `sgl_kernel`, `sglang` |
| 3 | 261 | Testing Agent | `{tests[], compare_kind, baseline_func_resolved}` | `cuda` |
| 4 | 308 | Profiling Agent (baseline) | `{baseline_metrics{}, profile_evidence_path}` | `cuda`, `ncu`, `nsys` (in prompt text) |
| 5 | 354 | Planning Agent (per-iter) | `{proposed_optimization{}, expected_effect}` | `cuda` |
| 6 | 399 | Coding Agent (per-iter) | `{kernel_code, files_changed[]}` | `cuda` |
| 7 | 437 | Eval (Testing + Profiling combined, per-iter) | `{compiled, correct, latency_ms, metrics{}, log}` | `cuda`, `ncu` |
| 8 | 499 | Lesson distillation (per-iter) | `{lesson_md, generalization}` | `cuda` |
| 9 | 529 | Post-process | `{post_steps[], handoff_notes}` | `sglang`, `sgl_kernel` (when `integration_mode: sglang`) |
| 10 | 573 | Final report | `{report_md}` | `cuda` |

Net: **10 `agent()` calls; ~8 driver-gated seams** (#9 only if
`INTEGRATION_MODE === 'sglang'`; #10 summary-only).

**c-3-2 — loop unlock + fixture key count.** Single iterative loop at
~L350 (`for (iter = 0; iter < MAX_ITERATIONS; iter++)`). Branches:
`integration_mode: 'standalone' | 'sglang'` selects the post-process
shape; `args.kernel_path === ''` fork at L142 triggers L179 generation
fallback. For full-render fixture pin `iterations: 2`,
`integration_mode: 'standalone'` (skip #9): `setup.fallback?`, `setup`,
`tests`, `baseline_profile`, `plan.0`, `code.0`, `eval.0`, `lesson.0`,
`plan.1`, `code.1`, `eval.1`, `lesson.1`, `post_process`, `final_report` =
**14 keys**.

**c-3-3 — banned-API audit.** **CLEAN.** Zero hits. Byte-identity gate
**viable**.

**c-3-4 — manifest status.** **EXISTS, v1.0** (10.4K, no
`schema_version`, no `backend:` block). Extend per P5c.

**c-3-5 — tier classification.** **CLEAN confirmed.** The
README + manifest describe Astra as "production CUDA kernel optimization
with structured planning/coding/testing/profiling agents" — but the
multi-agent topology (plan → code → eval → distill) is backend-agnostic
verbatim. The CUDA-ness lives in prompt text + the optional `sglang`
integration. **Caveat — `integration_mode: 'sglang'` is vendor-locked
(NVIDIA + sglang).** Resolution: keep `sglang` mode reachable on the
legacy path; on the driver path, throw if `args.integration_mode ===
'sglang'` and `driver.backend_id !== 'cuda'` (a §6.4-style guard
extension specific to Astra). `supported: [cuda, triton]`, `default:
cuda`, `matrix_eligible: true` (matrix-smoke only exercises
`integration_mode: 'standalone'`).

**c-3-6 — kernel-path convention.** **Hybrid (KDA-style).** L122 `let
INITIAL_KERNEL_PATH = args.kernel_path || ''` with the same three-way
input check as KDA. Same resolution: AccelOpt-style for user-supplied
path; KernelAgent-workspace-local for generated mode (path is
`${EXP_DIR}/generated/<filename>`, reassigned to
`INITIAL_KERNEL_PATH` after L212). **No third helper needed.**

### 2.4 StitchCUDA — `StitchCUDA/stitchcuda-kernel-optimization.js` (636 lines, 21.6K)

**c-4-1 — `agent()` site inventory.** 6 call sites (sparsest of the
five — large bodies per call, lots of static logic in between).

| # | Line | Label/phase | Schema shape (top keys) | Vendor tokens in prompt |
|---|---|---|---|---|
| 1 | 84 | Setup StitchCUDA | `{cuda_version, target_architecture, pytorch_available, kernel_spec{}, kernelbench_config{}, replan_heuristics{}, max_attempts}` | `cuda` (heavy — "CUDA version", "sm_80/89/90", "PyTorch load_inline", "KernelBench"), `nvcc` (implicit via compile-config) |
| 2 | 197 | Adaptive replan (when triggered) | `{new_strategy, revised_heuristics{}}` | `cuda` |
| 3 | 263 | Planner (per-iter) | `{plan{}, anchors[], priority}` | `cuda` |
| 4 | 357 | Coder (per-iter) | `{kernel_code, files[]}` | `cuda`, `__global__` (in prompt code-fence guidance) |
| 5 | 416 | Verifier (per-iter) | `{compiled, correct, latency_ms, log, replan_signal: boolean}` | `cuda`, `nvcc`, `pytest`/`load_inline` |
| 6 | 544 | Report | `{report_md}` | `cuda` |

Net: **6 `agent()` calls; ~5 driver-gated seams** (#6 summary-only).

**c-4-2 — loop unlock + fixture key count.** Single iterative loop
with **adaptive replan** branching: `for (attempt = 0; attempt <
max_attempts; attempt++)`. The verifier returns `replan_signal: true`
when consecutive failures cross the threshold (`compile_failure_threshold`
or `correctness_failure_threshold` from setup); that triggers L197's
**conditional** replan `agent()`. For full-render fixture pin
`max_attempts: 2` and craft `verify.0` to return `replan_signal: true`
to exercise the replan branch: `setup`, `plan.0`, `code.0`, `verify.0`,
`replan.0`, `plan.1`, `code.1`, `verify.1`, `report` = **9 keys**.

**c-4-3 — banned-API audit.** **CLEAN.** Zero hits. Byte-identity gate
**viable**.

**c-4-4 — manifest status.** **EXISTS, v1.0** (largest manifest in the
batch at 14.9K — heavy method documentation). No `schema_version`, no
`backend:` block. Extend per P5c.

**c-4-5 — tier classification.** **CLEAN — but reclassification
worth a second look.** The name "StitchCUDA" is suggestive of vendor
binding; the manifest method describes "three-agent
planner/coder/verifier orchestration around KernelBench + PyTorch
load_inline". KernelBench is CUDA-only as a benchmark suite, but as a
**method** the orchestration topology is backend-agnostic — the
verifier just needs a `(compiled, correct, latency_ms)` triple, which
the triton driver's `run.sh` envelope produces verbatim. **Verdict:
clean confirmed**, with the same caveat as Astra `integration_mode:
sglang` — the KernelBench harness wiring may not generalize. P5c emits
the manifest with `supported: [cuda, triton]`, `default: cuda`,
`matrix_eligible: true`, but the StitchCUDA Stage-B `USE_DRIVER` path
**throws** if the user explicitly requests `kernelbench_config.benchmark_suite`
combined with non-cuda driver (Astra-style intersectional guard).

**c-4-6 — kernel-path convention.** **KernelAgent-workspace-local
pattern** — verified: only 4 `args.<x>` references in the body
(`args.language`, `args.problem_type`, plus two in suitability), no
`args.kernel_path`. The kernel is generated by the Coder agent (L357)
and the Verifier (L416) executes it; no file path passed in. The
workspace path is computed as `${args.workspace ||
'/tmp/stitchcuda'}/attempt_${attempt}/kernel${driver.source_ext}`.
**No third helper needed** — P5b KernelAgent helper covers this
verbatim with a per-attempt path template.

### 2.5 STARK — `STARK/stark-kernel-optimization.js` (840 lines, 31.6K)

**c-5-1 — `agent()` site inventory.** 8 call sites (largest workflow
in the batch by line count; agent calls are well-segregated from the
MCTS/ε-greedy tree machinery).

| # | Line | Label/phase | Schema shape (top keys) | Vendor tokens in prompt |
|---|---|---|---|---|
| 1 | 340 | Generation fallback (no `kernel_path`) | `{generated_kernel_path, initial_candidates[], initial_generation_result}` | `cuda`, `kernel_path`/`result_path` |
| 2 | 376 | Setup (read reference kernel + structure) | `{reference_summary, lang_fence, anchors[], …}` | `cuda` (heavy — reference kernel is embedded in `\`\`\`cuda` fence) |
| 3 | 413 | Root evaluation | `{compiled, correct, runtime, kernel_code, log}` | `cuda` |
| 4 | 504 | Debug agent (per-iter, on failure) | `{kernel_code, fix_summary}` | `cuda`, `\`\`\`cuda` fence (sibling-pattern code blocks) |
| 5 | 542 | Plan agent (per-iter) | `{plan{}, grounded_instructions[], anchors[]}` | `cuda` |
| 6 | 600 | Code agent (per-iter) | `{kernel_code, anchor_resolutions[]}` | `cuda`, `\`\`\`cuda`, explicit "CUDA code" |
| 7 | 648 | Eval agent (per-iter) | `{compiled, correct, runtime, log}` | `cuda` |
| 8 | 738 | Report | `{report_md, leaderboard[]}` | `cuda` |

Net: **8 `agent()` calls; ~7 driver-gated seams** (#8 summary-only).

**c-5-2 — loop unlock + fixture key count.** MCTS-shaped: `for (budget
= 0; budget < BUDGET; budget++)` (BUDGET defaults 30). Per-iteration
branches: `selectNode()` returns one of {best-by-score, leaf-biased
random, full-random fallback} — **controlled by `Math.random()`,
see c-5-3**; node status fork {compile-fail → debug, success → plan →
code → eval}; root expansion fork (first N_ROOT iterations expand root).
**Even with BUDGET pinned to 2**, the `selectNode()` randomness means
the fixture cannot pin the selected node deterministically without
addressing c-5-3 first.

Conditional on c-5-3 resolution (option A: seeded RNG), key count for
BUDGET=2: `setup.fallback?`, `setup`, `root_eval`, `select.0 → plan.0,
code.0, eval.0`, `select.1 → plan.1, code.1, eval.1` (assume both succeed,
no debug) = **9–10 keys**. With one debug fork add `debug.1` = 11.

**c-5-3 — banned-API audit.** **BLOCKER. `Math.random()` × 5 in
`selectNode()`** (lines 212, 213, 217, 228, 231). Specifically:

- L212–213: `fallback` and full-tree fallback random pick.
- L217: ε coin flip (`const coin = Math.random()`).
- L228: leaf-biased explore random pick.
- L231: expandable fallback random pick.

These calls are in pure JS body (not inside an `agent()` prompt), so
they directly affect which `(phase, node_id)` keys the fixture must
supply per iteration. **Byte-identity gate is NOT viable as written.**

**Three resolution options (must pick ONE in Stage-A, before any
retrofit edit):**

- **Option A — Seeded RNG arg (recommended).** Introduce a new
  optional `args.rng_seed: number | null` (default null = today's
  `Math.random()` behavior). When set, replace all 5 `Math.random()`
  call sites with a deterministic xorshift32 PRNG seeded from
  `args.rng_seed`. Fixture pins `rng_seed: 42` (or similar). This is a
  **pre-retrofit refactor** committed under Stage A as a separate
  task (call it A0) — NOT under Stage B (Stage B is `backend_dir`-gated
  additive). The legacy path (no `rng_seed`) preserves today's
  `Math.random()` behavior byte-identically. Effort: ~30 LoC.
- **Option B — Per-label-shape gate (escape hatch).** Drop byte-identity
  for STARK; fall back to a "per-label-shape" gate per P5b Risk-2 escape
  hatch: assert prompt count + per-label length-bounded snippet match.
  Cheaper to land but **weaker merge gate**; future STARK edits could
  drift prose without test signal. Discouraged.
- **Option C — Mock `selectNode` in capture.** Patch the harness to
  inject a deterministic `selectNode` for the capture phase only. Adds
  test-only coupling between `print-workflow-prompts.js` and STARK
  internals; ugly. Discouraged.

**Recommendation: Option A.** It's a clean refactor, byte-preserving on
the legacy path, and useful beyond P5c (reproducible MCTS runs are a
desirable property in their own right). This plan assumes Option A
chosen; if the implementer picks Option B/C, this plan's STARK section
needs amendment before Stage A.

**c-5-4 — manifest status.** **EXISTS, v1.0** (smallest of the four
existing manifests at 10.4K — sparse method documentation). No
`schema_version`, no `backend:` block. Extend per P5c.

**c-5-5 — tier classification.** **CLEAN — confirmed.** Per §7.2 row 1.
The STARK method (tree search + ε-greedy expansion + grounded-instruction
anchor resolution + debug/plan/code/eval rotation) is backend-agnostic
verbatim. The CUDA-ness lives in (i) prompt text, (ii) the `\`\`\`cuda`
fence in `buildPlanContext`/`buildCodeContext`/`buildDebugContext`
(lines 263, 275, 280, 289, 293, 304, 313, 319, 416, 608), and (iii)
`WORKFLOW_SUITABILITY` literal. **All swappable via driver
`lang_fence` + `idioms`.** `supported: [cuda, triton]`, `default:
cuda`, `matrix_eligible: true`.

**c-5-6 — kernel-path convention.** **Hybrid (AccelOpt + KDA-style).**
L113 `let REF_KERNEL_PATH = args.kernel_path || ''` with the same
three-way fork as KDA + Astra. Additional input: `args.test_harness_path`
(separate from kernel — exists as standalone). For driver dispatch:
AccelOpt-style for the user-supplied reference kernel; the **candidate
kernels generated per-node** are workspace-local at
`${EXP_DIR}/node_${node_id}${driver.source_ext}` — KernelAgent-style.
**No third helper needed.**

---

## 3. Per-workflow tier table

Locked in this sub-plan; emitted into each `<workflow>/manifest.yaml`'s
`backend:` block in Stage A Task A1.

| Workflow | `portability` | `matrix_eligible` | `method_supported_backends` | `supported` | `default` | Kernel-path pattern | Topology | Manifest status |
|---|---|---|---|---|---|---|---|---|
| KDA | `clean` | `true` | `any` | `[cuda, triton]` | `cuda` | AccelOpt + KernelAgent (hybrid) | Iterative-loop (draft→plan→implement→validate, candidate-at-a-time) | **CREATE** |
| CUDALLM | `clean` | `true` | `any` | `[cuda, triton]` | `cuda` | KernelAgent-workspace-local (per-iter × per-sample template) | Iterative-loop nested (iter × sample) | EXTEND v1.0→v1.1 |
| Astra | `clean` | `true` | `any` | `[cuda, triton]` | `cuda` | AccelOpt + KernelAgent (hybrid) | Iterative-loop (plan→code→eval→distill) + post-process branch | EXTEND v1.0→v1.1 |
| StitchCUDA | `clean` | `true` | `any` | `[cuda, triton]` | `cuda` | KernelAgent-workspace-local (per-attempt template) | Iterative-loop with adaptive-replan branch | EXTEND v1.0→v1.1 |
| STARK | `clean` | `true` | `any` | `[cuda, triton]` | `cuda` | AccelOpt + KernelAgent (hybrid) | MCTS/ε-greedy tree-exploration (debug/plan/code/eval rotation) | EXTEND v1.0→v1.1 **+ Option-A seeded-RNG refactor (A0)** |

**Intersectional guards (Astra + StitchCUDA only):** when `USE_DRIVER`
and a vendor-binding mode arg is set, throw with a clear message:

- Astra: `args.integration_mode === 'sglang'` + `driver.backend_id !==
  'cuda'` → throw.
- StitchCUDA: `args.kernelbench_config.benchmark_suite` user-set +
  `driver.backend_id !== 'cuda'` → throw.

These guards are NOT in P5b (AdaExplore + KernelAgent had no such
modes). They are documented as a **Stage B1 task addendum** for the two
workflows.

**No new third canonical kernel-path helper needed.** All 5 workflows
fit the P5b two-helper alphabet (AccelOpt-pattern for user-supplied
file; KernelAgent-pattern for workspace-local generated file).

---

## 4. Dependencies

### 4.1 Hard predecessors

- **P5a merged.** Manifest v1.1 schema in `_meta/manifests/schema.yaml`;
  `validate-workflow.js` warns on `supported_languages:` literal +
  CUDA tokens in clean-tier prompts; `validate-backend.{js,py}` L0
  green. Every Stage-A manifest emitted by P5c is validated by P5a's
  validator before commit.
- **P5b merged.** The 7-stage retrofit checklist proven on two
  workflows; harness helpers (`capturePrompts`,
  `print-workflow-prompts.js`, `schema-stub.js`, `run-workflow.js`) at
  `_meta/tools/`; the column-0 helper `workspaceKernelPath(args,
  driver)` documented in `_substrate/BACKEND-DRIVER-SDK.md`.

### 4.2 Soft predecessors

- P3 cuda + triton drivers L0 conformant (precondition of P5b, carries
  through).
- P4 AccelOpt pilot (pattern source).

### 4.3 Successors

- **P5d** depends on P5c (the lower-complexity batch reuses any spike
  outcomes; specifically the Astra/StitchCUDA intersectional-guard
  pattern is a candidate generalization).
- **P5e** matrix-smoke depends on P5b minimum; P5c's 5 retrofitted
  matrix-eligible workflows further enrich the matrix-smoke coverage.

### 4.4 Inside P5c: per-workflow ordering

Workflows within P5c are **independent** — no inter-workflow file
dependency. The recommended execution order (§6) is by complexity, not
dependency.

---

## 5. Task breakdown (per-workflow, A→B→C stages)

Each workflow ships **9 tasks** (5 workflows × 9 = **45 tasks** total).
Tasks A1–A4 are Stage A (fixtures + frozen golden + byte-identity
test); B1–B3 are Stage B (workflow `.js` retrofit, additive,
`backend_dir`-gated); C1–C2 are Stage C (test trio: guard +
cross-backend dry-run; byte-identity test was front-loaded in A4 so
Stage B can TDD against it).

**STARK has an extra task A0 — the seeded-RNG refactor — committed
BEFORE A1.** STARK total: 10 tasks; batch total **46 tasks**.

This section enumerates **WHICH seams** with file:line citations from
§2 spikes; per-seam edit detail is left to the execution agent per the
P5b convention.

### 5.1 KDA — 9 tasks, ~6 driver-gated seams

**Files touched:**
- **Create** `KDA/manifest.yaml` (v1.1).
- **Edit** `KDA/kda-kernel-workflow.js`.
- **Create** `_meta/tools/fixtures/kda-{args,agent-returns,golden,args-triton}.json`.
- **Create** `_meta/tools/fixtures/KDA-GOLDEN-BASELINE-SHA.txt`.
- **Create** `_meta/tools/test/kda-{byte-identity,guard,triton-dryrun}.test.js`.
- **Append** to `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md`.

**Task A1 — Manifest (v1.1) creation.** Author from scratch per
template; required keys: `schema_version: "1.1"`, `name`, `entrypoint`,
plus `backend:` block from §3 row 1. Validate with
`node _meta/tools/validate-workflow.js KDA/manifest.yaml` — green.
*(Mirrors P5b A1, with **create** instead of **edit**.)*

**Task A2 — Harness fixtures (pre-retrofit capture).** Create
`kda-args.json` (no `backend_dir`; pin `max_candidates: 2`,
`kernel_path: '/tmp/kda-fixture/baseline.cu'`); `kda-agent-returns.json`
with the **~9 keys** enumerated in c-1-2; cross-backend args
`kda-args-triton.json` (with `backend_dir:
_substrate/backends/triton`). Capture `kda-golden.json` via
`print-workflow-prompts.js`; commit
`KDA-GOLDEN-BASELINE-SHA.txt = git rev-parse HEAD`. *(Mirrors P5b A3.)*

**Task A3 — Cross-backend args fixture.** Created as part of A2 above
for parsimony; this slot is reserved to follow P5b's six-task Stage A
pattern. Optional: split A2 → A2 (legacy) + A3 (cross-backend) for
review readability.

**Task A4 — Byte-identity test (pre-Stage-B).** Create
`kda-byte-identity.test.js` that diffs legacy-path `capturePrompts`
output against `kda-golden.json`. **MUST be committed before any
Stage-B edit** so Stage B can TDD against it. *(Mirrors P5b C1; moved
to Stage A per the P5c TDD ordering — every Stage-B task in P5c re-runs
the byte-identity test green.)*

**Task B1 — `USE_DRIVER` scaffolding + §6.4 guard + load-driver agent
+ column-0 helpers.** Edit `KDA/kda-kernel-workflow.js`:

- Column-0 named consts for every legacy literal Stage B2/B3 will gate:
  `LEGACY_INSPECT_LANG_TOKEN = 'CUDA'`,
  `LEGACY_VALIDATE_COMMAND_HINT = 'ncu-report-skill'`, etc. (full list
  emerges during edit; ~6 consts to match the ~6 seams).
- New args: `backend_dir`, `backend` (alias `language` per §6.4),
  `kernel_filename` (default driven by `driver.source_ext`).
- Path helpers per BACKEND-DRIVER-SDK §6.1: `driverPath`, `driverSh`,
  `driverManifest` (cached), `driverIdioms` (cached).
- §6.4 guard resolution + conflict throws (4 cases: `backend_dir` set
  alone → derive `backend`; both set + mismatch → throw;
  `language ≠ backend` → throw; `backend_dir` unset + `backend` set →
  throw).
- Load-driver `agent()` call (only when `USE_DRIVER`); first agent in
  Setup; `JSON_PASSTHROUGH` schema; cached.
- Selects AccelOpt-pattern helper when `KERNEL_PATH = args.kernel_path`,
  KernelAgent-pattern when `KERNEL_PATH` was reassigned post-generation
  (per c-1-6).
- `USE_DRIVER = Boolean(args.backend_dir)`.

**Acceptance:** byte-identity test green (no observable rendered-string
change — all new consts referenced only in not-yet-added `USE_DRIVER`
branches).

**Task B2 — Wrap prompt-token seams.** Edit prompts at agent calls #2
(L214 Inspect), #3 (L271 Plan/draft), #4 (L339 Plan/exec), #5 (L403
Implement), #6 (L451 Validate) — gate the CUDA-specific tokens
(`cuda`, `ncu-report-skill`, code-fence `\`\`\`cuda`) on `USE_DRIVER`:

```js
const langToken = USE_DRIVER
  ? `${driverIdioms().lang_fence}`
  : LEGACY_INSPECT_LANG_TOKEN
```

Same pattern as P5b AdaExplore S1/S3/S4/S6. **4 prompt-token seams.**

**Acceptance:** legacy byte-identical; triton-dry-run (C2) zero CUDA
tokens.

**Task B3 — Wrap command seams + Layer-A envelope.** Edit:

- Agent call #1 (L177 Setup/Generation fallback): when `USE_DRIVER`,
  the generation prompt instructs writing to a workspace-local path
  with `driver.source_ext`; the generated path is wrapped via
  `workspaceKernelPath`.
- Agent call #6 (L451 Validate): the `test_command` / `benchmark_command`
  interpolation becomes `driverSh('run.sh', '--kernel',
  ${KERNEL_PATH}, …)` on the driver path; `driverSh('profile.sh', …)`
  for perf metrics.
- Assemble Layer-A evidence envelope on the driver path: `build → run →
  profile → to_evidence → diagnose → assemble`. Map driver
  `{compiled, correct, latency_ms, metrics}` onto KDA's
  `{validation_result, perf_metrics}` schema.
- `anti_cheat.py` invoked before scoring.

**2 command seams.** Mirrors P5b AdaExplore S2/S5.

**Acceptance:** legacy byte-identical; driver path produces valid
envelope under stubbed agent-returns.

**Task C1 — Guard unit test.** Create `kda-guard.test.js`; 4 subtests
matching P5b C2 verbatim.

**Task C2 — Triton dry-run test.** Create `kda-triton-dryrun.test.js`;
assert zero `nvcc`, `ncu`, `__global__`, `PYBIND11_MODULE`,
`cuda_runtime\.h`, `NCU Profile Results`, `\`\`\`cuda` tokens; assert
rendered validate-command starts with
`bash _substrate/backends/triton/run.sh`. Append KDA row to
`DEFERRED-GPU-VERIFICATION.md`.

**Definition of Done (KDA):**
- All 9 tasks committed.
- `node --test _meta/tools/test/kda-*.test.js` green (3 files).
- Byte-identity invariant: with no `args.backend_dir`,
  `capturePrompts(KDA, kda-args.json, kda-agent-returns.json) ===
  kda-golden.json` (byte-for-byte).
- Manifest passes `validate-workflow.js` + `validate-backend.{js,py}`
  L0 (mainfest-scope subset).
- Substrate diff-guard green.

### 5.2 CUDALLM — 9 tasks, ~7 driver-gated seams

**Files touched:**
- **Edit** `CUDALLM/manifest.yaml` (add v1.1 header + backend block).
- **Edit** `CUDALLM/cudallm-fsr-kernel-generation.js`.
- **Create** `_meta/tools/fixtures/cudallm-{args,agent-returns,golden,args-triton}.json`.
- **Create** `_meta/tools/fixtures/CUDALLM-GOLDEN-BASELINE-SHA.txt`.
- **Create** `_meta/tools/test/cudallm-{byte-identity,guard,triton-dryrun}.test.js`.
- **Append** to `DEFERRED-GPU-VERIFICATION.md`.

**Task A1 — Manifest extension.** Add `schema_version: "1.1"` header
+ `backend:` block from §3 row 2 to existing `CUDALLM/manifest.yaml`.
Preserve all existing `source`/`workflow`/`method`/`topology`
sections verbatim.

**Task A2 — Harness fixtures.** Create the 4 fixture files. Pin
`iterations: 2`, `samples_per_feature_set: 2` to keep the 16-key
`agent-returns` manageable (largest in the batch — see c-2-2).

**Task A3 — Cross-backend args fixture.** As §5.1 A3 (split optional).

**Task A4 — Byte-identity test (pre-Stage-B).** As §5.1 A4 against
`cudallm-golden.json`.

**Task B1 — Scaffolding.** Same as §5.1 B1. **Caveat:** the
`feature_catalog` seam (agent call #2, L226) is a heavyweight
prompt-token seam — see B2 below. Out-of-scope deferred item (L3
tightening) flagged here.

**Task B2 — Wrap prompt-token seams.** Edit agent calls #1 (L182
Setup), #2 (L226 Catalog), #3 (L278 Test plan), #4 (L324 Selection),
#5 (L364 Generation), #6 (L403 Evaluation), #7 (L464 Reinforce). **6
prompt-token seams.**

- **#2 (Catalog) note:** the legacy prompt contains a 30+ line CUDA
  feature list ("shared memory tiling", "warp shuffle", "tensor core
  mma", "ldmatrix"). On the driver path, gate the entire list block on
  `USE_DRIVER ? (driverIdioms().feature_catalog ||
  LEGACY_TRITON_FEATURE_FALLBACK) : LEGACY_CUDA_FEATURE_CATALOG`.
  Today the triton driver has no `feature_catalog` idiom, so the
  fallback string is used; this is a **deferred L3-tightening item**,
  documented in the workflow header comment.

**Acceptance:** legacy byte-identical; triton-dry-run zero CUDA
tokens.

**Task B3 — Wrap command seams + Layer-A envelope.** Edit agent call
#5 (L364 Generation) to write to `cudallmCandidatePath(iter, sample,
driver)`; agent call #6 (L403 Evaluation) to invoke driver
`build/run/profile/to_evidence/diagnose` and map onto the
`{compiled, correct, latency_ms, log}` schema. **1 command seam +
filename seam.**

**Task C1 — Guard test.** As §5.1 C1.

**Task C2 — Triton dry-run.** As §5.1 C2.

**Definition of Done (CUDALLM):**
- All 9 tasks committed.
- 3 test files green.
- Byte-identity invariant: no `args.backend_dir` → matches
  `cudallm-golden.json` byte-for-byte across all 16 fixture keys.
- Manifest validation green.
- Substrate diff-guard green.
- L3 `feature_catalog` deferred item filed in workflow header comment +
  follow-up issue.

### 5.3 Astra — 9 tasks, ~8 driver-gated seams

**Files touched:**
- **Edit** `Astra/manifest.yaml` (v1.1 + backend block).
- **Edit** `Astra/astra-kernel-optimization.js`.
- **Create** `_meta/tools/fixtures/astra-{args,agent-returns,golden,args-triton}.json`.
- **Create** `_meta/tools/fixtures/ASTRA-GOLDEN-BASELINE-SHA.txt`.
- **Create** `_meta/tools/test/astra-{byte-identity,guard,triton-dryrun}.test.js`.
- **Append** to `DEFERRED-GPU-VERIFICATION.md`.

**Task A1 — Manifest extension.** §3 row 3. Pin
`integration_mode: 'standalone'` as the matrix-smoke default in a
comment for downstream P5e.

**Task A2 — Harness fixtures.** 14 keys per c-3-2; pin
`iterations: 2`, `integration_mode: 'standalone'`,
`kernel_path: '/tmp/astra-fixture/baseline.cu'`.

**Task A3 — Cross-backend args fixture.** Split optional.

**Task A4 — Byte-identity test.** As §5.1 A4 against
`astra-golden.json`.

**Task B1 — Scaffolding + Astra-specific intersectional guard.** Same
as §5.1 B1 PLUS: add the **§3 intersectional guard** —
`args.integration_mode === 'sglang' && USE_DRIVER &&
driver.backend_id !== 'cuda'` → throw with message naming both. Add
matching subtest to C1.

**Task B2 — Wrap prompt-token seams.** Edit agent calls #1 (L179
Setup/Generation), #2 (L218 Setup), #3 (L261 Testing), #4 (L308
Profiling), #5 (L354 Planning), #6 (L399 Coding), #8 (L499 Lesson). **7
prompt-token seams.**

**Task B3 — Wrap command seams + Layer-A envelope.** Edit:
- Agent call #4 (L308 Profiling) — baseline profile gathered via
  driver `profile.sh` + `to_evidence.py` when `USE_DRIVER`.
- Agent call #7 (L437 Eval) — full driver envelope. Map driver
  metrics onto Astra's `{compiled, correct, latency_ms, metrics{}}`
  schema. **2 command seams.**
- Agent call #9 (L529 post-process) — only edits applicable on the
  `integration_mode: 'standalone'` path; sglang path is throw-guarded
  in B1.

**Task C1 — Guard test (+ Astra-specific subtest).** As §5.1 C1, add a
**5th subtest:** `args.integration_mode = 'sglang' && backend_dir =
triton` → throws.

**Task C2 — Triton dry-run.** As §5.1 C2.

**Definition of Done (Astra):**
- All 9 tasks committed.
- 3 test files green; guard test has 5 subtests (1 extra for
  intersectional guard).
- Byte-identity invariant: no `args.backend_dir` → matches
  `astra-golden.json` byte-for-byte. Both modes (`standalone` and
  `sglang`) byte-identical on the legacy path; `sglang` mode is
  driver-path-incompatible (throws as designed).
- Manifest validation green.
- Substrate diff-guard green.

### 5.4 StitchCUDA — 9 tasks, ~5 driver-gated seams

**Files touched:**
- **Edit** `StitchCUDA/manifest.yaml`.
- **Edit** `StitchCUDA/stitchcuda-kernel-optimization.js`.
- **Create** `_meta/tools/fixtures/stitchcuda-{args,agent-returns,golden,args-triton}.json`.
- **Create** `_meta/tools/fixtures/STITCHCUDA-GOLDEN-BASELINE-SHA.txt`.
- **Create** `_meta/tools/test/stitchcuda-{byte-identity,guard,triton-dryrun}.test.js`.
- **Append** to `DEFERRED-GPU-VERIFICATION.md`.

**Task A1 — Manifest extension.** §3 row 4.

**Task A2 — Harness fixtures.** 9 keys per c-4-2; pin
`max_attempts: 2`; craft `verify.0` to return `replan_signal: true`
to exercise the conditional replan branch (L197 agent call #2).

**Task A3 — Cross-backend args.** Split optional.

**Task A4 — Byte-identity test.**

**Task B1 — Scaffolding + StitchCUDA-specific intersectional guard.**
Add the §3 intersectional guard: `args.kernelbench_config?.benchmark_suite`
user-set + `USE_DRIVER && driver.backend_id !== 'cuda'` → throw.

**Task B2 — Wrap prompt-token seams.** Edit agent calls #1 (L84
Setup — heavy CUDA tokens: `cuda_version`, `sm_80/89/90`, `nvcc`,
`load_inline`), #2 (L197 Replan), #3 (L263 Planner), #4 (L357 Coder
— includes `__global__` in code-fence guidance), #5 (L416 Verifier).
**5 prompt-token seams** (large per-seam due to L84 setup's heavy
content).

**Task B3 — Wrap command seams + Layer-A envelope.** Edit:
- Agent call #5 (L416 Verifier): replace the `pytest`/`load_inline`
  invocation with `driverSh('build.sh', …) → driverSh('run.sh', …) →
  driverSh('profile.sh', …) → to_evidence → diagnose → assemble`.
  Map onto verifier's `{compiled, correct, latency_ms, log,
  replan_signal}`. The `replan_signal` decision stays in workflow JS
  body (heuristic on consecutive failure counts). **1 command seam.**

**Task C1 — Guard test (+ StitchCUDA-specific subtest).** 5 subtests.

**Task C2 — Triton dry-run.**

**Definition of Done (StitchCUDA):**
- All 9 tasks committed.
- 3 test files green; guard test has 5 subtests.
- Byte-identity invariant: no `args.backend_dir` → matches
  `stitchcuda-golden.json` byte-for-byte. Both `replan_signal` paths
  (true / false) covered.
- Manifest validation green.
- Substrate diff-guard green.

### 5.5 STARK — 10 tasks (A0 + 9), ~7 driver-gated seams

**Extra task A0 — Seeded-RNG refactor (BLOCKER resolution).** See
c-5-3. Replace 5 `Math.random()` call sites in `selectNode()` (lines
212, 213, 217, 228, 231) with a deterministic xorshift32 PRNG seeded
from `args.rng_seed` when set. When `args.rng_seed === undefined ||
args.rng_seed === null`, fall through to native `Math.random()` —
**legacy byte-identical**. New optional arg
`rng_seed: number | null` documented in the workflow header.

```js
// Column-0 helper (top of file)
const RNG_SEED = args.rng_seed
let _rngState = (RNG_SEED != null) ? (RNG_SEED | 0) || 1 : null
function rng() {
  if (_rngState === null) return Math.random()
  // xorshift32
  let s = _rngState
  s ^= s << 13; s ^= s >>> 17; s ^= s << 5
  _rngState = s
  return ((s >>> 0) / 0x1_0000_0000)
}
```

Then every `Math.random()` call in `selectNode()` becomes `rng()`.
Commit A0 **before** A1 so the Stage-A golden capture (A2) can pin
`rng_seed: 42` without touching `Math.random()` semantics for legacy
callers.

**Files touched (A0):** `STARK/stark-kernel-optimization.js` only.

**Acceptance (A0):** existing tests still pass; a new ephemeral unit
test (kept or rolled into A4) asserts `rng()` is deterministic given a
seed and equals `Math.random()` semantics (in distribution sense — not
exact value) when unseeded.

**Files touched (A1–C2):**
- **Edit** `STARK/manifest.yaml`.
- **Edit** `STARK/stark-kernel-optimization.js` (continued — Stage B
  edits on top of A0).
- **Create** `_meta/tools/fixtures/stark-{args,agent-returns,golden,args-triton}.json`.
- **Create** `_meta/tools/fixtures/STARK-GOLDEN-BASELINE-SHA.txt`.
- **Create** `_meta/tools/test/stark-{byte-identity,guard,triton-dryrun}.test.js`.
- **Append** to `DEFERRED-GPU-VERIFICATION.md`.

**Task A1 — Manifest extension.** §3 row 5. Document the
`rng_seed` arg in the manifest's `args.optional` block.

**Task A2 — Harness fixtures.** ~10–11 keys per c-5-2; pin
`iterations: 2` (BUDGET=2), `rng_seed: 42`,
`kernel_path: '/tmp/stark-fixture/reference.cu'`. The fixture must
also pin the leaderboard/tree state implied by the agent-returns
(node IDs map deterministically given seeded RNG + fixture
`compiled/correct/runtime` returns).

**Task A3 — Cross-backend args.** Split optional.

**Task A4 — Byte-identity test.** Note: the test reads
`STARK-GOLDEN-BASELINE-SHA.txt` and asserts it matches `git
rev-parse HEAD` only loosely (provenance log, not gate); the actual
gate is the JSON diff against `stark-golden.json`.

**Task B1 — Scaffolding.** Same as §5.1 B1. **No intersectional
guard for STARK** (no vendor-bound mode args).

**Task B2 — Wrap prompt-token seams.** Edit agent calls #1 (L340
generation fallback), #2 (L376 Setup), #3 (L413 Root eval), #4 (L504
Debug), #5 (L542 Plan), #6 (L600 Code), #7 (L648 Eval). **7
prompt-token seams.** Special attention: the `buildPlanContext`,
`buildCodeContext`, `buildDebugContext` helpers (lines 263, 275, 280,
289, 293, 304, 313, 319) **emit code-fence strings** `\`\`\`cuda` ;
these helpers must accept a `lang_fence` arg (default `'cuda'`) and
the driver path passes `driverIdioms().lang_fence`. Edit those three
context builders as a single seam-group.

**Task B3 — Wrap command seams + Layer-A envelope.** Edit agent
calls #3 (L413 Root eval) and #7 (L648 Eval) — both invoke the
driver `build/run/profile/to_evidence/diagnose` pipeline; the
returned envelope maps onto STARK's per-node `{compiled, correct,
runtime, log}` schema. **2 command seams.** Note: STARK's tree
machinery (`tree`, `leaderboard`, `nodes`) consumes the envelope but
the data shape is preserved verbatim — no tree-machinery edits.

**Task C1 — Guard test.** As §5.1 C1.

**Task C2 — Triton dry-run.** As §5.1 C2 PLUS: assert the rendered
context-builder strings use `\`\`\`python` fence (driver
`lang_fence`) when `backend_dir = triton`, not `\`\`\`cuda`.

**Definition of Done (STARK):**
- All 10 tasks committed (A0 + 9).
- 3 test files green; byte-identity test pins `rng_seed: 42`.
- Byte-identity invariant: with no `args.backend_dir` AND
  `rng_seed: 42`, `capturePrompts` matches `stark-golden.json`
  byte-for-byte. **Without `rng_seed`**, legacy `Math.random()` is
  used and the rendered prompt set is non-deterministic — gate is
  N/A for that path.
- Manifest validation green; manifest documents `rng_seed`.
- Substrate diff-guard green.
- A0's xorshift32 helper is the ONLY net-new pure-JS function
  introduced; column-0 placement; commented.

---

## 6. Execution units / commit clusters

P5c proposes **one workflow per dispatch cluster** — each cluster is
an independent dev branch (`dev/p5c-<workflow>-*`) that merges into
`dev/solver-substrate`. Within a cluster, ordering is A1→A2→A3→A4 →
B1→B2→B3 → C1→C2 (STARK prepends A0).

### 6.1 Recommended dispatch order (simplest first)

Ordered by spike complexity (seam count + fixture key count + risk
flags):

1. **CUDALLM** — **PILOT** (simplest after P5b; no banned APIs; rich
   existing manifest reduces A1 risk; KernelAgent-workspace-local
   kernel-path pattern straight from P5b).
2. **StitchCUDA** — 5 seams (lowest seam count), 9 fixture keys
   (lowest fixture density), 1 intersectional guard.
3. **KDA** — 6 seams, 9 keys, manifest **create** (small extra step
   vs the others which **extend**).
4. **Astra** — 8 seams, 14 keys, 1 intersectional guard.
5. **STARK** — 7 seams, 10–11 keys, **+A0 seeded-RNG refactor
   blocker**, largest workflow (840 LoC).

### 6.2 Batching (review-readable PRs)

Per the parent plan §P5c hint "May ship as 2–3 sub-units":

- **P5c.1 — Simplest 3 (CUDALLM, StitchCUDA, KDA)** — single
  dispatch round; can be reviewed as one stacked PR or 3 sequential
  PRs against `dev/solver-substrate`. ~27 tasks total. Median
  per-workflow commit count: 9.
- **P5c.2 — Astra + STARK** — second dispatch round; ships after
  P5c.1 merges so any bug discovered in the first round is fixed
  before exposing the intersectional-guard pattern (Astra) and the
  A0 seeded-RNG refactor (STARK). ~19 tasks (9 + 10).

### 6.3 Commit count estimate

| Workflow | Tasks | Commits |
|---|---|---|
| CUDALLM | 9 | 9 |
| StitchCUDA | 9 | 9 |
| KDA | 9 | 9 |
| Astra | 9 | 9 |
| STARK | 10 | 10 |
| **Total** | **46** | **46** |

Within parent §P5c's 35–50 envelope.

### 6.4 Parallelism

Per-workflow clusters are independent (§4.4). Within a cluster, tasks
have a strict topological order (A0 → A1 → A2/A3 → A4 → B1 → B2 → B3
→ C1 → C2). Dispatching multiple workflows in parallel is **safe** —
no shared files except `DEFERRED-GPU-VERIFICATION.md` (append-only,
trivial merge) and the fixture/test directories (per-workflow file
prefixes prevent collision).

P5c.1 batch (3 workflows) can be dispatched as 3 parallel sub-agents
per `superpowers:dispatching-parallel-agents`. P5c.2 (Astra + STARK)
also parallel-safe.

---

## 7. Definition of Done

### 7.1 Per-workflow gates

| Workflow | Test count | Byte-identity invariant |
|---|---|---|
| KDA | 3 (byte-identity + guard + triton-dryrun) | `capturePrompts(KDA, kda-args.json [no backend_dir], kda-agent-returns.json) === kda-golden.json` byte-for-byte; 9 keys exercised. |
| CUDALLM | 3 | Same shape; **16 keys** exercised (largest fixture). |
| Astra | 3 (guard has +1 subtest = 5 total subtests) | Same shape; 14 keys; both `integration_mode` modes byte-identical on legacy. |
| StitchCUDA | 3 (guard has +1 subtest = 5 total subtests) | Same shape; 9 keys; both `replan_signal` true/false paths covered. |
| STARK | 3 (+ A0 unit test if not folded) | Same shape **conditional on `rng_seed: 42`**; 10–11 keys; legacy default `Math.random()` path is NOT byte-identity-gated (cannot be — see c-5-3). |

### 7.2 Manifest conformance (all 5)

Each `<workflow>/manifest.yaml`:

- `schema_version: "1.1"` header present.
- `backend:` block with `portability`, `matrix_eligible`,
  `method_supported_backends`, `supported`, `default` keys per §3
  table.
- Passes `node _meta/tools/validate-workflow.js
  <workflow>/manifest.yaml` (P5a's LLM-driven checklist; warns only
  on the now-correctly-widened `supported_languages` literal in the
  `.js` body — verified gone after Stage B1).
- Passes `validate-backend.{js,py}` L0 manifest-scope subset (the
  full L0 also checks substrate scripts; that's a P3 concern, not
  P5c's).

### 7.3 Cross-cutting gates

- `node --test _meta/tools/test/*.test.js` green end-to-end across
  the full repo (no regression of any P3/P4/P5a/P5b test).
- `git diff --stat _substrate/` shows **zero** changes touched by
  P5c (substrate diff-guard).
- Triton dry-run zero CUDA-token leak for all 5 workflows (5 ×
  `triton-dryrun.test.js`).
- Guard tests: 4 baseline subtests per workflow; +1 intersectional
  subtest for Astra + StitchCUDA.
- `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md` has 5 new rows
  (one per workflow).
- All 46 task commits have `Co-Authored-By` trailer per repo
  convention.

### 7.4 What P5c does NOT verify

- End-to-end driver-path execution (build → run → profile →
  to_evidence → diagnose → assemble) on actual GPU — deferred via
  `DEFERRED-GPU-VERIFICATION.md`.
- Cross-backend prompt-leak beyond the 7 enumerated regex tokens
  per workflow — deferred to P5e CI audit (per P5b Risk-3).
- `idioms.methods.<name>` L3 method-coverage tightening (CUDALLM
  `feature_catalog`, P5b reviser-hints) — explicit deferred item.
- Matrix-smoke CI wiring — P5e.
- Documentation updates referencing the 5 new retrofitted workflows
  — P5f.

---

## 8. Open risks + mitigations

### R1 — STARK `Math.random()` blocker (Spike c-5-3)

**Severity:** HIGH — without resolution, STARK cannot ship in P5c
under a byte-identity gate.

**Status:** Option A (seeded RNG, new optional `args.rng_seed`)
proposed in §2.5 + §5.5 task A0. Option B (per-label-shape gate)
and Option C (mock harness) are documented as escape hatches.

**Mitigation:** Land Option A as STARK task A0 BEFORE any Stage-A
fixture work. The refactor is ~30 LoC, legacy byte-preserving
(when `args.rng_seed` is absent/null, native `Math.random()` is
used unchanged), and useful beyond P5c. If Option A is rejected by
the execution agent, this plan's §5.5 needs amendment before
dispatch.

### R2 — CUDALLM `feature_catalog` driver-side absence

**Severity:** MEDIUM — the driver path uses a generic fallback
prompt for the feature catalog, which weakens triton-path output
quality.

**Mitigation:** Document the gap in the workflow header comment;
file as a P5e/L3 follow-up issue. Byte-identity gate is unaffected
(legacy path unchanged). Triton dry-run still passes (no CUDA
tokens leak; fallback text is generic).

### R3 — Astra `integration_mode: 'sglang'` is vendor-locked at the mode level

**Severity:** LOW — intersectional guard throws cleanly with a clear
message; the §6.4 guard pattern (P5b) extends naturally.

**Mitigation:** §5.3 Task B1 adds the guard; §5.3 Task C1 adds the
5th subtest. The pattern is documented as a generalizable
"intersectional guard" for downstream P5d (KernelBand's φ-gate
threshold may need similar treatment).

### R4 — StitchCUDA `kernelbench_config.benchmark_suite` is vendor-locked

**Severity:** LOW — same shape as R3, same mitigation; isolated to
the StitchCUDA workflow.

### R5 — Hidden vendor-isms in large prompts (P5b Risk-3 carryover)

**Severity:** MEDIUM — STARK's 8 agent calls + 3 context builders
contain hundreds of lines of prose; subtle CUDA-isms ("warp",
"shared memory", "shfl_sync") may slip past the 7-token blacklist.

**Mitigation:** Extend each workflow's `triton-dryrun.test.js`
blacklist iteratively as failures are caught downstream (same
pattern as P5b). Add `// VENDOR-LANG-LEAK-AUDIT` comments near
each prompt template in Stage B2 for future-grep traceability.
Accept residual risk — clean-tier *topology* retrofits, not
zero-leak vendor migrations.

### R6 — Astra agent-returns map size (14 keys) is brittle

**Severity:** LOW–MEDIUM — same shape as P5b Risk-2; the
mismatch-reporter from P5b C1 step 2 (first divergent slice
snippet) surfaces typos on first run.

**Mitigation:** §5.3 Task A2 enumerates all 14 keys explicitly;
review the agent-returns JSON in PR carefully. Fallback: shrink
`iterations` to 1 (key count drops to 9–10), accept reduced
coverage.

### R7 — Manifest "extend vs create" inconsistency

**Severity:** LOW — KDA creates from scratch; 4 others extend
existing. A subtle bug class: the 4 existing manifests already
have rich `source`/`workflow`/`method`/`topology` sections; P5c's
extension must NOT delete or reorder them.

**Mitigation:** Stage A1 for the 4 EXTEND workflows uses pure
**Edit** (add `schema_version` header at top + `backend:` block at
bottom or per-schema convention); no structural rewrite. KDA's
A1 follows the v1.1 template verbatim with no legacy section to
preserve.

### R8 — Intersectional guard pattern not yet in SDK doc

**Severity:** LOW — new pattern introduced by Astra + StitchCUDA;
not in `_substrate/BACKEND-DRIVER-SDK.md` after P5b.

**Mitigation:** P5f documentation pass picks this up. P5c flags
the pattern in a workflow-header comment and in this plan §3 +
§9 for visibility. NO SDK edits in P5c (out of scope per §1.3).

---

## 9. Pattern divergences from P5b

P5c introduces **two new patterns** not present in P5b. Both are
isolated additions, not breakage:

### 9.1 Intersectional guards (Astra + StitchCUDA)

**Pattern:** when a workflow has a mode arg that binds it to a
specific vendor (Astra `integration_mode: 'sglang'`, StitchCUDA
`kernelbench_config.benchmark_suite`), the §6.4 guard is extended
with an intersectional check:

```js
if (USE_DRIVER && args.integration_mode === 'sglang' &&
    driverManifest().backend_id !== 'cuda') {
  throw new Error(
    `Astra integration_mode='sglang' requires backend_dir to be a CUDA driver; ` +
    `got backend_dir=${args.backend_dir} (backend_id=${driverManifest().backend_id})`
  )
}
```

**Generality:** P5d's KernelBand φ-gate threshold and any future
vendor-bound mode arg can adopt this pattern. Candidate for
generalization into a column-0 helper
`assertVendorBoundModeCompatible(modeArg, allowedBackends)` in P5d
or P5f. **NOT generalized in P5c** — only 2 instances; helper would
be premature abstraction.

### 9.2 Pre-Stage-A determinism refactor (STARK A0)

**Pattern:** when a clean-tier workflow has banned APIs
(`Math.random()`/`Date.now()`/etc) that would break byte-identity,
prepend a Stage A0 task that introduces a seeded-RNG (or
fixed-clock) arg and replaces the banned API call sites. **Legacy
behavior preserved** when the new arg is unset.

**Generality:** P5d/P5e may discover further banned-API hits;
adopt A0 as the canonical pre-Stage-A remediation. **NOT
generalized into a checklist item in P5c** — only 1 instance;
document in this plan for forward reference. Candidate for SDK doc
in P5f.

### 9.3 No new kernel-path helper

Confirmed in §3 + §2: all 5 P5c workflows fit the P5b
two-helper alphabet (AccelOpt-pattern + KernelAgent-pattern).
**No third canonical helper introduced.** This is a
positive-confirmation outcome of the spike phase — the P5b
hardening generalizes cleanly to the mid-complexity batch.

### 9.4 No vendor-locked demotion

All 5 spikes confirmed **clean** tier despite suggestive names
(StitchCUDA, CUDALLM). The CUDA-flavoring lives in prompt text +
`WORKFLOW_SUITABILITY` literals, both swappable via driver
idioms. **No spec §7.2 amendment needed** for P5c.

---

## 10. Estimated commit count

| Stage | Per-workflow | Across 5 (STARK has +A0) |
|---|---|---|
| A (A0?+A1+A2+A3+A4) | 4 (KDA/CUDALLM/Astra/StitchCUDA) / 5 (STARK) | 21 |
| B (B1+B2+B3) | 3 | 15 |
| C (C1+C2) | 2 | 10 |
| **Total** | **9 / 10** | **46** |

Within parent §P5c's 35–50 envelope.

---

## 11. Recommendation

**Proceed with P5c as planned, with explicit STARK A0 (Option A
seeded-RNG) decision recorded BEFORE dispatch.** The four other
workflows (KDA, CUDALLM, Astra, StitchCUDA) are clean-API and
follow the P5b template with two well-bounded new patterns
(intersectional guard for Astra + StitchCUDA; no other novelties).

**Pilot recommendation: dispatch CUDALLM first.** Rationale:

- No banned APIs (clean Spike c-2-3).
- Existing rich manifest (lower A1 risk than KDA's create-from-scratch).
- Pure KernelAgent-workspace-local kernel-path (no hybrid logic).
- No intersectional guard (simpler than Astra + StitchCUDA).
- Largest fixture (16 keys, c-2-2) — if the fixture machinery
  scales here, the rest are easier.
- The `feature_catalog` deferred item exercises the L3-tightening
  flagging discipline established by P5b, surfacing the
  intersectional-guard discussion (Astra/StitchCUDA) for the
  second batch with maximum context.

**Sequencing recommendation:** P5c.1 = CUDALLM → StitchCUDA → KDA
(dispatch as 3 parallel sub-agents per
`superpowers:dispatching-parallel-agents`; merge in any order);
THEN P5c.2 = Astra + STARK (parallel, after STARK A0 lands as a
separate prep commit on `dev/solver-substrate`).

**Out of scope (deferred, summarized):**

- L3 `idioms.methods.<name>` tightening (CUDALLM `feature_catalog`;
  P5b reviser-hints) — P5e or P5f follow-up issue.
- Intersectional-guard helper generalization — P5d candidate.
- Pre-Stage-A determinism refactor as SDK-doc checklist item — P5f.
- Matrix-smoke wiring — P5e.
- Docs pass — P5f.
- GPU verification — `DEFERRED-GPU-VERIFICATION.md`.
- KSearch tier reclassification (per P5 master plan) — P5d spike.

---

## Appendix A — Spike-to-task mapping

| Spike | Drives task(s) |
|---|---|
| c-N-1 (agent inventory) | B2, B3 (per-seam edit targets) |
| c-N-2 (fixture keys) | A2 (key enumeration), A4 (gate scope) |
| c-N-3 (banned APIs) | A0 (STARK only); else A4 gate viability |
| c-N-4 (manifest status) | A1 (create vs extend) |
| c-N-5 (tier confirm) | A1 (backend block contents), B1 (guard behavior), C1 (guard tests) |
| c-N-6 (kernel-path pattern) | B1 (helper selection), B3 (command seam shape) |

## Appendix B — File touch summary

```
Modified workflow files (5):
  KDA/kda-kernel-workflow.js                          # +scaffolding +6 seam edits
  CUDALLM/cudallm-fsr-kernel-generation.js            # +scaffolding +7 seam edits
  Astra/astra-kernel-optimization.js                  # +scaffolding +8 seam edits +intersectional guard
  StitchCUDA/stitchcuda-kernel-optimization.js        # +scaffolding +5 seam edits +intersectional guard
  STARK/stark-kernel-optimization.js                  # A0 seeded RNG +scaffolding +7 seam edits

Manifest files (1 create, 4 extend):
  KDA/manifest.yaml                                   # CREATE
  CUDALLM/manifest.yaml                               # extend v1.0 → v1.1
  Astra/manifest.yaml                                 # extend v1.0 → v1.1
  StitchCUDA/manifest.yaml                            # extend v1.0 → v1.1
  STARK/manifest.yaml                                 # extend v1.0 → v1.1

Fixtures (4 per workflow × 5 = 20 JSON + 5 SHA = 25 files):
  _meta/tools/fixtures/kda-{args,agent-returns,golden,args-triton}.json
  _meta/tools/fixtures/KDA-GOLDEN-BASELINE-SHA.txt
  …(same shape × CUDALLM, Astra, StitchCUDA, STARK)…

Tests (3 per workflow × 5 = 15 test files):
  _meta/tools/test/kda-{byte-identity,guard,triton-dryrun}.test.js
  …(same shape × CUDALLM, Astra, StitchCUDA, STARK)…

Appends (1 file):
  _meta/tools/test/DEFERRED-GPU-VERIFICATION.md      # +5 rows

NO edits under:
  _substrate/**                                      # diff-guard sacred
  _meta/tools/{schema-stub,run-workflow,print-workflow-prompts,validate-workflow,validate-backend,generate-workflow}.js
  _templates/**, _meta/templates/**
  AdaExplore/**, KernelAgent/**, AccelOpt/**         # P4/P5b territory
```

Total net file ops: ~50 (5 workflow edits + 5 manifests + 25
fixtures + 15 tests + 1 append).




