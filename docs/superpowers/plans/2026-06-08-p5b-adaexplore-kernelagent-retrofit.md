# P5b — AdaExplore + KernelAgent retrofit (Phase 2, first wave)

> **Branch:** `dev/p5b-plan` for the plan; implementation lands on dev branches
> (`dev/p5b-adaexplore-*`, `dev/p5b-kernelagent-*`) and merges into
> `dev/solver-substrate`.
>
> **Parent:** `docs/superpowers/plans/2026-06-08-p5-clean-workflow-migration.md`
> §P5b. Pattern source: `docs/superpowers/plans/2026-06-08-accelopt-driver-pilot.md`
> (P4 — the executed pilot). Schema source: `docs/superpowers/plans/2026-06-08-p5a-schema-and-generator.md`
> (manifest v1.1 + `validate-workflow.js`).
>
> **Scope summary:** Generalize the AccelOpt P4 7-stage retrofit checklist and
> apply it to two workflows — `AdaExplore/adaexplore-kernel-optimization.js`
> (MCTS topology, CUDA-flavored today) and
> `KernelAgent/kernelagent-triton-synthesis.js` (routing+parallel-seeds topology,
> Triton-flavored today). Both are §7.2 **clean tier** per the parent plan.
>
> **Pre-retrofit spike findings (already completed):** see §1 below — six spikes
> resolved on `dev/p5b-plan` before this plan was written.

---

## 1. Pre-retrofit spike resolutions

The parent plan §P5b "Open questions" lists two spikes. Investigation of the
two workflow `.js` files plus the existing `_substrate/backends/{cuda,triton}/`
drivers and the P4 AccelOpt retrofit produced six concrete resolutions. They
gate the Stage-B retrofit edits below.

### Spike #1 — Canonical `args.kernel_path` discipline

**Question (parent §P5b OQ1):** Do AdaExplore and KernelAgent use
`args.kernel_path` like AccelOpt, or do they expose method-specific
source-input arg names that break `driverSh('build.sh', --source ${...})`?

**Resolution — AdaExplore:** Source input is **not** a single arg. The
workflow materializes an *operator spec* via the Setup `agent()` call (line
~373 of `adaexplore-kernel-optimization.js`); the kernel under evaluation is a
candidate node body produced inside the MCTS rollout, not a path passed in.
The user-facing input is `args.problem_spec` (operator description) plus
`args.evaluator_command` (the eval harness). For driver dispatch we therefore
introduce **a generated workspace-local path** (`workspace/candidate_<node_id>.py`)
that the driver `build.sh` consumes; `args.kernel_path` is **not** added to
AdaExplore's arg schema. Driver dispatch wraps the existing `evaluator_command`
seam (~line 594), not a source path.

**Resolution — KernelAgent:** Source input is **also not** a single arg. Per
line ~134 (`let testCode = ''`) and the Setup phase (~line 270), the workflow
*generates* both the kernel and the test harness from `args.problem_spec`. The
user-provided input arg is `args.problem_spec` plus optional
`args.test_command`. Like AdaExplore, driver dispatch consumes the generated
kernel file at a workspace path (`workspace/kernel.py` per line ~529 — the
file the agent is instructed to write); the seam to wrap is the **test
execution** call (~line 515 / 721).

**Implication for both:** The P4 `driverSh('build.sh', --source ${args.kernel_path})`
pattern does **not** apply unchanged. Both retrofits use
`driverSh('build.sh', --source ${workspaceKernelPath})` where
`workspaceKernelPath` is a column-0 const computed from
`${args.workspace}/${KERNEL_FILENAME}`. This is the **one** structural
deviation from the AccelOpt template; everything else (USE_DRIVER gate, guard
agent, JSON_PASSTHROUGH, Layer-A envelope) is identical.

### Spike #2 — MCTS / parallel-seeds determinism

**Question (parent §P5b OQ2):** Does AdaExplore's MCTS rollout use
`Math.random`/`Date.now` that would break the byte-identity gate?

**Resolution — AdaExplore:** Grep of `adaexplore-kernel-optimization.js`
shows the body uses **no `Math.random`, no `Date.now`, no `performance.now`**
in seam-rendered prompts. The MCTS state (UCB1 select, expand, score) is
**deterministic** given the agent-returns map: `ucb1Score`, `expandUcb1Score`,
`scoreTuple`, `avgReward`, `blendedReward` are pure functions of node stats,
and node stats come from `agent()` returns the harness controls. The
`selectNode()` function deterministically picks the max-UCB node; ties broken
by insertion order (`nodes.push`). **Byte-identity gate is viable.**

**Resolution — KernelAgent:** Same audit — no `Math.random`/`Date.now` in the
body. The "parallel seeds" topology is sequential `agent()` calls in the
JS body (the parallelism is logical, expressed in prompts to the agent, not in
JS-level concurrency). **Byte-identity gate is viable.**

**Caveat — agent-returns map must unlock the full loop.** For AdaExplore this
means the `agentReturns` fixture has to supply: Setup result with
`evaluator_command` + materialized `operator_code`; per-iteration Select
results (node id), Expand results (new kernel body), Evaluate results (reward
triple `{compiled, correct, speedup}`), Backprop is body-pure (no `agent()`),
and optional AdaptMemory results. The fixture per-iteration must therefore
key on the (phase, iteration_index) tuple, identical to how AccelOpt's P4
fixture keys on `(phase, beam_index)`.

### Spike #3 — KernelAgent backend posture (clean vs method-intrinsic)

**Question (parent §P5b table row 2):** Is the *method* (routing + parallel
seeds + test-harness verification) Triton-only, or is it backend-agnostic and
only the prompts are Triton-flavored?

**Resolution:** **Clean tier confirmed.** The topology is
language-agnostic. The body has no Triton-specific control flow (no
`@triton.jit` parsing, no Triton compile checks in JS); the Triton-ness lives
in the prompt text (`"You are a Triton kernel..."` at lines 270, 515, 721)
and in the `.py` file the agent is instructed to write. Swapping the driver
swaps the lang_fence (Triton→CUDA `__global__`) and the build/run/profile
shell, but the routing-and-parallel-seeds method survives. Manifest emits
`portability: clean`, `supported: [cuda, triton]`, `default: triton`
(preserving today's behavior).

### Spike #4 — Backend manifest defaults

**Question (implicit; parent §P5b exit criterion 4):** What
`supported`/`default` does each manifest declare under P5a schema v1.1?

**Resolution:**

| Workflow | `portability` | `matrix_eligible` | `supported` | `default` | `method_supported_backends` |
|---|---|---|---|---|---|
| AdaExplore | `clean` | `true` | `[cuda, triton]` | `cuda` (preserves today) | `any` |
| KernelAgent | `clean` | `true` | `[cuda, triton]` | `triton` (preserves today) | `any` |

AdaExplore is the first matrix_eligible workflow listed in P5e's matrix smoke
test — gating it on `triton` dry-run as part of P5b is what unlocks P5e's
exit criterion.

### Spike #5 — Number of backend-laden seams per workflow

**Question (implicit; sizing the Stage-B task split):** How many seams need
`USE_DRIVER ? <driver> : <legacy>` gating?

**Resolution (AdaExplore — 7 seams):**

| # | Line ~ | Seam | Group |
|---|---|---|---|
| 1 | 354 | Setup prompt — evaluator contract language ("Triton kernel" → backend lang_fence) | prompt-token |
| 2 | 373 | Setup result — `evaluator_command` template (CUDA `nvcc`/Triton `python -m triton`) | command |
| 3 | 486 | Large-step prompt — operator interface preservation language | prompt-token |
| 4 | 548 | Reviser suggestions list — backend-specific perf hints | prompt-token |
| 5 | 594 | Evaluate phase — `evaluatorCommand = renderCommand(...)` (THE driver entry point) | command |
| 6 | 607 | Evaluate prompt — "run evaluator command" instructions | prompt-token |
| 7 | 780 | Report epilog — "did not call AdaExplore repository" boilerplate (no backend tokens; **no edit needed**) | none |

Net: **6 driver-gated seams** (S1–S6). Stage-B splits as 3 tasks: prompt-token
(S1/3/4/6), command (S2/5), guard+helpers+manifest+args (the always-emit
scaffolding).

**Resolution (KernelAgent — 9 seams):**

| # | Line ~ | Seam | Group |
|---|---|---|---|
| 1 | 252 | Routing/complexity-analysis prompt — `"Triton kernel"` lang token | prompt-token |
| 2 | 270 | Test-harness gen prompt — `"Triton kernel test engineer"` + Triton fence | prompt-token |
| 3 | 347 | Per-seed synthesis prompt — `"Generate Triton kernel"` | prompt-token |
| 4 | 383 | Seed-aggregator prompt — `"select best Triton kernel"` | prompt-token |
| 5 | 515 | Verification call — `"Execute the test harness"` + run command | command + prompt-token |
| 6 | 529 | Test-harness write instruction — `test_kernel.py` filename | filename |
| 7 | 530 | Test-harness execute — user-provided test_command or generated | command |
| 8 | 721 | Refinement verification — second test-harness call | command + prompt-token |
| 9 | 305 | Log message — "Test harness generated" (informational, **no edit**) | none |

Net: **8 driver-gated seams** (S1–S8). Stage-B splits as 4 tasks: prompt-token
(S1/2/3/4/5p/8p), command (S5c/7/8c), filename (S6 — driver-supplied via
`manifest.source_ext`/`aux_ext`), guard+scaffolding.

### Spike #6 — Driver `idioms.json` coverage of the methods these workflows invoke

**Question (implicit; L3 of BACKEND-DRIVER-SDK):** Do the existing
`_substrate/backends/{cuda,triton}/idioms.json` files cover every
`method_gate.TABLE` method these workflows trigger?

**Resolution:** Both workflows invoke methods at the **topology layer**
(MCTS, routing+seeds), not the kernel-rewrite layer where `method_gate.TABLE`
entries live. Stage-B's `USE_DRIVER` gate consumes only:
- `manifest.compiler.invoke` (`build.sh`)
- `manifest.runner.invoke` (`run.sh`)
- `manifest.profiler.invoke` (`profile.sh`)
- `manifest.profiler.to_evidence`
- `idioms.lang_fence`
- `idioms.impl_requirements`
- `manifest.source_ext` / `manifest.aux_ext` (for spike #5 row 6)

All seven keys are present in both `cuda/manifest.json`+`cuda/idioms.json` and
`triton/manifest.json`+`triton/idioms.json` (verified — see
`_substrate/backends/REGISTRY.md`). **No driver edits required by P5b.** Any
method-specific `idioms.methods.<name>` lookups (e.g. for kernel-rewrite
sub-methods in P5c) are out of scope for this sub-plan.

---

## 2. Per-workflow tier classification (locked in this sub-plan)

| Workflow | Tier (§7.2) | `portability` | `matrix_eligible` | `method_supported_backends` | `supported` | `default` | Topology |
|---|---|---|---|---|---|---|---|
| AdaExplore | row 1 (clean) | `clean` | `true` | `any` | `[cuda, triton]` | `cuda` | MCTS + failure-driven skill memory |
| KernelAgent | row 1 (clean) | `clean` | `true` | `any` | `[cuda, triton]` | `triton` | Routing + parallel seeds + verification |

Both are matrix_eligible. **AdaExplore is the one P5e wires into matrix smoke**
(parent plan §P5e exit criterion 1); KernelAgent's matrix-eligibility is
prepared but not exercised by CI until a later P-step.

---

## 3. File structure

| Path | Stage | Action |
|---|---|---|
| `AdaExplore/adaexplore-kernel-optimization.js` | B | Edit — additive, `backend_dir`-gated retrofit (6 seams) |
| `AdaExplore/manifest.yaml` | A0 | Create — v1.1 schema with backend block |
| `KernelAgent/kernelagent-triton-synthesis.js` | B | Edit — additive, `backend_dir`-gated retrofit (8 seams) |
| `KernelAgent/manifest.yaml` | A0 | Edit — add v1.1 backend block (file exists today) |
| `_meta/tools/fixtures/adaexplore-args.json` | A | Create — pinned args (no `backend_dir`) |
| `_meta/tools/fixtures/adaexplore-agent-returns.json` | A | Create — deterministic per-(phase, iter) returns |
| `_meta/tools/fixtures/adaexplore-golden.json` | A | Create — pre-retrofit prompt capture |
| `_meta/tools/fixtures/ADAEXPLORE-GOLDEN-BASELINE-SHA.txt` | A | Create — repo SHA the golden was captured at |
| `_meta/tools/fixtures/adaexplore-args-triton.json` | A | Create — triton dry-run args (with `backend_dir`) |
| `_meta/tools/fixtures/kernelagent-args.json` | A | Create — pinned args, no `backend_dir` (today's path = triton) |
| `_meta/tools/fixtures/kernelagent-agent-returns.json` | A | Create |
| `_meta/tools/fixtures/kernelagent-golden.json` | A | Create — pre-retrofit prompts |
| `_meta/tools/fixtures/KERNELAGENT-GOLDEN-BASELINE-SHA.txt` | A | Create |
| `_meta/tools/fixtures/kernelagent-args-cuda.json` | A | Create — cuda dry-run args |
| `_meta/tools/test/adaexplore-byte-identity.test.js` | C | Create — diff legacy-path against `adaexplore-golden.json` |
| `_meta/tools/test/adaexplore-guard.test.js` | C | Create — §6.4 resolution + conflict-throw |
| `_meta/tools/test/adaexplore-triton-dryrun.test.js` | C | Create — assert no CUDA tokens leak |
| `_meta/tools/test/kernelagent-byte-identity.test.js` | C | Create — diff legacy-path against `kernelagent-golden.json` |
| `_meta/tools/test/kernelagent-guard.test.js` | C | Create |
| `_meta/tools/test/kernelagent-cuda-dryrun.test.js` | C | Create — KernelAgent's "dry-run other backend" is CUDA, not Triton |
| `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md` | C | Edit — append both workflows |

**No edits** to `_substrate/**` (substrate diff-guard must stay green —
exit criterion 5). **No edits** to `_meta/tools/{schema-stub,run-workflow,print-workflow-prompts,validate-workflow}.js`
(those are P5a-owned).

---

## 4. Testability note (node-renderable vs GPU-deferred)

Every Stage-B prompt change is verified by `node --test _meta/tools/test/*.test.js`
on macOS with no GPU. The end-to-end driver path (`build.sh` → `run.sh` →
`profile.sh` → `to_evidence.py` → `diagnose.py` → Layer-A envelope) is
**GPU-deferred** for both workflows — append to
`_meta/tools/test/DEFERRED-GPU-VERIFICATION.md` per Stage C Task 14. Layer-A
envelope assembly **is** unit-tested on the no-GPU path by stubbing
`diagnose.py` output in the agent-returns map (matching the P4 pattern at
AccelOpt Task 11).

---

# Stage A — Harness fixtures + pre-retrofit goldens

> Mirrors P4 Stage A. The goldens MUST be captured from **TODAY's pre-retrofit**
> `.js` bodies and committed BEFORE any Stage-B edit. The
> `*-GOLDEN-BASELINE-SHA.txt` file records the tree the golden was captured at;
> Stage C Task 1's byte-identity test greps it for human-readable provenance.

## Task A1 — AdaExplore manifest.yaml (v1.1 schema)

### Files
- **Create** `AdaExplore/manifest.yaml`.

### Steps
1. Author manifest per P5a v1.1 schema. Required fields:
   ```yaml
   schema_version: "1.1"
   name: adaexplore-kernel-optimization
   entrypoint: adaexplore-kernel-optimization.js
   backend:
     portability: clean
     matrix_eligible: true
     method_supported_backends: any
     supported: [cuda, triton]
     default: cuda
   ```
2. Run `node _meta/tools/validate-workflow.js AdaExplore/manifest.yaml`;
   expect green (P5a Task 7 must already pass).

### Acceptance
`node --test _meta/tools/test/validate-workflow.test.js` still green; no
P5a-checklist warnings emitted for the new manifest.

## Task A2 — KernelAgent manifest.yaml (extend existing)

### Files
- **Edit** `KernelAgent/manifest.yaml` (file exists today; add `backend:` block).

### Steps
1. Add the `backend:` block — `portability: clean`,
   `matrix_eligible: true`, `method_supported_backends: any`,
   `supported: [cuda, triton]`, `default: triton`.
2. Bump `schema_version` to `"1.1"`.
3. Validate as A1.

### Acceptance
Same as A1.

## Task A3 — AdaExplore harness fixtures (pre-retrofit capture)

### Files
- **Create** `_meta/tools/fixtures/adaexplore-args.json`.
- **Create** `_meta/tools/fixtures/adaexplore-agent-returns.json`.
- **Create** `_meta/tools/fixtures/adaexplore-golden.json`.
- **Create** `_meta/tools/fixtures/ADAEXPLORE-GOLDEN-BASELINE-SHA.txt`.

### Steps
1. **`adaexplore-args.json`** — pinned args, **no `backend_dir`**:
   ```json
   {
     "problem_spec": "elementwise_relu(x: Tensor) -> Tensor",
     "evaluator_command": "python eval.py --kernel {kernel_path}",
     "workspace": "/tmp/adaexplore-fixture",
     "language": "triton",
     "max_iterations": 2,
     "exploration_constant": 1.4,
     "skill_memory_path": "/tmp/adaexplore-fixture/memory.txt",
     "memory_update": false
   }
   ```
   Set `max_iterations: 2` to keep the golden small but exercise both
   large-step and small-step expand paths.
2. **`adaexplore-agent-returns.json`** — keyed on
   `(phase_label, iteration_index)`. Required keys:
   - `setup` → `{operator_code, evaluator_command, workspace_files}`
   - `select.0`, `select.1` → `{selected_node_id, action: "expand_large" | "expand_small"}`
   - `expand_large.0` → `{node_id, kernel_body, plan_title}`
   - `expand_small.1` → `{node_id, kernel_body, suggestions}`
   - `evaluate.0`, `evaluate.1` → `{compiled, correct, speedup, latency_ms, log}`
   - `report` → `{best_node_id, summary}`
3. **`adaexplore-golden.json`** — produced by running
   `node _meta/tools/print-workflow-prompts.js AdaExplore/adaexplore-kernel-optimization.js \
     --args _meta/tools/fixtures/adaexplore-args.json \
     --agent-returns _meta/tools/fixtures/adaexplore-agent-returns.json \
     --json > _meta/tools/fixtures/adaexplore-golden.json`.
   Verify it contains every `agent()` prompt string + every `bash()` argv +
   every `log()` line, in order.
4. **`ADAEXPLORE-GOLDEN-BASELINE-SHA.txt`** — `git rev-parse HEAD` of the
   tree the golden was captured against; commit alongside the golden.

### Acceptance
The four files exist; `adaexplore-golden.json` is reproducible (re-running
the capture against the same SHA + same fixture yields byte-identical JSON).

## Task A4 — AdaExplore triton dry-run args fixture

### Files
- **Create** `_meta/tools/fixtures/adaexplore-args-triton.json`.

### Steps
Same as A3 step 1, but **with** `"backend_dir": "_substrate/backends/triton"`
and `"backend": "triton"`. This fixture feeds the Stage-C triton-dry-run test
once Stage-B lands (no golden — the test asserts negative-token invariants,
not byte-identity).

### Acceptance
File exists; valid JSON; no `backend_dir` collision with `args.backend` per
§6.4 resolution rules (resolution: `backend_dir` wins, `args.backend`
auto-derived from manifest).

## Task A5 — KernelAgent harness fixtures (pre-retrofit capture)

### Files
- **Create** `_meta/tools/fixtures/kernelagent-args.json`.
- **Create** `_meta/tools/fixtures/kernelagent-agent-returns.json`.
- **Create** `_meta/tools/fixtures/kernelagent-golden.json`.
- **Create** `_meta/tools/fixtures/KERNELAGENT-GOLDEN-BASELINE-SHA.txt`.

### Steps
1. **`kernelagent-args.json`** — pinned args, **no `backend_dir`** (today's
   path is triton):
   ```json
   {
     "problem_spec": "softmax(x: Tensor[B,N]) -> Tensor[B,N]",
     "workspace": "/tmp/kernelagent-fixture",
     "language": "triton",
     "num_seeds": 2,
     "model_profile": "sonnet",
     "test_command": null
   }
   ```
2. **`kernelagent-agent-returns.json`** — keyed on
   `(phase_label, seed_index)`:
   - `route` → `{complexity, model_choice}`
   - `harness` → `{test_code}`
   - `synth.0`, `synth.1` → `{kernel_code}`
   - `aggregate` → `{best_seed_index}`
   - `verify` → `{passed, log}`
   - `refine` → `{kernel_code, suggestions}`
   - `verify_refined` → `{passed, log}`
3. **`kernelagent-golden.json`** — captured via the same
   `print-workflow-prompts.js` invocation as A3 step 3.
4. **`KERNELAGENT-GOLDEN-BASELINE-SHA.txt`** — `git rev-parse HEAD`.

### Acceptance
As A3.

## Task A6 — KernelAgent cuda dry-run args fixture

### Files
- **Create** `_meta/tools/fixtures/kernelagent-args-cuda.json`.

### Steps
Same as A5 step 1, plus `"backend_dir": "_substrate/backends/cuda"`,
`"backend": "cuda"`. Feeds the Stage-C cuda-dry-run test.

### Acceptance
As A4.

---

# Stage B — Workflow `.js` retrofits (`backend_dir`-gated, legacy path byte-identical)

> Mirrors P4 Stage B. Both retrofits are **additive**: with no `args.backend_dir`,
> every rendered string is byte-identical to the pre-retrofit golden captured in
> Stage A. With `args.backend_dir`, the workflow consumes the driver via the
> `USE_DRIVER` gate.
>
> **TDD discipline:** Stage B may not edit the workflow `.js` until both
> Stage-A goldens and SHAs are committed. After each seam-group edit, run
> the byte-identity test for the workflow (created in Stage C Task C1/C4 —
> commit Stage C scaffolding alongside Stage A so this test exists at
> Stage-B start).

## Task B1 (AdaExplore) — Path helpers, guard agent, new args, `USE_DRIVER` gate, manifest plumb

### Files
- **Edit** `AdaExplore/adaexplore-kernel-optimization.js`.

### Steps
1. **Column-0 named consts** for every legacy literal that the seams S1–S6
   will gate (P4 col-0 rule — load-bearing for diff readability):
   - `LEGACY_SETUP_LANG_TOKEN = 'Triton kernel'`
   - `LEGACY_EVALUATOR_COMMAND_TEMPLATE = ...` (the L373 template)
   - `LEGACY_LARGE_STEP_INTERFACE_LANG = 'PyTorch operator interface'`
   - `LEGACY_REVISER_PERF_HINTS = ['Apply a small, local perf...', ...]`
   - `LEGACY_EVALUATE_RUN_INSTRUCTION = '3. Run the evaluator command...'`
2. **New args** in the schema block at the top of the file:
   - `backend_dir` (string, optional)
   - `backend` (string, optional; aliases `language` per §6.4)
   - `kernel_filename` (string, optional, default `"candidate_kernel.py"`
     for triton, `"candidate_kernel.cu"` for cuda — derived from driver
     `source_ext` when `USE_DRIVER`)
3. **Path helpers** (per BACKEND-DRIVER-SDK §6.1):
   - `driverPath(rel)` → `${args.backend_dir}/${rel}`
   - `driverSh(script, ...flags)` → `bash ${driverPath(script)} ${flags...}`
   - `driverManifest()` → reads `${args.backend_dir}/manifest.json` via the
     load-driver `agent()` (added below); cached at module scope.
   - `driverIdioms()` → same for `idioms.json`.
4. **§6.4 guard resolution + conflict-throw**:
   - If `args.backend_dir` set, derive `args.backend` from
     `driverManifest().backend_id`; if `args.backend` was also user-set and
     mismatches, **throw**.
   - If `args.language` and `args.backend` both set and mismatch (post-normalize),
     **throw**.
   - `USE_DRIVER = Boolean(args.backend_dir)`.
5. **Load-driver `agent()` call** (only when `USE_DRIVER`) — the first
   `agent()` in Setup, before the existing Setup call. Returns
   `{manifest, idioms}`; cached for the run. `JSON_PASSTHROUGH` schema per
   P4 Task 5 (load-driver returns parsed JSON, not free text).
6. **Manifest plumb** to `manifest.yaml`'s `backend.default` — when
   `USE_DRIVER` is false but `args.backend` is set, this is a user error
   (driver dispatch requires `backend_dir`); for the byte-identity path
   neither is set → legacy literal path runs unchanged.

### Acceptance
With **no `args.backend_dir`**: `adaexplore-byte-identity.test.js` green
(no observable rendered-string change from any of the above — every new
const is referenced only inside the not-yet-added `USE_DRIVER` branches in
B2/B3).

## Task B2 (AdaExplore) — Wrap prompt-token seams S1, S3, S4, S6

### Files
- **Edit** `AdaExplore/adaexplore-kernel-optimization.js`.

### Steps
1. **S1 (line ~354 Setup prompt):** replace the inline `"Triton kernel"`
   token with
   ```js
   const setupLangToken = USE_DRIVER
     ? `${driverIdioms().lang_fence} kernel`
     : LEGACY_SETUP_LANG_TOKEN
   ```
   and interpolate. (Driver: `triton/idioms.json` → `lang_fence: "python"`
   → reads `"python kernel"`; we deliberately accept this less-pretty
   substitution — the prompt is for the agent, not the user. P4 made the
   identical tradeoff at AccelOpt S3.)
2. **S3 (line ~486 Large-step prompt):** gate the interface-preservation
   language similarly via `driverIdioms().impl_requirements`.
3. **S4 (line ~548 Reviser suggestions):** when `USE_DRIVER`, pull
   per-backend perf hints from `driverIdioms().methods.<perf_method>.prompt_guidance`
   when present, else fall back to the legacy list. **Out-of-scope deviation
   from P4:** if no matching `methods.<perf_method>` entry exists (current
   state per Spike #6), the driver path uses the legacy list. Record this
   as a deferred L3-tightening item.
4. **S6 (line ~607 Evaluate prompt):** swap the "run evaluator command"
   wording to reference the driver `run.sh` path when `USE_DRIVER`.

### Acceptance
Legacy path byte-identical. Triton dry-run (Stage C Task C3):
- No `cuda` / `__global__` / `nvcc` / `PYBIND11_MODULE` / `cuda_runtime.h` /
  `NCU Profile Results` / ` ```cuda ` tokens anywhere in the rendered prompt
  set.

## Task B3 (AdaExplore) — Wrap command seams S2, S5 + Layer-A envelope

### Files
- **Edit** `AdaExplore/adaexplore-kernel-optimization.js`.

### Steps
1. **S2 (line ~373 Setup result `evaluator_command` template):** if
   `USE_DRIVER`, the Setup `agent()` is instructed to return
   `evaluator_command = driverSh('run.sh', '--kernel', workspaceKernelPath, '--problem', problemSpecPath)`
   (composed via a column-0 helper `driverEvaluatorCommand(workspaceKernelPath)`).
2. **S5 (line ~594 `evaluatorCommand = renderCommand(...)`):** the actual
   eval dispatch. When `USE_DRIVER`:
   1. `bash` the driver `build.sh` (column-0 `runDriverBuild()`).
   2. `bash` the driver `run.sh` → captures the
      `{compiled, correct, latency_ms}` envelope per the run.sh contract.
   3. `bash` the driver `profile.sh` → captures the pointer.
   4. `python` the driver `to_evidence.py` → canonical metrics.
   5. `python _substrate/diagnose.py` (optional, via the stub agent-return
      under test) → bottleneck class.
   6. Assemble Layer-A evidence envelope per the P4 AccelOpt Task 11 shape
      (`{anti_cheat, metrics, vendor, coverage, bottleneck_class}`); fold
      into the existing `{compiled, correct, speedup}` reward triple by
      mapping `latency_ms` → `speedup` against a baseline arg.
3. **Baseline shape:** AdaExplore's reward triple `(compiled, correct, speedup)`
   is preserved; the driver path simply **computes** `speedup` from
   measured `latency_ms` and a `baseline_latency_ms` arg (new optional arg,
   default null → speedup unavailable on driver path until user supplies it).
4. **`anti_cheat.py`** is invoked on the driver `run.sh` output before
   scoring (P4 pattern at Task 9).

### Acceptance
Legacy path byte-identical. Driver path produces a valid Layer-A envelope
under the Stage C byte-identity test's `--with-driver` mode (the test
stubs `agent()` returns for build/run/profile/diagnose so the envelope is
assembled deterministically without a GPU).

## Task B4 (KernelAgent) — Path helpers, guard, args, `USE_DRIVER` gate

### Files
- **Edit** `KernelAgent/kernelagent-triton-synthesis.js`.

### Steps
Identical structure to B1, with these per-workflow specifics:
1. **Column-0 named consts:**
   - `LEGACY_ROUTE_LANG_TOKEN = 'Triton kernel'`
   - `LEGACY_HARNESS_LANG_PROMPT = 'You are a Triton kernel test engineer.'`
   - `LEGACY_SYNTH_PROMPT_PREFIX = 'Generate Triton kernel'`
   - `LEGACY_AGGREGATE_PROMPT = 'select best Triton kernel'`
   - `LEGACY_VERIFY_HARNESS_INSTRUCTION = 'Execute the test harness ...'`
   - `LEGACY_TEST_FILENAME = 'test_kernel.py'`
   - `LEGACY_KERNEL_FILENAME = 'kernel.py'`
2. **New args:** `backend_dir`, `backend`, `kernel_filename` (default driven
   by driver `source_ext` when `USE_DRIVER`), `test_filename` (default
   `test_kernel${aux_ext_or_py}`).
3. Path helpers + §6.4 resolution + load-driver `agent()` — identical to B1.

### Acceptance
Legacy path byte-identical to `kernelagent-golden.json`.

## Task B5 (KernelAgent) — Wrap prompt-token seams S1–S5p, S8p

### Files
- **Edit** `KernelAgent/kernelagent-triton-synthesis.js`.

### Steps
1. **S1 (line ~252 routing prompt):** swap `LEGACY_ROUTE_LANG_TOKEN` for
   `${driverIdioms().lang_fence}` derivation.
2. **S2 (line ~270 test-harness gen prompt):** same; also swap the inline
   Triton fence ` ```python ` to `${driverIdioms().lang_fence}` fence.
3. **S3 (line ~347 per-seed synth prompt):** swap `LEGACY_SYNTH_PROMPT_PREFIX`
   per driver lang token.
4. **S4 (line ~383 seed-aggregator prompt):** swap `LEGACY_AGGREGATE_PROMPT`.
5. **S5p (line ~515 verification prompt-token portion):** swap the
   "Triton kernel" wording.
6. **S8p (line ~721 refinement verification prompt-token portion):** swap
   the same wording.

### Acceptance
Legacy byte-identical. CUDA dry-run (Stage C Task C6): zero Triton tokens
(`@triton.jit`, `triton.language`, ` ```python ` fence around a Triton
kernel) in driver-path-rendered prompts.

## Task B6 (KernelAgent) — Wrap command + filename seams S5c, S6, S7, S8c + Layer-A envelope

### Files
- **Edit** `KernelAgent/kernelagent-triton-synthesis.js`.

### Steps
1. **S6 (line ~529 test-harness filename):** replace `'test_kernel.py'`
   literal with `args.test_filename` (default derived from driver
   `aux_ext`).
2. **S7 (line ~530 test-harness execute command):** when `USE_DRIVER`,
   build the execute command via
   `driverSh('run.sh', '--kernel', workspaceKernelPath, '--test', testPath)`;
   else use the legacy `args.test_command || generated_command`.
3. **S5c (line ~515 verification call command):** mirror S7.
4. **S8c (line ~721 refinement verification command):** mirror S7.
5. **Layer-A envelope** on the driver path: identical scaffolding to B3 step
   2 — `build → run → profile → to_evidence → diagnose → assemble`. The
   KernelAgent reward shape is binary `{passed, log}` per seed; the driver
   path enriches with `{metrics, bottleneck_class, vendor}` for the
   aggregator's `bestSeedIndex` decision (richer signal than today's pass/fail).

### Acceptance
Legacy byte-identical. Driver-path envelope valid; CUDA dry-run green.

---

# Stage C — Test trio per workflow + deferred-GPU checklist

## Task C1 (AdaExplore) — Byte-identity test

### Files
- **Create** `_meta/tools/test/adaexplore-byte-identity.test.js`.

### Steps
1. `node:test` suite. Reads `_meta/tools/fixtures/adaexplore-args.json` +
   `adaexplore-agent-returns.json`; invokes `capturePrompts` (P4 harness);
   diffs the resulting JSON against `adaexplore-golden.json`.
2. On mismatch, print first divergent index + first 200 chars of each side.
3. Read `ADAEXPLORE-GOLDEN-BASELINE-SHA.txt` and log it for provenance.
4. Add a **second** sub-test: re-run with
   `adaexplore-args-triton.json` (has `backend_dir`) + a stubbed
   agent-returns that includes the load-driver result; assert the run
   completes without throw (semantic check; the actual prompt content is
   asserted by C3, not here).

### Acceptance
`node --test _meta/tools/test/adaexplore-byte-identity.test.js` green
against the pre-retrofit golden AND green again after each Stage-B task.

## Task C2 (AdaExplore) — Guard unit test

### Files
- **Create** `_meta/tools/test/adaexplore-guard.test.js`.

### Steps
Three subtests:
1. `args.backend_dir` set, `args.backend` unset → derives `backend` from
   manifest; no throw.
2. `args.backend_dir = triton`, `args.backend = cuda` → throws with
   message mentioning both values.
3. `args.language = triton`, `args.backend = cuda` → throws.
4. `args.backend_dir` unset, `args.backend = cuda` → throws (driver
   dispatch requires `backend_dir`; no implicit-resolve).

### Acceptance
All four subtests green.

## Task C3 (AdaExplore) — Triton dry-run test

### Files
- **Create** `_meta/tools/test/adaexplore-triton-dryrun.test.js`.

### Steps
1. Run `capturePrompts` with `adaexplore-args-triton.json` + a stubbed
   agent-returns that exercises both expand_large + expand_small.
2. Assert every captured string is **free of**:
   - regex `/\bnvcc\b/`
   - regex `/\bncu\b/`
   - regex `/\b__global__\b/`
   - regex `/PYBIND11_MODULE/`
   - regex `/cuda_runtime\.h/`
   - regex `/NCU Profile Results/`
   - regex `/```cuda\b/`
3. Assert the rendered evaluator command starts with
   `bash _substrate/backends/triton/run.sh`.

### Acceptance
All token assertions pass; evaluator command shape matches.

## Task C4 (KernelAgent) — Byte-identity test

### Files
- **Create** `_meta/tools/test/kernelagent-byte-identity.test.js`.

Identical structure to C1, against `kernelagent-golden.json`.

## Task C5 (KernelAgent) — Guard unit test

### Files
- **Create** `_meta/tools/test/kernelagent-guard.test.js`.

Identical structure to C2.

## Task C6 (KernelAgent) — CUDA dry-run test

### Files
- **Create** `_meta/tools/test/kernelagent-cuda-dryrun.test.js`.

### Steps
Mirror C3 but assert **no Triton tokens** leak when `backend_dir =
_substrate/backends/cuda`:
- regex `/@triton\.jit/`
- regex `/triton\.language/`
- regex `/import triton\b/`
- regex `/```python\s*[\s\S]*@triton/` (Python fence around Triton code)

Assert rendered test-execute command starts with
`bash _substrate/backends/cuda/run.sh`.

## Task C7 — Append both workflows to deferred-GPU checklist

### Files
- **Edit** `_meta/tools/test/DEFERRED-GPU-VERIFICATION.md`.

### Steps
Append two rows under the existing AccelOpt + driver-pilot entries:
- `AdaExplore/adaexplore-kernel-optimization.js` — driver path
  (`build.sh`→`run.sh`→`profile.sh`→`to_evidence.py`→`diagnose.py` →
  Layer-A envelope) is node-renderable + unit-tested with stubs; the
  end-to-end GPU run is deferred. Owner: (unassigned). Trigger: P5e
  matrix smoke or GPU CI tier.
- `KernelAgent/kernelagent-triton-synthesis.js` — same.

### Acceptance
File renders; no `markdownlint` complaints if the repo runs one.

---

# Exit criteria (testable, no GPU)

Mirrors parent §P5b "Exit criteria":
1. Both workflows render byte-identical prompts on the legacy path
   (`adaexplore-byte-identity.test.js` + `kernelagent-byte-identity.test.js`
   green against the Stage-A goldens).
2. Triton dry-run green for AdaExplore (`adaexplore-triton-dryrun.test.js`);
   CUDA dry-run green for KernelAgent (`kernelagent-cuda-dryrun.test.js`).
   Zero cross-backend token leak in either direction.
3. Guard tests green: §6.4 resolution + conflict-throw behavior matches
   spec (`adaexplore-guard.test.js`, `kernelagent-guard.test.js`).
4. Both workflows' manifests declare `backend:` block per P5a v1.1 schema:
   `portability: clean`, `matrix_eligible: true`,
   `method_supported_backends: any`, `supported: [cuda, triton]`. Defaults
   per §2 above.
5. Substrate diff-guard still green (`git diff --stat _substrate/` shows
   zero changes touched by this sub-plan).
6. `node --test _meta/tools/test/*.test.js` green end-to-end (no
   regression of any P3/P4/P5a test).

---

# Risks + mitigations

### Risk 1 — Spike #1 fragility: workspace-local kernel path is a new convention

The deviation from P4's `args.kernel_path` (Spike #1) introduces a
**workspace-relative path computed in JS** that the driver `build.sh`
consumes. If P5c/P5d workflows adopt yet a third convention, the driver
contract drifts.

**Mitigation:** Stage-B B1/B4 introduce a column-0 helper
`workspaceKernelPath(args, driver)` and document it in the workflow file
header as the canonical pattern for "method generates kernel content" (vs
P4's "user provides kernel file"). Add a one-line note to
`_substrate/BACKEND-DRIVER-SDK.md` §"How workflows call drivers" calling
out both patterns. If a third pattern emerges in P5c, file an ADR before
introducing it.

### Risk 2 — AdaExplore MCTS agent-returns map is large and brittle

AdaExplore's `agentReturns` fixture must key on `(phase, iteration)` tuples
through 2 iterations of MCTS with both large + small expand branches —
roughly 12–15 distinct keys. P4 AccelOpt had ~8 keys for beam_size=2. A
typo in a key silently skips that branch and the rendered prompt diverges
in non-obvious ways.

**Mitigation:** Task A3 step 2 lists every required key explicitly. The
byte-identity test's mismatch reporter (C1 step 2) prints the first
divergent string slice, which surfaces missing-key skips on first run. If
debugging cost spikes anyway, fall back parent-plan §P5b OQ2's escape
hatch: a per-label-shape gate (assert prompt **count** + per-label
length-bounded snippet matches) instead of full byte-identity for
AdaExplore only. KernelAgent keeps the strict gate.

### Risk 3 — Hidden Triton-isms in KernelAgent prompts beyond the 8 enumerated seams

Spike #5 enumerated 8 seams from line-by-line grep, but free-text prompt
bodies (300+ lines of prose across 8 `agent()` calls) may contain
Triton-leaning vocabulary the regex didn't catch (e.g. "block pointer",
"tl.load", "program_id"). The CUDA dry-run test (C6) only checks for the
4 most-obvious Triton tokens; subtler leaks ship.

**Mitigation:** Extend C6's token-blacklist iteratively as failures are
caught downstream. Add a `// TRITON-LANG-LEAK-AUDIT` comment near each
prompt template in B5 marking lines that were reviewed; future P-steps can
grep for it. Accept residual risk for P5b — these are clean-tier
*topology* retrofits, not zero-leak vendor migrations; full leak audit is
a P5e CI-time concern.

---

# Estimated commit count

Per parent plan §P5b: 14–20 commits. This plan's task split:

| Stage | Tasks | Commits |
|---|---|---|
| A | A1, A2, A3, A4, A5, A6 | 6 |
| B (AdaExplore) | B1, B2, B3 | 3 |
| B (KernelAgent) | B4, B5, B6 | 3 |
| C | C1, C2, C3, C4, C5, C6, C7 | 7 |
| **Total** | | **19** |

Within parent's 14–20 envelope.

---

# Depends on / blocks

- **Depends on:** P5a complete (manifest v1.1 schema + `validate-workflow.js`
  + the P5a-shipped `print-workflow-prompts.js`/`capturePrompts` harness
  reused by Stage A). Also depends on P4 (the AccelOpt pilot — pattern
  source).
- **Blocks:** P5c (mid-complexity batch reuses the Spike #1 + Spike #2
  resolutions baked into B1/B4); P5e matrix-smoke CI (needs AdaExplore
  matrix_eligible + triton dry-run as the first matrix-eligible
  retrofitted workflow).

---

# Recommendation

**Proceed with the plan as written.** The two workflows are well-matched to
the P4 AccelOpt template — same `node --test`-on-macOS testability story,
same `backend_dir`-gated additive retrofit shape, byte-identity gate viable
per Spike #2. The one structural deviation from P4 (workspace-local kernel
path per Spike #1) is contained to two column-0 helpers and one SDK-doc
footnote.

**Sequencing recommendation:** execute AdaExplore first (B1→B2→B3 +
C1→C2→C3), then KernelAgent (B4→B5→B6 + C4→C5→C6), then C7. This minimizes
context-switching across the two `.js` files and lets Risk 2 (MCTS fixture
brittleness) be resolved before exposing the same fixture machinery to
KernelAgent's simpler routing+seeds topology — where failures would be
harder to attribute to the fixture vs the gate.

**Out of scope (deferred):**
- L3 driver-coverage tightening for the reviser perf-hints seam (Task B2
  step 3) — file as a follow-up issue.
- Cross-backend prompt-leak audit beyond the 8 enumerated tokens (Risk 3
  mitigation) — fold into P5e CI work.
- Adding `kernel_path` as the canonical arg name across all P5b+ workflows
  — keep workspace-local convention; revisit only if a third pattern
  emerges.




