# P5a — Schema + Generator/Template Prep + Deterministic L0 Backend Validator

> **Status:** detailed implementation plan. Decomposes the §5 sub-plan "P5a"
> from `docs/superpowers/plans/2026-06-08-p5-clean-workflow-migration.md` into
> dependency-ordered Tasks, each subagent-executable. **Contract-only:** no
> workflow `.js` file is touched in this sub-plan.
>
> **Spec of record:** `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md`
> (§9.2 generator/template/schema; §9.3 driver conformance L0; §4.4 manifest
> fields; §4.9 L0 contract).
> **Pattern of record (TDD ordering, per-task contracts):**
> `docs/superpowers/plans/2026-06-08-accelopt-driver-pilot.md`.
> **Master plan:** `docs/superpowers/plans/2026-06-08-p5-clean-workflow-migration.md`
> §5 "P5a".
>
> **For agentic workers:** REQUIRED SUB-SKILLS:
> `superpowers:test-driven-development`, `superpowers:subagent-driven-development`,
> `superpowers:verification-before-completion`.

---

## 1. Goal

Land the §9.2 schema and generator changes so newly generated clean
workflows emit driver-shaped code by default and every manifest declares
its `backend:` posture; ship the §9.3 deterministic L0 backend
validator's `validate-backend` invariants — but **without editing any
workflow `.js`**. The output of P5a is the contract that P5b/c/d
retrofits will conform to.

### Scope (precise, bounded)

1. `_meta/manifests/schema.yaml` → v1.1 with a top-level `backend:`
   block, extended `args.optional[]` (`backend`, `profile_command`),
   `ncu_command` retained as a documented deprecated alias.
2. `_meta/tools/generate-workflow.js` agent-prompt edits at L263–282
   (model-args prompt) and L483–525 (generate prompt). Prompt
   instructions only — no token substitution; per spec §9.2 the
   generator is not a substitution engine.
3. `_meta/tools/validate-workflow.js` — append two prompt-checklist
   items (LLM-driven warnings; not hard gates).
4. `_meta/templates/{iterative-loop,search-based,single-pass,tree-exploration}.js`
   guidance-text + input-policy comment edits (L70–75 of
   `iterative-loop.js`).
5. Extend `_substrate/backends/validate_backend.py` with deterministic
   L0 backend invariants (manifest/idioms conform, no driver edits a
   universal substrate script, `backend_id == dir`). **No Node
   duplicate.**
6. New regression test:
   `_meta/tools/test/generator-prompt-schema.test.js` — pins the
   generate-workflow agent's `schema:` keys.
7. New backend-validator tests (Python pytest under
   `_substrate/tests/`) covering the new invariants on cuda + triton.

### Non-goals (deferred to later P5 sub-plans, or out of scope)

- Editing any workflow `.js` body (P5b/c/d/e do that).
- `_meta/manifests/*.yaml` populating its new `backend:` block — each
  retrofit sub-plan (P5b–P5e) adds the block per workflow.
- A Node `validate-backend.js` duplicate of the Python validator
  (rejected — see spike #3).
- `_tools/`↔`_meta/tools/` and `_templates/`↔`_meta/templates/`
  deduping (pre-existing duplication; flagged, not deepened — spec §2.3).
- §9.3 matrix smoke + CI wiring (P5e).
- `_substrate/BACKEND-DRIVER-SDK.md` and `REGISTRY.md` rewrites (P5f).
- Real-hardware GPU verification (opt-in tier inherited from P4).

---

## 2. Pre-flight spike resolutions

The master plan flagged 4 open questions. Each resolved here with cited
evidence. These become facts P5a builds on; no further investigation
needed during Task execution.

### Spike 1 — KSearch tier (clean vs triton-intrinsic)

**Decision:** KSearch is **clean / multi-backend / matrix_eligible**.
No demotion needed; no spec §7.2 amendment.

**Evidence:**
- `KSearch/ksearch-kernel-optimization.js:17`:
  `supported_languages: ['triton', 'cuda', 'python']` — already a
  three-language declaration, not triton-locked.
- `KSearch/ksearch-kernel-optimization.js:109`:
  `const LANGUAGE = args.language || 'triton'` — `language` is an
  arg-driven polymorphism switch (same shape as AccelOpt's L135
  pattern verified in spec §2.2).
- `_meta/manifests/ksearch.yaml:34-35`:
  `method.category: "tree_exploration"` — the **topology** is
  world-model decision tree; tree exploration is backend-agnostic by
  construction.
- No `@triton.jit` / `tl.load` / `nvcc` / `ncu` literal in the workflow
  body (grep result on `ksearch-kernel-optimization.js`: only the
  three suitability-list mentions of `triton`/`cuda`, plus `'triton'`
  as the LANGUAGE default).

**Consequence for P5a:** v1.1 schema accepts KSearch as `portability:
clean`, `matrix_eligible: true`, `supported: [cuda, triton]` (or
`[any]` per P5d's per-workflow decision); P5d retrofits it through the
standard checklist with no special-case.

### Spike 2 — Manifest tree location (`_meta/manifests/` vs `_manifests/`)

**Decision:** `_meta/manifests/` is canonical. P5a edits **only**
`_meta/manifests/schema.yaml`. `_manifests/schema.yaml` gets a
one-line deprecation header comment pointing to `_meta/` (does NOT
copy the v1.1 content; the duplicate tree is frozen pending the
out-of-scope dedup work).

**Evidence:**
- Spec §2.3:
  > "`SOLVER-SDK.md:117` names `_meta/tools/validate-workflow.js` as
  > the conformance checker, so **`_meta/` is canonical.** All new
  > tooling … and all edits to the generator/validator/templates/
  > manifest-schema land in the `_meta/` tree. The plan **must not
  > deepen** the `_tools/`↔`_meta/tools/` divergence."
- Both trees byte-identical today (verified via `diff
  _meta/manifests/schema.yaml _manifests/schema.yaml` → identical;
  same for `_meta/templates/` vs `_templates/`).
- `_meta/manifests/` has 13 files; `_manifests/` has 8 (missing:
  `archagent.yaml`, `fact.yaml`, `gpuforecasters.yaml`, `stitchcuda.yaml`,
  `xe-forge.yaml`). The `_meta/` tree is already the strict superset.
- One stale reference exists at
  `_meta/tools/generate-workflow.js:394`: comment string
  `"The manifest must conform to the schema at _manifests/schema.yaml"`.
  P5a's Task 4 corrects this string to `_meta/manifests/schema.yaml`
  (one-token doc fix; not a behavioral change).

**Consequence for P5a:** every "edit the schema" / "edit the
templates" Task targets the `_meta/` path. The `_manifests/` and
`_templates/` mirrors get a one-line deprecation header but no v1.1
content (avoids deepening divergence).

### Spike 3 — Python (`validate_backend.py`) vs Node (`validate-backend.js`)

**Decision:** **Keep Python.** Extend `_substrate/backends/validate_backend.py`
with the new L0 invariants. **Do not ship a Node duplicate.** Spec
§9.2/§9.3 references to `_meta/tools/validate-backend.js` are
corrected via the SDK doc's existing "Deviations" section (P5f
formalizes the spec-text correction).

**Evidence:**
- `_substrate/BACKEND-DRIVER-SDK.md:142-153` already records the
  deviation:
  > "The L0 validator is Python in `_substrate/backends/`, not Node
  > in `_meta/tools/`, and ships no JSON-Schema files. … Any future
  > generator (§9.2) work expecting `_schema/*.json` must consume this
  > Python validator instead."
- `_substrate/backends/validate_backend.py` exists (133 lines,
  stdlib-only, imports live `method_gate.TABLE`).
- `_substrate/tests/test_validate_backend.py` covers positives + 4
  named negatives (`backend_id_mismatch`,
  `idiom_references_unknown_method`, `non_canonical_metric_key`,
  `bogus_bottleneck_class`).
- A Node duplicate would force a second source of truth for the
  `KNOWN_METHODS` set (currently derived live from `method_gate.TABLE`
  by `import method_gate`); duplicating in JS would require either
  parsing `method_gate.py` from JS or freezing a `KNOWN_METHODS`
  constant — both bug farms.
- Master plan §5 "P5a" explicitly says: *"per SDK doc's 'Deviations'
  section, Python won; settle in P5a's implementation plan, do not
  ship both."*

**Consequence for P5a:** Task 6/7 extend the Python validator and
its pytest. Master-plan amendment is NOT required (the SDK
"Deviations" section already carries the decision); a spec-text
correction is bundled into P5f.

### Spike 4 — `capturePrompts` on `generate-workflow.js`

**Decision:** The P4 `capturePrompts` harness works on
`generate-workflow.js` **as-is**. No harness extension needed for
P5a's regression test.

**Evidence:**
- `_meta/tools/print-workflow-prompts.js:15-19`: `capturePrompts({
  workflowPath, args, agentReturns })` shells through
  `_meta/tools/lib/run-workflow.js`.
- `_meta/tools/lib/run-workflow.js:25-34`: strips the leading
  `export ` token + textually wraps the body in
  `(async function(){ … })()`. This handles top-level `await` (which
  `generate-workflow.js` uses heavily — L335, L391, L483) and
  top-level `return`.
- `generate-workflow.js:1`: `export const meta = { … }` — it IS itself
  a workflow `.js`; the harness contract applies.
- `generate-workflow.js` body uses ONLY the stubbed runtime globals:
  `args`, `agent`, `phase`, `parallel`, `pipeline`, `log`. No
  `readFile`, no `WebFetch`, no `fs`, no `import`. The sandbox in
  `run-workflow.js:71-89` provides every global the body references
  (Math, Promise, console, JSON). The agents return canned objects
  via `agentReturns`, satisfying every downstream `?.field` chain.
- The `schemaStub` fallback (`_meta/tools/lib/schema-stub.js`) returns
  a minimal valid object even when an agent label is not in
  `agentReturns`, so a partial `agentReturns` is sufficient.

**Consequence for P5a:** Task 5's regression test wires
`capturePrompts({ workflowPath: <generate-workflow.js> })` with a
fixed `args` (`paper_url`, `repo_url`, `output_dir`, `method_name`)
and an `agentReturns` map keyed by the 5 generator labels (research,
model-topology, model-phases, model-args, model-angles,
assemble-manifest, generate-workflow, validate). It captures the
prompts emitted by `model-args` and `generate-workflow`, asserts the
`schema:` keys of the captured `agent()` opts deep-equal a pinned
baseline. **No harness extension; no new test infrastructure.**

---

## 3. Hard dependencies

P5a starts only after both of these are merged on `main`:

1. **P4 — AccelOpt driver pilot** merged
   (`docs/superpowers/plans/2026-06-08-accelopt-driver-pilot.md`).
   Provides `_meta/tools/print-workflow-prompts.js`,
   `_meta/tools/lib/run-workflow.js`,
   `_meta/tools/lib/schema-stub.js`, the test directory layout, and
   the `capturePrompts({workflowPath, args, agentReturns})` API used by
   Task 5.
2. **P5 master plan** merged
   (`docs/superpowers/plans/2026-06-08-p5-clean-workflow-migration.md`).
   This document is the master's §5 sub-plan; merging the master
   first ensures the dependency graph & non-goals are committed.

Soft preconditions (already met on the current `dev/solver-substrate`
branch — verify at Task 1 start, **abort and escalate** if any
regressed):

- P1: `_substrate/backends/validate_backend.py` present (verified —
  133 lines).
- P2: `_substrate/diagnose.py` vendor profile + null rule edits
  present.
- P3: `_substrate/backends/cuda/` and `_substrate/backends/triton/`
  present and L0-conformant against today's validator (verified —
  `manifest.json`/`idioms.json` present in both).

---

## 4. Task breakdown

8 Tasks, dependency-ordered. Each Task ends with green tests; each
test is added **before** its implementation (TDD anchor). Tasks
group into 4 commit clusters in §5.

> **TDD ordering rule:** for every Task that has a `Tests` block,
> the test commit lands first (red), then the implementation commit
> turns it green. No combined "test+impl" commit. The exception is
> Task 1 (the schema doc edit) which has no behavioral assertion
> attached — it's a doc edit + a deep-equal parse check the next
> Task adds.

### Task 1 — Add v1.1 `backend:` block to `_meta/manifests/schema.yaml`

**Goal:** Extend the manifest schema with the §9.2 `backend:` block
and `args.optional[]` additions. Bump the header to v1.1. **Doc edit
only** — the schema is YAML comments + example fields, not a parser
artifact (today's repo has no YAML schema validator that gates this
file).

**Files touched:**
- `_meta/manifests/schema.yaml` (in-place edit; header bump + new
  `backend:` section + extended `args.optional` examples).
- `_manifests/schema.yaml` (append a 3-line deprecation header
  comment; **do not copy v1.1 content**).

**Pre-conditions:** none beyond §3.

**Implementation steps:**

1. Bump the header at L1–17:
   - Change `Manifest Schema v1.0` → `Manifest Schema v1.1`.
   - Add a `# v1.1 changelog:` block of comment lines listing the
     three additions: `backend:` block; `args.optional[]` learns
     `backend` and `profile_command`; `ncu_command` retained as a
     deprecated alias.
2. Insert a new top-level `backend:` section between the existing
   `topology:` (L86) and `phases:` (L127) sections. Use this exact
   YAML (preserves the file's `[R]/[O]` annotation style):

   ```yaml
   # ---------------------------------------------------------------------------
   # BACKEND POSTURE — Which (language × vendor) drivers this workflow can run
   # ---------------------------------------------------------------------------
   # Added v1.1. Mirrors _substrate/backends/<id>/manifest.json's contract.
   # See docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md §4.4 + §7.2.
   backend:
     # [R] Backend ids this workflow's method can honor.
     #     Use ['any'] when the method is fully backend-agnostic (clean tier).
     #     Use a concrete list (e.g. ['cuda','triton']) for vendor_locked methods.
     #     Single-element list (e.g. ['rocm']) for method_intrinsic.
     #     Every entry must equal a directory name under _substrate/backends/.
     supported: []
     # [R] Default backend id when args.backend is not provided.
     #     May be null → forces callers to pass --backend explicitly (clean tier convention).
     default: null
     # [R] Whether this workflow participates in the matrix-smoke CI (§9.3).
     #     true     = run every supported × every matrix driver
     #     partial  = run a declared subset (e.g. {cuda} only)
     #     false    = excluded (method_intrinsic always false)
     matrix_eligible: false
     # [R] Tier per spec §7.2. One of: clean | vendor_locked | method_intrinsic.
     portability: ""
     # [R when portability != clean] The vendor/tool this method is intrinsic to.
     #     clean         → omit
     #     vendor_locked → e.g. "ncu" / "torch.profiler"
     #     method_intrinsic → e.g. "cutlass" / "rocm"
     intrinsic_to: ""
     # [O] Capability floor — only set on vendor_locked or capability-sensitive workflows.
     #     metrics:    [dram_pct, sm_pct]   # min set the driver must populate
     #     problem_types: [cuda-kernel-optimization]
     requires_capability: {}
   ```

3. Extend `args.optional[]` example block at L176–181 to include the
   neutral backend args. Append (do not replace) these example entries
   directly after the existing optional template:

   ```yaml
     # --- v1.1 additions: neutral backend-axis args ---
     - name: "backend"
       type: "string"
       default: "null"
       description: "Backend id (matches _substrate/backends/<id>); null → use backend.default from manifest"
       example: "cuda"
     - name: "backend_dir"
       type: "string"
       default: "''"
       description: "Path to the driver dir; empty → legacy inline-prompt path (back-compat)"
       example: "_substrate/backends/cuda"
     - name: "profile_command"
       type: "string"
       default: "''"
       description: "Neutral profile command template (replaces legacy ncu_command). Empty → driver-resolved."
       example: "ncu --csv --target-processes all -o {out} {bin}"
     - name: "ncu_command"
       type: "string"
       default: "''"
       description: "DEPRECATED v1.1 alias for profile_command. Retained for back-compat; remove in v1.3."
       example: ""
   ```

4. Document the deprecation window explicitly in the v1.1 changelog
   header: **`ncu_command` is removed in v1.3 (two minor versions after
   v1.1; v1.2 emits a generator-prompt warning; v1.3 stops emitting the
   alias).** This resolves the master-plan open question
   ("`ncu_command` deprecation window length — pick a concrete number
   in P5a (1 or 2 minor versions)"). **2 minor versions chosen** —
   one version's warning + one version's removal gives downstream
   workflow authors a real migration cycle without indefinite legacy
   support.

5. Edit `_manifests/schema.yaml` (the mirror): prepend exactly:
   ```yaml
   # DEPRECATED MIRROR — see _meta/manifests/schema.yaml for the authoritative v1.1+ schema.
   # This file is preserved for back-compat against tools still hard-coding the _manifests/ path
   # (see _meta/tools/generate-workflow.js:394 stale reference — corrected in P5a Task 4).
   # Do not edit. Spec §2.3.
   ```
   **No** v1.1 content copy — by deliberate policy the mirror stays
   at v1.0 until the dedup work (out of scope) runs.

**Tests:** Task 1 ships no test of its own (this is a docs/YAML
edit; the parser check is added by Task 2). The merge gate for Task
1 is: `python3 -c "import yaml; yaml.safe_load(open('_meta/manifests/schema.yaml'))"`
returns without exception. Add this as a one-liner shell smoke in the
commit message.

**Exit criteria:**
- `_meta/manifests/schema.yaml` parses as YAML.
- Header line 2 reads `Manifest Schema v1.1`.
- `backend:` section present between `topology:` and `phases:`.
- `args.optional[]` lists `backend`, `backend_dir`, `profile_command`,
  `ncu_command` (with deprecation note in description).
- `_manifests/schema.yaml` has the 3-line deprecation header; rest
  byte-identical to its pre-task state.

---

### Task 2 — Backward-compat fixture test (TDD anchor for Task 3+)

**Goal:** Prove every existing manifest under `_meta/manifests/*.yaml`
continues to parse against the v1.1 schema with the `backend:` block
absent. The v1.1 `backend:` block must be **optional** at the
schema-doc level until each workflow's P5b/c/d retrofit populates it
(this matches master-plan exit criterion §5 P5a #1).

**Files touched:**
- `_substrate/tests/test_manifest_parse_v11.py` (NEW; pytest).
- No production code touched.

**Pre-conditions:** Task 1 merged.

**Implementation steps:**

1. Create `_substrate/tests/test_manifest_parse_v11.py`. Stdlib +
   `yaml` (already used elsewhere in the repo; if absent, the test
   should skip with a clear message — but `pyyaml` is in the repo's
   `_substrate/tests/` deps, verify via `python3 -c "import yaml"`
   in the spike commit).
2. The test enumerates `_meta/manifests/*.yaml` excluding `schema.yaml`.
3. For each manifest:
   - `yaml.safe_load(open(p))` returns a dict (not None, not a list).
   - `source`, `workflow`, `method`, `topology`, `phases`, `args`
     keys present (the v1.0 minimum surface — unchanged in v1.1).
   - Either `backend` key is absent (back-compat) OR — if present —
     `backend['supported']` is a list, `backend['portability']` is one
     of `{clean, vendor_locked, method_intrinsic}`, and
     `backend['matrix_eligible']` is `True | False | "partial"`.
4. **Conformance fixture for the v1.1 happy path:** add one synthetic
   manifest at `_substrate/tests/fixtures/manifest_v11_minimal.yaml`
   that does populate the `backend:` block (`supported: [cuda]`,
   `default: cuda`, `matrix_eligible: true`, `portability: clean`).
   Test asserts this parses + passes the v1.1 backend-block checks.
5. **Conformance fixture for an existing real manifest:** explicitly
   pin `_meta/manifests/accelopt.yaml` as the back-compat
   conformance fixture (master plan §5 calls this out:
   "Call out which existing manifest is your conformance fixture").
   Add a dedicated test method `test_accelopt_yaml_v10_still_parses_under_v11`
   that asserts the file parses, has all v1.0 keys, and has no
   `backend:` block (i.e. legacy shape).

**Test commands:**
```
cd _substrate && python3 -m pytest tests/test_manifest_parse_v11.py -v
```

**Exit criteria:**
- 13 existing manifests (post-Task 1) all parse green under the v1.1
  contract.
- The synthetic `backend:`-populated fixture parses + passes the
  block-shape check.
- AccelOpt manifest pinned as the legacy back-compat fixture.

---

### Task 3 — Edit `generate-workflow.js` model-args prompt (L263–282)

**Goal:** Teach the model-args agent prompt to extract
`native_backend` + `portability_class` from the manifest and emit
`backend`/`backend_dir` as standard optional args. Mark `ncu_command`
as deprecated in the prompt text. **Prompt instructions only — no
substitution; per spec §9.2.**

**Files touched:**
- `_meta/tools/generate-workflow.js` (L252–289 region).

**Pre-conditions:** Task 2 green (so the regression test in Task 5
has a baseline schema to compare against — see Task 5).

**Implementation steps:**

1. Inside the existing `agent('...', { label: 'model-args', ... })`
   prompt block (template literal beginning at L252), keep the
   existing "Standard args patterns" list and append a new
   sub-section before the "Method-specific args" line (L284):

   ```
   # Backend-axis args (v1.1 contract; see _meta/manifests/schema.yaml `backend:` block):
   - backend: backend id (matches a directory under _substrate/backends/). Use the manifest's `backend.default` when present; null forces explicit --backend.
   - backend_dir: optional driver-dir path; empty string → legacy inline-prompt path. Required for any driver-shaped emission.
   - profile_command: neutral profile-command template; preferred over the legacy ncu_command alias.
   - driver_shell_prefix: optional prefix for invoking driver .sh entrypoints (rarely set).
   ```

2. Edit the existing `Do not emit concrete default commands ...` block
   (currently L282) to **demote** `ncu_command` from "permitted user-
   supplied arg" status to a deprecated alias:

   - Old: `…through compile_command, test_command, benchmark_command, profile_command, ncu_command, or method-specific tool args.`
   - New: `…through compile_command, test_command, benchmark_command, profile_command, or method-specific tool args. The legacy ncu_command is a deprecated alias for profile_command (scheduled for removal in manifest schema v1.3); emit it only when the source paper or manifest explicitly names it for back-compat, and mark it deprecated in the arg description.`

3. Update the model-args agent's `schema:` (L292–324) — leave the
   keys unchanged (the regression test in Task 5 will fail loudly if
   any key drifts). Add a comment line above the schema literal:
   `// SCHEMA PINNED — see _meta/tools/test/generator-prompt-schema.test.js`.

**Tests:** Task 5 introduces the regression test that pins this
schema. Task 3 itself produces no new test file; its merge gate is
Task 5's test going green.

**Exit criteria:**
- Prompt text contains the new "Backend-axis args (v1.1 contract)"
  sub-section.
- `ncu_command` reframed as deprecated alias.
- Schema object keys unchanged (deep-equal to pre-edit).
- File still parses as JS:
  `node --check _meta/tools/generate-workflow.js`.

---

### Task 4 — Edit `generate-workflow.js` generate prompt (L483–525)

**Goal:** Teach the generate-workflow agent to emit the §6.4 driver-
dispatch split (`method_supported_backends` / `default_backend` /
`requires_capability`), the §6.1 path-helper block, and the §6.2
Setup `load-driver` agent gated on `USE_DRIVER`. Tighten the
"Critical constraints" to forbid naming `nvcc`/`ncu`/`@triton.jit`/
`__global__`/`PYBIND11_MODULE` in any clean-tier workflow's body
prompts. Fix the L394 `_manifests/schema.yaml` reference.

**Files touched:**
- `_meta/tools/generate-workflow.js` (L394 + L483–525 region).

**Pre-conditions:** Task 3 merged.

**Implementation steps:**

1. **L394 one-token fix** (the stale `_manifests/` reference from
   Spike 2 evidence): change
   `"The manifest must conform to the schema at _manifests/schema.yaml."`
   → `"The manifest must conform to the schema at _meta/manifests/schema.yaml."`

2. In the `assemble-manifest` agent prompt (L391–470), add a new
   "## Backend posture" data section between "## Topology" (L421) and
   "## Phases" (L427):

   ```
   ## Backend posture (v1.1; see _meta/manifests/schema.yaml `backend:` block):
   - supported: ${JSON.stringify(paperResearch?.supported_backends || ['any'])}
   - default: ${paperResearch?.default_backend || 'null'}
   - matrix_eligible: ${paperResearch?.matrix_eligible !== false}
   - portability: ${paperResearch?.portability_class || 'clean'}
   - intrinsic_to: ${paperResearch?.intrinsic_to || ''}
   ```

   And append `backend` to the "Generate a complete, valid YAML
   manifest. Include ALL sections: source, workflow, inputs, method,
   topology, phases, plan_angles, args, correctness, return_fields,
   template." enumeration at L454 →
   `…, args, backend, correctness, return_fields, template.`

3. In the `generate-workflow` agent prompt (L483–525), edit the
   "Generation Rules" enumeration (currently 12 numbered items):

   - **Rule 2 (current):** keep the `WORKFLOW_SUITABILITY` emission
     but tighten its body. Replace:

     > "Immediately after meta, emit `WORKFLOW_SUITABILITY` with
     > supported_languages, supported_problem_types, problem_types,
     > reason, plus `assertWorkflowSuitability()` …"

     with:

     > "Immediately after meta, emit `WORKFLOW_SUITABILITY` with the
     > §6.4 split:
     >   - `method_supported_backends` (from manifest `backend.supported`;
     >     `['any']` for clean methods),
     >   - `default_backend` (from manifest `backend.default`; may be `null`),
     >   - `requires_capability` (from manifest `backend.requires_capability`;
     >     `{}` if absent),
     >   - `supported_problem_types`, `problem_types`, `reason` (unchanged
     >     from v1.0).
     > Plus `assertWorkflowSuitability()` that hard-fails when explicit
     > `args.backend` is not in `method_supported_backends`, when
     > `args.problem_type` is incompatible, or when a manifest declares
     > `portability: method_intrinsic` and `args.backend` is off-list
     > (throwing the intrinsic-reason message verbatim). Preserve the
     > legacy `supported_languages` key as an alias of
     > `method_supported_backends` for one minor version (v1.2)."

   - **New Rule 5a** (insert between current Rules 5 and 6):

     > "5a. If `args.backend_dir` is provided (the driver path), emit
     > the §6.1 path-helper block immediately after the arg const
     > declarations:
     >
     > ```js
     > const USE_DRIVER = Boolean(BACKEND_DIR)
     > const driverSh = (script, ...flags) =>
     >   `${DRIVER_SHELL_PREFIX || ''} ${BACKEND_DIR}/${script} ${flags.join(' ')}`.trim()
     > const driverPy = (script, ...flags) =>
     >   `${SUBSTRATE_COMMAND_PREFIX} ${BACKEND_DIR}/${script} ${flags.join(' ')}`
     > ```
     >
     > and a §6.2 `load-driver` Setup agent that reads
     > `${BACKEND_DIR}/manifest.json` + `${BACKEND_DIR}/idioms.json`,
     > gated on `USE_DRIVER`. Per the §6.4 contract: never emit
     > `load-driver` outside `USE_DRIVER`."

   - **Rule 10 (tighten):** replace:

     > "Never hardcode an evaluator/compiler/profiler command in
     > Usage examples or agent prompts. Describe the JSON/artifact
     > contract and consume user-provided command args."

     with:

     > "Never hardcode an evaluator/compiler/profiler command in
     > Usage examples or agent prompts. For workflows with
     > manifest `backend.portability: clean`, never name the literal
     > tokens `nvcc`, `ncu`, `__global__`, `PYBIND11_MODULE`,
     > `cuda_runtime.h`, `@triton.jit`, or `tl.load` in any agent
     > prompt body — these are backend-specific idioms and must
     > arrive through the driver's `idioms.json` (read by the
     > `load-driver` agent) and be interpolated as
     > `${IDIOMS.lang_fence}`/`${IDIOMS.impl_requirements}`/etc.
     > Describe the JSON/artifact contract and consume user-provided
     > command args."

4. Update the generate-workflow agent's `schema:` (L528–536) —
   leave the keys unchanged (`workflow_code`, `filename`, `directory`).
   Add a comment: `// SCHEMA PINNED — see generator-prompt-schema.test.js`.

**Tests:** Same as Task 3 — Task 5's regression test pins all
agent schemas across this file.

**Exit criteria:**
- L394 reference corrected.
- `assemble-manifest` prompt includes the `## Backend posture`
  section and `backend` in the manifest-section enumeration.
- `generate-workflow` prompt has Rule 2 rewritten, Rule 5a inserted,
  Rule 10 tightened.
- `node --check _meta/tools/generate-workflow.js` passes.
- All agent `schema:` literals deep-equal pre-edit (verified by Task 5).

---

### Task 5 — Generator-prompt-schema regression test

**Goal:** Pin the agent `schema:` outputs of `generate-workflow.js`
so that future prompt edits (P5a Tasks 3+4 + any subsequent generator
work) cannot accidentally drift the structured-output keys downstream
consumers rely on. This is the master-plan-required regression test.

**Files touched:**
- `_meta/tools/test/generator-prompt-schema.test.js` (NEW; uses
  `node:test` runner per repo convention from
  `_meta/tools/test/print-workflow-prompts.test.js`).
- `_meta/tools/fixtures/generate-workflow-args.json` (NEW; fixed
  args object).
- `_meta/tools/fixtures/generate-workflow-agent-returns.json` (NEW;
  the agentReturns map covering every agent label in the generator).

**Pre-conditions:** Tasks 3 and 4 merged.

**Implementation steps:**

1. Create `_meta/tools/fixtures/generate-workflow-args.json`:
   ```json
   {
     "paper_url": "https://arxiv.org/abs/0000.00000",
     "repo_url": "https://github.com/example/example",
     "output_dir": "/tmp/p5a-fixture",
     "method_name": "P5aFixture",
     "partial_manifest": ""
   }
   ```

2. Create `_meta/tools/fixtures/generate-workflow-agent-returns.json`
   keyed by every agent `label:` in `generate-workflow.js`. From the
   file: `research-paper`, `research-repo` (verify name in spike when
   writing — was Agent 2 in `parallel([])`), `model-topology`,
   `model-phases`, `model-args`, `model-angles`, `assemble-manifest`,
   `generate-workflow`. Provide a minimal `{}` (`schemaStub` falls
   back) for each — but for `assemble-manifest` set
   `{"manifest_yaml": "# fixture", "method_name": "P5aFixture",
   "workflow_name": "p5afixture-kernel-optimization"}` so the
   subsequent `generate-workflow` prompt can interpolate without
   throwing.

3. Create `_meta/tools/test/generator-prompt-schema.test.js`:

   ```js
   'use strict'
   const test = require('node:test')
   const assert = require('node:assert/strict')
   const fs = require('node:fs')
   const path = require('node:path')
   const { capturePrompts } = require(path.resolve(
     __dirname, '..', 'print-workflow-prompts.js'))

   const ROOT = path.resolve(__dirname, '..', '..', '..')
   const WORKFLOW = path.join(ROOT, '_meta', 'tools', 'generate-workflow.js')
   const ARGS = require(path.resolve(__dirname, '..',
     'fixtures', 'generate-workflow-args.json'))
   const RETURNS = require(path.resolve(__dirname, '..',
     'fixtures', 'generate-workflow-agent-returns.json'))

   // The EXACT set of agent-schema property keys we pin. Any future drift to
   // these keys is a P5a contract violation and must update this test + the
   // master plan in the same commit.
   const PINNED_SCHEMAS = {
     'model-args': {
       required_args: ['name', 'type', 'description', 'example'],
       optional_args: ['name', 'type', 'default_value', 'description', 'example'],
     },
     'generate-workflow': {
       top: ['workflow_code', 'filename', 'directory'],
     },
     'assemble-manifest': {
       top: ['manifest_yaml', 'method_name', 'workflow_name'],
     },
     // Add more labels as the generator evolves; each entry is intentional.
   }

   test('generate-workflow.js agent schemas are stable (P5a regression)',
     async () => {
       const calls = await capturePrompts({
         workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
       const byLabel = Object.fromEntries(calls.map(c => [c.label, c]))
       for (const label of Object.keys(PINNED_SCHEMAS)) {
         assert.ok(byLabel[label],
           `expected agent call labeled '${label}' to be captured`)
       }
       // The schema_keys assertion deep-equals against PINNED_SCHEMAS.
       // (Implementation note: capturePrompts captures prompts, not
       //  schemas; if schemas aren't on the captured record, extend
       //  run-workflow.js to also capture opts.schema. Verify in
       //  spike-during-implementation — likely a 1-line addition to
       //  agentStub's calls.push.)
     })
   ```

   > **Implementation note for the subagent executing Task 5:**
   > `_meta/tools/lib/run-workflow.js:42-45` currently pushes
   > `{ seq, label, phase, prompt }` only. The schema is NOT
   > captured. Task 5 will extend `agentStub` to also push the
   > `opts.schema` reference: `calls.push({ seq: seq++, label,
   > phase, prompt, schema: opts && opts.schema })`. This is a
   > 1-line additive change with no impact on existing tests (the
   > new key is ignored by callers that destructure
   > `{seq,label,phase,prompt}`). Add a small companion test under
   > `_meta/tools/test/print-workflow-prompts.test.js` asserting the
   > new `schema` key is present and is a JSON-Schema object.

4. **TDD ordering:** the test commit lands FIRST and is expected to
   fail (red) — because Task 3/4's prompt edits are already merged,
   the test will fail not on schema drift but on the missing
   `schema` field. The Task 5 follow-up commit adds the 1-line
   `run-workflow.js` extension turning the test green.

**Test command:**
```
cd _meta/tools && node --test test/generator-prompt-schema.test.js
```
Also run the full test suite to confirm no regression:
```
cd _meta/tools && node --test test/*.test.js
```

**Exit criteria:**
- `generator-prompt-schema.test.js` exists; runs green.
- `run-workflow.js` captures `opts.schema` on each call (verified by
  a companion assertion in `print-workflow-prompts.test.js`).
- All pre-existing tests under `_meta/tools/test/*.test.js` still
  green (AccelOpt byte-identity, guard, triton dry-run,
  print-workflow-prompts).

---

### Task 6 — Extend `validate_backend.py` with new L0 invariants

**Goal:** Add the deterministic backend invariants the master plan
calls out beyond today's validator coverage: (a) no driver edits a
universal substrate script (byte-equality against a frozen baseline);
(b) explicit `backend_id == basename(dir)` is already covered;
**add**: `manifest.profiler.invoke == "profile.sh"` (and equivalent
for `compiler`/`runner`) — closing the spec §4.2 footnote *"asserts
the declared invoke values equal those fixed names (no divergence
allowed)"*.

**Files touched:**
- `_substrate/backends/validate_backend.py` (extend; no breaking change).
- `_substrate/tests/test_validate_backend.py` (extend with the new
  negative fixtures).
- `_substrate/tests/fixtures/<new-fixture-dirs>/` (NEW; ≥3 crafted
  negatives required by master-plan exit criterion #2).

**Pre-conditions:** Task 5 merged.

**Implementation steps:**

1. **TDD anchor — write the tests first.** Add to
   `_substrate/tests/test_validate_backend.py` 3 new test methods:
   - `test_invoke_field_must_be_fixed_name`: fixture
     `bad_invoke_compiler/` where `manifest.json` declares
     `compiler.invoke: "compile.sh"` (wrong) → assert validator FAILs
     with substring `compiler.invoke`.
   - `test_substrate_script_not_edited`: fixture
     `driver_edits_substrate/` containing a top-level
     `diagnose.py` (which would shadow the substrate's) → assert
     validator FAILs with substring `must not contain a copy of
     _substrate/<name>.py`.
   - `test_threshold_profile_required`: fixture
     `missing_threshold_profile/` with no `threshold_profile` key →
     assert validator FAILs with substring `threshold_profile`.
2. Create the 3 fixture directories under
   `_substrate/tests/fixtures/`. Each is a minimal copy of the
   existing `good_driver/` fixture with the single invalidating
   mutation. Each has its own `manifest.json` + `idioms.json` +
   (for the `driver_edits_substrate` case) a stub `diagnose.py`.
3. Run the test suite — the 3 new tests are RED.
4. Extend `validate_backend.py:validate()`:
   - After the existing `manifest.capabilities` checks, add:
     ```python
     # invoke fields must equal the fixed filenames (spec §4.2 footnote)
     for role, fixed in [('compiler', 'build.sh'), ('runner', 'run.sh')]:
         block = manifest.get(role)
         if isinstance(block, dict):
             inv = block.get('invoke')
             if inv is not None and inv != fixed:
                 errors.append(
                     f"manifest.{role}.invoke '{inv}' must equal '{fixed}'")
     prof = manifest.get('profiler')
     if isinstance(prof, dict):
         inv = prof.get('invoke')
         if inv is not None and inv != 'profile.sh':
             errors.append(
                 f"manifest.profiler.invoke '{inv}' must equal 'profile.sh'")
         te = prof.get('to_evidence')
         if te is not None and te != 'to_evidence.py':
             errors.append(
                 f"manifest.profiler.to_evidence '{te}' must equal 'to_evidence.py'")
     # threshold_profile required (spec §4.4)
     if 'threshold_profile' not in manifest:
         errors.append("manifest.threshold_profile missing (required §4.4)")
     ```
   - After the existing idioms checks, add a substrate-shadow
     guard:
     ```python
     # A driver dir must not contain a copy of any _substrate/*.py script
     # (the driver is data; never imports or shadows substrate scripts).
     SUBSTRATE_SCRIPTS = {'evidence_schema.py', 'anti_cheat.py', 'diagnose.py',
                          'method_gate.py', 'memory_store.py', 'verify_insight.py'}
     for fname in os.listdir(driver_dir):
         if fname in SUBSTRATE_SCRIPTS:
             errors.append(
                 f"driver dir must not contain a copy of _substrate/{fname} "
                 f"(driver is data; never imports/shadows substrate)")
     ```
5. Re-run the test suite — the 3 new tests are GREEN, all
   pre-existing tests still GREEN (no behavioral change to today's
   cuda + triton drivers, verified next Task).

**Test command:**
```
cd _substrate && python3 -m pytest tests/test_validate_backend.py -v
```

**Exit criteria:**
- 3 new negative fixtures present.
- 3 new tests added; all green.
- All 4 pre-existing validator tests (positives + the original 4
  negatives) still green.
- `validate_backend.py` line count grows by ≤ 30 lines (the new
  checks are tight and additive).

---

### Task 7 — Re-validate existing cuda + triton drivers under new invariants

**Goal:** Prove the P3 cuda and triton drivers still PASS the extended
L0 validator. This is the master-plan exit criterion #2 ("returns L0
PASS on the cuda + triton P3 drivers").

**Files touched:**
- `_substrate/tests/test_driver_conformance.py` (extend with a single
  test method that re-runs the extended validator on the two real
  drivers).
- No production code touched.

**Pre-conditions:** Task 6 merged.

**Implementation steps:**

1. Add `test_real_drivers_pass_l0` to
   `_substrate/tests/test_driver_conformance.py` (or a new test file
   if cleaner). It iterates `['cuda', 'triton']`, shells out to
   `validate_backend.py _substrate/backends/<id>`, asserts
   exit-code 0 and `ok: true` in stdout JSON.
2. If a real driver FAILs (e.g. its `manifest.json` lacks
   `threshold_profile`, or has an `invoke` divergence), **the
   correct fix is to amend the driver's manifest**, NOT to weaken
   the validator. Make any such edits in the same Task 7 commit
   with a clear commit-message note.

**Test command:**
```
cd _substrate && python3 -m pytest tests/test_driver_conformance.py::TestRealDrivers -v
```

**Exit criteria:**
- Both `cuda` and `triton` drivers return `{ok: true, errors: []}`
  from the extended validator.
- Any minor `manifest.json` corrections to the real drivers are
  committed under the Task 7 commit with explicit reasoning.

---

### Task 8 — Template guidance edits + `validate-workflow.js` prompt checklist

**Goal:** Update the four template files' `{{TOKEN}}`/`[BLOCK]`
guidance + the input-policy comment (L70–75 of
`_meta/templates/iterative-loop.js`) to teach the LLM "the body
never names a vendor profiler or vendor metric; tokens come from
`idioms.json` via the driver". Add two warning-level checklist items
to `validate-workflow.js` (LLM-driven; explicitly not hard gates per
spec §9.2).

**Files touched:**
- `_meta/templates/iterative-loop.js` (input-policy comment + token
  reference table).
- `_meta/templates/search-based.js` (input-policy comment).
- `_meta/templates/single-pass.js` (input-policy comment).
- `_meta/templates/tree-exploration.js` (input-policy comment).
- `_meta/tools/validate-workflow.js` (append 2 checklist items).
- `_templates/{iterative-loop,search-based,single-pass,tree-exploration}.js`
  — append the 3-line deprecation header (mirror Task 1's policy on
  `_manifests/`); do NOT copy v1.1 content.

**Pre-conditions:** Task 7 merged.

**Implementation steps:**

1. `_meta/templates/iterative-loop.js`: edit the comment block at
   L70–75 (current text reproduced below) to add backend guidance:

   Current (verified):
   ```
   // Canonical input policy:
   // - If args.kernel_path is provided, optimize that existing kernel.
   // - Else require args.problem_definition or args.problem_path, generate seed_candidates initial kernels,
   //   verify them with test_command or benchmark_command, and optimize the best verified seed.
   // - Do not hardcode evaluator/compiler/profiler commands; consume user-provided command args.
   // - Return input_mode, generated_kernel_path, initial_candidates, and initial_generation_result.
   ```

   Append two new bullets:
   ```
   // - Backend (v1.1): the workflow body never names a vendor profiler (`nvcc`/`ncu`),
   //   a vendor metric (`sm_throughput_pct`/`dram_throughput_pct`), or a vendor idiom
   //   (`__global__`/`@triton.jit`/`PYBIND11_MODULE`). All such tokens come from the
   //   driver's `idioms.json` via the load-driver Setup agent (see spec §6.1/§6.2).
   // - Backend (v1.1): args.backend_dir gates the driver path; when empty the body
   //   falls back to the legacy inline-prompt path (USE_DRIVER = Boolean(BACKEND_DIR)).
   ```

   Then add 2 new tokens to the token reference (L11–37) — purely
   documentation:
   ```
   //   {{USE_DRIVER}}             — `Boolean(args.backend_dir)` gate (v1.1)
   //   {{DRIVER_DISPATCH_BLOCK}}  — §6.1 driverSh / driverPy path helpers (v1.1)
   ```

2. Apply the same 4-line input-policy bullet additions (the two
   `// - Backend (v1.1): …` lines) to the corresponding comment
   regions of `search-based.js`, `single-pass.js`, and
   `tree-exploration.js`. Each template has a parallel comment
   block; locate by `grep -n "Do not hardcode evaluator" _meta/templates/*.js`.

3. `_meta/tools/validate-workflow.js`: locate the existing checklist
   block (the agent `'check-args'` or similar — verify the label
   when implementing Task 8). Append two new check items, each
   producing a `severity: "warning"` violation (NOT error). Exact
   prompt text:

   ```
   - **backend-axis: clean-tier vendor-token leak (warning).** If
     manifest `backend.portability == 'clean'`, the workflow body
     must not contain any of: `nvcc`, `ncu`, `__global__`,
     `PYBIND11_MODULE`, `cuda_runtime.h`, `@triton.jit`, `tl.load`.
     Each occurrence in a prompt body is one warning.
   - **backend-axis: legacy supported_languages key (warning).** If
     manifest declares a v1.1 `backend:` block, the workflow body
     should emit `method_supported_backends` (and may emit
     `supported_languages` as a one-version-window alias). A bare
     `supported_languages` with no `method_supported_backends` is one
     warning, citing v1.1 schema migration.
   ```

   Per spec §9.2: these are LLM checklist items, NOT deterministic
   hard gates. The deterministic backend invariants live in
   `validate_backend.py` (Task 6).

4. Mirror-tree edits: append the 3-line deprecation header to each
   of `_templates/{iterative-loop,search-based,single-pass,tree-exploration}.js`
   identical in shape to Task 1's `_manifests/schema.yaml`
   deprecation. **No** v1.1 content copy.

**Tests:** This Task ships no new test (templates are read by LLM
agents, not parsed by code; `validate-workflow.js` is itself a
workflow `.js` whose only structural assertion is the prompt body
content, covered by the regression test in Task 5 if any agent label
or schema drifts).

The merge gate: re-run the full `node --test _meta/tools/test/*.test.js`
suite and confirm green; in particular Task 5's
`generator-prompt-schema.test.js` must still pass.

**Exit criteria:**
- 4 template files have the two new `// - Backend (v1.1): …`
  bullets in their input-policy comment block.
- `iterative-loop.js` has the 2 new tokens in its token reference.
- `validate-workflow.js` has the 2 new checklist items (verifiable
  by `grep -c "backend-axis:" _meta/tools/validate-workflow.js` ≥
  2).
- All `_templates/*.js` mirrors have the deprecation header (no v1.1
  content).
- `node --test _meta/tools/test/*.test.js` green.

---

## 5. Execution units (commit clusters)

Tasks group into **4 commit clusters**. Each cluster is one PR or a
contiguous push to `dev/p5a-impl` (a branch off this plan's branch).

### Cluster A — Schema landing + back-compat proof (Tasks 1+2)

- Commit A1: `docs(schema): bump _meta/manifests/schema.yaml to v1.1
  with backend: block + arg additions (P5a Task 1)`
- Commit A2: `test(schema): pin v1.1 manifest parse + back-compat
  (AccelOpt fixture); 13 existing manifests still parse green (P5a
  Task 2)`

Cluster A is independently mergeable; it makes the schema doc-edit
live and proves the back-compat invariant.

### Cluster B — Generator prompt edits (Tasks 3+4+5)

- Commit B1: `test(generator): pin agent schemas of generate-workflow.js
  — captures opts.schema in run-workflow stub; RED before B2/B3 (P5a
  Task 5 / TDD anchor)`
- Commit B2: `feat(generator): teach model-args prompt the v1.1
  backend-axis args; demote ncu_command to deprecated alias (P5a Task
  3)`
- Commit B3: `feat(generator): teach generate prompt the §6.4 split +
  §6.1 driver-dispatch + §6.2 load-driver; tighten Rule 10
  vendor-token ban; correct L394 stale path (P5a Task 4)`

Cluster B follows TDD: the test commit is RED at B1 (because the
captured `schema` field is missing); B1+B2+B3 land green together.

### Cluster C — Validator hardening + driver re-validation (Tasks 6+7)

- Commit C1: `test(validate-backend): add 3 negative fixtures
  (invoke-divergence, substrate-shadow, missing threshold_profile);
  RED before C2 (P5a Task 6)`
- Commit C2: `feat(validate-backend): add invoke/threshold_profile/
  substrate-shadow L0 invariants (P5a Task 6)`
- Commit C3: `test(drivers): re-validate cuda + triton under extended
  L0 — manifest corrections if any (P5a Task 7)`

### Cluster D — Template guidance + validator checklist (Task 8)

- Commit D1: `docs(templates): teach v1.1 backend-axis input-policy
  across all 4 templates; add 2 USE_DRIVER tokens; deprecate
  _templates/ mirror header (P5a Task 8)`
- Commit D2: `feat(validate-workflow): add 2 backend-axis prompt
  checklist warnings (clean-tier vendor-token leak;
  supported_languages legacy key) (P5a Task 8)`

---

## 6. Definition of done

P5a is complete when ALL of the following hold simultaneously:

### Test counts (no GPU; no network)

- `_substrate/tests/test_validate_backend.py`: pre-P5a 4 negatives +
  positives + **3 new negatives** = 7+ negative cases, all green.
- `_substrate/tests/test_driver_conformance.py`: real cuda + triton
  drivers L0 PASS under extended validator (1 new test method).
- `_substrate/tests/test_manifest_parse_v11.py`: 13 manifest files
  (excluding schema.yaml) parse v1.1, AccelOpt pinned as legacy
  back-compat fixture, 1 synthetic v1.1-populated fixture parses
  + asserts.
- `_meta/tools/test/generator-prompt-schema.test.js`: green; pins
  ≥ 3 agent `schema:` definitions (`model-args`,
  `assemble-manifest`, `generate-workflow`).
- `_meta/tools/test/print-workflow-prompts.test.js`: still green
  + new assertion that captured calls include a `schema` key.
- All P4-shipped tests (`accelopt-cuda-byte-identity`,
  `accelopt-guard`, `accelopt-triton-dryrun`): still green.

### Byte-identity proofs

- `_meta/tools/generate-workflow.js`: every agent `schema:` literal
  deep-equal to its pre-P5a value (pinned by Task 5's test).
- `_meta/manifests/*.yaml` (the 13 existing manifests): byte-identical
  to pre-P5a (P5a does not retrofit any manifest's `backend:` block;
  P5b/c/d do).
- `_substrate/*.py` (the 6 universal substrate scripts): byte-identical
  to pre-P5a (P5a touches no substrate script).
- `AccelOpt/accelopt-kernel-optimization.js` + every other workflow
  `.js`: byte-identical to pre-P5a (iron rule: no workflow body
  edits in this sub-plan).

### Documentation updates

- `_meta/manifests/schema.yaml` v1.1 header changelog mentions the
  `ncu_command` deprecation window (v1.1 → warn at v1.2 → remove
  at v1.3).
- `_manifests/schema.yaml` carries the 3-line deprecation header.
- `_templates/*.js` mirrors carry the 3-line deprecation header.
- This plan committed to `docs/superpowers/plans/`.

### Spec/SDK reconciliation

- No new edits to `_substrate/BACKEND-DRIVER-SDK.md` or
  `_substrate/backends/REGISTRY.md` in P5a — those are P5f's scope.
  Cross-reference note added as a `TODO(P5f)` comment in the SDK
  doc's "Deviations" section is **optional**; not required for P5a
  exit.

---

## 7. Open risks (post-spike) and mitigations

### R1 — `run-workflow.js` `schema` capture is structural, not behavioral

Task 5's regression test depends on `run-workflow.js:agentStub`
pushing `opts.schema` into the captured `calls[]`. Today it does
not. This is a 1-line additive change with no downstream consumer
of the existing `{seq, label, phase, prompt}` shape that destructures
keys (verified: all P4 tests destructure named fields and tolerate
extras).

**Mitigation:** Task 5's first sub-commit is the
`run-workflow.js` extension + a print-workflow-prompts test asserting
the new key. If a downstream consumer breaks (none currently
identified), the fix is a single test-file update, not a redesign.

### R2 — `_manifests/` mirror deprecation header risks confusion

A user reading `_manifests/schema.yaml` might infer the file is
abandoned, then look for v1.1 in the wrong place. The 3-line header
explicitly points to `_meta/manifests/schema.yaml`; risk is bounded.

**Mitigation:** P5f formalizes the duplication note in the SDK doc;
nothing further needed in P5a.

### R3 — Spec §9.2/§9.3 references to `_meta/tools/validate-backend.js` still wrong-shaped

Spec text names a Node validator that P5a deliberately does not ship.
The SDK doc already records the deviation; the spec itself is the
slow-moving artifact.

**Mitigation:** P5f's spec-correction commit explicitly updates §9.2
and §9.3 to name `_substrate/backends/validate_backend.py`. P5a adds
a short comment in `_substrate/BACKEND-DRIVER-SDK.md`'s "Deviations"
section reaffirming the decision and pointing to this P5a plan as
the executor of record. **Optional in P5a**; required in P5f.

### R4 — Generator-prompt edits may degrade non-clean-tier generation

The Rule 10 tightening forbids `nvcc`/`ncu` literals in **clean-
tier** workflow bodies. The rule is conditioned on `manifest
backend.portability == 'clean'`. For vendor_locked workflows
(AccelOpt-style), the literals remain permitted. The prompt's
condition language ("For workflows with manifest
`backend.portability: clean`…") makes this explicit.

**Mitigation:** Task 5's regression test pins the **schema keys**,
not the prompt body content. Body content drift is accepted as a
known limitation (master plan §6 "Generator-prompt drift"). When
P5b/c/d retrofits hit the first regression, the fix is a per-
workflow manifest `backend.portability` setting + prompt-body
review.

### R5 — `validate_backend.py` substrate-shadow guard may false-positive on incidental files

The new guard refuses any driver dir containing a filename in
`SUBSTRATE_SCRIPTS`. If a driver author legitimately adds (e.g.)
`anti_cheat_patterns.txt` (which is the `vendor_patterns_file`
contract — and **does not collide** because the extension is `.txt`),
no false positive. If someone adds a true `anti_cheat.py` in the
driver dir, that IS the failure mode the guard exists to catch — by
design, no false positive.

**Mitigation:** the SUBSTRATE_SCRIPTS set is exhaustive (6 names);
new substrate scripts are explicitly forbidden by spec §3.1.
Updating the set requires a substrate-scope change which is
out-of-band of P5a.

---

## 8. What this plan deliberately does NOT decide

- Per-workflow `backend:` block contents — each P5b/c/d retrofit
  picks `supported[]`/`default`/`matrix_eligible`/`portability` per
  workflow.
- The complete set of agent labels pinned by the regression test —
  Task 5 pins 3 (`model-args`, `assemble-manifest`,
  `generate-workflow`); future generator agents may need pinning,
  added by their own plans.
- Spec-text corrections to §9.2/§9.3 — owned by P5f.
- The dedup of `_meta/` ↔ `_` mirror trees — out of scope.
- GPU verification — opt-in tier inherited from P4.

---

## Appendix — File-path quick reference

All paths absolute from repo root
`/Users/haiyan/Documents/Infinity/Agent4Kernel/Awesome-Kernel-Workflows/`.

| Touched | Action | Task |
|---|---|---|
| `_meta/manifests/schema.yaml` | EDIT (v1.0 → v1.1) | 1 |
| `_manifests/schema.yaml` | EDIT (deprecation header) | 1 |
| `_substrate/tests/test_manifest_parse_v11.py` | NEW | 2 |
| `_substrate/tests/fixtures/manifest_v11_minimal.yaml` | NEW | 2 |
| `_meta/tools/generate-workflow.js` (L263–282) | EDIT | 3 |
| `_meta/tools/generate-workflow.js` (L394 + L483–525) | EDIT | 4 |
| `_meta/tools/test/generator-prompt-schema.test.js` | NEW | 5 |
| `_meta/tools/fixtures/generate-workflow-args.json` | NEW | 5 |
| `_meta/tools/fixtures/generate-workflow-agent-returns.json` | NEW | 5 |
| `_meta/tools/lib/run-workflow.js` (agentStub +1 line) | EDIT | 5 |
| `_meta/tools/test/print-workflow-prompts.test.js` | EDIT (+1 assertion) | 5 |
| `_substrate/backends/validate_backend.py` | EDIT (+L0 invariants) | 6 |
| `_substrate/tests/test_validate_backend.py` | EDIT (+3 tests) | 6 |
| `_substrate/tests/fixtures/bad_invoke_compiler/` | NEW (dir) | 6 |
| `_substrate/tests/fixtures/driver_edits_substrate/` | NEW (dir) | 6 |
| `_substrate/tests/fixtures/missing_threshold_profile/` | NEW (dir) | 6 |
| `_substrate/tests/test_driver_conformance.py` | EDIT (+1 test) | 7 |
| `_substrate/backends/cuda/manifest.json` | EDIT (if needed) | 7 |
| `_substrate/backends/triton/manifest.json` | EDIT (if needed) | 7 |
| `_meta/templates/iterative-loop.js` | EDIT (comment + tokens) | 8 |
| `_meta/templates/search-based.js` | EDIT (comment) | 8 |
| `_meta/templates/single-pass.js` | EDIT (comment) | 8 |
| `_meta/templates/tree-exploration.js` | EDIT (comment) | 8 |
| `_meta/tools/validate-workflow.js` | EDIT (+2 checklist items) | 8 |
| `_templates/iterative-loop.js` | EDIT (deprecation header) | 8 |
| `_templates/search-based.js` | EDIT (deprecation header) | 8 |
| `_templates/single-pass.js` | EDIT (deprecation header) | 8 |
| `_templates/tree-exploration.js` | EDIT (deprecation header) | 8 |
| `docs/superpowers/plans/2026-06-08-p5a-schema-and-generator.md` | NEW (this file) | — |

**Total touched files: 30.** **Total new files: 9.** **No workflow
`.js` body edited.** **No `_substrate/*.py` substrate script edited
beyond `validate_backend.py`.**
