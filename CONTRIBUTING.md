# Contributing to Awesome-Kernel-Workflows

Guide for authoring new workflows and avoiding recurring failure modes.

## Quick start

```bash
# From an existing KerSor session or manually:
node _tools/generate-workflow.js  # or _meta/tools/generate-workflow.js
```

The scaffold templates already avoid every anti-pattern below by construction. If you're writing a workflow from scratch or migrating an old one, read on.

## CI is your safety net, not a substitute for knowing the rules

Every anti-pattern in this document is enforced by a CI test — a PR that violates any of them will hard-fail. **The point of this document is to save you the round-trip.** You can always trust CI to catch mistakes; you'll ship faster if you don't make them.

Run tests locally before pushing:

```bash
node --test _meta/tools/test/*.test.js
node scripts/check-canonical-args.js
./scripts/validate-manifests.sh
```

---

## Anti-patterns

### 1. Workflow-runtime forbidden APIs

**Symptom:** Workflow dispatches successfully, agent runs for a while (30–100K tokens), then crashes silently on resume or produces inconsistent results across replays.

**Root cause:** The Workflow tool runtime forbids `Date.now()`, `Math.random()`, and argless `new Date()` — they return different values on resume, causing cached branches to diverge from live execution. AWK issue #22 (KernelAgent).

**Don't:**
```js
const sessionDir = `${EXP_DIR}/session_${Date.now()}`
const jitter = Math.random() * 0.1
const startedAt = new Date().toISOString()
```

**Do:**
```js
const sessionDir = `${EXP_DIR}/session_${args.run_index ?? args.round_index ?? 'run'}`
// If you truly need randomness, take a seed via args and use a deterministic PRNG.
// Don't emit timestamps at all — the trace/ledger handles timing.
```

Also forbidden: **line-leading ESM `import`** — the entrypoint is loaded as a script, not a module.

**Enforcement:**
- `_meta/tools/test/kernelagent-runtime-patterns.test.js` (source-level assertion; extend to your workflow)
- KerSor `scripts/generate-catalog.sh` forbidden-API scan marks offending workflows `known_broken`; the selector then hard-vetoes them

**Related:** KerSor #31 subtype 1.

---

### 2. `assertWorkflowSuitability` in workflow code

**Symptom:** Workflow gets dispatched by KerSor selector (which reads `manifest.yaml`), then throws mid-dispatch when its own JS predicate rejects the same args KerSor already vetted. Wastes the round.

**Root cause:** Workflows used to re-implement eligibility logic in a `WORKFLOW_SUITABILITY` const + `assertWorkflowSuitability()` function — a dual source of truth with `manifest.yaml`. When the two disagreed (e.g. KDA's manifest `[cuda, cutlass, rocm, metax, ascendc]` vs its JS `['cuda', 'ascendc']`), the selector picked, the workflow rejected. AWK #24 eliminated this.

**Don't:**
```js
const WORKFLOW_SUITABILITY = {
  supported_languages: ['cuda'],
  supported_problem_types: ['cuda-kernel-optimization'],
}
function assertWorkflowSuitability(args) {
  if (!WORKFLOW_SUITABILITY.supported_languages.includes(args.language)) {
    throw new Error(`unsupported language ${args.language}`)
  }
}
assertWorkflowSuitability(args)
```

**Do:** Declare eligibility in `manifest.yaml`:
```yaml
routing:
  languages: [cuda]
  backends: [cuda]
  accepts:
    problem_type:
      - cuda-kernel-optimization
    constraints: []
```

If you need runtime backend resolution (e.g. AccelOpt-style), emit a small `resolveBackend()` helper — but it must **not throw** on eligibility; it only normalizes/derives. Keep `normalizeSuitabilityValue` only if used for arg-conflict checks (e.g. `cuda` ↔ `cuda_cpp` aliasing).

**Enforcement:**
- `_meta/tools/test/routing-accepts-single-source.test.js` — no `assertWorkflowSuitability` in any workflow `.js`
- `scripts/check-canonical-args.js` — the anti-pattern is in the "forbidden snippets" list

**Related:** KerSor #31, KerSor PR #35 (selector consumes `accepted_problem_types`).

---

### 3. `manifest.yaml` without `routing.accepts:` or with an empty `problem_type`

**Symptom:** Your workflow's `known_broken` is null in the catalog but it never gets picked; or it gets picked for every task including totally wrong ones.

**Root cause:** KerSor selector uses `routing.accepts.problem_type` as a hard filter. Empty list = "accepts any" (selector will not veto on this axis). Missing declaration = your workflow behaves like `accepts_any` today, but this is fragile — if the axis becomes required, your workflow silently becomes broken.

**Do:** Always declare, match your workflow's actual capability:

```yaml
routing:
  accepts:
    problem_type:
      - cuda-kernel-optimization     # specific: <backend>-kernel-<mode>
      - cuda-kernel-generation
      - gpu-kernel-optimization       # neutral: gpu-kernel-<mode> (cross-vendor)
      - kernel-generation             # universal: kernel-<mode> (task-only)
    constraints: []                   # forward-compat; no constraint declared today
```

**Task mode** is `optimization` when `seed_origin=provided_kernel`, else `generation`. **Only declare tokens your workflow can actually handle.**

**Enforcement:** KerSor `tests/test_routing_accepts_filter.py` — selector hard-vetoes with a recorded reason when your session's derived `problem_type` is in none of your listed accepts.

**Related:** KerSor PR #35, AWK #24.

---

### 4. Substrate CLI: wrong flag names

**Symptom:** Workflow calls substrate `run.sh`, immediately gets `err_envelope "unknown arg" 3`. Every dispatch fails identically.

**Root cause:** Each substrate declares its accepted flags in `_substrate/backends/<B>/flags.yaml`; the `run.sh` parser is generated from that schema. Workflows historically passed `--kernel/--test/--result` — none of cuda/metal/metax/triton accept them. AWK #25 established the SSOT and fixed workflow call-sites; new workflows must not regress.

**Don't:**
```js
driverSh('run.sh', `--kernel ${kernelFilename()} --test ${testFilename()}`)
```

**Do:**
```js
driverSh('run.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${resultPath}`)
```

Baseline substrate flag set (all 6 substrates): `--artifact --problem --out --reps --rtol --atol --baseline`. `ascend` additionally accepts `--kernel --op --language --mkb-root --eval-runner --baseline-latency` (plus `--result` as an alias for `--out`). `rocm` additionally accepts `--bench-cmd --correct-regex --latency-regex --test-cmd`. Read your target substrate's `flags.yaml` — never guess.

**Enforcement:**
- `_meta/tools/test/substrate-flags-contract.test.js` — part (b) is enforcing: any `driverSh('run.sh', ...)` call with a flag not in the substrate schema hard-fails
- The substrate `run.sh` itself will `exit 3` on unknown flags at dispatch time (immediate failure, not silent)

**Related:** KerSor #31 subtype 3 (declared no-op — AKW CI is the enforcer).

---

### 5. Unwrapped `await agent()` calls

**Symptom:** Workflow runs fine for 5 rounds, then crashes with `TypeError: Cannot read properties of null (reading 'bottleneck_class')`. A transient 429 or model unavailability made `agent()` return null; the next line dereferenced it.

**Root cause:** `agent()` returns null on terminal API errors after retries. Bare property access blows up. `agentRetry()` handles this: bounded retries with backoff on transient failures, explicit `allowNull` semantics if you can proceed without a result. AWK #20.

**Don't:**
```js
const diag = await agent('diagnose.py', {...})
const bclass = diag.bottleneck_class  // BOOM on null
```

**Do:**
```js
const diag = await agentRetry(
  () => agent('diagnose.py', {...}),
  { retries: 3, allowNull: false }   // hard fail if all retries return null
)
const bclass = diag.bottleneck_class

// Or, when null is acceptable:
const optional = await agentRetry(
  () => agent('optional_analysis.py', {...}),
  { retries: 2, allowNull: true }
)
if (optional) { /* use optional.field */ }
```

**Enforcement:**
- `_meta/tools/test/agent-retry-guard-lint.test.js` — lints every `await agent(` and requires it to be inside `agentRetry(`
- `_meta/tools/test/agent-retry-null-safety.test.js` — catches bare property access on `agentRetry` results without a null check when `allowNull: true`

**Migration helper:** `scripts/add-agent-retry-scaffolding.js` auto-wraps existing calls (used during AWK #20 rollout).

**Related:** AWK #20.

---

### 6. CWD pollution

**Symptom:** After a workflow runs, files like `_dev/`, `solution.py`, `kernels/*.cu` appear in the KerSor session root or user's project directory instead of `${EXP_DIR}/run-N/`.

**Root cause:** Workflow shell commands or agent tool calls that don't explicitly chdir or path-prefix write to the process CWD (which is whatever the orchestrator was in). `collect-stray-outputs.sh` sweeps these up post-hoc, but the pollution is visible to users and can hide real state.

**Don't:**
```js
`bash -c 'nvcc kernel.cu -o kernel_bin && ./kernel_bin > result.json'`
```

**Do:**
```js
`bash -c 'cd ${EXP_DIR}/run-${round} && nvcc kernel.cu -o kernel_bin && ./kernel_bin > result.json'`
// or, better, use full paths:
`nvcc ${EXP_DIR}/run-${round}/kernel.cu -o ${EXP_DIR}/run-${round}/kernel_bin`
```

Every substrate command already uses `--out ${resultPath}` where `resultPath` is under `EXP_DIR`. Your own build/eval steps should do the same.

**Enforcement:** No specific test; caught by human review or observed as stray files in `.kersor/<session>/` root. `collect-stray-outputs.sh` moves them into place post-hoc as a safety net.

---

### 7. Hardcoded model tier

**Symptom:** Workflow fails on some deployments because it requires a model tier that's unavailable (e.g. `haiku` on an infini-ai proxy without the haiku model).

**Root cause:** Workflows historically pinned `model: 'haiku'` for mechanical tasks. When the tier is unavailable, the entire workflow degrades. AWK #19 defaults `model_mechanical` to `sonnet`.

**Don't:**
```js
{ model: 'haiku', ... }
{ model: 'claude-haiku-4-5', ... }
```

**Do:**
```js
{ model: args.model_mechanical || 'sonnet', ... }
// or inherit from the parent workflow model:
{ /* no model field — inherits main-loop model */ }
```

**Enforcement:** No blocking test; recommended patch: use `scripts/patch-model-tier.js` when migrating.

**Related:** AWK #19.

---

## Tooling map

| Anti-pattern | CI test | Auto-fix / scaffold |
|---|---|---|
| Forbidden APIs (§1) | `kernelagent-runtime-patterns.test.js` + KerSor catalog scan | — |
| `assertWorkflowSuitability` (§2) | `routing-accepts-single-source.test.js` + `check-canonical-args.js` | Scaffold rule 2 |
| `routing.accepts` shape (§3) | `test_routing_accepts_filter.py` (KerSor) | Scaffold rule 2a |
| Substrate flag names (§4) | `substrate-flags-contract.test.js` (part b) | Scaffold rule 2a |
| `agentRetry` wrapping (§5) | `agent-retry-guard-lint.test.js` + `agent-retry-null-safety.test.js` | `scripts/add-agent-retry-scaffolding.js` |
| CWD pollution (§6) | — (human review) | Scaffold rule 2a template shows `${EXP_DIR}` paths |
| Model tier (§7) | — | `scripts/patch-model-tier.js` |

---

## Before submitting

- [ ] Ran `node --test _meta/tools/test/*.test.js` locally — all green
- [ ] Ran `node scripts/check-canonical-args.js` — no missing snippet warnings
- [ ] Ran `./scripts/validate-manifests.sh` and `./scripts/count-workflows.sh` — no errors
- [ ] `routing.accepts.problem_type` in your manifest is non-empty and matches capability
- [ ] No `Date.now()`, `Math.random()`, `new Date()` (argless), or ESM `import` in the workflow `.js`
- [ ] Every `await agent(...)` wrapped in `agentRetry(...)`
- [ ] All substrate `driverSh('run.sh', ...)` calls use only flags declared in the target substrate's `flags.yaml`
- [ ] All file writes go under `${EXP_DIR}`
- [ ] No hardcoded `model: 'haiku'` — use `args.model_*` or omit

## References

- Runtime rules and Workflow tool contract: KerSor `docs/integration-contract.md`
- SSOT-judgment criteria (when to unify sources of truth, when not to): KerSor `docs/integration-contract.md` §"Judging SSOT proposals"
- Closed issues that shaped these rules: AWK #12, #13, #19, #20, #22, #24, #25; KerSor #31, #32, #33, #34
