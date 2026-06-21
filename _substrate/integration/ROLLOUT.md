# Integration-strategist rollout — COMPLETE (23/29 workflows wired)

> **Status: DONE.** 23 of 29 workflows now wire the integration-strategist (all
> optimizers). All pass wfcheck + the bug-class guard (0 findings). The recipe below
> is kept as the reference for future workflows + the guard is the permanent safety net.

Goal: let every workflow handle inference-engine embedded operators (llama.cpp / PyTorch
aten / vLLM …), not just Generalist. Generalist is the **GPU-validated reference**
(CONFIRM v2/v5); this document was the recipe to apply that treatment to the other
~23 standalone-default workflows, and the `tests/test_workflow_embedded_safety.py`
guard is the net that keeps the rollout honest.

## The treatment = 7 components

| # | Component | Shared snippet? | Per-workflow? |
|---|-----------|-----------------|---------------|
| 1 | integration-strategist gate (can_standalone → method) | insertable | where Setup lives |
| 2 | `USE_DRIVER_STANDALONE` / skip standalone build when embedded | — | gate the existing build.sh/run.sh envelope |
| 3 | embedded eval branching (inplace: backup+restore; dispatch: `__embeddedEvalPlan`) | `__embeddedEvalPlan` already shared via `scripts/patch-embedded-eval.js` | wire the branch |
| 4 | **serial** eval for embedded (NOT parallel) | — | gate the candidate-eval loop |
| 5 | profiling-strategist full Profile branching + native→perf downgrade | insertable | where the Profile phase is |
| 6 | perf_heuristic Profile writes `heuristic_bclass` (diagnose contract, dacbd4f) | prompt line | in the perf_heuristic branch |
| 7 | manifest `integration_patterns: [standalone, embedded_dispatch, embedded_inplace]` | manifest edit | — |

Components 1, 5, 6 are **insertable snippets** (see below). Components 2, 3, 4 are
**structural** — each workflow's eval topology differs, so they are guided by the
checklist + enforced by the guard, not auto-patched. Component 7 is a manifest edit.

## Insertable snippets (copy from Generalist, adapt paths)

### 1. integration-strategist gate (Setup)
```js
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _profManifest = (USE_DRIVER && BACKEND_DIR) ? `${BACKEND_DIR}/manifest.json` : `${SUBSTRATE}/backends/cuda/manifest.json`
  const _integ = await agent(`Read ${KERNEL_PATH}; classify can_compile_standalone (yes|no|uncertain) ... Then `
    + substrateInstruction('integration/integration_strategist.py',
      `resolve --kernel "${KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '<json>' --cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl`)
    + ` Return stdout JSON verbatim.`, { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH })
  if (_integ && _integ.method) INTEGRATION_DECISION = _integ
}
const USE_DRIVER_STANDALONE = USE_DRIVER && INTEGRATION_DECISION.method === 'standalone'
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) { await agent(`cp -a "${KERNEL_PATH}" "${ORIGINAL_BACKUP}" ...`, {...}) }
```

### 2 + 4. serial embedded eval (gate the candidate loop)
```js
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
let evaluated
if (IS_EMBEDDED) {
  evaluated = []
  for (let i = 0; i < plans.length; i++) {
    try { const r = await evalOne(plans[i], i); if (r) evaluated.push(r) }
    catch (e) { log(`cand ${i+1} embedded eval failed: ${e.message||e}`) }
  }
} else {
  evaluated = (await parallel(plans.map((p, i) => () => evalOne(p, i)))).filter(Boolean)
}
```

### 5. profiling-strategist full Profile branching + native→perf downgrade
```js
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }
{ /* resolve via profiling/profiling_strategist.py — see Generalist */ if (_pd?.method) PROFILING_DECISION = _pd }
// A-O1 closure: strategist may pick native_profiler (which ncu) but ncu_command was stripped by
// KerSor preflight (K-O2) because counters blocked — downgrade so Profile uses perf_heuristic.
if (PROFILING_DECISION.method === 'native_profiler' && !NCU_CMD) {
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', ... }
}
// then in the Profile phase, branch on PROFILING_DECISION.method — NOT a hardcoded ncu_command path.
```

### 6. perf_heuristic Profile writes heuristic_bclass (dacbd4f contract)
In the perf_heuristic Profile branch, instruct the agent to also write
`heuristic_bclass` / `bottleneck_hint` (qualitative: memory/compute/latency-bound) so
`diagnose.py` does not fall to `unknown`.

## Per-workflow checklist (apply to each rollout target)

1. [ ] Add the integration-strategist gate (snippet 1) at Setup.
2. [ ] Gate the existing standalone build/run envelope on `USE_DRIVER_STANDALONE`.
3. [ ] Add embedded eval branching (snippet: inplace backup+restore; dispatch via `__embeddedEvalPlan` — run `scripts/patch-embedded-eval.js <Workflow>` first to inline it).
4. [ ] **Make embedded eval serial** (snippet 2+4). This is the #1 bug-class — do NOT leave `parallel(` for embedded.
5. [ ] Add `ORIGINAL_BACKUP` (once, Setup) + unconditional exit restore for embedded_inplace.
6. [ ] Wire the full profiling-strategist Profile branching + native→perf downgrade (snippet 5).
7. [ ] perf_heuristic Profile writes `heuristic_bclass` (snippet 6).
8. [ ] manifest: add `integration_patterns: [standalone, embedded_dispatch, embedded_inplace]`.
9. [ ] `wfcheck` + `tests/test_workflow_embedded_safety.py` GREEN.
10. [ ] GPU smoke (1 embedded kernel, e.g. mmq.cuh) before relying on it.

## Safety rules the guard enforces (`tests/test_workflow_embedded_safety.py`)

- **parallel-embedded-race**: embedded eval wired + `await parallel(` + no serial gate → FAIL.
- **inplace-no-restore**: `embedded_inplace` without `ORIGINAL_BACKUP` + exit restore → FAIL.
- **profile-ignores-strategist**: `PROFILING_DECISION` resolved but never branched on (`.method`) → FAIL.

A newly-wired workflow MUST pass this guard. It catches the three Generalist bug-classes
without per-workflow human review.

## Rollout order (cohorts)

1. **Driver-path family** (shared build.sh structure, closest to Generalist):
   AKO4X, KDA, Astra, KernelFoundry, KernelFoundryDx, AdaExplore, KSearch, ReGraphT,
   CUDALLM, STARK, StitchCUDA, KernelBand, KernelSkill, AccelOpt. (14)
2. **Bespoke raw-ncu family** (inline ncu, no driver): cuPilot, CutlassGEMM, KEET,
   KernelBlaster, WarpSpeed. (5)
3. **Already embedded-capable** (have `__embeddedEvalPlan` inlined, need the gate +
   profiling branching to be Generalist-level): ARGUS, CUDAAgent, FACT, GPUForecasters. (4)
4. Generators (TritorX, KernelAgent): standalone-only generators — embedded generation is
   a separate question (no dispatch-wiring for generated kernels yet). Defer.

Per cohort: wire → guard + wfcheck GREEN → 1 GPU smoke → next.

## The non-architecture risk (does NOT go away with rollout)

CONFIRM v5 showed **~80% of the non-determinism is the third-party MaaS proxy model**
(glm-5.2 StructuredOutput death-loops), NOT the wiring. Rolling out to 23 workflows =
**23× exposure** to that instability. The wiring is correct; the execution substrate
(model) is flaky. Before relying on rolled-out workflows, EITHER run them on a stable
native model OR add model-instability guards (StructuredOutput retry caps / downgrade).
The guard + checklist make the wiring SAFE; they do not make the model stable.
