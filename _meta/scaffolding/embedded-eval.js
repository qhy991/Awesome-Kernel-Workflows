// embedded-eval.js — CANONICAL default-scaffolding snippet for the embedded-
// dispatch evaluation substrate (the contract + the eval-plan builder).
//
// This is NOT a runnable workflow. It is the single source of truth for the
// EMBEDDING_CONTRACT (the prompt preamble that tells an LLM it is authoring a
// kernel that lives INSIDE a larger project) + __embeddedEvalPlan (builds the
// register/build/test/benchmark/unregister command sequence for a candidate
// against a project's adapter). Inlined by 6 workflows (ARGUS, CUDAAgent, FACT,
// GPUForecasters, Generalist, StitchCUDA) — byte-identical across all 6.
//
// WHY: the block was inlined by `scripts/patch-embedded-eval.js` with no SSOT
// file (the codemod held the canonical text). This file gives it a home —
// matching the agent-retry.js / arg-guard.js / backend-axis.js convention — so a
// fix to the contract or the eval-plan builder is one edit here + a
// `patch-embedded-eval.js --refresh` to propagate.
//
// CONSTRAINT: the Workflow runtime forbids `Date.now()` / `Math.random()` /
// argless `new Date()`. This helper uses none. `ctx.params`/`unregParams` are
// opaque pass-through strings the substrate does not parse.
//
// USAGE (inline near the top of a workflow that supports embedded dispatch):
//
//   // --- BEGIN inlined embedded-eval scaffolding (from _meta/scaffolding/embedded-eval.js) ---
//   <paste EMBEDDING_CONTRACT + __embeddedEvalPlan>
//   // --- END inlined embedded-eval scaffolding ---
//
//   const plan = __embeddedEvalPlan({ adapter, variant, source, projectRoot, buildCmd, testCmd, benchmarkCmd })

const EMBEDDING_CONTRACT = [
  'EMBEDDED-DISPATCH CONTRACT (this kernel is NOT standalone):',
  '',
  'You are authoring a kernel that lives INSIDE a larger project and is wired into',
  'its dispatch table. It cannot be compiled on its own. Therefore:',
  '',
  '1. Emit a COMPLETE source file (e.g. a .cuh) that matches the reference',
  '   dispatch signature exactly -- same entry-point shape, template params, and',
  '   launch-bounds conventions as the reference file. Do NOT add a main(), a',
  '   standalone harness, or top-level test code.',
  '2. Use ONLY symbols/headers the project already provides (project headers,',
  '   template instantiations, dispatch macros). Do not invent include paths.',
  '3. Do NOT register, build, or benchmark the variant yourself, and do NOT name',
  '   any symbol with the variant suffix -- the workflow + adapter handle wiring.',
  '4. Return ONLY the file contents plus a short rationale citing the concrete',
  '   design choice (tile shape, register budget, pipelining, GQA packing, etc.).',
].join('\n')

// Build the ordered evaluation commands for one candidate against a
// contract-conforming adapter. All fields are plain strings the caller already
// resolved from `args`. `params`/`unregParams` are opaque pass-through strings
// (e.g. "--dkq 256 --dv 256 --cmake-build-dir /p/build") that the substrate does
// not parse -- they belong to the project's adapter.
function __embeddedEvalPlan(ctx) {
  const adapter = ctx.adapter                       // e.g. 'python "/abs/llamacpp_register_variant.py"'
  const variant = ctx.variant                       // unique variant name for this candidate
  const source = ctx.source                         // path to the candidate source file on disk
  const root = ctx.projectRoot                       // --project-root
  const params = ctx.params || ''                    // opaque register params pass-through
  const unregParams = ctx.unregParams || ''          // opaque unregister params pass-through
  const q = (s) => `"${s}"`
  const reg = `${adapter} register --variant ${variant} --source ${q(source)} --project-root ${q(root)}${params ? ' ' + params : ''}`.trim()
  const unreg = `${adapter} unregister --variant ${variant} --project-root ${q(root)}${unregParams ? ' ' + unregParams : ''}`.trim()
  const list = `${adapter} list --project-root ${q(root)}`
  return {
    register: reg,
    list,
    // Project-native build/test/benchmark, run VERBATIM with the variant's env
    // gate set so the project binary dispatches to this candidate.
    build: ctx.buildCmd ? `KERSOR_VARIANT=${variant} ${ctx.buildCmd}` : '',
    test: ctx.testCmd ? `KERSOR_VARIANT=${variant} ${ctx.testCmd}` : '',
    benchmark: ctx.benchmarkCmd ? `KERSOR_VARIANT=${variant} ${ctx.benchmarkCmd}` : '',
    unregister: unreg,
    // Human-orderable sequence + the non-negotiable cleanup invariant.
    order: ['register', 'list', 'build', 'test', 'benchmark', 'unregister'],
    cleanupInvariant: `On ANY failure or non-improvement, run the unregister command and confirm via list that ${variant} is gone, leaving the project byte-exact pristine.`,
  }
}
