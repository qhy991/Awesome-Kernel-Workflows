// _substrate/embedded/embedded_eval.js
//
// Shared "embedded-dispatch" evaluation substrate. A workflow uses this to
// evaluate a candidate kernel that cannot be compiled standalone -- it must be
// registered into a project's dispatch table, built by the project's own build
// system, tested/benchmarked, then unregistered byte-exact.
//
// Two roles, like _substrate/grounding.js:
//
//   1. Canonical source (human-readable + single source of truth).
//   2. The region between the EMBEDDED_INLINE_BEGIN / END sentinels below is
//      INLINED verbatim into each opted-in workflow .js by
//      `scripts/patch-embedded-eval.js`. The workflow runtime does not support
//      ES module imports (see the InPlacePatch inlining fix), so we inline
//      rather than `import`.
//
// What a workflow gets after inlining:
//
//   EMBEDDING_CONTRACT          -- prompt fragment appended to proposal subagents
//                                  in embedded mode (emit a dispatch-compatible
//                                  source file, NOT a standalone TU).
//   __embeddedEvalPlan(ctx)     -- returns the ordered Bash command strings for
//                                  register -> build -> test -> bench -> unregister
//                                  against any contract-conforming adapter
//                                  (see ADAPTER_CONTRACT.md).
//
// The standalone path is unchanged: when a workflow's integration_pattern is
// `standalone`, it keeps doing its existing `{kernel_path}` / `{result_path}`
// substitution and never touches this substrate.

// >>> EMBEDDED_INLINE_BEGIN (do not edit inlined copies; edit this source + re-run scripts/patch-embedded-eval.js) <<<
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
// <<< EMBEDDED_INLINE_END >>>

export { EMBEDDING_CONTRACT, __embeddedEvalPlan as embeddedEvalPlan }
