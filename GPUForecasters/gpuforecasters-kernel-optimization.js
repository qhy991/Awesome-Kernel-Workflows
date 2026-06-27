export const meta = {
  name: 'gpuforecasters-kernel-optimization',
  description: 'Kernel optimization with learned speedup forecasting and PUCT search',
  whenToUse: 'Use for kernel optimization guided by learned performance forecasting models',
  phases: [
    { title: 'Setup', detail: 'Initialize forecasting models and search parameters' },
    { title: 'Train Forecasters', detail: 'Train surrogate models for speedup prediction' },
    { title: 'Calibration', detail: 'Calibrate abstention thresholds for forecasters' },
    { title: 'PUCT Search', detail: 'Tree search with forecaster-guided exploration' },
    { title: 'Refinement', detail: 'Refine promising candidates with focused search' },
    { title: 'Validation', detail: 'Validate final candidates on target hardware' },
    { title: 'Report', detail: 'Generate optimization report' },
  ],
  // Embedded-dispatch args (optional; default behavior is standalone). When
  // integration_pattern starts with "embedded", candidates are evaluated against
  // a project's register adapter instead of {kernel_path}/{result_path} substitution.
  args: [
    { name: 'integration_pattern', detail: 'standalone (default) | embedded[-*]. "embedded" routes evaluation through a register adapter.' },
    { name: 'register_script', detail: 'Path to a contract-conforming adapter (e.g. scripts/llamacpp_register_variant.py). Required when embedded.' },
    { name: 'project_root', detail: 'Project root the adapter wires the variant into. Alias: ggml_root. Required when embedded.' },
    { name: 'reference_cuh', detail: 'Reference dispatch .cuh whose signature the candidate must match exactly. Alias: reference_file.' },
    { name: 'register_params', detail: 'Opaque pass-through to the adapter (e.g. "--dkq 256 --dv 256 --cmake-build-dir /p/build").' },
  ],
};

// --- BEGIN model-tier (auto-inserted by scripts/patch-model-tier.js) ---
// Tier-based model routing: mechanical steps (run substrate scripts, parse
// JSON) use cheaper models; profile steps (run eval/ncu) use mid-tier;
// judgment steps (plan/implement/report) use the top tier. Tuneable via
// args.model_{mechanical,profile,judgment}.
const MODEL = {
  mechanical: (typeof args !== 'undefined' && args && args.model_mechanical) || 'haiku',
  profile: (typeof args !== 'undefined' && args && args.model_profile) || 'sonnet',
  judgment: (typeof args !== 'undefined' && args && args.model_judgment) || 'opus',
}
// __modelTierApplied
// --- END model-tier ---

const WORKFLOW_NAME = 'gpuforecasters-kernel-optimization'

// --- shared profiling-strategist plumbing (benchmark-heavy forecaster workflow:
// no native profiler of its own, so the strategist resolves against the CUDA
// substrate manifest and the forecast/evaluate prompt below honors its decision). ---
const SUBSTRATE = args.substrate_dir || '_substrate'
const PY = args.substrate_command_prefix || ''
const BACKEND_MANIFEST = args.backend_manifest || `${SUBSTRATE}/backends/cuda/manifest.json`
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }
function substrateInstruction(script, cliArgs) {
  const p = `${SUBSTRATE}/${script}`
  return PY ? `Run exactly: \`${PY} ${p} ${cliArgs}\`.`
            : `No substrate_command_prefix for ${p} ${cliArgs}; do not invent an interpreter.`
}


// --- BEGIN inlined arg_guard (Workflow runtime parses scripts as bare scripts,
//                              not ES modules; static imports are rejected) ---
function __unwrapArgs(rawArgs) {
  if (rawArgs == null) return {}
  if (typeof rawArgs === 'object' && !Array.isArray(rawArgs)) return rawArgs
  if (typeof rawArgs === 'string') {
    const trimmed = rawArgs.trim()
    if (trimmed === '') return {}
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        const parsed = JSON.parse(trimmed)
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed
        throw new Error('arg_guard: parsed JSON value is not a plain object')
      } catch (e) { throw new Error(`arg_guard: invalid JSON args: ${e.message}`) }
    }
    const out = {}
    const re = /(\w[\w.-]*)=("(?:\\\\\"|[^"])*"|\'(?:\\\\\'|[^\'])*\'|\S+)/g
    let m
    while ((m = re.exec(trimmed)) !== null) {
      let v = m[2]
      if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
        v = v.slice(1, -1)
      }
      out[m[1]] = v
    }
    if (Object.keys(out).length === 0) {
      throw new Error(`arg_guard: workflow args is a non-empty string but contains no key=value pairs and is not JSON. First 160 chars: ${trimmed.slice(0, 160)}`)
    }
    return out
  }
  throw new Error(`arg_guard: workflow args has unexpected type: ${typeof rawArgs}`)
}
// eslint-disable-next-line no-global-assign
args = __unwrapArgs(typeof args === 'undefined' ? undefined : args)
// --- END inlined arg_guard ---

// --- BEGIN inlined agent-retry scaffolding (from _meta/scaffolding/agent-retry.js) ---
async function agentRetry(fn, opts) {
  const retries = (opts && opts.retries != null) ? opts.retries : 5
  let lastError = null
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const result = await fn()
      if (result != null) return result
      // null = agent skipped mid-run OR terminal subagent failure (e.g. transient 429) — retry.
    } catch (e) {
      lastError = e
    }
  }
  if (lastError) throw lastError
  // All attempts returned null (agent skipped mid-run OR a terminal subagent
  // failure such as a sustained 429). FAIL-SAFE DEFAULT: throw an attributable
  // error instead of returning null. A null return would later hit an unguarded
  // deref (`diag.bottleneck_class`, `impl.code`, ...) and crash the run with a
  // cryptic TypeError — issue #20. Throwing here makes the round abort cleanly
  // with a recorded reason, and inside `parallel()` a throwing thunk simply
  // resolves to a null slot that `.filter(Boolean)` drops (graceful). Callers
  // that INTENTIONALLY degrade on a missing result opt out with `{ allowNull: true }`.
  if (opts && opts.allowNull === true) return null
  throw new Error(
    `agentRetry: "${(opts && opts.label) || 'agent'}" returned null after ${retries + 1} attempt(s) ` +
    `(agent skipped or terminal API failure after retries).`,
  )
}

/**
 * Null-guard a REQUIRED structured field. Throws a clear, attributable error
 * (instead of a cryptic TypeError) when an agent returned null/malformed output,
 * so the run fails loudly at the dereference rather than producing garbage.
 */
function expect(obj, field, ctx) {
  if (obj == null || obj[field] == null) {
    throw new Error(
      `agentRetry: required field "${field}" is missing${ctx ? ' from ' + ctx : ''} ` +
      `(agent returned null or a malformed result after retries).`,
    )
  }
  return obj[field]
}

/**
 * Null-guard an OPTIONAL structured field with a fallback (no throw).
 * Use for deref points that have a sensible default (e.g. `[]`, `''`, `0`).
 */
function guard(obj, field, fallback) {
  if (obj == null || obj[field] == null) return fallback
  return obj[field]
}
// --- END inlined agent-retry scaffolding ---
// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.

// --- BEGIN embedded-eval substrate (auto-inlined by scripts/patch-embedded-eval.js) ---
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
// --- END embedded-eval substrate ---

const WORKFLOW_SUITABILITY = {
  supported_languages: ['cuda', 'metal'],
  supported_problem_types: ['cuda-kernel-optimization', 'gpu-kernel-optimization', 'kernel-search'],
  problem_types: ['CUDA/GPU/Metal kernel search with speedup forecaster', 'PUCT optimization with execute-or-abstain feedback'],
  reason: 'GPU Forecasters expects GPU speedup evaluator feedback and a CUDA/Metal-oriented search/evaluation loop.',
}

function normalizeSuitabilityValue(value) {
  const raw = String(value || '').trim().toLowerCase().replace(/_/g, '-')
  const aliases = {
    'c++': 'cpp',
    cxx: 'cpp',
    cplusplus: 'cpp',
    cute: 'cute-dsl',
    hip: 'rocm',
    'intel-xpu': 'xpu',
    optimize: 'kernel-optimization',
    optimization: 'kernel-optimization',
    generate: 'kernel-generation',
    generation: 'kernel-generation',
    explain: 'performance-explanation',
    explanation: 'performance-explanation',
  }
  return aliases[raw] || raw
}

function supportsSuitabilityValue(supported, requested) {
  return supported.includes(requested) || supported.some(value => value.endsWith(`-${requested}`))
}

function assertWorkflowSuitability() {
  const requestedLanguage = normalizeSuitabilityValue(args.language)
  if (requestedLanguage && requestedLanguage !== 'auto') {
    const supported = WORKFLOW_SUITABILITY.supported_languages.map(normalizeSuitabilityValue)
    if (!supported.includes(requestedLanguage)) {
      throw new Error(
        `${WORKFLOW_NAME} is not suitable for language="${args.language}". ` +
        `Supported languages/backends: ${WORKFLOW_SUITABILITY.supported_languages.join(', ')}. ` +
        `Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }
  }

  const requestedProblemType = normalizeSuitabilityValue(args.problem_type)
  if (requestedProblemType && requestedProblemType !== 'auto') {
    const supportedProblemTypes = (WORKFLOW_SUITABILITY.supported_problem_types || []).map(normalizeSuitabilityValue)
    if (supportedProblemTypes.length && !supportsSuitabilityValue(supportedProblemTypes, requestedProblemType)) {
      throw new Error(
        `${WORKFLOW_NAME} is not suitable for problem_type="${args.problem_type}". ` +
        `Supported problem types: ${WORKFLOW_SUITABILITY.supported_problem_types.join(', ')}. ` +
        `Typical use cases: ${WORKFLOW_SUITABILITY.problem_types.join('; ')}. ` +
        `Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }
  }
}

assertWorkflowSuitability()

const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const KERNEL_PATH = args.kernel_path || ''
const OP_DESC = args.op_description || args.operation || 'CUDA kernel'
const LANGUAGE = args.language || 'cuda'
const TARGET_GPU = args.target_gpu || ''
const TEST_CMD = args.test_command || ''
const BENCHMARK_CMD = args.benchmark_command || ''
const BUILD_CMD = args.build_command || ''
const USER_NOTE = args.note || args.notes || ''
// Embedded-dispatch mode: gate everything below behind EMBEDDED so the
// standalone path is byte-identical to before when integration_pattern is absent.
// NOTE: EMBEDDED is the STATIC seed (from args.integration_pattern) used for arg
// validation and backward compat. The RUNTIME authority is IS_EMBEDDED, derived
// from the integration-strategist gate inside main(); EMBEDDED is reassigned to
// IS_EMBEDDED there so the authoring/eval prompts below honor the runtime decision.
const INTEGRATION_PATTERN = (args.integration_pattern || 'standalone')
let EMBEDDED = INTEGRATION_PATTERN.startsWith('embedded')
// RUNTIME_INTEGRATION_METHOD is set by the integration-strategist gate inside main();
// until then it mirrors the static seed. Module-scope helpers (taskContract /
// embeddedEvidenceContract) read it so their eval-contract text reflects the runtime
// decision, not the static arg.
let RUNTIME_INTEGRATION_METHOD = EMBEDDED ? 'embedded_dispatch' : 'standalone'
const REGISTER_SCRIPT = args.register_script || ''
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const REFERENCE_FILE = args.reference_cuh || args.reference_file || ''
const REGISTER_PARAMS = args.register_params || ''  // opaque pass-through e.g. "--dkq 256 --dv 256 --cmake-build-dir /p/build"
const BASELINE_DESCRIPTION = args.baseline || args.baseline_description || args.baseline_notes || ''
const BASELINE_LATENCY_MS = args.baseline_latency_ms ?? args.baseline_perf_ms ?? args.baseline_time_ms ?? args.baseline_perf ?? ''
const BASELINE_RESULT_PATH = args.baseline_result_path || ''
const EXP_DIR = args.exp_dir || '/tmp/gpuforecasters_exp'
const REQUESTED_TRAINING_BUDGET = args.curriculum_size || args.training_budget || ''
const REQUESTED_GPU_BUDGET = args.gpu_budget || args.iterations || ''
const REQUESTED_PUCT_C = args.puct_c || args.puct_exploration_constant || ''
const REQUESTED_TREE_DEPTH = args.tree_depth || args.tree_depth_limit || ''
function assertEmbeddedArgs() {
  if (!EMBEDDED) return
  const missing = []
  if (!REGISTER_SCRIPT) missing.push('register_script')
  if (!PROJECT_ROOT) missing.push('project_root (or ggml_root)')
  if (!BUILD_CMD) missing.push('build_command')
  if (!TEST_CMD) missing.push('test_command')
  if (!BENCHMARK_CMD) missing.push('benchmark_command')
  if (missing.length) {
    throw new Error(
      `${WORKFLOW_NAME}: integration_pattern="${INTEGRATION_PATTERN}" (embedded dispatch) requires ` +
      `the following non-empty args: ${missing.join(', ')}. ` +
      `Provide a contract-conforming register_script and the project's build/test/benchmark commands ` +
      `(see _substrate/embedded/ADAPTER_CONTRACT.md), or use integration_pattern="standalone".`
    )
  }
}

assertEmbeddedArgs()

const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : (PROBLEM_DEFINITION || PROBLEM_PATH || USER_NOTE ? 'generate_then_optimize' : 'unspecified_task')
const EVIDENCE_MODE = (TEST_CMD && BENCHMARK_CMD)
  ? 'measured_correctness_and_performance'
  : (TEST_CMD ? 'correctness_only' : (BENCHMARK_CMD ? 'benchmark_only' : 'conservative_missing_evidence'))

function formatContractValue(value) {
  if (value === undefined || value === null || value === '') return '(not provided)'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function positiveNumber(value) {
  const n = Number(value)
  return Number.isFinite(n) && n > 0 ? n : null
}

const PROVIDED_BASELINE_LATENCY_MS = positiveNumber(BASELINE_LATENCY_MS)

function taskContract() {
  return `# Task Contract
- input_mode: ${INPUT_MODE}
- problem_definition: ${formatContractValue(PROBLEM_DEFINITION)}
- problem_path: ${formatContractValue(PROBLEM_PATH)}
- kernel_path: ${formatContractValue(KERNEL_PATH)}
- op_description: ${formatContractValue(OP_DESC)}
- language: ${formatContractValue(LANGUAGE)}
- target_gpu: ${formatContractValue(TARGET_GPU)}
- exp_dir: ${EXP_DIR}
- user_note: ${formatContractValue(USER_NOTE)}

# Baseline Contract
- baseline: ${formatContractValue(BASELINE_DESCRIPTION)}
- baseline_latency_ms: ${formatContractValue(BASELINE_LATENCY_MS)}
- baseline_result_path: ${formatContractValue(BASELINE_RESULT_PATH)}

# Evidence Commands
- test_command: ${formatContractValue(TEST_CMD)}
- benchmark_command: ${formatContractValue(BENCHMARK_CMD)}
- evidence_mode: ${EVIDENCE_MODE}

# Evidence Rules
1. Treat user_note as authoritative task context. If it contains validation commands, baseline details, tolerances, or constraints, preserve and follow them.
2. When test_command is provided, run it exactly with {kernel_path} and {result_path} substitutions for every candidate that claims correctness.
3. When benchmark_command is provided, run it exactly with {kernel_path} and {result_path} substitutions before reporting measured latency or speedup.
4. Compute speedup against baseline_latency_ms when provided; otherwise obtain a measured baseline through benchmark_command before claiming measured speedup.
5. If a required command is missing or cannot run, mark measured evidence unavailable. Do not invent measured correctness, latency, or speedup.
6. Materialize generated candidates under exp_dir so evidence artifacts can be inspected.${EMBEDDED ? '\n\n' + embeddedEvidenceContract(RUNTIME_INTEGRATION_METHOD) : ''}`
}

// Sanitize a candidate/round identity into a valid adapter variant name.
function sanitizeVariant(name) {
  return String(name || 'cand').replace(/[^A-Za-z0-9_]/g, '_')
}

// Proposal-prompt appendix: how to AUTHOR an embedded-dispatch candidate.
function embeddedProposalContract() {
  return [
    EMBEDDING_CONTRACT,
    '',
    'REFERENCE DISPATCH FILE: ' + (REFERENCE_FILE || '(not provided)'),
    REFERENCE_FILE
      ? 'Read ' + REFERENCE_FILE + ' and match its dispatch signature EXACTLY ' +
        '(entry-point shape, template params, launch-bounds conventions). Emit a ' +
        'COMPLETE dispatch-compatible .cuh -- never a standalone translation unit, ' +
        'main(), or test harness.'
      : 'No reference file provided; still emit a COMPLETE dispatch-compatible .cuh ' +
        'matching the project dispatch signature, not a standalone translation unit.',
  ].join('\n')
}

// Evidence-prompt appendix: how to EVALUATE one embedded candidate. Branches on the
// runtime integration method: embedded_dispatch uses the register adapter; embedded_inplace
// swaps the candidate into the project file in place (a pristine backup is taken once at
// Setup and restored on every eval + unconditionally on exit).
function embeddedEvidenceContract(method) {
  const m = method || 'embedded_dispatch'
  if (m === 'embedded_inplace') {
    return [
      '# Embedded-Inplace Evidence (integration method=embedded_inplace)',
      'This kernel is NOT standalone. Do NOT use {kernel_path}/{result_path} substitution.',
      'Each candidate REPLACES the project kernel file in place; a pristine backup was taken',
      'once at Setup. For every candidate:',
      '1. RESTORE pristine first (defensive): cp -a <backup> <project_kernel_path>',
      '2. APPLY candidate in place: cp <candidatePath> <project_kernel_path>',
      '3. BUILD the project (project-native build_command)',
      '4. TEST (correctness) and BENCHMARK (latency) against the built project',
      '5. ALWAYS restore the pristine original (even on failure/non-improvement):',
      '   cp -a <backup> <project_kernel_path>',
      '6. HARD REQUIREMENT (cleanup invariant): leave the project byte-exact pristine.',
      '7. Parse correctness and latency ONLY from the command stdout, under the SAME',
      '   grounding/anti-fabrication rules above. Never invent measured correctness, latency,',
      '   or speedup; if a command is missing or fails, mark measured evidence unavailable.',
    ].join('\n')
  }
  return [
    '# Embedded-Dispatch Evidence (integration method=embedded_dispatch)',
    'This kernel is NOT standalone. Do NOT use {kernel_path}/{result_path} substitution.',
    'Each candidate is a complete dispatch-compatible .cuh evaluated against the project',
    'register adapter using the workflow-provided embedded eval plan. For every candidate:',
    '1. Derive a unique variant name from the candidate/round identity (sanitized to [A-Za-z0-9_]).',
    '2. Build the plan via the workflow helper (adapter: python "' + REGISTER_SCRIPT + '", project-root: ' + PROJECT_ROOT + ').',
    '3. Run IN ORDER: plan.register, plan.list (confirm the variant is registered),',
    '   plan.build, plan.test (correctness), plan.benchmark (latency),',
    '   then ALWAYS plan.unregister (even on failure/non-improvement) and confirm via',
    '   plan.list that the variant is gone.',
    '4. HARD REQUIREMENT (cleanup invariant): on ANY failure or non-improvement you MUST',
    '   unregister and confirm removal, leaving the project byte-exact pristine.',
    '5. Parse correctness and latency ONLY from the command stdout, under the SAME',
    '   grounding/anti-fabrication rules above. Never invent measured correctness, latency,',
    '   or speedup; if a command is missing or fails, mark measured evidence unavailable.',
  ].join('\n')
}

// Build the ordered eval plan for one embedded candidate (reuses the inlined substrate).
// Used to render concrete, copy-runnable plan commands into the eval-phase prompts so the
// evaluating subagent runs the exact register/list/build/test/benchmark/unregister sequence.
function embeddedPlanFor(candidatePath, variantSeed) {
  return __embeddedEvalPlan({
    adapter: 'python "' + REGISTER_SCRIPT + '"',
    variant: sanitizeVariant(variantSeed),
    source: candidatePath,
    projectRoot: PROJECT_ROOT,
    params: REGISTER_PARAMS,
    buildCmd: BUILD_CMD,
    testCmd: TEST_CMD,
    benchmarkCmd: BENCHMARK_CMD,
  })
}

// GPU Forecasters: Kernel optimization with learned performance prediction
// Based on arXiv:2605.31464 (MIT)
// Implements surrogate models with abstention + PUCT tree search

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it in the forecast/evaluate prompt below. The agent only CLASSIFIES the
// task (fuzzy op_class/size); the substrate DETERMINISTICALLY picks the method and
// STAMPS confidence by method (measured/inferred/hypothesized) -- the model must
// NOT assign confidence itself. See _substrate/profiling/README.md. Defaults to
// native_profiler so the happy-path forecast/execute loop is unchanged if the
// decision is ignored; only overwritten on a valid strategist response. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured' }

async function main() {
  // ============================================================================
  // Phase 1: Setup
  // ============================================================================
  phase('Setup');

  // Classify the kernel and resolve the profiling method via the shared
  // profiling-strategist (deterministic method pick + confidence stamp).
  {
    const _pd = await agentRetry(() => agent(
      `Classify the kernel under optimization. Source: ` +
      (KERNEL_PATH ? `read ${KERNEL_PATH}` : `operation "${OP_DESC}"${PROBLEM_DEFINITION ? ` / spec:\n${PROBLEM_DEFINITION.substring(0, 1500)}` : ''}`) + `.\n` +
      `Pick op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
      substrateInstruction('profiling/profiling_strategist.py',
        `resolve --backend-manifest ${BACKEND_MANIFEST} --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl`) +
      ` Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
      { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    if (_pd && _pd.method) PROFILING_DECISION = _pd
  }
  log(`Profiling method: ${PROFILING_DECISION.method} (confidence=${PROFILING_DECISION.confidence})`)

  // --- integration-strategist: DECIDE standalone vs embedded_* at runtime. This
  // REPLACES the static `EMBEDDED` (from args.integration_pattern) as the authority
  // for which eval path runs: the strategist classifies whether the kernel can
  // compile as a single TU and routes accordingly. GPUForecasters has no backend
  // driver, so standalone runs its native {kernel_path}/{result_path} evidence
  // loop, IS_EMBEDDED runs project-native register/build/test/benchmark.
  // Backward compat: seed the strategist's can-standalone probe from the existing
  // static EMBEDDED arg so legacy `integration_pattern=embedded*` callers still
  // route to the embedded path when the strategist is unavailable or uncertain.
  // Additive: when method==='standalone' the path below is byte-identical to before.
  // See _substrate/integration/README.md and _substrate/integration/ROLLOUT.md.
  let INTEGRATION_DECISION = { method: EMBEDDED ? 'embedded_dispatch' : 'standalone', build_fidelity: 'isolated', reversible: true }
  if (KERNEL_PATH) {
    const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
    const _integ = await agentRetry(() => agent(
      `Read ${KERNEL_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
      `(use no when the file cannot compile as a single TU — e.g. a llama.cpp .cuh with project-only deps). Then ` +
      substrateInstruction('integration/integration_strategist.py',
        `resolve --kernel "${KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
        `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl`) +
      ` Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
      { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    if (_integ && _integ.method) INTEGRATION_DECISION = _integ
  }
  log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
  if (INTEGRATION_DECISION.method === 'derive_adapter') {
    throw new Error('integration-strategist returned derive_adapter — provide project_root + register_script + build/test/benchmark commands')
  }
  const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
  RUNTIME_INTEGRATION_METHOD = INTEGRATION_DECISION.method
  const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
  if (ORIGINAL_BACKUP) {
    await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
      { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  }
  // If the strategist routed to an embedded path, validate the required wiring now
  // (the static `EMBEDDED` arg check above only fires when integration_pattern was
  // passed; the strategist can decide embedded even when it was not).
  if (IS_EMBEDDED) {
    const missing = []
    if (INTEGRATION_DECISION.method === 'embedded_dispatch' && !REGISTER_SCRIPT) missing.push('register_script')
    if (!PROJECT_ROOT) missing.push('project_root (or ggml_root)')
    if (!BUILD_CMD) missing.push('build_command')
    if (!TEST_CMD) missing.push('test_command')
    if (!BENCHMARK_CMD) missing.push('benchmark_command')
    if (missing.length) {
      throw new Error(`integration-strategist routed to ${INTEGRATION_DECISION.method} but missing: ${missing.join(', ')}`)
    }
  }
  // A-O1 closure: GPUForecasters has no native profiler/driver, so a native_profiler
  // choice cannot be honored without a standalone benchmark command -> downgrade to
  // perf_heuristic (which writes heuristic_bclass in the eval prompts so diagnose
  // stays defined).
  if (PROFILING_DECISION.method === 'native_profiler' && !BENCHMARK_CMD) {
    log(`profiling: native_profiler but no benchmark command -> downgrade to perf_heuristic`)
    PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
      profiler_name: 'gpuforecasters-perf', rationale: 'native_profiler but no benchmark_command -> perf_heuristic' }
  }
  // Promote the runtime decision to EMBEDDED so the authoring/eval prompts below
  // route on IS_EMBEDDED (runtime) rather than the static arg. When standalone,
  // IS_EMBEDDED===false===EMBEDDED so the standalone path is byte-identical.
  EMBEDDED = IS_EMBEDDED

  const setupResult = await agentRetry(() => agent(
    `Set up GPU Forecasters optimization environment:

${taskContract()}

1. Identify target kernel and optimization space
2. Configure surrogate forecasting models:
   - Model types (MLP, Transformer, etc.)
   - Training budget
   - Abstention strategy (native vs calibrated)
3. Set up PUCT search parameters:
   - Exploration constant
   - Simulation budget
   - Tree depth limit
4. Prepare baseline implementation from kernel_path or generate an initial kernel if only problem context is provided
5. Configure execution backend (Modal, local GPU, etc.)
6. Preserve the evidence commands and baseline contract for all later phases${EMBEDDED ? `

# Embedded-Dispatch Authoring Contract
When you author or modify any kernel candidate (here or in later phases), follow this:
${embeddedProposalContract()}` : ''}

Return JSON:
{
  "kernel_name": "kernel name",
  "optimization_space": {
    "parameters": ["param1", "param2", ...],
    "search_space_size": <int>
  },
  "forecaster_models": ["model1", "model2", ...],
  "training_budget": <int>,
  "puct_exploration_constant": <float>,
  "puct_simulation_budget": <int>,
  "tree_depth_limit": <int>,
  "baseline_perf": <float>,
  "backend": "modal|local",
  "target_gpu": "A100|H100|V100|..."
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"forecaster_search_setup","speedup":null,"note":"<kernel name + search space size + forecaster models + baseline_perf, one line>"}`,
    {
      label: 'Setup GPUForecasters',
      phase: 'Setup',
      schema: {
        type: 'object',
        properties: {
          kernel_name: { type: 'string' },
          optimization_space: { type: 'object' },
          forecaster_models: { type: 'array', items: { type: 'string' } },
          training_budget: { type: 'integer' },
          puct_exploration_constant: { type: 'number' },
          puct_simulation_budget: { type: 'integer' },
          tree_depth_limit: { type: 'integer' },
          baseline_perf: { type: 'number' },
          backend: { type: 'string' },
          target_gpu: { type: 'string' },
        },
        required: ['kernel_name', 'optimization_space', 'forecaster_models', 'baseline_perf'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!setupResult) {
    log('Setup failed');
    return { success: false, reason: 'setup_failed' };
  }

  if (PROVIDED_BASELINE_LATENCY_MS !== null) {
    setupResult.baseline_perf = PROVIDED_BASELINE_LATENCY_MS
  }
  setupResult.target_gpu = TARGET_GPU || setupResult.target_gpu || 'NVIDIA GPU'

  log(`Optimizing ${setupResult.kernel_name} on ${setupResult.target_gpu}`);
  log(`Search space: ${setupResult.optimization_space.search_space_size} configurations`);
  log(`Forecaster models: ${setupResult.forecaster_models.join(', ')}`);

  const trainingBudget = REQUESTED_TRAINING_BUDGET || setupResult.training_budget || 100;
  const puctExploration = REQUESTED_PUCT_C || setupResult.puct_exploration_constant || 1.0;
  const puctSimulations = REQUESTED_GPU_BUDGET || setupResult.puct_simulation_budget || 500;
  const treeDepthLimit = REQUESTED_TREE_DEPTH || setupResult.tree_depth_limit || 10;

  // Track optimization history
  const executionLog = [];
  let bestConfig = null;
  let bestSpeedup = 1.0;

  // ============================================================================
  // Phase 2: Train Forecasters
  // ============================================================================
  phase('Train Forecasters');

  log(`Training surrogate models with budget ${trainingBudget} evaluations...`);

  const trainingResult = await agentRetry(() => agent(
    `Train surrogate forecasting models:

${taskContract()}

Target: ${setupResult.kernel_name}
Training budget: ${trainingBudget} kernel executions
Forecaster models: ${setupResult.forecaster_models.join(', ')}

Training process:
1. Sample initial configurations (random, LHS, Sobol)
2. Materialize each config under ${EXP_DIR}/training/${EMBEDDED ? `
   When authoring each candidate, follow the Embedded-Dispatch Authoring Contract above.` : ''}
3. ${EMBEDDED
    ? (INTEGRATION_DECISION.method === 'embedded_inplace'
        ? `For each candidate written at <candidatePath>, evaluate it IN PLACE (a pristine backup was taken at Setup):
   - restore pristine: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}
   - apply candidate: cp <candidatePath> ${KERNEL_PATH}
   - build: ${BUILD_CMD}
   - test (correctness): ${TEST_CMD}
   - benchmark (latency): ${BENCHMARK_CMD || TEST_CMD}
   - then ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH} (cleanup invariant — leave the project byte-exact pristine).
   Parse correctness from test stdout and latency from benchmark stdout, then measure speedup against the baseline contract.`
        : `For each candidate written at <candidatePath> with a unique sanitized <variantName>, build the eval plan and run, IN ORDER:
   - plan.register, then plan.list (confirm <variantName> is registered)
   - plan.build, then plan.test (correctness), then plan.benchmark (latency)
   - then ALWAYS plan.unregister and confirm via plan.list that <variantName> is gone (cleanup invariant).
   Concrete example for candidate "${EXP_DIR}/training/cand_001.cuh", variant "cand_001"
   (these exact command strings are produced by the workflow's embedded eval plan):
${(() => { const p = embeddedPlanFor(`${EXP_DIR}/training/cand_001.cuh`, 'cand_001'); return [
     '     register:   ' + p.register,
     '     list:       ' + p.list,
     '     build:      ' + p.build,
     '     test:       ' + p.test,
     '     benchmark:  ' + p.benchmark,
     '     unregister: ' + p.unregister,
   ].join('\n'); })()}
   Parse correctness from plan.test stdout and latency from plan.benchmark stdout, then measure speedup against the baseline contract.`)
    : 'Run test_command before accepting correctness when provided'}
4. ${EMBEDDED
    ? (INTEGRATION_DECISION.method === 'embedded_inplace'
        ? 'On ANY failure or non-improvement still restore the pristine original (cp -a the backup back) so the project stays byte-exact pristine.'
        : 'On ANY failure or non-improvement still run plan.unregister and confirm removal so the project stays byte-exact pristine.')
    : `Run benchmark_command on ${setupResult.target_gpu} when provided and measure speedup against the baseline contract`}
5. Store evaluator JSON artifacts beside each candidate
6. Reject or label candidates with missing/failed correctness evidence
7. Collect training dataset: (config, speedup) pairs only from measured or explicitly labeled forecast-only evidence
8. Train each forecaster model:
   - Input: configuration vector plus kernel/code context
   - Output: predicted speedup + uncertainty estimate
9. Implement abstention mechanism:
   - Native abstention: model-intrinsic uncertainty (e.g., dropout variance)
   - Calibrated abstention: learned threshold based on prediction error
10. Validate on hold-out set

Return JSON:
{
  "training_samples": <int>,
  "training_dataset_size": <int>,
  "trained_models": [
    {
      "model_name": "model1",
      "architecture": "MLP|Transformer|...",
      "train_mae": <float>,
      "val_mae": <float>,
      "abstention_rate": <float>,
      "abstention_strategy": "native|calibrated"
    },
    ...
  ],
  "best_training_speedup": <float>,
  "best_training_config": "config description"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (best_training_speedup is a measured speedup from training evaluations, or null if no measured evidence):
{"workflow":"${WORKFLOW_NAME}","phase":"Train Forecasters","ts":"<ts>","status":"done","technique":"surrogate_forecaster_training","speedup":<number or null>,"note":"<#training samples + #models trained + best val MAE + best training config>"}`,
    {
      label: 'Train forecasters',
      phase: 'Train Forecasters',
      schema: {
        type: 'object',
        properties: {
          training_samples: { type: 'integer' },
          training_dataset_size: { type: 'integer' },
          trained_models: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                model_name: { type: 'string' },
                architecture: { type: 'string' },
                train_mae: { type: 'number' },
                val_mae: { type: 'number' },
                abstention_rate: { type: 'number' },
                abstention_strategy: { type: 'string' },
              },
              required: ['model_name', 'train_mae', 'abstention_rate'],
            },
          },
          best_training_speedup: { type: 'number' },
          best_training_config: { type: 'string' },
        },
        required: ['training_samples', 'trained_models', 'best_training_speedup'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!trainingResult || trainingResult.trained_models.length === 0) {
    log('Forecaster training failed');
    return { success: false, reason: 'training_failed' };
  }

  log(`Trained ${trainingResult.trained_models.length} forecaster models`);
  for (const model of trainingResult.trained_models) {
    log(`  ${model.model_name}: MAE=${model.train_mae.toFixed(3)}, abstain=${(model.abstention_rate * 100).toFixed(1)}%`);
  }

  // Update best from training
  if (trainingResult.best_training_speedup > bestSpeedup) {
    bestSpeedup = trainingResult.best_training_speedup;
    bestConfig = trainingResult.best_training_config;
  }

  // ============================================================================
  // Phase 3: Calibration
  // ============================================================================
  phase('Calibration');

  log('Calibrating abstention thresholds...');

  const calibrationResult = await agentRetry(() => agent(
    `Calibrate abstention thresholds for forecasters:

${taskContract()}

Trained models: ${trainingResult.trained_models.map(m => m.model_name).join(', ')}

Calibration process:
1. Collect uncertainty estimates on validation set
2. Correlate uncertainty with prediction error
3. Find optimal abstention threshold:
   - Minimize: prediction error on non-abstained samples
   - Subject to: abstention rate ≤ target (e.g., 20%)
4. For ensemble: combine predictions when all agree, abstain if any abstains
5. Measure calibrated performance:
   - MAE on non-abstained predictions
   - Abstention rate
   - Coverage (1 - abstention_rate)

Return JSON:
{
  "calibrated_models": [
    {
      "model_name": "model1",
      "calibrated_threshold": <float>,
      "calibrated_mae": <float>,
      "calibrated_abstention_rate": <float>,
      "coverage": <float>
    },
    ...
  ],
  "ensemble_strategy": "unanimous|majority|weighted",
  "ensemble_mae": <float>,
  "ensemble_abstention_rate": <float>,
  "ensemble_coverage": <float>
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Calibration","ts":"<ts>","status":"done","technique":"abstention_threshold_calibration","speedup":null,"note":"<ensemble strategy + ensemble MAE + ensemble coverage + abstention rate>"}`,
    {
      label: 'Calibrate forecasters',
      phase: 'Calibration',
      schema: {
        type: 'object',
        properties: {
          calibrated_models: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                model_name: { type: 'string' },
                calibrated_threshold: { type: 'number' },
                calibrated_mae: { type: 'number' },
                calibrated_abstention_rate: { type: 'number' },
                coverage: { type: 'number' },
              },
              required: ['model_name', 'calibrated_mae', 'coverage'],
            },
          },
          ensemble_strategy: { type: 'string' },
          ensemble_mae: { type: 'number' },
          ensemble_abstention_rate: { type: 'number' },
          ensemble_coverage: { type: 'number' },
        },
        required: ['calibrated_models', 'ensemble_mae'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!calibrationResult) {
    log('Calibration failed');
    return { success: false, reason: 'calibration_failed' };
  }

  log(`Ensemble calibrated: MAE=${calibrationResult.ensemble_mae.toFixed(3)}, coverage=${(calibrationResult.ensemble_coverage * 100).toFixed(1)}%`);

  // ============================================================================
  // Phase 4: PUCT Search
  // ============================================================================
  phase('PUCT Search');

  log(`Running PUCT tree search with ${puctSimulations} simulations...`);

  const puctResult = await agentRetry(() => agent(
    `Perform PUCT (Polynomial Upper Confidence Trees) search:

${taskContract()}

Search parameters:
- Exploration constant (c_puct): ${puctExploration}
- Simulation budget: ${puctSimulations}
- Tree depth limit: ${treeDepthLimit}

Forecaster ensemble:
- Models: ${calibrationResult.calibrated_models.map(m => m.model_name).join(', ')}
- Ensemble strategy: ${calibrationResult.ensemble_strategy}
- Abstention rate: ${(calibrationResult.ensemble_abstention_rate * 100).toFixed(1)}%

PUCT algorithm:
1. Initialize root node with baseline config
2. For each simulation:
   a. Selection: traverse tree using PUCT formula
      PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
      where:
      - Q(s,a) = mean reward (speedup) from (s,a)
      - P(s,a) = prior from forecaster
      - N(s) = visit count of state s
      - N(s,a) = visit count of (s,a)
   b. Expansion: expand node with forecaster-predicted actions
   c. Simulation:
${EMBEDDED
  ? (INTEGRATION_DECISION.method === 'embedded_inplace'
      ? `      - If forecaster abstains: materialize the config under ${EXP_DIR}/puct/ and evaluate it in place:
        author a dispatch .cuh per the Authoring Contract, then run IN ORDER: restore pristine
        (cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}), apply candidate (cp <candidatePath> ${KERNEL_PATH}),
        build (${BUILD_CMD}), test (${TEST_CMD}), benchmark (${BENCHMARK_CMD || TEST_CMD}), then ALWAYS restore
        (cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}) — cleanup invariant, leave the project byte-exact pristine.
        Parse correctness from test stdout and latency from benchmark stdout.
      - Else: use forecaster prediction
      - If test fails, treat the candidate as invalid regardless of predicted speedup
      - Use the measured benchmark latency as the authoritative measured speedup`
      : `      - If forecaster abstains: materialize the config under ${EXP_DIR}/puct/ and execute the embedded eval plan:
        author a dispatch .cuh per the Authoring Contract, then run IN ORDER plan.register, plan.list, plan.build,
        plan.test, plan.benchmark, then ALWAYS plan.unregister + confirm removal via plan.list (cleanup invariant).
        Parse correctness from plan.test stdout and latency from plan.benchmark stdout.
      - Else: use forecaster prediction
      - If plan.test fails, treat the candidate as invalid regardless of predicted speedup
      - Use the measured plan.benchmark latency as the authoritative measured speedup`)
  : `      - If forecaster abstains: materialize the config under ${EXP_DIR}/puct/ and execute user-provided evidence commands for ground truth
      - Else: use forecaster prediction
      - If test_command fails, treat the candidate as invalid regardless of predicted speedup
      - If benchmark_command is available, use its evaluator JSON as the authoritative measured speedup`}
   d. Backpropagation: update Q values along path
3. Return best config from tree (highest Q value)

Track:
- Total GPU executions (should be << simulation budget due to abstention)
- Tree statistics (depth, breadth, nodes explored)
- Best config found

Profiling-strategist selected method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'). If method==='native_profiler', you MAY use ncu metrics as forecaster features. If method==='perf_heuristic', derive memory-vs-compute-bound features from benchmark throughput (latency, GFLOPS/GB-s) and tag them evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'; also write heuristic_bclass (memory_bound|compute_bound|latency_bound) so diagnose does not fall to unknown. If method==='static', reason from source only (confidence='hypothesized'). Never fabricate profiler counters.

Return JSON:
{
  "simulations": ${puctSimulations},
  "total_executions": <int>,
  "abstention_saved_executions": <int>,
  "tree_nodes_explored": <int>,
  "tree_max_depth": <int>,
  "best_config": "config description",
  "best_speedup": <float>,
  "search_trajectory": [
    {"step": <int>, "speedup": <float>, "executed": true/false},
    ...
  ]
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (candidate_id is the best config found; best_speedup is the measured speedup of that config, or null if not measured):
{"workflow":"${WORKFLOW_NAME}","phase":"PUCT Search","ts":"<ts>","status":"done","candidate_id":"<best config description>","technique":"puct_tree_search","speedup":<number or null>,"note":"<total GPU executions + executions saved by abstention + nodes explored + best config>"}`,
    {
      label: 'PUCT search',
      phase: 'PUCT Search',
      schema: {
        type: 'object',
        properties: {
          simulations: { type: 'integer' },
          total_executions: { type: 'integer' },
          abstention_saved_executions: { type: 'integer' },
          tree_nodes_explored: { type: 'integer' },
          tree_max_depth: { type: 'integer' },
          best_config: { type: 'string' },
          best_speedup: { type: 'number' },
          search_trajectory: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                step: { type: 'integer' },
                speedup: { type: 'number' },
                executed: { type: 'boolean' },
              },
            },
          },
        },
        required: ['simulations', 'total_executions', 'best_speedup', 'best_config'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!puctResult) {
    log('PUCT search failed');
    return { success: false, reason: 'puct_failed' };
  }

  log(`PUCT search complete: ${puctResult.total_executions} GPU executions (saved ${puctResult.abstention_saved_executions} via forecasters)`);
  log(`Best speedup: ${puctResult.best_speedup.toFixed(3)}x`);

  // Update best
  if (puctResult.best_speedup > bestSpeedup) {
    bestSpeedup = puctResult.best_speedup;
    bestConfig = puctResult.best_config;
  }

  executionLog.push(...(puctResult.search_trajectory || []));

  // ============================================================================
  // Phase 5: Refinement
  // ============================================================================
  phase('Refinement');

  log('Refining top candidates with local search...');

  const refinementResult = await agentRetry(() => agent(
    `Refine best configuration found:

${taskContract()}

Best config from PUCT: ${puctResult.best_config}
Best speedup: ${puctResult.best_speedup.toFixed(3)}x

Refinement strategies:
1. Local search around best config:
   - Grid search in neighborhood
   - Gradient-based refinement (if applicable)
2. Ablation studies:
   - Test impact of individual optimizations
   - Identify critical parameters
3. Multi-start local search from top-k PUCT nodes
4. Fine-grained parameter tuning

Execute promising refinements on GPU (use forecasters to filter).
Materialize each refinement under ${EXP_DIR}/refinement/ and use the evidence commands exactly when provided.${EMBEDDED ? (INTEGRATION_DECISION.method === 'embedded_inplace' ? `
For each refinement, author a dispatch .cuh per the Authoring Contract and evaluate it IN PLACE: run IN ORDER
restore pristine (cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}), apply candidate (cp <candidatePath> ${KERNEL_PATH}),
build (${BUILD_CMD}), test (${TEST_CMD}), benchmark (${BENCHMARK_CMD || TEST_CMD}), then ALWAYS restore
(cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}) — cleanup invariant. Parse correctness/latency from command stdout only.` : `
For each refinement, author a dispatch .cuh per the Authoring Contract and evaluate it via the embedded eval
plan: run IN ORDER plan.register, plan.list, plan.build, plan.test, plan.benchmark, then ALWAYS plan.unregister
and confirm removal via plan.list (cleanup invariant). Parse correctness/latency from command stdout only.`) : ''}
Only promote a refinement as measured if correctness passes and benchmark evidence is available.

Return JSON:
{
  "refinement_candidates": <int>,
  "refinement_executions": <int>,
  "best_refined_config": "config description",
  "best_refined_speedup": <float>,
  "improvement_over_puct": <float>,
  "ablation_insights": "brief insights"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (candidate_id is the best refined config; best_refined_speedup is the measured speedup, or null if not measured):
{"workflow":"${WORKFLOW_NAME}","phase":"Refinement","ts":"<ts>","status":"done","candidate_id":"<best refined config description>","technique":"local_search_refinement","speedup":<number or null>,"note":"<#refinement candidates + #executions + improvement over PUCT + ablation insight>"}`,
    {
      label: 'Refine candidates',
      phase: 'Refinement',
      schema: {
        type: 'object',
        properties: {
          refinement_candidates: { type: 'integer' },
          refinement_executions: { type: 'integer' },
          best_refined_config: { type: 'string' },
          best_refined_speedup: { type: 'number' },
          improvement_over_puct: { type: 'number' },
          ablation_insights: { type: 'string' },
        },
        required: ['refinement_executions', 'best_refined_speedup', 'best_refined_config'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!refinementResult) {
    log('Refinement failed, using PUCT result');
  } else {
    log(`Refinement complete: ${refinementResult.refinement_executions} additional executions`);
    log(`Best refined speedup: ${refinementResult.best_refined_speedup.toFixed(3)}x (${refinementResult.improvement_over_puct > 0 ? '+' : ''}${(refinementResult.improvement_over_puct * 100).toFixed(1)}%)`);

    // Update best
    if (refinementResult.best_refined_speedup > bestSpeedup) {
      bestSpeedup = refinementResult.best_refined_speedup;
      bestConfig = refinementResult.best_refined_config;
    }
  }

  // ============================================================================
  // Phase 6: Validation
  // ============================================================================
  phase('Validation');

  log('Validating best configuration...');

  const validationResult = await agentRetry(() => agent(
    `Validate best configuration:

${taskContract()}

Best config: ${bestConfig}
Best speedup: ${bestSpeedup.toFixed(3)}x

Validation:
1. Materialize the final candidate under ${EXP_DIR}/final/${EMBEDDED ? (INTEGRATION_DECISION.method === 'embedded_inplace'
   ? `
   Author it as a dispatch .cuh per the Authoring Contract, then evaluate it IN PLACE: run IN ORDER
   restore pristine (cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}), apply candidate (cp <candidatePath> ${KERNEL_PATH}),
   build (${BUILD_CMD}), test (${TEST_CMD}), benchmark (${BENCHMARK_CMD || TEST_CMD}), then ALWAYS restore
   (cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}) — cleanup invariant. Use test stdout for correctness and benchmark
   stdout for latency; do NOT use {kernel_path}/{result_path} substitution.`
   : `
   Author it as a dispatch .cuh per the Authoring Contract, then evaluate via the embedded eval plan:
   run IN ORDER plan.register, plan.list, plan.build, plan.test, plan.benchmark, then ALWAYS plan.unregister
   and confirm removal via plan.list (cleanup invariant). Use plan.test for correctness and plan.benchmark
   stdout for latency; do NOT use {kernel_path}/{result_path} substitution.`) : ''}
2. Run test_command exactly if provided and fail validation if correctness fails
3. Run benchmark_command exactly if provided; parse its JSON artifact as authoritative measured performance
4. Execute on target hardware (${setupResult.target_gpu}) multiple times when benchmark_command supports repeated runs
5. Measure performance statistics:
   - Mean speedup
   - Std dev
   - Min/max
6. Verify correctness (output matches baseline)
7. Profile hardware utilization:
   - SM occupancy
   - Memory bandwidth
   - Compute throughput
8. Test on different input sizes (if applicable)
9. Compare with baseline and other methods
10. If evidence_mode is conservative_missing_evidence, return validation_passed=false unless the user note explicitly authorizes static-only validation

Profiling-strategist selected method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'). If method==='native_profiler', you MAY use ncu metrics as forecaster features. If method==='perf_heuristic', derive memory-vs-compute-bound features from benchmark throughput (latency, GFLOPS/GB-s) and tag them evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'; also write heuristic_bclass (memory_bound|compute_bound|latency_bound) so diagnose does not fall to unknown. If method==='static', reason from source only (confidence='hypothesized'). Never fabricate profiler counters.

Return JSON:
{
  "config": "${bestConfig}",
  "validation_runs": <int>,
  "mean_speedup": <float>,
  "std_speedup": <float>,
  "min_speedup": <float>,
  "max_speedup": <float>,
  "correctness_passed": true/false,
  "hardware_utilization": {
    "sm_occupancy_pct": <float>,
    "memory_bandwidth_pct": <float>,
    "compute_throughput_pct": <float>
  },
  "validation_passed": true/false
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (status="done" if correctness passed AND validation passed, else "error"; speedup is the measured mean_speedup number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Validation","ts":"<ts>","status":"<done|error>","candidate_id":"<validated config>","technique":"target_hardware_validation","speedup":<number or null>,"note":"<mean +/- std speedup + correctness pass/fail + validation pass/fail; or the failure reason>"}`,
    {
      label: 'Validate best config',
      phase: 'Validation',
      schema: {
        type: 'object',
        properties: {
          config: { type: 'string' },
          validation_runs: { type: 'integer' },
          mean_speedup: { type: 'number' },
          std_speedup: { type: 'number' },
          min_speedup: { type: 'number' },
          max_speedup: { type: 'number' },
          correctness_passed: { type: 'boolean' },
          hardware_utilization: { type: 'object' },
          validation_passed: { type: 'boolean' },
        },
        required: ['mean_speedup', 'correctness_passed', 'validation_passed'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!validationResult || !validationResult.validation_passed) {
    log('Validation failed');
    // embedded_inplace exit safety net: restore pristine original before the early
    // return so the project is left byte-exact even when validation fails mid-flow.
    if (ORIGINAL_BACKUP) {
      await agentRetry(() => agent(`Exit restore (unconditional, validation failed): run \`cp -a "${ORIGINAL_BACKUP}" "${KERNEL_PATH}"\` and confirm.`,
        { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Validation', schema: JSON_PASSTHROUGH }), { retries: 5 })
    }
    return {
      success: false,
      reason: 'validation_failed',
      best_config: bestConfig,
      best_speedup: bestSpeedup,
      input_mode: INPUT_MODE,
      evidence_mode: EVIDENCE_MODE,
      test_command: TEST_CMD,
      benchmark_command: BENCHMARK_CMD,
      baseline_latency_ms: setupResult.baseline_perf,
      exp_dir: EXP_DIR,
    };
  }

  log(`Validation passed: ${validationResult.mean_speedup.toFixed(3)}x ± ${validationResult.std_speedup.toFixed(3)}x`);

  // ============================================================================
  // Phase 7: Report
  // ============================================================================
  phase('Report');

  const report = await agentRetry(() => agent(
    `Generate GPU Forecasters optimization report:

${taskContract()}

Summary:
- Kernel: ${setupResult.kernel_name}
- Target GPU: ${setupResult.target_gpu}
- Search space: ${setupResult.optimization_space.search_space_size} configs
- Training budget: ${trainingResult.training_samples} samples
- PUCT simulations: ${puctResult.simulations}
- Total GPU executions: ${trainingResult.training_samples + puctResult.total_executions + (refinementResult?.refinement_executions || 0)}
- Executions saved by forecasters: ${puctResult.abstention_saved_executions}

Results:
- Baseline: ${setupResult.baseline_perf.toFixed(3)} ms
- Best speedup: ${validationResult.mean_speedup.toFixed(3)}x ± ${validationResult.std_speedup.toFixed(3)}x
- Best config: ${validationResult.config}

Forecaster performance:
${calibrationResult.calibrated_models.map(m => `  ${m.model_name}: MAE=${m.calibrated_mae.toFixed(3)}, coverage=${(m.coverage * 100).toFixed(1)}%`).join('\n')}

Generate report with:
1. Executive summary
2. Search efficiency analysis (executions saved)
3. Forecaster accuracy and calibration
4. PUCT search trajectory visualization
5. Best configuration analysis
6. Hardware utilization breakdown
7. Comparison with baselines

Return JSON:
{
  "summary": "brief summary",
  "best_speedup": ${validationResult.mean_speedup},
  "total_executions": <int>,
  "executions_saved": ${puctResult.abstention_saved_executions},
  "search_efficiency": <float>,
  "report_path": "path/to/report.md"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (speedup is the final best validated speedup number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","technique":"optimization_report","speedup":<number or null>,"note":"<final best speedup + total executions + executions saved by forecasters + report path>"}`,
    {
      label: 'Generate report',
      phase: 'Report',
      schema: {
        type: 'object',
        properties: {
          summary: { type: 'string' },
          best_speedup: { type: 'number' },
          total_executions: { type: 'integer' },
          executions_saved: { type: 'integer' },
          search_efficiency: { type: 'number' },
          report_path: { type: 'string' },
        },
        required: ['summary', 'best_speedup', 'total_executions'],
      },
    }
  ), { retries: 5, allowNull: true });

  // ============================================================================
  // Return final results
  // ============================================================================

  // embedded_inplace exit safety net: unconditionally restore the pristine original
  // so the project is left byte-exact regardless of how the workflow terminated.
  if (ORIGINAL_BACKUP) {
    await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${KERNEL_PATH}"\` and confirm.`,
      { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
  }

  return {
    success: true,
    method: 'GPU Forecasters',
    approach: 'Learned speedup forecasting + PUCT search',
    input_mode: INPUT_MODE,
    evidence_mode: EVIDENCE_MODE,
    kernel: setupResult.kernel_name,
    target_gpu: setupResult.target_gpu,
    exp_dir: EXP_DIR,
    test_command: TEST_CMD,
    benchmark_command: BENCHMARK_CMD,
    user_note_present: !!USER_NOTE,
    search_space_size: setupResult.optimization_space.search_space_size,
    training_budget: trainingResult.training_samples,
    forecaster_models: trainingResult.trained_models.map(m => m.model_name),
    ensemble_mae: calibrationResult.ensemble_mae,
    ensemble_coverage: calibrationResult.ensemble_coverage,
    puct_simulations: puctResult.simulations,
    total_executions: trainingResult.training_samples + puctResult.total_executions + (refinementResult?.refinement_executions || 0),
    executions_saved: puctResult.abstention_saved_executions,
    baseline_perf: setupResult.baseline_perf,
    best_speedup: validationResult.mean_speedup,
    speedup_std: validationResult.std_speedup,
    best_config: validationResult.config,
    hardware_utilization: validationResult.hardware_utilization,
    validation_passed: validationResult.validation_passed,
    report: report?.report_path,
    summary: report?.summary,
  };
}

// Execute the workflow
return await main();
