export const meta = {
  name: 'ksearch-kernel-optimization',
  description: 'World-model-guided tree search for GPU kernel optimization (K-Search methodology)',
  whenToUse: 'When you need to explore a large kernel design space systematically rather than iterating on a single optimization path. Uses a co-evolving decision tree (world model) to track hypotheses, guide action selection, and backtrack from failed strategies. Best for complex kernels where multiple orthogonal design decisions interact (e.g., MLA attention, MoE routing, fused operators).',
  phases: [
    { title: 'Setup', detail: 'Read kernel spec, evaluate baseline performance' },
    { title: 'Initialize', detail: 'Build initial world model decision tree with design hypotheses' },
    { title: 'Select', detail: 'Choose highest-scoring open frontier action node' },
    { title: 'Generate', detail: 'Generate and iteratively improve kernel code for selected action' },
    { title: 'Evaluate', detail: 'Compile, run, and measure kernel against workloads' },
    { title: 'Refine', detail: 'Update tree: attach solution (success) or downgrade + backtrack (failure)' },
    { title: 'Report', detail: 'Final optimization report with search trajectory' },
  ],
}

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

const WORKFLOW_NAME = 'ksearch-kernel-optimization'


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

const WORKFLOW_SUITABILITY = {
  supported_languages: ['triton', 'cuda', 'python'],
  supported_problem_types: ['gpu-kernel-optimization', 'kernel-search'],
  problem_types: ['world-model-guided kernel/operator search', 'benchmark-driven optimization with evaluator feedback'],
  reason: 'K-Search relies on an evaluator-backed search tree and is intended for kernel/operator optimization in supported implementation languages.',
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

function resolveBackendAxis() {
  const b = args.backend ? normalizeSuitabilityValue(args.backend) : null
  const l = args.language ? normalizeSuitabilityValue(args.language) : null
  if (b && l && b !== l) {
    throw new Error(`Conflicting args: backend="${args.backend}" vs language="${args.language}". Pass only one.`)
  }
  if (args.backend && !args.backend_dir) {
    throw new Error(`args.backend="${args.backend}" requires args.backend_dir; driver dispatch has no implicit-resolve path.`)
  }
  return b || l || null
}
const RESOLVED_BACKEND = resolveBackendAxis()
const USE_DRIVER = !!args.backend_dir

if (!USE_DRIVER) {
  assertWorkflowSuitability()
}

// =============================================================================
// K-Search: Co-Evolving World Model Kernel Optimization Workflow
// =============================================================================
//
// Source: "K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model"
//         arXiv:2602.19128 — Shiyi Cao, Ziming Mao, Joseph E. Gonzalez, Ion Stoica
//
// Core idea: maintain a JSON decision tree (world model) encoding kernel design
// hypotheses. Each cycle selects the highest-scoring open action node, generates
// code with multiple improve attempts, evaluates, then refines (success) or
// backtracks (failure). The tree co-evolves with the solutions.
//
// Usage:
//   Workflow({name: 'ksearch-kernel-optimization', args: {
//     problem_path: '/path/to/spec.yaml',
//     op_description: 'MLA decode attention kernel',
//     language: 'triton',                 // triton | cuda | python
//     target_gpu: 'H100',
//     iterations: 10,                     // search cycles
//     attempts_per_cycle: 5,              // generate/improve rounds per cycle
//     stagnation_window: 3,              // non-improving attempts before cycle ends
//     max_difficulty: 4,                  // max action difficulty (1-5)
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     kernel_path: '/path/to/baseline.py',
//     rtol: 0.01,
//     atol: 0.01,
//     exp_dir: '/tmp/ksearch_exp',
//   }})
//
// =============================================================================

const PROBLEM_DEFINITION = args.problem_definition || ''
const KERNEL_SPEC_PATH = args.problem_path
const OP_DESC = args.op_description || 'GPU kernel'
const LANGUAGE = args.language || 'triton'
const TARGET_GPU = args.target_gpu || 'H100'
const MAX_CYCLES = args.iterations || 10
const ATTEMPTS_PER_CYCLE = args.attempts_per_cycle || 5
const STAGNATION_WINDOW = args.stagnation_window || 3
const MAX_DIFFICULTY = args.max_difficulty || 4
const BENCH_CMD = args.benchmark_command || ''
const BASELINE_CODE_PATH = args.kernel_path || ''
const INPUT_MODE = BASELINE_CODE_PATH ? 'optimize_existing' : 'generate_then_optimize'
const RTOL = args.rtol || 0.01
const ATOL = args.atol || 0.01
const EXP_DIR = args.exp_dir || '/tmp/ksearch_exp'

if (!KERNEL_SPEC_PATH && !PROBLEM_DEFINITION && !BASELINE_CODE_PATH) {
  throw new Error('Provide one of problem_path, problem_definition, or kernel_path')
}

// --- Backend driver wiring (P5d Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_LANG_TOKEN = LANGUAGE
const LEGACY_FENCE_TOKEN = LANGUAGE
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

// --- Project-native integration (embedded kernels via integration-strategist) ---
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const BUILD_CMD = args.build_command || ''
const PROJECT_BENCH_CMD = args.benchmark_command || BENCH_CMD || ''
const REGISTER_SCRIPT = args.register_script || ''

function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}

let DRIVER = null
let DRIVER_LANG_FENCE = LEGACY_FENCE_TOKEN
let DRIVER_IMPL_REQUIREMENTS = ''
let DRIVER_SOURCE_EXT = ''
let DRIVER_BACKEND_ID = RESOLVED_BACKEND || ''

function langToken(legacy) {
  return USE_DRIVER ? DRIVER_LANG_FENCE : legacy
}
function fenceToken() {
  return USE_DRIVER ? DRIVER_LANG_FENCE : LEGACY_FENCE_TOKEN
}

function ksearchNodeKernelPath(label) {
  const ext = USE_DRIVER ? (DRIVER_SOURCE_EXT || '.py') : '.py'
  return `${EXP_DIR}/${label}${ext}`
}

// State
let decisionTree = null
let solutionDb = []
let bestSolution = null
let bestMetric = null
let baselineMetric = null
let specText = ''
let cycleCount = 0
let globalRound = 0

// =============================================================================
// Phase 1: Setup — Read spec, evaluate baseline
// =============================================================================
phase('Setup')

if (USE_DRIVER) {
  DRIVER = await agentRetry(() => agent(
    `Load the backend driver at ${BACKEND_DIR} and return its manifest plus idioms verbatim.\n` +
    `1. Run exactly: \`cat ${driverPath('manifest.json')}\` and parse JSON.\n` +
    `2. Run exactly: \`cat ${driverPath('idioms.json')}\` and parse JSON.\n` +
    `Return {present, backend_id, source_ext, aux_ext, lang_fence, impl_requirements, methods}.`,
    { model: MODEL.mechanical, label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (!DRIVER || DRIVER.present === false) {
    throw new Error(`No backend driver present at ${BACKEND_DIR}. Provide a valid backend_dir or omit it for the legacy path.`)
  }
  if (RESOLVED_BACKEND && DRIVER.backend_id && normalizeSuitabilityValue(DRIVER.backend_id) !== RESOLVED_BACKEND) {
    throw new Error(`backend_dir manifest backend_id="${DRIVER.backend_id}" conflicts with args.backend/language="${RESOLVED_BACKEND}".`)
  }
  DRIVER_LANG_FENCE = DRIVER.lang_fence || DRIVER_LANG_FENCE
  DRIVER_IMPL_REQUIREMENTS = DRIVER.impl_requirements || ''
  DRIVER_SOURCE_EXT = DRIVER.source_ext || DRIVER_SOURCE_EXT
  DRIVER_BACKEND_ID = DRIVER.backend_id || DRIVER_BACKEND_ID
  log(`Driver loaded: ${DRIVER_BACKEND_ID} (fence=${DRIVER_LANG_FENCE})`)
}

const setupResult = await agentRetry(() => agent(`You are a GPU kernel optimization expert. Read and analyze the kernel specification.

# Task
Read the kernel specification file at: ${KERNEL_SPEC_PATH || '(not provided)'}
If problem_definition is provided, use it as the authoritative kernel specification:
${PROBLEM_DEFINITION || '(not provided)'}
${BASELINE_CODE_PATH ? `Also read the baseline kernel at: ${BASELINE_CODE_PATH}` : ''}

# Analyze and return:
1. **spec_text**: The full specification text (problem definition, input/output formats, constraints)
2. **op_type**: Classification of the operation (e.g., "mla_attention", "moe_routing", "gemm", "softmax")
3. **input_shapes**: Description of input tensor shapes and data types
4. **output_shape**: Expected output shape and type
5. **constraints**: List of hard constraints (numerical precision, memory limits, etc.)
6. **baseline_code**: The baseline kernel code (if available)
7. **key_challenges**: What makes this kernel hard to optimize?
8. **design_dimensions**: Orthogonal axes of the design space (e.g., tiling strategy, memory hierarchy usage, parallelism decomposition, algorithmic variant)

Target language: ${langToken(LANGUAGE)}
Target GPU: ${TARGET_GPU}
Operation: ${OP_DESC}

Return structured analysis.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"spec_analysis","speedup":null,"note":"<op_type + key challenges + design dimensions, one line>"}`, {
  label: 'read-spec',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      spec_text: { type: 'string' },
      op_type: { type: 'string' },
      input_shapes: { type: 'string' },
      output_shape: { type: 'string' },
      constraints: { type: 'array', items: { type: 'string' } },
      baseline_code: { type: 'string' },
      key_challenges: { type: 'array', items: { type: 'string' } },
      design_dimensions: { type: 'array', items: { type: 'string' } },
    },
    required: ['spec_text', 'op_type'],
  },
}), { retries: 5 })

specText = setupResult.spec_text
const opType = setupResult.op_type

// Evaluate baseline
const baselineEval = await agentRetry(() => agent(`You are a kernel evaluation expert. Evaluate the baseline kernel to establish reference performance.

# Kernel Spec:
${specText.substring(0, 2000)}

# Baseline Code:
\`\`\`${langToken(LANGUAGE)}
${(setupResult.baseline_code || '').substring(0, 3000)}
\`\`\`

# Evaluation Instructions:
${BENCH_CMD ? `Run: ${BENCH_CMD}` : 'No benchmark_command provided; perform static characterization only and mark measured evidence unavailable.'}

Establish the baseline metric. If a benchmark command is available, compile and run it.
If not, analyze the code only; do not report estimated performance as measured.

The metric is mean_vs_baseline_factor (this IS the baseline, so it should be 1.0).
Also report absolute latency if measurable.

Return evaluation results.`, {
  label: 'eval-baseline',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      baseline_metric: { type: 'number' },
      baseline_latency_ms: { type: 'number' },
      eval_passed: { type: 'boolean' },
      performance_profile: { type: 'string' },
      bottleneck_analysis: { type: 'string' },
    },
    required: ['baseline_metric', 'eval_passed'],
  },
}), { retries: 5 })

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it below. The agent only classifies the task (fuzzy); the substrate stamps
// confidence by method (measured/inferred/hypothesized) — not the agent. See
// _substrate/profiling/README.md. Falls back to native_profiler if undecided. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }
if (USE_DRIVER) {
  const _pd = await agentRetry(() => agent(
    `Read ${BASELINE_CODE_PATH || ksearchNodeKernelPath('ksearch_root')}; classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${BACKEND_DIR}/manifest.json --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}

// --- integration-strategist: route build/test mode (standalone vs embedded_*). ---
const INTEG_KERNEL_PATH = BASELINE_CODE_PATH || ksearchNodeKernelPath('ksearch_root')
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Read ${INTEG_KERNEL_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${INTEG_KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
    `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_integ && _integ.method) INTEGRATION_DECISION = _integ
}
log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error('integration-strategist returned derive_adapter — provide project_root + build/test commands')
}
const USE_DRIVER_STANDALONE = USE_DRIVER && INTEGRATION_DECISION.method === 'standalone'
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Byte-exact backup (once): run \`cp -a "${INTEG_KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// A-O1 closure: native_profiler chosen but no standalone driver build to profile → perf_heuristic.
if (PROFILING_DECISION.method === 'native_profiler' && !USE_DRIVER_STANDALONE) {
  log(`profiling: native_profiler but not standalone driver build -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'test-harness-perf', rationale: 'native_profiler but no standalone driver build -> perf_heuristic' }
}

if (USE_DRIVER_STANDALONE) {
  const kPath = BASELINE_CODE_PATH || ksearchNodeKernelPath('ksearch_root')
  const buildOut = `${EXP_DIR}/ksearch_root.artifact`
  const profOut = `${EXP_DIR}/ksearch_root.prof.native`
  await agentRetry(() => agent(
    `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
    `Return its stdout JSON verbatim.`,
    { model: MODEL.mechanical, label: 'driver-build-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  const runOut = await agentRetry(() => agent(
    `${driverSh('run.sh', `--artifact ${buildOut} --kernel ${kPath}`)}\n` +
    `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
    { model: MODEL.profile, label: 'driver-run-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  let evidenceOut = null
  if (PROFILING_DECISION.method === 'native_profiler') {
    await agentRetry(() => agent(
      `${driverSh('profile.sh', `--artifact ${buildOut} --kernel ${kPath} --out ${profOut}`)}\n` +
      `Return {ok, native_path}.`,
      { model: MODEL.profile, label: 'driver-profile-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
    evidenceOut = await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
      `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
      { model: MODEL.mechanical, label: 'driver-to-evidence-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  } else {
    // Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run a native profiler.
    // Reuse the run.sh throughput above; when method==='perf_heuristic', normalize via the strategist normalizer.
    const _norm = PROFILING_DECISION.normalizer || 'perf_to_evidence.py'
    if (PROFILING_DECISION.method === 'perf_heuristic') {
      evidenceOut = await agentRetry(() => agent(
        `Normalize the run.sh throughput into canonical metrics.\n` +
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${_norm} --baseline ${EXP_DIR}/ksearch_root.result.json --run-json '${JSON.stringify(runOut || {})}'\`.\n` +
        `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}. ` +
        `Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.`,
        { model: MODEL.mechanical, label: 'driver-heuristic-evidence-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    }
  }
  await agentRetry(() => agent(
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
    `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
    { model: MODEL.mechanical, label: 'driver-diagnose-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  await agentRetry(() => agent(
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${EXP_DIR}/ksearch_root.result.json\`.\n` +
    `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
    { model: MODEL.mechanical, label: 'driver-anti-cheat-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  baselineEval.driver_envelope = {
    latency_ms: Number((runOut && runOut.latency_ms) || 0),
    backend_id: DRIVER_BACKEND_ID,
    profiling_method: PROFILING_DECISION.method,
    profiling_confidence: PROFILING_DECISION.confidence,
  }
}

baselineMetric = baselineEval.baseline_metric || 1.0
bestMetric = baselineMetric
log(`Baseline: metric=${baselineMetric}, latency=${baselineEval.baseline_latency_ms || 'N/A'}ms`)
log(`Bottleneck: ${baselineEval.bottleneck_analysis || 'unknown'}`)

// =============================================================================
// Phase 2: Initialize — Build the world model decision tree
// =============================================================================
phase('Initialize')

const initResult = await agentRetry(() => agent(`You are a kernel optimization architect. Build an initial world model decision tree for systematic design space exploration.

# Kernel Specification:
${specText.substring(0, 3000)}

# Operation: ${OP_DESC} (${opType})
# Language: ${langToken(LANGUAGE)}
# Target GPU: ${TARGET_GPU}
# Baseline performance: metric=${baselineMetric}, latency=${baselineEval.baseline_latency_ms || 'N/A'}ms
# Bottleneck: ${baselineEval.bottleneck_analysis || 'unknown'}

# Design Dimensions Identified:
${(setupResult.design_dimensions || []).map((d, i) => `${i + 1}. ${d}`).join('\n')}

# Key Challenges:
${(setupResult.key_challenges || []).map((c, i) => `${i + 1}. ${c}`).join('\n')}

# World Model Structure

Build a decision tree where:
- Root node (id: "root") is a dummy anchor
- Level 1+ nodes represent design DECISIONS (what aspect to optimize)
- Each node has:
  - node_id: unique string identifier
  - parent_id: parent node id
  - node_type: "decision" | "action"
  - decision: what design dimension this addresses
  - choice: the specific strategy chosen
  - action: {title, description, difficulty_1_to_5, score_0_to_1, expected_vs_baseline_factor}
  - status: "open" (no solution attached yet) | "solved" | "failed"
  - children: array of child node ids

# Requirements:
1. Create at least 5 open action nodes across different design dimensions
2. Each action should be concrete enough to implement (not vague like "optimize memory")
3. Assign difficulty 1-5 (1=simple parameter tuning, 5=complete algorithmic redesign)
4. Assign score 0-1 (expected value of pursuing this action, based on bottleneck analysis)
5. Assign expected_vs_baseline_factor (realistic expected speedup if successful)
6. Cover diverse strategies: don't put all nodes in the same design dimension
7. Order by estimated impact: highest-value, lowest-difficulty actions should have higher scores

Return the complete decision tree.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Initialize","ts":"<ts>","status":"done","technique":"world_model_init","speedup":null,"note":"<node_count + open_actions + design dimensions covered, one line>"}`, {
  label: 'init-tree',
  phase: 'Initialize',
  schema: {
    type: 'object',
    properties: {
      decision_tree: { type: 'object' },
      node_count: { type: 'number' },
      open_actions: { type: 'number' },
      design_dimensions: { type: 'array', items: { type: 'string' } },
      initial_hypotheses: { type: 'array', items: { type: 'string' } },
    },
    required: ['decision_tree', 'node_count', 'open_actions'],
  },
}), { retries: 5 })

decisionTree = initResult.decision_tree
log(`World model initialized: ${initResult.node_count} nodes, ${initResult.open_actions} open actions`)
log(`Dimensions: ${(initResult.design_dimensions || []).join(', ')}`)

// =============================================================================
// Search Cycles — Select → Generate/Improve → Evaluate → Refine/Backtrack
// =============================================================================

for (let cycle = 0; cycle < MAX_CYCLES; cycle++) {
  log(`\n=== Cycle ${cycle + 1}/${MAX_CYCLES} | Best: ${bestMetric?.toFixed(3) || 'N/A'}x | Solutions: ${solutionDb.length} ===`)

  // ===========================================================================
  // Cycle Start: Propose action nodes to ensure frontier has enough candidates
  // (K-Search calls propose_action_nodes() at the start of every cycle)
  // ===========================================================================
  phase('Select')

  const proposeResult = await agentRetry(() => agent(`You are a world model manager. Ensure the decision tree has enough high-quality open action nodes on the frontier.

# Current Decision Tree:
${JSON.stringify(decisionTree, null, 2).substring(0, 5000)}

# Frontier Requirements:
- There must be at least 3 open frontier action nodes (status="open", action.title non-empty)
- At least one must have score_0_to_1 > 0.5
- A frontier node is "executable" only if its parent has a solution attached OR it is a direct child of root
- If requirements are already met, return the tree unchanged

# If requirements NOT met:
- Add 2-3 new open action nodes in under-explored design dimensions
- Each new node needs: node_id, parent_id, action.title, action.description, difficulty_1_to_5, score_0_to_1, expected_vs_baseline_factor
- Prefer attaching new actions as children of the best-performing solved nodes (builds on success)

# Current best metric: ${bestMetric || baselineMetric}

Return the (possibly updated) tree and a count of open frontier nodes.`, {
    label: `propose-${cycle}`,
    phase: 'Select',
    schema: {
      type: 'object',
      properties: {
        updated_tree: { type: 'object' },
        open_frontier_count: { type: 'number' },
        nodes_added: { type: 'number' },
      },
      required: ['updated_tree', 'open_frontier_count'],
    },
  }), { retries: 5, allowNull: true })

  if (proposeResult && proposeResult.updated_tree) {
    decisionTree = proposeResult.updated_tree
  }

  // ===========================================================================
  // Select — Choose best frontier action node
  // K-Search selection: sort by (-score, +difficulty, -overall_rating, node_id)
  // Hard constraint: only executable frontier nodes with difficulty <= MAX_DIFFICULTY
  // ===========================================================================

  const selection = await agentRetry(() => agent(`You are a search strategy expert implementing the K-Search action selection algorithm.

# Decision Tree (current state):
${JSON.stringify(decisionTree, null, 2).substring(0, 6000)}

# Selection Algorithm (DETERMINISTIC — follow exactly):
1. Identify all "open frontier" nodes: status="open" AND action.title is non-empty AND (parent has solution_id OR parent is root)
2. Filter: only keep nodes with difficulty_1_to_5 <= ${MAX_DIFFICULTY}
3. Sort remaining by: (-score_0_to_1, +difficulty_1_to_5, -overall_rating_0_to_10, +node_id alphabetically)
4. Select the FIRST node after sorting (highest utility)

If no nodes pass the filter, relax difficulty constraint and try again with all candidates.
If still no candidates, return selected_node_id = null (search exhausted).

# Current State:
- Best metric: ${bestMetric || 'baseline only'}
- Solutions: ${solutionDb.length}
- Cycle: ${cycle + 1}/${MAX_CYCLES}

# Also provide:
- parent_solution_code: code from the parent node's attached solution (or baseline code if parent is root)
- parent_metric: the score of the parent node's solution
- context_for_generation: ancestor path decisions + sibling outcomes (compact)

Return selection result.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Select","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle}","technique":"frontier_selection","speedup":null,"note":"<selected node_id + action_title + score + difficulty, one line>"}`, {
    label: `select-${cycle}`,
    phase: 'Select',
    schema: {
      type: 'object',
      properties: {
        selected_node_id: { type: 'string' },
        action_title: { type: 'string' },
        action_description: { type: 'string' },
        action_score: { type: 'number' },
        action_difficulty: { type: 'number' },
        parent_solution_code: { type: 'string' },
        parent_metric: { type: 'number' },
        parent_is_root: { type: 'boolean' },
        context_for_generation: { type: 'object' },
        reasoning: { type: 'string' },
      },
      required: ['selected_node_id', 'action_title'],
    },
  }), { retries: 5, allowNull: true })

  if (!selection || !selection.selected_node_id) {
    log('No viable action nodes remain — search exhausted.')
    break
  }

  const activeNodeId = selection.selected_node_id
  const parentCode = selection.parent_solution_code || setupResult.baseline_code || ''
  const parentIsRoot = selection.parent_is_root || false
  const baseScore = selection.parent_metric || baselineMetric

  log(`Selected: "${selection.action_title}" (node=${activeNodeId}, score=${selection.action_score || '?'}, difficulty=${selection.action_difficulty || '?'})`)

  // ===========================================================================
  // Phase: Generate — Multi-attempt code generation with dual stagnation detection
  //
  // K-Search has TWO stagnation counters:
  //   1. no_improve_streak: consecutive rounds not beating cycle_best_score
  //   2. no_improve_over_base_streak: consecutive rounds where cycle_best <= parent score
  // Either reaching STAGNATION_WINDOW terminates the cycle.
  //
  // Prompt selection (4 branches):
  //   - Attempt 1: "generate from action" (with or without base code)
  //   - Attempts 2+: "debug" (no passing solution yet) OR "improve" (have passing solution)
  //     Each further splits on whether base code exists.
  // ===========================================================================
  phase('Generate')

  let cycleBestCode = null
  let cycleBestEval = null
  let cycleBestScore = -1
  let currentRawCode = null  // tracks the LAST generated code (for debug prompts)
  let noImproveStreak = 0
  let noImproveOverBaseStreak = 0
  let hasPassedInCycle = false

  for (let attempt = 0; attempt < ATTEMPTS_PER_CYCLE; attempt++) {
    globalRound++
    const isFirstAttempt = attempt === 0

    // Compact WM section injected into every codegen prompt (K-Search does this)
    const wmSection = `\n\n# World Model (persistent decision tree — use it to guide design):\n${JSON.stringify(decisionTree, null, 2).substring(0, 3000)}`

    // Determine base_for_debug: whichever of parentCode and cycleBestCode has higher score
    const baseForDebug = (cycleBestCode && cycleBestScore > baseScore) ? cycleBestCode : parentCode
    const baseForDebugLabel = (cycleBestCode && cycleBestScore > baseScore) ? 'cycle_best' : 'parent'

    let genResult

    if (isFirstAttempt) {
      // Attempt 1: generate from action (with or without base code)
      genResult = await agentRetry(() => agent(`You are an expert ${langToken(LANGUAGE)} kernel developer. Generate a high-performance kernel implementing a SPECIFIC optimization action.

# Operation: ${OP_DESC} (${opType})
# Target: ${TARGET_GPU}
# Language: ${langToken(LANGUAGE)}

# Kernel Specification:
${specText.substring(0, 2000)}

# Action to implement: "${selection.action_title}"
${selection.action_description || ''}

${parentCode ? `# Base code (from parent node — start from this and apply the action):
\`\`\`${langToken(LANGUAGE)}
${parentCode.substring(0, 4000)}
\`\`\`` : '# No base code available — implement from specification directly.'}

# Tree context (ancestor decisions):
${JSON.stringify(selection.context_for_generation || {}).substring(0, 1500)}
${wmSection}

# Requirements:
1. Output COMPLETE, COMPILABLE ${langToken(LANGUAGE)} code
2. Implement ONLY the specified action — keep everything else close to base
3. Must be functionally correct (outputs within rtol=${RTOL}, atol=${ATOL})
4. Target ${TARGET_GPU} architecture
5. Include all necessary imports/headers

Return the complete kernel code.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Generate","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle}-a${attempt}","technique":"<the optimization action implemented this round>","speedup":null,"note":"<key design choices made, one line>"}`, {
        label: `gen-${cycle}-${attempt}`,
        phase: 'Generate',
        schema: {
          type: 'object',
          properties: {
            code: { type: 'string' },
            implementation_notes: { type: 'string' },
            design_choices: { type: 'array', items: { type: 'string' } },
          },
          required: ['code'],
        },
      }), { retries: 5, allowNull: true })
    } else if (!hasPassedInCycle) {
      // Attempts 2+, NO passing solution yet: DEBUG prompt
      // Uses currentRawCode (last attempt's code) as the buggy code to fix
      genResult = await agentRetry(() => agent(`You are an expert ${langToken(LANGUAGE)} kernel developer. The previous attempt has bugs or fails correctness. Debug and fix it.

# Operation: ${OP_DESC} (${opType})
# Target: ${TARGET_GPU}
# Language: ${langToken(LANGUAGE)}

# Kernel Specification:
${specText.substring(0, 1500)}

# Action: "${selection.action_title}"
${selection.action_description || ''}

${parentCode ? `# Base code (known-good reference, from ${baseForDebugLabel}):
\`\`\`${langToken(LANGUAGE)}
${baseForDebug.substring(0, 3000)}
\`\`\`` : ''}

# Buggy code (last attempt — FIX THIS):
\`\`\`${langToken(LANGUAGE)}
${(currentRawCode || '').substring(0, 4000)}
\`\`\`

# Previous evaluation (shows what went wrong):
${JSON.stringify(cycleBestEval || {}, null, 2).substring(0, 1500)}

# Debug round: ${attempt + 1}/${ATTEMPTS_PER_CYCLE}
# Priority: FIX CORRECTNESS FIRST, then optimize performance.
${wmSection}

Return the fixed kernel code.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Generate","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle}-a${attempt}","technique":"debug_fix","speedup":null,"note":"<bugs fixed / changes made this round, one line>"}`, {
        label: `debug-${cycle}-${attempt}`,
        phase: 'Generate',
        schema: {
          type: 'object',
          properties: {
            code: { type: 'string' },
            changes_made: { type: 'string' },
            bugs_fixed: { type: 'array', items: { type: 'string' } },
          },
          required: ['code'],
        },
      }), { retries: 5, allowNull: true })
    } else {
      // Attempts 2+, HAVE a passing solution: IMPROVE prompt
      // Focus on performance, not correctness
      genResult = await agentRetry(() => agent(`You are an expert ${langToken(LANGUAGE)} kernel developer. You have a working solution — improve its performance.

# Operation: ${OP_DESC} (${opType})
# Target: ${TARGET_GPU}
# Language: ${langToken(LANGUAGE)}

# Kernel Specification:
${specText.substring(0, 1500)}

${parentCode ? `# Base code (reference, from ${baseForDebugLabel}):
\`\`\`${langToken(LANGUAGE)}
${baseForDebug.substring(0, 3000)}
\`\`\`` : ''}

# Current working code (improve this):
\`\`\`${langToken(LANGUAGE)}
${(currentRawCode || cycleBestCode || '').substring(0, 4000)}
\`\`\`

# Current performance:
- Last attempt metric: ${cycleBestEval?.metric_value || 'unknown'}
- vs parent (${baseScore}): ${cycleBestScore > baseScore ? 'BEATING' : 'NOT YET BEATING'} parent
- vs global best (${bestMetric}): ${cycleBestScore > bestMetric ? 'NEW BEST' : 'below best'}

# Improvement round: ${attempt + 1}/${ATTEMPTS_PER_CYCLE}
# Target: beat metric ${Math.max(bestMetric || 0, baseScore)}
${wmSection}

Focus on PERFORMANCE OPTIMIZATION. The code is already correct.
Return improved kernel code.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Generate","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle}-a${attempt}","technique":"performance_improve","speedup":null,"note":"<performance change attempted this round, one line>"}`, {
        label: `improve-${cycle}-${attempt}`,
        phase: 'Generate',
        schema: {
          type: 'object',
          properties: {
            code: { type: 'string' },
            changes_made: { type: 'string' },
            expected_improvement: { type: 'string' },
          },
          required: ['code'],
        },
      }), { retries: 5, allowNull: true })
    }

    if (!genResult || !genResult.code) continue

    // Track the LAST generated code (used in debug prompts for next attempt)
    currentRawCode = genResult.code

    // =========================================================================
    // Phase: Evaluate
    // =========================================================================
    phase('Evaluate')

    const evalResult = await agentRetry(() => agent(`You are a kernel evaluation expert. Evaluate this ${langToken(LANGUAGE)} kernel for correctness and performance.

# Kernel Code:
\`\`\`${langToken(LANGUAGE)}
${genResult.code.substring(0, 4000)}
\`\`\`

# Kernel Specification (for correctness reference):
${specText.substring(0, 1500)}

# Evaluation Steps:

## 1. Compilation Check
- Is the code syntactically valid ${langToken(LANGUAGE)}?
- All imports/includes present?

## 2. Correctness Check
- Implements specification correctly?
- No race conditions, out-of-bounds, precision issues?
- Outputs match reference within rtol=${RTOL}, atol=${ATOL}?

## 3. Performance Measurement
${BENCH_CMD ? `Run benchmark: ${BENCH_CMD}` : 'Estimate performance via code analysis.'}
- Latency (ms)
- metric_value = mean_vs_baseline_factor (higher = better, baseline=${baselineMetric})

## 4. Performance Analysis
- Primary bottleneck?
- Underutilized hardware resources?

# Context:
- Baseline: ${baselineMetric}, Parent: ${baseScore}, Global best: ${bestMetric || baselineMetric}
- Action: "${selection.action_title}"

Return evaluation.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if it compiled AND passed correctness, else "error"; speedup is the measured speedup_vs_baseline number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"cycle-${cycle}-a${attempt}","speedup":<number or null>,"technique":"<action under test>","note":"<valid? metric_value + latency; or the failure reason>"}`, {
      label: `eval-${cycle}-${attempt}`,
      phase: 'Evaluate',
      schema: {
        type: 'object',
        properties: {
          is_valid: { type: 'boolean' },
          metric_value: { type: 'number' },
          latency_ms: { type: 'number' },
          speedup_vs_baseline: { type: 'number' },
          pass_rate: { type: 'string' },
          error_log: { type: 'string' },
          performance_analysis: { type: 'string' },
          remaining_bottleneck: { type: 'string' },
        },
        required: ['is_valid', 'metric_value'],
      },
    }), { retries: 5, allowNull: true })

    if (USE_DRIVER_STANDALONE) {
      const suffix = `${cycle}-${attempt}`
      const kPath = ksearchNodeKernelPath(`cycle_${cycle}_a${attempt}`)
      const buildOut = `${EXP_DIR}/cycle_${cycle}_a${attempt}.artifact`
      const profOut = `${EXP_DIR}/cycle_${cycle}_a${attempt}.prof.native`
      await agentRetry(() => agent(
        `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
        `Return its stdout JSON verbatim.`,
        { model: MODEL.mechanical, label: `driver-build-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      const runOut = await agentRetry(() => agent(
        `${driverSh('run.sh', `--artifact ${buildOut} --kernel ${kPath}`)}\n` +
        `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
        { model: MODEL.profile, label: `driver-run-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      let evidenceOut
      if (PROFILING_DECISION.method === 'native_profiler') {
        await agentRetry(() => agent(
          `${driverSh('profile.sh', `--artifact ${buildOut} --kernel ${kPath} --out ${profOut}`)}\n` +
          `Return {ok, native_path}.`,
          { model: MODEL.profile, label: `driver-profile-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
        evidenceOut = await agentRetry(() => agent(
          `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
          `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
          { model: MODEL.mechanical, label: `driver-to-evidence-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      } else {
        // profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}');
        // do NOT run profile.sh. run.sh above already gave throughput; normalize via the strategist's normalizer.
        const _norm = PROFILING_DECISION.normalizer || 'perf_to_evidence.py'
        evidenceOut = await agentRetry(() => agent(
          `Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run profile.sh. ` +
          `run.sh returned latency_ms=${(runOut && runOut.latency_ms) || 'null'}. ` +
          `If method='perf_heuristic', normalize that throughput into canonical metrics by running exactly: ` +
          `\`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${_norm} --baseline ${EXP_DIR}/cycle_${cycle}_a${attempt}.result.json --run-json '${JSON.stringify(runOut || {})}'\`. ` +
          `Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
          `Also write heuristic_bclass (memory_bound|compute_bound|latency_bound) based on the throughput ratio so diagnose.py does not fall to unknown. ` +
          `Return stdout JSON verbatim {ok:true, metrics:{latency_ms:<from run.sh>,dram_pct:<from perf or null>,sm_pct:<from perf or null>,occupancy:null}, coverage:[...], source_backend:'${DRIVER_BACKEND_ID}'}.`,
          { model: MODEL.mechanical, label: `driver-to-evidence-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      }
      const diagOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
        `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
        { model: MODEL.mechanical, label: `driver-diagnose-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${EXP_DIR}/cycle_${cycle}_a${attempt}.result.json\`.\n` +
        `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
        { model: MODEL.mechanical, label: `driver-anti-cheat-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      evalResult.driver_envelope = {
        latency_ms: Number((runOut && runOut.latency_ms) || 0),
        bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
        backend_id: DRIVER_BACKEND_ID,
      }
    } else if (IS_EMBEDDED) {
      // --- Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch) ---
      // SERIAL: this runs inside the per-attempt `for (let attempt...)` loop, which is
      // already sequential — no `await parallel(` over candidates — so embedded modes that
      // mutate the shared project file (inplace) or share the project build (dispatch) never
      // race. KSearch evaluates one attempt at a time; do NOT parallelize this branch.
      const kPath = ksearchNodeKernelPath(`cycle_${cycle}_a${attempt}`)
      const variant = `ksearch_${cycle}_${attempt}`.replace(/[^A-Za-z0-9_]/g, '_')
      let embLatency = 0, embMetrics = {}, embBclass = 'unknown'
      if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
        const embResult = await agentRetry(() => agent(
          `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project kernel: ${INTEG_KERNEL_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
          `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${INTEG_KERNEL_PATH}\n` +
          `2. Apply candidate: cp ${kPath} ${INTEG_KERNEL_PATH}\n3. Build: ${BUILD_CMD}\n4. Test: ${PROJECT_BENCH_CMD}\n5. Benchmark: ${PROJECT_BENCH_CMD}\n` +
          `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${INTEG_KERNEL_PATH}\n` +
          `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-inplace-${cycle}-${attempt}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        embLatency = Number(embResult?.latency_ms || 0)
        embBclass = embResult?.heuristic_bclass || 'unknown'
        embMetrics = embResult?.metrics || { latency_ms: embLatency }
      } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
        const _plan = typeof __embeddedEvalPlan === 'function'
          ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: PROJECT_BENCH_CMD, benchmarkCmd: PROJECT_BENCH_CMD })
          : null
        if (_plan) {
          const embResult = await agentRetry(() => agent(
            `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
            `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
            { model: MODEL.mechanical, label: `embedded-dispatch-${cycle}-${attempt}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
          embLatency = Number(embResult?.latency_ms || 0)
          embBclass = embResult?.heuristic_bclass || 'unknown'
          embMetrics = embResult?.metrics || { latency_ms: embLatency }
        }
      }
      evalResult.driver_envelope = { latency_ms: embLatency, metrics: embMetrics, bottleneck_class: embBclass, backend_id: 'embedded' }
    }

    if (!evalResult) continue

    // Track solution
    solutionDb.push({
      id: `cycle_${cycle}_attempt_${attempt}`,
      code: genResult.code,
      eval: evalResult,
      node_id: activeNodeId,
    })

    const roundScore = evalResult.is_valid ? evalResult.metric_value : -1
    const allPassed = evalResult.is_valid

    // Update cycle best (K-Search: only update if passed AND score > cycle_best_score)
    if (allPassed && roundScore > cycleBestScore) {
      cycleBestCode = genResult.code
      cycleBestEval = evalResult
      cycleBestScore = roundScore
      hasPassedInCycle = true
      noImproveStreak = 0
    } else {
      noImproveStreak++
    }

    // Second stagnation counter: cycle best vs parent/base score
    if (cycleBestScore > 0 && baseScore > 0) {
      if (cycleBestScore > baseScore) {
        noImproveOverBaseStreak = 0
      } else {
        noImproveOverBaseStreak++
      }
    }

    // Dual stagnation detection (K-Search terminates cycle on EITHER)
    if (noImproveStreak >= STAGNATION_WINDOW || noImproveOverBaseStreak >= STAGNATION_WINDOW) {
      log(`Stagnation after ${attempt + 1} attempts (streak=${noImproveStreak}, over_base=${noImproveOverBaseStreak}) — ending cycle`)
      break
    }
  }

  // ===========================================================================
  // Phase: Refine or Backtrack
  // K-Search: cycleSucceeded = at least one PASSED eval in this cycle
  // ===========================================================================
  phase('Refine')

  const cycleSucceeded = hasPassedInCycle

  if (cycleSucceeded) {
    // Update global best
    if (bestMetric === null || cycleBestScore > bestMetric) {
      bestMetric = cycleBestScore
      bestSolution = { code: cycleBestCode, eval: cycleBestEval, node_id: activeNodeId }
      log(`NEW GLOBAL BEST: ${bestMetric.toFixed(3)}x vs baseline`)
    }

    // Refine the tree — attach solution, update scores, add continuation children
    // K-Search hard requirement: the solved node MUST have at least one open child after refine
    const refineResult = await agentRetry(() => agent(`You are a world model manager. The search cycle SUCCEEDED. Update the decision tree.

# Outcome:
- Node: ${activeNodeId}
- Action: "${selection.action_title}"
- Achieved metric: ${cycleBestScore} (vs baseline ${baselineMetric}, vs parent ${baseScore})
- Speedup vs baseline: ${cycleBestEval.speedup_vs_baseline || 'N/A'}x
- Performance analysis: ${cycleBestEval.performance_analysis || 'N/A'}
- Remaining bottleneck: ${cycleBestEval.remaining_bottleneck || 'unknown'}
- Global best: ${bestMetric}

# Current Decision Tree:
${JSON.stringify(decisionTree, null, 2).substring(0, 5000)}

# Tasks (ALL REQUIRED):
1. **Attach solution**: Mark node ${activeNodeId} as status="solved", record metric=${cycleBestScore}
2. **Update ancestor scores**: Increase scores of ancestors if result exceeded expected_vs_baseline_factor; decrease slightly if below
3. **MANDATORY — Add continuation children**: You MUST add at least 2 NEW open action nodes as children of ${activeNodeId}. This is a HARD REQUIREMENT — the solved node must have open children for the search to continue.
   - Each child should address the remaining bottleneck (${cycleBestEval.remaining_bottleneck || 'unknown'}) or explore orthogonal improvements
   - Assign realistic difficulty (1-5) and score (0-1)
4. **Reflect**: Add a note with CURRENT observation, FOLLOW_THROUGH items, and UPDATE_BELIEF adjustments

Return the updated tree. The solved node MUST have at least one open child action node.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Refine","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle}","technique":"attach_solution","speedup":<the achieved metric number, or null>,"note":"<node solved + new continuation actions added + reflection, one line>"}`, {
      label: `refine-${cycle}`,
      phase: 'Refine',
      schema: {
        type: 'object',
        properties: {
          updated_tree: { type: 'object' },
          new_actions_added: { type: 'number' },
          score_updates: { type: 'array', items: { type: 'string' } },
          reflection: { type: 'string' },
        },
        required: ['updated_tree', 'new_actions_added'],
      },
    }), { retries: 5, allowNull: true })

    if (refineResult && refineResult.updated_tree) {
      decisionTree = refineResult.updated_tree
      // Hard fallback: if refine didn't add children, we note it (in real K-Search this inserts a deterministic node)
      if ((refineResult.new_actions_added || 0) < 1) {
        log(`WARNING: refine did not add continuation children — search may stall on this branch`)
      }
      log(`Refined: +${refineResult.new_actions_added || 0} new actions. ${refineResult.reflection || ''}`)
    }
  } else {
    // Backtrack — downgrade node (note_action_too_hard)
    const backtrackResult = await agentRetry(() => agent(`You are a world model manager. The search cycle FAILED — no passing solution was produced. Update the decision tree.

# Outcome:
- Node: ${activeNodeId}
- Action attempted: "${selection.action_title}" (difficulty: ${selection.action_difficulty || '?'})
- Result: NO PASSED SOLUTION in ${Math.min(ATTEMPTS_PER_CYCLE, noImproveStreak + 1)} attempts
- Best attempt: ${cycleBestEval ? `is_valid=${cycleBestEval.is_valid}, metric=${cycleBestEval.metric_value}` : 'all failed'}
- Error: ${cycleBestEval?.error_log || 'compilation/correctness failure'}

# Current Decision Tree:
${JSON.stringify(decisionTree, null, 2).substring(0, 5000)}

# Tasks:
1. **Downgrade node**: Mark ${activeNodeId} status="failed", reduce score_0_to_1 significantly, increase difficulty_1_to_5 by 1 (cap at 5)
2. **Failure analysis**: Why did this action fail? (too complex, wrong prerequisite, incompatible with target arch, etc.)
3. **Add recovery actions**: Add 1-2 NEW easier alternative nodes (lower difficulty, different approach to same dimension)
4. **Update siblings**: If this failure implies sibling strategies are also risky, reduce their scores

Return the updated tree.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Refine","ts":"<ts>","status":"error","candidate_id":"cycle-${cycle}","technique":"backtrack","speedup":null,"note":"<node downgraded + failure analysis + recovery actions added, one line>"}`, {
      label: `backtrack-${cycle}`,
      phase: 'Refine',
      schema: {
        type: 'object',
        properties: {
          updated_tree: { type: 'object' },
          failure_analysis: { type: 'string' },
          recovery_actions: { type: 'array', items: { type: 'string' } },
          downgraded_node: { type: 'string' },
        },
        required: ['updated_tree'],
      },
    }), { retries: 5, allowNull: true })

    if (backtrackResult && backtrackResult.updated_tree) {
      decisionTree = backtrackResult.updated_tree
      log(`Backtracked: ${backtrackResult.failure_analysis || 'action too hard'}`)
      log(`Recovery: ${(backtrackResult.recovery_actions || []).join(', ')}`)
    }
  }

  cycleCount++
}

// =============================================================================
// Final Report
// =============================================================================
phase('Report')

const topSolutions = solutionDb
  .filter(s => s.eval?.is_valid)
  .sort((a, b) => (b.eval.metric_value || 0) - (a.eval.metric_value || 0))
  .slice(0, 5)

const finalReport = await agentRetry(() => agent(`Write a concise technical report on this K-Search kernel optimization campaign.

# K-Search Optimization Results
- Operation: ${OP_DESC} (${opType})
- Language: ${langToken(LANGUAGE)}, Target: ${TARGET_GPU}
- Baseline metric: ${baselineMetric}
- Best metric achieved: ${bestMetric}
- Overall speedup: ${bestMetric ? (bestMetric / baselineMetric).toFixed(2) : 'N/A'}x
- Cycles completed: ${cycleCount}/${MAX_CYCLES}
- Total solutions evaluated: ${solutionDb.length}
- Valid solutions: ${solutionDb.filter(s => s.eval?.is_valid).length}

# Best Solution (node: ${bestSolution?.node_id || 'none'}):
\`\`\`${langToken(LANGUAGE)}
${(bestSolution?.code || '').substring(0, 3000)}
\`\`\`

# Top 5 Solutions:
${topSolutions.map((s, i) => `${i + 1}. ${s.id} (node=${s.node_id}, metric=${s.eval.metric_value})`).join('\n')}

# Final Decision Tree State:
${JSON.stringify(decisionTree, null, 2).substring(0, 3000)}

# Write:
1. Search trajectory: which actions were attempted in what order, success/failure pattern
2. Key insights: what design decisions yielded the most improvement?
3. World model evolution: how did the tree structure change over time?
4. Failed strategies: what was tried and abandoned, and why?
5. Remaining opportunities: what open actions in the tree look promising for future exploration?

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","technique":"search_summary","speedup":<the best metric vs baseline number, or null>,"note":"<best node + cycles completed + winning strategy, one line>"}`, {
  label: 'final-report',
  phase: 'Report',
}), { retries: 5 })

// embedded_inplace exit safety net — ALWAYS restore the project file byte-exact.
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${INTEG_KERNEL_PATH}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: KERNEL_SPEC_PATH,
  kernel_path: BASELINE_CODE_PATH,
  generated_kernel_path: bestSolution?.path || '',
  initial_candidates: solutionDb.filter(s => s.cycle === 0),
  initial_generation_result: {
    verified: solutionDb.some(s => s.eval?.correct),
    selected_candidate_id: bestSolution?.id || '',
  },
  best_metric: bestMetric,
  best_solution_code: bestSolution?.code || '',
  cycles_completed: cycleCount,
  solutions_evaluated: solutionDb.length,
  decision_tree: decisionTree,
  solution_lineage: topSolutions.map(s => ({
    id: s.id,
    node_id: s.node_id,
    metric: s.eval.metric_value,
  })),
  report: finalReport,
  baseline_metric: baselineMetric,
  speedup_over_baseline: bestMetric ? bestMetric / baselineMetric : 1.0,
}
