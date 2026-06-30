export const meta = {
  name: 'regrapht-kernel-optimization',
  description: 'Reasoning-graph-guided CUDA optimization with Monte Carlo Graph Search (ReGraphT methodology)',
  whenToUse: 'When you want to guide a local or smaller coding model with reusable CUDA optimization trajectories instead of relying only on free-form prompting. Builds or loads a CUDA Reasoning Graph, selects graph paths with Monte Carlo Graph Search, generates candidates from retrieved optimization examples, and updates the graph from real compile/correctness/benchmark feedback.',
  phases: [
    { title: 'Setup', detail: 'Read source kernel/spec, evaluator contract, optional trace corpus, and optional existing reasoning graph' },
    { title: 'BuildGraph', detail: 'Construct or refresh the CUDA Reasoning Graph from optimization trajectories and method labels' },
    { title: 'Select', detail: 'Use Monte Carlo Graph Search to choose a promising optimization path' },
    { title: 'Generate', detail: 'Generate a CUDA candidate conditioned on selected graph examples and method sequence' },
    { title: 'Evaluate', detail: 'Compile, correctness-test, and benchmark the generated candidate' },
    { title: 'UpdateGraph', detail: 'Backpropagate measured reward, record evidence, and expand/relabel graph nodes if needed' },
    { title: 'Report', detail: 'Return best kernel, selected reasoning paths, final graph statistics, and next opportunities' },
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

const WORKFLOW_NAME = 'regrapht-kernel-optimization'


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


// =============================================================================
// ReGraphT — CUDA Reasoning Graph + Monte Carlo Graph Search Workflow
// =============================================================================
//
// Source: "From Large to Small: Transferring CUDA Optimization Expertise via
//         Reasoning Graph"
//         arXiv:2510.19873 — Junfeng Gong, Zhiyi Wei, Junying Chen,
//         Cheng Liu, Huawei Li
//
// Boundary:
//   This workflow operationalizes ReGraphT as an agent-executable inference and
//   optimization loop. It does not train or fine-tune a small language model.
//   Instead, it builds or loads a CUDA Reasoning Graph from optimization traces,
//   traverses it with Monte Carlo Graph Search, prompts a coding agent with the
//   selected method/example path, and trusts only evaluator evidence.
// adaptation_scope: training_free_inference — this covers the graph-guided
// inference/use phase, not full small-model transfer training.
//
// Usage:
//   Workflow({name: 'regrapht-kernel-optimization', args: {
//     kernel_path: '/path/to/source.cu',    // alias: kernel_path
//     op_description: 'Sequential stencil kernel to CUDA',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     trace_corpus_path: '/path/to/llm_optimization_traces.jsonl',
//     graph_path: '/path/to/regraph.json',
//     baseline_command: '<user-provided baseline command with {result_path}>',
//     iterations: 20,
//     rollouts_per_select: 12,
//     exploration_weight: 1.4,
//     max_path_length: 4,
//     target_gpu: 'H100',
//     exp_dir: '/tmp/regrapht_exp',
//   }})
//
// Evaluator JSON contract:
//   benchmark_command should write JSON at {result_path}:
//   {
//     "compiled": true,
//     "correct": true,
//     "speedup": 1.23,
//     "kernel_time_ms": 0.12,
//     "baseline_time_ms": 0.15,
//     "error_message": "",
//     "error_type": ""
//   }
//
// =============================================================================

// --- Required Args ---
let SOURCE_CODE_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = SOURCE_CODE_PATH ? 'optimize_existing' : 'generate_then_optimize'
const OP_DESC = args.op_description || 'CUDA kernel optimization task'
const EVAL_CMD = args.benchmark_command || ''

// --- Optional Args ---
const TRACE_CORPUS_PATH = args.trace_corpus_path || ''
const GRAPH_PATH = args.graph_path || ''
const BASELINE_CMD = args.baseline_command || ''
const BUDGET = args.iterations || 20
const ROLLOUTS_PER_SELECT = args.rollouts_per_select || 12
const EXPLORATION_WEIGHT = args.exploration_weight ?? 1.4
const MAX_PATH_LENGTH = args.max_path_length || 4
const TARGET_GPU = args.target_gpu || 'H100'
const EXP_DIR = args.exp_dir || '/tmp/regrapht_exp'
const ADAPTATION_SCOPE = 'training_free_inference'
const LANGUAGE = args.language || 'cuda'
const SEED_CANDIDATES = args.seed_candidates || 3

if (!SOURCE_CODE_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

// --- Backend driver wiring (P5d Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_LANG_TOKEN = LANGUAGE
const LEGACY_FENCE_TOKEN = LANGUAGE
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

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

function regraphtNodeKernelPath(label) {
  const ext = USE_DRIVER ? (DRIVER_SOURCE_EXT || '.cu') : '.cu'
  return `${EXP_DIR}/${label}${ext}`
}

// --- State ---
let sourceCode = ''
let baselineMetric = 1.0
let bestCandidate = null
let graph = {
  nodes: [
    {
      id: 'v_init',
      method: 'initial sequential/source code',
      visits: 0,
      reward: 0,
      examples: [],
    },
  ],
  edges: [],
}
let evaluatedCandidates = []
let selectedPaths = []
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null

function nodeById(id) {
  return graph.nodes.find(node => node.id === id)
}

function outgoingEdges(nodeId) {
  return graph.edges.filter(edge => edge.from === nodeId)
}

function averageReward(nodeId) {
  const node = nodeById(nodeId)
  if (!node || !node.visits) return 0
  return (node.reward || 0) / node.visits
}

function edgeScore(edge, parentVisits) {
  const child = nodeById(edge.to)
  if (!child || !child.visits) return Infinity
  const exploitation = averageReward(edge.to)
  const exploration = EXPLORATION_WEIGHT * Math.sqrt(Math.log(Math.max(parentVisits, 1)) / child.visits)
  return exploitation + exploration + (edge.prior || 0)
}

function bestMeasured() {
  const valid = evaluatedCandidates.filter(item => item.eval?.correct)
  if (!valid.length) return null
  return valid.reduce((best, item) => (item.eval.speedup || 0) > (best.eval.speedup || 0) ? item : best, valid[0])
}

function graphStats() {
  return {
    nodes: graph.nodes.length,
    edges: graph.edges.length,
    visited_nodes: graph.nodes.filter(node => node.visits > 0).length,
    evaluated_candidates: evaluatedCandidates.length,
    correct_candidates: evaluatedCandidates.filter(item => item.eval?.correct).length,
  }
}

// =============================================================================
// Phase 1: Setup
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

if (INPUT_MODE === 'generate_then_optimize') {
  const generated = await agentRetry(() => agent(`No kernel_path was provided. Generate and verify an initial CUDA source file before building the ReGraphT reasoning graph.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- op_description: ${OP_DESC}
- language: ${langToken(LANGUAGE)}
- target_gpu: ${TARGET_GPU}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- benchmark_command: ${EVAL_CMD || '(not provided)'}
- baseline_command: ${BASELINE_CMD || '(not provided)'}

# Contract
Generate ${SEED_CANDIDATES} complete candidates under ${EXP_DIR}/generated/. Run benchmark_command if available using {kernel_path}/{result_path}. Return the best verified generated kernel path.`, {
    label: 'generate-initial-kernel',
    phase: 'Setup',
    schema: {
      type: 'object',
      properties: {
        generated_kernel_path: { type: 'string' },
        initial_candidates: { type: 'array', items: { type: 'object' } },
        initial_generation_result: { type: 'object' },
      },
      required: ['generated_kernel_path', 'initial_candidates', 'initial_generation_result'],
    },
  }), { retries: 5 })
  initialCandidates = generated.initial_candidates || []
  initialGenerationResult = generated.initial_generation_result || { verified: false }
  generatedKernelPath = generated.generated_kernel_path || ''
  if (!generatedKernelPath) throw new Error('Generation mode did not produce generated_kernel_path')
  if (EVAL_CMD && initialGenerationResult.verified === false) throw new Error('No generated seed passed benchmark evidence')
  SOURCE_CODE_PATH = generatedKernelPath
}

const setupResult = await agentRetry(() => agent(`Read the CUDA optimization task and evaluator contract.

# Inputs
- kernel_path: ${SOURCE_CODE_PATH}
- operation: ${OP_DESC}
- target_gpu: ${TARGET_GPU}
- benchmark_command: ${EVAL_CMD || '(missing; do not create an evaluator command; measured evidence is unavailable)'}
- baseline_command: ${BASELINE_CMD || '(optional baseline command not provided)'}
- exp_dir: ${EXP_DIR}

# Tasks
1. Read the source code or task spec from kernel_path.
2. Identify the operation, input/output contract, and target kernel entry points.
3. If baseline_command is present, run it or describe the exact command to run and parse its JSON result.
4. State the required evaluator JSON contract. Evaluation evidence must come from benchmark_command; if it is missing, mark evidence as unavailable.
5. Identify CUDA optimization dimensions relevant to this task.

Return structured setup data.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"task_setup","speedup":null,"note":"<operation type + key optimization dimensions + whether baseline evidence is available, one line>"}`, {
  label: 'setup-task',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      source_code: { type: 'string' },
      operation_type: { type: 'string' },
      input_contract: { type: 'string' },
      output_contract: { type: 'string' },
      kernel_entry_points: { type: 'array', items: { type: 'string' } },
      baseline_metric: { type: 'number' },
      baseline_time_ms: { type: 'number' },
      evaluator_contract: { type: 'string' },
      optimization_dimensions: { type: 'array', items: { type: 'string' } },
    },
    required: ['source_code', 'operation_type', 'evaluator_contract'],
  },
}), { retries: 5 })

sourceCode = setupResult.source_code || ''
baselineMetric = setupResult.baseline_metric || 1.0

// =============================================================================
// Phase 2: BuildGraph
// =============================================================================
phase('BuildGraph')

const graphResult = await agentRetry(() => agent(`Build or refresh a CUDA Reasoning Graph for ReGraphT.

# Paper-derived graph contract
- A node represents a CUDA optimization method or intermediate optimization state.
- A directed edge represents a transition between two methods.
- The graph may contain cycles.
- Each node and edge should retain examples extracted from optimization trajectories.
- Method labels must be normalized so equivalent LLM phrases map to one method.

# Inputs
- graph_path: ${GRAPH_PATH || '(none; construct from trace corpus and task dimensions)'}
- trace_corpus_path: ${TRACE_CORPUS_PATH || '(none; synthesize a minimal seed graph from task dimensions)'}
- operation: ${OP_DESC}
- target_gpu: ${TARGET_GPU}
- setup optimization dimensions: ${(setupResult.optimization_dimensions || []).join(', ') || 'unknown'}

# Source code excerpt
\`\`\`${fenceToken()}
${sourceCode.substring(0, 5000)}
\`\`\`

# Tasks
1. If graph_path exists, read the graph and normalize it to {nodes, edges}.
2. If trace_corpus_path exists, extract optimization trajectories, method labels, examples, and transitions.
3. If no corpus exists, create a conservative seed graph using standard CUDA method nodes relevant to this operation.
4. Include v_init as the start node.
5. For each edge, include a prior score and one or more example snippets when available.

Return a graph suitable for Monte Carlo Graph Search.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"BuildGraph","ts":"<ts>","status":"done","technique":"reasoning_graph_construction","speedup":null,"note":"<node count + edge count + how built (loaded graph_path / extracted from trace corpus / synthesized seed), one line>"}`, {
  label: 'build-regraph',
  phase: 'BuildGraph',
  schema: {
    type: 'object',
    properties: {
      graph: { type: 'object' },
      method_labels: { type: 'array', items: { type: 'string' } },
      trace_count: { type: 'number' },
      normalization_notes: { type: 'array', items: { type: 'string' } },
    },
    required: ['graph', 'method_labels'],
  },
}), { retries: 5, allowNull: true })

if (graphResult?.graph?.nodes && graphResult?.graph?.edges) {
  graph = graphResult.graph
}

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it below. The agent only classifies the task (fuzzy); the substrate stamps
// confidence by method (measured/inferred/hypothesized) -- not the agent. See
// _substrate/profiling/README.md. Falls back to native_profiler if undecided. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }
if (USE_DRIVER) {
  const _pd = await agentRetry(() => agent(
    `Read ${SOURCE_CODE_PATH}; classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    `run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${BACKEND_DIR}/manifest.json --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'BuildGraph', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}

// --- integration-strategist: route build/test mode (standalone vs embedded_*). ---
// Project-native args for inference-engine embedded operators (e.g. llama.cpp .cuh).
// BENCH_CMD would collide with EVAL_CMD/benchmark_command, so use PROJECT_BENCH_CMD.
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const BUILD_CMD = args.build_command || ''
const PROJECT_BENCH_CMD = args.project_benchmark_command || EVAL_CMD || ''
const REGISTER_SCRIPT = args.register_script || ''
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _profManifest = (USE_DRIVER && BACKEND_DIR) ? `${BACKEND_DIR}/manifest.json` : `${SUBSTRATE}/backends/cuda/manifest.json`
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Read ${SOURCE_CODE_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${SOURCE_CODE_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
    `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'BuildGraph', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
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
  await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${SOURCE_CODE_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'BuildGraph', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// A-O1 closure: native_profiler but no standalone driver path (embedded/legacy) → perf_heuristic
if (PROFILING_DECISION.method === 'native_profiler' && !USE_DRIVER_STANDALONE) {
  log(`profiling: native_profiler but no standalone driver path -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'test-harness-perf', rationale: 'native_profiler but no standalone driver -> perf_heuristic' }
}

if (USE_DRIVER_STANDALONE) {
  const kPath = SOURCE_CODE_PATH || regraphtNodeKernelPath('regrapht_root')
  const buildOut = `${EXP_DIR}/regrapht_root.artifact`
  const profOut = `${EXP_DIR}/regrapht_root.prof.native`
  await agentRetry(() => agent(
    `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
    `Return its stdout JSON verbatim.`,
    { model: MODEL.mechanical, label: 'driver-build-root', phase: 'BuildGraph', schema: JSON_PASSTHROUGH }), { retries: 5 })
  await agentRetry(() => agent(
    `${driverSh('run.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${buildOut}.run.json`)}\n` +
    `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
    { model: MODEL.profile, label: 'driver-run-root', phase: 'BuildGraph', schema: JSON_PASSTHROUGH }), { retries: 5 })
  if (PROFILING_DECISION.method === 'native_profiler') {
    await agentRetry(() => agent(
      `${driverSh('profile.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${buildOut}.run.json --out ${profOut}`)}\n` +
      `Return {ok, native_path}.`,
      { model: MODEL.profile, label: 'driver-profile-root', phase: 'BuildGraph', schema: JSON_PASSTHROUGH }), { retries: 5 })
    const evidenceOut = await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
      `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
      { model: MODEL.mechanical, label: 'driver-to-evidence-root', phase: 'BuildGraph', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
      `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
      { model: MODEL.mechanical, label: 'driver-diagnose-root', phase: 'BuildGraph', schema: JSON_PASSTHROUGH }), { retries: 5 })
  } else {
    // profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}');
    // do NOT run profile.sh / to_evidence.py. Normalize run.sh throughput via the strategist normalizer when perf_heuristic.
    if (PROFILING_DECISION.method === 'perf_heuristic') {
      const _normalizer = PROFILING_DECISION.normalizer || 'perf_to_evidence.py'
      await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${_normalizer} --baseline ${EXP_DIR}/regrapht_root.result.json\`.\n` +
        `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage}. ` +
        `Also write heuristic_bclass (memory_bound|compute_bound|latency_bound) based on the throughput ratio so diagnose.py does not fall to unknown. ` +
        `Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.`,
        { model: MODEL.mechanical, label: 'driver-perf-heuristic-root', phase: 'BuildGraph', schema: JSON_PASSTHROUGH }), { retries: 5 })
    }
    await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '{}' \`.\n` +
      `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
      { model: MODEL.mechanical, label: 'driver-diagnose-root', phase: 'BuildGraph', schema: JSON_PASSTHROUGH }), { retries: 5 })
  }
  await agentRetry(() => agent(
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${EXP_DIR}/regrapht_root.result.json\`.\n` +
    `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
    { model: MODEL.mechanical, label: 'driver-anti-cheat-root', phase: 'BuildGraph', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

// =============================================================================
// Main Loop: Select -> Generate -> Evaluate -> UpdateGraph
// =============================================================================
for (let attempt = 0; attempt < BUDGET; attempt++) {
  log(`\n=== ReGraphT attempt ${attempt + 1}/${BUDGET} | best=${bestCandidate?.eval?.speedup || 0}x | graph=${graph.nodes.length} nodes/${graph.edges.length} edges ===`)

  phase('Select')

  const selection = await agentRetry(() => agent(`Select a promising CUDA optimization path with Monte Carlo Graph Search.

# ReGraph state
\`\`\`json
${JSON.stringify(graph, null, 2).substring(0, 10000)}
\`\`\`

# Selection constraints
- Start from v_init.
- Run ${ROLLOUTS_PER_SELECT} conceptual MCGS rollouts.
- Use UCT-style selection: average_reward + exploration + edge prior.
- Limit path length to ${MAX_PATH_LENGTH}.
- Prefer paths whose examples fit the current operation and target GPU.
- Do not select methods contradicted by evaluator failures in prior attempts.

# Prior evaluated candidates
\`\`\`json
${JSON.stringify(evaluatedCandidates.slice(-8), null, 2).substring(0, 8000)}
\`\`\`

Return the selected method path and the examples that should condition generation.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is MCGS attempt ${attempt}):
{"workflow":"${WORKFLOW_NAME}","phase":"Select","ts":"<ts>","status":"done","candidate_id":"attempt_${attempt}","technique":"mcgs_path_selection","speedup":null,"note":"<selected method sequence + selection rationale, one line>"}`, {
    label: `select-${attempt}`,
    phase: 'Select',
    schema: {
      type: 'object',
      properties: {
        path_node_ids: { type: 'array', items: { type: 'string' } },
        method_sequence: { type: 'array', items: { type: 'string' } },
        selected_examples: { type: 'array', items: { type: 'object' } },
        selection_rationale: { type: 'string' },
        expected_speedup: { type: 'number' },
      },
      required: ['path_node_ids', 'method_sequence', 'selection_rationale'],
    },
  }), { retries: 5 })

  selectedPaths.push(selection)

  phase('Generate')

  const generation = await agentRetry(() => agent(`Generate a CUDA optimization candidate using the selected ReGraphT path.

# Operation
${OP_DESC}

# Target GPU
${TARGET_GPU}

# Original/source code
\`\`\`${fenceToken()}
${sourceCode.substring(0, 8000)}
\`\`\`

# Selected CUDA optimization method sequence
${(selection.method_sequence || []).map((item, i) => `${i + 1}. ${item}`).join('\n')}

# Retrieved optimization examples
\`\`\`json
${JSON.stringify(selection.selected_examples || [], null, 2).substring(0, 10000)}
\`\`\`

# Evaluator contract
${setupResult.evaluator_contract}

# Hard requirements
1. Use the selected method sequence only when it is suitable for the current code.
2. Produce complete compilable CUDA/C++ code, not a patch fragment.
3. Preserve correctness according to the input/output contract.
4. If a selected method is unsuitable, explain why and apply the next suitable method in the path.
5. Do not claim speedup without evaluator evidence.

Return candidate code and suitability decisions for each method.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is MCGS attempt ${attempt}):
{"workflow":"${WORKFLOW_NAME}","phase":"Generate","ts":"<ts>","status":"done","candidate_id":"attempt_${attempt}","technique":"<the main optimization method actually applied this candidate>","speedup":null,"note":"<applied vs skipped methods + what changed in the code>"}`, {
    label: `generate-${attempt}`,
    phase: 'Generate',
    schema: {
      type: 'object',
      properties: {
        candidate_code: { type: 'string' },
        applied_methods: { type: 'array', items: { type: 'string' } },
        skipped_methods: { type: 'array', items: { type: 'string' } },
        suitability_notes: { type: 'string' },
        implementation_notes: { type: 'string' },
      },
      required: ['candidate_code', 'applied_methods'],
    },
  }), { retries: 5 })

  phase('Evaluate')

  const evaluation = await agentRetry(() => agent(`Evaluate the generated CUDA candidate with real evidence.

# Candidate code
\`\`\`${fenceToken()}
${(generation.candidate_code || '').substring(0, 12000)}
\`\`\`

# Evaluation command
${EVAL_CMD || '(no benchmark_command provided)'}

# Result path convention
Use ${EXP_DIR}/regrapht_attempt_${attempt}.json as {result_path}.
Use ${EXP_DIR}/regrapht_attempt_${attempt}.cu as {kernel_path}.

# Required behavior
1. Materialize the candidate at the kernel path if command execution is available.
2. Run benchmark_command with {kernel_path} and {result_path} substitutions if provided.
3. Parse evaluator JSON. If no command is provided, mark correct=false and explain missing evidence.
4. Correctness and speedup must be based on evaluator output only.

Return evaluator evidence.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if correctness passed, else "error"; speedup is the measured speedup number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"attempt_${attempt}","speedup":<number or null>,"technique":"<methods under test>","note":"<compiled? correct? measured kernel/baseline ms; or the failure reason>"}`, {
    label: `evaluate-${attempt}`,
    phase: 'Evaluate',
    schema: {
      type: 'object',
      properties: {
        compiled: { type: 'boolean' },
        correct: { type: 'boolean' },
        speedup: { type: 'number' },
        kernel_time_ms: { type: 'number' },
        baseline_time_ms: { type: 'number' },
        result_path: { type: 'string' },
        error_message: { type: 'string' },
        error_type: { type: 'string' },
        evidence_summary: { type: 'string' },
      },
      required: ['compiled', 'correct', 'speedup'],
    },
  }), { retries: 5 })

  if (USE_DRIVER_STANDALONE) {
    const suffix = `${attempt}`
    const kPath = regraphtNodeKernelPath(`regrapht_attempt_${attempt}`)
    const buildOut = `${EXP_DIR}/regrapht_attempt_${attempt}.artifact`
    const profOut = `${EXP_DIR}/regrapht_attempt_${attempt}.prof.native`
    await agentRetry(() => agent(
      `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
      `Return its stdout JSON verbatim.`,
      { model: MODEL.mechanical, label: `driver-build-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    const runOut = await agentRetry(() => agent(
      `${driverSh('run.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${buildOut}.run.json`)}\n` +
      `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
      { model: MODEL.profile, label: `driver-run-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    await agentRetry(() => agent(
      `${driverSh('profile.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${buildOut}.run.json --out ${profOut}`)}\n` +
      `Return {ok, native_path}.`,
      { model: MODEL.profile, label: `driver-profile-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    const evidenceOut = await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
      `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
      { model: MODEL.mechanical, label: `driver-to-evidence-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    const diagOut = await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
      `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
      { model: MODEL.mechanical, label: `driver-diagnose-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${EXP_DIR}/regrapht_attempt_${attempt}.result.json\`.\n` +
      `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
      { model: MODEL.mechanical, label: `driver-anti-cheat-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    evaluation.driver_envelope = {
      latency_ms: Number((runOut && runOut.latency_ms) || 0),
      bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
      backend_id: DRIVER_BACKEND_ID,
    }
  } else if (IS_EMBEDDED) {
    // --- Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch) ---
    // Serial: this MCGS main loop is a sequential `for (attempt...)` loop, so there is no
    // race on the shared project source (inplace) or shared project build (dispatch).
    const kPath = regraphtNodeKernelPath(`regrapht_attempt_${attempt}`)
    const variant = `regrapht_${attempt}`.replace(/[^A-Za-z0-9_]/g, '_')
    let embLatency = 0, embMetrics = {}, embBclass = 'unknown'
    if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
      const embResult = await agentRetry(() => agent(
        `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project kernel: ${SOURCE_CODE_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
        `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${SOURCE_CODE_PATH}\n` +
        `2. Apply candidate: cp ${kPath} ${SOURCE_CODE_PATH}\n3. Build: ${BUILD_CMD}\n4. Test: ${args.test_command || PROJECT_BENCH_CMD}\n5. Benchmark: ${PROJECT_BENCH_CMD || args.test_command}\n` +
        `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${SOURCE_CODE_PATH}\n` +
        `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
        { model: MODEL.mechanical, label: `embedded-inplace-${attempt}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      embLatency = Number(embResult?.latency_ms || 0); embBclass = embResult?.heuristic_bclass || 'unknown'; embMetrics = embResult?.metrics || { latency_ms: embLatency }
    } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
      const _plan = typeof __embeddedEvalPlan === 'function'
        ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: args.test_command || PROJECT_BENCH_CMD, benchmarkCmd: PROJECT_BENCH_CMD || args.test_command })
        : null
      if (_plan) {
        const embResult = await agentRetry(() => agent(
          `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
          `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-dispatch-${attempt}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        embLatency = Number(embResult?.latency_ms || 0); embBclass = embResult?.heuristic_bclass || 'unknown'; embMetrics = embResult?.metrics || { latency_ms: embLatency }
      }
    }
    evaluation.driver_envelope = { latency_ms: embLatency, metrics: embMetrics, bottleneck_class: embBclass, backend_id: 'embedded' }
  }

  const candidateRecord = {
    id: `attempt_${attempt}`,
    path_node_ids: selection.path_node_ids || [],
    methods: generation.applied_methods || [],
    code: generation.candidate_code || '',
    eval: evaluation,
  }
  evaluatedCandidates.push(candidateRecord)

  const measuredBest = bestMeasured()
  if (measuredBest && (!bestCandidate || (measuredBest.eval.speedup || 0) > (bestCandidate.eval.speedup || 0))) {
    bestCandidate = measuredBest
  }

  phase('UpdateGraph')

  const update = await agentRetry(() => agent(`Update the CUDA Reasoning Graph from measured evaluator feedback.

# Selected path
\`\`\`json
${JSON.stringify(selection, null, 2).substring(0, 6000)}
\`\`\`

# Generated methods
${(generation.applied_methods || []).join(' -> ') || '(none)'}

# Evaluator result
\`\`\`json
${JSON.stringify(evaluation, null, 2)}
\`\`\`

# Current graph
\`\`\`json
${JSON.stringify(graph, null, 2).substring(0, 10000)}
\`\`\`

# Update rules
1. Increment visits along selected nodes.
2. Add reward equal to speedup for correct candidates, small positive reward for compile-only progress, and zero/negative evidence for failures.
3. Attach the concrete optimization example to the traversed edge or final node.
4. If the candidate used a new method label, add or relabel a node instead of duplicating equivalent methods.
5. Add 1-3 follow-up edges from the best promising node when evidence suggests a next method.
6. If graph_path is provided, write the updated graph there.

Return the updated graph and update summary.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is MCGS attempt ${attempt}):
{"workflow":"${WORKFLOW_NAME}","phase":"UpdateGraph","ts":"<ts>","status":"done","candidate_id":"attempt_${attempt}","technique":"reward_backprop","speedup":null,"note":"<reward applied + added/relabeled nodes/edges, one line>"}`, {
    label: `update-graph-${attempt}`,
    phase: 'UpdateGraph',
    schema: {
      type: 'object',
      properties: {
        updated_graph: { type: 'object' },
        reward: { type: 'number' },
        added_nodes: { type: 'array', items: { type: 'string' } },
        added_edges: { type: 'array', items: { type: 'string' } },
        relabel_notes: { type: 'array', items: { type: 'string' } },
        wrote_graph_path: { type: 'string' },
      },
      required: ['updated_graph', 'reward'],
    },
  }), { retries: 5, allowNull: true })

  if (update?.updated_graph?.nodes && update?.updated_graph?.edges) {
    graph = update.updated_graph
  }
}

// =============================================================================
// Phase 7: Report
// =============================================================================
phase('Report')

const finalGraphStats = graphStats()
const finalReport = await agentRetry(() => agent(`Write a concise technical report for this ReGraphT optimization run.

# Operation
${OP_DESC}

# Boundary
This workflow used ReGraphT as a training-free inference loop. It built or loaded a CUDA Reasoning Graph, selected paths with Monte Carlo Graph Search, generated candidates from graph-conditioned examples, and used evaluator evidence for reward.

# Adaptation scope
${ADAPTATION_SCOPE}

# Final graph stats
\`\`\`json
${JSON.stringify(finalGraphStats, null, 2)}
\`\`\`

# Best candidate
\`\`\`json
${JSON.stringify(bestCandidate ? {
  id: bestCandidate.id,
  methods: bestCandidate.methods,
  eval: bestCandidate.eval,
} : null, null, 2)}
\`\`\`

# Best code excerpt
\`\`\`${fenceToken()}
${(bestCandidate?.code || '').substring(0, 5000)}
\`\`\`

# Selected paths
\`\`\`json
${JSON.stringify(selectedPaths.slice(-10), null, 2).substring(0, 10000)}
\`\`\`

Cover:
1. Which reasoning-graph methods were most useful.
2. Which methods were unsuitable or failed, based on evaluator evidence.
3. Whether the best candidate is trustworthy.
4. Which open graph paths should be tried next.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the best measured candidate (speedup is the best measured speedup number, or null if no correct candidate):
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","technique":"final_report","speedup":<number or null>,"note":"<best candidate id + most useful reasoning-graph methods + whether result is trustworthy>"}`, {
  label: 'final-report',
  phase: 'Report',
}), { retries: 5 })

// embedded_inplace exit safety net: unconditionally restore the project source.
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${SOURCE_CODE_PATH}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  kernel_path: SOURCE_CODE_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  operation: OP_DESC,
  baseline_metric: baselineMetric,
  best_speedup: bestCandidate?.eval?.speedup || 0,
  best_kernel_code: bestCandidate?.code || '',
  best_candidate_id: bestCandidate?.id || '',
  attempts: BUDGET,
  evaluated_candidates: evaluatedCandidates.length,
  correct_candidates: evaluatedCandidates.filter(item => item.eval?.correct).length,
  adaptation_scope: ADAPTATION_SCOPE,
  selected_paths: selectedPaths,
  graph,
  graph_stats: finalGraphStats,
  report: finalReport,
}
