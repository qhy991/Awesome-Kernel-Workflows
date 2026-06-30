export const meta = {
  name: 'adaexplore-kernel-optimization',
  description: 'Standalone AdaExplore-style workflow: failure-driven skill memory plus diversity-preserving MCTS for Triton kernel optimization',
  whenToUse: 'When an agent must optimize Triton GPU kernels from a PyTorch operator specification without calling the AdaExplore repository. Requires a user-provided evaluator command for measured correctness/performance evidence. Explore mode reads skill memory; Adapt mode may update skill memory from failure logs.',
  phases: [
    { title: 'Setup', detail: 'Materialize operator spec, evaluator contract, experiment folders, and skill memory' },
    { title: 'Select', detail: 'Use UCB1 plus expand-UCB1 to choose whether to deepen or create a sibling' },
    { title: 'Expand', detail: 'Large step proposes structurally new kernels; small step applies reviser-guided surgical edits' },
    { title: 'Evaluate', detail: 'Run a real compile/correctness/performance harness; do not rely on LLM self-judgment' },
    { title: 'Backpropagate', detail: 'Update tree reward statistics from measured results' },
    { title: 'AdaptMemory', detail: 'Optionally distill failed evaluated logs into scored "You cannot..." constraints' },
    { title: 'Report', detail: 'Return best measured kernel, search tree stats, and memory changes' },
  ],
  requiredSkills: [],
  optionalSkills: [],
  skillMemoryContract: {
    type: 'method_memory_file',
    path_arg: 'skill_memory_path',
    update_arg: 'memory_update',
    rule_format: 'You cannot ... || score',
  },
}

// __modelTierApplied (declaration pre-existing)

const WORKFLOW_NAME = 'adaexplore-kernel-optimization'


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
// AdaExplore-Style Standalone Kernel Optimization
// =============================================================================
//
// Boundary:
//   This workflow intentionally does NOT call the AdaExplore repository entrypoint.
//   It turns the method into an agent-executable workflow that can run in a
//   generic workspace when given a PyTorch operator spec and an evaluator.
//
// Method:
//   1. Adapt: failures can be summarized into reusable "You cannot..." rules.
//   2. Explore: MCTS alternates large structural proposals and small local edits.
//
// Important execution contract:
//   - Correctness and speedup must come from a real user-provided evaluator
//     command. The evaluator result is authoritative.
//   - In normal benchmark Explore mode, memory is read-only.
//   - Memory update is explicit via memory_update/adapt mode and is driven by
//     evaluated failure logs, not by speculative LLM self-assessment.
//
// Usage:
//   Workflow({name: 'adaexplore-kernel-optimization', args: {
//     problem_definition: 'class Model(nn.Module): ...',
//     op_description: 'Fused LayerNorm + GELU activation',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     baseline_command: '<user-provided baseline command with {baseline_path}>',
//     skill_memory_path: 'general_memory.txt',
//     mode: 'explore',               // 'explore' or 'adapt'
//     memory_update: false,          // explore defaults read-only
//     iterations: 50,
//     small_step_limit: 2,
//     p_large: 0.2,
//     exploration_weight: 0.3,
//     expand_exploration_ratio: 1.0,
//     reward_alpha: 0.0,             // 0 = avg reward, 1 = max reward
//     diversity_pool_size: 5,
//     pool_size_extra_max: 0,
//     correctness_atol: 0.05,
//     correctness_rtol: 0.05,
//     exp_dir: '/tmp/adaexplore_workflow_exp',
//   }})
//
// Evaluator JSON contract:
//   The evaluator should write JSON at {result_path} with:
//     {
//       "compiled": true,
//       "correct": true,
//       "speedup": 1.23,
//       "kernel_time_ms": 0.12,
//       "baseline_time_ms": 0.15,
//       "error_message": "",
//       "error_type": ""
//     }
//
// =============================================================================

// --- Required Args ---
const OPERATOR_SPEC = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const OP_DESC = args.op_description || 'PyTorch operator'

// --- Optional Args ---
const MODE = args.mode || 'explore'
const BASELINE_CMD = args.baseline_command || ''
const EVAL_CMD = args.benchmark_command || ''
const SKILL_MEMORY_PATH = args.skill_memory_path || ''
const MEMORY_UPDATE = args.memory_update ?? (MODE === 'adapt')
const STEPS = args.iterations || 30
const SMALL_STEP_LIMIT = args.small_step_limit || 2
const P_LARGE = args.p_large ?? 0.2
const EXPLORATION_WEIGHT = args.exploration_weight ?? 0.3
const EXPAND_EXPLORATION_RATIO = args.expand_exploration_ratio ?? 1.0
const EXPAND_EXPLORATION_WEIGHT = EXPLORATION_WEIGHT * EXPAND_EXPLORATION_RATIO
const REWARD_ALPHA = args.reward_alpha ?? 0.0
const DIVERSITY_POOL_SIZE = args.diversity_pool_size || args.pool_size || 5
const POOL_SIZE_EXTRA_MAX = args.pool_size_extra_max ?? 0
const CORRECTNESS_ATOL = args.correctness_atol ?? 0.05
const CORRECTNESS_RTOL = args.correctness_rtol ?? 0.05
const EXP_DIR = args.exp_dir || '/tmp/adaexplore_workflow_exp'
const KERNEL_PATH = args.kernel_path || ''
const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const MAX_MEMORY_RULES = args.max_memory_rules || 40
const EST_PER_ROUND = args.est_tokens_per_round || 60000

// --- Backend driver wiring (P5b Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_SETUP_LANG_TOKEN = 'Triton'
const LEGACY_LARGE_STEP_INTERFACE_LANG = 'PyTorch operator interface'
const LEGACY_REVISER_PERF_HINT = 'Apply a small, local performance or correctness fix based on the evaluator logs.'
const LEGACY_EVALUATE_RUN_INSTRUCTION = 'Run the evaluator command if provided. If not provided, do not create one; mark compiled=false, correct=false, speedup=0, and explain missing evidence.'
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

// --- Project-native integration (embedded kernels via integration-strategist) ---
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const BUILD_CMD = args.build_command || ''
const BENCH_CMD = args.benchmark_command || EVAL_CMD || ''
const REGISTER_SCRIPT = args.register_script || ''

function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}
function workspaceKernelPath(stepIndex, isLarge, ext) {
  return `${EXP_DIR}/kernels/step_${stepIndex + 1}_${isLarge ? 'large' : 'small'}${ext}`
}

let DRIVER = null
let DRIVER_LANG_FENCE = LEGACY_SETUP_LANG_TOKEN.toLowerCase()
let DRIVER_IMPL_REQUIREMENTS = ''
let DRIVER_SOURCE_EXT = '.py'
let DRIVER_BACKEND_ID = RESOLVED_BACKEND || ''

function setupLangToken() {
  return USE_DRIVER ? `${DRIVER_LANG_FENCE} kernel` : `${LEGACY_SETUP_LANG_TOKEN} kernel`
}
function largeStepInterfaceLang() {
  return USE_DRIVER && DRIVER_IMPL_REQUIREMENTS
    ? DRIVER_IMPL_REQUIREMENTS
    : `Preserve the ${LEGACY_LARGE_STEP_INTERFACE_LANG} expected by the evaluator.`
}
function reviserDefaultHint() {
  if (!USE_DRIVER) return LEGACY_REVISER_PERF_HINT
  const m = (DRIVER && DRIVER.methods) || {}
  const guidance = (m.vectorized_load_store && m.vectorized_load_store.prompt_guidance) ||
                   (m.memory_coalescing && m.memory_coalescing.prompt_guidance) ||
                   ''
  return guidance || LEGACY_REVISER_PERF_HINT
}
function evaluateRunInstruction() {
  if (!USE_DRIVER) return LEGACY_EVALUATE_RUN_INSTRUCTION
  return `${driverSh('run.sh', '--artifact {kernel_path} --problem ${PROBLEM_PATH} --out {result_path}')} If the driver run.sh exits non-zero, mark compiled=false, correct=false, speedup=0 and capture stderr.`
}

if (!KERNEL_PATH && !OPERATOR_SPEC && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

// --- Model routing ---
const MODEL = {
  mechanical: args.model_mechanical || 'haiku',  // runs shell/scripts, parses output, bookkeeping
  profile: args.model_profile || 'sonnet',       // profiling / metric analysis
  judgment: args.model_judgment || 'opus',       // planning, code gen/edit/debug, final report
}

// --- State: MCTS Tree ---
let mctsNodes = []
let globalBest = { code: '', speedup: 0, score: [0, 0, 0], id: 'none' }
let skillMemory = [] // [{rule: 'You cannot ...', score: 1}]
let failureLogs = []

function scoreTuple(compiled, correct, speedup) {
  return [compiled ? 1 : 0, correct ? 1 : 0, correct ? (speedup || 0) : 0]
}

function compareScore(a, b) {
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const av = a[i] || 0
    const bv = b[i] || 0
    if (av !== bv) return av - bv
  }
  return 0
}

function computeReward(compiled, correct, speedup) {
  if (!compiled) return 0.0
  if (!correct) return 0.05
  const clipped = Math.max(0.1, Math.min(speedup || 0, 10.0))
  return 0.4 + (clipped - 0.1) * (1.2 / 9.9)
}

function avgReward(node) {
  return node.visits > 0 ? node.totalReward / node.visits : 0
}

function blendedReward(node) {
  return REWARD_ALPHA * (node.maxReward || 0) + (1 - REWARD_ALPHA) * avgReward(node)
}

function ucb1Score(node) {
  if (node.visits === 0) return Infinity
  const parent = node.parentId ? mctsNodes.find(n => n.id === node.parentId) : null
  const exploitation = blendedReward(node)
  if (!parent || parent.visits === 0) return exploitation
  const exploration = EXPLORATION_WEIGHT * Math.sqrt(Math.log(parent.visits) / node.visits)
  return exploitation + exploration
}

function expandUcb1Score(node) {
  if (!node.children.length) return Infinity
  if (EXPAND_EXPLORATION_WEIGHT <= 0) return -Infinity
  const children = node.children.map(cid => mctsNodes.find(n => n.id === cid)).filter(Boolean)
  const qValue = Math.max(...children.map(child => blendedReward(child)))
  const nExpand = children.length
  if (node.visits === 0) return qValue
  return qValue + EXPAND_EXPLORATION_WEIGHT * Math.sqrt(Math.log(node.visits) / (nExpand ** 2))
}

function shouldExpand(node) {
  if (!node.children.length) return true
  const children = node.children.map(cid => mctsNodes.find(n => n.id === cid)).filter(Boolean)
  const bestChildScore = Math.max(...children.map(ucb1Score))
  return expandUcb1Score(node) > bestChildScore
}

function getNode(id) {
  return mctsNodes.find(n => n.id === id)
}

function pathToRoot(node) {
  const path = []
  let current = node
  while (current) {
    path.push(current)
    current = current.parentId ? getNode(current.parentId) : null
  }
  return path.reverse()
}

function smallStepComponent(largeNode) {
  const component = []
  const queue = [largeNode]
  const seen = new Set()
  while (queue.length) {
    const current = queue.shift()
    if (!current || seen.has(current.id)) continue
    seen.add(current.id)
    component.push(current)
    for (const childId of current.children) {
      const child = getNode(childId)
      if (child && child.stepType === 'small') queue.push(child)
    }
  }
  return component
}

function bestCorrectNode(nodes) {
  const correctNodes = nodes.filter(n => n.correct && n.speedup > 0)
  if (!correctNodes.length) return null
  return correctNodes.reduce((best, node) => compareScore(node.score, best.score) > 0 ? node : best, correctNodes[0])
}

function buildDiversePool(selectedNode) {
  const selected = []
  const path = pathToRoot(selectedNode)
  const pathLargeNodes = path.filter(n => n.stepType === 'large')

  for (const largeNode of pathLargeNodes) {
    const best = bestCorrectNode(smallStepComponent(largeNode))
    if (best) selected.push(best)
  }

  if (selected.length < DIVERSITY_POOL_SIZE && POOL_SIZE_EXTRA_MAX > 0) {
    const selectedIds = new Set(selected.map(n => n.id))
    const pathIds = new Set(path.map(n => n.id))
    const extras = []
    for (const node of mctsNodes) {
      if (node.stepType !== 'large' || pathIds.has(node.id)) continue
      const best = bestCorrectNode(smallStepComponent(node))
      if (best && !selectedIds.has(best.id)) extras.push(best)
    }
    extras.sort((a, b) => compareScore(b.score, a.score))
    selected.push(...extras.slice(0, Math.min(POOL_SIZE_EXTRA_MAX, DIVERSITY_POOL_SIZE - selected.length)))
  }

  return selected.slice(0, DIVERSITY_POOL_SIZE)
}

function selectNode() {
  let node = mctsNodes[0]
  while (node.children.length) {
    if (shouldExpand(node)) return node
    const children = node.children.map(cid => getNode(cid)).filter(Boolean)
    node = children.reduce((best, child) => ucb1Score(child) > ucb1Score(best) ? child : best, children[0])
  }
  return node
}

function memoryLines(limit = MAX_MEMORY_RULES) {
  return skillMemory
    .slice()
    .sort((a, b) => (b.score || 0) - (a.score || 0))
    .slice(0, limit)
    .map(item => `${item.rule} || ${item.score || 1}`)
}

function normalizeMemory(rawRules) {
  return (rawRules || [])
    .map(item => {
      if (typeof item === 'string') {
        const [rule, score] = item.split('||')
        return { rule: rule.trim(), score: Number(score || 1) || 1 }
      }
      return { rule: String(item.rule || '').trim(), score: Number(item.score || 1) || 1 }
    })
    .filter(item => item.rule.startsWith('You cannot'))
}

function mergeMemoryRule(rule) {
  if (!rule || !rule.startsWith('You cannot')) return
  const existing = skillMemory.find(item => item.rule === rule)
  if (existing) existing.score = (existing.score || 1) + 1
  else skillMemory.push({ rule, score: 1 })
}

function renderCommand(template, replacements) {
  let command = template || ''
  for (const [key, value] of Object.entries(replacements)) {
    command = command.split(`{${key}}`).join(String(value))
  }
  return command
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
    `Return {present, backend_id, source_ext, lang_fence, impl_requirements, methods}.`,
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

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it in the per-step Evaluate driver envelope. The agent only classifies the
// task (fuzzy); the substrate stamps confidence by method (measured/inferred/
// hypothesized) -- not the agent. See _substrate/profiling/README.md. Falls back
// to native_profiler if undecided. Classifies the operator/task once at Setup;
// candidate kernels inherit the decision for the whole MCTS run. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }
if (USE_DRIVER) {
  const _pd = await agentRetry(() => agent(
    `Read the operator spec for this run (${KERNEL_PATH || PROBLEM_PATH || 'the inline problem_definition'}) and classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${BACKEND_DIR}/manifest.json --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`.\n` +
    `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}

// --- integration-strategist: route build/test mode (standalone vs embedded_*). ---
// Lets AdaExplore optimize inference-engine embedded operators (e.g. llama.cpp .cuh)
// that cannot compile as a single TU, not just standalone kernels. ADDITIVE:
// default stays standalone, so the legacy path is byte-identical.
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _kForIntg = KERNEL_PATH || PROBLEM_PATH || `${EXP_DIR}/reference.py`
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Classify can_compile_standalone for the operator under optimization (${_kForIntg}) as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. a llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${_kForIntg}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
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
  await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// A-O1 closure: embedded operators have no driver-provided native profiler
// (profile.sh targets standalone build artifacts) → downgrade native_profiler to
// perf_heuristic so the embedded eval normalizes throughput instead of probing ncu.
if (IS_EMBEDDED && PROFILING_DECISION.method === 'native_profiler') {
  log(`profiling: native_profiler but embedded operator has no native profiler -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'embedded-perf', rationale: 'native_profiler but embedded operator -> perf_heuristic' }
}

const setupResult = await agentRetry(() => agent(`Set up a standalone AdaExplore-style ${setupLangToken()} optimization run.

# Hard boundary
Do not call the AdaExplore repository or any AdaExplore Python entrypoint. This workflow must run from the operator spec, local files, and the evaluator command provided here.

# Operator source
${KERNEL_PATH ? `Read the operator spec from: ${KERNEL_PATH}` : ''}
${PROBLEM_PATH ? `Read the operator spec from problem_path: ${PROBLEM_PATH}` : ''}
${OPERATOR_SPEC ? `\`\`\`python\n${OPERATOR_SPEC.substring(0, 5000)}\n\`\`\`` : ''}

# Operation
${OP_DESC}

# Experiment directory
Create:
- ${EXP_DIR}/kernels
- ${EXP_DIR}/eval
- ${EXP_DIR}/logs
- ${EXP_DIR}/memory

# Evaluator contract
${EVAL_CMD
  ? `Use this evaluator command template for every candidate:\n${EVAL_CMD}`
  : `No evaluator command was provided. Do not build or infer one. Describe the required evaluator contract and mark measured correctness/performance evidence as unavailable.`}

# Baseline
${BASELINE_CMD
  ? `Run this baseline command template once if needed:\n${BASELINE_CMD}`
  : 'If the evaluator reports speedup directly, no separate baseline command is required. Otherwise mark baseline measurement as unavailable unless a baseline_command is provided.'}

# Skill memory
${SKILL_MEMORY_PATH
  ? `Read existing skill memory from ${SKILL_MEMORY_PATH}. Lines may be "You cannot ... || score".`
  : 'Start with empty skill memory.'}

Return the materialized operator code, evaluator command template, optional baseline time, and loaded memory rules.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"workspace_setup","note":"<operator + evaluator/baseline availability + loaded skill-memory rule count, one line>"}`, {
  label: 'setup',
  phase: 'Setup',
  model: MODEL.mechanical,
  schema: {
    type: 'object',
    properties: {
      operator_code: { type: 'string' },
      evaluator_command: { type: 'string' },
      baseline_time_ms: { type: 'number' },
      hardware_info: { type: 'string' },
      initial_skill_memory: { type: 'array', items: { type: 'string' } },
      reference_path: { type: 'string' },
    },
    required: ['operator_code'],
  },
}), { retries: 5 })

const operatorCode = setupResult.operator_code || OPERATOR_SPEC
const evaluatorCommandTemplate = setupResult.evaluator_command || EVAL_CMD
const baselineTime = setupResult.baseline_time_ms || 0
const referencePath = setupResult.reference_path || `${EXP_DIR}/reference.py`
skillMemory = normalizeMemory(setupResult.initial_skill_memory)

mctsNodes.push({
  id: 'root',
  parentId: null,
  code: '',
  compiled: false,
  correct: false,
  speedup: 0,
  score: [0, 0, 0],
  reward: 0,
  totalReward: 0,
  maxReward: 0,
  visits: 1,
  children: [],
  stepType: 'root',
})

log(`Setup: mode=${MODE} | steps=${STEPS} | memory_update=${MEMORY_UPDATE} | skill_rules=${skillMemory.length}`)

// =============================================================================
// MCTS Search Loop
// =============================================================================

for (let searchStep = 0; searchStep < STEPS; searchStep++) {
  if (typeof budget !== 'undefined' && budget.total && budget.remaining() < EST_PER_ROUND) { log(`token budget ~exhausted — stop`); break }

  phase('Select')

  const selectedNode = selectNode()
  const numSmallChildren = selectedNode.children
    .map(cid => getNode(cid))
    .filter(n => n && n.stepType === 'small')
    .length
  const forceLarge = selectedNode.id === 'root' || numSmallChildren >= SMALL_STEP_LIMIT
  // Use a deterministic pseudo-random based on step index (Math.random() is unavailable in workflow scripts)
  const pseudoRandom = ((searchStep * 2654435761) >>> 0) / 4294967296  // Knuth multiplicative hash
  const randomLarge = pseudoRandom < P_LARGE
  const isLargeStep = forceLarge || randomLarge

  log(`Step ${searchStep + 1}/${STEPS} | selected=${selectedNode.id} | expand=${isLargeStep ? 'large' : 'small'} | nodes=${mctsNodes.length}`)

  phase('Expand')

  let newKernelCode = ''
  let expandNotes = ''

  if (isLargeStep) {
    const diversePool = buildDiversePool(selectedNode)
    const poolContext = diversePool.length
      ? diversePool.map((node, i) => `## Context ${i + 1}: node=${node.id}, speedup=${node.speedup.toFixed(3)}\n\`\`\`python\n${node.code.substring(0, 2200)}\n\`\`\``).join('\n\n')
      : 'No correct diverse context yet.'

    const proposerResult = await agentRetry(() => agent(`You are the AdaExplore Large-Step Proposer.

# Goal
Generate a structurally new ${USE_DRIVER ? `${DRIVER_LANG_FENCE} implementation` : 'Triton implementation'} for the PyTorch operator. This is a broad exploration step, not a local patch.

# PyTorch reference
\`\`\`python
${operatorCode.substring(0, 5000)}
\`\`\`

# Operation
${OP_DESC}

# Hardware
${setupResult.hardware_info || 'NVIDIA GPU'}

# Skill memory constraints
${memoryLines(30).join('\n') || 'No constraints yet.'}

# Diverse context pool
Use these only as inspiration. Avoid copying their structure.
${poolContext}

# Requirements
1. Return complete ${USE_DRIVER ? `${DRIVER_LANG_FENCE} code containing ${DRIVER_BACKEND_ID || 'backend'} kernel(s)` : 'Python code containing Triton kernel(s)'} and a callable wrapper.
2. ${USE_DRIVER && DRIVER_IMPL_REQUIREMENTS ? DRIVER_IMPL_REQUIREMENTS : `Preserve the ${LEGACY_LARGE_STEP_INTERFACE_LANG} expected by the evaluator.`}
3. Respect atol=${CORRECTNESS_ATOL}, rtol=${CORRECTNESS_RTOL}.
4. Prefer structural diversity: different tiling, decomposition, fusion, mapping, or memory strategy from the pool.
5. Do not call any AdaExplore repository code.

Return the complete candidate kernel code.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is large-step proposal for candidate node-${searchStep + 1}-L):
{"workflow":"${WORKFLOW_NAME}","phase":"Expand","ts":"<ts>","status":"done","candidate_id":"node-${searchStep + 1}-L","technique":"<the structural strategy you chose, e.g. tiling/fusion/decomposition>","speedup":null,"note":"<how this candidate differs structurally from the diverse pool>"}`, {
      label: `propose-${searchStep + 1}`,
      phase: 'Expand',
      model: MODEL.judgment,
      schema: {
        type: 'object',
        properties: {
          kernel_code: { type: 'string' },
          strategy: { type: 'string' },
          novelty_vs_pool: { type: 'string' },
        },
        required: ['kernel_code'],
      },
    }), { retries: 5, allowNull: true })

    newKernelCode = proposerResult?.kernel_code || ''
    expandNotes = proposerResult?.strategy || proposerResult?.novelty_vs_pool || ''
  } else {
    const pathContext = pathToRoot(selectedNode)
      .filter(node => node.code)
      .slice(-5)
      .map(node => `## Node ${node.id}: ${node.correct ? 'correct' : 'incorrect'}, speedup=${node.speedup.toFixed(3)}\n\`\`\`python\n${node.code.substring(0, 1800)}\n\`\`\``)
      .join('\n\n')

    const reviserResult = await agentRetry(() => agent(`You are the AdaExplore Reviser.

# Task
Inspect the selected kernel and its recent path. Produce 1-3 concrete local improvement suggestions. Do not rewrite the kernel.

# Selected kernel: node ${selectedNode.id}
\`\`\`python
${selectedNode.code.substring(0, 5000)}
\`\`\`

# Recent path context
${pathContext || 'No prior kernel context.'}

# Selected node metrics
compiled=${selectedNode.compiled}, correct=${selectedNode.correct}, speedup=${selectedNode.speedup}
error=${selectedNode.errorMessage || ''}

# Skill memory constraints
${memoryLines(20).join('\n') || 'No constraints yet.'}

Return specific, surgical suggestions only.`, {
      label: `revise-${searchStep + 1}`,
      phase: 'Expand',
      model: MODEL.judgment,
      schema: {
        type: 'object',
        properties: {
          suggestions: { type: 'array', items: { type: 'string' } },
        },
        required: ['suggestions'],
      },
    }), { retries: 5, allowNull: true })

    const suggestions = reviserResult?.suggestions || [reviserDefaultHint()]

    const tunerResult = await agentRetry(() => agent(`You are the AdaExplore Tuner.

# Task
Apply the reviser suggestions as surgical edits. Preserve the overall structure of the selected kernel. Do not regenerate from scratch.

# Current kernel
\`\`\`python
${selectedNode.code.substring(0, 6000)}
\`\`\`

# Suggestions
${suggestions.map((s, i) => `${i + 1}. ${s}`).join('\n')}

# Skill memory constraints
${memoryLines(20).join('\n') || 'No constraints yet.'}

# Requirements
1. Return complete Python code.
2. Make targeted edits only.
3. Preserve the evaluator-facing interface.
4. Do not call any AdaExplore repository code.

Return the edited candidate kernel code.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is small-step surgical edit for candidate node-${searchStep + 1}-S):
{"workflow":"${WORKFLOW_NAME}","phase":"Expand","ts":"<ts>","status":"done","candidate_id":"node-${searchStep + 1}-S","technique":"<the main local edit you applied>","speedup":null,"note":"<the surgical changes made vs the selected kernel>"}`, {
      label: `tune-${searchStep + 1}`,
      phase: 'Expand',
      model: MODEL.judgment,
      schema: {
        type: 'object',
        properties: {
          kernel_code: { type: 'string' },
          changes_applied: { type: 'array', items: { type: 'string' } },
        },
        required: ['kernel_code'],
      },
    }), { retries: 5, allowNull: true })

    newKernelCode = tunerResult?.kernel_code || selectedNode.code || ''
    expandNotes = (tunerResult?.changes_applied || []).join('; ')
  }

  phase('Evaluate')

  const kernelPath = USE_DRIVER
    ? workspaceKernelPath(searchStep, isLargeStep, DRIVER_SOURCE_EXT)
    : `${EXP_DIR}/kernels/step_${searchStep + 1}_${isLargeStep ? 'large' : 'small'}.py`
  const resultPath = `${EXP_DIR}/eval/step_${searchStep + 1}_result.json`
  const evaluatorCommand = USE_DRIVER
    ? `${SH ? SH + ' ' : ''}${BACKEND_DIR}/run.sh --artifact ${kernelPath} --problem ${PROBLEM_PATH} --out ${resultPath}`
    : renderCommand(evaluatorCommandTemplate, {
    kernel_path: kernelPath,
    result_path: resultPath,
    reference_path: referencePath,
    baseline_path: `${EXP_DIR}/eval/baseline.json`,
    exp_dir: EXP_DIR,
  })

  const evalResult = await agentRetry(() => agent(`Evaluate this candidate with real execution evidence.

# Hard rules
1. Write the candidate code exactly to: ${kernelPath}
2. Do not judge correctness or speedup by inspection.
3. ${evaluateRunInstruction()}
4. The evaluator must compile the ${USE_DRIVER ? DRIVER_LANG_FENCE : 'Triton'} code, compare against PyTorch reference with atol=${CORRECTNESS_ATOL}, rtol=${CORRECTNESS_RTOL}, and measure speed if possible.
5. Write or read JSON result at: ${resultPath}
6. Do not call any AdaExplore repository code.

# Evaluator command
${evaluatorCommand || '(No benchmark_command provided; measured evidence unavailable.)'}

# Candidate code
\`\`\`${USE_DRIVER ? DRIVER_LANG_FENCE : 'python'}
${newKernelCode.substring(0, 9000)}
\`\`\`

# PyTorch reference
\`\`\`python
${operatorCode.substring(0, 5000)}
\`\`\`

Return the parsed evaluation result.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if the candidate compiled AND was correct, else "error"; speedup is the measured speedup number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"node-${searchStep + 1}-${isLargeStep ? 'L' : 'S'}","speedup":<number or null>,"technique":"measured_evaluation","note":"<compiled? correct? speedup; or the evaluator failure reason>"}`, {
    label: `eval-${searchStep + 1}`,
    phase: 'Evaluate',
    model: MODEL.mechanical,
    schema: {
      type: 'object',
      properties: {
        compiled: { type: 'boolean' },
        correct: { type: 'boolean' },
        speedup: { type: 'number' },
        kernel_time_ms: { type: 'number' },
        baseline_time_ms: { type: 'number' },
        error_message: { type: 'string' },
        error_type: { type: 'string' },
        result_path: { type: 'string' },
      },
      required: ['compiled', 'correct', 'speedup'],
    },
  }), { retries: 5 })

  const compiled = Boolean(evalResult.compiled)
  const correct = Boolean(evalResult.correct)
  let speedup = Number(evalResult.speedup || 0)

  let driverEnvelope = null
  if (USE_DRIVER_STANDALONE) {
    const buildOut = `${EXP_DIR}/eval/step_${searchStep + 1}_artifact`
    const profOut = `${EXP_DIR}/eval/step_${searchStep + 1}_prof.native`
    await agentRetry(() => agent(
      `${driverSh('build.sh', `--source ${kernelPath} --out ${buildOut}`)}\n` +
      `Return its stdout JSON verbatim.`,
      { model: MODEL.mechanical, label: `driver-build-${searchStep + 1}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    const runOut = await agentRetry(() => agent(
      `${driverSh('run.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${resultPath}`)}\n` +
      `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
      { model: MODEL.profile, label: `driver-run-${searchStep + 1}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    let evidenceOut = null
    if (PROFILING_DECISION.method === 'native_profiler') {
      const profileOut = await agentRetry(() => agent(
        `${driverSh('profile.sh', `--artifact ${buildOut} --kernel ${kernelPath} --out ${profOut}`)}\n` +
        `Return {ok, native_path}.`,
        { model: MODEL.profile, label: `driver-profile-${searchStep + 1}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      evidenceOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
        `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
        { model: MODEL.mechanical, label: `driver-to-evidence-${searchStep + 1}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    } else {
      // Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run profile.sh / a native profiler.
      // run.sh above already produced throughput. When method='perf_heuristic', normalize that throughput into canonical metrics via the strategist normalizer.
      if (PROFILING_DECISION.method === 'perf_heuristic') {
        evidenceOut = await agentRetry(() => agent(
          `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${PROFILING_DECISION.normalizer || 'perf_to_evidence.py'} --baseline ${resultPath}\`.\n` +
          `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend, heuristic_bclass}. ` +
          `Also write heuristic_bclass (memory_bound|compute_bound|latency_bound) based on the throughput ratio so diagnose.py does not fall to unknown. ` +
          `Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.`,
          { model: MODEL.mechanical, label: `driver-to-evidence-${searchStep + 1}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      }
    }
    const diagOut = await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
      `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
      { model: MODEL.mechanical, label: `driver-diagnose-${searchStep + 1}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    const antiCheatOut = await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kernelPath} --result ${resultPath}\`.\n` +
      `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
      { model: MODEL.mechanical, label: `driver-anti-cheat-${searchStep + 1}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    const measuredLatency = Number((runOut && runOut.latency_ms) || (evidenceOut && evidenceOut.metrics && evidenceOut.metrics.latency_ms) || 0)
    const baselineLatency = Number(args.baseline_latency_ms || 0)
    const driverSpeedup = (baselineLatency > 0 && measuredLatency > 0) ? baselineLatency / measuredLatency : 0
    if (driverSpeedup > 0) speedup = driverSpeedup
    driverEnvelope = {
      anti_cheat: antiCheatOut || {},
      metrics: (evidenceOut && evidenceOut.metrics) || {},
      vendor: DRIVER && DRIVER.hw_vendor || '',
      coverage: (evidenceOut && evidenceOut.coverage) || [],
      bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
      latency_ms: measuredLatency,
      baseline_latency_ms: baselineLatency,
      backend_id: DRIVER_BACKEND_ID,
      profiling_method: PROFILING_DECISION.method,
      profiling_confidence: PROFILING_DECISION.confidence,
    }
  } else if (IS_EMBEDDED) {
    // --- Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch) ---
    // SERIAL by construction: the MCTS loop evaluates exactly ONE candidate per
    // searchStep (no `await parallel(` over candidates), so embedded modes never run
    // concurrently — no race on the shared KERNEL_PATH (inplace) or project build (dispatch).
    const variant = `adaexplore_${searchStep + 1}_${isLargeStep ? 'L' : 'S'}`.replace(/[^A-Za-z0-9_]/g, '_')
    let embLatency = 0, embMetrics = {}, embBclass = 'unknown'
    if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
      const embResult = await agentRetry(() => agent(
        `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kernelPath} | project kernel: ${KERNEL_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
        `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
        `2. Apply candidate: cp ${kernelPath} ${KERNEL_PATH}\n3. Build: ${BUILD_CMD}\n4. Test: ${TEST_CMD}\n5. Benchmark: ${BENCH_CMD || TEST_CMD}\n` +
        `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
        `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
        { model: MODEL.mechanical, label: `embedded-inplace-${searchStep + 1}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      embLatency = Number(embResult?.latency_ms || 0)
      embBclass = embResult?.heuristic_bclass || 'unknown'
      embMetrics = embResult?.metrics || { latency_ms: embLatency }
    } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
      const _plan = typeof __embeddedEvalPlan === 'function'
        ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kernelPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: TEST_CMD, benchmarkCmd: BENCH_CMD || TEST_CMD })
        : null
      if (_plan) {
        const embResult = await agentRetry(() => agent(
          `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
          `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-dispatch-${searchStep + 1}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        embLatency = Number(embResult?.latency_ms || 0)
        embBclass = embResult?.heuristic_bclass || 'unknown'
        embMetrics = embResult?.metrics || { latency_ms: embLatency }
      }
    }
    const embBaselineLatency = Number(args.baseline_latency_ms || 0)
    const embSpeedup = (embBaselineLatency > 0 && embLatency > 0) ? embBaselineLatency / embLatency : 0
    if (embSpeedup > 0) speedup = embSpeedup
    driverEnvelope = {
      metrics: embMetrics,
      bottleneck_class: embBclass,
      latency_ms: embLatency,
      baseline_latency_ms: embBaselineLatency,
      backend_id: 'embedded',
      profiling_method: PROFILING_DECISION.method,
      profiling_confidence: PROFILING_DECISION.confidence,
    }
  }

  const reward = computeReward(compiled, correct, speedup)
  const newScore = scoreTuple(compiled, correct, speedup)
  const newNodeId = `node-${searchStep + 1}-${isLargeStep ? 'L' : 'S'}`
  const newNode = {
    id: newNodeId,
    parentId: selectedNode.id,
    code: newKernelCode,
    compiled,
    correct,
    speedup,
    score: newScore,
    reward,
    totalReward: reward,
    maxReward: reward,
    visits: 0,
    children: [],
    stepType: isLargeStep ? 'large' : 'small',
    errorMessage: evalResult.error_message || '',
    errorType: evalResult.error_type || '',
    kernelPath,
    resultPath: evalResult.result_path || resultPath,
    notes: expandNotes,
    ...(driverEnvelope ? { driver_envelope: driverEnvelope } : {}),
  }
  mctsNodes.push(newNode)
  selectedNode.children.push(newNodeId)

  if (compareScore(newScore, globalBest.score) > 0) {
    globalBest = { code: newKernelCode, speedup, score: newScore, id: newNodeId }
    log(`  NEW BEST: node=${newNodeId} score=${JSON.stringify(newScore)} speedup=${speedup.toFixed(3)}x`)
  }

  phase('Backpropagate')

  let current = newNode
  while (current) {
    current.visits += 1
    current.totalReward += reward
    current.maxReward = Math.max(current.maxReward || 0, reward)
    current = current.parentId ? getNode(current.parentId) : null
  }

  if (!compiled || !correct) {
    failureLogs.push({
      node_id: newNodeId,
      kernel_excerpt: newKernelCode.substring(0, 2500),
      error_type: evalResult.error_type || (compiled ? 'correctness' : 'compile'),
      error_message: evalResult.error_message || 'unknown evaluator failure',
      result_path: evalResult.result_path || resultPath,
    })
  }

  log(`  Step ${searchStep + 1}: compiled=${compiled} correct=${correct} speedup=${speedup.toFixed(3)} reward=${reward.toFixed(3)} nodes=${mctsNodes.length} best=${globalBest.id}`)
}

// =============================================================================
// Phase 6: Optional memory adaptation
// =============================================================================
phase('AdaptMemory')

let memoryUpdateReport = { updated: false, new_rules: 0, total_rules: skillMemory.length }

if (MEMORY_UPDATE && failureLogs.length) {
  const memoryResult = await agentRetry(() => agent(`Update standalone AdaExplore skill memory from evaluated failure logs.

# Hard rules
1. Only use the provided failure logs and evaluator messages.
2. Extract minimal one-line constraints.
3. Each rule must start with "You cannot".
4. If the failure does not imply a precise prohibited action, skip it.
5. Deduplicate against existing memory. If duplicate, increment score.
6. If skill_memory_path is provided, write updated memory back as "rule || score" lines.

# Existing memory
${memoryLines(MAX_MEMORY_RULES).join('\n') || 'No existing rules.'}

# skill_memory_path
${SKILL_MEMORY_PATH || '(no path provided; return memory in workflow output only)'}

# Failure logs
\`\`\`json
${JSON.stringify(failureLogs, null, 2).substring(0, 12000)}
\`\`\`

Return updated memory rules.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"AdaptMemory","ts":"<ts>","status":"done","technique":"failure_to_constraint_distillation","note":"<how many new You-cannot rules distilled from failure logs, one line>"}`, {
    label: 'adapt-memory',
    phase: 'AdaptMemory',
    model: MODEL.mechanical,
    schema: {
      type: 'object',
      properties: {
        rules: { type: 'array', items: { type: 'string' } },
        wrote_path: { type: 'string' },
      },
      required: ['rules'],
    },
  }), { retries: 5, allowNull: true })

  const before = skillMemory.length
  for (const rule of memoryResult?.rules || []) {
    const normalized = String(rule).split('||')[0].trim()
    mergeMemoryRule(normalized)
  }
  memoryUpdateReport = {
    updated: true,
    new_rules: Math.max(0, skillMemory.length - before),
    total_rules: skillMemory.length,
    wrote_path: memoryResult?.wrote_path || SKILL_MEMORY_PATH || '',
  }
}

// =============================================================================
// Phase 7: Report
// =============================================================================
phase('Report')

const treeStats = {
  total_nodes: mctsNodes.length,
  large_steps: mctsNodes.filter(n => n.stepType === 'large').length,
  small_steps: mctsNodes.filter(n => n.stepType === 'small').length,
  compiled_count: mctsNodes.filter(n => n.compiled).length,
  correct_count: mctsNodes.filter(n => n.correct).length,
  best_node_id: globalBest.id,
  best_score: globalBest.score,
  best_speedup: globalBest.speedup,
}

const finalReport = await agentRetry(() => agent(`Write a concise technical report for this standalone AdaExplore-style optimization run.

# Operation
${OP_DESC}

# Boundary
This workflow did not call the AdaExplore repository entrypoint. It implemented the method directly through agent generation, evaluator execution, MCTS bookkeeping, and optional memory adaptation.

# Tree stats
\`\`\`json
${JSON.stringify(treeStats, null, 2)}
\`\`\`

# Best kernel excerpt
\`\`\`python
${globalBest.code.substring(0, 5000)}
\`\`\`

# Memory update
\`\`\`json
${JSON.stringify(memoryUpdateReport, null, 2)}
\`\`\`

# Top skill memory
${memoryLines(15).join('\n') || 'No skill memory rules.'}

Cover:
1. Which expansion mode was most useful.
2. Whether the evaluator evidence is strong enough to trust the best kernel.
3. What failure patterns should guide the next run.
4. Any remaining optimization opportunities.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the best measured result (speedup is the best measured speedup number, or null if none verified):
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","candidate_id":"${globalBest.id}","speedup":<number or null>,"technique":"final_report","note":"<which expansion mode won + whether evaluator evidence supports the best kernel, one line>"}`, {
  label: 'final-report',
  phase: 'Report',
  model: MODEL.judgment,
}), { retries: 5 })

// --- exit restore (embedded_inplace safety net) ---
// The per-step inplace eval ALWAYS restores after each candidate, but on any abnormal
// exit the project tree could be left dirty. Restore pristine unconditionally before
// returning so the embedded_inplace path never leaves the original kernel mutated.
if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit-restore: run \`cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\` and confirm the original kernel is restored byte-exact.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  input_mode: INPUT_MODE,
  problem_definition: OPERATOR_SPEC,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: globalBest.id !== 'none' ? `${EXP_DIR}/${globalBest.id}.py` : '',
  initial_candidates: mctsNodes.filter(n => n.parent_id == null),
  initial_generation_result: {
    verified: globalBest.speedup > 0,
    selected_candidate_id: globalBest.id || '',
  },
  operation: OP_DESC,
  mode: MODE,
  memory_update: MEMORY_UPDATE,
  baseline_time_ms: baselineTime,
  best_speedup: globalBest.speedup,
  best_score: globalBest.score,
  best_kernel_code: globalBest.code,
  best_node_id: globalBest.id,
  mcts_steps: STEPS,
  tree_stats: treeStats,
  failure_count: failureLogs.length,
  skill_memory: memoryLines(MAX_MEMORY_RULES),
  skill_memory_size: skillMemory.length,
  memory_update_report: memoryUpdateReport,
  report: finalReport,
}
