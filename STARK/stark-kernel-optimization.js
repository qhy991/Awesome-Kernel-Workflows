export const meta = {
  name: 'stark-kernel-optimization',
  description: 'Strategic Team of Agents for Refining Kernels: multi-agent collaboration with tree-search, grounded instruction, and dynamic context windows (STARK methodology)',
  whenToUse: 'When you need to optimize GPU kernels through a structured multi-agent pipeline that separates planning, coding, and debugging, with strategic tree search and grounded instruction anchoring. Effective for KernelBench-style tasks where monolithic agents get trapped in local optima.',
  phases: [
    { title: 'Setup', detail: 'Read reference kernel, initialize search tree and leaderboard' },
    { title: 'Select', detail: 'ε-greedy node selection from tree memory with root throttling and dead-branch pruning' },
    { title: 'Plan', detail: 'Plan agent proposes optimization with grounded instruction anchors' },
    { title: 'Code', detail: 'Code agent realizes grounded instructions into executable kernel' },
    { title: 'Debug', detail: 'Debug agent repairs failing kernels using sibling context' },
    { title: 'Evaluate', detail: 'Compile, correctness test, and runtime profiling' },
    { title: 'Update', detail: 'Append child to tree, update leaderboard' },
    { title: 'Report', detail: 'Return best kernel from leaderboard with full trajectory' },
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

const WORKFLOW_NAME = 'stark-kernel-optimization'


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

// --- BEGIN typed-args (channel ② experience_excerpts) ---
// Cross-session priors travel here as a typed array (see KerSor
// agents/dispatch-arg-synthesizer.md), independent of op_description so the
// solver can treat them as distinct lower-authority signals.
const EXPERIENCE_EXCERPTS = Array.isArray(args.experience_excerpts) ? args.experience_excerpts : []
function __experienceBlock() {
  if (!EXPERIENCE_EXCERPTS.length) return ''
  const lines = EXPERIENCE_EXCERPTS.map(e => {
    const kind = (e && e.kind) || 'note'
    const directive = (e && e.directive) || 'inform'
    const claim = (e && e.claim) || (typeof e === 'string' ? e : JSON.stringify(e))
    return `- [${kind}/${directive}] ${claim}`
  })
  return `\n# Cross-session experience excerpts (channel ② — priors from past sessions; LOWER authority than current-round evidence):\n${lines.join('\n')}\n`
}

// Channel ③: typed prior-attempt context (attempt_evidence + attempt_plan).
// KerSor's dispatch-arg-synthesizer reads run-{N-1}/analysis.json and
// round-{N}-selection.json and emits both as typed JSON objects on args.
// Solvers consume them as a HIGHER-authority signal than HANDOFF prose.
const ATTEMPT_EVIDENCE = (args.attempt_evidence && typeof args.attempt_evidence === 'object') ? args.attempt_evidence : null
const ATTEMPT_PLAN = (args.attempt_plan && typeof args.attempt_plan === 'object') ? args.attempt_plan : null
const FAILED_STRATEGY_IDS = (ATTEMPT_EVIDENCE && Array.isArray(ATTEMPT_EVIDENCE.transfer_items))
  ? ATTEMPT_EVIDENCE.transfer_items.filter(i => i && i.kind === 'failed_strategy' && i.id).map(i => i.id)
  : []
function __attemptBlock() {
  if (!ATTEMPT_EVIDENCE && !ATTEMPT_PLAN) return ''
  const parts = ['\n# Prior attempt context (channel ③ — TYPED, machine-verified; HIGHER authority than handoff prose):']
  if (FAILED_STRATEGY_IDS.length > 0) {
    parts.push(`## HARD CONSTRAINT — do NOT re-propose any of these failed-strategy ids: ${FAILED_STRATEGY_IDS.join(', ')}`)
  }
  if (ATTEMPT_EVIDENCE) {
    const j = JSON.stringify(ATTEMPT_EVIDENCE, null, 2)
    parts.push('## Prior attempt evidence (last round):\n```json\n' + (j.length > 4000 ? j.slice(0, 4000) + '\n... [truncated to 4000 chars]' : j) + '\n```')
  }
  if (ATTEMPT_PLAN && Array.isArray(ATTEMPT_PLAN.candidate_plans)) {
    parts.push('## Routing-suggested candidate plans:\n```json\n' + JSON.stringify({phase_intent: ATTEMPT_PLAN.phase_intent, candidate_plans: ATTEMPT_PLAN.candidate_plans}, null, 2) + '\n```')
  }
  return parts.join('\n') + '\n'
}
// --- END typed-args ---
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
// STARK — Strategic Team of Agents for Refining Kernels Workflow
// =============================================================================
//
// Implements the STARK paper (arXiv:2510.16996, Meta/Duke 2025):
//   Multi-Agent Collaboration + Grounded Instruction + Dynamic Context Windows
//   + Strategic Tree Search (ε-greedy with domain-specific adaptations)
//
// Key innovations addressed:
//   1. Monolithic agent design → Plan / Code / Debug role separation
//   2. Naive exploration → ε-greedy over tree memory with root throttling
//   3. Planning-implementation gap → Grounded instruction with span anchors
//   4. Context blindness → Agent-specific dynamic context windows
//
// Usage:
//   Workflow({name: 'stark-kernel-optimization', args: {
//     kernel_path: '/path/to/reference.cu',
//     test_harness_path: '/path/to/test.py',
//     iterations: 30,
//     epsilon: 0.35,
//     n_root: 5,
//     n_child_max: 3,
//     leaderboard_size: 5,
//     plan_model: 'claude-sonnet-4-20250514',
//     code_model: 'claude-sonnet-4-20250514',
//     debug_model: 'claude-sonnet-4-20250514',
//     plan_temperature: 0.8,
//     code_temperature: 0.1,
//     debug_temperature: 0.1,
//     compile_command: '<user-provided compile command with {kernel_path}/{result_path}>',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     exp_dir: '/tmp/stark_exp',
//   }})
//
// =============================================================================

// --- Required Args ---
let REF_KERNEL_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = REF_KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const TEST_HARNESS_PATH = args.test_harness_path

// --- Optional Args ---
const BUDGET = args.iterations || 30
const EPSILON = args.epsilon !== undefined ? args.epsilon : 0.35
const N_ROOT = args.n_root || 5
const N_CHILD_MAX = args.n_child_max || 3
const LEADERBOARD_SIZE = args.leaderboard_size || 5
const PLAN_MODEL = args.plan_model || 'claude-sonnet-4-20250514'
const CODE_MODEL = args.code_model || 'claude-sonnet-4-20250514'
const DEBUG_MODEL = args.debug_model || 'claude-sonnet-4-20250514'
const PLAN_TEMP = args.plan_temperature !== undefined ? args.plan_temperature : 0.8
const CODE_TEMP = args.code_temperature !== undefined ? args.code_temperature : 0.1
const DEBUG_TEMP = args.debug_temperature !== undefined ? args.debug_temperature : 0.1
const COMPILE_CMD = args.compile_command || ''
const BENCHMARK_CMD = args.benchmark_command || ''
const EXP_DIR = args.exp_dir || '/tmp/stark_exp'
const LANGUAGE = args.language || 'cuda'
const TARGET_GPU = args.target_gpu || 'unknown GPU'
const SEED_CANDIDATES = args.seed_candidates || 3

// Seeded RNG (P5c STARK A0). Deterministic xorshift32, used in selectNode().
// The Workflow runtime FORBIDS the host language's nondeterministic float RNG
// (non-determinism breaks resume — cached branches diverge; the KerSor catalog
// forbidden-API scan marks any workflow containing that token known_broken —
// AWK #50). So rng() is ALWAYS seeded: args.rng_seed when pinned, else a fixed
// deterministic seed (1). Callers that want per-run exploration variety must
// opt in by passing rng_seed.
const RNG_SEED = (args.rng_seed !== undefined && args.rng_seed !== null) ? args.rng_seed : 1
let _rngState = ((RNG_SEED | 0) || 1) >>> 0
function rng() {
  let s = _rngState | 0
  s ^= s << 13
  s ^= s >>> 17
  s ^= s << 5
  _rngState = s >>> 0
  return (_rngState / 0x1_0000_0000)
}

if (!REF_KERNEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

// --- Backend driver wiring (P5c Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_GENERATION_LANG_TOKEN = 'CUDA'
const LEGACY_SETUP_LANG_TOKEN = 'CUDA'
const LEGACY_DEBUG_LANG_TOKEN = 'CUDA'
const LEGACY_PLAN_LANG_TOKEN = 'CUDA'
const LEGACY_CODE_LANG_TOKEN = 'CUDA'
const LEGACY_EVAL_LANG_TOKEN = 'CUDA'
const LEGACY_SOURCE_EXT = '.cu'
const LEGACY_FENCE_TOKEN = 'cuda'
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

// --- Project-native integration (embedded kernels via integration-strategist) ---
// STARK already declares BENCHMARK_CMD (benchmark_command) for its own eval prompts;
// reuse it as the project bench command (avoid a duplicate const that breaks wfcheck).
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const BUILD_CMD = args.build_command || ''
const PROJECT_BENCH_CMD = args.project_benchmark_command || BENCHMARK_CMD || ''
const REGISTER_SCRIPT = args.register_script || ''

function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}

let DRIVER = null
let DRIVER_LANG_FENCE = LEGACY_FENCE_TOKEN
let DRIVER_IMPL_REQUIREMENTS = ''
let DRIVER_SOURCE_EXT = LEGACY_SOURCE_EXT
let DRIVER_BACKEND_ID = RESOLVED_BACKEND || ''

function langToken(legacy) {
  return USE_DRIVER ? DRIVER_LANG_FENCE : legacy
}
function fenceToken() {
  return USE_DRIVER ? DRIVER_LANG_FENCE : LEGACY_FENCE_TOKEN
}
// Hybrid kernel-path helper (KDA/Astra-style): AccelOpt-pattern for the
// user-supplied REF_KERNEL_PATH; KernelAgent-workspace-local for per-node
// candidate kernels written under EXP_DIR.
function starkGeneratedKernelPath() {
  const ext = USE_DRIVER ? DRIVER_SOURCE_EXT : LEGACY_SOURCE_EXT
  return `${EXP_DIR}/generated/kernel${ext}`
}
function starkNodeKernelPath(nodeId) {
  const ext = USE_DRIVER ? DRIVER_SOURCE_EXT : LEGACY_SOURCE_EXT
  return `${EXP_DIR}/${nodeId}${ext}`
}

// --- State ---
let tree = []                    // Array of nodes: {id, parent_id, kernel_code, plan, anchors, runtime, correct, compile_ok, logs, children, status}
let leaderboard = []             // Top-r nodes by runtime (lower = better)
let rootNode = null              // The reference kernel (node 0)
let attemptCount = 0
let bestKernel = null
let bestRuntime = null
let referenceKernelCode = ''
let referenceDescription = ''
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null

// =============================================================================
// Helper: Node management
// =============================================================================

function getNode(id) {
  return tree.find(n => n.id === id)
}

function getChildren(nodeId) {
  return tree.filter(n => n.parent_id === nodeId)
}

function getSiblings(nodeId) {
  const node = getNode(nodeId)
  if (!node || node.parent_id === null) return []
  return tree.filter(n => n.parent_id === node.parent_id && n.id !== nodeId)
}

function isLeaf(nodeId) {
  return getChildren(nodeId).length === 0
}

function isExpandable(nodeId) {
  const node = getNode(nodeId)
  if (!node) return false
  // Root throttling: root cannot be selected if it already has n_root children
  if (node.id === 'root' && getChildren('root').length >= N_ROOT) return false
  // Dead-branch pruning: if node has > n_child_max children and all fail, mark dead
  const children = getChildren(nodeId)
  if (children.length > N_CHILD_MAX) {
    const allFailed = children.every(c => c.status === 'failed')
    if (allFailed) return false
  }
  return true
}

function getScore(node) {
  // Lower runtime = better. Failed/crashed nodes get +inf.
  if (!node.compile_ok) return Infinity
  if (!node.correct) return Infinity
  if (node.runtime === null || node.runtime === undefined) return Infinity
  return node.runtime
}

// =============================================================================
// Helper: ε-greedy selection with leaf-biased exploration
// =============================================================================

function selectNode() {
  // Collect all expandable nodes
  const expandable = tree.filter(n => isExpandable(n.id))
  if (expandable.length === 0) {
    // Fallback: select any node that hasn't been expanded too much
    const fallback = tree.filter(n => {
      const c = getChildren(n.id)
      return c.length < N_CHILD_MAX * 2
    })
    if (fallback.length > 0) return fallback[Math.floor(rng() * fallback.length)]
    return tree[Math.floor(rng() * tree.length)]
  }

  // With probability (1 - epsilon), pick the best node by score (exploitation)
  const coin = rng()
  if (coin > EPSILON) {
    // Exploitation: pick the best-scoring expandable node
    const sorted = [...expandable].sort((a, b) => getScore(a) - getScore(b))
    return sorted[0]
  }

  // With probability epsilon, explore
  // Leaf-biased: sample uniformly from expandable leaves
  const leaves = expandable.filter(n => isLeaf(n.id))
  if (leaves.length > 0) {
    return leaves[Math.floor(rng() * leaves.length)]
  }
  // Fallback to all expandable
  return expandable[Math.floor(rng() * expandable.length)]
}

// =============================================================================
// Helper: Build agent-specific context windows (Section 4.4)
// =============================================================================

function buildPlanContext(nodeId, langFence) {
  const node = getNode(nodeId)
  if (!node) return ''
  const children = getChildren(nodeId)
  const topR = leaderboard.slice(0, LEADERBOARD_SIZE).filter(n => n.id !== nodeId)
  const fence = langFence || LEGACY_FENCE_TOKEN

  let ctx = `# Context Window for PLAN Agent\n\n## Selected Node (id=${nodeId})\n`
  ctx += `- Runtime: ${node.runtime !== null ? node.runtime + ' ms' : 'N/A'}\n`
  ctx += `- Correct: ${node.correct}\n`
  ctx += `- Compile OK: ${node.compile_ok}\n`

  ctx += `\n## Children of Selected Node (${children.length})\n`
  for (const c of children) {
    ctx += `- ${c.id}: runtime=${c.runtime !== null ? c.runtime + 'ms' : 'N/A'}, correct=${c.correct}, compile=${c.compile_ok}\n`
    if (c.plan) {
      ctx += `  Plan: ${c.plan.substring(0, 200)}...\n`
    }
  }

  ctx += `\n## Global Leaderboard (Top-${LEADERBOARD_SIZE})\n`
  for (let i = 0; i < topR.length; i++) {
    const l = topR[i]
    ctx += `${i + 1}. ${l.id}: ${l.runtime}ms\n`
  }

  ctx += `\n## Source Reference Kernel\n\`\`\`${fence}\n${referenceKernelCode.substring(0, 3000)}\n\`\`\`\n`

  return ctx
}

function buildCodeContext(nodeId, langFence) {
  const node = getNode(nodeId)
  if (!node) return ''
  const children = getChildren(nodeId)
  const siblings = getSiblings(nodeId)
  const fence = langFence || LEGACY_FENCE_TOKEN

  let ctx = `# Context Window for CODE Agent\n\n## Selected Node (id=${nodeId})\n`
  ctx += `Selected kernel code:\n\`\`\`${fence}\n${node.kernel_code.substring(0, 2500)}\n\`\`\`\n`

  ctx += `\n## Children of Selected Node (${children.length})\n`
  for (const c of children) {
    if (c.correct && c.runtime !== null) {
      ctx += `- ${c.id}: SUCCESS, ${c.runtime}ms. Code snippet:\n\`\`\`${fence}\n${c.kernel_code.substring(0, 800)}\n\`\`\`\n`
    } else if (!c.compile_ok || !c.correct) {
      ctx += `- ${c.id}: FAILED — ${c.logs?.substring(0, 200) || 'unknown error'}\n`
    }
  }

  ctx += `\n## Sibling Nodes (${siblings.length}) — Transferable patches\n`
  for (const s of siblings) {
    if (s.correct && s.runtime !== null) {
      ctx += `- ${s.id}: ${s.runtime}ms. Key implementation:\n\`\`\`${fence}\n${s.kernel_code.substring(0, 600)}\n\`\`\`\n`
    }
  }

  ctx += `\n## Source Reference Kernel\n\`\`\`${fence}\n${referenceKernelCode.substring(0, 1500)}\n\`\`\`\n`

  return ctx
}

function buildDebugContext(nodeId, langFence) {
  const node = getNode(nodeId)
  if (!node) return ''
  const siblings = getSiblings(nodeId)
  const fence = langFence || LEGACY_FENCE_TOKEN

  let ctx = `# Context Window for DEBUG Agent\n\n## Failing Node (id=${nodeId})\n`
  ctx += `Failing kernel code:\n\`\`\`${fence}\n${node.kernel_code.substring(0, 2500)}\n\`\`\`\n`
  ctx += `\n## Error Logs\n\`\`\`\n${(node.logs || '').substring(0, 2000)}\n\`\`\`\n`
  ctx += `\n## Original Plan (if any)\n${node.plan || 'No plan recorded'}\n`
  ctx += `\n## Anchors (if any)\n${node.anchors || 'No anchors recorded'}\n`

  ctx += `\n## Sibling Kernels (${siblings.length}) — Local fix patterns\n`
  for (const s of siblings) {
    ctx += `### ${s.id}\n`
    if (s.correct) {
      ctx += `CORRECT — ${s.runtime}ms:\n\`\`\`${fence}\n${s.kernel_code.substring(0, 800)}\n\`\`\`\n`
    } else {
      ctx += `Also failing — ${s.logs?.substring(0, 200) || 'unknown'}\n`
    }
  }

  ctx += `\n## Source Reference Kernel\n\`\`\`${fence}\n${referenceKernelCode.substring(0, 1500)}\n\`\`\`\n`

  return ctx
}

// =============================================================================
// Helper: Update leaderboard
// =============================================================================

function updateLeaderboard() {
  const candidates = tree.filter(n => n.correct && n.runtime !== null && n.runtime !== undefined)
  candidates.sort((a, b) => a.runtime - b.runtime)
  leaderboard = candidates.slice(0, LEADERBOARD_SIZE)
}

// =============================================================================
// Phase 1: Setup — Read reference kernel, init tree
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
  const generated = await agentRetry(() => agent(`No kernel_path was provided. Generate and verify an initial ${langToken(LEGACY_GENERATION_LANG_TOKEN)} kernel before constructing the STARK search tree.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- language: ${LANGUAGE}
- target_gpu: ${TARGET_GPU}
- seed_candidates: ${SEED_CANDIDATES}
- test_harness_path: ${TEST_HARNESS_PATH || '(not provided)'}

# Evidence Commands
- compile_command: ${COMPILE_CMD || '(not provided)'}
- benchmark_command: ${BENCHMARK_CMD || '(not provided)'}

# Contract
Generate ${SEED_CANDIDATES} complete candidates under ${EXP_DIR}/generated/. Run available commands using {kernel_path}/{result_path}. Return the best verified generated kernel path as the root/reference node.`, {
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
  if (BENCHMARK_CMD && initialGenerationResult.verified === false) throw new Error('No generated seed passed benchmark evidence')
  REF_KERNEL_PATH = generatedKernelPath
}

const setupResult = await agentRetry(() => agent(`Read the reference kernel file at: ${REF_KERNEL_PATH}

Analyze it and return:
1. Full kernel source code
2. Operation type and algorithm description
3. Launch configuration (grid/block dimensions if visible)
4. Current optimization level assessment
5. Potential optimization directions

Return ONLY a JSON object with:
- kernel_code: string (full source)
- op_type: string
- algorithm_description: string
- launch_config: string
- optimization_assessment: string
- potential_directions: array of strings

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"read_reference_kernel","note":"<op_type + optimization assessment + top potential direction, one line>"}`, {
  label: 'setup-read-reference',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      kernel_code: { type: 'string' },
      op_type: { type: 'string' },
      algorithm_description: { type: 'string' },
      launch_config: { type: 'string' },
      optimization_assessment: { type: 'string' },
      potential_directions: { type: 'array', items: { type: 'string' } },
    },
    required: ['kernel_code', 'op_type', 'algorithm_description'],
  },
}), { retries: 5 })

referenceKernelCode = setupResult.kernel_code
referenceDescription = setupResult.algorithm_description
log(`Reference loaded: ${setupResult.op_type}, ${referenceKernelCode.length} chars`)

// Evaluate reference kernel as root node
const rootEval = await agentRetry(() => agent(`Evaluate this reference kernel for correctness and performance.

# Kernel Code
\`\`\`${fenceToken()}
${referenceKernelCode.substring(0, 4000)}
\`\`\`

# Compile and Run
${COMPILE_CMD ? `Compile: ${COMPILE_CMD}` : 'No compile_command provided; perform static compileability review only.'}
${BENCHMARK_CMD ? `Benchmark: ${BENCHMARK_CMD}` : 'No benchmark_command provided; do not invent a test harness.'}

Return JSON with:
- compile_ok: boolean
- correct: boolean (passes test)
- runtime_ms: number (or null if failed)
- logs: string (compiler/test output)`, {
  label: 'setup-eval-root',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      compile_ok: { type: 'boolean' },
      correct: { type: 'boolean' },
      runtime_ms: { type: 'number' },
      logs: { type: 'string' },
    },
    required: ['compile_ok', 'correct', 'runtime_ms', 'logs'],
  },
}), { retries: 5 })

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it below. The agent only classifies the task (fuzzy); the substrate stamps
// confidence by method (measured/inferred/hypothesized) -- not the agent. See
// _substrate/profiling/README.md. Falls back to native_profiler if undecided. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }
if (USE_DRIVER) {
  const _pd = await agentRetry(() => agent(
    `Read ${REF_KERNEL_PATH}; classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    `run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${BACKEND_DIR}/manifest.json --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`.\n` +
    `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}

// --- integration-strategist: route build/test mode (standalone vs embedded_*). ---
// Lets STARK handle inference-engine embedded operators (e.g. llama.cpp .cuh) that
// cannot compile as a single TU, not just standalone kernels. Default = standalone
// (legacy path byte-identical when no project_root/build/register args are passed).
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _profManifest = (USE_DRIVER && BACKEND_DIR) ? `${BACKEND_DIR}/manifest.json` : `${SUBSTRATE}/backends/cuda/manifest.json`
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Read ${REF_KERNEL_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${REF_KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
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
  await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${REF_KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// A-O1 closure: native_profiler but no standalone driver path (embedded/legacy) → perf_heuristic
if (PROFILING_DECISION.method === 'native_profiler' && !USE_DRIVER_STANDALONE) {
  log(`profiling: native_profiler but no standalone driver path -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'test-harness-perf', rationale: 'native_profiler but no standalone driver -> perf_heuristic' }
}

if (USE_DRIVER_STANDALONE) {
  const kPath = REF_KERNEL_PATH
  const buildOut = `${EXP_DIR}/stark_root.artifact`
  const profOut = `${EXP_DIR}/stark_root.prof.native`
  await agentRetry(() => agent(
    `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
    `Return its stdout JSON verbatim.`,
    { model: MODEL.mechanical, label: 'driver-build-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  const runOut = await agentRetry(() => agent(
    `${driverSh('run.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${buildOut}.run.json`)}\n` +
    `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
    { model: MODEL.profile, label: 'driver-run-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  let evidenceOut = null
  if (PROFILING_DECISION.method === 'native_profiler') {
    await agentRetry(() => agent(
      `${driverSh('profile.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${buildOut}.run.json --out ${profOut}`)}\n` +
      `Return {ok, native_path}.`,
      { model: MODEL.profile, label: 'driver-profile-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
    evidenceOut = await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
      `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
      { model: MODEL.mechanical, label: 'driver-to-evidence-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  } else {
    // Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run the native profiler.
    // run.sh/latency path above already produced throughput. When method='perf_heuristic',
    // normalize that throughput via the strategist normalizer, tagging bottlenecks
    // evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.
    evidenceOut = await agentRetry(() => agent(
      `Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run a native profiler. ` +
      `Normalize the run.sh throughput (latency_ms=${(runOut && runOut.latency_ms) || 'N/A'}) into canonical metrics via ` +
      `run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${PROFILING_DECISION.normalizer || 'perf_to_evidence.py'} --baseline ${EXP_DIR}/stark_root.result.json\`.\n` +
      `Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
      `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
      { model: MODEL.mechanical, label: 'driver-to-evidence-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  }
  await agentRetry(() => agent(
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
    `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
    { model: MODEL.mechanical, label: 'driver-diagnose-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  await agentRetry(() => agent(
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${EXP_DIR}/stark_root.result.json\`.\n` +
    `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
    { model: MODEL.mechanical, label: 'driver-anti-cheat-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  rootEval.driver_envelope = {
    latency_ms: Number((runOut && runOut.latency_ms) || 0),
    metrics: (evidenceOut && evidenceOut.metrics) || {},
    backend_id: DRIVER_BACKEND_ID,
  }
}

// Initialize root node
rootNode = {
  id: 'root',
  parent_id: null,
  kernel_code: referenceKernelCode,
  plan: 'Reference kernel (no optimization applied)',
  anchors: '',
  runtime: rootEval.correct ? rootEval.runtime_ms : null,
  correct: rootEval.correct,
  compile_ok: rootEval.compile_ok,
  logs: rootEval.logs,
  children: [],
  status: rootEval.correct ? 'ok' : 'failed',
}

tree.push(rootNode)
leaderboard = [rootNode]

if (!rootEval.correct) {
  log('WARNING: Reference kernel itself fails correctness test. Proceeding with caution.')
} else {
  log(`Root runtime: ${rootEval.runtime_ms}ms`)
}

bestKernel = rootNode.kernel_code
bestRuntime = rootNode.runtime

// =============================================================================
// Main Loop: Select → Plan/Code/Debug → Evaluate → Update (Algorithm 1)
// =============================================================================

for (let t = 0; t < BUDGET; t++) {
  attemptCount = t + 1
  log(`\n=== Attempt ${attemptCount}/${BUDGET} ===`)

  // ===========================================================================
  // Phase 2: Select — ε-greedy node selection
  // ===========================================================================
  phase('Select')

  const selectedNode = selectNode()
  const selectedId = selectedNode.id
  log(`Selected node: ${selectedId} (score=${getScore(selectedNode)}, status=${selectedNode.status})`)

  // ===========================================================================
  // Phase 3: Plan OR Debug
  // ===========================================================================

  let newKernelCode = ''
  let newPlan = ''
  let newAnchors = ''
  let isDebugPath = false

  if (!selectedNode.compile_ok || !selectedNode.correct) {
    // HasBug(i) → Debug path
    phase('Debug')
    isDebugPath = true
    log(`Debug path for ${selectedId}: ${selectedNode.logs?.substring(0, 200) || 'unknown error'}`)

    const debugCtx = buildDebugContext(selectedId, fenceToken())

    const debugResult = await agentRetry(() => agent(`You are a kernel debugging expert. Fix the failing kernel using sibling patterns.

${debugCtx}

# Instructions
1. Analyze the error logs carefully
2. Look at sibling kernels for successful fix patterns (e.g., off-by-one guards, stride/index alignment, launch parameter tweaks, shared memory sizing)
3. Make MINIMAL changes to fix the bug — preserve the overall approach
4. Return the COMPLETE fixed kernel code

Return a JSON object with:
- kernel_code: string (complete fixed ${langToken(LEGACY_DEBUG_LANG_TOKEN)} kernel)
- fix_summary: string (brief description of what was wrong and how it was fixed)
- confidence: 'high' | 'medium' | 'low'

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Debug","ts":"<ts>","status":"done","candidate_id":"node_${attemptCount}","technique":"debug_fix","speedup":null,"note":"<what was wrong + how it was fixed + confidence, one line>"}`, {
      label: `debug-${selectedId}-a${attemptCount}`,
      phase: 'Debug',
      schema: {
        type: 'object',
        properties: {
          kernel_code: { type: 'string' },
          fix_summary: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['kernel_code', 'fix_summary'],
      },
    }), { retries: 5 })

    newKernelCode = debugResult.kernel_code
    newPlan = selectedNode.plan || 'Debug fix'
    newAnchors = selectedNode.anchors || ''
    log(`Debug result: ${debugResult.fix_summary} (confidence: ${debugResult.confidence})`)

  } else {
    // Normal path: Plan + Code
    phase('Plan')

    const planCtx = buildPlanContext(selectedId, fenceToken())

    const planResult = await agentRetry(() => agent(`You are a kernel optimization strategist. Propose a concrete, actionable optimization plan with grounded instruction anchors.

${planCtx}

# Instructions
1. Propose ONE specific optimization (e.g., shared memory tiling, vectorized loads, loop unrolling, warp-level primitives, register blocking)
2. Provide GROUNDED INSTRUCTIONS by inserting span anchors in the kernel code:
   Use markers like:
   <<<IMPROVE BEGINS: optimization_name>>>
   // existing code to be modified
   <<<IMPROVE ENDS: optimization_name>>>
3. The anchors must wrap EXACTLY the code spans that need modification
4. Include a brief rationale for WHY this optimization is chosen
5. Consider the code agent's demonstrated capabilities (seen in children nodes) — don't propose instructions beyond what has been shown to work

Return a JSON object with:
- plan: string (text description of the optimization strategy)
- anchored_scaffold: string (kernel code with <<<IMPROVE BEGINS/ENDS>>> anchors)
- anchors: array of {name, begin_line, end_line, description}
- rationale: string
${__attemptBlock()}${__experienceBlock()}
# Recent genome trajectory (read BEFORE proposing)
Run \`tail -20 ${EXP_DIR}/genome.jsonl 2>/dev/null\` to see prior attempts across the tree this session. Use it to: (a) avoid re-proposing optimizations that already regressed on sibling or ancestor nodes, (b) spot patterns the per-node sibling summary may have missed. If the file is empty or missing, ignore this and rely on the tree context above.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Plan","ts":"<ts>","status":"done","candidate_id":"node_${attemptCount}","technique":"<the single optimization you propose, e.g. shared_memory_tiling>","speedup":null,"note":"<rationale for this optimization, one line>"}`, {
      label: `plan-${selectedId}-a${attemptCount}`,
      phase: 'Plan',
      schema: {
        type: 'object',
        properties: {
          plan: { type: 'string' },
          anchored_scaffold: { type: 'string' },
          anchors: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                name: { type: 'string' },
                begin_line: { type: 'number' },
                end_line: { type: 'number' },
                description: { type: 'string' },
              },
              required: ['name', 'begin_line', 'end_line'],
            },
          },
          rationale: { type: 'string' },
        },
        required: ['plan', 'anchored_scaffold', 'anchors', 'rationale'],
      },
    }), { retries: 5 })

    newPlan = planResult.plan
    newAnchors = JSON.stringify(planResult.anchors)
    log(`Plan: ${planResult.plan.substring(0, 120)}...`)
    log(`Anchors: ${planResult.anchors.map(a => a.name).join(', ')}`)

    // =========================================================================
    // Phase 4: Code — Realize grounded instructions
    // =========================================================================
    phase('Code')

    const codeCtx = buildCodeContext(selectedId, fenceToken())

    const codeResult = await agentRetry(() => agent(`You are a kernel coding expert. Realize the grounded instructions into executable ${langToken(LEGACY_CODE_LANG_TOKEN)} code.

${codeCtx}

# Grounded Instructions from Plan Agent
Plan: ${planResult.plan}

Anchored Scaffold:
\`\`\`${fenceToken()}
${planResult.anchored_scaffold.substring(0, 4000)}
\`\`\`

# Anchor Details
${planResult.anchors.map(a => `- ${a.name} (lines ${a.begin_line}-${a.end_line}): ${a.description}`).join('\n')}

# Instructions
1. Replace each <<<IMPROVE BEGINS/ENDS>>> anchor with concrete, correct ${langToken(LEGACY_CODE_LANG_TOKEN)} code
2. Preserve all code OUTSIDE the anchors exactly as-is
3. Use successful patterns from sibling kernels when appropriate
4. Ensure the resulting code is syntactically valid and compilable
5. Do NOT use <<<IMPROVE>>> markers in the final output — they must be fully resolved

Return a JSON object with:
- kernel_code: string (complete, anchor-free ${langToken(LEGACY_CODE_LANG_TOKEN)} kernel)
- implementation_notes: string (brief notes on how anchors were resolved)
- anchor_resolutions: array of {name, resolution_summary}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Code","ts":"<ts>","status":"done","candidate_id":"node_${attemptCount}","technique":"<the optimization you realized into code>","speedup":null,"note":"<how the anchors were resolved into concrete code, one line>"}`, {
      label: `code-${selectedId}-a${attemptCount}`,
      phase: 'Code',
      schema: {
        type: 'object',
        properties: {
          kernel_code: { type: 'string' },
          implementation_notes: { type: 'string' },
          anchor_resolutions: { type: 'array', items: { type: 'object' } },
        },
        required: ['kernel_code', 'implementation_notes'],
      },
    }), { retries: 5 })

    newKernelCode = codeResult.kernel_code
    log(`Code generated: ${newKernelCode.length} chars`)
  }

  // ===========================================================================
  // Phase 5: Evaluate — Compile, correctness test, runtime
  // ===========================================================================
  phase('Evaluate')

  const evalResult = await agentRetry(() => agent(`Evaluate this kernel for correctness and performance.

# Kernel Code
\`\`\`${fenceToken()}
${newKernelCode.substring(0, 4000)}
\`\`\`

# Compile and Run
${COMPILE_CMD ? `Compile: ${COMPILE_CMD}` : 'No compile_command provided; perform static compileability review only.'}
${BENCHMARK_CMD ? `Benchmark: ${BENCHMARK_CMD}` : 'No benchmark_command provided; do not invent a test harness.'}

# Reference for comparison
Baseline runtime: ${bestRuntime !== null ? bestRuntime + 'ms' : 'N/A'}

Return JSON with:
- compile_ok: boolean
- correct: boolean (passes correctness test)
- runtime_ms: number (wall-clock, lower is better; null if failed)
- speedup_vs_baseline: number (or null)
- logs: string (compiler + test output)
- error_category: 'compile_error' | 'runtime_error' | 'correctness_fail' | 'timeout' | 'none'

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if compiled AND correct, else "error"; speedup is the measured speedup_vs_baseline number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"node_${attemptCount}","speedup":<number or null>,"technique":"evaluate_candidate","note":"<compile_ok? correct? runtime_ms; or the error_category and failure reason>"}`, {
    label: `eval-a${attemptCount}`,
    phase: 'Evaluate',
    schema: {
      type: 'object',
      properties: {
        compile_ok: { type: 'boolean' },
        correct: { type: 'boolean' },
        runtime_ms: { type: 'number' },
        speedup_vs_baseline: { type: 'number' },
        logs: { type: 'string' },
        error_category: { type: 'string', enum: ['compile_error', 'runtime_error', 'correctness_fail', 'timeout', 'none'] },
      },
      required: ['compile_ok', 'correct', 'runtime_ms', 'logs', 'error_category'],
    },
  }), { retries: 5 })

  if (USE_DRIVER_STANDALONE) {
    const kPath = starkNodeKernelPath(`node_${attemptCount}`)
    const buildOut = `${EXP_DIR}/node_${attemptCount}.artifact`
    const profOut = `${EXP_DIR}/node_${attemptCount}.prof.native`
    await agentRetry(() => agent(
      `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
      `Return its stdout JSON verbatim.`,
      { model: MODEL.mechanical, label: `driver-build-${attemptCount}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    const runOut = await agentRetry(() => agent(
      `${driverSh('run.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${buildOut}.run.json`)}\n` +
      `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
      { model: MODEL.profile, label: `driver-run-${attemptCount}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    let evidenceOut = null
    if (PROFILING_DECISION.method === 'native_profiler') {
      await agentRetry(() => agent(
        `${driverSh('profile.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${buildOut}.run.json --out ${profOut}`)}\n` +
        `Return {ok, native_path}.`,
        { model: MODEL.profile, label: `driver-profile-${attemptCount}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      evidenceOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
        `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
        { model: MODEL.mechanical, label: `driver-to-evidence-${attemptCount}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    } else {
      // profiling-strategist chose method='perf_heuristic' (or non-native); do NOT run profile.sh.
      // run.sh above is the measurement; normalize that throughput via the strategist normalizer and
      // write heuristic_bclass so diagnose.py does not fall to unknown (dacbd4f contract).
      evidenceOut = await agentRetry(() => agent(
        `Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run a native profiler. ` +
        `run.sh returned latency_ms=${(runOut && runOut.latency_ms) || 'null'}. ` +
        `Normalize that throughput into canonical metrics via ` +
        `run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${PROFILING_DECISION.normalizer || 'perf_to_evidence.py'} --baseline ${EXP_DIR}/node_${attemptCount}.result.json\`. ` +
        `Also write heuristic_bclass (memory_bound|compute_bound|latency_bound) based on the throughput ratio. ` +
        `Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
        `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, heuristic_bclass, coverage, source_backend}.`,
        { model: MODEL.mechanical, label: `driver-to-evidence-${attemptCount}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    }
    const diagOut = await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
      `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
      { model: MODEL.mechanical, label: `driver-diagnose-${attemptCount}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${EXP_DIR}/node_${attemptCount}.result.json\`.\n` +
      `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
      { model: MODEL.mechanical, label: `driver-anti-cheat-${attemptCount}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    evalResult.driver_envelope = {
      latency_ms: Number((runOut && runOut.latency_ms) || 0),
      metrics: (evidenceOut && evidenceOut.metrics) || {},
      bottleneck_class: (diagOut && diagOut.bottleneck_class) || (evidenceOut && evidenceOut.heuristic_bclass) || 'unknown',
      backend_id: DRIVER_BACKEND_ID,
    }
  } else if (IS_EMBEDDED) {
    // --- Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch).
    // SERIAL: STARK's main attempt loop is already a sequential for-loop (one candidate
    // per iteration, no parallel(); see eval above), so there is no race on the shared
    // KERNEL_PATH/project build. ---
    const kPath = starkNodeKernelPath(`node_${attemptCount}`)
    const variant = `stark_node_${attemptCount}`.replace(/[^A-Za-z0-9_]/g, '_')
    let embLatency = 0, embMetrics = {}, embBclass = 'unknown'
    if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
      const embResult = await agentRetry(() => agent(
        `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project kernel: ${REF_KERNEL_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
        `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${REF_KERNEL_PATH}\n` +
        `2. Apply candidate: cp ${kPath} ${REF_KERNEL_PATH}\n3. Build: ${BUILD_CMD}\n4. Test: ${args.test_command || TEST_HARNESS_PATH || PROJECT_BENCH_CMD}\n5. Benchmark: ${PROJECT_BENCH_CMD || args.test_command || TEST_HARNESS_PATH}\n` +
        `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${REF_KERNEL_PATH}\n` +
        `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
        { model: MODEL.mechanical, label: `embedded-inplace-${attemptCount}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      embLatency = Number(embResult?.latency_ms || 0); embBclass = embResult?.heuristic_bclass || 'unknown'; embMetrics = embResult?.metrics || { latency_ms: embLatency }
    } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
      const _plan = typeof __embeddedEvalPlan === 'function'
        ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: args.test_command || TEST_HARNESS_PATH || PROJECT_BENCH_CMD, benchmarkCmd: PROJECT_BENCH_CMD || args.test_command || TEST_HARNESS_PATH })
        : null
      if (_plan) {
        const embResult = await agentRetry(() => agent(
          `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
          `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-dispatch-${attemptCount}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        embLatency = Number(embResult?.latency_ms || 0); embBclass = embResult?.heuristic_bclass || 'unknown'; embMetrics = embResult?.metrics || { latency_ms: embLatency }
      }
    }
    evalResult.driver_envelope = { latency_ms: embLatency, metrics: embMetrics, bottleneck_class: embBclass, backend_id: 'embedded' }
  }

  // ===========================================================================
  // Phase 6: Update — Append to tree, update leaderboard
  // ===========================================================================
  phase('Update')

  const childId = `node_${attemptCount}`
  const childNode = {
    id: childId,
    parent_id: selectedId,
    kernel_code: newKernelCode,
    plan: newPlan,
    anchors: newAnchors,
    runtime: evalResult.correct ? evalResult.runtime_ms : null,
    correct: evalResult.correct,
    compile_ok: evalResult.compile_ok,
    logs: evalResult.logs,
    children: [],
    status: evalResult.correct ? 'ok' : 'failed',
    attempt: attemptCount,
    is_debug_path: isDebugPath,
    speedup: evalResult.speedup_vs_baseline,
  }

  tree.push(childNode)
  selectedNode.children.push(childId)

  // Update best
  if (evalResult.correct && evalResult.runtime_ms !== null) {
    if (bestRuntime === null || evalResult.runtime_ms < bestRuntime) {
      bestRuntime = evalResult.runtime_ms
      bestKernel = newKernelCode
      log(`NEW BEST: ${childId} — ${evalResult.runtime_ms}ms (speedup: ${evalResult.speedup_vs_baseline}x)`)
    } else {
      log(`${childId}: ${evalResult.runtime_ms}ms (speedup: ${evalResult.speedup_vs_baseline}x)`)
    }
  } else {
    log(`${childId}: FAILED — ${evalResult.error_category} (${evalResult.logs?.substring(0, 150) || 'no logs'})`)
  }

  // Update leaderboard
  updateLeaderboard()
  log(`Leaderboard: ${leaderboard.length} entries, best=${leaderboard[0]?.runtime || 'N/A'}ms`)
}

// =============================================================================
// Phase 7: Report — Final summary
// =============================================================================
phase('Report')

const bestNode = leaderboard[0] || rootNode
const allCorrect = tree.filter(n => n.correct)
const allFailed = tree.filter(n => !n.correct && n.id !== 'root')

const report = await agentRetry(() => agent(`Generate a comprehensive optimization report for this STARK session.

# Session Statistics
- Total attempts: ${attemptCount}
- Correct kernels: ${allCorrect.length}
- Failed attempts: ${allFailed.length}
- Best runtime: ${bestRuntime !== null ? bestRuntime + 'ms' : 'N/A'}
- Initial runtime: ${rootNode.runtime !== null ? rootNode.runtime + 'ms' : 'N/A'}
- Best speedup: ${bestNode?.speedup || 'N/A'}x

# Best Kernel (id=${bestNode.id})
\`\`\`${fenceToken()}
${bestNode.kernel_code.substring(0, 4000)}
\`\`\`

# Plan for Best Kernel
${bestNode.plan || 'N/A'}

# Tree Summary
Nodes by depth:
- Root: 1 node, ${rootNode.runtime}ms
- Level 1: ${getChildren('root').length} nodes
- Level 2+: ${tree.filter(n => n.parent_id !== 'root' && n.parent_id !== null).length} nodes

Failed breakdown:
${allFailed.reduce((acc, n) => {
  const cat = n.compile_ok ? (n.correct ? 'ok' : 'correctness') : 'compile'
  acc[cat] = (acc[cat] || 0) + 1
  return acc
}, {})}

Return a JSON object with:
- outcome: 'success' | 'partial' | 'failed'
- summary: string (1-2 sentence executive summary)
- best_runtime_ms: number
- baseline_runtime_ms: number
- speedup: number
- total_attempts: number
- correct_kernels: number
- failed_kernels: number
- best_kernel_code: string
- best_kernel_plan: string
- best_kernel_id: string
- optimization_trajectory: array of {attempt, node_id, runtime, speedup, plan_summary}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the session outcome (status="done" if outcome is success or partial, else "error"; speedup is the best speedup number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"<done|error>","speedup":<number or null>,"technique":"session_report","note":"<outcome + best runtime vs baseline + best_kernel_id, one line>"}`, {
  label: 'report-final',
  phase: 'Report',
  schema: {
    type: 'object',
    properties: {
      outcome: { type: 'string', enum: ['success', 'partial', 'failed'] },
      summary: { type: 'string' },
      best_runtime_ms: { type: 'number' },
      baseline_runtime_ms: { type: 'number' },
      speedup: { type: 'number' },
      total_attempts: { type: 'number' },
      correct_kernels: { type: 'number' },
      failed_kernels: { type: 'number' },
      best_kernel_code: { type: 'string' },
      best_kernel_plan: { type: 'string' },
      best_kernel_id: { type: 'string' },
      optimization_trajectory: { type: 'array', items: { type: 'object' } },
    },
    required: ['outcome', 'summary', 'total_attempts', 'correct_kernels', 'failed_kernels'],
  },
}), { retries: 5, allowNull: true })

// embedded_inplace exit safety net (unconditional restore of the project kernel)
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${REF_KERNEL_PATH}"\` and confirm. ALWAYS restore.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  kernel_path: REF_KERNEL_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  outcome: report?.outcome || 'unknown',
  summary: report?.summary || '',
  best_runtime_ms: bestRuntime,
  baseline_runtime_ms: rootNode.runtime,
  speedup: bestNode?.speedup || (rootNode.runtime && bestRuntime ? rootNode.runtime / bestRuntime : null),
  total_attempts: attemptCount,
  correct_kernels: allCorrect.length,
  failed_kernels: allFailed.length,
  best_kernel_id: bestNode?.id,
  best_kernel_code: bestKernel || '',
  best_kernel_plan: bestNode?.plan || '',
  reference_kernel_code: referenceKernelCode,
  tree_summary: tree.map(n => ({
    id: n.id,
    parent_id: n.parent_id,
    runtime: n.runtime,
    correct: n.correct,
    compile_ok: n.compile_ok,
    status: n.status,
    attempt: n.attempt,
    is_debug_path: n.is_debug_path,
  })),
  leaderboard: leaderboard.map(n => ({
    id: n.id,
    runtime: n.runtime,
    speedup: n.speedup,
  })),
  optimization_trajectory: report?.optimization_trajectory || [],
}
