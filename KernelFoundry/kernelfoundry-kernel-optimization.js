export const meta = {
  name: 'kernelfoundry-kernel-optimization',
  description: 'Evolutionary MAP-Elites quality-diversity search with meta-prompt evolution and templated parameter tuning (KernelFoundry methodology)',
  whenToUse: 'When generating GPU kernels (SYCL/CUDA/Triton) from PyTorch operator specs and you need diverse, high-quality solutions across multiple optimization strategies. Prevents mode collapse via behavioral-descriptor archive. Best for KernelBench-style tasks, custom operators, and cross-platform (Intel/NVIDIA) kernel generation.',
  phases: [
    { title: 'Setup', detail: 'Parse task specification, establish baseline, initialize MAP-Elites archive' },
    { title: 'Select', detail: 'Sample parent(s) from archive using gradient-informed selection strategy' },
    { title: 'Vary', detail: 'LLM generates offspring via mutation guided by meta-evolved prompts' },
    { title: 'Evaluate', detail: 'Compile, validate correctness, benchmark, classify behavioral coordinates' },
    { title: 'Insert', detail: 'Update archive if offspring improves its behavioral cell; track transitions' },
    { title: 'Evolve-Prompts', detail: 'Meta-prompter analyzes outcomes, evolves prompt sections co-operatively' },
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

const WORKFLOW_NAME = 'kernelfoundry-kernel-optimization'


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

// --- BEGIN inlined typed-args (from _meta/scaffolding/typed-args.js) ---
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
// --- END inlined typed-args ---
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


// --- BEGIN inlined backend-axis (resolve) scaffolding (from _meta/scaffolding/backend-axis.js) ---
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
// --- END inlined backend-axis (resolve) scaffolding ---


// =============================================================================
// KernelFoundry: Hardware-Aware Evolutionary GPU Kernel Optimization
// =============================================================================
//
// Source: "KernelFoundry: Hardware-aware evolutionary GPU kernel optimization"
//         Wiedemann, Leboutet, Paulitsch, Wofk, Ummenhofer
//         Intel Corporation, arXiv:2603.12440, 2026
//
// Three key mechanisms:
//   1. MAP-Elites with kernel-specific behavioral descriptors:
//      - d_mem ∈ {0,1,2,3}: memory access pattern (scalar → multi-level hierarchy)
//      - d_algo ∈ {0,1,2,3}: algorithmic structure (direct translation → novel algorithm)
//      - d_sync ∈ {0,1,2,3}: parallelism coordination (none → global coordination)
//      → 4³ = 64 behavioral cells, each holding the best kernel for that strategy
//
//   2. Meta-prompt evolution: 4 evolvable prompt sections co-evolve with kernels
//      - optimization_philosophy, optimization_strategies, common_pitfalls, analysis_guidance
//      - A separate meta-prompter LLM analyzes outcomes and proposes SEARCH/REPLACE edits
//
//   3. Templated parameter optimization: kernels can declare configurable parameters
//      (tile sizes, work-group dims, unroll factors) evaluated independently per config
//
// Gradient-informed evolution:
//   ∇F (fitness gradient): which behavioral directions improve fitness
//   ∇R (improvement-rate gradient): which directions have high improvement probability
//   ∇E (exploration gradient): which empty/low-quality cells to explore
//   Combined: ∇ = α∇F + β∇R + γ∇E (default: 0.4, 0.4, 0.2)
//
// Usage:
//   Workflow({name: 'kernelfoundry-kernel-optimization', args: {
//     problem_definition: 'class Model(nn.Module): ...',
//     op_description: 'Fused softmax + dropout',
//     language: 'sycl',                    // 'sycl', 'cuda', 'triton'
//     target_gpu: 'Intel Arc B580',          // or 'NVIDIA A6000', etc.
//     test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     descriptor_result_path: '/tmp/kernelfoundry_exp/descriptors/latest.json',
//     archive_update_result_path: '/tmp/kernelfoundry_exp/archive/updates.jsonl',
//     generations: 40,
//     meta_prompt_interval: 10,
//     speedup_target: 2.0,
//     selection_strategy: 'mixed',
//   }})
//
// =============================================================================

// --- Required Args ---
const TASK_SPEC = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const OP_DESC = args.op_description || 'GPU kernel'

// --- Optional Args ---
const TARGET_LANG = args.language || 'cuda'
const TARGET_HW = args.target_gpu || 'NVIDIA GPU'
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || ''
const DESCRIPTOR_RESULT_PATH = args.descriptor_result_path || `${args.exp_dir || '/tmp/kernelfoundry_exp'}/descriptors/latest.json`
const ARCHIVE_UPDATE_RESULT_PATH = args.archive_update_result_path || `${args.exp_dir || '/tmp/kernelfoundry_exp'}/archive/updates.jsonl`
const GENERATIONS = args.generations || 30
const META_PROMPT_INTERVAL = args.meta_prompt_interval || 10
const SPEEDUP_TARGET = args.speedup_target || 2.0
const SELECTION_STRATEGY = args.selection_strategy || 'mixed'
const EXP_DIR = args.exp_dir || '/tmp/kernelfoundry_exp'
const KERNEL_PATH = args.kernel_path || ''
const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const EVIDENCE_MODE = (TEST_CMD && BENCH_CMD) ? 'measured' : 'conservative_missing_evidence'

// --- Backend driver wiring (P5d Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''

// --- Project-native integration (embedded kernels via integration-strategist) ---
// PROJECT_BUILD_CMD/REGISTER_SCRIPT are new; benchmark/test reuse the existing
// BENCH_CMD/TEST_CMD consts (do NOT redeclare them).
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const PROJECT_BUILD_CMD = args.build_command || ''
const REGISTER_SCRIPT = args.register_script || ''

const LEGACY_LANG_TOKEN = TARGET_LANG
const LEGACY_FENCE_TOKEN = TARGET_LANG
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

// --- BEGIN inlined backend-axis (driver) scaffolding (from _meta/scaffolding/backend-axis.js) ---
function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}
// --- END inlined backend-axis (driver) scaffolding ---
function substrateInstruction(script, cliArgs) {
  const p = `${SUBSTRATE}/${script}`
  return PY ? `Run exactly: \`${PY} ${p} ${cliArgs}\`.`
            : `No substrate_command_prefix for ${p} ${cliArgs}; do not invent an interpreter.`
}

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it below. The agent only classifies the task (fuzzy); the substrate stamps
// confidence by method (measured/inferred/hypothesized) -- not the agent. See
// _substrate/profiling/README.md. Falls back to native_profiler if undecided. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }

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
function kernelPathForGeneration(gen) {
  const ext = USE_DRIVER ? (DRIVER_SOURCE_EXT || '.cu') : '.cu'
  return `${EXP_DIR}/gen_${gen}${ext}`
}

// --- State: MAP-Elites Archive ---
// 4x4x4 = 64 cells, indexed by (d_mem, d_algo, d_sync)
let archive = {}           // key="d_mem,d_algo,d_sync" → {code, fitness, speedup, id}
let transitions = []       // [{parent_cell, child_cell, delta_f, outcome, gen}]
let generation = 0
let globalBest = { code: '', fitness: 0, speedup: 0, cell: '' }

// Meta-prompt evolvable sections
let metaPrompt = {
  optimization_philosophy: 'Prioritize memory bandwidth utilization before compute optimization. Minimize global memory accesses through data reuse.',
  optimization_strategies: 'Memory: coalesced loads, shared/local memory tiling, vectorized access (vec4/vec8). Compute: loop unrolling, fused operations, register blocking. Parallelism: work-group barriers for shared memory, sub-group shuffles for reductions.',
  common_pitfalls: 'Avoid bank conflicts in shared memory. Do not assume specific work-group sizes. Guard against out-of-bounds access for non-power-of-2 dimensions.',
  analysis_guidance: 'Before writing code: identify the memory access pattern, determine arithmetic intensity, choose tiling strategy based on data reuse distance.',
}
let promptArchive = []     // [{prompt_sections, best_fitness, generation}]

// Behavioral descriptor classification
const BEHAVIORAL_DIMS = {
  d_mem: ['scalar/strided/uncoalesced', 'coalesced/vectorized (vec4, aligned)', 'shared/local memory with explicit tiling', 'multi-level hierarchy (SLM + reg blocking + prefetch)'],
  d_algo: ['direct PyTorch translation', 'fused operations (single-pass)', 'reformulated algorithm (online norm, flash pattern)', 'novel/asymptotically improved algorithm'],
  d_sync: ['no synchronization (embarrassingly parallel)', 'work-group barriers', 'sub-group primitives (shuffles, reductions, broadcast)', 'global coordination (atomics, multi-pass with sync)'],
}

// Fitness function (Section 3.2)
function computeFitness(compiled, correct, speedup) {
  if (!compiled) return 0.0
  if (!correct) return 0.1
  const sNorm = Math.min(1.0, speedup / SPEEDUP_TARGET)
  return 0.5 + 0.5 * sNorm
}

// =============================================================================
// Phase 1: Setup — Parse task, baseline, initialize archive
// =============================================================================
phase('Setup')

if (USE_DRIVER) {
  DRIVER = await agentRetry(() => agent(
    `Load the backend driver at ${BACKEND_DIR} and return its manifest plus idioms verbatim.\n` +
    `1. Run exactly: \`cat ${driverPath('manifest.json')}\` and parse JSON.\n` +
    `2. Run exactly: \`cat ${driverPath('idioms.json')}\` and parse JSON.\n` +
    `Return {present, backend_id, source_ext, aux_ext, lang_fence, impl_requirements, methods}.`,
    { model: MODEL.mechanical, label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  // #31b: load-driver fallback. The agent is `{ allowNull: true }`, so a
  // transient failure (sustained 429 / agent skipped) returns null after
  // retries — distinct from `present===false` (the manifest explicitly says no
  // driver, a real config error). The workflow sandbox cannot cat+parse the
  // JSON files directly (all Bash runs through agent()), so on a transient null
  // we degrade to the legacy path (no idioms) with a loud warning rather than
  // aborting the whole round. `present===false` still throws.
  if (!DRIVER) {
    log(`WARNING: load-driver agent returned null after retries — continuing without backend idioms (legacy path). Verify ${BACKEND_DIR}/manifest.json + idioms.json are readable by the agent.`)
  } else if (DRIVER.present === false) {
    throw new Error(`No backend driver present at ${BACKEND_DIR}. Provide a valid backend_dir or omit it for the legacy path.`)
  } else {
    if (RESOLVED_BACKEND && DRIVER.backend_id && normalizeSuitabilityValue(DRIVER.backend_id) !== RESOLVED_BACKEND) {
      throw new Error(`backend_dir manifest backend_id="${DRIVER.backend_id}" conflicts with args.backend/language="${RESOLVED_BACKEND}".`)
    }
    DRIVER_LANG_FENCE = DRIVER.lang_fence || DRIVER_LANG_FENCE
    DRIVER_IMPL_REQUIREMENTS = DRIVER.impl_requirements || ''
    DRIVER_SOURCE_EXT = DRIVER.source_ext || DRIVER_SOURCE_EXT
    DRIVER_BACKEND_ID = DRIVER.backend_id || DRIVER_BACKEND_ID
    log(`Driver loaded: ${DRIVER_BACKEND_ID} (fence=${DRIVER_LANG_FENCE})`)
  }
}

// profiling-strategist: classify the task once and let the substrate stamp the
// method+confidence. Honored in the per-generation Evaluate driver-profile branch.
if (USE_DRIVER) {
  const _pd = await agentRetry(() => agent(
    `${KERNEL_PATH ? `Read ${KERNEL_PATH}; ` : 'Read the operator spec below; '}` +
    `classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    substrateInstruction('profiling/profiling_strategist.py',
      `resolve --backend-manifest ${driverPath('manifest.json')} --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl`) +
    ` Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
  log(`Profiling-strategist: method=${PROFILING_DECISION.method} confidence=${PROFILING_DECISION.confidence} normalizer=${PROFILING_DECISION.normalizer}`)
}

// --- integration-strategist: route build/test mode (standalone vs embedded_*). ---
// Lets KernelFoundry handle inference-engine embedded operators (e.g. llama.cpp .cuh)
// not just standalone kernels. Default standalone => legacy path byte-identical.
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _probe = JSON.stringify({ compiler: true, project_build: !!PROJECT_BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `${KERNEL_PATH ? `Read ${KERNEL_PATH}; ` : 'Read the operator spec; '}` +
    `classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    substrateInstruction('integration/integration_strategist.py',
      `resolve --kernel "${KERNEL_PATH || EXP_DIR + '/operator.spec'}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' --cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl`) +
    ` Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
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
// A-O1 closure: native_profiler chosen but driver-profile path unavailable when not
// running the standalone driver envelope (KernelFoundry has no ncu_binary arg, so key
// on !USE_DRIVER_STANDALONE) -> downgrade so Profile uses perf_heuristic instead.
if (PROFILING_DECISION.method === 'native_profiler' && !USE_DRIVER_STANDALONE) {
  log(`profiling: native_profiler but not standalone-driver -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'test-harness-perf', rationale: 'native_profiler but not USE_DRIVER_STANDALONE -> perf_heuristic' }
}

const setupResult = await agentRetry(() => agent(`You are a GPU kernel optimization expert setting up the KernelFoundry evolutionary search.

# Task:
${KERNEL_PATH ? `Read kernel/operator from: ${KERNEL_PATH}` : ''}
${TASK_SPEC ? `\`\`\`python\n${TASK_SPEC.substring(0, 3000)}\n\`\`\`` : '(Determine from op_description)'}

# Operation: ${OP_DESC}
# Target language: ${langToken(LEGACY_LANG_TOKEN)} (SYCL/CUDA/Triton)
# Target hardware: ${TARGET_HW}
# Evidence contract:
- descriptor_result_path: ${DESCRIPTOR_RESULT_PATH}
- archive_update_result_path: ${ARCHIVE_UPDATE_RESULT_PATH}
- evidence_mode: ${EVIDENCE_MODE}
- If evidence_mode is conservative_missing_evidence, behavioral descriptors and MAP-Elites insertion decisions are not strict paper evidence.

# Setup Tasks:
1. Parse the operator specification (inputs, outputs, shapes, dtypes)
2. Establish PyTorch baseline performance: ${BENCH_CMD || '(estimate)'}
3. Identify the optimization feature space for this operator:
   - What memory access patterns are possible? (d_mem: 0-3)
   - What algorithmic reformulations exist? (d_algo: 0-3)
   - What parallelism levels apply? (d_sync: 0-3)
4. Initialize experiment: mkdir -p ${EXP_DIR}/{kernels,archive,prompts}

Return operator analysis and baseline.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"map_elites_setup","note":"<operator type + baseline ms + feasible behavioral cells, one line>"}`, {
  label: 'setup',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      operator_code: { type: 'string' },
      operator_type: { type: 'string' },
      input_shapes: { type: 'string' },
      baseline_time_ms: { type: 'number' },
      hardware_info: { type: 'string' },
      feasible_cells: { type: 'array', items: { type: 'string' } },
    },
    required: ['operator_code', 'baseline_time_ms'],
  },
}), { retries: 5 })

const operatorCode = setupResult.operator_code
const baselineTime = setupResult.baseline_time_ms

log(`Setup: ${setupResult.operator_type} on ${TARGET_HW} | Baseline: ${baselineTime}ms | Target: ${SPEEDUP_TARGET}x | Language: ${TARGET_LANG}`)

// =============================================================================
// MAP-Elites Evolutionary Loop
// =============================================================================

for (generation = 0; generation < GENERATIONS; generation++) {

  // ===========================================================================
  // Phase 2: Select — Sample parent(s) from archive
  // ===========================================================================
  phase('Select')

  // Gradient-informed selection (Section 3.3)
  const occupiedCells = Object.keys(archive)
  let selectedParent = null
  let selectionReason = ''

  if (occupiedCells.length === 0) {
    selectionReason = 'empty archive — generate from scratch'
  } else {
    // Mixed selection: uniform(0.3) + fitness-proportionate(0.4) + curiosity(0.3)
    const selIdx = (generation * 13 + 7) % 10
    if (selIdx < 3) {
      // Uniform: random occupied cell
      const cellKey = occupiedCells[(generation * 3) % occupiedCells.length]
      selectedParent = archive[cellKey]
      selectionReason = `uniform selection from cell [${cellKey}]`
    } else if (selIdx < 7) {
      // Fitness-proportionate
      const sorted = occupiedCells.map(k => archive[k]).sort((a, b) => b.fitness - a.fitness)
      selectedParent = sorted[generation % Math.min(3, sorted.length)]
      selectionReason = `fitness-proportionate (fitness=${selectedParent.fitness.toFixed(2)})`
    } else {
      // Curiosity-driven: pick cell with highest estimated improvement potential
      const sorted = occupiedCells.map(k => archive[k]).sort((a, b) => a.fitness - b.fitness)
      selectedParent = sorted[0]
      selectionReason = `curiosity-driven (lowest fitness cell, room to improve)`
    }
  }

  // Compute gradient hints from transition history
  let gradientHints = ''
  if (transitions.length >= 5) {
    const recentImprovements = transitions.filter(t => t.outcome === 'improvement').slice(-5)
    if (recentImprovements.length > 0) {
      gradientHints = `Recent successful directions: ${recentImprovements.map(t => `${t.parent_cell}→${t.child_cell} (+${t.delta_f.toFixed(2)})`).join(', ')}`
    }
  }

  log(`Gen ${generation + 1}/${GENERATIONS} | Archive: ${occupiedCells.length}/64 cells | Selection: ${selectionReason}`)

  // ===========================================================================
  // Phase 3: Vary — LLM generates offspring with meta-evolved prompts
  // ===========================================================================
  phase('Vary')

  const parentContext = selectedParent
    ? `\n# Parent Kernel (from cell [${selectedParent.cell}], fitness=${selectedParent.fitness.toFixed(2)}, speedup=${selectedParent.speedup.toFixed(2)}x):\n\`\`\`${fenceToken()}\n${selectedParent.code.substring(0, 4000)}\n\`\`\``
    : ''

  const varyResult = await agentRetry(() => agent(`You are a GPU kernel generator for the KernelFoundry evolutionary framework.
Generate a ${TARGET_LANG.toUpperCase()} kernel that implements the given operator.

# Operator to Implement:
\`\`\`python
${operatorCode.substring(0, 2500)}
\`\`\`

# Operation: ${OP_DESC}
# Target: ${langToken(LEGACY_LANG_TOKEN)} on ${TARGET_HW}
# Baseline: ${baselineTime}ms | Speedup target: ${SPEEDUP_TARGET}x

# === EVOLVED OPTIMIZATION GUIDANCE (meta-prompt) ===

## Optimization Philosophy:
${metaPrompt.optimization_philosophy}

## Optimization Strategies:
${metaPrompt.optimization_strategies}

## Common Pitfalls:
${metaPrompt.common_pitfalls}

## Analysis Guidance:
${metaPrompt.analysis_guidance}

# === END EVOLVED GUIDANCE ===
${parentContext}

${gradientHints ? `# Gradient Hints (from evolutionary history):\n${gradientHints}` : ''}

# Generation Requirements:
1. Produce a COMPLETE, COMPILABLE ${langToken(LEGACY_LANG_TOKEN)} kernel
2. Include all necessary headers/imports
3. If mutating a parent: make MEANINGFUL structural changes, not just parameter tweaks
4. Try to explore a DIFFERENT optimization strategy than the parent (different memory pattern, algorithm, or parallelism level)
5. You may optionally produce a TEMPLATED kernel with configurable parameters (tile_size, work_group_size, unroll_factor) alongside a dispatch function

Return the kernel code and its optimization strategy description.
${__attemptBlock()}${__experienceBlock()}
# Recent genome trajectory (read BEFORE varying)
Run \`tail -20 ${EXP_DIR}/genome.jsonl 2>/dev/null\` to see prior generations this session (every Vary/Evaluate step has self-reported). Use it to: (a) avoid producing an offspring whose strategy matches a recently-regressed sibling, (b) spot evolutionary patterns the gradientHints summary may have lost. If the file is empty or missing, ignore this and rely on the parent context above.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is generation ${generation}):
{"workflow":"${WORKFLOW_NAME}","phase":"Vary","ts":"<ts>","status":"done","candidate_id":"gen${generation}","technique":"<the mutation/variation strategy you applied: memory pattern, algorithm, or parallelism change>","note":"<how this offspring differs from the parent>"}`, {
    label: `vary-${generation}`,
    phase: 'Vary',
    schema: {
      type: 'object',
      properties: {
        kernel_code: { type: 'string' },
        strategy_description: { type: 'string' },
        memory_pattern: { type: 'string' },
        algorithm_type: { type: 'string' },
        parallelism_level: { type: 'string' },
        is_templated: { type: 'boolean' },
        template_params: { type: 'array', items: { type: 'string' } },
      },
      required: ['kernel_code', 'strategy_description'],
    },
  }), { retries: 5, allowNull: true })

  const offspringCode = varyResult?.kernel_code || ''

  // ===========================================================================
  // Phase 4: Evaluate — Compile + correctness + benchmark + classify
  // ===========================================================================
  phase('Evaluate')

  const evalResult = await agentRetry(() => agent(`You are a kernel evaluator for KernelFoundry. Evaluate this ${langToken(LEGACY_LANG_TOKEN)} kernel.

# Kernel Code:
\`\`\`${fenceToken()}
${offspringCode.substring(0, 5000)}
\`\`\`

# Reference Operator:
\`\`\`python
${operatorCode.substring(0, 1500)}
\`\`\`

# Evaluation Steps:
1. **Compile**: Can this ${langToken(LEGACY_LANG_TOKEN)} kernel compile? Check syntax, headers, type correctness.
${TEST_CMD ? `   Run: ${TEST_CMD}` : ''}
2. **Correctness**: Does it produce numerically equivalent output?
   Tolerance: relative precision ν < 0.01 in 99% of outputs.
3. **Performance**: Measure execution time.
${BENCH_CMD ? `   Run: ${BENCH_CMD}` : `   Estimate speedup over baseline (${baselineTime}ms).`}
   speedup = ${baselineTime}ms / kernel_time_ms

4. **Behavioral Classification** (assign coordinates 0-3 for each dimension):
   - d_mem (Memory Access Pattern):
     0: Scalar, strided, or uncoalesced access
     1: Coalesced/vectorized (vec4, aligned loads)
     2: Shared/local memory with explicit tiling
     3: Multi-level hierarchy (SLM + register blocking + prefetch)
   - d_algo (Algorithmic Structure):
     0: Direct PyTorch translation
     1: Fused operations (single-pass over data)
     2: Reformulated algorithm (online norm, flash pattern)
     3: Novel/asymptotically improved algorithm
   - d_sync (Parallelism Coordination):
     0: No synchronization (embarrassingly parallel)
     1: Work-group barriers (group::barrier)
     2: Sub-group primitives (shuffles, reductions, broadcast)
     3: Global coordination (atomics, multi-pass with sync)

   Use STATIC PATTERN MATCHING on the code (not runtime behavior).

Return evaluation results.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is generation ${generation}; status="done" if it compiled AND was correct, else "error"; speedup is the measured speedup number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"gen${generation}","speedup":<number or null>,"technique":"<behavioral cell d_mem,d_algo,d_sync you classified>","note":"<compiled? correct? speedup; or the failure reason>"}`, {
    label: `eval-${generation}`,
    phase: 'Evaluate',
    schema: {
      type: 'object',
      properties: {
        compiled: { type: 'boolean' },
        correct: { type: 'boolean' },
        speedup: { type: 'number' },
        kernel_time_ms: { type: 'number' },
        d_mem: { type: 'number' },
        d_algo: { type: 'number' },
        d_sync: { type: 'number' },
        error_message: { type: 'string' },
        performance_notes: { type: 'string' },
      },
      required: ['compiled', 'correct', 'speedup', 'd_mem', 'd_algo', 'd_sync'],
    },
  }), { retries: 5 })

  if (USE_DRIVER_STANDALONE) {
    const suffix = `${generation}`
    const kPath = kernelPathForGeneration(generation)
    const buildOut = `${EXP_DIR}/gen_${generation}.artifact`
    const profOut = `${EXP_DIR}/gen_${generation}.prof.native`
    const resultPath = `${EXP_DIR}/gen_${generation}.result.json`
    await agentRetry(() => agent(
      `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
      `Return its stdout JSON verbatim.`,
      { model: MODEL.mechanical, label: `driver-build-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    const runOut = await agentRetry(() => agent(
      `${driverSh('run.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${resultPath}`)}\n` +
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
      // do NOT run profile.sh. run.sh above already produced throughput. If perf_heuristic,
      // normalize that throughput into canonical metrics via the strategist normalizer.
      if (PROFILING_DECISION.method === 'perf_heuristic') {
        evidenceOut = await agentRetry(() => agent(
          `Throughput is in ${resultPath} from run.sh. Normalize it into canonical metrics via ` +
          substrateInstruction('profiling/' + (PROFILING_DECISION.normalizer || 'perf_to_evidence.py'), `--baseline ${resultPath} --peak-gflops <device_peak_gflops> --peak-gbs <device_peak_gbs>`) +
          ` Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
          `Also write heuristic_bclass (memory_bound|compute_bound|latency_bound) based on the throughput ratio so diagnose.py does not fall to unknown. ` +
          `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, heuristic_bclass, coverage, source_backend}.`,
          { model: MODEL.mechanical, label: `driver-to-evidence-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      }
    }
    const diagOut = await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
      `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
      { model: MODEL.mechanical, label: `driver-diagnose-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --source ${kPath} --metrics ${resultPath}\`.\n` +
      `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
      { model: MODEL.mechanical, label: `driver-anti-cheat-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    evalResult.driver_envelope = {
      latency_ms: Number((runOut && runOut.latency_ms) || 0),
      bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
      backend_id: DRIVER_BACKEND_ID,
      profiling_method: PROFILING_DECISION.method,
      profiling_confidence: PROFILING_DECISION.confidence,
    }
  } else if (IS_EMBEDDED) {
    // --- Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch) ---
    // KernelFoundry is MAP-Elites: one offspring per generation, evaluated inside this
    // serial `for (generation...)` loop. No `await parallel(` candidate eval here, so the
    // embedded branch is already serial — no race on the shared KERNEL_PATH / project build.
    const kPath = kernelPathForGeneration(generation)
    const variant = `kf_gen_${generation}`.replace(/[^A-Za-z0-9_]/g, '_')
    let embLatency = 0, embMetrics = {}, embBclass = 'unknown'
    // Materialize the offspring source to kPath first so build/test/bench can find it.
    await agentRetry(() => agent(`Write the offspring kernel source to ${kPath} (mkdir -p its parent dir first):\n\`\`\`${fenceToken()}\n${offspringCode.substring(0, 6000)}\n\`\`\``,
      { model: MODEL.mechanical, label: `embedded-materialize-${generation}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
      const embResult = await agentRetry(() => agent(
        `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project kernel: ${KERNEL_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
        `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
        `2. Apply candidate: cp ${kPath} ${KERNEL_PATH}\n3. Build: ${PROJECT_BUILD_CMD}\n4. Test: ${TEST_CMD}\n5. Benchmark: ${BENCH_CMD || TEST_CMD}\n` +
        `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
        `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
        { model: MODEL.mechanical, label: `embedded-inplace-${generation}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      embLatency = Number(embResult?.latency_ms || 0)
      embBclass = embResult?.heuristic_bclass || 'unknown'
      embMetrics = embResult?.metrics || { latency_ms: embLatency }
    } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
      const _plan = typeof __embeddedEvalPlan === 'function'
        ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: PROJECT_BUILD_CMD, testCmd: TEST_CMD, benchmarkCmd: BENCH_CMD || TEST_CMD })
        : null
      if (_plan) {
        const embResult = await agentRetry(() => agent(
          `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
          `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-dispatch-${generation}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        embLatency = Number(embResult?.latency_ms || 0)
        embBclass = embResult?.heuristic_bclass || 'unknown'
        embMetrics = embResult?.metrics || { latency_ms: embLatency }
      }
    }
    evalResult.driver_envelope = { latency_ms: embLatency, metrics: embMetrics, bottleneck_class: embBclass, backend_id: 'embedded' }
  }

  const fitness = computeFitness(evalResult.compiled, evalResult.correct, evalResult.speedup || 0)
  const cellKey = `${evalResult.d_mem || 0},${evalResult.d_algo || 0},${evalResult.d_sync || 0}`

  // ===========================================================================
  // Phase 5: Insert — Update archive if offspring improves its cell
  // ===========================================================================
  phase('Insert')

  const existingElite = archive[cellKey]
  let outcome = 'neutral'

  if (!existingElite || fitness > existingElite.fitness) {
    archive[cellKey] = {
      code: offspringCode,
      fitness: fitness,
      speedup: evalResult.speedup || 0,
      cell: cellKey,
      id: `gen${generation}`,
      strategy: varyResult.strategy_description,
    }
    outcome = existingElite ? 'improvement' : 'discovery'

    if (fitness > globalBest.fitness) {
      globalBest = { code: offspringCode, fitness, speedup: evalResult.speedup || 0, cell: cellKey }
    }
  } else {
    outcome = fitness < existingElite.fitness ? 'regression' : 'neutral'
  }

  // Record transition
  const parentCell = selectedParent?.cell || 'none'
  transitions.push({
    parent_cell: parentCell,
    child_cell: cellKey,
    delta_f: existingElite ? fitness - existingElite.fitness : fitness,
    outcome: outcome,
    gen: generation,
  })

  const statusIcon = outcome === 'improvement' ? '↑' : outcome === 'discovery' ? '★' : outcome === 'regression' ? '↓' : '='
  log(`  ${statusIcon} Cell [${cellKey}] fitness=${fitness.toFixed(2)} speedup=${(evalResult.speedup || 0).toFixed(2)}x | Archive: ${Object.keys(archive).length}/64 | Best: ${globalBest.speedup.toFixed(2)}x`)

  // ===========================================================================
  // Phase 6: Evolve-Prompts — Meta-prompter updates evolvable sections
  // ===========================================================================
  if ((generation + 1) % META_PROMPT_INTERVAL === 0 && generation > 0) {
    phase('Evolve-Prompts')

    const recentOutcomes = transitions.slice(-META_PROMPT_INTERVAL)
    const improvements = recentOutcomes.filter(t => t.outcome === 'improvement' || t.outcome === 'discovery')
    const failures = recentOutcomes.filter(t => t.outcome === 'regression' || t.outcome === 'neutral')

    const metaResult = await agentRetry(() => agent(`You are the KernelFoundry Meta-Prompter (Section 3.5).
Your job is to evolve the optimization guidance prompts based on recent evolutionary outcomes.

# Current Evolvable Prompt Sections:
## optimization_philosophy:
${metaPrompt.optimization_philosophy}

## optimization_strategies:
${metaPrompt.optimization_strategies}

## common_pitfalls:
${metaPrompt.common_pitfalls}

## analysis_guidance:
${metaPrompt.analysis_guidance}

# Recent Outcomes (last ${META_PROMPT_INTERVAL} generations):
- Improvements/discoveries: ${improvements.length}
- Regressions/neutral: ${failures.length}
- Successful transitions: ${improvements.map(t => `${t.parent_cell}→${t.child_cell}`).join(', ') || 'none'}

# Top archive entries:
${Object.entries(archive).sort((a, b) => b[1].fitness - a[1].fitness).slice(0, 5).map(([k, v]) => `[${k}] fitness=${v.fitness.toFixed(2)} speedup=${v.speedup.toFixed(2)}x: ${v.strategy?.substring(0, 60)}`).join('\n')}

# Meta-Prompting Rules (Section 3.5):
1. Diagnose which guidance was MISSING, MISLEADING, or INSUFFICIENT for recent outcomes
2. Prescribe targeted SEARCH/REPLACE updates to the 4 evolvable sections
3. Successful strategies should be REINFORCED; failed advice should be PRUNED
4. Keep each section concise (2-4 sentences)

Return updated prompt sections.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this meta-evolution ran at generation ${generation}):
{"workflow":"${WORKFLOW_NAME}","phase":"Evolve-Prompts","ts":"<ts>","status":"done","candidate_id":"gen${generation}","technique":"meta_prompt_evolution","note":"<which evolvable sections changed and why, one line>"}`, {
      label: `meta-prompt-${generation}`,
      phase: 'Evolve-Prompts',
      schema: {
        type: 'object',
        properties: {
          optimization_philosophy: { type: 'string' },
          optimization_strategies: { type: 'string' },
          common_pitfalls: { type: 'string' },
          analysis_guidance: { type: 'string' },
          evolution_rationale: { type: 'string' },
        },
        required: ['optimization_philosophy', 'optimization_strategies', 'common_pitfalls', 'analysis_guidance'],
      },
    }), { retries: 5, allowNull: true })

    if (metaResult) {
      metaPrompt = {
        optimization_philosophy: metaResult.optimization_philosophy || metaPrompt.optimization_philosophy,
        optimization_strategies: metaResult.optimization_strategies || metaPrompt.optimization_strategies,
        common_pitfalls: metaResult.common_pitfalls || metaPrompt.common_pitfalls,
        analysis_guidance: metaResult.analysis_guidance || metaPrompt.analysis_guidance,
      }
      promptArchive.push({ prompt_sections: { ...metaPrompt }, best_fitness: globalBest.fitness, generation })
      log(`  Meta-prompt evolved (gen ${generation + 1}): ${metaResult.evolution_rationale?.substring(0, 80) || 'updated'}`)
    }
  }
}

// =============================================================================
// Final Report
// =============================================================================
phase('Evaluate')

const finalReport = await agentRetry(() => agent(`Write a concise technical report on KernelFoundry MAP-Elites optimization.

# Results
- Operation: ${OP_DESC}
- Target: ${langToken(LEGACY_LANG_TOKEN)} on ${TARGET_HW}
- Baseline: ${baselineTime}ms
- Best speedup: ${globalBest.speedup.toFixed(2)}x (cell [${globalBest.cell}])
- Generations: ${GENERATIONS}
- Archive coverage: ${Object.keys(archive).length}/64 cells
- Total improvements: ${transitions.filter(t => t.outcome === 'improvement').length}
- Total discoveries: ${transitions.filter(t => t.outcome === 'discovery').length}
- Evidence mode: ${EVIDENCE_MODE}
- Descriptor artifact: ${DESCRIPTOR_RESULT_PATH}
- Archive update artifact: ${ARCHIVE_UPDATE_RESULT_PATH}

# Archive (top cells):
${Object.entries(archive).sort((a, b) => b[1].fitness - a[1].fitness).slice(0, 10).map(([k, v]) => `[${k}] ${v.speedup.toFixed(2)}x — ${v.strategy?.substring(0, 60)}`).join('\n')}

# Best Kernel:
\`\`\`${fenceToken()}
${globalBest.code.substring(0, 3000)}
\`\`\`

# Final Meta-Prompt State:
${Object.entries(metaPrompt).map(([k, v]) => `${k}: ${v}`).join('\n')}

Write:
1. Quality-diversity analysis: how well did the archive cover the behavioral space?
2. Meta-prompt evolution: how did the guidance change and what impact did it have?
3. Most effective optimization strategies discovered
4. Hardware awareness: evidence of hardware-specific optimizations

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (final report; speedup is the best speedup found, or null if none):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"done","candidate_id":"final","speedup":<number or null>,"technique":"<best cell strategy>","note":"<archive coverage + best speedup + most effective strategy, one line>"}`, {
  label: 'final-report',
  phase: 'Evaluate',
}), { retries: 5 })

// embedded_inplace exit safety net: unconditionally restore pristine original.
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${KERNEL_PATH}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  baseline_id: args.fair_baseline_id || null,  // #32: echo the frozen fair baseline KerSor handed us (contract.env::baseline_id via dispatch-args.json); null when undeclared -> check-acceptance-gate.sh Check 2c skips (back-compat).
  input_mode: INPUT_MODE,
  problem_definition: TASK_SPEC,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: globalBest.code ? `${EXP_DIR}/best_kernel.${TARGET_LANG === 'cuda' ? 'cu' : TARGET_LANG}` : '',
  initial_candidates: [],
  initial_generation_result: {
    verified: globalBest.speedup > 0,
    selected_candidate_id: globalBest.id || '',
  },
  operation: OP_DESC,
  language: TARGET_LANG,
  target_gpu: TARGET_HW,
  baseline_time_ms: baselineTime,
  best_speedup: globalBest.speedup,
  best_cell: globalBest.cell,
  best_kernel_code: globalBest.code,
  generations: GENERATIONS,
  archive_coverage: Object.keys(archive).length,
  archive_summary: Object.entries(archive).sort((a, b) => b[1].fitness - a[1].fitness).slice(0, 10).map(([k, v]) => ({ cell: k, speedup: v.speedup, strategy: v.strategy })),
  improvements: transitions.filter(t => t.outcome === 'improvement').length,
  discoveries: transitions.filter(t => t.outcome === 'discovery').length,
  final_meta_prompt: metaPrompt,
  prompt_evolution_history: promptArchive.length,
  descriptor_result_path: DESCRIPTOR_RESULT_PATH,
  archive_update_result_path: ARCHIVE_UPDATE_RESULT_PATH,
  evidence_mode: EVIDENCE_MODE,
  report: finalReport,
}
