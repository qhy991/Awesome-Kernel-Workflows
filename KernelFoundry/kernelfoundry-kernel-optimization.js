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
// --- BEGIN sol-execbench-eval substrate (auto-inlined by scripts/patch-sol-execbench-eval.js) ---
const SOL_SOLUTION_CONTRACT = [
  'SOL-EXECBENCH SOLUTION CONTRACT (this task is evaluated by the sol-execbench CLI):',
  '',
  'You are authoring a kernel that will be packaged into a solution.json and run by',
  'the sol-execbench harness, which compiles it internally. Therefore:',
  '',
  '1. Emit a COMPLETE kernel source plus a torch binding that exposes the task',
  '   entry point (run(...)). Do NOT write a standalone main()/CLI harness.',
  '2. Match the task reference signature exactly (same argument order/dtypes).',
  '3. Do NOT package, compile, or benchmark yourself — the workflow + substrate',
  '   handle pack -> sol-execbench -> parse. Return only the kernel + binding.',
].join('\n')

function __solQ(s) { return `"${String(s).replace(/"/g, '\\"')}"` }

function __solExecbenchEvalPlan(ctx) {
  const substrateDir = ctx.substrateDir            // abs path to _substrate/integration
  const kernelSource = ctx.kernelSource            // path to candidate kernel on disk
  const contractEnv = ctx.contractEnv              // path to session contract.env
  const solutionOut = ctx.solutionOut              // where to write solution.json
  const benchOut = ctx.benchOut                    // where sol-execbench writes bench.jsonl
  const normalizedOut = ctx.normalizedOut || ''    // optional canonical measurement JSON
  const solCli = ctx.solCli                        // e.g. /abs/sol-execbench/.venv/bin/sol-execbench
  const taskDir = ctx.taskDir                      // FlashInfer-Bench/<task> dir
  const benchConfig = ctx.benchConfig              // --config path
  const seedDir = ctx.seedDir                      // cd target for the run
  const cvd = ctx.cudaVisibleDevices || '0'
  const ld = ctx.ldLibraryPath ? `LD_LIBRARY_PATH=${__solQ(ctx.ldLibraryPath)}:$LD_LIBRARY_PATH ` : ''
  const env = ctx.envPrefix ? `${String(ctx.envPrefix).trim()} ` : ''
  const definition = ctx.definitionPath ? ` --definition ${__solQ(ctx.definitionPath)}` : ''

  const pack = `python3 ${__solQ(substrateDir + '/pack_sol_candidate.py')} --kernel ${__solQ(kernelSource)} --contract ${__solQ(contractEnv)} --out ${__solQ(solutionOut)}`
  const run = `cd ${__solQ(seedDir)} && ${env}${ld}CUDA_VISIBLE_DEVICES=${cvd} ${__solQ(solCli)} ${__solQ(taskDir)}${definition} --solution ${__solQ(solutionOut)} --config ${__solQ(benchConfig)} -o ${__solQ(benchOut)}`
  const parse = `python3 ${__solQ(substrateDir + '/parse_sol_bench.py')} ${__solQ(benchOut)}${normalizedOut ? ` --out ${__solQ(normalizedOut)}` : ''}`

  return {
    pack,
    run,
    parse,
    order: ['pack', 'run', 'parse'],
    cleanupInvariant: 'solution.json + bench.jsonl are per-candidate scratch files in the run dir; overwrite freely. No project source is mutated (non-mutating method).',
  }
}
// --- END sol-execbench-eval substrate ---


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

// --- BEGIN inlined runtime-safe-point scaffolding (from _meta/scaffolding/runtime-safe-point.js) ---
async function __workflowRuntimeSafePoint(ctx) {
  const checkpointPath = ctx.checkpointPath || `${ctx.expDir}/checkpoint.json`
  const materialize = ctx.materializeBest && ctx.bestKernelPath && ctx.bestKernelSourcePath
    ? `Atomically copy the exact bytes from immutable candidate ${ctx.bestKernelSourcePath} to ${ctx.bestKernelPath}. ` +
      `Use a small Python program: read the source as bytes, require its SHA-256 to equal ` +
      `${ctx.bestKernelExpectedSha256 || '<missing-required-sha256>'}, write a temporary file in the destination directory, ` +
      `fsync it, then os.replace it. Recompute the destination SHA-256 and fail if it differs. ` +
      `Never regenerate, reformat, or reconstruct the source from a prompt.`
    : ctx.materializeBest && ctx.bestKernelPath && ctx.bestKernelCode
    ? `Atomically write this exact best source to ${ctx.bestKernelPath} using a temporary file in the same directory followed by rename:\n` +
      `\`\`\`${ctx.bestLanguage || ''}\n${ctx.bestKernelCode}\n\`\`\``
    : (ctx.bestKernelPath
      ? `Preserve the existing best source at ${ctx.bestKernelPath}; do not rewrite it.`
      : 'There is no verified best source yet; do not create a best-kernel file.')

  return agentRetry(() => agent(`Workflow runtime safe point.

1. ${materialize}
2. Check cooperative termination:
   - termination file: ${ctx.terminationFile || '<none>'}
   - deadline epoch: ${ctx.deadlineEpoch || 0}
   A non-empty termination file requests stop. If it contains JSON, use its
   "reason"; otherwise use "supervisor_request". A positive deadline requests
   stop when the current epoch from \`date +%s\` is at or beyond it.
3. Start from this exact checkpoint object:
${JSON.stringify(ctx.checkpoint)}
   If step 2 requests stop, set termination_requested=true and set
   termination_reason to the observed reason. Otherwise preserve the planned
   termination fields. Atomically write the resulting JSON to ${checkpointPath}
   using a temporary file in the same directory followed by os.replace/rename.
   Do not change metric.name or metric.value.
4. Return only the termination decision and checkpoint path.
`, {
    model: MODEL.mechanical,
    label: ctx.label,
    phase: ctx.phase,
    schema: {
      type: 'object',
      properties: {
        termination_requested: { type: 'boolean' },
        termination_reason: { type: 'string' },
        checkpoint_path: { type: 'string' },
      },
      required: ['termination_requested', 'checkpoint_path'],
    },
  }), { retries: 5 })
}
// --- END inlined runtime-safe-point scaffolding ---

// --- BEGIN inlined turn-timeout scaffolding (from _meta/scaffolding/turn-timeout.js) ---
const TURN_TIMEOUT_MS = (args.turn_timeout_min || 12) * 60 * 1000  // per-turn wall-clock cap

/**
 * Wrap a doer-turn promise with a wall-clock cap. On expiry the returned
 * promise rejects with `turn-timeout: <label> exceeded Ns`. Degrades to a
 * passthrough when the runtime has no timers or TURN_TIMEOUT_MS <= 0.
 */
function withTurnTimeout(promise, label) {
  if (typeof setTimeout !== 'function' || !(TURN_TIMEOUT_MS > 0)) return promise
  let timer
  const guard = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`turn-timeout: ${label} exceeded ${Math.round(TURN_TIMEOUT_MS / 1000)}s`)),
      TURN_TIMEOUT_MS)
  })
  return Promise.race([promise, guard]).finally(() => {
    if (typeof clearTimeout === 'function') clearTimeout(timer)
  })
}
// --- END inlined turn-timeout scaffolding ---
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
//     stop_on_target: true,                 // false keeps fixed-budget research mode
//     target_patience: 2,                   // consecutive safe points at/above target
//     min_generations: 2,
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
const STOP_ON_TARGET = args.stop_on_target !== false
const TARGET_PATIENCE = Math.max(1, Number(args.target_patience || 2))
const MIN_GENERATIONS = Math.max(1, Number(args.min_generations || 2))
const SELECTION_STRATEGY = args.selection_strategy || 'mixed'
const EXP_DIR = args.exp_dir || '/tmp/kernelfoundry_exp'
const TERMINATION_FILE = args.termination_file || ''
const DEADLINE_EPOCH = Number(args.deadline_epoch || 0)
const CHECKPOINT_PATH = `${EXP_DIR}/checkpoint.json`
const KERNEL_PATH = args.kernel_path || ''
const TASK_IDENTITY_PATH = PROBLEM_PATH || KERNEL_PATH
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
const INTEGRATION_PATTERN = args.integration_pattern || 'standalone'
const SOL_CLI = args.sol_cli || ''
const SOL_TASK_DIR = args.sol_task_dir || ''
const SOL_BENCH_CONFIG = args.sol_bench_config || ''
const SOL_SEED_DIR = args.sol_seed_dir || EXP_DIR
const SOL_CVD = args.sol_cuda_visible_devices || '0'
const SOL_LD_LIBRARY_PATH = args.sol_ld_library_path || ''
const SOL_ENV_PREFIX = args.sol_env_prefix || ''
const SOL_DEFINITION_PATH = args.sol_definition_path || ''
const SOL_SUBSTRATE_DIR = args.sol_substrate_dir || `${SUBSTRATE}/integration`

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
  const legacyExt = TARGET_LANG === 'cuda' ? '.cu'
    : (TARGET_LANG === 'triton' || TARGET_LANG === 'python') ? '.py'
    : `.${TARGET_LANG}`
  const ext = USE_DRIVER ? (DRIVER_SOURCE_EXT || legacyExt) : legacyExt
  return `${EXP_DIR}/gen_${gen}${ext}`
}

function bestKernelPath() {
  const ext = TARGET_LANG === 'cuda' ? 'cu'
    : (TARGET_LANG === 'triton' || TARGET_LANG === 'python') ? 'py'
    : TARGET_LANG
  return `${EXP_DIR}/best_kernel.${ext}`
}

function harnessCommand(template, kernelPath, resultPath) {
  if (!template) return ''
  const hasResultPlaceholder = template.includes('{result_path}')
  let command = template.split('{kernel_path}').join(kernelPath)
  command = command.split('{result_path}').join(resultPath)
  return hasResultPlaceholder ? command : `${command} > "${resultPath}"`
}

// --- State: MAP-Elites Archive ---
// 4x4x4 = 64 cells, indexed by (d_mem, d_algo, d_sync)
let archive = {}           // key="d_mem,d_algo,d_sync" → {code, fitness, speedup, id}
let transitions = []       // [{parent_cell, child_cell, delta_f, outcome, gen}]
let generation = 0
let globalBest = {
  code: '', fitness: 0, speedup: 0, cell: '', id: '',
  compiled: false, correct: false, result_path: '', strategy: '',
  candidate_path: '', candidate_sha256: '', measurement_sha256: '',
  binding_path: '', binding_sha256: '', task_path: '', task_sha256: '',
  task_fingerprint_kind: '',
}
let generationsCompleted = 0
let targetStreak = 0
let terminationReason = 'generation_limit'
let checkpointedBestId = ''

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
let INTEGRATION_DECISION = {
  method: INTEGRATION_PATTERN === 'sol_execbench_solution' ? 'sol_execbench_solution' : 'standalone',
  build_fidelity: INTEGRATION_PATTERN === 'sol_execbench_solution' ? 'production' : 'isolated',
  reversible: true,
}
{
  const _probe = JSON.stringify({ compiler: true, project_build: !!PROJECT_BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true, sol_execbench_cli: !!SOL_CLI })
  const _preferred = INTEGRATION_PATTERN === 'sol_execbench_solution' ? ' --preferred-method sol_execbench_solution' : ''
  const _integ = await agentRetry(() => agent(
    `${KERNEL_PATH ? `Read ${KERNEL_PATH}; ` : 'Read the operator spec; '}` +
    `classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    substrateInstruction('integration/integration_strategist.py',
      `resolve --kernel "${KERNEL_PATH || EXP_DIR + '/operator.spec'}" --can-standalone <yes|no|uncertain>${_preferred} --host-probe '${_probe}' --cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl`) +
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
const IS_SOL = INTEGRATION_DECISION.method === 'sol_execbench_solution'
if (IS_SOL) {
  const missing = [
    ['sol_cli', SOL_CLI], ['sol_task_dir', SOL_TASK_DIR],
    ['sol_bench_config', SOL_BENCH_CONFIG], ['sol_seed_dir', SOL_SEED_DIR],
    ['sol_substrate_dir', SOL_SUBSTRATE_DIR],
  ].filter(([, value]) => !value).map(([name]) => name)
  if (missing.length) throw new Error(`sol_execbench_solution requires non-empty: ${missing.join(', ')}`)
}
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

${IS_SOL ? SOL_SOLUTION_CONTRACT : ''}

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

  const candidatePath = kernelPathForGeneration(generation)
  const generationResultPath = `${EXP_DIR}/gen_${generation}_result.json`
  const testCommand = harnessCommand(TEST_CMD, candidatePath, generationResultPath)
  const benchmarkCommand = harnessCommand(BENCH_CMD, candidatePath, generationResultPath)

  await agentRetry(() => agent(`Materialize the exact KernelFoundry candidate below at ${candidatePath}.
Create the parent directory first. Write the complete source byte-for-byte without
summarizing, repairing, or reformatting it. Use an atomic temporary file + rename.

\`\`\`${fenceToken()}
${offspringCode}
\`\`\`

Return {"written":true,"path":"${candidatePath}"}.`, {
    model: MODEL.mechanical,
    label: `materialize-${generation}`,
    phase: 'Evaluate',
    schema: {
      type: 'object',
      properties: {
        written: { type: 'boolean' },
        path: { type: 'string' },
      },
      required: ['written', 'path'],
    },
  }), { retries: 5 })

  const evalResult = IS_SOL
    ? await (async () => {
      const variant = `kf_gen_${generation}`.replace(/[^A-Za-z0-9_]/g, '_')
      const plan = __solExecbenchEvalPlan({
        substrateDir: SOL_SUBSTRATE_DIR,
        kernelSource: candidatePath,
        contractEnv: `${EXP_DIR}/contract.env`,
        solutionOut: `${EXP_DIR}/${variant}.solution.json`,
        benchOut: `${EXP_DIR}/${variant}.bench.jsonl`,
        normalizedOut: generationResultPath,
        solCli: SOL_CLI,
        taskDir: SOL_TASK_DIR,
        benchConfig: SOL_BENCH_CONFIG,
        seedDir: SOL_SEED_DIR,
        cudaVisibleDevices: SOL_CVD,
        ldLibraryPath: SOL_LD_LIBRARY_PATH,
        envPrefix: SOL_ENV_PREFIX,
        definitionPath: SOL_DEFINITION_PATH,
      })
      return agentRetry(() => agent(`Evaluate this already-materialized KernelFoundry candidate through the authoritative sol-execbench contract.

Candidate: ${candidatePath}
Run exactly in order:
1. PACK: ${plan.pack}
2. RUN: ${plan.run}
3. PARSE: ${plan.parse}

The parser writes the canonical measurement JSON at ${generationResultPath}.
Read only that JSON. Require compiled=true, correct=true, and n_pass==n_total>0
before accepting the candidate. Use its exact geomean_speedup as speedup. Do not
recompute a mean-latency ratio or reuse another generation's output.
Classify d_mem/d_algo/d_sync from ${candidatePath} only. ${plan.cleanupInvariant}

Return the parsed result.`, {
        label: `sol-eval-${generation}`,
        phase: 'Evaluate',
        model: MODEL.mechanical,
        schema: {
          type: 'object',
          properties: {
            compiled: { type: 'boolean' },
            correct: { type: 'boolean' },
            speedup: { type: 'number' },
            metric_name: { type: 'string' },
            result_path: { type: 'string' },
            n_pass: { type: 'number' },
            n_total: { type: 'number' },
            kernel_time_ms: { type: 'number' },
            d_mem: { type: 'number' },
            d_algo: { type: 'number' },
            d_sync: { type: 'number' },
            error_message: { type: 'string' },
            performance_notes: { type: 'string' },
          },
          required: [
            'compiled', 'correct', 'speedup', 'metric_name', 'result_path',
            'n_pass', 'n_total', 'd_mem', 'd_algo', 'd_sync',
          ],
        },
      }), { retries: 5 })
    })()
    : await agentRetry(() => agent(`You are a kernel evaluator for KernelFoundry. Evaluate the already-materialized ${langToken(LEGACY_LANG_TOKEN)} kernel.

# Candidate Path:
${candidatePath}

# Reference Operator:
\`\`\`python
${operatorCode.substring(0, 1500)}
\`\`\`

# Evaluation Steps:
1. **Compile + Correctness**:
${testCommand ? `   Run exactly: ${testCommand}` : `   Inspect and test the candidate conservatively; no measured harness was supplied.`}
2. **Correctness**: Does it produce numerically equivalent output?
   Tolerance: relative precision ν < 0.01 in 99% of outputs.
3. **Canonical Performance**:
${benchmarkCommand
  ? (benchmarkCommand === testCommand
      ? `   The command above is the combined correctness+benchmark harness; run it only once.`
      : `   Run exactly: ${benchmarkCommand}`)
  : `   No benchmark harness was supplied. You may describe an estimate, but return speedup=0.`}
   Read the harness JSON at ${generationResultPath}. The ONLY authoritative
   performance value is its top-level geomean_speedup. Return speedup equal to
   that exact number. NEVER recompute a ratio from mean/reference latency and
   never substitute a self-reported estimate. compiled/correct and n_pass/n_total
   must also come from this same JSON. If measured JSON is absent or invalid,
   return compiled=false, correct=false, speedup=0.

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
        metric_name: { type: 'string' },
        result_path: { type: 'string' },
        n_pass: { type: 'number' },
        n_total: { type: 'number' },
        kernel_time_ms: { type: 'number' },
        d_mem: { type: 'number' },
        d_algo: { type: 'number' },
        d_sync: { type: 'number' },
        error_message: { type: 'string' },
        performance_notes: { type: 'string' },
      },
      required: [
        'compiled', 'correct', 'speedup', 'metric_name', 'result_path',
        'd_mem', 'd_algo', 'd_sync',
      ],
    },
  }), { retries: 5 })

  // Bind the immutable candidate bytes to the exact harness JSON before either
  // can enter the archive.  KerSor independently recomputes these hashes again
  // at acceptance/finalization; this agent produces the evidence, not trust.
  let candidateBinding = {
    verified: false, binding_path: '', binding_sha256: '',
    candidate_sha256: '', measurement_sha256: '',
    task_path: '', task_sha256: '', task_fingerprint_kind: '',
  }
  if (EVIDENCE_MODE === 'measured') {
    const generationBindingPath = `${EXP_DIR}/bindings/gen_${generation}.json`
    const canonicalEval = await agentRetry(() => agent(`Create the immutable
KernelFoundry source-measurement binding for candidate gen${generation}.

Run one small deterministic Python program; do not infer or repair anything:
1. Read ${candidatePath} as bytes and compute SHA-256.
2. Read ${generationResultPath} as bytes, compute SHA-256, then json.load it.
3. Require top-level compiled=true, correct=true, numeric geomean_speedup, and
   numeric n_pass == n_total > 0. Do not calculate a ratio or read a mean.
4. If ${TASK_IDENTITY_PATH || '<none>'} names an existing file, hash its exact
   bytes as task_sha256, record its absolute task_path, and set
   task_fingerprint_kind="file_sha256". Otherwise hash the UTF-8 bytes of the
   exact problem-definition string obtained by json.loads of
   ${JSON.stringify(TASK_SPEC)}; record task_path=null and
   task_fingerprint_kind="inline_problem_definition".
5. Atomically write this exact schema to ${generationBindingPath}:
   schema_version=1, workflow="${WORKFLOW_NAME}", candidate_id="gen${generation}",
   candidate_path (absolute), candidate_sha256, measurement_path (absolute),
   measurement_sha256, task_path/task_sha256, task_fingerprint_kind,
   metric_name="geomean_speedup",
   measurement_metric_field="geomean_speedup", metric_value (the exact JSON
   value), compiled=true, correct=true, n_pass, n_total.
6. Compute the binding file SHA-256. Append one compact JSON line containing
   event="candidate_bound", candidate_id, candidate_sha256, measurement_sha256,
   binding_path, binding_sha256, metric_value, n_pass, n_total to
   ${ARCHIVE_UPDATE_RESULT_PATH}. Create parent directories first.
7. Return the recorded fields. If any requirement fails, do not write a binding
   or archive event and return verified=false, compiled=false, correct=false,
   speedup=0.
`, {
      model: MODEL.mechanical,
      label: `canonical-bind-${generation}`,
      phase: 'Evaluate',
      schema: {
        type: 'object',
        properties: {
          verified: { type: 'boolean' },
          compiled: { type: 'boolean' },
          correct: { type: 'boolean' },
          speedup: { type: 'number' },
          n_pass: { type: 'number' },
          n_total: { type: 'number' },
          metric_name: { type: 'string' },
          result_path: { type: 'string' },
          binding_path: { type: 'string' },
          binding_sha256: { type: 'string' },
          candidate_sha256: { type: 'string' },
          measurement_sha256: { type: 'string' },
          task_path: { type: 'string' },
          task_sha256: { type: 'string' },
          task_fingerprint_kind: { type: 'string' },
        },
        required: [
          'verified', 'compiled', 'correct', 'speedup', 'metric_name',
          'result_path', 'binding_path', 'binding_sha256',
          'candidate_sha256', 'measurement_sha256', 'task_sha256',
          'task_fingerprint_kind',
        ],
      },
    }), { retries: 5 })
    const bindingVerified = (
      canonicalEval.verified === true &&
      /^[0-9a-f]{64}$/i.test(canonicalEval.candidate_sha256 || '') &&
      /^[0-9a-f]{64}$/i.test(canonicalEval.measurement_sha256 || '') &&
      /^[0-9a-f]{64}$/i.test(canonicalEval.binding_sha256 || '') &&
      /^[0-9a-f]{64}$/i.test(canonicalEval.task_sha256 || '') &&
      ['file_sha256', 'inline_problem_definition'].includes(canonicalEval.task_fingerprint_kind) &&
      Boolean(canonicalEval.binding_path)
    )
    evalResult.compiled = bindingVerified && canonicalEval.compiled === true
    evalResult.correct = bindingVerified && canonicalEval.correct === true
    evalResult.speedup = bindingVerified ? Number(canonicalEval.speedup || 0) : 0
    evalResult.n_pass = canonicalEval.n_pass
    evalResult.n_total = canonicalEval.n_total
    evalResult.metric_name = 'geomean_speedup'
    evalResult.result_path = generationResultPath
    candidateBinding = {
      verified: bindingVerified,
      binding_path: canonicalEval.binding_path || generationBindingPath,
      binding_sha256: canonicalEval.binding_sha256 || '',
      candidate_sha256: canonicalEval.candidate_sha256 || '',
      measurement_sha256: canonicalEval.measurement_sha256 || '',
      task_path: canonicalEval.task_path || '',
      task_sha256: canonicalEval.task_sha256 || '',
      task_fingerprint_kind: canonicalEval.task_fingerprint_kind || '',
    }
  }

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
  const candidateSpeedup = evalResult.speedup || 0
  const improvesCell = (
    !existingElite ||
    fitness > existingElite.fitness ||
    (fitness === existingElite.fitness && candidateSpeedup > existingElite.speedup)
  )

  if (improvesCell) {
    archive[cellKey] = {
      code: offspringCode,
      fitness: fitness,
      speedup: candidateSpeedup,
      cell: cellKey,
      id: `gen${generation}`,
      strategy: varyResult?.strategy_description || '',
      compiled: evalResult.compiled,
      correct: evalResult.correct,
      result_path: generationResultPath,
      candidate_path: candidatePath,
      candidate_sha256: candidateBinding.candidate_sha256,
      measurement_sha256: candidateBinding.measurement_sha256,
      binding_path: candidateBinding.binding_path,
      binding_sha256: candidateBinding.binding_sha256,
      task_path: candidateBinding.task_path,
      task_sha256: candidateBinding.task_sha256,
      task_fingerprint_kind: candidateBinding.task_fingerprint_kind,
    }
    outcome = existingElite ? 'improvement' : 'discovery'

    if (
      fitness > globalBest.fitness ||
      (fitness === globalBest.fitness && candidateSpeedup > globalBest.speedup)
    ) {
      globalBest = {
        code: offspringCode,
        fitness,
        speedup: candidateSpeedup,
        cell: cellKey,
        id: `gen${generation}`,
        compiled: evalResult.compiled,
        correct: evalResult.correct,
        result_path: generationResultPath,
        candidate_path: candidatePath,
        candidate_sha256: candidateBinding.candidate_sha256,
        measurement_sha256: candidateBinding.measurement_sha256,
        binding_path: candidateBinding.binding_path,
        binding_sha256: candidateBinding.binding_sha256,
        task_path: candidateBinding.task_path,
        task_sha256: candidateBinding.task_sha256,
        task_fingerprint_kind: candidateBinding.task_fingerprint_kind,
        strategy: varyResult?.strategy_description || '',
      }
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
  // Phase 5.5: Atomic checkpoint + cooperative termination safe point
  // ===========================================================================
  generationsCompleted = generation + 1
  const targetSatisfied = (
    EVIDENCE_MODE === 'measured' &&
    globalBest.compiled === true &&
    globalBest.correct === true &&
    globalBest.speedup >= SPEEDUP_TARGET
  )
  targetStreak = targetSatisfied ? targetStreak + 1 : 0
  const shouldStopOnTarget = (
    STOP_ON_TARGET &&
    generationsCompleted >= MIN_GENERATIONS &&
    targetStreak >= TARGET_PATIENCE
  )
  const plannedTerminationReason = shouldStopOnTarget ? 'speedup_target_reached' : null
  const finalKernelPath = bestKernelPath()
  const bestChanged = Boolean(globalBest.code && globalBest.id !== checkpointedBestId)
  const checkpointPayload = {
    schema_version: 1,
    workflow: WORKFLOW_NAME,
    progress: {
      unit: 'generation',
      completed: generationsCompleted,
      requested: GENERATIONS,
    },
    generation,
    generations_completed: generationsCompleted,
    generations_requested: GENERATIONS,
    compiled: globalBest.compiled === true,
    correct: globalBest.correct === true,
    metric: {
      name: 'geomean_speedup',
      value: globalBest.speedup || 0,
    },
    target: SPEEDUP_TARGET,
    target_met: targetSatisfied,
    target_streak: targetStreak,
    best_candidate_id: globalBest.id || null,
    best_kernel_path: globalBest.code ? finalKernelPath : null,
    measured_candidate_path: globalBest.candidate_path || null,
    measured_candidate_sha256: globalBest.candidate_sha256 || null,
    result_path: globalBest.result_path || null,
    measurement_sha256: globalBest.measurement_sha256 || null,
    artifact_binding_path: globalBest.binding_path || null,
    artifact_binding_sha256: globalBest.binding_sha256 || null,
    task_path: globalBest.task_path || null,
    task_sha256: globalBest.task_sha256 || null,
    task_fingerprint_kind: globalBest.task_fingerprint_kind || null,
    archive_entries: Object.entries(archive).map(([cell, elite]) => ({
      cell,
      candidate_id: elite.id,
      fitness: elite.fitness,
      speedup: elite.speedup,
      compiled: elite.compiled === true,
      correct: elite.correct === true,
      candidate_path: elite.candidate_path || null,
      candidate_sha256: elite.candidate_sha256 || null,
      measurement_path: elite.result_path || null,
      measurement_sha256: elite.measurement_sha256 || null,
      binding_path: elite.binding_path || null,
      binding_sha256: elite.binding_sha256 || null,
      task_fingerprint_kind: elite.task_fingerprint_kind || null,
    })),
    evidence: globalBest.result_path ? {
      kind: 'raw_json',
      compiled_field: 'compiled',
      correct_field: 'correct',
      metric_field: 'geomean_speedup',
    } : {
      kind: 'workflow_verified',
    },
    termination_requested: Boolean(plannedTerminationReason),
    termination_reason: plannedTerminationReason,
  }
  const safePoint = await __workflowRuntimeSafePoint({
    expDir: EXP_DIR,
    checkpointPath: CHECKPOINT_PATH,
    terminationFile: TERMINATION_FILE,
    deadlineEpoch: DEADLINE_EPOCH,
    checkpoint: checkpointPayload,
    bestKernelPath: globalBest.code ? finalKernelPath : null,
    bestKernelSourcePath: globalBest.candidate_path || null,
    bestKernelExpectedSha256: globalBest.candidate_sha256 || null,
    bestKernelCode: globalBest.candidate_path ? null : globalBest.code,
    bestLanguage: fenceToken(),
    materializeBest: bestChanged,
    label: `checkpoint-${generation}`,
    phase: 'Insert',
  })
  if (bestChanged) checkpointedBestId = globalBest.id

  if (safePoint.termination_requested) {
    terminationReason = safePoint.termination_reason || 'supervisor_request'
    log(`  Cooperative stop at generation ${generationsCompleted}: ${terminationReason}`)
    break
  }
  if (shouldStopOnTarget) {
    terminationReason = 'speedup_target_reached'
    log(`  Target ${SPEEDUP_TARGET}x held for ${targetStreak} safe points; stopping after ${generationsCompleted} generations`)
    break
  }

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

let finalReport = ''
if (terminationReason !== 'generation_limit') {
  finalReport = `KernelFoundry stopped at a generation safe point: ${terminationReason}. ` +
    `Completed ${generationsCompleted}/${GENERATIONS} generations; best verified ` +
    `harness geomean speedup ${globalBest.speedup.toFixed(6)}x; ` +
    `archive coverage ${Object.keys(archive).length}/64.`
} else {
  finalReport = await agentRetry(() => agent(`Write a concise technical report on KernelFoundry MAP-Elites optimization.

# Results
- Operation: ${OP_DESC}
- Target: ${langToken(LEGACY_LANG_TOKEN)} on ${TARGET_HW}
- Baseline: ${baselineTime}ms
- Best speedup: ${globalBest.speedup.toFixed(2)}x (cell [${globalBest.cell}])
- Generations completed: ${generationsCompleted}/${GENERATIONS}
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
}

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
  generated_kernel_path: globalBest.code ? bestKernelPath() : '',
  best_candidate_id: globalBest.id || '',
  artifact_binding_required: EVIDENCE_MODE === 'measured',
  artifact_binding_path: globalBest.binding_path || '',
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
  canonical_metric: {
    name: 'geomean_speedup',
    value: globalBest.speedup,
  },
  best_cell: globalBest.cell,
  best_kernel_code: globalBest.code,
  generations: GENERATIONS,
  generations_completed: generationsCompleted,
  termination_reason: terminationReason,
  target_met: (
    EVIDENCE_MODE === 'measured' &&
    globalBest.compiled === true &&
    globalBest.correct === true &&
    globalBest.speedup >= SPEEDUP_TARGET
  ),
  checkpoint_path: CHECKPOINT_PATH,
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
