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
// --- BEGIN sol-execbench-eval substrate (auto-inlined by scripts/patch-sol-execbench-eval.js) ---
const SOL_SOLUTION_CONTRACT = [
  'SOL-EXECBENCH SOLUTION CONTRACT (this task is evaluated by the sol-execbench CLI):',
  '',
  'You are authoring a kernel that will be packaged into a solution.json and run by',
  'the sol-execbench harness, which compiles it internally. Therefore:',
  '',
  '1. Emit a COMPLETE candidate with the task entry point run(...). CUDA C++',
  '   requires a torch PYBIND11_MODULE binding; Python/Triton requires a',
  '   module-level def run(...). Do NOT write a standalone main()/CLI harness.',
  '2. Match the task reference signature exactly (same argument order/dtypes).',
  '3. Do NOT package, compile, or benchmark yourself — the workflow + substrate',
  '   handle pack -> sol-execbench -> parse. Return only the runnable source.',
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

  const pack = `rm -f -- ${__solQ(solutionOut)} && python3 ${__solQ(substrateDir + '/pack_sol_candidate.py')} --kernel ${__solQ(kernelSource)} --contract ${__solQ(contractEnv)} --out ${__solQ(solutionOut)}`
  const clearRunOutputs = [benchOut, normalizedOut].filter(Boolean).map(__solQ).join(' ')
  const run = `rm -f -- ${clearRunOutputs} && test -s ${__solQ(solutionOut)} && cd ${__solQ(seedDir)} && ${env}${ld}CUDA_VISIBLE_DEVICES=${cvd} ${__solQ(solCli)} ${__solQ(taskDir)}${definition} --solution ${__solQ(solutionOut)} --config ${__solQ(benchConfig)} -o ${__solQ(benchOut)}`
  const parse = `test -s ${__solQ(benchOut)} && python3 ${__solQ(substrateDir + '/parse_sol_bench.py')} ${__solQ(benchOut)} --contract ${__solQ(contractEnv)}${normalizedOut ? ` --out ${__solQ(normalizedOut)}` : ''}`

  return {
    pack,
    run,
    parse,
    order: ['pack', 'run', 'parse'],
    cleanupInvariant: 'solution.json + bench.jsonl are per-candidate scratch files in the run dir; each stage clears its own stale outputs and requires the preceding artifact. No project source is mutated (non-mutating method).',
  }
}

async function __solExecbenchEvaluate(ctx) {
  // Claude's legacy Workflow host does not yet expose this optional primitive.
  // Keep the prompt-driven path as a compatibility edge, while KerSor's Host
  // owns exact source materialization and PACK/RUN/PARSE without an LLM turn.
  if (typeof evaluate !== 'function') return null
  return evaluate({
    protocol: 'sol-execbench-v1',
    label: ctx.label || 'sol-eval',
    phase: ctx.phase || 'Evaluate',
    candidatePath: ctx.kernelSource,
    candidateSource: ctx.candidateSource,
    substrateDir: ctx.substrateDir,
    contractEnv: ctx.contractEnv,
    solutionOut: ctx.solutionOut,
    benchOut: ctx.benchOut,
    normalizedOut: ctx.normalizedOut || `${ctx.benchOut}.result.json`,
    solCli: ctx.solCli,
    taskDir: ctx.taskDir,
    benchConfig: ctx.benchConfig,
    seedDir: ctx.seedDir,
    cudaVisibleDevices: ctx.cudaVisibleDevices || '0',
    ldLibraryPath: ctx.ldLibraryPath || '',
    envPrefix: ctx.envPrefix || '',
    definitionPath: ctx.definitionPath || '',
    timeoutSeconds: ctx.timeoutSeconds || 0,
  })
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
// The hard "do not re-propose" constraint is owned by the cumulative transfer
// object, where a failed_strategy stops applying only when a later
// validated_win supersedes it. Deriving it from the previous round alone drops
// a strategy that failed in round 1 and simply was not retried in round 2.
// KerSor emits the cumulative ids as `failed_strategy_ids`; the per-round
// derivation stays as the fallback for a dispatch that predates that channel.
const FAILED_STRATEGY_IDS = Array.isArray(args.failed_strategy_ids)
  ? args.failed_strategy_ids.filter(id => typeof id === 'string' && id)
  : ((ATTEMPT_EVIDENCE && Array.isArray(ATTEMPT_EVIDENCE.transfer_items))
    ? ATTEMPT_EVIDENCE.transfer_items.filter(i => i && i.kind === 'failed_strategy' && i.id).map(i => i.id)
    : [])
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
// Per-turn wall-clock watchdog (parity with CUDAAgent #12/#14). KSearch already
// bounds the *eval* step with EVAL_TIMEOUT_SEC (shell `timeout Ns`), but a hung
// non-eval agent() turn (propose/select/generate stalled in_progress) had no
// wall-clock cap and could stall the search indefinitely. Wrapping the generate
// doer turn bounds it; on expiry the attempt loop breaks (treated like
// stagnation) and the search continues with the next cycle rather than hanging.
const TURN_TIMEOUT_MS = (args.turn_timeout_min || 12) * 60 * 1000  // per-turn wall-clock cap
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
//     seed_candidates: 4,                 // independent seed candidates generated concurrently
//     stagnation_window: 3,              // non-improving attempts before cycle ends
//     max_difficulty: 4,                  // max action difficulty (1-5)
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     test_command: '<correctness/test command — when separate from benchmark>',
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
const _requestedSeedCandidates = Number(args.seed_candidates)
const SEED_CANDIDATES = Math.max(1, Math.min(
  ATTEMPTS_PER_CYCLE,
  Number.isFinite(_requestedSeedCandidates) ? Math.floor(_requestedSeedCandidates) : 4,
))
const STAGNATION_WINDOW = args.stagnation_window || 3
const RUN_STAGNATION_LIMIT = args.run_stagnation_limit || 3  // #31a: consecutive cycles with no global-best improvement -> stop early (parity with CUDAAgent STAGNATION_LIMIT). A cycle where the doer kept failing lands here too, since a failed cycle produces no new global best.
const MAX_DIFFICULTY = args.max_difficulty || 4
const BENCH_CMD = args.benchmark_command || ''
const TEST_CMD = args.test_command || ''
const BASELINE_CODE_PATH = args.kernel_path || ''
const INPUT_MODE = BASELINE_CODE_PATH ? 'optimize_existing' : 'generate_then_optimize'
const RTOL = args.rtol || 0.01
const ATOL = args.atol || 0.01
const EXP_DIR = args.exp_dir || '/tmp/ksearch_exp'
const TERMINATION_FILE = args.termination_file || ''
const DEADLINE_EPOCH = Number(args.deadline_epoch || 0)
const CHECKPOINT_PATH = `${EXP_DIR}/checkpoint.json`
// Wall-clock budget for a single eval attempt (compile + test + benchmark).
// Without this, fattn R1 ran 90+ minutes of correctness rechecks with no
// benchmark latency emitted. The agent prompts surface the budget; the
// embedded paths additionally wrap build/test/benchmark with coreutils
// `timeout` so a runaway command is killed at the shell layer too.
const EVAL_TIMEOUT_SEC = Number.isFinite(args.eval_timeout_sec) && args.eval_timeout_sec > 0
  ? Math.floor(args.eval_timeout_sec)
  : 600
const TIMEOUT_WRAP = `timeout ${EVAL_TIMEOUT_SEC}s`

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
// Embedded eval: prefer a dedicated correctness command. When none is provided,
// fall back to PROJECT_BENCH_CMD for back-compat (current default) but log so
// the operator can see test==benchmark is in effect. The NMSE correctness gate
// only protects you if the test command actually measures correctness, not
// latency-only.
const PROJECT_TEST_CMD = args.test_command || TEST_CMD || PROJECT_BENCH_CMD || ''
if (PROJECT_TEST_CMD && PROJECT_BENCH_CMD && PROJECT_TEST_CMD === PROJECT_BENCH_CMD) {
  log(`KSearch: test_command == benchmark_command (no separate test). NMSE/correctness gate runs the same command as the perf measurement; pass a distinct args.test_command if your harness has a correctness-only path.`)
}
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

// --- BEGIN inlined backend-axis (driver) scaffolding (from _meta/scaffolding/backend-axis.js) ---
function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}
// --- END inlined backend-axis (driver) scaffolding ---

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
  const ext = USE_DRIVER
    ? (DRIVER_SOURCE_EXT || '.py')
    : (LANGUAGE === 'cuda' ? '.cu' : '.py')
  return `${EXP_DIR}/${label}${ext}`
}
function bestKernelPath() {
  const ext = USE_DRIVER
    ? (DRIVER_SOURCE_EXT || '.py')
    : (LANGUAGE === 'cuda' ? '.cu' : '.py')
  return `${EXP_DIR}/best_kernel${ext}`
}

// State
let decisionTree = null
let solutionDb = []
let bestSolution = null
let bestMetric = null
let runStagnation = 0  // #31a: consecutive cycles with no global-best improvement (run-level circuit breaker)
let baselineMetric = null
let specText = ''
let cycleCount = 0
let globalRound = 0
let terminationReason = 'cycle_limit'
let checkpointedBestId = ''

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

// SOL carries its honest reference latency in every official workload row, so a
// separate LLM baseline turn cannot add evidence. Other integrations retain the
// legacy baseline characterization path.
const baselineEval = INTEGRATION_PATTERN === 'sol_execbench_solution'
  ? {
      baseline_metric: 1.0,
      baseline_latency_ms: 0,
      eval_passed: true,
      performance_profile: 'owned by sol-execbench per-workload reference rows',
      bottleneck_analysis: 'deferred to measured candidate feedback',
    }
  : await agentRetry(() => agent(`You are a kernel evaluation expert. Evaluate the baseline kernel to establish reference performance.

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
let INTEGRATION_DECISION = {
  method: INTEGRATION_PATTERN === 'sol_execbench_solution' ? 'sol_execbench_solution' : 'standalone',
  build_fidelity: INTEGRATION_PATTERN === 'sol_execbench_solution' ? 'production' : 'isolated',
  reversible: true,
}
if (INTEGRATION_PATTERN !== 'sol_execbench_solution') {
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true, sol_execbench_cli: !!SOL_CLI })
  const _preferred = INTEGRATION_PATTERN === 'sol_execbench_solution' ? ' --preferred-method sol_execbench_solution' : ''
  const _integ = await agentRetry(() => agent(
    `Read ${INTEG_KERNEL_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${INTEG_KERNEL_PATH}" --can-standalone <yes|no|uncertain>${_preferred} --host-probe '${_probe}' ` +
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
    `${driverSh('run.sh', `--artifact ${buildOut} --problem ${KERNEL_SPEC_PATH} --out ${buildOut}.run.json`)}\n` +
    `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
    { model: MODEL.profile, label: 'driver-run-root', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  let evidenceOut = null
  if (PROFILING_DECISION.method === 'native_profiler') {
    await agentRetry(() => agent(
      `${driverSh('profile.sh', `--artifact ${buildOut} --problem ${KERNEL_SPEC_PATH} --out ${buildOut}.run.json --out ${profOut}`)}\n` +
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
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --source ${kPath} --metrics ${EXP_DIR}/ksearch_root.result.json\`.\n` +
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

if (initResult.decision_tree && typeof initResult.decision_tree === 'object' &&
    Object.keys(initResult.decision_tree).length > 0) {
  decisionTree = initResult.decision_tree
} else {
  // Structured output can satisfy the numeric count fields while returning an
  // empty tree object. Seed a small executable frontier so Select cannot report
  // exhaustion before K-Search has evaluated a single action.
  const fallbackActions = [
    ['fallback-regime-dispatch', 'runtime dispatch and candidate portfolio', 'Implement workload-aware dispatch across small, mid, and large M regimes.'],
    ['fallback-small-m', 'algorithm by M regime', 'Optimize the small-M latency and B-streaming regime without regressing other shapes.'],
    ['fallback-mid-m', 'tensor-core path and tile geometry', 'Tune tensor-core tiling for the enumerated mid-M regime.'],
    ['fallback-large-m', 'memory staging and operand layout', 'Improve large-M tensor-core staging and operand reuse.'],
    ['fallback-tail', 'tail strategy and epilogue', 'Handle irregular M tails with efficient predication and vectorized stores.'],
  ]
  decisionTree = {
    root: {
      node_id: 'root', parent_id: null, node_type: 'decision',
      decision: 'root', choice: 'measured baseline', status: 'solved',
      solution_id: 'baseline', children: fallbackActions.map(([nodeId]) => nodeId),
    },
  }
  fallbackActions.forEach(([nodeId, decision, description], index) => {
    decisionTree[nodeId] = {
      node_id: nodeId, parent_id: 'root', node_type: 'action', decision,
      choice: description, status: 'open', children: [],
      action: {
        title: nodeId.replace(/^fallback-/, '').replace(/-/g, '_'),
        description,
        difficulty_1_to_5: index === 0 ? 2 : 3,
        score_0_to_1: 0.75 - index * 0.05,
        expected_vs_baseline_factor: 1.05,
      },
    }
  })
  log(`Init-tree returned an empty tree; seeded deterministic fallback frontier (${fallbackActions.length} open actions).`)
}
log(`World model initialized: ${Object.keys(decisionTree).length} nodes, ${Object.values(decisionTree).filter(node => node?.status === 'open').length} open actions`)
log(`Dimensions: ${(initResult.design_dimensions || []).join(', ')}`)

// #43: resume from checkpoint if present (mechanical agent reads it verbatim).
// Captures the 5 in-memory state vars (decisionTree, solutionDb, bestSolution,
// bestMetric, cycleCount) so a crashed search resumes at the next cycle instead
// of losing cycles 1..N. Uses the existing mechanical-agent file-IO pattern
// (same as load-driver); zero new sandbox dependencies.
const _checkpoint = await agentRetry(() => agent(
  `Read ${EXP_DIR}/checkpoint.json (if it exists) and return its contents as a parsed JSON object, verbatim.\n` +
  `If the file does not exist, return {"present": false}. Do NOT modify, summarize, or re-encode the JSON.\n` +
  `Return the parsed object.`,
  { model: MODEL.mechanical, label: 'load-checkpoint', phase: 'Setup', schema: JSON_PASSTHROUGH }),
  { retries: 3, allowNull: true })
let _startCycle = 0
if (_checkpoint && _checkpoint.present !== false && _checkpoint.cycle != null) {
  decisionTree = _checkpoint.decisionTree || decisionTree
  solutionDb = _checkpoint.solutionDb || solutionDb
  bestSolution = _checkpoint.bestSolution || bestSolution
  if (_checkpoint.bestMetric != null) bestMetric = _checkpoint.bestMetric
  _startCycle = Math.min(Number(_checkpoint.cycle) || 0, MAX_CYCLES)
  cycleCount = _startCycle
  checkpointedBestId = _checkpoint.best_candidate_id || bestSolution?.id || ''
  log(`RESUMED from checkpoint: cycle ${_startCycle}, best ${bestMetric != null ? bestMetric.toFixed(3) + 'x' : 'N/A'}, ${solutionDb.length} solutions`)
}

// =============================================================================
// Search Cycles — Select → Generate/Improve → Evaluate → Refine/Backtrack
// =============================================================================

for (let cycle = _startCycle; cycle < MAX_CYCLES; cycle++) {
  log(`\n=== Cycle ${cycle + 1}/${MAX_CYCLES} | Best: ${bestMetric?.toFixed(3) || 'N/A'}x | Solutions: ${solutionDb.length} ===`)
  const bestAtCycleStart = bestMetric  // #31a: snapshot global best at cycle start to detect run-level stagnation

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

  if (proposeResult && proposeResult.updated_tree &&
      typeof proposeResult.updated_tree === 'object' &&
      Object.keys(proposeResult.updated_tree).length > 0) {
    decisionTree = proposeResult.updated_tree
  } else if (proposeResult && proposeResult.updated_tree) {
    log('Propose returned an empty tree; preserving the existing decision tree.')
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
        selected_node_id: { type: ['string', 'null'] },
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

  if (!selection || !selection.selected_node_id ||
      ['null', 'search_exhausted'].includes(String(selection.selected_node_id).toLowerCase())) {
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
  let cycleBestPath = null  // AWK #59: path of the cycle-best candidate (authoritative; code is compat)
  let cycleBestEval = null
  let cycleBestScore = -1
  let currentRawCode = null  // tracks the LAST generated code (for debug prompts)
  let noImproveStreak = 0
  let noImproveOverBaseStreak = 0
  let hasPassedInCycle = false

  // The world model and parent are immutable during this fan-out. Only kernel
  // generation is concurrent; evaluation below stays serial so GPU timing and
  // embedded project mutation retain one owner.
  const wmSection =
    `\n\n# World Model (persistent decision tree — use it to guide design):\n${JSON.stringify(decisionTree, null, 2).substring(0, 3000)}` +
    (IS_SOL ? `\n\n${SOL_SOLUTION_CONTRACT}` : '')
  const seedIndexes = Array.from({ length: SEED_CANDIDATES }, (_, seedAttempt) => seedAttempt)
  const generateSeedCandidate = (attempt) => {
      const variantPath = ksearchNodeKernelPath(`cycle_${cycle}_a${attempt}`)  // AWK #58/#59: absolute path the gen agent writes to (survives teardown; driver envelope already reads from this kPath)
      const diversityDirective = SEED_CANDIDATES > 1
        ? `\n\n# Parallel seed branch: ${attempt + 1}/${SEED_CANDIDATES}\nChoose a materially distinct implementation strategy from the other branches while preserving the selected world-model action.`
        : ''
      return withTurnTimeout(agentRetry(() => agent(`You are an expert ${langToken(LANGUAGE)} kernel developer. Generate a high-performance kernel implementing a SPECIFIC optimization action.

# Operation: ${OP_DESC} (${opType})
# Target: ${TARGET_GPU}
# Language: ${langToken(LANGUAGE)}

# Kernel Specification:
${specText.substring(0, 2000)}

# Action to implement: "${selection.action_title}"
${selection.action_description || ''}${diversityDirective}

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
6. PATCH-FIRST / NO-TRUNCATION (AWK #52): emit the kernel from the first line to the LAST closing brace. When parent code exists, edit ONLY the action-relevant spans and preserve the rest verbatim — do NOT rewrite unrelated regions (large whole-file rewrites are the #1 cause of mid-kernel truncation). Do NOT emit a skeleton/stub body. Your output is checked by \`${SUBSTRATE}/code_integrity.py\` — truncated or empty-body output is rejected and the attempt is discarded.
7. NATIVE INTRINSICS FOR THE TARGET ARCH (AWK #53): on Blackwell sm_100 use \`tcgen05.mma\` (+ TMEM) — not Hopper \`wgmma\`/\`mma.async\`; on Hopper sm_90 use \`wgmma\`. See \`${SUBSTRATE}/knowledge/sm100-blackwell.md\` (reference; arch-mismatch gating is enforced vendor-neutrally at the KerSor injection layer, KerSor #70).
8. PERSIST (AWK #58/#59): Write the COMPLETE kernel to ${variantPath} (absolute path — the single source of truth for eval + the driver envelope; \`code\` is a display/compat payload only and may truncate for >20KB kernels). Return variant_path = this path.

Return the complete kernel code + variant_path.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Generate","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle}-a${attempt}","technique":"<the optimization action implemented this round>","speedup":null,"note":"<key design choices made, one line>"}`, {
        label: `gen-${cycle}-${attempt}`,
        phase: 'Generate',
        schema: {
          type: 'object',
          properties: {
            variant_path: { type: 'string' },
            code: { type: 'string' },
            implementation_notes: { type: 'string' },
            design_choices: { type: 'array', items: { type: 'string' } },
          },
          required: ['variant_path', 'code'],
        },
      }), { retries: 5, allowNull: true }), `gen-${cycle}-${attempt}`)
  }
  const seedCandidates = await parallel(seedIndexes.map((seedAttempt) => () => generateSeedCandidate(seedAttempt)))
  log(`Generated ${seedCandidates.filter(Boolean).length}/${SEED_CANDIDATES} seed candidates with bounded parallel fan-out; evaluation remains serial.`)

  for (let attempt = 0; attempt < ATTEMPTS_PER_CYCLE; attempt++) {
    globalRound++

    // Once every independent seed has been measured, continue the dependent
    // chain from the strongest passing seed rather than whichever branch was
    // evaluated last.
    if (attempt === SEED_CANDIDATES && cycleBestCode) {
      currentRawCode = cycleBestCode
    }

    // Determine base_for_debug: whichever of parentCode and cycleBestCode has higher score
    const baseForDebug = (cycleBestCode && cycleBestScore > baseScore) ? cycleBestCode : parentCode
    const baseForDebugLabel = (cycleBestCode && cycleBestScore > baseScore) ? 'cycle_best' : 'parent'

    let genResult

    try {
    if (attempt < SEED_CANDIDATES) {
      genResult = seedCandidates[attempt]
    } else if (!hasPassedInCycle) {
      // Attempts 2+, NO passing solution yet: DEBUG prompt
      // Uses currentRawCode (last attempt's code) as the buggy code to fix
      genResult = await withTurnTimeout(agentRetry(() => agent(`You are an expert ${langToken(LANGUAGE)} kernel developer. The previous attempt has bugs or fails correctness. Debug and fix it.

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
      }), { retries: 5, allowNull: true }), `debug-${cycle}-${attempt}`)
    } else {
      // Attempts 2+, HAVE a passing solution: IMPROVE prompt
      // Focus on performance, not correctness
      genResult = await withTurnTimeout(agentRetry(() => agent(`You are an expert ${langToken(LANGUAGE)} kernel developer. You have a working solution — improve its performance.

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
      }), { retries: 5, allowNull: true }), `improve-${cycle}-${attempt}`)
    }
    } catch (e) {
      log(`  Cycle ${cycle + 1} attempt ${attempt + 1}: turn watchdog tripped — ending cycle (${e && e.message ? e.message : e})`)
      break
    }

    if (!genResult || !genResult.code) continue

    // Track the LAST generated code (used in debug prompts for next attempt)
    currentRawCode = genResult.code

    // =========================================================================
    // Phase: Evaluate
    // =========================================================================
    phase('Evaluate')

    const evalResult = IS_SOL
      ? await (async () => {
        const variant = `ksearch_c${cycle}_a${attempt}`.replace(/[^A-Za-z0-9_]/g, '_')
        const candidatePath = `${EXP_DIR}/${variant}${LANGUAGE === 'cuda' ? '.cu' : '.py'}`
        const normalizedPath = `${EXP_DIR}/${variant}.result.json`
        const evalContext = {
          substrateDir: SOL_SUBSTRATE_DIR,
          kernelSource: candidatePath,
          candidateSource: genResult.code,
          contractEnv: `${EXP_DIR}/contract.env`,
          solutionOut: `${EXP_DIR}/${variant}.solution.json`,
          benchOut: `${EXP_DIR}/${variant}.bench.jsonl`,
          normalizedOut: normalizedPath,
          solCli: SOL_CLI,
          taskDir: SOL_TASK_DIR,
          benchConfig: SOL_BENCH_CONFIG,
          seedDir: SOL_SEED_DIR,
          cudaVisibleDevices: SOL_CVD,
          ldLibraryPath: SOL_LD_LIBRARY_PATH,
          envPrefix: SOL_ENV_PREFIX,
          definitionPath: SOL_DEFINITION_PATH,
          label: `sol-eval-${cycle}-${attempt}`,
          phase: 'Evaluate',
        }
        const direct = await __solExecbenchEvaluate(evalContext)
        if (direct) {
          const nPass = Number(direct.n_pass || 0)
          const nTotal = Number(direct.n_total || 0)
          const valid = direct.compiled === true && direct.correct === true && nTotal > 0 && nPass === nTotal
          return {
            is_valid: valid,
            metric_value: Number(direct.speedup || 0),
            latency_ms: Number(direct.candidate_latency_aggregate_ms || 0),
            speedup_vs_baseline: Number(direct.speedup || 0),
            pass_rate: `${nPass}/${nTotal}`,
            error_log: direct.stderr || '',
            performance_analysis: `host-owned ${direct.protocol} stage=${direct.stage}`,
            remaining_bottleneck: direct.failure_code || '',
          }
        }
        const plan = __solExecbenchEvalPlan(evalContext)
        return agentRetry(() => agent(`Evaluate this K-Search candidate through the authoritative sol-execbench contract.

1. Atomically write the exact candidate below to ${candidatePath}.
\`\`\`${langToken(LANGUAGE)}
${genResult.code}
\`\`\`
2. Run exactly in order:
   PACK: ${plan.pack}
   RUN: ${plan.run}
   PARSE: ${plan.parse}
3. The parse step prints exactly one authoritative line:
   SPEEDUP=<aggregate> REDUCTION=<contract reduction> STATUS=<PASS|FAIL> WORKLOADS=<passed>/<total>
Set is_valid only when STATUS=PASS and passed==total>0. Set metric_value and
speedup_vs_baseline to SPEEDUP. Do not infer missing values or reuse another
candidate's output. ${plan.cleanupInvariant}

Return the parsed result.`, {
          model: MODEL.mechanical,
          label: `sol-eval-${cycle}-${attempt}`,
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
            required: ['is_valid', 'metric_value', 'speedup_vs_baseline', 'pass_rate'],
          },
        }), { retries: 5 })
      })()
      : await agentRetry(() => agent(`You are a kernel evaluation expert. Evaluate this ${langToken(LANGUAGE)} kernel for correctness and performance.

WALL-CLOCK BUDGET: ${EVAL_TIMEOUT_SEC}s for this whole eval attempt (compile + correctness + benchmark combined). If you exceed it on correctness alone with no benchmark latency yet, RETURN EARLY with {is_valid:false, latency_ms:null, metric_value:null, reason:"timeout_in_correctness"} — do not keep retrying. A budget-exceeded attempt is itself useful signal; a 90-minute correctness loop is not.

# Kernel Source (authoritative — Read the FULL kernel from this path; the snippet below is orientation only, AWK #61):
${genResult.variant_path || 'n/a'}

# Kernel Code (orientation snippet):
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
    }), { retries: 5 })

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
        `${driverSh('run.sh', `--artifact ${buildOut} --problem ${KERNEL_SPEC_PATH} --out ${buildOut}.run.json`)}\n` +
        `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
        { model: MODEL.profile, label: `driver-run-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      let evidenceOut
      if (PROFILING_DECISION.method === 'native_profiler') {
        await agentRetry(() => agent(
          `${driverSh('profile.sh', `--artifact ${buildOut} --problem ${KERNEL_SPEC_PATH} --out ${buildOut}.run.json --out ${profOut}`)}\n` +
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
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --source ${kPath} --metrics ${EXP_DIR}/cycle_${cycle}_a${attempt}.result.json\`.\n` +
        `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
        { model: MODEL.mechanical, label: `driver-anti-cheat-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      // AWK #52: code-integrity gate — reject truncated / stub-body kernels before
      // their "speedup" can enter the beam/memory. Catches the 018 (6/6 truncated)
      // and L2-054 (whole-file stub) failure modes that anti_cheat does not see.
      await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/code_integrity.py --source ${kPath}\`.\n` +
        `Return stdout JSON verbatim {valid, flags}. A non-zero exit (valid=false) means the candidate was truncated or is a stub — record it as a failed attempt and do not promote it.`,
        { model: MODEL.mechanical, label: `driver-code-integrity-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
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
          `WALL-CLOCK BUDGET: ${EVAL_TIMEOUT_SEC}s per command — when any single step exceeds the budget, abort the attempt, restore pristine, and return {compiled:false, correct:false, latency_ms:null, heuristic_bclass:"unknown", metrics:{latency_ms:null}, reason:"timeout"}.\n` +
          `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${INTEG_KERNEL_PATH}\n` +
          `2. Apply candidate: cp ${kPath} ${INTEG_KERNEL_PATH}\n3. Build: ${TIMEOUT_WRAP} ${BUILD_CMD}\n4. Test: ${TIMEOUT_WRAP} ${PROJECT_TEST_CMD}\n5. Benchmark: ${TIMEOUT_WRAP} ${PROJECT_BENCH_CMD}\n` +
          `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${INTEG_KERNEL_PATH}\n` +
          `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-inplace-${cycle}-${attempt}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        embLatency = Number(embResult?.latency_ms || 0)
        embBclass = embResult?.heuristic_bclass || 'unknown'
        embMetrics = embResult?.metrics || { latency_ms: embLatency }
      } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
        const _plan = typeof __embeddedEvalPlan === 'function'
          ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: `${TIMEOUT_WRAP} ${BUILD_CMD}`, testCmd: `${TIMEOUT_WRAP} ${PROJECT_TEST_CMD}`, benchmarkCmd: `${TIMEOUT_WRAP} ${PROJECT_BENCH_CMD}` })
          : null
        if (_plan) {
          const embResult = await agentRetry(() => agent(
            `EMBEDDED-DISPATCH EVAL (serial).\n` +
            `WALL-CLOCK BUDGET: ${EVAL_TIMEOUT_SEC}s per command — when any step exceeds the budget, unregister, return {compiled:false, correct:false, latency_ms:null, heuristic_bclass:"unknown", metrics:{latency_ms:null}, reason:"timeout"}.\n` +
            `Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
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
      variant_path: genResult.variant_path || null,  // AWK #59
      eval: evalResult,
      node_id: activeNodeId,
    })

    const roundScore = evalResult.is_valid ? evalResult.metric_value : -1
    const allPassed = evalResult.is_valid

    // Update cycle best (K-Search: only update if passed AND score > cycle_best_score)
    if (allPassed && roundScore > cycleBestScore) {
      cycleBestCode = genResult.code
      cycleBestPath = genResult.variant_path || cycleBestPath  // AWK #59
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

    // Do not abandon already-generated seeds halfway through their serial
    // measurement reduction. After the seed batch, preserve the original dual
    // stagnation rule for dependent debug/improve turns.
    const seedBatchMeasured = attempt + 1 >= SEED_CANDIDATES
    if (seedBatchMeasured &&
        (noImproveStreak >= STAGNATION_WINDOW || noImproveOverBaseStreak >= STAGNATION_WINDOW)) {
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
      bestSolution = {
        id: `cycle_${cycle}_best`,
        code: cycleBestCode,
        path: cycleBestPath,
        eval: cycleBestEval,
        node_id: activeNodeId,
      }
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

    if (refineResult && refineResult.updated_tree &&
        typeof refineResult.updated_tree === 'object' &&
        Object.keys(refineResult.updated_tree).length > 0) {
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

    if (backtrackResult && backtrackResult.updated_tree &&
        typeof backtrackResult.updated_tree === 'object' &&
        Object.keys(backtrackResult.updated_tree).length > 0) {
      decisionTree = backtrackResult.updated_tree
      log(`Backtracked: ${backtrackResult.failure_analysis || 'action too hard'}`)
      log(`Recovery: ${(backtrackResult.recovery_actions || []).join(', ')}`)
    }
  }

  // #31a: Run-level circuit breaker (parity with CUDAAgent STAGNATION_LIMIT). If
  // the global best has not improved for RUN_STAGNATION_LIMIT consecutive cycles,
  // the search has plateaued — stop early instead of burning the remaining
  // MAX_CYCLES budget. (A cycle where the doer kept failing also lands here:
  // it produces no new global best, so bestMetric is unchanged at cycle end.)
  if (bestMetric === bestAtCycleStart) {
    runStagnation++
  } else {
    runStagnation = 0
  }
  cycleCount = cycle + 1
  const plannedStall = runStagnation >= RUN_STAGNATION_LIMIT
  const materializedBestPath = bestSolution?.code ? bestKernelPath() : null
  const bestChanged = Boolean(bestSolution?.code && bestSolution.id !== checkpointedBestId)
  const speedupOverBaseline = (
    bestMetric != null && baselineMetric > 0
      ? bestMetric / baselineMetric
      : 0
  )
  const checkpointPayload = {
    schema_version: 1,
    workflow: WORKFLOW_NAME,
    progress: {
      unit: 'cycle',
      completed: cycleCount,
      requested: MAX_CYCLES,
    },
    cycle: cycleCount,
    cycles_completed: cycleCount,
    cycles_requested: MAX_CYCLES,
    compiled: bestSolution?.eval?.is_valid === true,
    correct: bestSolution?.eval?.is_valid === true,
    metric: {
      name: 'speedup_over_baseline',
      value: speedupOverBaseline,
    },
    baseline_metric: baselineMetric,
    best_metric: bestMetric,
    best_candidate_id: bestSolution?.id || null,
    best_kernel_path: materializedBestPath,
    result_path: bestSolution?.eval?.result_path || null,
    evidence: { kind: 'workflow_verified' },
    termination_requested: plannedStall,
    termination_reason: plannedStall ? 'stalled' : null,
    decisionTree,
    bestMetric,
    bestSolution,
    solutionDb,
    runtime_metadata: {
      checkpoint_written_at: 'cycle-' + cycleCount + '-end',
      workflow: WORKFLOW_NAME,
    },
  }
  const safePoint = await __workflowRuntimeSafePoint({
    expDir: EXP_DIR,
    checkpointPath: CHECKPOINT_PATH,
    terminationFile: TERMINATION_FILE,
    deadlineEpoch: DEADLINE_EPOCH,
    checkpoint: checkpointPayload,
    bestKernelPath: materializedBestPath,
    bestKernelCode: bestSolution?.code || '',
    bestLanguage: fenceToken(),
    materializeBest: bestChanged,
    label: `checkpoint-${cycle}`,
    phase: 'Refine',
  })
  if (bestChanged) checkpointedBestId = bestSolution.id

  if (safePoint.termination_requested && safePoint.termination_reason !== 'stalled') {
    terminationReason = safePoint.termination_reason || 'supervisor_request'
    log(`Cooperative stop after cycle ${cycleCount}: ${terminationReason}`)
    break
  }
  if (plannedStall) {
    terminationReason = 'stalled'
    log(`Run-level stagnation: no global-best improvement for ${runStagnation} consecutive cycles — stopping early (parity: CUDAAgent STAGNATION_LIMIT).`)
    break
  }
}

// =============================================================================
// Final Report
// =============================================================================
phase('Report')

const topSolutions = solutionDb
  .filter(s => s.eval?.is_valid)
  .sort((a, b) => (b.eval.metric_value || 0) - (a.eval.metric_value || 0))
  .slice(0, 5)

let finalReport = ''
if (terminationReason !== 'cycle_limit') {
  finalReport = `KSearch stopped at a cycle safe point: ${terminationReason}. ` +
    `Completed ${cycleCount}/${MAX_CYCLES} cycles; best verified speedup ` +
    `${bestMetric != null && baselineMetric > 0 ? (bestMetric / baselineMetric).toFixed(6) : '0'}x.`
} else {
  finalReport = await agentRetry(() => agent(`Write a concise technical report on this K-Search kernel optimization campaign.

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
}

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
  generated_kernel_path: bestSolution?.code ? bestKernelPath() : '',
  initial_candidates: solutionDb.filter(s => s.cycle === 0),
  initial_generation_result: {
    verified: solutionDb.some(s => s.eval?.is_valid),
    selected_candidate_id: bestSolution?.id || '',
  },
  best_metric: bestMetric,
  best_solution_code: bestSolution?.code || '',
  best_kernel_path: bestSolution?.code ? bestKernelPath() : null,
  cycles_completed: cycleCount,
  termination_reason: terminationReason,
  checkpoint_path: CHECKPOINT_PATH,
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
