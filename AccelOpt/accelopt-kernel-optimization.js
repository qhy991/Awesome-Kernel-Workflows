export const meta = {
  name: 'accelopt-kernel-optimization',
  description: 'Self-improving kernel optimization loop with profiler-driven evidence (AccelOpt methodology; NCU on the cuda backend)',
  whenToUse: 'When you need to iteratively optimize a GPU kernel through plan-execute-profile-learn cycles. Uses the backend driver\'s profiler (e.g. Nsight Compute on cuda) for evidence-based bottleneck classification rather than guessing.',
  phases: [
    { title: 'Setup', detail: 'Read target kernel, build via driver, profile baseline' },
    { title: 'Plan', detail: 'Generate optimization plans guided by profiler data + candidate beam context' },
    { title: 'Execute', detail: 'Implement optimized kernels from each plan' },
    { title: 'Evaluate', detail: 'Profile variants, per-branch dedup, update candidate beam' },
    { title: 'Learn', detail: 'Threshold-filtered slow-fast pairs → reusable patterns (AccelOpt format)' },
    { title: 'Iterate', detail: 'Feed sampled experience + beam state into next optimization round' },
  ],
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
    candidateLanguage: ctx.candidateLanguage || '',
    baselineSolutionPath: ctx.baselineSolutionPath || '',
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
    bindingOut: ctx.bindingOut || '',
    bindingWorkflow: ctx.bindingWorkflow || '',
    candidateId: ctx.candidateId || '',
    timeoutSeconds: ctx.timeoutSeconds || 0,
  }).then(__solGuardHarnessFault)
}

// A `compiled: false` from the evaluator does not always mean the candidate is
// bad.  `invalid_request` and `infrastructure_error` are the harness refusing or
// failing before the candidate was ever built, and callers that map any
// non-success onto compile_error burn refine turns and a stagnation budget on a
// misconfiguration.  Observed: a candidate staged outside the evaluation roots
// was rejected at preflight, reported three times as `compile_error`, and the run
// stopped at the stagnation limit having never compiled anything.  Surface a
// harness fault as a harness fault and stop, because retrying cannot fix it.
function __solGuardHarnessFault(result) {
  const HARNESS_FAULTS = ['invalid_request', 'infrastructure_error']
  if (result && HARNESS_FAULTS.includes(result.failure_code)) {
    const detail = result.stderr || result.stdout || ''
    throw new Error(
      `sol-execbench harness fault (${result.failure_code}) at stage `
      + `${result.stage || 'unknown'}: ${String(detail).slice(0, 400)} `
      + '- this is a harness or wiring fault, not a candidate compile error',
    )
  }
  return result
}

// --- sol-execbench wiring (Host-owned PACK/RUN/PARSE; no LLM turn) -----------
// These 15 workflows declared only the standalone path, so their benchmark was a
// command string embedded in a prompt for a read-only activation with no shell.
// The read-only allowlist deliberately excludes execution, so the agent could
// never run it.  The Host can, and this is the protocol it exposes for exactly
// that.
const SOL_CLI = args.sol_cli || ''
const SOL_TASK_DIR = args.sol_task_dir || ''
const SOL_BENCH_CONFIG = args.sol_bench_config || ''
const SOL_SEED_DIR = args.sol_seed_dir || ''
const SOL_CVD = args.sol_cuda_visible_devices || '0'
const SOL_LD_LIBRARY_PATH = args.sol_ld_library_path || ''
const SOL_ENV_PREFIX = args.sol_env_prefix || ''
const SOL_DEFINITION_PATH = args.sol_definition_path || ''
const SOL_SUBSTRATE_DIR = args.sol_substrate_dir || ''
const SOL_AVAILABLE = Boolean(SOL_CLI && SOL_TASK_DIR && SOL_SUBSTRATE_DIR)

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

const WORKFLOW_NAME = 'accelopt-kernel-optimization'


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
      // Provider refusals and model-identity mismatches are terminal for this
      // request. Repeating a rejected prompt or paying for more calls on the
      // wrong model cannot repair either condition. The message check also
      // protects direct/older Hosts that lack the typed refusal code.
      if (e && (e.code === 'KERSOR_PROVIDER_SAFEGUARD_REFUSAL'
        || e.code === 'KERSOR_CLAUDE_MODEL_IDENTITY_MISMATCH'
        || /safeguards? flagged (?:this|the) message|provider safeguard refusal|(?:can't|cannot) help with this\.\s*Start a new session to continue/i.test(String(e.message || '')))) {
        throw e
      }
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
  let expired = false
  const guard = new Promise((_, reject) => {
    timer = setTimeout(
      () => {
        expired = true
        reject(new Error(`turn-timeout: ${label} exceeded ${Math.round(TURN_TIMEOUT_MS / 1000)}s`))
      },
      TURN_TIMEOUT_MS)
  })
  return Promise.race([promise, guard]).catch(async error => {
    if (expired) {
      // A race does not cancel agent(). Drain this Host-bounded activation so
      // no running call survives the workflow's return.
      try { await promise } catch (_) { /* preserve the guard error */ }
    }
    throw error
  }).finally(() => {
    if (typeof clearTimeout === 'function') clearTimeout(timer)
  })
}
// --- END inlined turn-timeout scaffolding ---
// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.

// Method metadata (eligibility moved to manifest routing.accepts; backend axis
// consumed by resolveBackend()). Kept fields are still referenced below.
const WORKFLOW_META = {
  method_supported_backends: ['cuda', 'triton', 'metax', 'cute-dsl'],
  default_backend: 'cuda',
  requires_capability: { bottleneck_classes: [], metrics: ['dram_pct', 'sm_pct'] },
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


function resolveBackend() {
  const b = args.backend ? normalizeSuitabilityValue(args.backend) : null
  const l = args.language ? normalizeSuitabilityValue(args.language) : null
  if (b && l && b !== l) {
    throw new Error(`Conflicting args: backend="${b}" vs language="${l}". Pass only one.`)
  }
  if (b) return b
  if (l) return l
  const ms = WORKFLOW_META.method_supported_backends
  if (Array.isArray(ms) && ms.length === 1) return normalizeSuitabilityValue(ms[0])
  return WORKFLOW_META.default_backend
}


const BACKEND = resolveBackend()

// =============================================================================
// AccelOpt Self-Improving Kernel Optimization Workflow (NCU-Enhanced, v2)
// =============================================================================
//
// Implements the AccelOpt paper's core loop (MLSys 2026, arXiv:2511.15915):
//   Plan → Execute → Profile → Summarize → Accumulate Experience → Repeat
//
// v2 enhancements aligned with the original AccelOpt system:
//   1. Candidate beam pool (topK kernels carried forward, not just single best)
//   2. Experience pool random sampling with capacity control
//   3. Parameterized selection heuristics (threshold filtering)
//   4. Per-branch deduplication (best sample per plan)
//   5. Experience format aligned with original system
//
// Usage:
//   Workflow({name: 'accelopt-kernel-optimization', args: {
//     kernel_path: '/path/to/kernel.cu',
//     op_description: 'Quantized GEMM Q4_0 weight * FP32 activation',
//     harness_path: '/path/to/harness.cu',
//     harness_build_cmd: '<user-provided harness build command>',
//     harness_run_args: '',
//     kernel_name_regex: 'forward_kernel',
//     ncu_binary: '<user-provided ncu binary path>',
//     exp_dir: '/path/to/experiment/output',
//     iterations: 3,
//     breadth: 3,
//     samples_per_plan: 2,
//     topk_candidates: 3,
//     max_experience_in_prompt: 8,
//     max_threshold: 1.05,
//     min_threshold: 1.05,
//     topk_learn: 5,
//     // Backend-driver wiring (optional; absent → legacy cuda inline-prompt path):
//     backend: 'cuda',
//     backend_dir: '_substrate/backends/cuda',
//     substrate_dir: '_substrate',
//     substrate_command_prefix: 'python3',
//     driver_shell_prefix: '',
//   }})
//
// =============================================================================

let KERNEL_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const OP_DESC = args.op_description || (args.language === 'cute-dsl' ? 'CuTe DSL kernel' : 'CUDA kernel')
const ITERATIONS = args.iterations || 2
const BREADTH = args.breadth || 3
const SAMPLES_PER_PLAN = args.samples_per_plan || 2

// NCU Configuration
const HARNESS_PATH = args.harness_path || ''
const HARNESS_BUILD_CMD = args.harness_build_cmd || ''
const HARNESS_RUN_ARGS = args.harness_run_args || ''
const KERNEL_NAME_REGEX = args.kernel_name_regex || ''
const NCU_BINARY = args.ncu_binary || ''
const EXP_DIR = args.exp_dir || '/tmp/accelopt_exp'
const TERMINATION_FILE = args.termination_file || ''
const DEADLINE_EPOCH = Number(args.deadline_epoch || 0)
const CHECKPOINT_PATH = `${EXP_DIR}/checkpoint.json`

// Fallback: non-NCU profiling commands
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || ''

if (!KERNEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

const LANGUAGE = args.language || 'cuda'
const CUTE_SOL = LANGUAGE === 'cute-dsl' && args.integration_pattern === 'sol_execbench_solution'
if (args.deadline_epoch != null && (!Number.isFinite(DEADLINE_EPOCH) || DEADLINE_EPOCH <= 0)) {
  throw new Error('AccelOpt deadline_epoch must be a positive epoch')
}
if (!CUTE_SOL && (TERMINATION_FILE || DEADLINE_EPOCH > 0)) {
  throw new Error('AccelOpt cooperative wall controls are supported only for CuTe SOL')
}
if (CUTE_SOL && !KERNEL_PATH) {
  throw new Error('AccelOpt CuTe SOL requires an inherited CuTe kernel_path; CUDA seed generation is not applicable')
}
const TARGET_GPU = args.target_gpu || 'unknown GPU'
const SEED_CANDIDATES = args.seed_candidates || 3
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null

// AccelOpt-aligned parameters (v2)
const TOPK_CANDIDATES = args.topk_candidates || 3
const MAX_EXPERIENCE_IN_PROMPT = args.max_experience_in_prompt || 8
const MAX_THRESHOLD = args.max_threshold || 1.05
const MIN_THRESHOLD = args.min_threshold || 1.05
const TOPK_LEARN = args.topk_learn || 5

const SUBSTRATE = args.substrate_dir || '_substrate'
const PY = args.substrate_command_prefix || ''
const SH = args.driver_shell_prefix || ''
const BACKEND_DIR = args.backend_dir || ''
const DRIVER_DIR = BACKEND_DIR || `${SUBSTRATE}/backends/${BACKEND}`
const USE_DRIVER = !!args.backend_dir

// --- Project-native integration (embedded operators via integration-strategist) ---
// For inference-engine embedded operators (e.g. llama.cpp .cuh) the candidate cannot
// compile as a standalone TU; it is built/tested inside the host project. BENCH_CMD
// already exists above for the legacy/standalone benchmark, so the project-native
// benchmark reuses it under a distinct name (PROJECT_BENCH_CMD) to avoid a duplicate
// `const` (which would break wfcheck). Absent these args -> standalone path is unchanged.
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const BUILD_CMD = args.build_command || ''
const PROJECT_BENCH_CMD = args.project_benchmark_command || BENCH_CMD || ''
const REGISTER_SCRIPT = args.register_script || ''

function substrateInstruction(script, cliArgs) {
  const p = `${SUBSTRATE}/${script}`
  return PY ? `Run exactly: \`${PY} ${p} ${cliArgs}\`.`
            : `No substrate_command_prefix for ${p} ${cliArgs}; do not invent an interpreter.`
}
function driverPy(script, cliArgs) {
  const p = `${DRIVER_DIR}/${script}`
  return PY ? `Run exactly: \`${PY} ${p} ${cliArgs}\`.`
            : `No substrate_command_prefix for ${p}; do not invent an interpreter.`
}
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${DRIVER_DIR}/${script} ${cliArgs}\`.`
}
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }
const DRIVER_EXT = '.json'

let IDIOMS = {
  lang_fence: 'cuda',
  impl_requirements:
    'Output a COMPLETE .cu file: all #includes, struct definitions, __global__ kernel(s), forward() wrapper, PYBIND11_MODULE',
  plan_angles: [
    'memory latency hiding: address long_scoreboard stalls via ILP, prefetching, async copies, or software pipelining',
    'memory coalescing and vectorization: fix uncoalesced accesses (sectors/request > 4), use float4/int4 loads',
    'occupancy and parallelism: address SM idle time, tail effects, or low achieved occupancy',
    'compute restructuring: tensor core usage, warp-level reductions, reduced synchronization',
    'data layout and tiling: shared memory staging, bank-conflict-free layouts, double-buffering',
  ],
  read_metric_guide: [
    '- If top stall is "long_scoreboard" (>40%): kernel is MEMORY-LATENCY-BOUND. Add ILP, async loads, or data reuse.',
    '- If top stall is "short_scoreboard" (>30%): heavy shared-mem or dep chains. Shorten chains, add ILP.',
    '- If top stall is "barrier" (>20%): too much __syncthreads. Use warp-level primitives.',
    '- If top stall is "math_pipe_throttle": actually compute-bound — good! Look elsewhere.',
    '- If DRAM throughput > 80%: bandwidth-bound. Reduce bytes read (compression, shared-mem reuse).',
    '- If DRAM throughput < 10% AND long_scoreboard high: latency-bound on L1, not DRAM.',
    '- If sectors/request > 5: uncoalesced access — big optimization opportunity.',
    '- If achieved occupancy << theoretical: stalls prevent filling SM, fix stall source first.',
    '- If waves/SM < 1: grid too small, parallelize more or use persistent kernel.',
    '- If registers/thread > 128: likely register spill — add __launch_bounds__.',
    '- NCU rule suggestions with "Est. Speedup: X%" are surprisingly accurate — prioritize them.',
  ].join('\n'),
  unsupported_methods: [],
}

if (CUTE_SOL) {
  IDIOMS = { ...IDIOMS, lang_fence: 'python',
    impl_requirements: 'Output one complete Python module with from cutlass import cute, @cute.jit kernels, and module-level run(...) matching the reference. Every workload must execute CuTe; no CUDA C++, Triton, pybind, torch.matmul, or placeholder.',
    plan_angles: ['CuTe tile and MMA decomposition', 'coalesced CuTe tensor views', 'shared-memory staging and pipeline depth', 'launch-grid and warp specialization'],
    read_metric_guide: 'Only complete official Host workload latency is measured in this CuTe SOL mode; NCU counters are unavailable. Do not invent stall, occupancy, SM, or DRAM percentages.',
    source_ext: '.py',
  }
}

// State
let experienceMemory = []       // Full pool of learned patterns (grows unbounded)
let lastIterNewPatterns = []    // Patterns discovered in the most recent Learn phase
let bestLatency = null
let bestKernelCode = null
let baselineLatency = null
let baselineNcuProfile = ''
let candidateBeam = []          // [{code, latency, speedup, ncuSummary, planTitle}]
let bottleneckClass = 'unknown'
let solBestHostCandidate = null

function legacyEvaluatePrompt(variant, bestLatency, ncuSetup) {
  return `You are a CUDA kernel evaluator using Nsight Compute. Evaluate this optimized kernel variant.

# Variant: ${variant.id} — Plan: "${variant.plan.title}"
# NCU Evidence for this plan: ${variant.plan.ncu_evidence}

# Kernel Code:
\`\`\`cuda
${variant.code.substring(0, 4000)}
\`\`\`

# Baseline Performance:
- Latency: ${bestLatency}ms
- Top stall: ${ncuSetup.top_stall_reason || 'unknown'}

# Evaluation Steps:

## Step 1: Static correctness check
- Race conditions? (check __syncthreads placement, shared-mem access patterns)
- Out-of-bounds? (check index computations against dimensions)
- Missing synchronization? (writes to shared followed by reads without barrier)
- Incorrect reductions? (warp shuffle masks, final-warp logic)

## Step 2: Compilability check
- All #includes present? (torch/extension.h, cuda_runtime.h, cuda_fp16.h, etc.)
- Valid CUDA syntax? (correct use of __global__, __shared__, __device__)
- PYBIND11_MODULE present?

## Step 3: Build and profile (if environment allows)
Use only the user-provided harness_build_cmd, harness_path/run args, and ncu_binary contract. If any required command is missing, do not invent one; return static correctness/compilability analysis and mark NCU fields as missing evidence.

## Step 4: Compare with baseline
Calculate speedup = baseline_latency / variant_latency.
Note which NCU metrics improved and which degraded.

If NCU is not available, provide your expert static analysis:
- Did the optimization address the identified bottleneck (${ncuSetup.top_stall_reason})?
- Would you expect sectors/request to decrease?
- Would occupancy change?
- Estimate speedup based on the targeted inefficiency.

Return evaluation results.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (variant ${variant.id}; status="done" if correct AND compilable, else "error"; speedup is your estimated_speedup number or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"${variant.id}","technique":"${variant.plan.title}","speedup":<number or null>,"note":"<correct? compilable? bottleneck addressed? or the failure reason>"}`
}

function legacyIterProfile(iter, candidateBeam, bestResult, bestLatency, baselineLatency, baselineNcuProfile) {
  return `
## NCU Profile Results (After Iteration ${iter + 1} — Best: "${candidateBeam[0].planTitle}")
- Latency: ${bestLatency}ms (${(baselineLatency / bestLatency).toFixed(2)}x speedup vs original)
- Bottleneck addressed: ${bestResult.evaluation.bottleneck_addressed ? 'YES' : 'NO'}
- New bottleneck: ${bestResult.evaluation.new_bottleneck || 'unknown'}
- Comparison: ${bestResult.evaluation.ncu_comparison}

Previous profile data for reference:
${baselineNcuProfile}`
}

function cuteIterEvidence(iter, candidateBeam, bestLatency, baselineLatency, priorEvidence) {
  return `
## Host full-workload latency (after iteration ${iter + 1})
- Best source: ${candidateBeam[0].planTitle}
- Latency: ${bestLatency}ms (${(baselineLatency / bestLatency).toFixed(2)}x vs the inherited CuTe seed)
- Profiler counters: unavailable; no NCU or IKET attribution

Prior measured evidence:
${priorEvidence}`
}

function cuteEvaluatePrompt(variant, bestLatency) {
  return `Evaluate this CuTe DSL Python module as a source-level hypothesis only.

# Candidate: ${variant.id}; plan: ${variant.plan.title}
# Current Host-measured latency: ${bestLatency}ms
# Candidate source:
\`\`\`python
${variant.code.substring(0, 4000)}
\`\`\`

Check likely correctness, CuTe syntax, entrypoint run(...), and the intended transformation. Do not claim compilation, correctness, latency, speedup, NCU counters, or IKET counters from this inspection. The separate official Host evaluation decides those results. Return a qualitative assessment with estimated_speedup=1 and estimated_latency_ms omitted.`
}

function cuteLearnPrompt(pair) {
  return `Analyze this measured slow-fast CuTe DSL pair for one reusable source-level optimization rule.

# Slow source:
\`\`\`python
${pair.slow.substring(0, 2500)}
\`\`\`
# Fast source:
\`\`\`python
${pair.fast.substring(0, 2500)}
\`\`\`
# Official Host full-workload speedup: ${pair.speedup.toFixed(2)}x
# Plan: ${pair.plan_title}

The available evidence is the two sources and their Host latency. No profiler counters were collected. State a source-structure trigger and a rule. Do not infer a hardware bottleneck or invent NCU or IKET evidence. Return {title,source_trigger,rule,original_snippet,optimized_snippet,why,is_antipattern}.`
}

function cuteFinalReportPrompt(op, baselineLatency, bestLatency, beam, experience, evidence, source) {
  return `Write a concise report for the CuTe DSL Host-latency adaptation of AccelOpt.

Operation: ${op}
Inherited CuTe seed latency: ${baselineLatency}ms
Best Host full-workload latency: ${bestLatency}ms
Measured speedup over CuTe seed: ${(baselineLatency / bestLatency).toFixed(2)}x
Candidate beam: ${beam.length}; source-level experience rules: ${experience.length}
Evidence: ${evidence}

Best CuTe source:
\`\`\`python
${source.substring(0, 3000)}
\`\`\`

Describe the source changes and measured outcomes. Explicitly say no NCU or IKET counters were collected. Do not attribute the speedup to an unmeasured hardware bottleneck.`
}

function legacyFinalReportPrompt(OP_DESC, opType, baselineLatency, bestLatency, ITERATIONS, candidateBeam, experienceMemory, baselineNcuProfile, bestKernelCode) {
  return `Write a concise technical optimization report.

# AccelOpt + NCU Optimization Results
- Operation: ${OP_DESC} (${opType})
- Baseline Latency: ${baselineLatency}ms
- Final Best Latency: ${bestLatency}ms
- Overall Speedup: ${(baselineLatency / bestLatency).toFixed(2)}x
- Iterations: ${ITERATIONS}
- Candidate Beam (final): ${candidateBeam.length} kernels
- Experience Patterns: ${experienceMemory.length}

# Initial NCU Diagnosis:
${baselineNcuProfile.substring(0, 1000)}

# Final Candidate Beam:
${candidateBeam.map((c, i) => `${i + 1}. "${c.planTitle}" — ${c.speedup.toFixed(2)}x (${c.latency.toFixed(3)}ms)`).join('\n')}

# Learned Optimization Knowledge Base:
${experienceMemory.map((e, i) => `${i + 1}. ${e}`).join('\n\n')}

# Final Kernel:
\`\`\`cuda
${bestKernelCode.substring(0, 3000)}
\`\`\`

Write:
1. NCU-driven optimization journey (what metrics → what actions → what results)
2. Which NCU patterns reliably predicted optimization opportunities
3. Anti-patterns: what NCU data looked promising but the optimization failed
4. Candidate beam evolution: how the population of solutions evolved
5. Remaining bottlenecks (what NCU shows for the final kernel)
6. Recommendations for further optimization with specific NCU metrics to target`
}

function legacyLearnPrompt(pair) {
  return `You are a CUDA optimization expert with NCU profiling expertise. Analyze this slow-fast kernel pair and extract a GENERAL, REUSABLE optimization insight.

# Slow Kernel:
\`\`\`cuda
${pair.slow.substring(0, 2500)}
\`\`\`

# Fast Kernel:
\`\`\`cuda
${pair.fast.substring(0, 2500)}
\`\`\`

# Speedup: ${pair.speedup.toFixed(2)}x
# This is a ${pair.type === 'positive' ? 'POSITIVE example (do this)' : 'NEGATIVE example (avoid this)'}

# NCU Evidence that motivated this optimization:
${pair.ncu_evidence || 'N/A'}

# NCU Metric Comparison (before vs after):
${pair.ncu_comparison || 'N/A'}

# Was the targeted bottleneck addressed? ${pair.bottleneck_addressed ? 'YES' : 'NO/UNKNOWN'}

## Your task:
Extract a GENERAL optimization rule following this EXACT format:

**{Short title}**
NCU trigger: {what metric/stall pattern signals this opportunity}
Rule: {one sentence — when you see X in NCU, do Y to the code}
Original code:
\`\`\`cuda
{2-5 lines of slow pattern}
\`\`\`
Optimized code:
\`\`\`cuda
{2-5 lines of fast pattern}
\`\`\`
Why: {hardware-level explanation}

Make the rule GENERAL enough to apply to other kernels (not specific to this one kernel).

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (pair "${pair.plan_title}", a ${pair.type} example at ${pair.speedup.toFixed(2)}x):
{"workflow":"${WORKFLOW_NAME}","phase":"Learn","ts":"<ts>","status":"done","candidate_id":"learn-${pair.plan_title}","technique":"<your extracted rule title>","speedup":${pair.speedup.toFixed(2)},"note":"<the general reusable rule you extracted, one line>"}`
}

function legacySetupReadPrompt() {
  return `Read the ${CUTE_SOL ? 'CuTe DSL Python' : 'CUDA'} kernel file at: ${KERNEL_PATH}

Analyze it and return a JSON object with:
- kernel_code: the full source code
- op_type: operation type (e.g., "quantized_gemm", "attention", "rmsnorm", "softmax")
- key_functions: list of key function names (especially ${CUTE_SOL ? '@cute.jit kernels and module-level run' : '__global__ kernels'})
- current_approach: brief description of the implementation strategy
- launch_config: if visible, the grid/block dimensions used
- shared_memory_usage: whether and how shared memory is used
- memory_access_patterns: description of global memory access patterns

Return ONLY the JSON object.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"baseline_read_analysis","speedup":null,"note":"<op_type + current approach + memory access pattern, one line>"}`
}

function driverSetupReadPrompt() {
  return `Read the ${BACKEND} kernel source file at: ${KERNEL_PATH}

Analyze it and return a JSON object with:
- kernel_code: the full source code
- op_type: operation type (e.g., "quantized_gemm", "attention", "rmsnorm", "softmax")
- key_functions: list of key function names (entry kernels (the backend's launch entrypoints))
- current_approach: brief description of the implementation strategy
- launch_config: if visible, the grid/block dimensions used
- shared_memory_usage: whether and how shared memory is used
- memory_access_patterns: description of global memory access patterns

Return ONLY the JSON object.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"baseline_read_analysis","speedup":null,"note":"<op_type + current approach + memory access pattern, one line>"}`
}

function legacyGenerateSeedPrompt() {
  return `No kernel_path was provided. Generate and verify an initial CUDA kernel before starting AccelOpt.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- op_description: ${OP_DESC}
- language: ${LANGUAGE}
- target_gpu: ${TARGET_GPU}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- test_command: ${TEST_CMD || '(not provided)'}
- benchmark_command: ${BENCH_CMD || '(not provided)'}

# Contract
1. If problem_path is provided, read it first.
2. Generate ${SEED_CANDIDATES} complete CUDA kernel candidates.
3. Materialize candidates under ${EXP_DIR}/generated/.
4. Run available commands with {kernel_path} and {result_path} substitutions.
5. Select the best candidate that compiles and passes correctness. If no real evaluator is available, select the strongest candidate and mark verified=false.
6. Return generated_kernel_path plus candidate and evidence metadata.`
}

function driverGenerateSeedPrompt() {
  return `No kernel_path was provided. Generate and verify an initial ${BACKEND} kernel before starting AccelOpt.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- op_description: ${OP_DESC}
- backend: ${BACKEND}
- target_gpu: ${TARGET_GPU}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- test_command: ${TEST_CMD || '(not provided)'}
- benchmark_command: ${BENCH_CMD || '(not provided)'}

# Contract
1. If problem_path is provided, read it first.
2. Generate ${SEED_CANDIDATES} complete ${BACKEND} kernel candidates.
3. Materialize complete kernels honoring: ${IDIOMS.impl_requirements}
4. Materialize candidates under ${EXP_DIR}/generated/.
5. Run available commands with {kernel_path} and {result_path} substitutions.
6. Select the best candidate that compiles and passes correctness. If no real evaluator is available, select the strongest candidate and mark verified=false.
7. Return generated_kernel_path plus candidate and evidence metadata.`
}

async function resolveInitialKernelFromProblem() {
  if (INPUT_MODE !== 'generate_then_optimize') return ''

  const generated = await agentRetry(() => agent(USE_DRIVER ? driverGenerateSeedPrompt() : legacyGenerateSeedPrompt(), {
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
  if ((TEST_CMD || BENCH_CMD) && initialGenerationResult.verified === false) {
    throw new Error('No generated seed passed correctness evidence')
  }
  return generatedKernelPath
}

// Helper: sample n items from array without replacement (Fisher-Yates partial)
function sampleWithoutReplacement(arr, n) {
  if (n >= arr.length) return [...arr]
  const copy = [...arr]
  const result = []
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor((copy.length - i) * (i / (i + 1 + copy.length)))
    // Deterministic pseudo-shuffle using index-based swap
    const swapIdx = (i * 7 + 3) % (copy.length - i) + i
    const temp = copy[i]
    copy[i] = copy[swapIdx]
    copy[swapIdx] = temp
    result.push(copy[i])
  }
  return result
}

// Helper: construct experience section for planner prompt (AccelOpt sampling logic)
function legacyExecutePrompt(bestKernelCode, plan, sampleIdx, SAMPLES_PER_PLAN) {
  return `You are an expert CUDA kernel developer. Implement this NCU-informed optimization plan as a complete, compilable kernel.

# Original Kernel:
\`\`\`cuda
${bestKernelCode.substring(0, 4000)}
\`\`\`

# Optimization Plan: "${plan.title}"
NCU Evidence: ${plan.ncu_evidence}
Plan: ${plan.plan}

# Requirements:
1. Output a COMPLETE .cu file: all #includes, struct definitions, __global__ kernel(s), forward() wrapper, PYBIND11_MODULE
2. Must be FUNCTIONALLY CORRECT (same output as baseline within FP tolerance)
3. Apply the plan faithfully — the plan is based on real NCU data, so the optimization targets a real bottleneck
4. Keep the forward() function signature unchanged
5. MUST compile with -lineinfo (don't use features that break debug info)
6. This is variant ${sampleIdx + 1}/${SAMPLES_PER_PLAN}

Return the complete CUDA code.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (plan "${plan.title}", sample ${sampleIdx}):
{"workflow":"${WORKFLOW_NAME}","phase":"Execute","ts":"<ts>","status":"done","candidate_id":"${plan.title}-v${sampleIdx}","technique":"${plan.title}","speedup":null,"note":"<the concrete code transformation you implemented, one line>"}`
}

function buildExperienceSection(experienceMemory, lastIterNewPatterns, maxInPrompt) {
  if (experienceMemory.length === 0) return ''

  let selected = []
  // Priority: new patterns from last iteration (like original_rewrite_list in AccelOpt)
  const newPatterns = [...lastIterNewPatterns]
  if (newPatterns.length >= maxInPrompt) {
    selected = newPatterns.slice(0, maxInPrompt)
  } else {
    selected = [...newPatterns]
    const remaining = maxInPrompt - selected.length
    // Fill remaining slots with random samples from the full pool (excluding already-selected)
    const pool = experienceMemory.filter(e => !newPatterns.includes(e))
    const sampled = sampleWithoutReplacement(pool, remaining)
    selected = selected.concat(sampled)
  }

  return `\n\n# Learned Optimization Patterns (${selected.length}/${experienceMemory.length} sampled)\n${selected.map((e, i) => `${i + 1}. ${e}`).join('\n\n')}`
}

// Helper: format candidate beam info for planner prompt
function buildBeamSection(candidateBeam, fence, evidenceLabel = 'NCU') {
  if (candidateBeam.length <= 1) return ''
  return `\n\n# Candidate Beam (top-${candidateBeam.length} kernels from previous iterations)\n${candidateBeam.map((c, i) => `## Candidate ${i + 1}: "${c.planTitle}" — ${c.speedup.toFixed(2)}x, ${c.latency.toFixed(3)}ms\n${evidenceLabel}: ${c.ncuSummary || 'N/A'}\n\`\`\`${fence}\n${c.code.substring(0, 1500)}\n\`\`\``).join('\n\n')}`
}

// =============================================================================
// Phase 1: Setup — Read kernel, build harness, NCU profile baseline
// =============================================================================
phase('Setup')

if (USE_DRIVER) {
  const driver = await agentRetry(() => agent(
    `Load the backend driver for backend="${BACKEND}".\n` +
    `1. Run exactly: \`cat ${DRIVER_DIR}/manifest${DRIVER_EXT}\` and parse JSON.\n` +
    `2. Run exactly: \`cat ${DRIVER_DIR}/idioms${DRIVER_EXT}\` and parse JSON.\n` +
    `If either is missing, return {present:false, reason:"no driver for backend ${BACKEND}"}.\n` +
    `Also compare manifest.capabilities against the required capability floor ` +
    `${JSON.stringify(WORKFLOW_META.requires_capability)};\n` +
    `if a required metric/class is missing return {present:true, capability_ok:false, missing:[...]}.\n` +
    `Return {present, capability_ok, missing, backend_id, source_ext, lang_fence, hw_vendor,\n` +
    `  profiler_name|null, profiler_format, capability_metrics, supported_classes, problem_types,\n` +
    `  requires_tools, impl_requirements, read_metric_guide,\n` +
    `  plan_angles:[...], unsupported_methods:[...],\n` +
    `  idioms:{<method>:{idiom,prompt_guidance}}}.`,
    { model: MODEL.mechanical, label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })

  if (!driver.present) {
    throw new Error(`No backend driver present for backend="${BACKEND}". Provide ${DRIVER_DIR}/ or pick a supported backend.`)
  }
  if (driver.capability_ok === false) {
    throw new Error(`backend="${BACKEND}" lacks required capability: ${(driver.missing || []).join(', ')}.`)
  }
  IDIOMS = {
    lang_fence: driver.lang_fence || IDIOMS.lang_fence,
    impl_requirements: driver.impl_requirements || IDIOMS.impl_requirements,
    plan_angles: (driver.plan_angles && driver.plan_angles.length) ? driver.plan_angles : IDIOMS.plan_angles,
    read_metric_guide: driver.read_metric_guide || IDIOMS.read_metric_guide,
    unsupported_methods: driver.unsupported_methods || [],
    profiler_name: driver.profiler_name || null,
    profiler_format: driver.profiler_format || '',
    source_ext: driver.source_ext || '.cu',
    ...(driver.idioms || {}),
  }
  log(`Driver loaded: ${BACKEND} (fence=${IDIOMS.lang_fence}, profiler=${IDIOMS.profiler_name || 'none'})`)
}

if (INPUT_MODE === 'generate_then_optimize') {
  KERNEL_PATH = await resolveInitialKernelFromProblem()
}

const setupResult = await agentRetry(() => agent(USE_DRIVER ? driverSetupReadPrompt() : legacySetupReadPrompt(), {
  label: 'read-baseline',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      kernel_code: { type: 'string' },
      op_type: { type: 'string' },
      key_functions: { type: 'array', items: { type: 'string' } },
      current_approach: { type: 'string' },
      launch_config: { type: 'string' },
      shared_memory_usage: { type: 'string' },
      memory_access_patterns: { type: 'string' },
    },
    required: ['kernel_code', 'op_type', 'key_functions', 'current_approach'],
  },
}), { retries: 5 })

const baselineKernel = setupResult.kernel_code
const opType = setupResult.op_type
log(`Baseline: ${opType}, kernels: ${setupResult.key_functions.join(', ')}`)

// NCU Profile the baseline
function legacyNcuBaselinePrompt(baselineKernel) {
  return `You are a CUDA profiling expert using Nsight Compute (ncu). Set up and run profiling for the baseline kernel only through the user-provided harness/profiler contract.

# Environment
- NCU binary: ${NCU_BINARY || '(not provided)'}
- Experiment directory: ${EXP_DIR}
- Kernel file: ${KERNEL_PATH}
- Kernel name regex for ncu -k: ${KERNEL_NAME_REGEX || '(auto-detect from kernel file)'}
- Harness path: ${HARNESS_PATH || '(not provided)'}
- Harness build command: ${HARNESS_BUILD_CMD || '(not provided)'}
- Harness run args: ${HARNESS_RUN_ARGS}

# Kernel Source:
\`\`\`cuda
${baselineKernel.substring(0, 4000)}
\`\`\`

# Instructions

## Step 1: Create run directory
\`\`\`bash
mkdir -p ${EXP_DIR}/baseline/{harness,reports,analysis}
\`\`\`

## Step 2: Build the profiling harness
If harness_path and harness_build_cmd are provided, use them exactly. If either is missing, do not invent a compiler or standalone harness; set ncu_available=false and explain the missing contract.

## Step 3: Run profiling
If ncu_binary is provided together with a runnable harness contract, run profiling using that user-provided contract and write reports under ${EXP_DIR}/baseline/reports/. Do not substitute a default compiler, default benchmark binary, or default command line.

## Step 4: Extract key metrics
Read the details page and the report to extract:
- gpu__time_duration.sum (kernel duration)
- sm__throughput.avg.pct_of_peak_sustained_elapsed
- dram__bytes_read.sum.pct_of_peak_sustained_elapsed
- sm__warps_active.avg.pct_of_peak_sustained_active (achieved occupancy)
- sm__maximum_warps_per_active_cycle_pct (theoretical occupancy)
- launch__waves_per_multiprocessor
- launch__registers_per_thread
- Top stall reasons (long_scoreboard, short_scoreboard, wait, barrier, etc.)
- Sectors/request for global loads
- L1/L2 hit rates
- NCU rule suggestions with Est. Speedup percentages

Execute only the steps backed by provided commands/artifacts. If ncu or the harness contract is unavailable, fall back to static code analysis and mark measured fields as missing.

Return a structured profile result.`
}
const LEGACY_NCU_BASELINE_SCHEMA = {
  type: 'object',
  properties: {
    latency_ms: { type: 'number' },
    sm_throughput_pct: { type: 'number' },
    dram_throughput_pct: { type: 'number' },
    achieved_occupancy_pct: { type: 'number' },
    theoretical_occupancy_pct: { type: 'number' },
    waves_per_sm: { type: 'number' },
    registers_per_thread: { type: 'number' },
    top_stall_reason: { type: 'string' },
    top_stall_pct: { type: 'number' },
    sectors_per_request: { type: 'number' },
    l1_hit_rate_pct: { type: 'number' },
    l2_hit_rate_pct: { type: 'number' },
    ncu_rule_suggestions: { type: 'array', items: { type: 'string' } },
    bottleneck_diagnosis: { type: 'string' },
    profile_summary: { type: 'string' },
    ncu_available: { type: 'boolean' },
  },
  required: ['latency_ms', 'bottleneck_diagnosis', 'profile_summary'],
}

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it below. The agent only classifies the task (fuzzy); the substrate stamps
// confidence by method (measured/inferred/hypothesized) -- not the agent. See
// _substrate/profiling/README.md. Falls back to native_profiler if undecided. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }
if (USE_DRIVER) {
  const _pd = await agentRetry(() => agent(
    `Read ${KERNEL_PATH}; classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    substrateInstruction('profiling/profiling_strategist.py',
      `resolve --backend-manifest ${DRIVER_DIR}/manifest${DRIVER_EXT} --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl`) +
    ` Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}

// --- integration-strategist: route build/test mode (standalone vs embedded_*).
// For an inference-engine embedded operator (e.g. llama.cpp .cuh referenced via
// kernel_path) can_compile_standalone=no, so the candidate is built/tested INSIDE the
// host project rather than as an isolated TU. The standalone path stays byte-identical. ---
// Honour an explicit caller declaration.  Without this the guard below only stops
// the model from changing the decision; the declaration itself still had no effect,
// so a caller asking for sol_execbench_solution silently got standalone.
let INTEGRATION_DECISION = args.integration_pattern === 'sol_execbench_solution'
  ? { method: 'sol_execbench_solution', build_fidelity: 'production', reversible: true }
  : { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _profManifest = (USE_DRIVER && BACKEND_DIR) ? `${BACKEND_DIR}/manifest${DRIVER_EXT}` : `${SUBSTRATE}/backends/cuda/manifest.json`
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Read ${KERNEL_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
    `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  // A caller that declared `integration_pattern` has already made this decision;
  // re-deciding it from an unvalidated model reply is how an explicit instruction
  // gets silently discarded.  Measured on B300: KDA was given
  // integration_pattern=sol_execbench_solution, ran the strategist anyway, adopted
  // the reply and logged `integration method = null`, which switched off the
  // deterministic sol-execbench path for the whole run.  Adopt only when the caller
  // said nothing, and only a method from the known set.
  if (!args.integration_pattern && _integ && ['standalone', 'embedded_inplace', 'embedded_dispatch', 'sol_execbench_solution', 'derive_adapter'].includes(_integ.method)) {
    INTEGRATION_DECISION = _integ
  }
}
log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error('integration-strategist returned derive_adapter — provide project_root + build/test commands')
}
const USE_DRIVER_STANDALONE = USE_DRIVER && INTEGRATION_DECISION.method === 'standalone'
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
const IS_SOL = INTEGRATION_DECISION.method === 'sol_execbench_solution'
if (CUTE_SOL) PROFILING_DECISION = { method: 'static', confidence: 'hypothesized',
  normalizer: null, profiler_name: null, rationale: 'CuTe SOL Host measures latency, not profiler counters' }
// The embedded operator file we swap in place is the project-referenced KERNEL_PATH.
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// A-O1 closure: native_profiler chosen but ncu unavailable, OR the embedded path has no
// reachable native profiler -> downgrade to perf_heuristic (run/bench gives throughput).
// On the driver-standalone path the native profiler is the backend's own tool
// (mcProfiler/msprof/rocprofv3/ncu via profile.sh), advertised as IDIOMS.profiler_name
// — NOT the NVIDIA ncu binary. Only the legacy non-driver path needs NCU_BINARY.
const NATIVE_PROFILER_REACHABLE = USE_DRIVER_STANDALONE ? !!IDIOMS.profiler_name : !!NCU_BINARY
if (PROFILING_DECISION.method === 'native_profiler' && (!NATIVE_PROFILER_REACHABLE || (IS_EMBEDDED && !USE_DRIVER_STANDALONE))) {
  log(`profiling: native_profiler unreachable (reachable=${NATIVE_PROFILER_REACHABLE}, profiler=${IDIOMS.profiler_name || 'none'}, ncu_binary=${!!NCU_BINARY}, embedded=${IS_EMBEDDED}) -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'project-native-perf', rationale: 'native_profiler unreachable -> perf_heuristic' }
}

let ncuSetup
if (USE_DRIVER_STANDALONE) {
  const profileResult = await agentRetry(() => agent(
    (IDIOMS.profiler_name && PROFILING_DECISION.method === 'native_profiler')
      ? `Profile the baseline kernel via the backend driver and normalize to canonical metrics.\n` +
        `Kernel: ${KERNEL_PATH}. Experiment dir: ${EXP_DIR}/baseline.\n` +
        `1. ` + driverSh('build.sh', `--source ${KERNEL_PATH} --out ${EXP_DIR}/baseline/artifact ${HARNESS_BUILD_CMD ? `--build-cmd "${HARNESS_BUILD_CMD}"` : ''}`) + `\n` +
        `2. ` + driverSh('profile.sh', `--artifact ${EXP_DIR}/baseline/artifact --problem ${PROBLEM_PATH || PROBLEM_DEFINITION || KERNEL_PATH} --out ${EXP_DIR}/baseline/prof.native`) + `\n` +
        `3. ` + driverPy('to_evidence.py', `--native ${EXP_DIR}/baseline/prof.native --format ${IDIOMS.profiler_format}`) + `\n` +
        `Return its stdout JSON verbatim: {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy,...}, coverage:[...], source_backend}. ` +
        `If the profiler exits 4 (unavailable), return {ok:true, metrics:{latency_ms:null,dram_pct:null,sm_pct:null,occupancy:null}, coverage:[], profiler_available:false}.`
      : `Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run ${IDIOMS.profiler_name || 'a native profiler'}. ` +
        `Build via ` + driverSh('build.sh', `--source ${KERNEL_PATH} --out ${EXP_DIR}/baseline/artifact`) + ` then ` +
        driverSh('run.sh', `--artifact ${EXP_DIR}/baseline/artifact --problem ${PROBLEM_PATH || KERNEL_PATH} --out ${EXP_DIR}/baseline/result.json`) + ` to get throughput (latency_ms, and GFLOPS/GB-s if the harness reports them). ` +
        `If method='perf_heuristic', normalize that throughput into canonical metrics via ` +
        substrateInstruction('profiling/' + (PROFILING_DECISION.normalizer || 'perf_to_evidence.py'), `--baseline ${EXP_DIR}/baseline/result.json --peak-gflops <device_peak_gflops> --peak-gbs <device_peak_gbs>`) +
        ` Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
        `Return {ok:true, metrics:{latency_ms:<from run.sh>,dram_pct:<from perf or null>,sm_pct:<from perf or null>,occupancy:null}, coverage:[...], profiler_available:false}.`,
    { model: MODEL.profile, label: 'ncu-baseline', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })

  const metrics = profileResult.metrics || {}
  const diag = await agentRetry(() => agent(
    `Write these metrics to ${EXP_DIR}/baseline/metrics.json:\n${JSON.stringify(metrics)}\n` +
    `${substrateInstruction('diagnose.py', `--metrics ${EXP_DIR}/baseline/metrics.json`)} Return stdout JSON verbatim {bottleneck_class, evidence}.`,
    { model: MODEL.mechanical, label: 'diagnose-baseline', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  bottleneckClass = diag.bottleneck_class || 'unknown'

  ncuSetup = {
    latency_ms: metrics.latency_ms,
    sm_throughput_pct: metrics.sm_pct,
    dram_throughput_pct: metrics.dram_pct,
    achieved_occupancy_pct: (metrics.occupancy != null) ? metrics.occupancy * 100 : undefined,
    bottleneck_diagnosis: `${bottleneckClass}: ${(diag.evidence || []).join('; ')}`,
    profile_summary: profileResult.profiler_available === false
      ? `profiler unavailable; static analysis only (class=${bottleneckClass})`
      : `class=${bottleneckClass}, metrics=${JSON.stringify(metrics)}`,
    profile_evidence: diag.evidence || [],
    profiler_available: profileResult.profiler_available !== false,
    _metrics: metrics,
    _coverage: profileResult.coverage || [],
  }
} else if (IS_EMBEDDED) {
  // --- Embedded baseline (integration-strategist → embedded_*): the operator cannot be
  // built as a standalone TU, so establish the baseline INSIDE the host project. For
  // embedded_inplace the operator file is already pristine on disk (backed up above), so
  // build/test/benchmark it as-is; for embedded_dispatch the unmodified project build is
  // the baseline. perf_heuristic only (no native profiler reachable). ---
  const embBaseline = await agentRetry(() => agent(
    `EMBEDDED BASELINE EVAL. Project operator file: ${KERNEL_PATH}` +
    (ORIGINAL_BACKUP ? ` | pristine backup: ${ORIGINAL_BACKUP}` : '') + `\n` +
    `Run IN ORDER (the operator on disk is the pristine baseline — do NOT modify it):\n` +
    `1. Build: ${BUILD_CMD || '(not provided)'}\n2. Test: ${TEST_CMD || '(not provided)'}\n3. Benchmark: ${PROJECT_BENCH_CMD || TEST_CMD || '(not provided)'}\n` +
    `Parse latency_ms + heuristic_bclass (memory_bound|compute_bound|latency_bound) from the throughput so diagnose.py does not fall to unknown. ` +
    `Tag bottleneck evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
    `Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
    { model: MODEL.profile, label: 'embedded-baseline', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  bottleneckClass = embBaseline?.heuristic_bclass || 'unknown'
  const embMetrics = embBaseline?.metrics || { latency_ms: Number(embBaseline?.latency_ms || 0) }
  ncuSetup = {
    latency_ms: Number(embBaseline?.latency_ms || 0),
    bottleneck_diagnosis: `${bottleneckClass}: embedded project-native baseline`,
    profile_summary: `embedded baseline (class=${bottleneckClass}); method=${INTEGRATION_DECISION.method}`,
    profile_evidence: [`embedded project-native throughput; heuristic_bclass=${bottleneckClass}`],
    profiler_available: false,
    _metrics: embMetrics,
    _coverage: [],
  }
} else if (CUTE_SOL) {
  ncuSetup = { latency_ms: null, bottleneck_diagnosis: 'Host latency only; profiler counters unavailable',
    profile_summary: 'CuTe SOL mode: no NCU counters measured', profile_evidence: [], _metrics: {} }
} else {
  ncuSetup = await agentRetry(() => agent(legacyNcuBaselinePrompt(baselineKernel), { model: MODEL.profile,
    label: 'ncu-baseline',
    phase: 'Setup',
    schema: LEGACY_NCU_BASELINE_SCHEMA,
  }), { retries: 5 })
}

if (IS_SOL) {
  if (!SOL_AVAILABLE) throw new Error('AccelOpt Sol requires complete Host evaluator configuration')
  if (typeof evaluate !== 'function') throw new Error('AccelOpt Sol requires Host evaluation')
  const seed = await __solExecbenchEvaluate({
    label: 'sol-seed-baseline', phase: 'Setup',
    substrateDir: SOL_SUBSTRATE_DIR,
    kernelSource: `${EXP_DIR}/host_seed.${CUTE_SOL ? 'py' : 'cu'}`,
    candidateLanguage: CUTE_SOL ? 'cute-dsl' : '',
    baselineSolutionPath: `${SOL_SEED_DIR}/seed.solution.json`,
    contractEnv: `${SOL_SEED_DIR}/contract.env`,
    solutionOut: `${EXP_DIR}/host_seed.solution.json`,
    benchOut: `${EXP_DIR}/host_seed.bench.jsonl`,
    solCli: SOL_CLI, taskDir: SOL_TASK_DIR, benchConfig: SOL_BENCH_CONFIG,
    seedDir: SOL_SEED_DIR, cudaVisibleDevices: SOL_CVD,
    ldLibraryPath: SOL_LD_LIBRARY_PATH, envPrefix: SOL_ENV_PREFIX,
    definitionPath: SOL_DEFINITION_PATH,
  })
  const latency = seed?.candidate_latency_aggregate_ms
  if (seed?.compiled !== true || seed?.correct !== true ||
      seed?.full_workload_set !== true || seed?.output_contract_valid !== true ||
      seed?.measurement_valid !== true ||
      typeof latency !== 'number' || !Number.isFinite(latency) || latency <= 0) {
    throw new Error('Host could not establish a complete measured Sol seed baseline')
  }
  ncuSetup = { ...ncuSetup, latency_ms: latency,
    profile_summary: `${ncuSetup?.profile_summary || ''}; Host Sol seed latency=${latency}ms` }
}
baselineLatency = ncuSetup.latency_ms
bestLatency = baselineLatency
bestKernelCode = baselineKernel

// Initialize candidate beam with baseline
candidateBeam = [{
  code: baselineKernel,
  latency: baselineLatency,
  speedup: 1.0,
  ncuSummary: ncuSetup.profile_summary || ncuSetup.bottleneck_diagnosis,
  planTitle: 'baseline',
}]

// Build the NCU profile string
function legacyBaselineProfile(ncuSetup) {
  return `
## NCU Profile Results (Baseline)
- Latency: ${ncuSetup.latency_ms} ms
- SM Throughput: ${ncuSetup.sm_throughput_pct || 'N/A'}% of peak
- DRAM Throughput: ${ncuSetup.dram_throughput_pct || 'N/A'}% of peak
- Achieved Occupancy: ${ncuSetup.achieved_occupancy_pct || 'N/A'}%
- Theoretical Occupancy: ${ncuSetup.theoretical_occupancy_pct || 'N/A'}%
- Waves/SM: ${ncuSetup.waves_per_sm || 'N/A'}
- Registers/Thread: ${ncuSetup.registers_per_thread || 'N/A'}
- Top Stall Reason: ${ncuSetup.top_stall_reason || 'N/A'} (${ncuSetup.top_stall_pct || 'N/A'}% of samples)
- Sectors/Request (global LD): ${ncuSetup.sectors_per_request || 'N/A'} (ideal=4)
- L1 Hit Rate: ${ncuSetup.l1_hit_rate_pct || 'N/A'}%
- L2 Hit Rate: ${ncuSetup.l2_hit_rate_pct || 'N/A'}%

## Bottleneck Diagnosis:
${ncuSetup.bottleneck_diagnosis}

## NCU Rule Suggestions:
${(ncuSetup.ncu_rule_suggestions || []).map(s => `- ${s}`).join('\n') || 'N/A'}
`
}
if (CUTE_SOL) {
  baselineNcuProfile = `Host full-workload seed latency: ${baselineLatency}ms. Profiler counters unavailable; no NCU or IKET attribution.`
} else if (USE_DRIVER || IS_EMBEDDED) {
  const m = ncuSetup._metrics || {}
  const lines = []
  if (m.latency_ms != null) lines.push(`- Latency: ${m.latency_ms} ms`)
  if (m.sm_pct != null) lines.push(`- SM Throughput: ${m.sm_pct}% of peak`)
  if (m.dram_pct != null) lines.push(`- DRAM Throughput: ${m.dram_pct}% of peak`)
  if (m.occupancy != null) lines.push(`- Achieved Occupancy: ${(m.occupancy * 100).toFixed(1)}%`)
  baselineNcuProfile = `
## Profile Results (Baseline, backend=${BACKEND}, class=${bottleneckClass})
${lines.join('\n')}

## Bottleneck Diagnosis:
${ncuSetup.bottleneck_diagnosis}

## Evidence:
${(ncuSetup.profile_evidence || []).map(s => `- ${s}`).join('\n') || 'N/A'}
`
} else {
  baselineNcuProfile = legacyBaselineProfile(ncuSetup)
}

log(`Baseline: ${baselineLatency}ms | ${ncuSetup.bottleneck_diagnosis}`)

// =============================================================================
// Iterative Self-Improvement Loop
// =============================================================================

let completedIterations = 0
let supervisorTerminationReason = null
for (let iter = 0; iter < ITERATIONS; iter++) {
  log(`\n=== Iteration ${iter + 1}/${ITERATIONS} | Best: ${bestLatency.toFixed(3)}ms (${(baselineLatency / bestLatency).toFixed(2)}x) | Beam: ${candidateBeam.length} | Experience: ${experienceMemory.length} patterns ===`)

  // ===========================================================================
  // Phase 2: Plan — Generate optimization plans GUIDED BY NCU DATA + BEAM
  // ===========================================================================
  phase('Plan')

  // Experience sampling (AccelOpt: construct_experience.py logic)
  const experienceSection = buildExperienceSection(experienceMemory, lastIterNewPatterns, MAX_EXPERIENCE_IN_PROMPT)

  // Candidate beam context for planner
  const beamSection = buildBeamSection(candidateBeam, IDIOMS.lang_fence,
    CUTE_SOL ? 'Host evidence' : 'NCU')

  const planAngles = IDIOMS.plan_angles

  const planPromptBase = CUTE_SOL
    ? `You are a CuTe DSL optimization expert. The Host measured the complete official workload, but no profiler counters are available. Propose one concrete source-level optimization from the correct inherited CuTe module.

# Operation: ${OP_DESC} (${opType})
# Current CuTe source:
${bestKernelCode.substring(0, 4000)}
# Measured latency: ${bestLatency} ms
${beamSection}
${experienceSection}
# Requirements: preserve module-level run(...), execute CuTe kernels for every workload, cite only observed latency and source structure, and do not invent NCU or IKET metrics. Return one structural optimization plan.`
    : USE_DRIVER
    ? `You are a ${BACKEND} kernel optimization expert. You have REAL ${IDIOMS.profiler_name || 'profiler'} profiling data for this kernel. Use it to generate ONE specific, evidence-based optimization plan.

# Operation: ${OP_DESC} (${opType})

# Current Best Implementation:
\`\`\`${IDIOMS.lang_fence}
${bestKernelCode.substring(0, 4000)}
\`\`\`

# PROFILING DATA (THIS IS REAL MEASURED DATA — base your plan on this):
${baselineNcuProfile}

# Current Performance:
- Latency: ${bestLatency}ms
- Speedup vs original baseline: ${(baselineLatency / bestLatency).toFixed(2)}x
${beamSection}
${experienceSection}

# How to read profile data for planning:
${IDIOMS.read_metric_guide}

# Optimization Plan Requirements:
1. CITE the specific profile metric(s) that justify your plan
2. Name the exact code region and transformation
3. Prefer STRUCTURAL changes over parameter tuning
4. Don't suggest lowering precision below the baseline
5. Estimate expected speedup based on the profile data
6. If candidate beam shows multiple approaches, consider COMBINING strengths from different candidates`
    : `You are a CUDA kernel optimization expert. You have REAL Nsight Compute (NCU) profiling data for this kernel. Use it to generate ONE specific, evidence-based optimization plan.

# Operation: ${OP_DESC} (${opType})

# Current Best Implementation:
\`\`\`cuda
${bestKernelCode.substring(0, 4000)}
\`\`\`

# NCU PROFILING DATA (THIS IS REAL MEASURED DATA — base your plan on this):
${baselineNcuProfile}

# Current Performance:
- Latency: ${bestLatency}ms
- Speedup vs original baseline: ${(baselineLatency / bestLatency).toFixed(2)}x
${beamSection}
${experienceSection}

# How to read NCU data for planning:
${IDIOMS.read_metric_guide}

# Optimization Plan Requirements:
1. CITE the specific NCU metric(s) that justify your plan
2. Name the exact code region and transformation
3. Prefer STRUCTURAL changes over parameter tuning
4. Don't suggest lowering precision below the baseline
5. Estimate expected speedup based on the NCU data (e.g., "NCU reports sectors/request=8.2; fixing to 4.0 should cut load time ~2x on those lines")
6. If candidate beam shows multiple approaches, consider COMBINING strengths from different candidates`

  const planSchema = CUTE_SOL
    ? {
        type: 'object',
        properties: {
          title: { type: 'string' },
          focus_area: { type: 'string' },
          source_evidence: { type: 'string' },
          plan: { type: 'string' },
          expected_impact: { type: 'string' },
          risk: { type: 'string' },
        },
        required: ['title', 'source_evidence', 'plan', 'expected_impact'],
      }
    : USE_DRIVER
    ? {
        type: 'object',
        properties: {
          title: { type: 'string' },
          focus_area: { type: 'string' },
          profile_evidence: { type: 'string' },
          ncu_evidence: { type: 'string' },
          analysis: { type: 'string' },
          plan: { type: 'string' },
          expected_impact: { type: 'string' },
          risk: { type: 'string' },
        },
        required: ['title', 'plan', 'expected_impact'],
      }
    : {
        type: 'object',
        properties: {
          title: { type: 'string' },
          focus_area: { type: 'string' },
          ncu_evidence: { type: 'string' },
          analysis: { type: 'string' },
          plan: { type: 'string' },
          expected_impact: { type: 'string' },
          risk: { type: 'string' },
        },
        required: ['title', 'ncu_evidence', 'plan', 'expected_impact'],
      }

  const plans = await parallel(
    Array.from({length: BREADTH}, (_, i) => () =>
      agentRetry(() => agent(`${planPromptBase}\n\n# YOUR FOCUS AREA: ${planAngles[i % planAngles.length]}\nYou are planner #${i + 1}/${BREADTH}. Focus on: ${planAngles[i % planAngles.length]}.
${__attemptBlock()}${__experienceBlock()}
# Recent genome trajectory (read BEFORE planning)
Run \`tail -20 ${EXP_DIR}/genome.jsonl 2>/dev/null\` to see prior attempts this session (every Plan/Execute/Evaluate/Learn step has self-reported here). Use it to: (a) avoid retrying any technique already attempted with a regression or null speedup, (b) spot multi-round patterns the per-iteration experience summary may have lost. If the file is empty or missing, ignore this and proceed with the inputs above.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is iteration ${iter}, planner ${i}):
{"workflow":"${WORKFLOW_NAME}","phase":"Plan","ts":"<ts>","status":"done","candidate_id":"iter-${iter}-plan-${i}","technique":"<your plan title / optimization move>","speedup":null,"note":"<the profile metric you cited + expected impact, one line>"}`, {
        label: `plan-${iter}-${i}`,
        phase: 'Plan',
        schema: planSchema,
      }), { retries: 5 })
    )
  )

  const validPlans = plans.filter(Boolean)
  const planEvidence = (p) => (p.source_evidence ?? p.profile_evidence ?? p.ncu_evidence)
  log(`Plans: ${validPlans.map(p => `${p.title} (evidence: ${String(planEvidence(p) || '').substring(0, 50)}...)`).join(' | ')}`)

  // ===========================================================================
  // Phase 3: Execute — Implement each plan
  // ===========================================================================
  phase('Execute')

  const implementations = await pipeline(
    validPlans,
    (plan) => parallel(
      Array.from({length: SAMPLES_PER_PLAN}, (_, sampleIdx) => () =>
        agentRetry(() => agent(CUTE_SOL
          ? `Implement the AccelOpt plan as a complete CuTe DSL Python module.
# Correct inherited CuTe source:
${bestKernelCode.substring(0, 4000)}
# Plan: ${plan.title}: ${plan.plan}
${IDIOMS.impl_requirements}
Keep the exact run(...) signature and full-workload semantics. Return the entire candidate source as code; no patch or estimated speedup.`
          : USE_DRIVER
          ? `You are an expert ${BACKEND} kernel developer. Implement this profiler-informed optimization plan as a complete, compilable kernel.

# Original Kernel:
\`\`\`${IDIOMS.lang_fence}
${bestKernelCode.substring(0, 4000)}
\`\`\`

# Optimization Plan: "${plan.title}"
Profiler Evidence: ${planEvidence(plan)}
Plan: ${plan.plan}

# Requirements:
1. ${IDIOMS.impl_requirements}
2. Must be FUNCTIONALLY CORRECT (same output as baseline within FP tolerance)
3. Apply the plan faithfully — the plan is based on real profiler data, so the optimization targets a real bottleneck
4. Keep the entrypoint signature unchanged
5. This is variant ${sampleIdx + 1}/${SAMPLES_PER_PLAN}

Return the complete ${BACKEND} code.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (iteration ${iter}, plan "${plan.title}", sample ${sampleIdx}):
{"workflow":"${WORKFLOW_NAME}","phase":"Execute","ts":"<ts>","status":"done","candidate_id":"iter-${iter}-${plan.title}-v${sampleIdx}","technique":"${plan.title}","speedup":null,"note":"<the concrete code transformation you implemented, one line>"}`
          : legacyExecutePrompt(bestKernelCode, plan, sampleIdx, SAMPLES_PER_PLAN), {
          label: `impl-${iter}-${plan.title.substring(0, 15)}-v${sampleIdx}`,
          phase: 'Execute',
          schema: {
            type: 'object',
            properties: {
              code: { type: 'string' },
              implementation_notes: { type: 'string' },
            },
            required: ['code'],
          },
        }), { retries: 5 })
      )
    )
  )

  const allVariants = []
  for (let planIdx = 0; planIdx < validPlans.length; planIdx++) {
    const planImpls = implementations[planIdx]
    if (!planImpls) continue
    for (let sIdx = 0; sIdx < planImpls.length; sIdx++) {
      const impl = planImpls[sIdx]
      if (impl && impl.code) {
        allVariants.push({
          plan: validPlans[planIdx],
          planIdx: planIdx,
          code: impl.code,
          id: `plan_${planIdx}_sample_${sIdx}`,
        })
      }
    }
  }

  log(`Generated ${allVariants.length} kernel variants`)

  // ===========================================================================
  // Phase 4: Evaluate — NCU profile each variant + per-branch dedup + beam update
  // ===========================================================================
  phase('Evaluate')

  const evalSchema = CUTE_SOL
    ? {
        type: 'object',
        properties: {
          is_correct: { type: 'boolean' },
          is_compilable: { type: 'boolean' },
          estimated_speedup: { type: 'number' },
          correctness_issues: { type: 'array', items: { type: 'string' } },
          performance_analysis: { type: 'string' },
        },
        required: ['is_correct', 'is_compilable', 'estimated_speedup'],
      }
    : USE_DRIVER
    ? {
        type: 'object',
        properties: {
          is_correct: { type: 'boolean' },
          is_compilable: { type: 'boolean' },
          estimated_latency_ms: { type: 'number' },
          estimated_speedup: { type: 'number' },
          correctness_issues: { type: 'array', items: { type: 'string' } },
          profile_comparison: { type: 'string' },
          ncu_comparison: { type: 'string' },
          bottleneck_addressed: { type: 'boolean' },
          new_bottleneck: { type: 'string' },
          performance_analysis: { type: 'string' },
        },
        required: ['is_correct', 'is_compilable', 'estimated_speedup'],
      }
    : {
        type: 'object',
        properties: {
          is_correct: { type: 'boolean' },
          is_compilable: { type: 'boolean' },
          estimated_latency_ms: { type: 'number' },
          estimated_speedup: { type: 'number' },
          correctness_issues: { type: 'array', items: { type: 'string' } },
          ncu_comparison: { type: 'string' },
          bottleneck_addressed: { type: 'boolean' },
          new_bottleneck: { type: 'string' },
          performance_analysis: { type: 'string' },
        },
        required: ['is_correct', 'is_compilable', 'estimated_speedup'],
      }
  const evalProfile = (e) => (e.profile_comparison ?? e.ncu_comparison)

  const evaluations = await parallel(
    allVariants.map((variant, varIdx) => () =>
      agentRetry(() => agent(CUTE_SOL
        ? cuteEvaluatePrompt(variant, bestLatency)
        : USE_DRIVER
        ? `You are a ${BACKEND} kernel evaluator. Evaluate this optimized kernel variant.

# Variant: ${variant.id} — Plan: "${variant.plan.title}"
# Profiler Evidence for this plan: ${planEvidence(variant.plan)}

# Kernel Code:
\`\`\`${IDIOMS.lang_fence}
${variant.code.substring(0, 4000)}
\`\`\`

# Baseline Performance:
- Latency: ${bestLatency}ms
- Bottleneck class: ${bottleneckClass}

# Evaluation Steps:

## Step 1: Static correctness check
- Race conditions, out-of-bounds, missing synchronization, incorrect reductions.

## Step 2: Compilability check
- Required: ${IDIOMS.impl_requirements}

## Step 3: Build and profile (if environment allows)
Use only the user-provided build/run/profile contract. If any required command is missing, do not invent one; return static correctness/compilability analysis and mark profile fields as missing evidence.

## Step 4: Compare with baseline
Calculate speedup = baseline_latency / variant_latency.
Note which profile metrics improved and which degraded.

If the profiler is not available, provide your expert static analysis:
- Did the optimization address the identified bottleneck (${bottleneckClass})?
- Estimate speedup based on the targeted inefficiency.

Return evaluation results.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (iteration ${iter}, variant ${variant.id}; status="done" if correct AND compilable, else "error"; speedup is your estimated_speedup number or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"iter-${iter}-${variant.id}","technique":"${variant.plan.title}","speedup":<number or null>,"note":"<correct? compilable? bottleneck addressed? or the failure reason>"}`
        : legacyEvaluatePrompt(variant, bestLatency, ncuSetup), {
        label: `eval-${variant.id}`,
        phase: 'Evaluate',
        schema: evalSchema,
      }), { retries: 5 })
    )
  )

  // --- Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch) ---
  // Candidate eval above is PARALLEL (LLM static/estimate). For an embedded operator the
  // REAL measurement must happen INSIDE the host project, which mutates a shared file
  // (inplace) or shares a single project build (dispatch) — so it CANNOT run concurrently.
  // This is a SEPARATE SERIAL for-loop (the parallel-embedded-race bug-class). It overwrites
  // each variant's estimated latency/speedup with the measured project-native result so all
  // downstream dedup/beam/learn logic is unchanged. Standalone path never enters here.
  if (IS_EMBEDDED) {
    for (let i = 0; i < allVariants.length; i++) {
      const variant = allVariants[i]
      const e = evaluations[i]
      if (!e) continue
      const kPath = `${EXP_DIR}/variants/iter${iter}/${variant.id}/kernel${IDIOMS.source_ext || '.cu'}`
      const variantTag = `accelopt_${iter}_${variant.id}`.replace(/[^A-Za-z0-9_]/g, '_')
      // Materialize the candidate source so the embedded eval can apply/register it.
      await agentRetry(() => agent(`Write the candidate kernel source to ${kPath} (mkdir -p its dir first).\n\n` +
        `\`\`\`${IDIOMS.lang_fence}\n${(variant.code || '').substring(0, 6000)}\n\`\`\`\n` +
        `Return {ok:true, path:"${kPath}"}.`,
        { model: MODEL.mechanical, label: `embedded-materialize-${variant.id}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      let embResult = null
      if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
        embResult = await agentRetry(() => agent(
          `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project operator file: ${KERNEL_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
          `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
          `2. Apply candidate: cp ${kPath} ${KERNEL_PATH}\n3. Build: ${BUILD_CMD}\n4. Test: ${TEST_CMD}\n5. Benchmark: ${PROJECT_BENCH_CMD || TEST_CMD}\n` +
          `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
          `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-inplace-${variant.id}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
        const _plan = typeof __embeddedEvalPlan === 'function'
          ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant: variantTag, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: TEST_CMD, benchmarkCmd: PROJECT_BENCH_CMD || TEST_CMD })
          : null
        if (_plan) {
          embResult = await agentRetry(() => agent(
            `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
            `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
            { model: MODEL.mechanical, label: `embedded-dispatch-${variant.id}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        }
      }
      if (embResult) {
        const embLatency = Number(embResult.latency_ms || 0)
        e.is_compilable = embResult.compiled !== false
        e.is_correct = embResult.correct !== false
        if (embLatency > 0) {
          e.estimated_latency_ms = embLatency
          e.estimated_speedup = baselineLatency / embLatency
        }
        e.performance_analysis = `embedded ${INTEGRATION_DECISION.method}; heuristic_bclass=${embResult.heuristic_bclass || 'unknown'}; ` + (e.performance_analysis || '')
      }
    }
  }

  // The parallel evaluation above is explicitly an LLM static estimate - the
  // schema field is named estimated_speedup and step 1 is a static correctness
  // check - and the embedded branch already exists to overwrite those estimates
  // with a real project-native measurement. Its comment notes "Standalone path
  // never enters here", which left the sol-execbench path on estimates alone.
  // Mirror the same overwrite for it. Unlike the embedded case this measures an
  // isolated candidate rather than mutating the host project, so it can run
  // concurrently.
  if (IS_SOL && SOL_AVAILABLE) {
    // Static agent judgments are authoring hints. They must never become beam
    // entries if the Host returns no complete measurement for a candidate.
    for (const e of evaluations.filter(Boolean)) {
      e.is_compilable = false
      e.is_correct = false
      e.estimated_speedup = 0
      e.estimated_latency_ms = null
    }
    const measured = await parallel(allVariants.map((variant, i) => async () => {
      if (!evaluations[i] || !(variant.code || '').trim()) return null
      const vid = String(variant.id).replace(/[^A-Za-z0-9_]/g, '_')
      const r = await __solExecbenchEvaluate({
        label: `sol-eval-${vid}`, phase: 'Evaluate',
        substrateDir: SOL_SUBSTRATE_DIR,
        kernelSource: `${EXP_DIR}/accelopt_${vid}.${CUTE_SOL ? 'py' : 'cu'}`,
        candidateSource: variant.code,
        candidateLanguage: CUTE_SOL ? 'cute-dsl' : '',
        bindingOut: `${EXP_DIR}/bindings/accelopt_${vid}.json`,
        bindingWorkflow: WORKFLOW_NAME,
        candidateId: String(variant.id),
        contractEnv: `${SOL_SEED_DIR || '.'}/contract.env`,
        solutionOut: `${EXP_DIR}/accelopt_${vid}.solution.json`,
        benchOut: `${EXP_DIR}/accelopt_${vid}.bench.jsonl`,
        solCli: SOL_CLI, taskDir: SOL_TASK_DIR, benchConfig: SOL_BENCH_CONFIG,
        seedDir: SOL_SEED_DIR, cudaVisibleDevices: SOL_CVD,
        ldLibraryPath: SOL_LD_LIBRARY_PATH, envPrefix: SOL_ENV_PREFIX,
        definitionPath: SOL_DEFINITION_PATH,
      })
      return r ? { i, r } : null
    }))
    for (const m of measured.filter(Boolean)) {
      const e = evaluations[m.i]
      const r = m.r
      const hostLatency = r.candidate_latency_aggregate_ms
      const hostValid = r.compiled === true && r.correct === true &&
        r.full_workload_set === true && r.output_contract_valid === true &&
        r.measurement_valid === true &&
        typeof hostLatency === 'number' && Number.isFinite(hostLatency) && hostLatency > 0
      if (hostValid && r.artifact_binding?.verified !== true) {
        throw new Error('AccelOpt Host artifact binding missing for a correct Sol candidate')
      }
      e.is_compilable = r.compiled === true
      e.is_correct = hostValid
      e.estimated_speedup = hostValid ? baselineLatency / hostLatency : 0
      e.estimated_latency_ms = hostValid ? hostLatency : null
      if (hostValid && hostLatency < baselineLatency &&
          (!solBestHostCandidate || hostLatency < solBestHostCandidate.latency)) {
        solBestHostCandidate = {
          id: String(allVariants[m.i].id), code: allVariants[m.i].code,
          path: r.candidate_path, sha256: r.candidate_sha256,
          binding_path: r.artifact_binding.binding_path,
          latency: hostLatency, gain: baselineLatency / hostLatency,
        }
      }
      e.performance_analysis = `host-measured sol-execbench: compiled=${r.compiled} correct=${r.correct} `
        + `speedup_vs_seed=${e.estimated_speedup} reference_speedup=${r.speedup} workloads=${r.n_pass}/${r.n_total}`
        + (r.failure_code ? ` failure_code=${r.failure_code}` : '')
        + `; ` + (e.performance_analysis || '')
      log(`Host-measured ${allVariants[m.i].id}: compiled=${r.compiled} correct=${r.correct} `
        + `speedup=${r.speedup} workloads=${r.n_pass}/${r.n_total}`)
    }
  }

  // Build results with evaluation data
  const results = []
  for (let i = 0; i < allVariants.length; i++) {
    const evalResult = evaluations[i]
    if (!evalResult) continue
    results.push({
      variant: allVariants[i],
      evaluation: evalResult,
      speedup: evalResult.estimated_speedup || 1.0,
    })
  }

  // Per-branch deduplication: keep only the best sample per plan (AccelOpt: select_candidates.py)
  const planBestMap = new Map()
  for (const r of results) {
    if (!r.evaluation.is_correct || !r.evaluation.is_compilable) continue
    const planKey = r.variant.planIdx
    const existing = planBestMap.get(planKey)
    if (!existing || r.speedup > existing.speedup) {
      planBestMap.set(planKey, r)
    }
  }
  const dedupedResults = [...planBestMap.values()]
  dedupedResults.sort((a, b) => b.speedup - a.speedup)

  // Update candidate beam (AccelOpt: select_candidates.py topK logic)
  const newCandidates = dedupedResults
    .filter(r => r.speedup > 1.0)
    .map(r => ({
      code: r.variant.code,
      latency: r.evaluation.estimated_latency_ms || (baselineLatency / r.speedup),
      speedup: baselineLatency / r.evaluation.estimated_latency_ms,
      ncuSummary: r.evaluation.ncu_comparison || r.evaluation.performance_analysis || '',
      planTitle: r.variant.plan.title,
    }))

  // Merge new candidates into beam, re-sort, keep topK
  const mergedBeam = [...candidateBeam, ...newCandidates]
  mergedBeam.sort((a, b) => a.latency - b.latency)
  candidateBeam = mergedBeam.slice(0, TOPK_CANDIDATES)

  // Update best from beam[0]
  if (candidateBeam.length > 0 && candidateBeam[0].latency < bestLatency) {
    bestKernelCode = candidateBeam[0].code
    bestLatency = candidateBeam[0].latency

    // Update NCU profile for next iteration
    const bestResult = dedupedResults.find(r => r.variant.code === candidateBeam[0].code)
    if (CUTE_SOL || (bestResult && bestResult.evaluation.ncu_comparison)) {
      baselineNcuProfile = CUTE_SOL
        ? cuteIterEvidence(iter, candidateBeam, bestLatency, baselineLatency, baselineNcuProfile)
        : USE_DRIVER
        ? `
## Profile Results (After Iteration ${iter + 1}, class=${bottleneckClass} — Best: "${candidateBeam[0].planTitle}")
- Latency: ${bestLatency}ms (${(baselineLatency / bestLatency).toFixed(2)}x speedup vs original)
- Bottleneck addressed: ${bestResult.evaluation.bottleneck_addressed ? 'YES' : 'NO'}
- New bottleneck: ${bestResult.evaluation.new_bottleneck || 'unknown'}
- Comparison: ${evalProfile(bestResult.evaluation)}

Previous profile data for reference:
${baselineNcuProfile}`
        : legacyIterProfile(iter, candidateBeam, bestResult, bestLatency, baselineLatency, baselineNcuProfile)
    }

    log(`NEW BEST: "${candidateBeam[0].planTitle}" — ${(baselineLatency / bestLatency).toFixed(2)}x, ~${bestLatency.toFixed(3)}ms`)
  }

  const improved = dedupedResults.filter(r => r.speedup > 1.0)
  const degraded = dedupedResults.filter(r => r.speedup < 1.0)
  log(`Results (deduped): ${improved.length} improved, ${degraded.length} degraded | Beam: [${candidateBeam.map(c => c.planTitle).join(', ')}]`)

  // ===========================================================================
  // Phase 5: Learn — Extract insights from slow-fast pairs + NCU metric diffs
  //
  // AccelOpt alignment:
  //   - Threshold filtering (max_threshold / min_threshold)
  //   - topk_learn total budget
  //   - Experience format: **title** + NCU trigger + code snippets
  // ===========================================================================
  phase('Learn')

  // Threshold-filtered selection (AccelOpt: rewrites_selection.py)
  const positiveFiltered = improved.filter(r => r.speedup > MAX_THRESHOLD)
  const negativeFiltered = degraded.filter(r => r.speedup < (1.0 / MIN_THRESHOLD))

  const maxPositive = Math.min(positiveFiltered.length, Math.ceil(TOPK_LEARN / 2))
  const selectedPositive = positiveFiltered.slice(0, maxPositive)
  const remainingSlots = Math.min(TOPK_LEARN - selectedPositive.length, negativeFiltered.length)
  const selectedNegative = negativeFiltered.slice(0, remainingSlots)

  const pairsToSummarize = []

  for (const r of selectedPositive) {
    pairsToSummarize.push({
      slow: baselineKernel,
      fast: r.variant.code,
      speedup: r.speedup,
      plan_title: r.variant.plan.title,
      ncu_evidence: r.variant.plan.ncu_evidence,
      ncu_comparison: r.evaluation.ncu_comparison || '',
      ...(USE_DRIVER ? {
        profile_evidence: planEvidence(r.variant.plan),
        profile_comparison: evalProfile(r.evaluation) || '',
      } : {}),
      bottleneck_addressed: r.evaluation.bottleneck_addressed,
      type: 'positive',
    })
  }

  for (const r of selectedNegative) {
    pairsToSummarize.push({
      slow: r.variant.code,
      fast: bestKernelCode,
      speedup: 1.0 / r.speedup,
      plan_title: r.variant.plan.title + ' [ANTI-PATTERN]',
      ncu_evidence: r.variant.plan.ncu_evidence,
      ncu_comparison: r.evaluation.ncu_comparison || '',
      ...(USE_DRIVER ? {
        profile_evidence: planEvidence(r.variant.plan),
        profile_comparison: evalProfile(r.evaluation) || '',
      } : {}),
      bottleneck_addressed: r.evaluation.bottleneck_addressed,
      type: 'negative',
    })
  }

  if (pairsToSummarize.length > 0) {
    const learnSchema = CUTE_SOL
      ? {
          type: 'object',
          properties: {
            title: { type: 'string' },
            source_trigger: { type: 'string' },
            rule: { type: 'string' },
            original_snippet: { type: 'string' },
            optimized_snippet: { type: 'string' },
            why: { type: 'string' },
            is_antipattern: { type: 'boolean' },
          },
          required: ['title', 'source_trigger', 'rule', 'original_snippet', 'optimized_snippet', 'why'],
        }
      : USE_DRIVER
      ? {
          type: 'object',
          properties: {
            title: { type: 'string' },
            profile_trigger: { type: 'string' },
            ncu_trigger: { type: 'string' },
            rule: { type: 'string' },
            original_snippet: { type: 'string' },
            optimized_snippet: { type: 'string' },
            why: { type: 'string' },
            is_antipattern: { type: 'boolean' },
          },
          required: ['title', 'rule', 'original_snippet', 'optimized_snippet', 'why'],
        }
      : {
          type: 'object',
          properties: {
            title: { type: 'string' },
            ncu_trigger: { type: 'string' },
            rule: { type: 'string' },
            original_snippet: { type: 'string' },
            optimized_snippet: { type: 'string' },
            why: { type: 'string' },
            is_antipattern: { type: 'boolean' },
          },
          required: ['title', 'ncu_trigger', 'rule', 'original_snippet', 'optimized_snippet', 'why'],
        }
    const summaries = await parallel(
      pairsToSummarize.map((pair) => () =>
        agentRetry(() => agent(CUTE_SOL
          ? cuteLearnPrompt(pair)
          : USE_DRIVER
          ? `You are a ${BACKEND} optimization expert with profiler expertise. Analyze this slow-fast kernel pair and extract a GENERAL, REUSABLE optimization insight.

# Slow Kernel:
\`\`\`${IDIOMS.lang_fence}
${pair.slow.substring(0, 2500)}
\`\`\`

# Fast Kernel:
\`\`\`${IDIOMS.lang_fence}
${pair.fast.substring(0, 2500)}
\`\`\`

# Speedup: ${pair.speedup.toFixed(2)}x
# This is a ${pair.type === 'positive' ? 'POSITIVE example (do this)' : 'NEGATIVE example (avoid this)'}

# Profiler Evidence that motivated this optimization:
${(pair.profile_evidence ?? pair.ncu_evidence) || 'N/A'}

# Profile Comparison (before vs after):
${(pair.profile_comparison ?? pair.ncu_comparison) || 'N/A'}

# Was the targeted bottleneck addressed? ${pair.bottleneck_addressed ? 'YES' : 'NO/UNKNOWN'}

## Your task:
Extract a GENERAL optimization rule following this EXACT format:

**{Short title}**
Profiler trigger: {what metric/pattern signals this opportunity}
Rule: {one sentence — when you see X in the profile, do Y to the code}
Original code:
\`\`\`${IDIOMS.lang_fence}
{2-5 lines of slow pattern}
\`\`\`
Optimized code:
\`\`\`${IDIOMS.lang_fence}
{2-5 lines of fast pattern}
\`\`\`
Why: {hardware-level explanation}

Make the rule GENERAL enough to apply to other kernels (not specific to this one kernel).
${__experienceBlock()}
# Recent genome trajectory (read BEFORE extracting the rule)
Run \`tail -30 ${EXP_DIR}/genome.jsonl 2>/dev/null\` to see prior attempts this session. Cross-check your candidate rule against the broader trajectory: does the pattern hold across multiple iterations, or is this slow/fast pair an outlier? Cite a second supporting (or contradicting) example if you find one. If the file is empty or missing, ignore this and rely on the slow/fast pair above.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (iteration ${iter}, pair "${pair.plan_title}", a ${pair.type} example at ${pair.speedup.toFixed(2)}x):
{"workflow":"${WORKFLOW_NAME}","phase":"Learn","ts":"<ts>","status":"done","candidate_id":"iter-${iter}-learn-${pair.plan_title}","technique":"<your extracted rule title>","speedup":${pair.speedup.toFixed(2)},"note":"<the general reusable rule you extracted, one line>"}`
          : legacyLearnPrompt(pair), {
          label: `learn-${pair.plan_title.substring(0, 20)}`,
          phase: 'Learn',
          schema: learnSchema,
        }), { retries: 5 })
      )
    )

    // Format experience entries aligned with AccelOpt's summarizer output format
    lastIterNewPatterns = []
    for (const s of summaries.filter(Boolean)) {
      const formatted = CUTE_SOL
        ? `**${s.title}**\nSource trigger: ${s.source_trigger}\n${s.rule}\nOriginal CuTe source:\n\`\`\`python\n${s.original_snippet}\n\`\`\`\nOptimized CuTe source:\n\`\`\`python\n${s.optimized_snippet}\n\`\`\`\nWhy: ${s.why}\nEvidence: official Host latency; profiler counters unavailable`
        : USE_DRIVER
        ? `**${s.title}**\nProfiler trigger: ${s.profile_trigger ?? s.ncu_trigger}\n${s.rule}\nOriginal code:\n\`\`\`${IDIOMS.lang_fence}\n${s.original_snippet}\n\`\`\`\nOptimized code:\n\`\`\`${IDIOMS.lang_fence}\n${s.optimized_snippet}\n\`\`\`\nWhy: ${s.why}`
        : `**${s.title}**\nNCU trigger: ${s.ncu_trigger}\n${s.rule}\nOriginal code:\n\`\`\`cuda\n${s.original_snippet}\n\`\`\`\nOptimized code:\n\`\`\`cuda\n${s.optimized_snippet}\n\`\`\`\nWhy: ${s.why}`
      experienceMemory.push(formatted)
      lastIterNewPatterns.push(formatted)
    }
    log(`Learned ${lastIterNewPatterns.length} patterns (threshold-filtered from ${pairsToSummarize.length} pairs). Pool: ${experienceMemory.length}`)
  } else {
    lastIterNewPatterns = []
    log(`No pairs passed threshold filters (max>${MAX_THRESHOLD}, min<${(1/MIN_THRESHOLD).toFixed(3)}).`)
  }

  phase('Iterate')
  // A ratio against an unmeasured baseline is not a speedup.  Observed on B300:
  // the profiling step reported "NOT MEASURED - no profiling was performed. Static
  // analysis only" and left baselineLatency at -1, after which this line still
  // printed "1.00x vs baseline" for two full iterations.  Say unmeasured instead of
  // inventing a number, which is the same fault the catalog already records against
  // gpuforecasters for reporting simulated speedups without compiling.
  const _measured = Number.isFinite(baselineLatency) && baselineLatency > 0
    && Number.isFinite(bestLatency) && bestLatency > 0
  const _ratio = _measured
    ? `${(baselineLatency / bestLatency).toFixed(2)}x vs baseline`
    : 'speedup unmeasured (no measured baseline)'
  log(`Iteration ${iter + 1} done. ${_ratio}. Beam size: ${candidateBeam.length}`)
  completedIterations = iter + 1
  if (CUTE_SOL) {
    const safePoint = await __workflowRuntimeSafePoint({
      expDir: EXP_DIR, checkpointPath: CHECKPOINT_PATH,
      terminationFile: TERMINATION_FILE, deadlineEpoch: DEADLINE_EPOCH,
      checkpoint: {
        schema_version: 1, workflow: WORKFLOW_NAME,
        progress: {unit: 'iteration', completed: completedIterations, requested: ITERATIONS},
        compiled: true, correct: true,
        metric: {name: 'speedup_vs_seed',
                 value: _measured ? baselineLatency / bestLatency : null},
        best_kernel_path: solBestHostCandidate?.path || null,
        termination_requested: false, termination_reason: null,
      },
      bestKernelPath: solBestHostCandidate?.path || null,
      materializeBest: false,
      label: `checkpoint-${iter + 1}`, phase: 'Iterate',
    })
    if (safePoint.termination_requested) {
      supervisorTerminationReason = safePoint.termination_reason || 'supervisor_request'
      log(`Cooperative stop after iteration ${completedIterations}: ${supervisorTerminationReason}`)
      break
    }
  }
}

// =============================================================================
// Final Report
// =============================================================================
const finalReport = supervisorTerminationReason
  ? `Stopped after iteration ${completedIterations}: ${supervisorTerminationReason}; measured best is retained.`
  : await agentRetry(() => agent(CUTE_SOL
  ? cuteFinalReportPrompt(OP_DESC, baselineLatency, bestLatency, candidateBeam,
    experienceMemory, baselineNcuProfile, bestKernelCode)
  : USE_DRIVER
  ? `Write a concise technical optimization report.

# AccelOpt (${BACKEND} backend, profiler=${IDIOMS.profiler_name || 'none'})
- Operation: ${OP_DESC} (${opType})
- Baseline Latency: ${baselineLatency}ms
- Final Best Latency: ${bestLatency}ms
- Overall Speedup: ${(baselineLatency / bestLatency).toFixed(2)}x
- Iterations: ${ITERATIONS}
- Candidate Beam (final): ${candidateBeam.length} kernels
- Experience Patterns: ${experienceMemory.length}
- Bottleneck class: ${bottleneckClass}

# Initial Profile Diagnosis:
${baselineNcuProfile.substring(0, 1000)}

# Final Candidate Beam:
${candidateBeam.map((c, i) => `${i + 1}. "${c.planTitle}" — ${c.speedup.toFixed(2)}x (${c.latency.toFixed(3)}ms)`).join('\n')}

# Learned Optimization Knowledge Base:
${experienceMemory.map((e, i) => `${i + 1}. ${e}`).join('\n\n')}

# Final Kernel:
\`\`\`${IDIOMS.lang_fence}
${bestKernelCode.substring(0, 3000)}
\`\`\`

Write:
1. Profiler-driven optimization journey (what metrics → what actions → what results)
2. Which profile patterns reliably predicted optimization opportunities
3. Anti-patterns: what profile data looked promising but the optimization failed
4. Candidate beam evolution: how the population of solutions evolved
5. Remaining bottlenecks (bottleneck_class=${bottleneckClass} for the final kernel)
6. Recommendations for further optimization with specific profiler metrics to target`
  : legacyFinalReportPrompt(OP_DESC, opType, baselineLatency, bestLatency, ITERATIONS, candidateBeam, experienceMemory, baselineNcuProfile, bestKernelCode), {
  label: 'final-report',
  phase: 'Iterate',
}), { retries: 5 })

let evidenceEnvelope = null
if (USE_DRIVER) {
  const insightItems = experienceMemory.slice(0, TOPK_LEARN).map(e => ({
    kind: 'bottleneck',
    directive: 'explore',
    evidence: IDIOMS.profiler_name ? 'ncu' : 'profile_heuristic',
    confidence: 'inferred',
    claim: e,
    actionable_hint: 'apply this learning when generating / improving the next candidate',
  }))
  const built = await agentRetry(() => agent(
    `Build a Layer-A evidence envelope for this AccelOpt run, then validate it.\n` +
    `1. ${substrateInstruction('evidence_schema.py', 'template')} to get the envelope shape.\n` +
    `2. Fill it: attempt_id="accelopt-${BACKEND}", backend="${BACKEND}", ` +
    `compiled=true, correct=true, speedup=${(baselineLatency / bestLatency)}, ` +
    `metrics=${JSON.stringify(ncuSetup._metrics || {})}, ` +
    `bottleneck_class="${bottleneckClass}", ` +
    `insights=${JSON.stringify(insightItems)}.\n` +
    `Each insight item already has {kind, directive, evidence, confidence, claim} as required by ` +
    `evidence_schema.py _validate_item.\n` +
    `3. Write it to ${EXP_DIR}/evidence.json.\n` +
    `4. ${substrateInstruction('evidence_schema.py', `validate ${EXP_DIR}/evidence.json`)}\n` +
    `Return {valid, normalized} (the validator stdout JSON verbatim).`,
    { label: 'assemble-evidence', phase: 'Iterate', schema: JSON_PASSTHROUGH }), { retries: 5 })
  evidenceEnvelope = built.valid ? (built.normalized || null) : null
  if (!built.valid) log(`WARN: Layer-A envelope failed evidence_schema validation`)
}

// embedded_inplace exit safety net: unconditionally restore the pristine operator file.
// The per-variant inplace eval always restores, but this is the belt-and-braces final
// restore in case a run aborted mid-eval (inplace-no-restore bug-class).
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${KERNEL_PATH}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Iterate', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: IS_SOL ? (solBestHostCandidate?.path || '') : generatedKernelPath,
  best_candidate_id: IS_SOL ? (solBestHostCandidate?.id || '') : '',
  artifact_binding_required: IS_SOL && Boolean(solBestHostCandidate),
  artifact_binding_path: IS_SOL ? (solBestHostCandidate?.binding_path || '') : '',
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  baseline_latency_ms: baselineLatency,
  best_latency_ms: IS_SOL ? (solBestHostCandidate?.latency || baselineLatency) : bestLatency,
  overall_speedup: IS_SOL ? (solBestHostCandidate?.gain || 1) : baselineLatency / bestLatency,
  iterations_completed: completedIterations,
  termination_reason: supervisorTerminationReason,
  candidate_beam: candidateBeam.map(c => ({
    plan_title: c.planTitle,
    latency_ms: c.latency,
    speedup: c.speedup,
  })),
  experience_patterns_count: experienceMemory.length,
  experience_patterns: experienceMemory,
  best_kernel_code: IS_SOL ? (solBestHostCandidate?.code || '') : bestKernelCode,
  ...(CUTE_SOL ? {
    evidence_mode: 'host_full_workload_latency',
    profiler_counters_available: false,
  } : {}),
  ncu_baseline_profile: baselineNcuProfile,
  report: finalReport,
  ...(USE_DRIVER ? {
    backend: BACKEND,
    baseline_profile: baselineNcuProfile,
    bottleneck_class: bottleneckClass,
    evidence: evidenceEnvelope,
  } : {}),
}
