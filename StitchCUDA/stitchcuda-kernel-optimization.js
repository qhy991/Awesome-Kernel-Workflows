export const meta = {
  name: 'stitchcuda-kernel-optimization',
  description: 'Three-agent orchestration for CUDA kernel synthesis with adaptive replanning',
  whenToUse: 'Use for generating optimized CUDA kernels via Planner-Coder-Verifier orchestration',
  phases: [
    { title: 'Setup', detail: 'Initialize StitchCUDA environment and orchestrator' },
    { title: 'Plan', detail: 'Generate initial optimization plan' },
    { title: 'Code', detail: 'Generate CUDA kernel implementation' },
    { title: 'Verify', detail: 'Verify correctness and performance' },
    { title: 'Replan', detail: 'Adaptive replanning when needed' },
    { title: 'Report', detail: 'Generate synthesis report' },
  ],
};

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

const WORKFLOW_NAME = 'stitchcuda-kernel-optimization'


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
const EXPDIR = args.exp_dir || '.'


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


// --- Backend driver wiring (P5c Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const WORKSPACE = args.workspace || '/tmp/stitchcuda'

// --- Project-native integration (embedded kernels via integration-strategist) ---
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const BUILD_CMD = args.build_command || ''
const BENCH_CMD = args.benchmark_command || ''
const TEST_CMD = args.test_command || ''
const REGISTER_SCRIPT = args.register_script || ''

const LEGACY_SETUP_LANG_TOKEN = 'CUDA'
const LEGACY_REPLAN_LANG_TOKEN = 'CUDA'
const LEGACY_PLAN_LANG_TOKEN = 'CUDA'
const LEGACY_CODE_LANG_TOKEN = 'CUDA'
const LEGACY_VERIFY_LANG_TOKEN = 'CUDA'
const LEGACY_SOURCE_EXT = '.cu'
const LEGACY_FENCE_TOKEN = 'cuda'
const LEGACY_CODE_FORMAT_HINT = `Use PyTorch load_inline compatible format:
   - __global__ kernel function
   - Template parameters if needed
   - Extern "C" wrapper if needed`
const LEGACY_VERIFY_TOOL_HINT = 'Profile with nsys/ncu if available'
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

// Intersectional guard (P5c plan §3 + §9.1): KernelBench harness is
// CUDA-only as a benchmark suite; refuse driver path when the user
// explicitly pins a benchmark_suite combined with a non-CUDA driver.
if (USE_DRIVER && args.kernelbench_config && args.kernelbench_config.benchmark_suite && RESOLVED_BACKEND && RESOLVED_BACKEND !== 'cuda') {
  throw new Error(
    `StitchCUDA kernelbench_config.benchmark_suite="${args.kernelbench_config.benchmark_suite}" requires backend_dir to be a CUDA driver; ` +
    `got backend_dir=${args.backend_dir} (resolved backend=${RESOLVED_BACKEND}).`
  )
}

// --- BEGIN inlined backend-axis (driver) scaffolding (from _meta/scaffolding/backend-axis.js) ---
function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}
// --- END inlined backend-axis (driver) scaffolding ---

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
function attemptKernelPath(attempt) {
  const ext = USE_DRIVER ? DRIVER_SOURCE_EXT : LEGACY_SOURCE_EXT
  return `${WORKSPACE}/attempt_${attempt}/kernel${ext}`
}
function codeFormatHint() {
  if (!USE_DRIVER) return LEGACY_CODE_FORMAT_HINT
  return DRIVER_IMPL_REQUIREMENTS || `Follow the ${DRIVER_LANG_FENCE} driver source-format conventions and provide a host launcher.`
}
function verifyToolHint() {
  return USE_DRIVER ? `Profile via the driver profile.sh envelope` : LEGACY_VERIFY_TOOL_HINT
}

// StitchCUDA: Three-agent orchestration for CUDA kernel synthesis
// Based on arXiv:2603.02637
// Planner → Coder → Verifier with adaptive replanning

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it in the driver-envelope profile branch. The agent only classifies the
// task (fuzzy); the substrate stamps confidence by method (measured/inferred/
// hypothesized) -- not the agent. See _substrate/profiling/README.md. Falls back
// to native_profiler if undecided. Happy path is unchanged when ignored. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }

// --- integration-strategist: route build/test mode (standalone vs embedded_*).
// Resolved in Setup (below) once KERNEL_PATH is known; declared here at module
// scope so the attempt loop and exit-restore can read the decision. Default keeps
// the legacy standalone path byte-identical when no embedded source is supplied. ---
const KERNEL_PATH = args.kernel_path || ''
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
let USE_DRIVER_STANDALONE = USE_DRIVER  // narrowed after integration-strategist resolves
let IS_EMBEDDED = false
let ORIGINAL_BACKUP = ''

async function main() {
  // ============================================================================
  // Phase 1: Setup
  // ============================================================================
  phase('Setup');

  if (USE_DRIVER) {
    DRIVER = await agentRetry(() => agent(
      `Load the backend driver at ${BACKEND_DIR} and return its manifest plus idioms verbatim.\n` +
      `1. Run exactly: \`cat ${driverPath('manifest.json')}\` and parse JSON.\n` +
      `2. Run exactly: \`cat ${driverPath('idioms.json')}\` and parse JSON.\n` +
      `Return {present, backend_id, source_ext, aux_ext, lang_fence, impl_requirements, methods}.`,
      { model: MODEL.mechanical, label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH }
    ), { retries: 5, allowNull: true });
    if (!DRIVER || DRIVER.present === false) {
      throw new Error(`No backend driver present at ${BACKEND_DIR}. Provide a valid backend_dir or omit it for the legacy path.`);
    }
    if (RESOLVED_BACKEND && DRIVER.backend_id && normalizeSuitabilityValue(DRIVER.backend_id) !== RESOLVED_BACKEND) {
      throw new Error(`backend_dir manifest backend_id="${DRIVER.backend_id}" conflicts with args.backend/language="${RESOLVED_BACKEND}".`);
    }
    DRIVER_LANG_FENCE = DRIVER.lang_fence || DRIVER_LANG_FENCE;
    DRIVER_IMPL_REQUIREMENTS = DRIVER.impl_requirements || '';
    DRIVER_SOURCE_EXT = DRIVER.source_ext || DRIVER_SOURCE_EXT;
    DRIVER_BACKEND_ID = DRIVER.backend_id || DRIVER_BACKEND_ID;
    log(`Driver loaded: ${DRIVER_BACKEND_ID} (fence=${DRIVER_LANG_FENCE})`);
  }

  // --- integration-strategist gate: route standalone vs embedded_* once we know
  // the source kernel. StitchCUDA is normally a generator (no kernel_path) -> the
  // default 'standalone' decision holds and the legacy path is byte-identical. When
  // args.kernel_path points at an inference-engine embedded operator (e.g. an
  // llama.cpp .cuh that cannot compile as a single TU), the strategist routes to
  // embedded_inplace / embedded_dispatch and the attempt loop evaluates in-project. ---
  if (KERNEL_PATH) {
    const _profManifest = (USE_DRIVER && BACKEND_DIR) ? `${BACKEND_DIR}/manifest.json` : `${SUBSTRATE}/backends/cuda/manifest.json`
    const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
    const _integ = await agentRetry(() => agent(
      `Read ${KERNEL_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
      `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
      `--kernel "${KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
      `--cache ${EXPDIR}/integ_cache.json --trajectory ${EXPDIR}/genome.jsonl\`. ` +
      `Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
      { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    if (_integ && _integ.method) INTEGRATION_DECISION = _integ
  }
  log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
  if (INTEGRATION_DECISION.method === 'derive_adapter') {
    throw new Error('integration-strategist returned derive_adapter — provide project_root + build/test commands')
  }
  USE_DRIVER_STANDALONE = USE_DRIVER && INTEGRATION_DECISION.method === 'standalone'
  IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
  ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXPDIR}/integ_original.backup` : ''
  if (ORIGINAL_BACKUP) {
    await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
      { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  }
  // embedded_inplace exit safety net: unconditionally restore the pristine original
  // before main() returns (covers no_successful_kernel + success paths). The
  // per-attempt embedded-inplace eval also ALWAYS restores; this is the belt-and-
  // suspenders exit restore the rollout guard requires.
  async function __exitRestore() {
    if (!ORIGINAL_BACKUP) return
    await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${KERNEL_PATH}"\` and confirm.`,
      { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
  }
  // A-O1 closure: native_profiler but ncu unavailable -> perf_heuristic (no ncu in
  // StitchCUDA's default contract; the driver profile.sh envelope provides native).
  if (PROFILING_DECISION.method === 'native_profiler' && !USE_DRIVER) {
    log(`profiling: native_profiler but no driver profile envelope -> downgrade to perf_heuristic`)
    PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
      profiler_name: 'verifier-perf', rationale: 'native_profiler but no driver profile envelope -> perf_heuristic' }
  }

  const setupResult = await agentRetry(() => agent(
    `Set up StitchCUDA orchestration environment:

1. Initialize ${langToken(LEGACY_SETUP_LANG_TOKEN)} environment:
   - ${langToken(LEGACY_SETUP_LANG_TOKEN)} version and user-provided compiler/toolchain
   - Target GPU architecture (sm_80, sm_89, sm_90, etc.)
   - PyTorch load_inline integration
2. Configure KernelBench evaluation:
   - Benchmark suite selection
   - Performance metrics
   - Correctness test suite
3. Set up three-agent orchestration:
   - Planner: strategic optimization planning
   - Coder: kernel code generation
   - Verifier: correctness and performance verification
4. Configure adaptive replanning heuristics:
   - Replan trigger: N consecutive compile failures
   - Replan trigger: N consecutive correctness failures
   - Replan trigger: performance stagnation (M iterations)
5. Identify target kernel specification:
   - Operation type
   - Input/output shapes
   - Data types
   - Performance baseline

Return JSON:
{
  "cuda_version": "12.x|11.x",
  "target_architecture": "sm_80|sm_89|sm_90",
  "pytorch_available": true/false,
  "kernel_spec": {
    "operation": "gemm|conv2d|attention|reduce|...",
    "shapes": "shape description",
    "dtypes": ["float32", "float16", ...],
    "baseline_gflops": <float or null>
  },
  "kernelbench_config": {
    "benchmark_suite": "suite name",
    "metrics": ["correctness", "performance", ...]
  },
  "replan_heuristics": {
    "compile_failure_threshold": <int>,
    "correctness_failure_threshold": <int>,
    "stagnation_iterations": <int>
  },
  "max_attempts": <int>
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"orchestration_setup","note":"<target operation + architecture + max_attempts, one line>"}`,
    {
      label: 'Setup StitchCUDA',
      phase: 'Setup',
      schema: {
        type: 'object',
        properties: {
          cuda_version: { type: 'string' },
          target_architecture: { type: 'string' },
          pytorch_available: { type: 'boolean' },
          kernel_spec: { type: 'object' },
          kernelbench_config: { type: 'object' },
          replan_heuristics: { type: 'object' },
          max_attempts: { type: 'integer' },
        },
        required: ['cuda_version', 'target_architecture', 'kernel_spec'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!setupResult) {
    log('Setup failed');
    return { success: false, reason: 'setup_failed' };
  }

  log(`Target: ${setupResult.kernel_spec.operation} on ${setupResult.target_architecture}`);
  log(`CUDA ${setupResult.cuda_version}, PyTorch: ${setupResult.pytorch_available ? 'Yes' : 'No'}`);
  log(`Max attempts: ${setupResult.max_attempts || 20}`);

  const kernelSpec = setupResult.kernel_spec;
  const maxAttempts = setupResult.max_attempts || 20;
  const replanHeuristics = setupResult.replan_heuristics || {
    compile_failure_threshold: 2,
    correctness_failure_threshold: 2,
    stagnation_iterations: 3,
  };

  // Orchestration state
  let currentPlan = null;
  let currentCode = null;
  let bestKernel = null;
  let bestPerformance = kernelSpec.baseline_gflops || 0;

  // Counters for replanning triggers
  let consecutiveCompileFailures = 0;
  let consecutiveCorrectnessFailures = 0;
  const performanceHistory = [];

  // Attempt loop
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    log(`\n=== Attempt ${attempt + 1}/${maxAttempts} ===`);

    // ==========================================================================
    // Replanning Decision
    // ==========================================================================

    const shouldReplan = (
      consecutiveCompileFailures >= replanHeuristics.compile_failure_threshold ||
      consecutiveCorrectnessFailures >= replanHeuristics.correctness_failure_threshold ||
      (performanceHistory.length >= replanHeuristics.stagnation_iterations &&
       isStagnant(performanceHistory, replanHeuristics.stagnation_iterations))
    );

    if (shouldReplan && attempt > 0) {
      log('Adaptive replanning triggered');
      phase('Replan');

      const replanResult = await agentRetry(() => agent(
        `Adaptive replanning triggered (Attempt ${attempt + 1}):

Current situation:
- Consecutive compile failures: ${consecutiveCompileFailures}
- Consecutive correctness failures: ${consecutiveCorrectnessFailures}
- Performance stagnation: ${isStagnant(performanceHistory, replanHeuristics.stagnation_iterations)}
- Recent performance: ${performanceHistory.slice(-3).map(p => p.toFixed(2)).join(' → ')} GFLOPS

Current plan summary:
${currentPlan?.plan_summary || 'No plan yet'}

Replanning strategy:
1. Diagnose root cause of failures:
   - Compile failures: likely syntax/API errors or invalid configurations
   - Correctness failures: likely algorithmic bugs or numerical issues
   - Stagnation: current approach hitting fundamental limits
2. Generate alternative approach:
   - Different optimization strategy
   - Different algorithm implementation
   - Different tiling/threading configuration
   - Fallback to simpler baseline if needed
3. Create new plan that avoids previous failure modes

Return JSON:
{
  "diagnosis": "root cause analysis",
  "alternative_approach": "description of new approach",
  "key_changes": ["change1", "change2", ...],
  "new_plan_summary": "summary of new plan"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Replan","ts":"<ts>","status":"done","candidate_id":"attempt-${attempt}","technique":"<the alternative approach you chose>","note":"<root-cause diagnosis + key changes, one line>"}`,
        {
          label: 'Replan',
          phase: 'Replan',
          schema: {
            type: 'object',
            properties: {
              diagnosis: { type: 'string' },
              alternative_approach: { type: 'string' },
              key_changes: { type: 'array', items: { type: 'string' } },
              new_plan_summary: { type: 'string' },
            },
            required: ['diagnosis', 'alternative_approach', 'new_plan_summary'],
          },
        }
      ), { retries: 5, allowNull: true });

      if (replanResult) {
        log(`Replanning: ${replanResult.alternative_approach}`);
        // Reset failure counters after replan
        consecutiveCompileFailures = 0;
        consecutiveCorrectnessFailures = 0;
      }
    }

    // ==========================================================================
    // Phase 2: Plan (Planner Agent)
    // ==========================================================================
    phase('Plan');

    const planContext = shouldReplan && attempt > 0
      ? `Replanned approach from previous attempt`
      : `Initial planning`;

    log(`Planner: ${planContext}...`);

    const planResult = await agentRetry(() => agent(
      `Generate optimization plan (Attempt ${attempt + 1}):

Kernel specification:
- Operation: ${kernelSpec.operation}
- Shapes: ${kernelSpec.shapes}
- Data types: ${kernelSpec.dtypes.join(', ')}
- Target: ${setupResult.target_architecture}

${shouldReplan && attempt > 0 ? `
Replanning context:
- Previous failures: compile=${consecutiveCompileFailures}, correctness=${consecutiveCorrectnessFailures}
- Performance history: ${performanceHistory.slice(-5).map(p => p.toFixed(2)).join(', ')} GFLOPS
` : ''}

Planning strategy:
1. High-level optimization approach:
   - Memory optimization strategy
   - Compute optimization strategy
   - Threading/block configuration
2. Decompose into implementation steps:
   - Data loading and layout
   - Core computation kernel
   - Memory hierarchy usage (shared memory, registers)
   - Output writing
3. Identify key optimizations:
   - Coalesced memory access
   - Shared memory usage
   - Register blocking
   - Instruction-level parallelism
   - Warp-level primitives
4. Specify constraints:
   - Resource limits (registers, shared memory)
   - Correctness requirements
   - Performance targets

Return JSON:
{
  "attempt": ${attempt + 1},
  "plan_summary": "high-level plan summary",
  "optimization_approach": "memory-bound|compute-bound|balanced",
  "key_strategies": [
    "strategy1",
    "strategy2",
    ...
  ],
  "implementation_steps": [
    {"step": 1, "description": "step description"},
    ...
  ],
  "threading_config": {
    "block_size": "block dimension",
    "grid_size": "grid dimension",
    "threads_per_block": <int>
  },
  "memory_strategy": "description of memory usage",
  "expected_bottleneck": "memory|compute|latency"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Plan","ts":"<ts>","status":"done","candidate_id":"attempt-${attempt}","technique":"<optimization_approach + top key strategy>","speedup":null,"note":"<plan summary + expected bottleneck, one line>"}`,
      {
        label: `Plan attempt ${attempt + 1}`,
        phase: 'Plan',
        schema: {
          type: 'object',
          properties: {
            attempt: { type: 'integer' },
            plan_summary: { type: 'string' },
            optimization_approach: { type: 'string' },
            key_strategies: { type: 'array', items: { type: 'string' } },
            implementation_steps: { type: 'array' },
            threading_config: { type: 'object' },
            memory_strategy: { type: 'string' },
            expected_bottleneck: { type: 'string' },
          },
          required: ['attempt', 'plan_summary', 'key_strategies', 'implementation_steps'],
        },
      }
    ), { retries: 5, allowNull: true });

    if (!planResult) {
      log('Planning failed, skipping this attempt');
      continue;
    }

    currentPlan = planResult;
    log(`Plan: ${planResult.plan_summary}`);
    log(`Strategies: ${planResult.key_strategies.join(', ')}`);

    // ==========================================================================
    // Phase 3: Code (Coder Agent)
    // ==========================================================================
    phase('Code');

    log('Coder: Generating CUDA kernel...');

    const codeResult = await agentRetry(() => agent(
      `Generate ${langToken(LEGACY_CODE_LANG_TOKEN)} kernel implementation (Attempt ${attempt + 1}):

Plan to implement:
${JSON.stringify(currentPlan, null, 2)}

Code generation:
1. Implement complete ${langToken(LEGACY_CODE_LANG_TOKEN)} kernel following the plan
2. ${codeFormatHint()}
3. Implement all steps from the plan:
${currentPlan.implementation_steps.map((s, idx) => `   ${idx + 1}. ${s.description}`).join('\n')}
4. Apply key optimizations:
${currentPlan.key_strategies.map((s, idx) => `   - ${s}`).join('\n')}
5. Include host launch code

Return JSON:
{
  "attempt": ${attempt + 1},
  "kernel_code": "complete ${langToken(LEGACY_CODE_LANG_TOKEN)} kernel code",
  "host_code": "host launch code",
  "kernel_name": "kernel function name",
  "implementation_notes": "notes on implementation choices"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Code","ts":"<ts>","status":"done","candidate_id":"attempt-${attempt}","technique":"<the main optimization you implemented this attempt>","note":"<kernel name + key implementation choices, one line>"}`,
      {
        label: `Code attempt ${attempt + 1}`,
        phase: 'Code',
        schema: {
          type: 'object',
          properties: {
            attempt: { type: 'integer' },
            kernel_code: { type: 'string' },
            host_code: { type: 'string' },
            kernel_name: { type: 'string' },
            implementation_notes: { type: 'string' },
          },
          required: ['attempt', 'kernel_code', 'kernel_name'],
        },
      }
    ), { retries: 5, allowNull: true });

    if (!codeResult) {
      log('Code generation failed, skipping this attempt');
      consecutiveCompileFailures++;
      continue;
    }

    currentCode = codeResult;
    log(`Generated kernel: ${codeResult.kernel_name}`);

    // ==========================================================================
    // Layer-A driver envelope (USE_DRIVER only): build -> run -> profile ->
    // to_evidence -> diagnose -> anti_cheat. Maps onto Verify schema below.
    // ==========================================================================
    let driverEnvelope = null;
    if (USE_DRIVER_STANDALONE) {
      const kPath = attemptKernelPath(attempt);
      const buildOut = `${WORKSPACE}/attempt_${attempt}/artifact`;
      const profOut = `${WORKSPACE}/attempt_${attempt}/profile.native`;
      await agentRetry(() => agent(
        `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
        `Return its stdout JSON verbatim.`,
        { model: MODEL.mechanical, label: `driver-build-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5 });
      const runOut = await agentRetry(() => agent(
        `${driverSh('run.sh', `--artifact ${buildOut} --out ${buildOut}.run.json`)}\n` +
        `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
        { model: MODEL.profile, label: `driver-run-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true });
      // profiling-strategist gate: classify this attempt's kernel (op_class+size),
      // resolve the method via the shared strategist (cached), then branch.
      const _pd = await agentRetry(() => agent(
        `Read ${kPath}; classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then run exactly: ` +
        `\`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${driverPath('manifest.json')} --task <op_class> --size <size> --cache ${EXPDIR}/prof_cache.json --trajectory ${EXPDIR}/genome.jsonl\`.\n` +
        `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
        { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true });
      if (_pd && _pd.method) PROFILING_DECISION = _pd
      let evidenceOut = null;
      if (PROFILING_DECISION.method === 'native_profiler') {
        await agentRetry(() => agent(
          `${driverSh('profile.sh', `--artifact ${buildOut} --out ${buildOut}.run.json --out ${profOut}`)}\n` +
          `Return {ok, native_path}.`,
          { model: MODEL.profile, label: `driver-profile-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5 });
        evidenceOut = await agentRetry(() => agent(
          `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
          `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
          { model: MODEL.mechanical, label: `driver-to-evidence-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true });
      } else {
        // Else path: rely on the existing run.sh/latency result; when method==='perf_heuristic',
        // normalize throughput via the strategist normalizer (substrate profiling/<normalizer>),
        // tagging bottlenecks evidence='profile_heuristic', confidence from the decision.
        const _normalizer = PROFILING_DECISION.normalizer || 'perf_to_evidence.py'
        log(`Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); skipping native profiler.`);
        if (PROFILING_DECISION.method === 'perf_heuristic') {
          evidenceOut = await agentRetry(() => agent(
            `Normalize the run.sh throughput into canonical metrics. Run exactly: ` +
            `\`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${_normalizer} --baseline ${WORKSPACE}/attempt_${attempt}/result.json --peak-gflops <device_peak_gflops> --peak-gbs <device_peak_gbs>\`.\n` +
            `First write the run.sh result to ${WORKSPACE}/attempt_${attempt}/result.json from: ${JSON.stringify(runOut || {})}\n` +
            `Also write heuristic_bclass (memory_bound|compute_bound|latency_bound) based on the throughput ratio so diagnose.py does not fall to unknown. ` +
            `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, heuristic_bclass, coverage, source_backend}. ` +
            `Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.`,
            { model: MODEL.mechanical, label: `driver-perf-heuristic-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true });
        }
      }
      const diagOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
        `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
        { model: MODEL.mechanical, label: `driver-diagnose-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true });
      const antiCheatOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --source ${kPath} --metrics ${buildOut}.run.json\`.\n` +
        `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
        { model: MODEL.mechanical, label: `driver-anti-cheat-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5 });
      driverEnvelope = {
        anti_cheat: antiCheatOut || {},
        metrics: (evidenceOut && evidenceOut.metrics) || {},
        vendor: (DRIVER && DRIVER.hw_vendor) || '',
        coverage: (evidenceOut && evidenceOut.coverage) || [],
        bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
        latency_ms: Number((runOut && runOut.latency_ms) || 0),
        backend_id: DRIVER_BACKEND_ID,
      };
    } else if (IS_EMBEDDED) {
      // --- Embedded eval (integration-strategist -> embedded_inplace / embedded_dispatch) ---
      // Serial by construction: this whole block runs inside the `for (let attempt...)`
      // attempt loop, which is sequential (Planner/Coder/Verifier, no `await parallel(`).
      // So there is no race on the shared project file (inplace) or project build (dispatch).
      const kPath = attemptKernelPath(attempt);
      const variant = `stitch_${attempt}`.replace(/[^A-Za-z0-9_]/g, '_');
      let embLatency = 0, embMetrics = {}, embBclass = 'unknown';
      if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
        const embResult = await agentRetry(() => agent(
          `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project kernel: ${KERNEL_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
          `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
          `2. Apply candidate: cp ${kPath} ${KERNEL_PATH}\n3. Build: ${BUILD_CMD}\n4. Test: ${TEST_CMD}\n5. Benchmark: ${BENCH_CMD || TEST_CMD}\n` +
          `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
          `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-inplace-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true });
        embLatency = Number(embResult?.latency_ms || 0);
        embBclass = embResult?.heuristic_bclass || 'unknown';
        embMetrics = embResult?.metrics || { latency_ms: embLatency };
      } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
        const _plan = typeof __embeddedEvalPlan === 'function'
          ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: TEST_CMD, benchmarkCmd: BENCH_CMD || TEST_CMD })
          : null;
        if (_plan) {
          const embResult = await agentRetry(() => agent(
            `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
            `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
            { model: MODEL.mechanical, label: `embedded-dispatch-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true });
          embLatency = Number(embResult?.latency_ms || 0);
          embBclass = embResult?.heuristic_bclass || 'unknown';
          embMetrics = embResult?.metrics || { latency_ms: embLatency };
        }
      }
      driverEnvelope = { latency_ms: embLatency, metrics: embMetrics, bottleneck_class: embBclass, backend_id: 'embedded' };
    }

    // ==========================================================================
    // Phase 4: Verify (Verifier Agent)
    // ==========================================================================
    phase('Verify');

    log('Verifier: Checking correctness and performance...');

    const verifyResult = await agentRetry(() => agent(
      `Verify ${langToken(LEGACY_VERIFY_LANG_TOKEN)} kernel (Attempt ${attempt + 1}):

Kernel to verify:
\`\`\`${fenceToken()}
${codeResult.kernel_code.substring(0, 2500)}${codeResult.kernel_code.length > 2500 ? '\n... (truncated)' : ''}
\`\`\`

Verification process:
1. Compile check:
   - Use the user-provided compile/build contract for ${setupResult.target_architecture}; if none is provided, perform static compileability review only.
   - Check for syntax errors, warnings
   - Verify resource usage (registers, shared memory)
2. Correctness check:
   - Run with test inputs
   - Compare with reference implementation
   - Check numerical accuracy (absolute/relative error)
   - Test edge cases
3. Performance check:
   - Benchmark on ${setupResult.target_architecture}
   - Measure execution time, GFLOPS
   - ${verifyToolHint()}
   - Compare with baseline: ${kernelSpec.baseline_gflops || 'N/A'} GFLOPS
4. KernelBench evaluation (if configured):
   - Run full benchmark suite
   - Aggregate scores

Return JSON:
{
  "attempt": ${attempt + 1},
  "compilation_success": true/false,
  "compilation_errors": ["error1", ...],
  "resource_usage": {
    "registers_per_thread": <int>,
    "shared_memory_bytes": <int>
  },
  "correctness_passed": true/false,
  "correctness_errors": ["error1", ...],
  "max_error": <float>,
  "performance_gflops": <float>,
  "execution_time_ms": <float>,
  "speedup_vs_baseline": <float>,
  "kernelbench_score": <float or null>,
  "verification_passed": true/false,
  "failure_reason": "compilation|correctness|performance|null"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if verification_passed, else "error"; speedup is the measured speedup_vs_baseline number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Verify","ts":"<ts>","status":"<done|error>","candidate_id":"attempt-${attempt}","speedup":<number or null>,"technique":"<technique under test>","note":"<compiled? correct? gflops; or the failure reason>"}`,
      {
        label: `Verify attempt ${attempt + 1}`,
        phase: 'Verify',
        schema: {
          type: 'object',
          properties: {
            attempt: { type: 'integer' },
            compilation_success: { type: 'boolean' },
            compilation_errors: { type: 'array', items: { type: 'string' } },
            resource_usage: { type: 'object' },
            correctness_passed: { type: 'boolean' },
            correctness_errors: { type: 'array', items: { type: 'string' } },
            max_error: { type: 'number' },
            performance_gflops: { type: 'number' },
            execution_time_ms: { type: 'number' },
            speedup_vs_baseline: { type: 'number' },
            verification_passed: { type: 'boolean' },
            failure_reason: { type: ['string', 'null'] },
          },
          required: ['attempt', 'compilation_success', 'correctness_passed', 'verification_passed'],
        },
      }
    ), { retries: 5, allowNull: true });

    if (!verifyResult) {
      log('Verification failed to run');
      consecutiveCompileFailures++;
      continue;
    }

    // Update counters based on verification result
    if (!verifyResult.compilation_success) {
      consecutiveCompileFailures++;
      consecutiveCorrectnessFailures = 0;
      log(`Compilation failed: ${verifyResult.compilation_errors.join(', ')}`);
      continue;
    } else {
      consecutiveCompileFailures = 0;
    }

    if (!verifyResult.correctness_passed) {
      consecutiveCorrectnessFailures++;
      log(`Correctness failed: ${verifyResult.correctness_errors.join(', ')}`);
      continue;
    } else {
      consecutiveCorrectnessFailures = 0;
    }

    // Verification fully passed
    log(`Verification passed: ${verifyResult.performance_gflops.toFixed(2)} GFLOPS (${verifyResult.speedup_vs_baseline.toFixed(2)}x)`);

    performanceHistory.push(verifyResult.performance_gflops);

    // Update best kernel
    if (verifyResult.performance_gflops > bestPerformance) {
      bestPerformance = verifyResult.performance_gflops;
      bestKernel = {
        attempt: attempt + 1,
        plan: currentPlan,
        code: currentCode,
        verification: verifyResult,
      };
      log(`New best kernel: ${bestPerformance.toFixed(2)} GFLOPS`);
    }

    // Early termination if very good performance achieved
    if (verifyResult.speedup_vs_baseline >= 2.0) {
      log('Excellent performance achieved, early termination');
      break;
    }
  }

  // ============================================================================
  // Phase 6: Report
  // ============================================================================
  phase('Report');

  if (!bestKernel) {
    log('No successful kernel found');
    await __exitRestore();
    return { success: false, reason: 'no_successful_kernel' };
  }

  const report = await agentRetry(() => agent(
    `Generate StitchCUDA synthesis report:

Orchestration summary:
- Target: ${kernelSpec.operation} on ${setupResult.target_architecture}
- Total attempts: ${performanceHistory.length} successful / ${maxAttempts} max
- Best performance: ${bestPerformance.toFixed(2)} GFLOPS
- Speedup: ${bestKernel.verification.speedup_vs_baseline.toFixed(2)}x
- Best attempt: ${bestKernel.attempt}

Best kernel plan:
${bestKernel.plan.plan_summary}
Key strategies: ${bestKernel.plan.key_strategies.join(', ')}

Performance trajectory:
${performanceHistory.map((p, idx) => `  Attempt ${idx + 1}: ${p.toFixed(2)} GFLOPS`).join('\n')}

Generate report with:
1. Executive summary
2. Orchestration overview (Planner-Coder-Verifier)
3. Replanning events and reasons
4. Performance progression
5. Best kernel analysis
6. Optimization breakdown

Return JSON:
{
  "summary": "brief summary",
  "total_attempts": ${performanceHistory.length},
  "successful_attempts": ${performanceHistory.length},
  "best_gflops": ${bestPerformance},
  "speedup": ${bestKernel.verification.speedup_vs_baseline},
  "best_attempt": ${bestKernel.attempt},
  "report_path": "path/to/report.md"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","technique":"synthesis_report","speedup":${bestKernel.verification.speedup_vs_baseline},"note":"<best attempt + best gflops + winning strategies, one line>"}`,
    {
      label: 'Generate report',
      phase: 'Report',
      schema: {
        type: 'object',
        properties: {
          summary: { type: 'string' },
          total_attempts: { type: 'integer' },
          successful_attempts: { type: 'integer' },
          best_gflops: { type: 'number' },
          speedup: { type: 'number' },
          best_attempt: { type: 'integer' },
          report_path: { type: 'string' },
        },
        required: ['summary', 'best_gflops'],
      },
    }
  ), { retries: 5, allowNull: true });

  // ============================================================================
  // Return final results
  // ============================================================================

  await __exitRestore();

  return {
    success: true,
    method: 'StitchCUDA',
    approach: 'Three-agent orchestration (Planner-Coder-Verifier)',
    kernel: kernelSpec.operation,
    target_architecture: setupResult.target_architecture,
    cuda_version: setupResult.cuda_version,
    total_attempts: performanceHistory.length,
    successful_attempts: performanceHistory.length,
    baseline_gflops: kernelSpec.baseline_gflops,
    best_gflops: bestPerformance,
    speedup: bestKernel.verification.speedup_vs_baseline,
    best_attempt: bestKernel.attempt,
    best_plan: bestKernel.plan.plan_summary,
    best_strategies: bestKernel.plan.key_strategies,
    performance_trajectory: performanceHistory,
    final_kernel: bestKernel.code.kernel_code,
    report: report?.report_path,
    summary: report?.summary,
  };
}

// Helper function to detect performance stagnation
function isStagnant(history, windowSize) {
  if (history.length < windowSize) return false;
  const recentWindow = history.slice(-windowSize);
  const avgRecent = recentWindow.reduce((a, b) => a + b, 0) / windowSize;
  const variance = recentWindow.reduce((sum, val) => sum + Math.pow(val - avgRecent, 2), 0) / windowSize;
  const coefficientOfVariation = Math.sqrt(variance) / avgRecent;
  // Stagnant if coefficient of variation < 5%
  return coefficientOfVariation < 0.05;
}

// Execute the workflow
return await main();
