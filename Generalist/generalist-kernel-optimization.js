export const meta = {
  name: 'generalist-kernel-optimization',
  description: 'Best-of-breed single solver built on the KerSor Solver SDK substrate: deterministic diagnosis, persistent state-keyed memory, method gating, anti-cheat reward, and a beam controller. Directly callable; also the strongest default member / hardest baseline for KerSor.',
  whenToUse: 'When you want one directly-callable kernel optimizer that internalizes the universal best-of-breed components (the 7 composable axes) with a single beam search topology. For cross-topology coverage, orchestrate with KerSor instead.',
  phases: [
    { title: 'Setup', detail: 'Read kernel + baseline, init beam, locate substrate scripts' },
    { title: 'Profile', detail: 'Profile current best (native profiler/benchmark) into normalized metrics' },
    { title: 'Diagnose', detail: 'diagnose.py -> shared bottleneck_class (Layer C)' },
    { title: 'Retrieve', detail: 'memory_store.py retrieve + method_gate.py allowed_methods (Layers D, E)' },
    { title: 'Plan', detail: 'Generate BREADTH plans, gated to allowed_methods, with grounded anchors' },
    { title: 'Evaluate', detail: 'Implement + eval each plan; anti_cheat.py reward; evidence_schema.py (Layers A, B)' },
    { title: 'Learn', detail: 'memory_store.py update with measured outcome; record dead-ends; beam top-K' },
    { title: 'Report', detail: 'Final report + Layer A evidence envelope' },
  ],
}

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

// __modelTierApplied (declaration pre-existing)

const WORKFLOW_NAME = 'generalist-kernel-optimization'


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
// Generalist Kernel Optimization — KerSor Solver SDK reference solver
// =============================================================================
//
// Topology: beam (the single chosen controller; topologies are exclusive — see
// _substrate/SOLVER-AUDIT.md). All best-of-breed COMPONENTS are delegated to the
// deterministic substrate scripts under _substrate/, run by agent Bash steps —
// the scripts are the fidelity anchor, not the prompts.
//
//   Layer A  evidence_schema.py   canonical attempt evidence (KerSor-native)
//   Layer B  anti_cheat.py        validity gate + robust reward (beat eager AND compile)
//   Layer C  diagnose.py          metrics -> shared bottleneck_class
//   Layer D  memory_store.py      persistent state-keyed memory (compounds across runs)
//   Layer E  method_gate.py       bottleneck_class -> allowed_methods
//   Layer F  (inline JS)          ceiling / stagnation / early-termination gates
//
// Usage:
//   Workflow({ name: 'generalist-kernel-optimization', args: {
//     kernel_path: '/path/to/kernel.cu',
//     op_description: 'Quantized GEMM Q4_0 weight x FP32 activation',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//       // benchmark_command MUST write JSON: {compiled, correct, candidate_latency_ms,
//       //   eager_latency_ms, compile_latency_ms, speedup, metrics:{dram_pct,sm_pct,occupancy,latency_ms}}
//     ncu_command: '<user-provided profiler command>', // optional; else metrics come from benchmark_command
//     substrate_dir: '/path/to/Awesome-Kernel-Workflows/_substrate',
//     substrate_command_prefix: '<user-provided runner for substrate scripts>',
//     exp_dir: '/path/to/experiment/output',
//     memory_db: '/path/to/experiment/memory.json', // persistent; defaults to exp_dir/memory.json
//     iterations: 3, breadth: 3, topk: 3, target_speedup: 1.5,
//   }})
//
// =============================================================================

// ---- Args / contract ----
let KERNEL_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const OP = args.op_description || 'kernel optimization'
const EVAL_CMD = args.benchmark_command
const NCU_CMD = args.ncu_command || ''
const EXP_DIR = args.exp_dir || '.'
const MEMORY_DB = args.memory_db || `${EXP_DIR}/memory.json`
const ITERATIONS = args.iterations || 3
const BREADTH = args.breadth || 3
const TOPK = args.topk || 3
const TARGET = args.target_speedup || 1.5
const LANGUAGE = args.language || 'cuda'
const TARGET_GPU = args.target_gpu || 'unknown GPU'
const SEED_CANDIDATES = args.seed_candidates || 3
const STAGNATION_EPS = 0.02   // < 2% improvement counts as no progress (Xe-Forge/KSearch)
const STAGNATION_LIMIT = 2    // consecutive stagnant rounds -> stop

// --- Project-native integration (embedded kernels, e.g. llama.cpp / PyTorch aten / vLLM,
//     via integration-strategist). Backend- and framework-agnostic: the strategist routes
//     on can_compile_standalone + host caps; nothing here is llama.cpp-specific. ---
const PROJECT_ROOT = args.project_root || args.ggml_root || ''  // ggml_root is a llama.cpp compat alias; project_root is canonical
const BUILD_CMD = args.build_command || ''
const BUILD_DIR = args.build_dir || ''  // optional CMake build dir for force-recompile-dependents
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || EVAL_CMD || ''
const REGISTER_SCRIPT = args.register_script || ''
const REGISTER_PARAMS = args.register_params || ''

function integrationHostProbeJson() {
  return JSON.stringify({
    compiler: true,
    project_build: !!BUILD_CMD,
    register_script: !!REGISTER_SCRIPT,
    runtime_registry: false,
    reversibility_net: true,
  })
}

function embeddedInplaceEvalBlock(kernelPath, runDir, originalBackup) {
  if (!BUILD_CMD || !TEST_CMD) {
    return '\n# embedded_inplace requires build_command and test_command; cannot eval without them.\n'
  }
  const bench = BENCH_CMD || TEST_CMD
  // Force-recompile guard: when the candidate is a header (.cuh / .h / …),
  // CMake/Ninja's dependency scan can miss the .cu translation units that
  // include it (depfile lag, same-second mtime, NVCC dep races). Inlined as
  // bash so the workflow does not require an external KerSor helper at the
  // AKW-submodule layer.
  const headerExts = /\.(cuh|h|hh|hpp|cuhpp|hxx)$/i
  const isHeader = headerExts.test(kernelPath)
  const headerDir = isHeader ? `$(dirname ${kernelPath})` : ''
  const touchSiblings = isHeader
    ? `2a. Force sibling TU recompile (defeats header-mtime races): find ${headerDir} -maxdepth 1 -type f \\( -name '*.cu' -o -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \\) -exec touch -c {} + ; echo touched siblings under ${headerDir}`
    : ''
  // Stale-object purge when build_dir is known: any .d depfile that
  // references the kernel path -> remove the corresponding .o so the next
  // build cannot reuse a stale object.
  const purgeStaleObjects = BUILD_DIR
    ? `2b. Purge stale objects keyed on this header: find ${BUILD_DIR} -type f -name '*.d' -exec grep -lF ${kernelPath} {} + 2>/dev/null | while IFS= read -r d; do rm -f "\${d%.d}.o" "\${d%.d}"; done; true`
    : ''
  const steps = [
    '',
    '# EMBEDDED-INPLACE EVALUATION (integration-strategist -> embedded_inplace; SERIAL — never run concurrently with another candidate)',
    `Candidate: ${runDir}/kernel | project kernel file: ${kernelPath} | pristine backup: ${originalBackup}`,
    'Run IN ORDER:',
    `1. Restore pristine FIRST (recovers if a prior candidate left the file dirty): cp -a ${originalBackup} ${kernelPath}`,
    `2. Apply candidate AND bump mtime: cp ${runDir}/kernel ${kernelPath} && touch -c ${kernelPath}`,
  ]
  if (touchSiblings) steps.push(touchSiblings)
  if (purgeStaleObjects) steps.push(purgeStaleObjects)
  steps.push(
    `3. Build: ${BUILD_CMD}`,
    `4. Test: ${TEST_CMD}`,
    `5. Benchmark: ${bench}`,
    `6. ALWAYS restore pristine: cp -a ${originalBackup} ${kernelPath} (project must be byte-exact pristine after eval)`,
    'Parse compiled/correct/latency/speedup strictly from command output; do NOT fabricate.',
  )
  return steps.join('\n')
}

function embeddedDispatchEvalBlock(kernelPath, runDir, variant) {
  if (!REGISTER_SCRIPT || !PROJECT_ROOT || !BUILD_CMD || !TEST_CMD) {
    return '\n# embedded_dispatch requires register_script, project_root, build_command, test_command.\n'
  }
  const plan = __embeddedEvalPlan({
    adapter: `python3 "${REGISTER_SCRIPT}"`,
    variant,
    source: `${runDir}/kernel`,
    projectRoot: PROJECT_ROOT,
    params: REGISTER_PARAMS,
    buildCmd: BUILD_CMD,
    testCmd: TEST_CMD,
    benchmarkCmd: BENCH_CMD || TEST_CMD,
  })
  return [
    '',
    '# EMBEDDED-DISPATCH EVALUATION (integration-strategist -> embedded_dispatch)',
    EMBEDDING_CONTRACT,
    `Reference dispatch signature: ${kernelPath}`,
    'Run IN ORDER:',
    `1. Register: ${plan.register}`,
    `2. List: ${plan.list}`,
    `3. Build: ${plan.build}`,
    `4. Test: ${plan.test}`,
    `5. Benchmark: ${plan.benchmark}`,
    `6. Unregister: ${plan.unregister}`,
    `7. List: ${plan.list} (confirm variant is GONE)`,
    plan.cleanupInvariant,
  ].join('\n')
}

function integrationEvalBlock(integMethod, kernelPath, runDir, variant, originalBackup) {
  if (integMethod === 'embedded_dispatch') return embeddedDispatchEvalBlock(kernelPath, runDir, variant)
  if (integMethod === 'embedded_inplace') return embeddedInplaceEvalBlock(kernelPath, runDir, originalBackup)
  return ''
}

// --- Backend driver wiring (P5e Stage B; off-by-default; legacy path byte-identical) ---
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

if (!KERNEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

// P0 — model/intelligence routing: mechanical steps don't need Opus (token savings).
const MODEL = {
  mechanical: args.model_mechanical || 'haiku',  // run substrate scripts, parse JSON
  profile: args.model_profile || 'sonnet',       // run eval/ncu, normalize metrics
  judgment: args.model_judgment || 'opus',       // plan / implement / report (override per kernel complexity)
}

function substrateInstruction(script, cliArgs) {
  const scriptPath = `${SUBSTRATE}/${script}`
  if (!PY) {
    return `No substrate_command_prefix provided for ${scriptPath} ${cliArgs}; do not invent an interpreter.`
  }
  return `Run exactly: \`${PY} ${scriptPath} ${cliArgs}\`.`
}
// P0 — token-budget wiring: rough per-unit output-token estimates for scaling + stopping.
const EST_PER_CANDIDATE = args.est_tokens_per_candidate || 20000
const EST_PER_ROUND = EST_PER_CANDIDATE * BREADTH + 15000
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null

// Schemas for structured agent returns
const METRICS_SCHEMA = {
  type: 'object',
  properties: {
    compiled: { type: 'boolean' }, correct: { type: 'boolean' },
    candidate_latency_ms: { type: ['number', 'null'] },
    eager_latency_ms: { type: ['number', 'null'] },
    compile_latency_ms: { type: ['number', 'null'] },
    speedup: { type: ['number', 'null'] },
    metrics: { type: 'object' },
  },
  required: ['compiled', 'correct', 'speedup', 'metrics'],
}
const ANTICHEAT_SCHEMA = {
  type: 'object',
  properties: {
    valid: { type: 'boolean' }, reward: { type: 'number' },
    recorded_speedup: { type: 'number' }, measured_speedup: { type: 'number' },
    reward_reason: { type: 'string' },
    flags: { type: 'array' }, blocking_flags: { type: 'array' },
  },
  required: ['valid', 'reward', 'recorded_speedup', 'measured_speedup'],
}

phase('Setup')
log(`Generalist solver | beam | breadth=${BREADTH} topk=${TOPK} iters=${ITERATIONS} target=${TARGET}x | models ${MODEL.mechanical}/${MODEL.profile}/${MODEL.judgment} | budget ${(typeof budget !== 'undefined' && budget.total) ? Math.round(budget.total / 1000) + 'k' : 'unbounded'}`)

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
  const generated = await agentRetry(() => agent(`No kernel_path was provided. Generate and verify an initial kernel before seeding the Generalist candidate beam.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- op_description: ${OP}
- language: ${langToken(LANGUAGE)}
- target_gpu: ${TARGET_GPU}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- benchmark_command: ${EVAL_CMD || '(not provided)'}
- ncu_command: ${NCU_CMD || '(not provided)'}

# Contract
Generate ${SEED_CANDIDATES} complete candidates under ${EXP_DIR}/generated/. Run benchmark_command if available using {kernel_path}/{result_path}. Return the best verified generated kernel path.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"seed_generation","speedup":<best seed verified speedup as number or null>,"note":"<how many of ${SEED_CANDIDATES} seeds verified + the chosen seed path, one line>"}`, {
    label: 'generate-initial-kernel',
    phase: 'Setup',
    model: MODEL.judgment,
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
  KERNEL_PATH = generatedKernelPath
}

// Baseline candidate seeds the beam.
let candidateBeam = [{ id: 'baseline', parent_id: null, code_path: KERNEL_PATH, speedup: 1.0, metrics: {}, planTitle: 'baseline' }]
let bestSpeedup = 1.0
let stagnantRounds = 0
let dryRounds = 0                 // P1.5 — consecutive rounds with no new 'measured' insight
const DRY_LIMIT = 2               // P1.5 — loop-until-dry stop threshold
const allAttempts = []
const verifiedInsights = []       // P1.4 — verified typed insights for the Layer A envelope

// --- integration-strategist: route build/test mode (standalone vs embedded_*).
// TRUST BOUNDARY (was bug 20260620-153001): the agent must NOT both classify AND
// "report" the script's decision in one shot — it can lie about either. Split into:
//   (1) heuristic OR classify-only agent  -> a fixed can_standalone value
//   (2) mechanical agent runs the python with that fixed CLI; stdout is parroted
//       AND the resulting integ_cache.json is read as authoritative.
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const probe = integrationHostProbeJson()
  const rootCli = PROJECT_ROOT ? ` --project-root "${PROJECT_ROOT}"` : ''
  const decisionPath = `${EXP_DIR}/integration_decision.json`
  const cachePath = `${EXP_DIR}/integ_cache.json`

  // (1) Decide can_standalone. Order: explicit user override (args.can_standalone) >
  // heuristic (header-like kernel WITH project build/test/root cannot be a single TU) >
  // classify agent. The override is the escape hatch for the rare standalone-compilable
  // .cuh that the heuristic would otherwise force to embedded_inplace.
  let canStandalone = 'uncertain'
  const cuhLike = /\.(cuh|cu\.h|inl|hpp|hh)$/i.test(KERNEL_PATH || '')
  const _userCS = typeof args.can_standalone === 'string' ? args.can_standalone.trim().toLowerCase() : ''
  if (_userCS === 'yes' || _userCS === 'no' || _userCS === 'uncertain') {
    canStandalone = _userCS
    log(`integration: can_standalone=${canStandalone} (user override via args.can_standalone)`)
  } else if (cuhLike && PROJECT_ROOT && BUILD_CMD && TEST_CMD) {
    canStandalone = 'no'
    log(`integration: heuristic forces can_standalone=no (header-like kernel + project build/test/root present)`)
  } else {
    const cls = await agentRetry(() => agent(
      `Read ${KERNEL_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
      `(use 'no' when the file cannot compile as a single translation unit — e.g. a llama.cpp .cuh ` +
      `with project-only deps, template-only headers, or kernels referenced through dispatch tables). ` +
      `Return ONLY the JSON {"can_standalone":"yes|no|uncertain","reason":"<one line>"}. ` +
      `Do not run any script; do not invent any other field.`,
      { model: MODEL.mechanical, label: 'integration-classify', phase: 'Setup',
        schema: { type: 'object',
                  properties: { can_standalone: { type: 'string' }, reason: { type: 'string' } },
                  required: ['can_standalone'] } }), { retries: 5, allowNull: true })
    canStandalone = (cls && cls.can_standalone) || 'uncertain'
  }
  log(`integration: can_standalone=${canStandalone}`)

  // (2) Mechanical runner — fixed CLI value; agent has no judgment surface.
  if (PY) {
    const _integ = await agentRetry(() => agent(
      `Run exactly: \`${PY} ${SUBSTRATE}/integration/integration_strategist.py resolve ` +
      `--kernel "${KERNEL_PATH}"${rootCli} --can-standalone ${canStandalone} ` +
      `--host-probe '${probe}' --cache ${cachePath} --trajectory ${EXP_DIR}/genome.jsonl ` +
      `> ${decisionPath}\`. ` +
      `Then run exactly: \`cat ${decisionPath}\` and return its stdout JSON verbatim. ` +
      `Do not modify, summarize, or invent any field.`,
      { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    if (_integ && _integ.method) INTEGRATION_DECISION = _integ
  } else {
    log('integration: no substrate_command_prefix set; keeping default standalone routing')
  }
}
log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error(
    'integration-strategist returned derive_adapter — provide project_root + build/test commands, ' +
    'or run /kersor:integrate to derive an adapter first'
  )
}
const USE_DRIVER_STANDALONE = USE_DRIVER && INTEGRATION_DECISION.method === 'standalone'

// embedded_inplace mutates the SHARED KERNEL_PATH (e.g. llama.cpp mmq.cuh). Back up the
// ORIGINAL once here so every candidate restores to pristine — never to a prior candidate's
// leftover. Serialized eval (Evaluate phase) + restore-pristine-first make this race-free.
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(
    `Byte-exact backup of the original embedded kernel: run \`cp -a "${KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm it succeeded. ` +
    `This backup is the pristine restore target for every candidate eval and the final exit restore. Do not modify it.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it below. The agent only classifies the task (fuzzy); the substrate stamps
// confidence by method (measured/inferred/hypothesized) — not the agent. See
// _substrate/profiling/README.md. Falls back to native_profiler if undecided. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }
{
  // Resolve for ALL integration modes (not only driver-path) so the per-iteration
  // Profile below honors the substrate ladder even on embedded/legacy runs. Default
  // the manifest to the cuda backend when no driver dir is present.
  const _profManifest = (USE_DRIVER && BACKEND_DIR) ? `${BACKEND_DIR}/manifest.json` : `${SUBSTRATE}/backends/cuda/manifest.json`
  const _pd = await agentRetry(() => agent(
    `Read ${KERNEL_PATH}; classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    substrateInstruction('profiling/profiling_strategist.py',
      `resolve --backend-manifest ${_profManifest} --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl`) +
    ` Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}

// A-O1 closure (K-O2 ↔ profiling-strategist SSOT): the strategist probes `which ncu`
// (binary present) and may pick native_profiler even when NCU hw counters are
// admin-blocked. KerSor harness-preflight (K-O2) detects the block and STRIPS
// ncu_command from dispatch-args. So if the strategist says native_profiler but no
// ncu_command arrived (stripped by preflight, or never provided), downgrade to
// perf_heuristic — the native path is unusable. This closes the loop without a
// cross-repo host-probe contract: preflight's strip is the signal.
if (PROFILING_DECISION.method === 'native_profiler' && !NCU_CMD) {
  log(`profiling: strategist chose native_profiler but ncu_command is absent (stripped by ` +
      `preflight because counters blocked, or not provided) -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred',
    normalizer: 'perf_to_evidence.py', profiler_name: 'test-harness-perf',
    rationale: 'native_profiler selected but ncu_command absent (K-O2 strip / not provided) -> perf_heuristic' }
}

function nsysEnrichSuffix() {
  if (USE_DRIVER && DRIVER_BACKEND_ID !== 'cuda') return ''
  return (
    `Path B (nsys enrich): if \`nsys\` is on PATH, you MAY run a short nsys timeline profile wrapping the same harness ` +
    `(e.g. \`nsys profile -o ${EXP_DIR}/prof.nsys-rep --trace=cuda,nvtx --sample=none --force-overwrite=true <harness argv>\`) ` +
    `to locate hot kernels and decode-vs-batch time share. nsys is enrich-only: do NOT claim dram_pct/sm_pct from nsys; ` +
    `keep evidence='profile_heuristic' and confidence='inferred' — never upgrade to measured.\n`
  )
}

function perfHeuristicProfileHint(evalCmd) {
  const profilerTerm = USE_DRIVER ? 'the native profiler' : 'ncu'
  const measurementInstruction = USE_DRIVER
    ? `Use the substrate driver run result for throughput and derive memory-vs-compute-bound hints from it; `
    : `Run the benchmark command \`${evalCmd}\` for throughput and derive memory-vs-compute-bound hints from it; `
  return (
    `Profiling-strategist chose method='perf_heuristic' (confidence='${PROFILING_DECISION.confidence}'); do NOT run ${profilerTerm}. ` +
    measurementInstruction +
    `tag any bottleneck evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
    `In metrics, set evidence='profile_heuristic' and either heuristic_bclass (memory_bound|compute_bound|latency_occupancy|overhead_bound) ` +
    `or bottleneck_hint (short free text, e.g. "L2/memory bound at 55% SM"); use null heuristic_bclass only if truly unknown. ` +
    nsysEnrichSuffix()
  )
}

async function runDriverMetricsEnvelope({ suffix, phaseName, kernelPath, artifactPath, resultPath, profilePath }) {
  await agentRetry(() => agent(
    `${driverSh('build.sh', `--source ${kernelPath} --out ${artifactPath}`)}\n` +
    `Return its stdout JSON verbatim.`,
    { model: MODEL.mechanical, label: `driver-build-${suffix}`, phase: phaseName, schema: JSON_PASSTHROUGH }), { retries: 5 })
  const runOut = await agentRetry(() => agent(
    `${driverSh('run.sh', `--artifact ${artifactPath} --problem ${PROBLEM_PATH} --out ${artifactPath}.run.json`)}\n` +
    `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
    { model: MODEL.profile, label: `driver-run-${suffix}`, phase: phaseName, schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  const profileOut = await agentRetry(() => agent(
    PROFILING_DECISION.method === 'native_profiler'
      ? `${driverSh('profile.sh', `--artifact ${artifactPath} --problem ${PROBLEM_PATH} --out ${artifactPath}.run.json --out ${profilePath}`)}\n` +
        `Return {ok, native_path}.`
      : `Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run the native profiler. ` +
        `Use driver-run-${suffix} throughput as the profiling source and return {ok:true, native_path:null, method:'${PROFILING_DECISION.method}', latency_ms:${(runOut && runOut.latency_ms) || 'null'}}.`,
    { model: MODEL.profile, label: `driver-profile-${suffix}`, phase: phaseName, schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  let evidenceOut = null
  if (PROFILING_DECISION.method === 'native_profiler') {
    const nativePath = (profileOut && (profileOut.native_path || profileOut.native_profile)) || profilePath
    evidenceOut = await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${nativePath}\`.\n` +
      `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
      { model: MODEL.mechanical, label: `driver-to-evidence-${suffix}`, phase: phaseName, schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  } else if (PROFILING_DECISION.method === 'perf_heuristic') {
    const normalizer = PROFILING_DECISION.normalizer || 'perf_to_evidence.py'
    evidenceOut = await agentRetry(() => agent(
      `Profiling-strategist chose method='perf_heuristic' (confidence='${PROFILING_DECISION.confidence}'); normalize driver-run-${suffix} throughput with the substrate normalizer. ` +
      `${substrateInstruction('profiling/' + normalizer, `--baseline ${resultPath} --peak-gflops <device_peak_gflops> --peak-gbs <device_peak_gbs>`)} ` +
      `Read device peaks from ${BACKEND_DIR}/manifest.json if present; otherwise estimate and say so. ` +
      `Tag emitted bottlenecks as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
      `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
      { model: MODEL.mechanical, label: `driver-to-evidence-${suffix}`, phase: phaseName, schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  } else {
    evidenceOut = await agentRetry(() => agent(
      `Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do not run a native profiler. ` +
      `Return canonical metrics with latency_ms from driver-run-${suffix} when available and null for missing counters.`,
      { model: MODEL.mechanical, label: `driver-to-evidence-${suffix}`, phase: phaseName, schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  }
  await agentRetry(() => agent(
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
    `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
    { model: MODEL.mechanical, label: `driver-diagnose-${suffix}`, phase: phaseName, schema: JSON_PASSTHROUGH }), { retries: 5 })
  await agentRetry(() => agent(
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kernelPath} --result ${resultPath}\`.\n` +
    `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
    { model: MODEL.mechanical, label: `driver-anti-cheat-${suffix}`, phase: phaseName, schema: JSON_PASSTHROUGH }), { retries: 5 })

  const metrics = (evidenceOut && evidenceOut.metrics) || {}
  const latency = metrics.latency_ms != null ? metrics.latency_ms : (runOut && runOut.latency_ms)
  return {
    compiled: runOut && runOut.compiled !== undefined ? !!runOut.compiled : true,
    correct: runOut && runOut.correct !== undefined ? !!runOut.correct : true,
    candidate_latency_ms: latency == null ? null : Number(latency),
    eager_latency_ms: null,
    compile_latency_ms: null,
    speedup: null,
    metrics: { latency_ms: latency == null ? null : Number(latency), ...metrics },
  }
}

// --- Baseline driver envelope (Layer-A, standalone driver-path only) ---
if (USE_DRIVER_STANDALONE) {
  const kPath = KERNEL_PATH || `${EXP_DIR}/baseline.kernel`
  const buildOut = `${EXP_DIR}/baseline.artifact`
  const profOut = `${EXP_DIR}/baseline.prof.native`
  await runDriverMetricsEnvelope({
    suffix: 'setup',
    phaseName: 'Setup',
    kernelPath: kPath,
    artifactPath: buildOut,
    resultPath: `${EXP_DIR}/baseline.result.json`,
    profilePath: profOut,
  })
}

for (let iter = 1; iter <= ITERATIONS; iter++) {
  const best = candidateBeam[0]
  log(`\n=== Iteration ${iter}/${ITERATIONS} | best ${bestSpeedup.toFixed(3)}x | beam ${candidateBeam.length} ===`)

  // P0 — token budget: stop if a full round can't fit; scale breadth to remaining budget.
  if (typeof budget !== 'undefined' && budget.total && budget.remaining() < EST_PER_ROUND) {
    log(`token budget ~exhausted (${Math.round(budget.remaining() / 1000)}k < ${Math.round(EST_PER_ROUND / 1000)}k/round) — stop`)
    break
  }
  const effBreadth = (typeof budget !== 'undefined' && budget.total)
    ? Math.max(1, Math.min(BREADTH, Math.floor(budget.remaining() / EST_PER_CANDIDATE)))
    : BREADTH

  // ---- Profile (Layer C input) ----
  phase('Profile')
  const metrics = USE_DRIVER_STANDALONE
    ? await runDriverMetricsEnvelope({
        suffix: `profile-${iter}`,
        phaseName: 'Profile',
        kernelPath: best.code_path,
        artifactPath: `${EXP_DIR}/run-${iter}/best.artifact`,
        resultPath: `${EXP_DIR}/run-${iter}/best.result.json`,
        profilePath: `${EXP_DIR}/run-${iter}/best.prof.native`,
      })
    : await agentRetry(() => agent(
      `Profile the current best kernel and produce normalized metrics.\n` +
      `Kernel: ${best.code_path}\nOp: ${OP}\n` +
      (PROFILING_DECISION.method === 'native_profiler' && NCU_CMD && EVAL_CMD
        ? `Run the user-provided ncu_command: \`${NCU_CMD}\` and the benchmark command \`${EVAL_CMD}\`.\n`
        : PROFILING_DECISION.method === 'perf_heuristic' && EVAL_CMD
          ? perfHeuristicProfileHint(EVAL_CMD)
          : EVAL_CMD
            ? `Run the benchmark command \`${EVAL_CMD}\` (writes a JSON result file).\n`
            : `No benchmark_command provided; do not invent an evaluator. Return missing/null measured metrics.\n`) +
      `Return the JSON exactly per the schema: compiled, correct, candidate_latency_ms, ` +
      `eager_latency_ms, compile_latency_ms, speedup, and metrics{dram_pct, sm_pct, occupancy, latency_ms}. ` +
      `Use null for unknown numbers. Do not fabricate; missing => null.` +
      `\n\n# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\n` +
      `Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\n` +
      `Then append:\n` +
      `{"workflow":"${WORKFLOW_NAME}","phase":"Profile","ts":"<ts>","status":"done","candidate_id":"iter-${iter}-best","technique":"profile","speedup":<measured speedup as number or null>,"note":"<measured latency_ms + dominant metric (dram_pct/sm_pct/occupancy), one line>"}`,
      { model: MODEL.profile, label: `profile-${iter}`, phase: 'Profile', schema: METRICS_SCHEMA }), { retries: 5 })

  // ---- Diagnose (Layer C, deterministic script) ----
  phase('Diagnose')
  const diag = await agentRetry(() => agent(
    `Write these metrics to ${EXP_DIR}/run-${iter}/metrics.json:\n${JSON.stringify(metrics.metrics || {})}\n` +
    `${substrateInstruction('diagnose.py', `--metrics ${EXP_DIR}/run-${iter}/metrics.json`)} ` +
    `Return its stdout JSON verbatim ({bottleneck_class, evidence}). If the substrate command is unavailable, return {bottleneck_class:"unknown", evidence:"missing_substrate_command_prefix"}.`,
    { label: `diagnose-${iter}`, phase: 'Diagnose', schema: JSON_PASSTHROUGH, model: MODEL.mechanical }), { retries: 5 })
  const bclass = diag.bottleneck_class || 'unknown'
  log(`bottleneck_class = ${bclass}`)

  // ---- Retrieve memory (Layer D) + gate methods (Layer E) ----
  phase('Retrieve')
  const [mem, gate] = await parallel([
    () => agentRetry(() => agent(
      `${substrateInstruction('memory_store.py', `--db ${MEMORY_DB} retrieve --class ${bclass}`)} ` +
      `Return its stdout JSON verbatim ({bottleneck_class, techniques, dead_ends}). If unavailable, return empty techniques/dead_ends with missing evidence.`,
      { label: `retrieve-${iter}`, phase: 'Retrieve', schema: JSON_PASSTHROUGH, model: MODEL.mechanical }), { retries: 5 }),
    () => agentRetry(() => agent(
      `${substrateInstruction('method_gate.py', `--class ${bclass} --metrics ${EXP_DIR}/run-${iter}/metrics.json --op-description ${JSON.stringify(OP)}`)} ` +
      `Return its stdout JSON verbatim ({bottleneck_class, allowed_methods, rationale}). If unavailable, return allowed_methods:[] with missing evidence.`,
      { label: `gate-${iter}`, phase: 'Retrieve', schema: JSON_PASSTHROUGH, model: MODEL.mechanical }), { retries: 5 }),
  ])
  const allowed = (gate && gate.allowed_methods) || []
  const priorTech = (mem && mem.techniques) || []
  const deadEnds = (mem && mem.dead_ends) || []
  log(`allowed_methods = ${allowed.join(', ')} | prior techniques = ${priorTech.length} | dead-ends = ${deadEnds.length}`)

  // ---- Plan: BREADTH gated plans with grounded anchors (STARK borrow) ----
  phase('Plan')
  const planContext =
    `# Bottleneck: ${bclass}\n# Allowed methods (you MUST stay within these): ${allowed.join(', ')}\n` +
    `# Prior techniques by confidence: ${JSON.stringify(priorTech.slice(0, 5))}\n` +
    `# Dead-ends to AVOID (re-allowed only if revalidate_if holds): ${JSON.stringify(deadEnds)}\n` +
    `# Current best: ${best.code_path} @ ${best.speedup.toFixed(2)}x`
  const plans = (await parallel(
    Array.from({ length: effBreadth }, (_, i) => () => agentRetry(() => agent(
      `You are planner #${i + 1}/${effBreadth} for a ${OP} kernel.\n${planContext}\n\n` +
      `Propose ONE optimization plan that uses ONLY an allowed method. Mark the exact code region to change with ` +
      `grounded anchors <<<IMPROVE BEGINS>>> ... <<<IMPROVE ENDS>>>. Pick the highest-confidence prior technique ` +
      `that fits, unless a dead-end forbids it. Return {method, plan, anchors}.` +
      __attemptBlock() +
      __experienceBlock() +
      `\n\n# Recent genome trajectory (read BEFORE picking the method)\n` +
      `Run \`tail -20 ${EXP_DIR}/genome.jsonl 2>/dev/null\` to see prior attempts this session. Use it to: (a) avoid picking a method already tried with regression in earlier rounds, (b) spot multi-round patterns the persistent memory may not surface yet. If the file is empty or missing, ignore this.\n` +
      `\n# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\n` +
      `Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\n` +
      `Then append:\n` +
      `{"workflow":"${WORKFLOW_NAME}","phase":"Plan","ts":"<ts>","status":"done","candidate_id":"iter-${iter}-plan-${i + 1}","technique":"<the allowed method you chose>","speedup":null,"note":"<one-line summary of the plan + why this method fits bottleneck ${bclass}>"}`,
      { label: `plan-${iter}-${i + 1}`, phase: 'Plan', schema: JSON_PASSTHROUGH, model: MODEL.judgment }), { retries: 5 })))
  ).filter(Boolean)

  // ---- Evaluate each plan: implement -> eval -> anti-cheat (Layers B, A) ----
  // worktree isolation is only safe for STANDALONE eval. For embedded_inplace/dispatch
  // the candidate must mutate the project tree (or the registry) at PROJECT_ROOT, and
  // a worktree of cwd (often a non-git scratch dir like LlamaCpp-Exp/) breaks both:
  // (a) cwd may not be a git repo -> "Cannot create agent worktree" hard-fail; (b)
  // even if it were, the worktree is the WRONG tree — build/test still target the
  // real project_root, so the swap+restore would happen to a copy nobody compiles.
  const useWorktree = INTEGRATION_DECISION.method === 'standalone'
  phase('Evaluate')
  const evalOne = async (p, i) => {
    const runDir = `${EXP_DIR}/run-${iter}/cand-${i + 1}`
    const variant = `gen_${iter}_${i + 1}`.replace(/[^A-Za-z0-9_]/g, '_')
    const integBlock = integrationEvalBlock(INTEGRATION_DECISION.method, KERNEL_PATH, runDir, variant, ORIGINAL_BACKUP)
    const embeddedProposal = INTEGRATION_DECISION.method === 'embedded_dispatch'
      ? `\n\n${EMBEDDING_CONTRACT}\nMatch dispatch signature of ${KERNEL_PATH} exactly.\n`
      : ''
    const m = await agentRetry(() => agent(
      `Implement this plan on a COPY of ${best.code_path} into ${runDir}/kernel, respecting the ` +
      `<<<IMPROVE BEGINS/ENDS>>> anchors. Method: ${p.method}. Plan: ${JSON.stringify(p.plan)}.\n` +
      embeddedProposal +
      (integBlock
        ? integBlock + `\nThen map measured results into the JSON metrics schema.`
        : EVAL_CMD
          ? `Then run the benchmark command \`${EVAL_CMD}\` and return the JSON metrics per schema.`
          : `No benchmark_command provided; do not invent an evaluator. Return compiled=false, correct=false, speedup=0, and missing evidence fields.`) +
      `\n\n# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\n` +
      `Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\n` +
      `Then append, using the values you just measured (status="done" if compiled AND correct, else "error"; speedup is the measured speedup number, or null if unavailable):\n` +
      `{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"iter-${iter}-cand-${i + 1}","technique":"${p.method}","speedup":<number or null>,"note":"<compiled? correct? + the failure reason if any, one line>"}`,
      { label: `impl-${iter}-${i + 1}`, phase: 'Evaluate', schema: METRICS_SCHEMA, model: MODEL.judgment, ...(useWorktree ? { isolation: 'worktree' } : {}) }), { retries: 5 })
    const ac = await agentRetry(() => agent(
      `Write these metrics to ${runDir}/metrics.json:\n${JSON.stringify({ ...m, claimed_speedup: m.speedup })}\n` +
      `${substrateInstruction('anti_cheat.py', `--source ${runDir}/kernel --metrics ${runDir}/metrics.json`)} ` +
      `Return its stdout JSON verbatim. Then ${substrateInstruction('evidence_schema.py', `validate ${runDir}/metrics.json`)} ` +
      `If substrate commands are unavailable, mark valid=false with blocking_flags:["missing_substrate_command_prefix"].`,
      { label: `anticheat-${iter}-${i + 1}`, phase: 'Evaluate', schema: ANTICHEAT_SCHEMA, model: MODEL.mechanical }), { retries: 5 })
    return { plan: p, metrics: m, anticheat: ac, code_path: `${runDir}/kernel`,
             recorded_speedup: ac.valid ? ac.recorded_speedup : 0,
             measured_speedup: ac.valid ? (ac.measured_speedup ?? 0) : 0 }
  }
  // Embedded modes mutate a shared file (inplace) or share a project build (dispatch):
  // they MUST evaluate SERIALLY — parallel candidates would clobber KERNEL_PATH / collide
  // on the project build. Standalone candidates are isolated (per-candidate .so) -> parallel.
  const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
  let evaluated
  if (IS_EMBEDDED) {
    evaluated = []
    for (let i = 0; i < plans.length; i++) {
      try { const r = await evalOne(plans[i], i); if (r) evaluated.push(r) }
      catch (e) { log(`iter ${iter} cand ${i + 1} embedded eval failed: ${(e && e.message) || e}`) }
    }
  } else {
    evaluated = (await parallel(plans.map((p, i) => () => evalOne(p, i)))).filter(Boolean)
  }

  // --- Per-iteration driver envelope (Layer-A, standalone driver-path only) ---
  if (USE_DRIVER_STANDALONE) {
    for (let ci = 0; ci < evaluated.length; ci++) {
      const suffix = `${iter}-${ci + 1}`
      const kPath = evaluated[ci].code_path
      const buildOut = `${EXP_DIR}/run-${iter}/cand-${ci + 1}.artifact`
      const profOut = `${EXP_DIR}/run-${iter}/cand-${ci + 1}.prof.native`
      await runDriverMetricsEnvelope({
        suffix,
        phaseName: 'Evaluate',
        kernelPath: kPath,
        artifactPath: buildOut,
        resultPath: `${EXP_DIR}/run-${iter}/cand-${ci + 1}.result.json`,
        profilePath: profOut,
      })
    }
  }

  // ---- Learn: update persistent memory (Layer D) per measured outcome ----
  phase('Learn')
  await parallel(evaluated.map((e) => () => {
    const updateArgs = `--db ${MEMORY_DB} update --class ${bclass} --technique ${e.plan.method} --speedup ${e.metrics.speedup || 0} --correct ${e.metrics.correct ? 1 : 0}`
    const deadendArgs = `--db ${MEMORY_DB} add-deadend --claim ${JSON.stringify(e.plan.method)} --why ${JSON.stringify(e.anticheat.reward_reason || (e.anticheat.blocking_flags || []).join(','))} --revalidate-if "metrics change"`
    return agentRetry(() => agent(
      `${substrateInstruction('memory_store.py', updateArgs)} ` +
      (e.anticheat.valid ? '' : `${substrateInstruction('memory_store.py', deadendArgs)} `) +
      `Return {updated:true} if the command ran. If unavailable, return {updated:false, reason:"missing_substrate_command_prefix"}.`,
      { label: `learn-${iter}-${e.plan.method}`, phase: 'Learn', schema: JSON_PASSTHROUGH, model: MODEL.mechanical }), { retries: 5 })
  }))

  // ---- P1.4: adversarial insight verification (LLM refuter judgment + deterministic downgrade) ----
  const roundInsightRaw = {
    kind: 'bottleneck', directive: 'explore',
    claim: `bottleneck_class=${bclass} dominates ${OP} at this shape`,
    actionable_hint: ({memory_bound: 'reduce DRAM traffic (tile for reuse, coalesce loads, shared memory)',
                      compute_bound: 'raise arithmetic intensity (tensor cores / SIMT-efficient ops, cut redundant work)',
                      latency_bound: 'raise occupancy / cut latency (larger blocks, fewer serial deps, pipeline)'})[bclass] || 're-examine the profile for the dominant cost and pick the next action',
    evidence: NCU_CMD ? 'ncu' : 'benchmark', confidence: 'measured', source_round: iter,
  }
  const refute = await agentRetry(() => agent(
    `Adversarially REFUTE this attribution against the MEASURED profile ` +
    `${JSON.stringify(metrics.metrics || {})}: "${roundInsightRaw.claim}". ` +
    `Default to refuted=true if the data does not clearly support it. Return {refuted, reason}.`,
    { label: `refute-${iter}`, phase: 'Learn', model: MODEL.judgment,
      schema: { type: 'object', properties: { refuted: { type: 'boolean' }, reason: { type: 'string' } }, required: ['refuted'] } }), { retries: 5 })
  const verified = await agentRetry(() => agent(
    `Write this insight to ${EXP_DIR}/run-${iter}/insight.json:\n${JSON.stringify(roundInsightRaw)}\n` +
    `${substrateInstruction('verify_insight.py', `--insight ${EXP_DIR}/run-${iter}/insight.json --refuted ${refute.refuted ? 1 : 0}`)} ` +
    `Return its stdout JSON verbatim. If unavailable, return {verified:false, reason:"missing_substrate_command_prefix"}.`,
    { label: `verify-insight-${iter}`, phase: 'Learn', schema: JSON_PASSTHROUGH, model: MODEL.mechanical }), { retries: 5, allowNull: true })
  verifiedInsights.push(verified)
  const producedMeasured = (verified && verified.confidence) === 'measured'
  log(`round insight confidence: ${(verified && verified.confidence) || 'n/a'}${refute.refuted ? ' (refuted -> downgraded)' : ''}`)

  // ---- Beam update (deterministic JS): merge, keep top-K by recorded speedup ----
  const newCandidates = evaluated
    .filter((e) => e.anticheat.valid && e.recorded_speedup > 0)
    .map((e, i) => ({ id: `it${iter}-c${i + 1}`, parent_id: best.id, code_path: e.code_path,
                      speedup: e.recorded_speedup, metrics: e.metrics.metrics || {}, planTitle: e.plan.method }))
  evaluated.forEach((e) => allAttempts.push({ iter, method: e.plan.method, valid: e.anticheat.valid,
                                              reward: e.anticheat.reward, speedup: e.recorded_speedup,
                                              measured_speedup: e.measured_speedup ?? 0 }))
  const merged = [...candidateBeam, ...newCandidates].sort((a, b) => b.speedup - a.speedup)
  candidateBeam = merged.slice(0, TOPK)

  // ---- Cost-control gates (Layer F, inline) ----
  const newBest = candidateBeam[0].speedup
  const improvement = (newBest - bestSpeedup) / Math.max(bestSpeedup, 1e-9)
  stagnantRounds = improvement < STAGNATION_EPS ? stagnantRounds + 1 : 0
  dryRounds = producedMeasured ? 0 : dryRounds + 1   // P1.5 loop-until-dry
  bestSpeedup = newBest
  log(`iter ${iter}: best now ${bestSpeedup.toFixed(3)}x (Δ ${(improvement * 100).toFixed(1)}%) | stagnant ${stagnantRounds} | dry ${dryRounds}`)
  if (bestSpeedup >= TARGET) { log(`target ${TARGET}x reached — stop (COMPLETE)`); break }
  if (stagnantRounds >= STAGNATION_LIMIT) { log(`stagnation limit reached — stop (STALLED)`); break }
  if (dryRounds >= DRY_LIMIT) { log(`loop-until-dry: ${DRY_LIMIT} rounds with no new measured insight — stop (STALLED)`); break }
}

// embedded_inplace exit safety net: unconditionally restore the pristine original so the
// project worktree is byte-exact on exit regardless of any candidate that failed to restore
// mid-eval. (Serialized eval + restore-pristine-first already keep it clean candidate-to-
// candidate; this guarantees a clean exit. Best kernel is persisted via best_kernel_code path.)
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(
    `Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${KERNEL_PATH}"\` and confirm the ` +
    `project kernel is now byte-equal to the pristine original backup. This always runs on exit.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

// ---- Report + Layer A evidence envelope ----
phase('Report')
const status = bestSpeedup >= TARGET ? 'converged' : ((stagnantRounds >= STAGNATION_LIMIT || dryRounds >= DRY_LIMIT) ? 'stalled' : 'budget_exhausted')
await agentRetry(() => agent(
  `Write a final optimization report for ${OP}.\n` +
  `Best speedup: ${bestSpeedup.toFixed(3)}x | status: ${status}\n` +
  `Beam (top-${TOPK}): ${JSON.stringify(candidateBeam.map((c) => ({ id: c.id, method: c.planTitle, speedup: c.speedup })))}\n` +
  `All attempts: ${JSON.stringify(allAttempts)}\n` +
  `Cover: which methods the gate allowed, what the persistent memory recommended, ` +
  `which attempts were rejected by anti-cheat and why, and the final best kernel path.` +
  `\n\n# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\n` +
  `Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\n` +
  `Then append:\n` +
  `{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","technique":"${candidateBeam[0].planTitle}","speedup":${bestSpeedup},"note":"final status=${status}; best ${bestSpeedup.toFixed(3)}x via the top beam method"}`,
  { label: 'final-report', phase: 'Report', model: MODEL.judgment }), { retries: 5 })

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  solver: 'generalist-kernel-optimization',
  topology: 'beam',
  overall_speedup: bestSpeedup,
  best_kernel_code: candidateBeam[0].code_path,
  convergence_status: status,
  beam: candidateBeam,
  attempts: allAttempts,
  insights: verifiedInsights,
  memory_db: MEMORY_DB,
}
