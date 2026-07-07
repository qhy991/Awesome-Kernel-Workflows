export const meta = {
  name: 'cupilot-kernel-optimization',
  description: 'Strategy-coordinated evolutionary multi-agent CUDA kernel optimization with roofline-guided prompting (cuPilot methodology)',
  whenToUse: 'When evolving CUDA kernels through multi-generation optimization that requires sophisticated strategies (tensor cores, tiling, pipelining, memory swizzling). Uses strategy as an intermediate semantic representation to decouple evolutionary crossover from low-level code, with roofline model guidance and RAG-based strategy initialization.',
  phases: [
    { title: 'Setup', detail: 'Read kernel spec, generate initial kernel, roofline classification, strategy pool init' },
    { title: 'Strategize', detail: 'SCE Manager generates/crosses strategies at the semantic level via RAG' },
    { title: 'Translate', detail: 'Strategy Translator applies strategies to kernel code' },
    { title: 'Revise', detail: 'Kernel Revisor: compile check → function check → NCU profiling → fix loop' },
    { title: 'Evolve', detail: 'Tournament selection + elitism preservation + strategy alignment + next generation' },
    { title: 'Report', detail: 'Best kernel, strategies applied, hardware utilization metrics' },
  ],
}

// __modelTierApplied (declaration pre-existing)

const WORKFLOW_NAME = 'cupilot-kernel-optimization'


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


// =============================================================================
// cuPilot: Strategy-Coordinated Multi-Agent Framework for CUDA Kernel Evolution
// =============================================================================
//
// Source: "cuPilot: A Strategy-Coordinated Multi-agent Framework for CUDA Kernel Evolution"
//         Chen, Wu, Li, Ma, Si, Hu, Yin, Yang
//         Southeast University / Tsinghua / Tsing Micro / NCTE
//         arXiv:2512.16465, 2025
//         https://github.com/champloo2878/cuPilot-Kernels
//
// Key insight: Conventional crossover prompting operates at the CODE level,
// requiring LLMs to traverse strategy identification → combination → synthesis
// in one shot. This fails as kernel complexity grows. cuPilot introduces
// STRATEGY as an intermediate semantic representation, decoupling:
//   - Strategy-level crossover (SCE Manager): combine optimization ideas
//   - Strategy-to-kernel translation (Strategy Translator): apply one strategy
//
// Three contributions:
//   1. Strategy-Coordinated Evolution (SCE) algorithm
//      - Strategy crossover at semantic level, not code level
//      - Tournament selection with elitism preservation
//      - Strategy alignment after kernel optimization
//   2. Roofline-guided prompting
//      - Classify kernel as compute-bound / memory-bound / middle-zone
//      - Guide strategy generation toward relevant optimizations
//   3. Strategy-level population initialization via RAG
//      - Historical (initial_kernel, optimized_kernel, strategy) triples
//      - RAG retrieval for similar kernels → bootstrap diverse strategies
//
// Multi-agent architecture (Figure 2):
//   - Roofline Prophet: positions kernel on roofline → guides prompting
//   - SCE Manager: manages population, crossover at strategy level
//   - Strategy Translator: applies strategy to kernel (strategy→code)
//   - Kernel Revisor: syntax check → function check → NCU profiling → fix
//   - Kernel Generator: initial vanilla kernel generation
//
// Usage:
//   Workflow({name: 'cupilot-kernel-optimization', args: {
//     kernel_spec: 'PyTorch operator code or description',
//     op_description: 'Standard GEMM (M×N×K, bf16)',
//     target_gpu: 'A100',
//     compile_command: '<user-provided compile command with {kernel_path}/{result_path}>',
//     test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
//     ncu_command: '<user-provided profiler command with {kernel_path}/{result_path}>',
//     roofline_result_path: '/tmp/cupilot_exp/roofline.json',
//     strategy_corpus_path: '/tmp/cupilot_exp/strategy_corpus.jsonl',
//     epochs: 3,
//     generations_per_epoch: 4,
//     population_size: 50,
//     strategy_pool_path: '',
//   }})
//
// =============================================================================

// --- Required Args ---
const PROBLEM_DEFINITION = args.problem_definition || args.kernel_spec || ''
const PROBLEM_PATH = args.problem_path || ''
const OP_DESC = args.op_description || 'CUDA kernel'

// --- Optional Args ---
const GPU_TARGET = args.target_gpu || 'A100'
const COMPILE_CMD = args.compile_command || ''
const TEST_CMD = args.test_command || ''
const NCU_CMD = args.ncu_command || ''
const ROOFLINE_RESULT_PATH = args.roofline_result_path || `${args.exp_dir || '/tmp/cupilot_exp'}/roofline.json`
const STRATEGY_CORPUS_PATH = args.strategy_corpus_path || args.strategy_pool_path || ''
const EPOCHS = args.epochs || 2
const GENERATIONS = args.generations_per_epoch || 4
const POP_SIZE = args.population_size || 30
const STRATEGY_POOL_PATH = args.strategy_pool_path || ''
const EXP_DIR = args.exp_dir || '/tmp/cupilot_exp'
const KERNEL_PATH = args.kernel_path || ''
const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const MAX_REVISE_LOOPS = args.max_revise_loops || 3
const EST_PER_ROUND = args.est_tokens_per_round || 60000
const LANGUAGE = args.language || 'cuda'
const SEED_CANDIDATES = args.seed_candidates || 3

// --- Model routing ---
const MODEL = {
  mechanical: args.model_mechanical || 'haiku',  // runs shell/scripts, parses output, population/strategy bookkeeping
  profile: args.model_profile || 'sonnet',       // profiling / roofline / NCU metric analysis
  judgment: args.model_judgment || 'opus',       // strategy gen, code translation/gen/edit/debug, final report
}
const EVIDENCE_MODE = (COMPILE_CMD && TEST_CMD && NCU_CMD && STRATEGY_CORPUS_PATH)
  ? 'measured'
  : 'conservative_missing_evidence'

// --- shared profiling-strategist plumbing (bespoke raw-ncu family: no backend
// driver, so the strategist resolves against the CUDA substrate manifest and
// the existing ncu prompt below honors its decision). ---
const SUBSTRATE = args.substrate_dir || '_substrate'
const PY = args.substrate_command_prefix || ''
const BACKEND_MANIFEST = args.backend_manifest || `${SUBSTRATE}/backends/cuda/manifest.json`
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }
function substrateInstruction(script, cliArgs) {
  const p = `${SUBSTRATE}/${script}`
  return PY ? `Run exactly: \`${PY} ${p} ${cliArgs}\`.`
            : `No substrate_command_prefix for ${p} ${cliArgs}; do not invent an interpreter.`
}

// --- Project-native integration (embedded inference-engine operators via
// integration-strategist). cuPilot is BESPOKE raw-ncu (no backend driver), so
// there is no standalone driver envelope to gate — only an embedded eval branch
// is ADDED when IS_EMBEDDED. Standalone/legacy benchmark path stays byte-identical. ---
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const BUILD_CMD = args.build_command || COMPILE_CMD
const PROJECT_BENCH_CMD = args.benchmark_command || args.bench_command || args.eval_command || ''
const REGISTER_SCRIPT = args.register_script || ''
if (!KERNEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}
const generatedKernelPath = KERNEL_PATH ? '' : `${EXP_DIR}/generated/cupilot_initial.cu`
const initialCandidates = []
const initialGenerationResult = KERNEL_PATH
  ? null
  : { verified: Boolean(TEST_CMD), selected_candidate_id: 'cupilot-initial', evidence_summary: TEST_CMD ? 'verification requested in setup' : 'unverified initial generation' }

// --- State ---
let population = []      // [{kernel, strategy, fitness, hwUtil}]
let strategyPool = []    // historical strategies for RAG
let bestKernel = { code: '', strategy: '', fitness: 0, speedup: 0 }
let rooflineClass = ''   // 'compute-bound' | 'memory-bound' | 'middle-zone'
let epoch = 0
let generation = 0

// =============================================================================
// Phase 1: Setup — Initial kernel, roofline classification, strategy pool
// =============================================================================
phase('Setup')

const setupResults = await parallel([
  // Agent 1: Generate initial kernel + roofline classification
  () => agentRetry(() => agent(`You are the cuPilot Kernel Generator + Roofline Prophet.

# Task:
1. Read the kernel specification:
${KERNEL_PATH ? `Read from: ${KERNEL_PATH}` : ''}
${PROBLEM_PATH ? `Read problem file: ${PROBLEM_PATH}` : ''}
${PROBLEM_DEFINITION ? `\`\`\`python\n${PROBLEM_DEFINITION.substring(0, 3000)}\n\`\`\`` : `Operation: ${OP_DESC}`}
- language: ${LANGUAGE}
- target_gpu: ${GPU_TARGET}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence contract:
- roofline_result_path: ${ROOFLINE_RESULT_PATH}
- strategy_corpus_path: ${STRATEGY_CORPUS_PATH || '(missing)'}
- evidence_mode: ${EVIDENCE_MODE}
- If evidence_mode is conservative_missing_evidence, roofline guidance, RAG initialization, and SCE decisions are workflow estimates, not strict cuPilot evidence.

2. Generate a functionally correct initial ${LANGUAGE} kernel (vanilla, unoptimized)
   - Include proper __global__ function, thread mapping, memory access
   - Must satisfy the user-provided compile_command/test_command when those contracts are present

3. Perform Roofline Classification (cuPilot Section 4.3):
   Analyze the kernel's arithmetic intensity (FLOPs / bytes transferred):
   - Compute peak FLOPS for ${GPU_TARGET}
   - Compute peak memory bandwidth for ${GPU_TARGET}
   - Calculate the intersection point (arithmetic intensity threshold)

   Classify as:
   - "compute-bound": kernel is to the RIGHT of intersection
     → Prioritize: SM throughput, tensor cores, branch efficiency, instruction scheduling
   - "memory-bound": kernel is to the LEFT of intersection
     → Prioritize: memory bandwidth, L2 hit rate, vectorized loads, memory padding
   - "middle-zone": kernel is BETWEEN the two intersection points
     → Need both compute AND memory optimizations

4. Generate initial optimization guidance based on roofline position.

Return the initial kernel and roofline analysis.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"initial_kernel_generation+roofline_classification","speedup":null,"note":"<roofline_class + arithmetic intensity + guidance summary, one line>"}`, {
    model: MODEL.judgment,
    label: 'setup-kernel-roofline',
    phase: 'Setup',
    schema: {
      type: 'object',
      properties: {
        initial_kernel: { type: 'string' },
        roofline_class: { type: 'string' },
        arithmetic_intensity: { type: 'number' },
        roofline_guidance: { type: 'string' },
        compute_metrics_to_focus: { type: 'array', items: { type: 'string' } },
        memory_metrics_to_focus: { type: 'array', items: { type: 'string' } },
      },
      required: ['initial_kernel', 'roofline_class', 'roofline_guidance'],
    },
  }), { retries: 5 }),

  // Agent 2: Initialize strategy pool (RAG from historical data)
  () => agentRetry(() => agent(`You are the cuPilot Strategy Pool Initializer (Section 4.4).

# Task:
Generate an initial strategy pool for kernel optimization. Each strategy is a concise, reusable optimization technique description.

# Operation: ${OP_DESC}
# GPU Target: ${GPU_TARGET}
${STRATEGY_POOL_PATH ? `# Historical strategy pool: ${STRATEGY_POOL_PATH}` : ''}

# Generate 8-12 diverse strategies spanning these categories:

## Compute Optimizations:
- Invoke Tensor Core (WMMA/MMA instructions for matrix operations)
- Thread/block configuration for maximum occupancy
- Instruction-level parallelism and scheduling

## Memory Optimizations:
- Tiling and shared memory staging
- Vectorized memory access (float4, LDG.128)
- Memory padding to avoid bank conflicts
- Shared memory swizzling for conflict-free access

## Execution Pipeline:
- Double buffering (overlap compute and memory)
- Multi-stage software pipeline
- Asynchronous memory copy (cp.async)

## Low-level:
- Layout/thread block swizzling for L2 locality
- PTX-level instruction optimization
- Register blocking for data reuse

For each strategy, provide:
- name: short identifier
- description: 1-2 sentence explanation of what to do
- applicable_when: roofline condition (compute-bound, memory-bound, or both)
- expected_metrics: which NCU metrics should improve

Return the strategy pool.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"strategy_pool_init","speedup":null,"note":"<number of strategies generated + categories covered, one line>"}`, {
    model: MODEL.judgment,
    label: 'setup-strategy-pool',
    phase: 'Setup',
    schema: {
      type: 'object',
      properties: {
        strategies: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              description: { type: 'string' },
              applicable_when: { type: 'string' },
              expected_metrics: { type: 'array', items: { type: 'string' } },
            },
            required: ['name', 'description'],
          },
        },
      },
      required: ['strategies'],
    },
  }), { retries: 5 }),
])

const kernelSetup = setupResults[0]
const poolSetup = setupResults[1]

rooflineClass = kernelSetup?.roofline_class || 'middle-zone'
strategyPool = (poolSetup?.strategies || []).map(s => s.name + ': ' + s.description)

// Initialize population with initial kernel + no strategy
const initialKernel = kernelSetup?.initial_kernel || ''
population = [{ kernel: initialKernel, strategy: 'baseline (no optimization)', fitness: 0, speedup: 1.0 }]
bestKernel = { code: initialKernel, strategy: 'baseline', fitness: 0, speedup: 1.0 }

log(`Setup: ${OP_DESC} | Roofline: ${rooflineClass} | Strategy pool: ${strategyPool.length} strategies`)
log(`Guidance: ${kernelSetup?.roofline_guidance?.substring(0, 100)}...`)

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it in the ncu/profiling prompt below. The agent only CLASSIFIES the task
// (fuzzy op_class/size); the substrate DETERMINISTICALLY picks the method and
// STAMPS confidence by method (measured/inferred/hypothesized) -- the model must
// NOT assign confidence itself. See _substrate/profiling/README.md. Defaults to
// native_profiler so happy-path ncu behavior is unchanged if the decision is ignored. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }
{
  const _pd = await agentRetry(() => agent(
    `Classify the kernel under optimization. Source: ` +
    (KERNEL_PATH ? `read ${KERNEL_PATH}` : `operation "${OP_DESC}"${PROBLEM_DEFINITION ? ` / spec:\n${PROBLEM_DEFINITION.substring(0, 1500)}` : ''}`) + `.\n` +
    `Pick op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    substrateInstruction('profiling/profiling_strategist.py',
      `resolve --backend-manifest ${BACKEND_MANIFEST} --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl`) +
    ` Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}
log(`Profiling method: ${PROFILING_DECISION.method} (confidence=${PROFILING_DECISION.confidence})`)

// --- integration-strategist: route build/test mode (standalone vs embedded_*).
// For an inference-engine embedded operator (e.g. llama.cpp .cuh referenced via
// KERNEL_PATH), can_compile_standalone=no, so the candidate is built/tested INSIDE
// the host project rather than as an isolated TU. cuPilot has NO backend driver, so
// the standalone path is just the existing inline ncu/benchmark prompt below; only an
// embedded eval branch is added when IS_EMBEDDED. Legacy path stays byte-identical. ---
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
if (KERNEL_PATH) {
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Read ${KERNEL_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    substrateInstruction('integration/integration_strategist.py',
      `resolve --kernel "${KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
      `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl`) +
    ` Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_integ && _integ.method) INTEGRATION_DECISION = _integ
}
log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error('integration-strategist returned derive_adapter — provide project_root + build/test commands')
}
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
// The embedded operator file we swap in place is the project-referenced KERNEL_PATH.
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// A-O1 closure: native_profiler chosen but no project-native profiler is reachable
// under the embedded path (cuPilot has no driver) → downgrade so Revise uses perf_heuristic.
if (PROFILING_DECISION.method === 'native_profiler' && IS_EMBEDDED && !NCU_CMD) {
  log(`profiling: native_profiler but embedded path with no ncu_command -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'project-native-perf', rationale: 'native_profiler unreachable on embedded path -> perf_heuristic' }
}

// =============================================================================
// Evolutionary Loop: Epochs × Generations
// =============================================================================

for (epoch = 0; epoch < EPOCHS; epoch++) {
  for (generation = 0; generation < GENERATIONS; generation++) {
    if (typeof budget !== 'undefined' && budget.total && budget.remaining() < EST_PER_ROUND) { log(`token budget ~exhausted — stop`); break }
    log(`\n=== Epoch ${epoch + 1}/${EPOCHS}, Gen ${generation + 1}/${GENERATIONS} | Pop: ${population.length} | Best: ${bestKernel.speedup.toFixed(2)}x ===`)

    // =========================================================================
    // Phase 2: Strategize — SCE Manager generates/crosses strategies
    // =========================================================================
    phase('Strategize')

    // Select parents via tournament selection
    const sortedPop = [...population].sort((a, b) => b.fitness - a.fitness)
    const parent1 = sortedPop[0]
    const parent2 = sortedPop[Math.min(1, sortedPop.length - 1)]

    const strategizeResult = await agentRetry(() => agent(`You are the cuPilot SCE Manager (Section 4.2).
Perform STRATEGY-LEVEL crossover and generate new optimization strategies.

# Roofline Classification: ${rooflineClass}
# Roofline Guidance:
${kernelSetup?.roofline_guidance || ''}

# Parent 1 (fitness=${parent1.fitness.toFixed(2)}):
Strategy: ${parent1.strategy}

# Parent 2 (fitness=${parent2.fitness.toFixed(2)}):
Strategy: ${parent2.strategy}

# Available Strategy Pool:
${strategyPool.slice(0, 15).map((s, i) => `${i + 1}. ${s}`).join('\n')}

# SCE Crossover Rules (Section 4.2):
1. Work at the STRATEGY level, NOT the code level
2. Combine optimization ideas from both parents into new strategies
3. Generate ${Math.min(POP_SIZE, 5)} new strategy combinations
4. Each strategy should be a coherent set of optimizations that work together
5. For ${rooflineClass} kernels, prioritize:
${rooflineClass === 'compute-bound' ? '   - Tensor core usage, compute throughput, SM utilization' : rooflineClass === 'memory-bound' ? '   - Memory bandwidth, L2 hit rate, vectorized access, memory padding' : '   - Both compute AND memory optimizations together'}
6. Include at least one "exploratory" strategy that tries something novel

Return new strategies for this generation.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is epoch ${epoch}, generation ${generation}):
{"workflow":"${WORKFLOW_NAME}","phase":"Strategize","ts":"<ts>","status":"done","candidate_id":"e${epoch}-g${generation}","technique":"sce_strategy_crossover","speedup":null,"note":"<number of new strategies + how parents were combined, one line>"}`, {
      model: MODEL.judgment,
      label: `strategize-e${epoch}-g${generation}`,
      phase: 'Strategize',
      schema: {
        type: 'object',
        properties: {
          new_strategies: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                strategy: { type: 'string' },
                rationale: { type: 'string' },
                parent_strategies_used: { type: 'array', items: { type: 'string' } },
              },
              required: ['strategy'],
            },
          },
        },
        required: ['new_strategies'],
      },
    }), { retries: 5, allowNull: true })

    const newStrategies = strategizeResult?.new_strategies || []
    log(`  Strategies: ${newStrategies.length} new combinations`)

    // =========================================================================
    // Phase 3: Translate — Apply strategies to kernel code
    // =========================================================================
    phase('Translate')

    const translatedKernels = await parallel(
      newStrategies.slice(0, 5).map((strat, idx) => () =>
        agentRetry(() => agent(`You are the cuPilot Strategy Translator (Section 4.1).
Apply this optimization strategy to produce an optimized CUDA kernel.

# Base Kernel:
\`\`\`cuda
${bestKernel.code.substring(0, 4000)}
\`\`\`

# Strategy to Apply:
${strat.strategy}

# Roofline: ${rooflineClass}
# GPU Target: ${GPU_TARGET}

# Translation Rules:
1. Apply the strategy faithfully — this is strategy-to-kernel translation
2. Output a COMPLETE, COMPILABLE .cu file
3. Maintain functional correctness (same output as baseline)
4. Use appropriate CUDA constructs for the strategy:
   - Tensor cores → wmma/mma intrinsics or cooperative groups
   - Tiling → shared memory with __syncthreads
   - Vectorized → float4/int4 loads
   - Double buffering → ping-pong shared memory buffers
   - Swizzling → XOR-based index remapping
5. Include proper error checking and bounds handling

Return the optimized kernel.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (epoch ${epoch}, generation ${generation}, strategy index ${idx}):
{"workflow":"${WORKFLOW_NAME}","phase":"Translate","ts":"<ts>","status":"done","candidate_id":"e${epoch}-g${generation}-s${idx}","technique":"<the concrete CUDA construct you applied for this strategy>","speedup":null,"note":"<strategy translated + main code change, one line>"}`, {
          model: MODEL.judgment,
          isolation: 'worktree',
          label: `translate-e${epoch}-g${generation}-s${idx}`,
          phase: 'Translate',
          schema: {
            type: 'object',
            properties: {
              kernel_code: { type: 'string' },
              strategies_applied: { type: 'array', items: { type: 'string' } },
            },
            required: ['kernel_code'],
          },
        }), { retries: 5 })
      )
    )

    // =========================================================================
    // Phase 4: Revise — Kernel Revisor: compile → function → profile → fix
    // =========================================================================
    phase('Revise')

    const revisedResults = await parallel(
      translatedKernels.filter(Boolean).map((tk, idx) => () =>
        agentRetry(() => agent(`You are the cuPilot Kernel Revisor (Section 4.1, Figure 2 right side).
Validate and refine this kernel through the revision loop.

# Kernel to Revise:
\`\`\`cuda
${(tk.kernel_code || '').substring(0, 4000)}
\`\`\`

# Strategy Applied: ${newStrategies[idx]?.strategy || 'unknown'}
# GPU: ${GPU_TARGET}

# Revision Loop (up to ${MAX_REVISE_LOOPS} iterations):

## Step 1: Compiler Check
${COMPILE_CMD ? `Run: ${COMPILE_CMD}` : 'Check: valid CUDA syntax, correct use of intrinsics, proper template parameters.'}
If syntax errors → fix them and retry.

## Step 2: Function Check
${TEST_CMD ? `Run: ${TEST_CMD}` : 'Verify numerical correctness against reference output.'}
If function errors → fix logic bugs and retry.

## Step 3: Performance Profiling
${NCU_CMD ? `Run the user-provided ncu_command: ${NCU_CMD}` : 'No ncu_command provided; estimate throughput, occupancy, and memory efficiency without claiming measured profiler evidence.'}
Extract:
- Speedup vs PyTorch baseline
- SM Throughput utilization
- Memory bandwidth utilization
- L2 cache hit rate

## Step 4: Profiling-guided refinement
If performance is suboptimal, use NCU feedback to make ONE targeted improvement.
Then return to Step 1.

If a kernel cannot be fixed after ${MAX_REVISE_LOOPS} attempts, mark it as failed.

Profiling-strategist selected method='${PROFILING_DECISION.method}', confidence='${PROFILING_DECISION.confidence}'. If method !== 'native_profiler', do NOT run ncu/Nsight; instead measure throughput (latency, and GFLOPS/GB-s if available) and stamp every emitted bottleneck with evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. If method === 'native_profiler', proceed with ncu as written.

Return the final revised kernel and its metrics.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if compiled AND correct, else "error"; speedup is the measured speedup number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Revise","ts":"<ts>","status":"<done|error>","candidate_id":"e${epoch}-g${generation}-k${idx}","speedup":<number or null>,"technique":"<strategy under revision>","note":"<compiled? correct? sm/mem util; revision iterations; or the failure reason>"}`, {
          model: MODEL.judgment,
          isolation: 'worktree',
          label: `revise-e${epoch}-g${generation}-k${idx}`,
          phase: 'Revise',
          schema: {
            type: 'object',
            properties: {
              kernel_code: { type: 'string' },
              compiled: { type: 'boolean' },
              correct: { type: 'boolean' },
              speedup: { type: 'number' },
              sm_utilization_pct: { type: 'number' },
              memory_utilization_pct: { type: 'number' },
              revision_iterations: { type: 'number' },
              strategies_confirmed: { type: 'array', items: { type: 'string' } },
            },
            required: ['kernel_code', 'compiled', 'correct', 'speedup'],
          },
        }), { retries: 5 })
      )
    )

    // =========================================================================
    // Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch)
    //
    // The Revise phase above runs candidates with `await parallel(...)` and an
    // INLINE ncu/benchmark step — that is the legacy/standalone path and stays
    // byte-identical when method=standalone. For inference-engine embedded
    // operators the inline ncu/benchmark prompt cannot apply (the .cuh only builds
    // inside the host project), so we re-evaluate each revised candidate IN-PROJECT.
    //
    // SERIAL by construction (this for-loop, NOT parallel): embedded_inplace mutates
    // the shared KERNEL_PATH operator file and embedded_dispatch shares the project
    // build, so candidates cannot be evaluated concurrently (parallel-embedded-race
    // bug-class). The measured embedded latency overrides the inline speedup so the
    // Evolve fitness below uses real in-project numbers.
    // =========================================================================
    if (IS_EMBEDDED) {
      for (let k = 0; k < revisedResults.length; k++) {
        const r = revisedResults[k]
        if (!r) continue
        const suffix = `e${epoch}-g${generation}-k${k}`
        const kPath = `${EXP_DIR}/variants/${suffix}/kernel.cu`
        const variant = `cupilot_${suffix}`.replace(/[^A-Za-z0-9_]/g, '_')
        // Materialize the candidate source so the embedded eval can apply/register it.
        await agentRetry(() => agent(`Write the candidate kernel source to ${kPath} (mkdir -p its dir first).\n\n` +
          `\`\`\`cuda\n${(r.kernel_code || '').substring(0, 6000)}\n\`\`\`\n` +
          `Return {ok:true, path:"${kPath}"}.`,
          { model: MODEL.mechanical, label: `embedded-materialize-${suffix}`, phase: 'Revise', schema: JSON_PASSTHROUGH }), { retries: 5 })
        let embLatency = 0, embMetrics = {}, embBclass = 'unknown', embCompiled = false, embCorrect = false
        if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
          const embResult = await agentRetry(() => agent(
            `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project operator file: ${KERNEL_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
            `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
            `2. Apply candidate: cp ${kPath} ${KERNEL_PATH}\n3. Build: ${BUILD_CMD}\n4. Test: ${TEST_CMD}\n5. Benchmark: ${PROJECT_BENCH_CMD || TEST_CMD}\n` +
            `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
            `Profiling-strategist chose method='${PROFILING_DECISION.method}'. If not native_profiler, do NOT run ncu; derive heuristic_bclass from the throughput ratio.\n` +
            `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
            { model: MODEL.mechanical, label: `embedded-inplace-${suffix}`, phase: 'Revise', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
          embLatency = Number(embResult?.latency_ms || 0)
          embBclass = embResult?.heuristic_bclass || 'unknown'
          embMetrics = embResult?.metrics || { latency_ms: embLatency }
          embCompiled = !!embResult?.compiled
          embCorrect = !!embResult?.correct
        } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
          const _plan = typeof __embeddedEvalPlan === 'function'
            ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: TEST_CMD, benchmarkCmd: PROJECT_BENCH_CMD || TEST_CMD })
            : null
          if (_plan) {
            const embResult = await agentRetry(() => agent(
              `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
              `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
              { model: MODEL.mechanical, label: `embedded-dispatch-${suffix}`, phase: 'Revise', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
            embLatency = Number(embResult?.latency_ms || 0)
            embBclass = embResult?.heuristic_bclass || 'unknown'
            embMetrics = embResult?.metrics || { latency_ms: embLatency }
            embCompiled = !!embResult?.compiled
            embCorrect = !!embResult?.correct
          }
        }
        // Override the inline verdict with the in-project measurement.
        r.driver_envelope = { latency_ms: embLatency, metrics: embMetrics, bottleneck_class: embBclass, backend_id: 'embedded' }
        if (embCompiled || embLatency > 0) { r.compiled = embCompiled; r.correct = embCorrect }
        // Convert measured embedded latency into a speedup vs the baseline latency,
        // if a baseline latency is available; else keep the candidate's reported speedup.
        if (embLatency > 0 && bestKernel.embeddedLatency && bestKernel.embeddedLatency > 0) {
          r.speedup = bestKernel.embeddedLatency / embLatency
        }
        r.embeddedLatency = embLatency
      }
    }

    // =========================================================================
    // Phase 5: Evolve — Tournament selection + elitism + strategy alignment
    // =========================================================================
    phase('Evolve')

    // Compute fitness and add to population
    for (let i = 0; i < revisedResults.length; i++) {
      const r = revisedResults[i]
      if (!r) continue
      const fitness = (!r.compiled || !r.correct) ? 0 : (0.5 + 0.5 * Math.min(1.0, (r.speedup || 0) / 3.0))
      const individual = {
        kernel: r.kernel_code || '',
        strategy: newStrategies[i]?.strategy || '',
        fitness: fitness,
        speedup: r.speedup || 0,
        hwUtil: { sm: r.sm_utilization_pct || 0, mem: r.memory_utilization_pct || 0 },
      }
      population.push(individual)

      // Update best
      if (r.correct && (r.speedup || 0) > bestKernel.speedup) {
        bestKernel = { code: r.kernel_code, strategy: newStrategies[i]?.strategy || '', fitness, speedup: r.speedup }
        // Carry the in-project measured latency forward as the reference for the next
        // generation's embedded speedup conversion (no-op on the standalone path).
        if (r.embeddedLatency && r.embeddedLatency > 0) bestKernel.embeddedLatency = r.embeddedLatency
        log(`  NEW BEST: ${bestKernel.speedup.toFixed(2)}x — strategy: ${bestKernel.strategy.substring(0, 60)}`)
      }
    }

    // Tournament selection + elitism preservation
    population.sort((a, b) => b.fitness - a.fitness)
    // Keep top individuals (elitism)
    const eliteCount = Math.max(2, Math.floor(population.length * 0.2))
    population = population.slice(0, Math.min(POP_SIZE, population.length))

    // Strategy alignment: update strategies to match their optimized kernels
    // (This is the StrategyAlignment step in Algorithm 1, line 7)

    const successfulStrategies = population.filter(p => p.speedup > 1.0).map(p => p.strategy)
    if (successfulStrategies.length > 0) {
      for (const s of successfulStrategies) {
        if (!strategyPool.includes(s) && s.length > 10) {
          strategyPool.push(s)
        }
      }
    }

    log(`  Pop: ${population.length} | Top3: ${population.slice(0, 3).map(p => p.speedup.toFixed(2) + 'x').join(', ')} | Pool: ${strategyPool.length} strategies`)
  }

  // End of epoch: select top kernels as initial kernels for next epoch
  log(`\n--- Epoch ${epoch + 1} complete. Best: ${bestKernel.speedup.toFixed(2)}x ---`)
}

// =============================================================================
// Phase 6: Report
// =============================================================================
phase('Report')

const finalReport = await agentRetry(() => agent(`Write a concise technical report on cuPilot evolutionary optimization.

# cuPilot Results
- Operation: ${OP_DESC}
- GPU: ${GPU_TARGET}
- Roofline classification: ${rooflineClass}
- Best speedup: ${bestKernel.speedup.toFixed(2)}x over PyTorch
- Epochs: ${EPOCHS}, Generations/epoch: ${GENERATIONS}
- Final population: ${population.length}
- Strategy pool: ${strategyPool.length} strategies
- Evidence mode: ${EVIDENCE_MODE}
- Roofline artifact: ${ROOFLINE_RESULT_PATH}
- Strategy corpus: ${STRATEGY_CORPUS_PATH || '(missing)'}

# Best Kernel Strategy:
${bestKernel.strategy}

# Best Kernel Code:
\`\`\`cuda
${bestKernel.code.substring(0, 3000)}
\`\`\`

# Top strategies in final population:
${population.slice(0, 5).map((p, i) => `${i + 1}. ${p.strategy.substring(0, 80)} (${p.speedup.toFixed(2)}x)`).join('\n')}

Write:
1. Evolutionary trajectory: how strategies evolved across generations
2. Roofline guidance effectiveness: did the classification help?
3. Strategy-level crossover: which combinations were most productive?
4. Hardware utilization achieved vs theoretical peak
5. Remaining optimization opportunities

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the final best result:
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","technique":"<winning best strategy summary>","speedup":<best speedup number or null>,"note":"<roofline class + best strategy + final population/pool sizes, one line>"}`, {
  model: MODEL.judgment,
  label: 'final-report',
  phase: 'Report',
}), { retries: 5 })

// embedded_inplace exit safety net: unconditionally restore the pristine operator file.
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${KERNEL_PATH}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  operation: OP_DESC,
  target_gpu: GPU_TARGET,
  roofline_class: rooflineClass,
  best_speedup: bestKernel.speedup,
  best_strategy: bestKernel.strategy,
  best_kernel_code: bestKernel.code,
  epochs: EPOCHS,
  generations: GENERATIONS,
  final_population_size: population.length,
  strategy_pool_size: strategyPool.length,
  top_strategies: population.slice(0, 5).map(p => ({ strategy: p.strategy, speedup: p.speedup })),
  roofline_result_path: ROOFLINE_RESULT_PATH,
  strategy_corpus_path: STRATEGY_CORPUS_PATH,
  evidence_mode: EVIDENCE_MODE,
  report: finalReport,
}
