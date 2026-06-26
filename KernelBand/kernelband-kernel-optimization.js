export const meta = {
  name: 'kernelband-kernel-optimization',
  description: 'Multi-Armed Bandit framework for kernel optimization with hardware-aware pruning, trace-driven clustering, and Lipschitz-continuous reward estimation (KernelBand methodology)',
  whenToUse: 'When optimizing GPU kernels (especially Triton) and you want principled exploration-exploitation balance across multiple optimization strategies. Ideal when the optimization space is large and you want to avoid wasting budget on physically implausible strategies. Uses profiling-based clustering to transfer knowledge between similar kernels and UCB-based action selection to steer exploration.',
  phases: [
    { title: 'Setup', detail: 'Parse kernel, identify strategies, configure hardware target and profiling' },
    { title: 'Profile', detail: 'Extract behavioral feature vector φ(k) via NCU profiling for each candidate' },
    { title: 'Cluster', detail: 'K-Means clustering on behavioral features, update cluster centroids' },
    { title: 'Select', detail: 'Hardware-constrained Masked UCB to select (cluster, strategy) pair' },
    { title: 'Generate', detail: 'LLM applies selected strategy to selected kernel candidate' },
    { title: 'Evaluate', detail: 'Compile, verify correctness, measure latency, compute reward' },
    { title: 'Update', detail: 'Update bandit statistics, cluster assignments, and candidate pool' },
    { title: 'Report', detail: 'Final results: best kernel, strategy statistics, exploration trajectory' },
  ],
}

// __modelTierApplied (declaration pre-existing)

const WORKFLOW_NAME = 'kernelband-kernel-optimization'


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
  return null
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

const WORKFLOW_SUITABILITY = {
  supported_languages: ['triton', 'cuda'],
  supported_problem_types: ['gpu-kernel-optimization', 'kernel-search'],
  problem_types: ['bandit-guided CUDA/Triton kernel search', 'hardware-aware profiling and pruning'],
  reason: 'KernelBand needs CUDA/Triton-style profiling features and evaluator rewards; unsupported languages lack the required feature/mask contract.',
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

function supportsSuitabilityValue(supported, requested) {
  return supported.includes(requested) || supported.some(value => value.endsWith(`-${requested}`))
}

function assertWorkflowSuitability() {
  const requestedLanguage = normalizeSuitabilityValue(args.language)
  if (requestedLanguage && requestedLanguage !== 'auto') {
    const supported = WORKFLOW_SUITABILITY.supported_languages.map(normalizeSuitabilityValue)
    if (!supported.includes(requestedLanguage)) {
      throw new Error(
        `${WORKFLOW_NAME} is not suitable for language="${args.language}". ` +
        `Supported languages/backends: ${WORKFLOW_SUITABILITY.supported_languages.join(', ')}. ` +
        `Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }
  }

  const requestedProblemType = normalizeSuitabilityValue(args.problem_type)
  if (requestedProblemType && requestedProblemType !== 'auto') {
    const supportedProblemTypes = (WORKFLOW_SUITABILITY.supported_problem_types || []).map(normalizeSuitabilityValue)
    if (supportedProblemTypes.length && !supportsSuitabilityValue(supportedProblemTypes, requestedProblemType)) {
      throw new Error(
        `${WORKFLOW_NAME} is not suitable for problem_type="${args.problem_type}". ` +
        `Supported problem types: ${WORKFLOW_SUITABILITY.supported_problem_types.join(', ')}. ` +
        `Typical use cases: ${WORKFLOW_SUITABILITY.problem_types.join('; ')}. ` +
        `Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }
  }
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

if (!USE_DRIVER) {
  assertWorkflowSuitability()
}

// =============================================================================
// KernelBand: Steering LLM-based Kernel Optimization via Hardware-Aware
//             Multi-Armed Bandits
// =============================================================================
//
// Source: "KernelBand: Steering LLM-based Kernel Optimization via
//          Hardware-Aware Multi-Armed Bandits"
//         Dezhi Ran, Shuxiao Xie, Mingfang Ji, Anmin Liu, Mengzhou Wu,
//         Yuan Cao, Yuzhe Guo, Hao Yu, Linyi Li, Yitao Hu, Wei Yang, Tao Xie
//         PKU / Tongming Lake / ECNU / Tianjin / HKUST / SFU / UTD / Fudan
//         arXiv:2511.18868, Feb 2026
//
// KernelBand formulates kernel optimization as a contextual Multi-Armed Bandit
// problem, explicitly separating code generation (LLM strength) from navigation
// of the optimization search space (bandit strength).
//
// Core Algorithm (Algorithm 1):
//   For t = 1..T:
//     1. Compute φ(k) for all k ∈ P (behavioral feature vector)
//     2. If t mod τ = 0 and |P| ≥ 2K: re-cluster via K-Means
//     3. Hardware-constrained selection:
//        - Compute mask M_{i,s} = 𝟙[h(k_c^(i))[Target(s)] < θ_sat]
//        - Select (I_t, S_t) = argmax UCB with mask
//     4. Sample k_t from cluster C_{I_t}
//     5. Generate k'_t = LLM(k_t, S_t)
//     6. If Verify(k'_t): compute reward r_t, update P, update μ̂, N
//
// Key Mechanisms:
//   1. Behavioral Feature Vector φ(k) = [T̄(k), n_reg, n_smem, d_block, η_occ]
//      - Kernels close in φ-space share similar bottlenecks (Lipschitz continuity)
//   2. Dynamic K-Means Clustering (K=3 default, re-cluster every τ=10 iters)
//      - Only profile cluster centroids (representative profiling) to save cost
//   3. Hardware Signature h(k) from NCU: throughput % for DRAM, L2, SM
//      - Pruning: strategy s valid for cluster i only if Target(s) < θ_sat (75%)
//   4. Masked UCB: argmax μ̂_{i,s} + c√(ln t / N_{i,s}) subject to M_{i,s} = 1
//
// Strategies S = {tiling, vectorization, fusion, pipeline, reordering, access_layout}
//
// Results: 1.91× geometric mean speedup on A100 (TritonBench-G, T=20)
//          35-50% higher speedup per dollar vs unguided methods
//
// Usage:
//   Workflow({name: 'kernelband-kernel-optimization', args: {
//     kernel_path: '/path/to/kernel.py',
//     op_description: 'Fused attention forward pass',
//     harness_path: '/path/to/benchmark.py',
//     compile_command: '<user-provided compile/import command with {kernel_path}/{result_path}>',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     ncu_command: '<user-provided profiler command with {kernel_path}/{result_path}>',
//     feature_vector_result_path: '/tmp/kernelband_exp/features.json',
//     hardware_signature_result_path: '/tmp/kernelband_exp/hardware_signature.json',
//     target_gpu: 'A100',
//     iterations: 20,
//     num_clusters: 3,
//     recluster_period: 10,
//     strategies: ['tiling', 'vectorization', 'fusion', 'pipeline', 'reordering', 'access_layout'],
//   }})
//
// =============================================================================

// --- Required Args ---
let KERNEL_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const OP_DESCRIPTION = args.op_description || 'GPU kernel'

// --- Model Routing ---
const MODEL = {
  mechanical: args.model_mechanical || 'haiku',  // runs shell/scripts, parses output, bandit/cluster bookkeeping
  profile: args.model_profile || 'sonnet',       // profiling / feature-vector / metric analysis
  judgment: args.model_judgment || 'opus',       // planning, code gen/edit/debug, final report
}

// --- Token Budget ---
const EST_PER_ROUND = args.est_tokens_per_round || 60000

// --- Optional Args ---
const HARNESS_PATH = args.harness_path || ''
const COMPILE_CMD = args.compile_command || ''
const BENCHMARK_CMD = args.benchmark_command || ''
const NCU_CMD = args.ncu_command || ''
const NCU_BINARY = args.ncu_binary || ''
const FEATURE_VECTOR_RESULT_PATH = args.feature_vector_result_path || `${args.exp_dir || '/tmp/kernelband_exp'}/features/latest.json`
const HARDWARE_SIGNATURE_RESULT_PATH = args.hardware_signature_result_path || `${args.exp_dir || '/tmp/kernelband_exp'}/profiles/hardware_signature.json`
const GPU_TARGET = args.target_gpu || 'A100'
const ITERATIONS = args.iterations || 20
const NUM_CLUSTERS = args.num_clusters || 3
const RECLUSTER_PERIOD = args.recluster_period || 10
const UCB_C = args.ucb_exploration || 2.0
LEGACY_SATURATION_THRESHOLD = 0.75
let SATURATION_THRESHOLD = args.saturation_threshold || LEGACY_SATURATION_THRESHOLD
const EXP_DIR = args.exp_dir || '/tmp/kernelband_exp'
const EVIDENCE_MODE = (COMPILE_CMD && BENCHMARK_CMD && (NCU_CMD || NCU_BINARY))
  ? 'measured'
  : 'conservative_missing_evidence'

if (!KERNEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

const LANGUAGE = args.language || 'triton'
const SEED_CANDIDATES = args.seed_candidates || 3
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null

// --- Backend driver wiring (P5d Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_LANG_TOKEN = LANGUAGE
const LEGACY_FENCE_TOKEN = LANGUAGE
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

// --- Project-native integration (embedded kernels via integration-strategist) ---
// For inference-engine embedded operators (e.g. llama.cpp .cuh) the candidate is
// built/tested INSIDE the host project instead of as an isolated TU. BENCHMARK_CMD
// is the existing standalone bench const; PROJECT_BENCH_CMD is the in-project bench.
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
let DRIVER_SOURCE_EXT = ''
let DRIVER_BACKEND_ID = RESOLVED_BACKEND || ''

function langToken(legacy) {
  return USE_DRIVER ? DRIVER_LANG_FENCE : legacy
}
function fenceToken() {
  return USE_DRIVER ? DRIVER_LANG_FENCE : LEGACY_FENCE_TOKEN
}

const STRATEGIES = args.strategies || [
  'tiling',
  'vectorization',
  'fusion',
  'pipeline',
  'reordering',
  'access_layout',
]

// --- Bandit State ---
let candidatePool = []
let clusters = []
let banditStats = {}
let bestKernel = { code: '', latency: Infinity, speedup: 0 }
let baselineLatency = 0
let totalReward = 0
let iterationLog = []

async function resolveInitialKernelFromProblem() {
  if (INPUT_MODE !== 'generate_then_optimize') return ''

  const generated = await agentRetry(() => agent(`No kernel_path was provided. Generate and verify an initial kernel before starting KernelBand.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- op_description: ${OP_DESCRIPTION}
- language: ${langToken(LANGUAGE)}
- target_gpu: ${GPU_TARGET}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- compile_command: ${COMPILE_CMD || '(not provided)'}
- benchmark_command: ${BENCHMARK_CMD || '(not provided)'}
- ncu_command: ${NCU_CMD || '(not provided)'}

# Contract
1. If problem_path is provided, read it first.
2. Generate ${SEED_CANDIDATES} complete kernel candidates.
3. Materialize candidates under ${EXP_DIR}/generated/.
4. Run available commands with {kernel_path} and {result_path} substitutions.
5. Select the best candidate that compiles and passes correctness or benchmark validation. If no real evaluator is available, select the strongest candidate and mark verified=false.
6. Return generated_kernel_path plus candidate and evidence metadata.`, {
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
  if (BENCHMARK_CMD && initialGenerationResult.verified === false) {
    throw new Error('No generated seed passed benchmark/correctness evidence')
  }
  return generatedKernelPath
}

// Initialize bandit statistics: N_{i,s} = 1, μ̂_{i,s} = 0.5 for all (i, s)
for (let i = 0; i < NUM_CLUSTERS; i++) {
  for (const s of STRATEGIES) {
    const key = `${i}_${s}`
    banditStats[key] = { count: 1, mean_reward: 0.5, mask: 1 }
  }
}

// =============================================================================
// Phase 1: Setup — Parse kernel, identify hardware, establish baseline
// =============================================================================
phase('Setup')

if (USE_DRIVER) {
  DRIVER = await agentRetry(() => agent(
    `Load the backend driver at ${BACKEND_DIR} and return its manifest plus idioms verbatim.\n` +
    `1. Run exactly: \`cat ${driverPath('manifest.json')}\` and parse JSON.\n` +
    `2. Run exactly: \`cat ${driverPath('idioms.json')}\` and parse JSON.\n` +
    `Return {present, backend_id, source_ext, aux_ext, lang_fence, impl_requirements, methods, saturation_threshold}.`,
    { model: MODEL.mechanical, label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
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
  // φ-gate driver resolution: driver may supply its own saturation threshold
  SATURATION_THRESHOLD = DRIVER.saturation_threshold != null ? DRIVER.saturation_threshold : LEGACY_SATURATION_THRESHOLD
  log(`Driver loaded: ${DRIVER_BACKEND_ID} (fence=${DRIVER_LANG_FENCE}, θ_sat=${SATURATION_THRESHOLD})`)
}

if (INPUT_MODE === 'generate_then_optimize') {
  KERNEL_PATH = await resolveInitialKernelFromProblem()
}

const setupResult = await agentRetry(() => agent(`You are setting up a KernelBand optimization session.

# Task
1. Read the kernel file: ${KERNEL_PATH}
2. Create experiment directory: mkdir -p ${EXP_DIR}/{candidates,profiles,logs}
3. Record profiling evidence artifacts:
   - feature_vector_result_path: ${FEATURE_VECTOR_RESULT_PATH}
   - hardware_signature_result_path: ${HARDWARE_SIGNATURE_RESULT_PATH}
   - evidence_mode: ${EVIDENCE_MODE}
   If evidence_mode is conservative_missing_evidence, do not claim strict KernelBand execution; mark phi, masks, and rewards as estimates.
4. Establish baseline performance:
   ${COMPILE_CMD ? `Compile: ${COMPILE_CMD}` : '(no compile_command provided; perform static compileability review only)'}
   ${BENCHMARK_CMD ? `Benchmark: ${BENCHMARK_CMD}` : '(no benchmark_command provided; do not invent one)'}
5. Run initial profiling to get hardware signature:
   ${NCU_CMD ? `Profile: ${NCU_CMD}` : NCU_BINARY ? `Use the user-provided ncu_binary (${NCU_BINARY}) only with the user-provided benchmark/harness contract.` : '(no ncu_command/ncu_binary provided; mark hardware signature as missing evidence)'}
   Extract: DRAM throughput %, L2 throughput %, SM throughput %
5. Extract behavioral features φ(k₀):
   - Normalized execution time T̄(k₀) (= 1.0 for baseline)
   - Registers per thread (n_reg)
   - Shared memory per block (n_smem)
   - Block dimension (d_block)
   - Occupancy (η_occ)

# Target Hardware: ${GPU_TARGET}
# Operation: ${OP_DESCRIPTION}
# Strategies available: ${STRATEGIES.join(', ')}

Return the baseline metrics and hardware signature.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"baseline_profiling","speedup":null,"note":"<baseline latency us + dominant hardware bottleneck (DRAM/L2/SM) one line>"}`, {
  label: 'setup',
  phase: 'Setup',
  model: MODEL.profile,
  schema: {
    type: 'object',
    properties: {
      kernel_code: { type: 'string' },
      baseline_latency_us: { type: 'number' },
      hardware_signature: {
        type: 'object',
        properties: {
          dram_throughput_pct: { type: 'number' },
          l2_throughput_pct: { type: 'number' },
          sm_throughput_pct: { type: 'number' },
          dominant_bottleneck: { type: 'string' },
        },
      },
      behavioral_features: {
        type: 'object',
        properties: {
          normalized_time: { type: 'number' },
          registers_per_thread: { type: 'number' },
          shared_mem_bytes: { type: 'number' },
          block_dimension: { type: 'number' },
          occupancy: { type: 'number' },
        },
      },
      platform_info: { type: 'string' },
    },
    required: ['baseline_latency_us'],
  },
}), { retries: 5 })

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it below. The agent only classifies the task (fuzzy op_class/size); the
// substrate strategist DETERMINISTICALLY picks the method and STAMPS confidence by
// method (measured/inferred/hypothesized) -- the model must NOT assign confidence.
// Falls back to native_profiler so the happy path is unchanged if the call returns
// nothing. Downstream diagnose.py bottleneck classification stays unchanged. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }
if (USE_DRIVER) {
  const _pd = await agentRetry(() => agent(
    `Read ${KERNEL_PATH}; classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    `run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${driverPath('manifest.json')} --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`.\n` +
    `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}

// --- integration-strategist: route build/test mode (standalone vs embedded_*).
// For an inference-engine embedded operator (e.g. llama.cpp .cuh referenced via
// KERNEL_PATH), can_compile_standalone=no, so the candidate is built/tested INSIDE
// the host project rather than as an isolated TU. Standalone path stays byte-identical. ---
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _kernelFile = KERNEL_PATH || `${EXP_DIR}/baseline.kernel`
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Read ${_kernelFile}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${_kernelFile}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
    `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  if (_integ && _integ.method) INTEGRATION_DECISION = _integ
}
log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error('integration-strategist returned derive_adapter — provide project_root + build/test commands')
}
const USE_DRIVER_STANDALONE = USE_DRIVER && INTEGRATION_DECISION.method === 'standalone'
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
// The embedded operator file we swap in place is the project-referenced KERNEL_PATH.
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// A-O1 closure: native_profiler chosen but no project-native profiler is reachable
// under the embedded path → downgrade to perf_heuristic (run.sh/bench gives throughput).
if (PROFILING_DECISION.method === 'native_profiler' && IS_EMBEDDED && !USE_DRIVER_STANDALONE) {
  log(`profiling: native_profiler but embedded path -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'project-native-perf', rationale: 'native_profiler unreachable on embedded path -> perf_heuristic' }
}

if (USE_DRIVER_STANDALONE) {
  const kPath = KERNEL_PATH || `${EXP_DIR}/baseline.kernel`
  const buildOut = `${EXP_DIR}/baseline.artifact`
  const profOut = `${EXP_DIR}/baseline.prof.native`
  await agentRetry(() => agent(
    `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
    `Return its stdout JSON verbatim.`,
    { model: MODEL.mechanical, label: 'driver-build-setup', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  const runOut = await agentRetry(() => agent(
    `${driverSh('run.sh', `--artifact ${buildOut} --kernel ${kPath}`)}\n` +
    `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
    { model: MODEL.profile, label: 'driver-run-setup', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  let evidenceOut
  if (PROFILING_DECISION.method === 'native_profiler') {
    await agentRetry(() => agent(
      `${driverSh('profile.sh', `--artifact ${buildOut} --kernel ${kPath} --out ${profOut}`)}\n` +
      `Return {ok, native_path}.`,
      { model: MODEL.profile, label: 'driver-profile-setup', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
    evidenceOut = await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
      `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
      { model: MODEL.mechanical, label: 'driver-to-evidence-setup', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  } else {
    evidenceOut = await agentRetry(() => agent(
      `Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run a native profiler (profile.sh). ` +
      `Use the run.sh latency/throughput from above. ` +
      (PROFILING_DECISION.method === 'perf_heuristic'
        ? `Normalize that throughput into canonical metrics via \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${PROFILING_DECISION.normalizer || 'perf_to_evidence.py'} --artifact ${buildOut} --kernel ${kPath} --out ${profOut}\`. Tag every emitted bottleneck with evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. `
        : ``) +
      `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend, profiler_available:false}.`,
      { model: MODEL.profile, label: 'driver-perf-evidence-setup', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  }
  await agentRetry(() => agent(
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
    `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
    { model: MODEL.mechanical, label: 'driver-diagnose-setup', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  await agentRetry(() => agent(
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${EXP_DIR}/baseline.result.json\`.\n` +
    `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
    { model: MODEL.mechanical, label: 'driver-anti-cheat-setup', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

baselineLatency = setupResult?.baseline_latency_us || 1000
const initialCode = setupResult?.kernel_code || ''
const hwSignature = setupResult?.hardware_signature || { dram_throughput_pct: 50, l2_throughput_pct: 50, sm_throughput_pct: 50 }
const initialFeatures = setupResult?.behavioral_features || { normalized_time: 1.0, registers_per_thread: 32, shared_mem_bytes: 0, block_dimension: 256, occupancy: 0.5 }

candidatePool.push({
  id: 0,
  code: initialCode,
  latency: baselineLatency,
  speedup: 1.0,
  features: initialFeatures,
  hw_signature: hwSignature,
  cluster: 0,
  source_strategy: 'initial',
})

clusters = [{ id: 0, centroid_features: initialFeatures, centroid_hw: hwSignature, members: [0] }]

log(`Setup: baseline ${baselineLatency}μs on ${GPU_TARGET} | ${STRATEGIES.length} strategies | K=${NUM_CLUSTERS} clusters | T=${ITERATIONS} iters`)
log(`Hardware: DRAM=${hwSignature.dram_throughput_pct}% L2=${hwSignature.l2_throughput_pct}% SM=${hwSignature.sm_throughput_pct}% → ${hwSignature.dominant_bottleneck || 'unknown'}-bound`)

// =============================================================================
// Main Bandit Loop: T iterations
// =============================================================================

for (let t = 1; t <= ITERATIONS; t++) {
  if (typeof budget !== 'undefined' && budget.total && budget.remaining() < EST_PER_ROUND) { log(`token budget ~exhausted — stop`); break }
  log(`\n--- Iteration ${t}/${ITERATIONS} ---`)

  // ===========================================================================
  // Periodic Re-clustering (every τ iterations when pool is large enough)
  // ===========================================================================
  const shouldRecluster = (t % RECLUSTER_PERIOD === 0) && (candidatePool.length >= 2 * NUM_CLUSTERS)

  if (shouldRecluster) {
    phase('Cluster')

    const clusterResult = await agentRetry(() => agent(`You are the KernelBand Dynamic Clustering module (Section 3.3).

# Task: Re-cluster the candidate kernel pool using K-Means on behavioral features.

# Candidate Pool (${candidatePool.length} kernels):
${candidatePool.map((c, idx) => `Kernel ${c.id}: φ = [T̄=${c.features.normalized_time.toFixed(3)}, reg=${c.features.registers_per_thread}, smem=${c.features.shared_mem_bytes}, block=${c.features.block_dimension}, occ=${c.features.occupancy.toFixed(3)}] speedup=${c.speedup.toFixed(2)}x`).join('\n')}

# Parameters:
- K = ${NUM_CLUSTERS} clusters
- Distance metric: Euclidean on normalized φ vectors

# Requirements:
1. Normalize each dimension to [0,1] before clustering
2. Assign each kernel to its nearest cluster centroid
3. Compute new centroids as the mean of cluster members
4. For each cluster centroid, identify the representative kernel (nearest to centroid)
   - This representative will be profiled for hardware signature updates

Return cluster assignments and centroids.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is bandit iteration ${t}):
{"workflow":"${WORKFLOW_NAME}","phase":"Cluster","ts":"<ts>","status":"done","candidate_id":"iter-${t}","technique":"kmeans_recluster","speedup":null,"note":"<resulting cluster sizes + what moved, one line>"}`, {
      label: `cluster-t${t}`,
      phase: 'Cluster',
      model: MODEL.mechanical,
      schema: {
        type: 'object',
        properties: {
          clusters: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                id: { type: 'number' },
                centroid_features: {
                  type: 'object',
                  properties: {
                    normalized_time: { type: 'number' },
                    registers_per_thread: { type: 'number' },
                    shared_mem_bytes: { type: 'number' },
                    block_dimension: { type: 'number' },
                    occupancy: { type: 'number' },
                  },
                },
                member_ids: { type: 'array', items: { type: 'number' } },
                representative_id: { type: 'number' },
              },
            },
          },
        },
        required: ['clusters'],
      },
    }), { retries: 5 })

    if (clusterResult?.clusters) {
      clusters = clusterResult.clusters.map(c => ({
        id: c.id,
        centroid_features: c.centroid_features,
        centroid_hw: hwSignature,
        members: c.member_ids || [],
      }))
      for (const c of clusterResult.clusters) {
        for (const mid of (c.member_ids || [])) {
          const cand = candidatePool.find(p => p.id === mid)
          if (cand) cand.cluster = c.id
        }
      }
      log(`Re-clustered: ${clusters.map(c => `C${c.id}(${c.members.length})`).join(', ')}`)
    }

    // Profile cluster centroids for hardware signature updates
    phase('Profile')

    const profileResult = await agentRetry(() => agent(`You are the KernelBand Representative Profiling module (Section 3.3).

# Task: Profile the representative kernel from each active cluster to update hardware signatures.

# Clusters:
${clusters.map(c => {
  const rep = candidatePool.find(p => p.id === (c.members[0] || 0))
  return `Cluster ${c.id}: representative kernel ${rep ? rep.id : 'N/A'}`
}).join('\n')}

# Profiling Contract:
${NCU_CMD ? `Run: ${NCU_CMD}` : NCU_BINARY ? `Use the user-provided ncu_binary (${NCU_BINARY}) with the user-provided benchmark/harness command; do not invent a profiler command.` : '(no profiler command provided; set profile evidence fields to missing)'}

# Extract for each cluster centroid:
- DRAM throughput % (memory bandwidth utilization)
- L2 cache throughput %
- SM throughput % (compute utilization)
These determine the hardware mask M_{i,s} for strategy pruning.

A strategy s is VALID for cluster i only if: h(k_c^(i))[Target(s)] < ${SATURATION_THRESHOLD * 100}%
- tiling targets: SM (compute)
- vectorization targets: DRAM (memory bandwidth)
- fusion targets: DRAM (reduce memory traffic)
- pipeline targets: SM (increase compute overlap)
- reordering targets: L2 (cache locality)
- access_layout targets: DRAM (coalescing)

Return updated hardware signatures and masks.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is bandit iteration ${t}):
{"workflow":"${WORKFLOW_NAME}","phase":"Profile","ts":"<ts>","status":"done","candidate_id":"iter-${t}","technique":"representative_profiling","speedup":null,"note":"<per-cluster DRAM/L2/SM throughput + how many (cluster,strategy) pairs pruned, one line>"}`, { model: MODEL.profile,
      label: `profile-t${t}`,
      phase: 'Profile',
      model: MODEL.profile,
      schema: {
        type: 'object',
        properties: {
          cluster_profiles: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                cluster_id: { type: 'number' },
                dram_throughput_pct: { type: 'number' },
                l2_throughput_pct: { type: 'number' },
                sm_throughput_pct: { type: 'number' },
                valid_strategies: { type: 'array', items: { type: 'string' } },
                pruned_strategies: { type: 'array', items: { type: 'string' } },
              },
            },
          },
        },
        required: ['cluster_profiles'],
      },
    }), { retries: 5 })

    if (profileResult?.cluster_profiles) {
      for (const cp of profileResult.cluster_profiles) {
        const cluster = clusters.find(c => c.id === cp.cluster_id)
        if (cluster) {
          cluster.centroid_hw = {
            dram_throughput_pct: cp.dram_throughput_pct,
            l2_throughput_pct: cp.l2_throughput_pct,
            sm_throughput_pct: cp.sm_throughput_pct,
          }
        }
        for (const s of STRATEGIES) {
          const key = `${cp.cluster_id}_${s}`
          if (banditStats[key]) {
            banditStats[key].mask = (cp.valid_strategies || STRATEGIES).includes(s) ? 1 : 0
          }
        }
      }
      const prunedCount = Object.values(banditStats).filter(v => v.mask === 0).length
      log(`Profiled centroids: ${prunedCount}/${Object.keys(banditStats).length} (cluster,strategy) pairs pruned`)
    }
  }

  // ===========================================================================
  // Action Selection: Masked UCB (Section 3.4)
  // ===========================================================================
  phase('Select')

  let bestUCB = -Infinity
  let selectedCluster = 0
  let selectedStrategy = STRATEGIES[0]

  for (let i = 0; i < Math.min(NUM_CLUSTERS, clusters.length); i++) {
    for (const s of STRATEGIES) {
      const key = `${i}_${s}`
      const stat = banditStats[key]
      if (!stat || stat.mask === 0) continue
      const ucb = stat.mean_reward + UCB_C * Math.sqrt(Math.log(t) / stat.count)
      if (ucb > bestUCB) {
        bestUCB = ucb
        selectedCluster = i
        selectedStrategy = s
      }
    }
  }

  // Sample kernel from selected cluster (prefer candidates with higher local potential)
  const clusterMembers = candidatePool.filter(c => c.cluster === selectedCluster)
  const selectedKernel = clusterMembers.length > 0
    ? clusterMembers.reduce((best, c) => c.speedup > best.speedup ? c : best, clusterMembers[0])
    : candidatePool[0]

  log(`Select: cluster=${selectedCluster}, strategy=${selectedStrategy}, kernel=${selectedKernel.id} (UCB=${bestUCB.toFixed(3)})`)

  // ===========================================================================
  // Code Generation: LLM applies strategy to kernel (Section 3.1)
  // ===========================================================================
  phase('Generate')

  const generateResult = await agentRetry(() => agent(`You are the KernelBand Code Generator. Apply a specific optimization strategy to the given kernel.

# Selected Action: (Cluster ${selectedCluster}, Strategy: ${selectedStrategy})
# Target Hardware: ${GPU_TARGET}
# Operation: ${OP_DESCRIPTION}

# Source Kernel (ID ${selectedKernel.id}, current speedup: ${selectedKernel.speedup.toFixed(2)}x):
\`\`\`${fenceToken()}
${(selectedKernel.code || '').substring(0, 6000)}
\`\`\`

# Strategy: ${selectedStrategy}
Apply the "${selectedStrategy}" optimization strategy. Specific guidance:

${selectedStrategy === 'tiling' ? `TILING: Restructure computation into tiles that fit in shared memory/registers.
- Choose tile sizes that match hardware cache/register constraints
- Ensure tiles cover the full computation without overlap or gaps
- Consider multi-level tiling (thread block → warp → thread)` : ''}
${selectedStrategy === 'vectorization' ? `VECTORIZATION: Increase memory throughput via wider loads/stores.
- Use vector types (float4, int4) for coalesced memory access
- Ensure alignment constraints are met
- Batch independent computations into vector operations` : ''}
${selectedStrategy === 'fusion' ? `FUSION: Reduce memory traffic by fusing multiple operations.
- Identify producer-consumer pairs that can share registers/shared memory
- Eliminate intermediate global memory writes
- Fuse elementwise operations with preceding/following kernels` : ''}
${selectedStrategy === 'pipeline' ? `PIPELINE: Overlap computation with memory operations.
- Double-buffer shared memory loads with computation
- Use async copy (cp.async) where available
- Stage computation to hide memory latency` : ''}
${selectedStrategy === 'reordering' ? `REORDERING: Improve cache locality via access pattern changes.
- Reorder loops for better spatial/temporal locality
- Swizzle memory layouts to avoid bank conflicts
- Rearrange thread-to-data mapping for coalescing` : ''}
${selectedStrategy === 'access_layout' ? `ACCESS & LAYOUT: Optimize memory layout and access patterns.
- Ensure coalesced global memory accesses (consecutive threads → consecutive addresses)
- Pad shared memory to avoid bank conflicts
- Choose row-major vs column-major based on access pattern` : ''}

# Constraints:
- Preserve functional correctness (same outputs for same inputs)
- Target the non-saturated resource (this strategy was selected because the target resource is below ${SATURATION_THRESHOLD * 100}% utilization)
- Generate a complete, compilable kernel

# Bandit Context:
- Iteration: ${t}/${ITERATIONS}
- This (cluster, strategy) pair has been tried ${banditStats[`${selectedCluster}_${selectedStrategy}`]?.count || 0} times
- Average reward so far: ${(banditStats[`${selectedCluster}_${selectedStrategy}`]?.mean_reward || 0).toFixed(3)}

Return the optimized kernel code.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is bandit iteration ${t}; the arm pulled is strategy ${selectedStrategy} on cluster ${selectedCluster}, source kernel ${selectedKernel.id}):
{"workflow":"${WORKFLOW_NAME}","phase":"Generate","ts":"<ts>","status":"done","candidate_id":"iter-${t}","technique":"${selectedStrategy}","speedup":null,"note":"<concrete change applied to kernel ${selectedKernel.id} under the ${selectedStrategy} strategy, one line>"}`, {
    label: `generate-t${t}-${selectedStrategy}`,
    phase: 'Generate',
    model: MODEL.judgment,
    schema: {
      type: 'object',
      properties: {
        optimized_kernel: { type: 'string' },
        changes_description: { type: 'string' },
        expected_improvement: { type: 'string' },
      },
      required: ['optimized_kernel'],
    },
  }), { retries: 5 })

  const generatedCode = generateResult?.optimized_kernel || ''

  // ===========================================================================
  // Evaluation: Compile, verify, benchmark (Section 3.1, Algorithm 1 line 19)
  // ===========================================================================
  phase('Evaluate')

  const evalResult = await agentRetry(() => agent(`You are the KernelBand Evaluation module. Verify correctness and measure performance.

# Generated Kernel:
\`\`\`${fenceToken()}
${generatedCode.substring(0, 6000)}
\`\`\`

# Evaluation Steps (Two-stage verification from Section 4.1):
1. **Call Accuracy**: Compile and run — check for runtime errors
   ${COMPILE_CMD || '(compile the generated kernel)'}
2. **Execution Accuracy**: Verify numerical equivalence via torch.allclose
   Compare outputs against reference (CPU ATen) across 10+ input shapes
3. **Performance Measurement**: Benchmark across dominant input shapes
   ${BENCHMARK_CMD || '(run the performance benchmark)'}
   Report latency in microseconds
4. **Feature Extraction**: Extract behavioral features φ(k'):
   - Normalized time: T̄(k') = latency(k') / ${baselineLatency} (baseline latency)
   - Registers per thread, shared memory, block dim, occupancy
   ${NCU_CMD ? `Profile: ${NCU_CMD}` : ''}

# Baseline latency: ${baselineLatency} μs
# Previous best: ${bestKernel.latency === Infinity ? 'N/A' : bestKernel.latency.toFixed(1) + ' μs (' + bestKernel.speedup.toFixed(2) + 'x)'}

Return evaluation results.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if it compiled AND passed correctness, else "error"; speedup is the measured speedup number vs baseline, or null if unavailable; this is bandit iteration ${t}, strategy ${selectedStrategy}):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"iter-${t}","technique":"${selectedStrategy}","speedup":<number or null>,"note":"<compiled? correct? measured latency us; or the failure reason>"}`, {
    label: `eval-t${t}`,
    phase: 'Evaluate',
    model: MODEL.mechanical,
    schema: {
      type: 'object',
      properties: {
        compiled: { type: 'boolean' },
        correct: { type: 'boolean' },
        latency_us: { type: 'number' },
        speedup: { type: 'number' },
        behavioral_features: {
          type: 'object',
          properties: {
            normalized_time: { type: 'number' },
            registers_per_thread: { type: 'number' },
            shared_mem_bytes: { type: 'number' },
            block_dimension: { type: 'number' },
            occupancy: { type: 'number' },
          },
        },
        hw_signature: {
          type: 'object',
          properties: {
            dram_throughput_pct: { type: 'number' },
            l2_throughput_pct: { type: 'number' },
            sm_throughput_pct: { type: 'number' },
          },
        },
        error_details: { type: 'string' },
      },
      required: ['compiled', 'correct'],
    },
  }), { retries: 5 })

  if (USE_DRIVER_STANDALONE) {
    const suffix = `t${t}`
    const kPath = `${EXP_DIR}/iter_${t}.kernel`
    const buildOut = `${EXP_DIR}/iter_${t}.artifact`
    const profOut = `${EXP_DIR}/iter_${t}.prof.native`
    await agentRetry(() => agent(
      `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
      `Return its stdout JSON verbatim.`,
      { model: MODEL.mechanical, label: `driver-build-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    const runOut = await agentRetry(() => agent(
      `${driverSh('run.sh', `--artifact ${buildOut} --kernel ${kPath}`)}\n` +
      `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
      { model: MODEL.profile, label: `driver-run-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    let evidenceOut
    if (PROFILING_DECISION.method === 'native_profiler') {
      await agentRetry(() => agent(
        `${driverSh('profile.sh', `--artifact ${buildOut} --kernel ${kPath} --out ${profOut}`)}\n` +
        `Return {ok, native_path}.`,
        { model: MODEL.profile, label: `driver-profile-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      evidenceOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
        `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
        { model: MODEL.mechanical, label: `driver-to-evidence-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    } else {
      // Profiling-strategist chose method='${PROFILING_DECISION.method}'; do NOT run the native profiler.
      // run.sh already produced throughput above; normalize it when method='perf_heuristic'.
      const _norm = PROFILING_DECISION.normalizer || 'perf_to_evidence.py'
      evidenceOut = await agentRetry(() => agent(
        `Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run profile.sh. ` +
        `run.sh returned latency_ms=${(runOut && runOut.latency_ms) || 'null'}. ` +
        (PROFILING_DECISION.method === 'perf_heuristic'
          ? `Normalize that throughput into canonical metrics via \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${_norm} --artifact ${buildOut} --kernel ${kPath} --out ${profOut}\`. ` +
            `Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
            `Also write heuristic_bclass (memory_bound|compute_bound|latency_bound) from the throughput ratio so diagnose.py does not fall to unknown. `
          : ``) +
        `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend, profiler_available:false}.`,
        { model: MODEL.mechanical, label: `driver-to-evidence-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    }
    await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
      `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
      { model: MODEL.mechanical, label: `driver-diagnose-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${EXP_DIR}/iter_${t}.result.json\`.\n` +
      `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
      { model: MODEL.mechanical, label: `driver-anti-cheat-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
  } else if (IS_EMBEDDED) {
    // --- Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch) ---
    // SERIAL by construction: this runs inside the main `for (let t...)` bandit loop, which
    // is itself serial (KernelBand evaluates one candidate per iteration — no `await parallel(`).
    // embedded_inplace mutates the shared KERNEL_PATH operator file and embedded_dispatch
    // shares the project build, so candidates cannot be evaluated concurrently
    // (parallel-embedded-race bug-class).
    const suffix = `t${t}`
    const kPath = `${EXP_DIR}/iter_${t}.kernel`
    const variant = `kband_${suffix}`.replace(/[^A-Za-z0-9_]/g, '_')
    // Materialize the candidate source so the embedded eval can apply/register it.
    await agentRetry(() => agent(`Write the candidate kernel source to ${kPath} (mkdir -p its dir first).\n\n` +
      `\`\`\`${fenceToken()}\n${(generatedCode || '').substring(0, 6000)}\n\`\`\`\n` +
      `Return {ok:true, path:"${kPath}"}.`,
      { model: MODEL.mechanical, label: `embedded-materialize-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    let embLatency = 0, embMetrics = {}, embBclass = 'unknown'
    if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
      const embResult = await agentRetry(() => agent(
        `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project operator file: ${KERNEL_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
        `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
        `2. Apply candidate: cp ${kPath} ${KERNEL_PATH}\n3. Build: ${BUILD_CMD}\n4. Test: ${COMPILE_CMD || BENCHMARK_CMD}\n5. Benchmark: ${PROJECT_BENCH_CMD || BENCHMARK_CMD}\n` +
        `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
        `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
        { model: MODEL.mechanical, label: `embedded-inplace-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      embLatency = Number(embResult?.latency_ms || 0)
      embBclass = embResult?.heuristic_bclass || 'unknown'
      embMetrics = embResult?.metrics || { latency_ms: embLatency }
    } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
      const _plan = typeof __embeddedEvalPlan === 'function'
        ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: COMPILE_CMD || BENCHMARK_CMD, benchmarkCmd: PROJECT_BENCH_CMD || BENCHMARK_CMD })
        : null
      if (_plan) {
        const embResult = await agentRetry(() => agent(
          `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
          `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-dispatch-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
        embLatency = Number(embResult?.latency_ms || 0)
        embBclass = embResult?.heuristic_bclass || 'unknown'
        embMetrics = embResult?.metrics || { latency_ms: embLatency }
      }
    }
    evalResult.driver_envelope = { latency_ms: embLatency, metrics: embMetrics, bottleneck_class: embBclass, backend_id: 'embedded' }
  }

  // ===========================================================================
  // Bandit Update (Algorithm 1, lines 20-23)
  // ===========================================================================
  phase('Update')

  const compiled = evalResult?.compiled || false
  const correct = evalResult?.correct || false
  const newLatency = evalResult?.latency_us || baselineLatency
  const newSpeedup = evalResult?.speedup || (baselineLatency / newLatency)

  // Reward: r_t = max(0, (T(k_t) - T(k'_t)) / T(k_t)), zero for failures
  let reward = 0
  if (compiled && correct && newLatency < selectedKernel.latency) {
    reward = Math.max(0, Math.min(1, (selectedKernel.latency - newLatency) / selectedKernel.latency))
  }

  // Update bandit statistics: incremental mean update
  const statKey = `${selectedCluster}_${selectedStrategy}`
  if (banditStats[statKey]) {
    banditStats[statKey].count += 1
    const n = banditStats[statKey].count
    banditStats[statKey].mean_reward += (reward - banditStats[statKey].mean_reward) / n
  }

  totalReward += reward

  // Add to candidate pool if valid
  if (compiled && correct) {
    const newId = candidatePool.length
    const features = evalResult?.behavioral_features || {
      normalized_time: newLatency / baselineLatency,
      registers_per_thread: selectedKernel.features.registers_per_thread,
      shared_mem_bytes: selectedKernel.features.shared_mem_bytes,
      block_dimension: selectedKernel.features.block_dimension,
      occupancy: selectedKernel.features.occupancy,
    }

    candidatePool.push({
      id: newId,
      code: generatedCode,
      latency: newLatency,
      speedup: baselineLatency / newLatency,
      features,
      hw_signature: evalResult?.hw_signature || selectedKernel.hw_signature,
      cluster: selectedCluster,
      source_strategy: selectedStrategy,
    })

    // Update best
    if (newLatency < bestKernel.latency) {
      bestKernel = { code: generatedCode, latency: newLatency, speedup: baselineLatency / newLatency }
      log(`  NEW BEST: ${newLatency.toFixed(1)}μs (${bestKernel.speedup.toFixed(2)}x) via ${selectedStrategy}`)
    }
  }

  iterationLog.push({
    t,
    cluster: selectedCluster,
    strategy: selectedStrategy,
    kernel_id: selectedKernel.id,
    compiled,
    correct,
    reward: reward.toFixed(4),
    latency: newLatency,
    speedup: compiled && correct ? (baselineLatency / newLatency).toFixed(2) : '0',
    cumulative_reward: totalReward.toFixed(3),
  })

  const statusEmoji = compiled && correct ? (reward > 0 ? '↑' : '→') : '✗'
  log(`  ${statusEmoji} t=${t}: ${selectedStrategy}@C${selectedCluster} → ${compiled && correct ? newLatency.toFixed(1) + 'μs (' + (baselineLatency / newLatency).toFixed(2) + 'x)' : 'FAIL'} r=${reward.toFixed(3)} Σr=${totalReward.toFixed(2)}`)
}

// =============================================================================
// Phase: Report — Final summary
// =============================================================================
phase('Report')

// Compute strategy statistics
const strategyStats = {}
for (const s of STRATEGIES) {
  const entries = iterationLog.filter(e => e.strategy === s)
  const successes = entries.filter(e => e.compiled && e.correct)
  strategyStats[s] = {
    attempts: entries.length,
    successes: successes.length,
    avg_reward: entries.length > 0 ? entries.reduce((sum, e) => sum + parseFloat(e.reward), 0) / entries.length : 0,
  }
}

const finalReport = await agentRetry(() => agent(`Write a KernelBand optimization report.

# KernelBand Results
- Target: ${GPU_TARGET}
- Operation: ${OP_DESCRIPTION}
- Iterations: ${ITERATIONS}
- Clusters: ${NUM_CLUSTERS} (re-cluster period: ${RECLUSTER_PERIOD})
- Evidence mode: ${EVIDENCE_MODE}
- Feature vector artifact: ${FEATURE_VECTOR_RESULT_PATH}
- Hardware signature artifact: ${HARDWARE_SIGNATURE_RESULT_PATH}

# Performance
- Baseline: ${baselineLatency} μs
- Best: ${bestKernel.latency === Infinity ? 'No improvement' : bestKernel.latency.toFixed(1) + ' μs'}
- Best Speedup: ${bestKernel.speedup.toFixed(2)}x
- Cumulative Reward: ${totalReward.toFixed(3)}
- Candidate Pool Size: ${candidatePool.length}

# Strategy Statistics:
${STRATEGIES.map(s => {
  const st = strategyStats[s]
  return `- ${s}: ${st.attempts} attempts, ${st.successes} successes, avg_reward=${st.avg_reward.toFixed(3)}`
}).join('\n')}

# Bandit State (final μ̂ and N):
${Object.entries(banditStats).map(([k, v]) => `  ${k}: μ̂=${v.mean_reward.toFixed(3)}, N=${v.count}, mask=${v.mask}`).join('\n')}

# Iteration Log (last 10):
${iterationLog.slice(-10).map(e => `  t=${e.t}: ${e.strategy}@C${e.cluster} → ${e.speedup}x (r=${e.reward})`).join('\n')}

Analyze:
1. Which strategies were most/least effective? Why (relate to hardware bottleneck)?
2. Did the bandit converge to exploiting the best strategy or keep exploring?
3. Was hardware-aware pruning effective? How many invalid strategies were avoided?
4. Clustering effectiveness: did kernels in the same cluster respond similarly?
5. Recommendations for further optimization (more iterations, different K, etc.)

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the final results above (speedup is the best speedup ${bestKernel.speedup.toFixed(2)} as a number, or null if no improvement):
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","technique":"<the winning strategy>","speedup":<number or null>,"note":"<best latency us + which strategy won + cumulative reward, one line>"}`, {
  label: 'report',
  phase: 'Report',
  model: MODEL.judgment,
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
  target_gpu: GPU_TARGET,
  operation: OP_DESCRIPTION,
  iterations: ITERATIONS,
  baseline_latency_us: baselineLatency,
  best_latency_us: bestKernel.latency === Infinity ? null : bestKernel.latency,
  best_speedup: bestKernel.speedup,
  cumulative_reward: totalReward,
  candidate_pool_size: candidatePool.length,
  strategy_stats: strategyStats,
  bandit_stats: banditStats,
  feature_vector_result_path: FEATURE_VECTOR_RESULT_PATH,
  hardware_signature_result_path: HARDWARE_SIGNATURE_RESULT_PATH,
  evidence_mode: EVIDENCE_MODE,
  iteration_log: iterationLog,
  report: finalReport,
}
