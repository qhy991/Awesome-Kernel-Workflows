export const meta = {
  name: 'astra-kernel-optimization',
  description: 'Astra-style multi-agent optimization loop for existing CUDA kernels from production frameworks',
  whenToUse: 'When you already have a CUDA kernel implementation and want a multi-agent loop to generate tests, profile baseline and candidates, plan optimizations, edit code, and preserve reintegration notes. Designed for SGLang-style production kernels where the task is not PyTorch-to-CUDA synthesis but improving an existing CUDA/PyBind implementation.',
  phases: [
    { title: 'Setup', detail: 'Read initial CUDA kernel, baseline/export function contract, and integration mode' },
    { title: 'PrepareTests', detail: 'Generate or validate representative correctness and performance tests' },
    { title: 'ProfileBaseline', detail: 'Measure baseline latency and identify kernel bottlenecks' },
    { title: 'Plan', detail: 'Planning agent proposes the next optimization from logs and profiles' },
    { title: 'Code', detail: 'Coding agent edits the current best CUDA kernel according to the plan' },
    { title: 'Evaluate', detail: 'Testing and profiling agents validate correctness and speedup' },
    { title: 'Record', detail: 'Record candidate evidence, update best kernel, and carry forward lessons' },
    { title: 'PostProcess', detail: 'Prepare reintegration notes and final drop-in patch guidance' },
    { title: 'Report', detail: 'Return best kernel, measured trajectory, failures, and follow-up opportunities' },
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

const WORKFLOW_NAME = 'astra-kernel-optimization'


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
// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.

const WORKFLOW_SUITABILITY = {
  supported_languages: ['cuda'],
  supported_problem_types: ['cuda-kernel-optimization'],
  problem_types: ['existing production CUDA/PyBind kernel optimization', 'test/profile/plan/code/evaluate loop for CUDA repositories'],
  reason: 'Astra is a production CUDA kernel optimization workflow and expects existing CUDA code, tests, and profiling artifacts.',
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
// Astra — Multi-Agent CUDA Kernel Performance Optimization Workflow
// =============================================================================
//
// Source: "Astra: A Multi-Agent System for GPU Kernel Performance Optimization"
//         arXiv:2509.07506 — Anjiang Wei, Tianran Sun, Yogesh Seenichamy,
//         Hang Song, Anne Ouyang, Azalia Mirhoseini, Ke Wang, Alex Aiken
//         NeurIPS 2025 Fourth Workshop on Deep Learning for Code
//         Repo: https://github.com/Anjiang-Wei/Astra
//
// Core idea:
//   Start from an existing CUDA kernel (for example, extracted from SGLang) and
//   coordinate specialized agents for testing, profiling, planning, and coding.
//   Each iteration edits the current best kernel, then accepts progress only
//   when real correctness and runtime evidence improves over the baseline.
//
// Usage:
//   Workflow({name: 'astra-kernel-optimization', args: {
//     kernel_path: '/path/to/kernel.cu',
//     compare_kind: 'rmsnorm',
//     baseline_func: 'fused_add_rmsnorm',
//     generated_export_func: 'sgl_fused_add_rmsnorm',
//     baseline_module: 'sgl_kernel',
//     test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     iterations: 5,
//     target_gpu: 'H100',
//     exp_dir: '/tmp/astra_exp',
//   }})
//
// Evaluator JSON contract:
//   test_command / benchmark_command should write JSON at {result_path}:
//   {
//     "compiled": true,
//     "correct": true,
//     "speedup": 1.23,
//     "runtime_ms": 0.12,
//     "baseline_runtime_ms": 0.15,
//     "error_message": "",
//     "profile_summary": ""
//   }
//
// =============================================================================

// --- Required Args ---
let INITIAL_KERNEL_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = INITIAL_KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'

// --- Optional Args matching Astra repo concepts ---
const COMPARE_KIND = args.compare_kind || 'generic'
const BASELINE_MODULE = args.baseline_module || 'sgl_kernel'
const BASELINE_FUNC = args.baseline_func || 'sgl_fused_add_rmsnorm'
const GENERATED_EXPORT_FUNC = args.generated_export_func || BASELINE_FUNC
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || ''
const MAX_ITERATIONS = args.iterations || 5
const TARGET_GPU = args.target_gpu || 'H100'
const EXP_DIR = args.exp_dir || '/tmp/astra_exp'
const INTEGRATION_MODE = args.integration_mode || 'standalone' // standalone | sglang
const LANGUAGE = args.language || 'cuda'
const SEED_CANDIDATES = args.seed_candidates || 3

// --- Project-native integration (embedded kernels via integration-strategist) ---
// BENCH_CMD already maps to args.benchmark_command above; reuse it as the embedded
// benchmark command (do NOT redeclare). TEST_CMD likewise reused for embedded test.
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const BUILD_CMD = args.build_command || ''
const REGISTER_SCRIPT = args.register_script || ''

if (!INITIAL_KERNEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

// --- Backend driver wiring (P5c Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_SETUP_LANG_TOKEN = 'CUDA'
const LEGACY_TESTING_LANG_TOKEN = 'CUDA'
const LEGACY_PROFILING_LANG_TOKEN = 'CUDA'
const LEGACY_PLAN_LANG_TOKEN = 'CUDA'
const LEGACY_CODE_LANG_TOKEN = 'CUDA'
const LEGACY_EVAL_LANG_TOKEN = 'CUDA'
const LEGACY_LESSON_LANG_TOKEN = 'CUDA'
const LEGACY_SOURCE_EXT = '.cu'
const LEGACY_FENCE_TOKEN = 'cuda'
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

// Intersectional guard (P5c plan §3 + §9.1): integration_mode='sglang' is
// vendor-locked (NVIDIA + sglang); refuse driver path when sglang mode is
// combined with a non-cuda backend.
if (USE_DRIVER && INTEGRATION_MODE === 'sglang' && RESOLVED_BACKEND && RESOLVED_BACKEND !== 'cuda') {
  throw new Error(
    `Astra integration_mode="sglang" requires backend_dir to be a CUDA driver; ` +
    `got backend_dir=${args.backend_dir} (resolved backend=${RESOLVED_BACKEND}).`
  )
}

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
// Hybrid kernel-path helper (KDA-style): AccelOpt-pattern for user-supplied
// kernel_path; KernelAgent-workspace-local for generated/per-iteration files.
function astraKernelPath(userKernelPath) {
  if (userKernelPath) return userKernelPath
  const ext = USE_DRIVER ? DRIVER_SOURCE_EXT : LEGACY_SOURCE_EXT
  return `${EXP_DIR}/generated/kernel${ext}`
}
function astraIterKernelPath(iteration) {
  const ext = USE_DRIVER ? DRIVER_SOURCE_EXT : LEGACY_SOURCE_EXT
  return `${EXP_DIR}/astra_iter_${iteration}${ext}`
}

// --- State ---
let initialKernelCode = ''
let currentBestCode = ''
let baselineProfile = null
let bestResult = null
let testSuite = []
let runLog = []
let lessons = []
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null

function scoreResult(result) {
  if (!result || !result.compiled) return [0, 0, 0]
  if (!result.correct) return [1, 0, 0]
  return [1, 1, result.speedup || 0]
}

function isBetter(candidate, incumbent) {
  if (!incumbent) return !!candidate?.correct
  const a = scoreResult(candidate)
  const b = scoreResult(incumbent)
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return a[i] > b[i]
  }
  return false
}

// =============================================================================
// Phase 1: Setup
// =============================================================================
phase('Setup')

if (USE_DRIVER) {
  DRIVER = await agent(
    `Load the backend driver at ${BACKEND_DIR} and return its manifest plus idioms verbatim.\n` +
    `1. Run exactly: \`cat ${driverPath('manifest.json')}\` and parse JSON.\n` +
    `2. Run exactly: \`cat ${driverPath('idioms.json')}\` and parse JSON.\n` +
    `Return {present, backend_id, source_ext, aux_ext, lang_fence, impl_requirements, methods}.`,
    { model: MODEL.mechanical, label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH })
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
  const generated = await agent(`No kernel_path was provided. Generate and verify an initial ${langToken(LEGACY_SETUP_LANG_TOKEN)} kernel before starting Astra optimization.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- compare_kind: ${COMPARE_KIND}
- baseline_func: ${BASELINE_FUNC}
- generated_export_func: ${GENERATED_EXPORT_FUNC}
- integration_mode: ${INTEGRATION_MODE}
- language: ${LANGUAGE}
- target_gpu: ${TARGET_GPU}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- test_command: ${TEST_CMD || '(not provided)'}
- benchmark_command: ${BENCH_CMD || '(not provided)'}

# Contract
${USE_DRIVER
  ? `Generate ${SEED_CANDIDATES} complete ${DRIVER_LANG_FENCE} candidates and write the best one to ${astraKernelPath('')}. Run available commands using {kernel_path}/{result_path}. Return the best verified generated kernel path.`
  : `Generate ${SEED_CANDIDATES} complete candidates under ${EXP_DIR}/generated/. Run available commands using {kernel_path}/{result_path}. Return the best verified generated kernel path.`}`, {
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
  })
  initialCandidates = generated.initial_candidates || []
  initialGenerationResult = generated.initial_generation_result || { verified: false }
  generatedKernelPath = generated.generated_kernel_path || ''
  if (!generatedKernelPath) throw new Error('Generation mode did not produce generated_kernel_path')
  if ((TEST_CMD || BENCH_CMD) && initialGenerationResult.verified === false) throw new Error('No generated seed passed correctness evidence')
  INITIAL_KERNEL_PATH = generatedKernelPath
}

const setup = await agent(`You are the Astra setup agent for production ${langToken(LEGACY_SETUP_LANG_TOKEN)} kernel optimization.

# Inputs
- kernel_path: ${INITIAL_KERNEL_PATH}
- compare_kind: ${COMPARE_KIND}
- baseline_module: ${BASELINE_MODULE}
- baseline_func: ${BASELINE_FUNC}
- generated_export_func: ${GENERATED_EXPORT_FUNC}
- integration_mode: ${INTEGRATION_MODE}
- target_gpu: ${TARGET_GPU}
- exp_dir: ${EXP_DIR}

# Tasks
1. Read the initial ${langToken(LEGACY_SETUP_LANG_TOKEN)} kernel from kernel_path.
2. Identify exported ${USE_DRIVER ? `${DRIVER_LANG_FENCE}` : 'PyBind/CUDA'} entry points and the function that must remain callable as generated_export_func.
3. Summarize the baseline function contract and compare_kind semantics.
4. Identify whether this should be optimized as a standalone kernel or reintegrated into SGLang-style code.
5. List likely performance-sensitive regions before profiling.

Return the kernel text and integration contract.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"kernel_intake","note":"<entry points + integration mode + likely hotspots, one line>"}`, {
  label: 'setup-astra',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      initial_kernel_code: { type: 'string' },
      kernel_summary: { type: 'string' },
      entry_points: { type: 'array', items: { type: 'string' } },
      integration_contract: { type: 'string' },
      likely_hotspots: { type: 'array', items: { type: 'string' } },
    },
    required: ['initial_kernel_code', 'integration_contract'],
  },
})

initialKernelCode = setup.initial_kernel_code || ''
currentBestCode = initialKernelCode

// =============================================================================
// Phase 2: PrepareTests
// =============================================================================
phase('PrepareTests')

const tests = await agent(`You are Astra's Testing Agent. Build a correctness and benchmark test suite for this ${langToken(LEGACY_TESTING_LANG_TOKEN)} kernel.

# Compare kind
${COMPARE_KIND}

# Baseline module/function
${BASELINE_MODULE}.${BASELINE_FUNC}

# Generated export function
${GENERATED_EXPORT_FUNC}

# User-provided commands
- test_command: ${TEST_CMD || '(not provided)'}
- benchmark_command: ${BENCH_CMD || '(not provided)'}

# Kernel summary
${setup.kernel_summary || ''}

# Requirements
1. If test_command is provided, treat it as authoritative and document its JSON contract.
2. If no command is provided, do not propose a concrete program or harness; describe the required command contract and mark evidence as missing.
3. Generate representative shapes for compare_kind. Cover small, medium, and large cases.
4. Include tolerance rules for floating-point comparison.
5. Do not accept performance measurements unless they come from benchmark_command or an explicit user-provided harness.

Return the test plan and shape suite.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"PrepareTests","ts":"<ts>","status":"done","technique":"test_suite","speedup":null,"note":"<number of shapes + tolerance + whether commands are authoritative or evidence missing>"}`, {
  label: 'prepare-tests',
  phase: 'PrepareTests',
  schema: {
    type: 'object',
    properties: {
      test_cases: { type: 'array', items: { type: 'object' } },
      correctness_tolerance: { type: 'string' },
      harness_plan: { type: 'string' },
      authoritative_commands: { type: 'array', items: { type: 'string' } },
    },
    required: ['test_cases', 'harness_plan'],
  },
})

testSuite = tests.test_cases || []

// =============================================================================
// Phase 3: ProfileBaseline
// =============================================================================
phase('ProfileBaseline')

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it in the Evaluate driver branch below. The agent only classifies the task
// (fuzzy); the substrate stamps confidence by method (measured/inferred/hypothesized)
// -- not the agent. See _substrate/profiling/README.md. Falls back to
// native_profiler if undecided. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }
if (USE_DRIVER) {
  const _pd = await agent(
    `Read ${INITIAL_KERNEL_PATH}; classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${BACKEND_DIR}/manifest.json --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'ProfileBaseline', schema: JSON_PASSTHROUGH })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}

// --- integration-strategist: route build/test mode (standalone vs embedded_*). ---
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agent(
    `Read ${INITIAL_KERNEL_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${INITIAL_KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
    `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'ProfileBaseline', schema: JSON_PASSTHROUGH })
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
  await agent(`Byte-exact backup: run \`cp -a "${INITIAL_KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'ProfileBaseline', schema: JSON_PASSTHROUGH })
}
// A-O1 closure: native_profiler but no standalone driver path (embedded/legacy) → perf_heuristic
if (PROFILING_DECISION.method === 'native_profiler' && !USE_DRIVER_STANDALONE) {
  log(`profiling: native_profiler but no standalone driver path -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'test-harness-perf', rationale: 'native_profiler but no standalone driver -> perf_heuristic' }
}

baselineProfile = await agent(`You are Astra's Profiling Agent. Establish the baseline profile for the initial kernel.

# Initial kernel
\`\`\`${fenceToken()}
${initialKernelCode.substring(0, 10000)}
\`\`\`

# Commands
- test_command: ${TEST_CMD || '(not provided)'}
- benchmark_command: ${BENCH_CMD || '(not provided)'}

# Test suite
\`\`\`json
${JSON.stringify(testSuite, null, 2).substring(0, 6000)}
\`\`\`

# Required behavior
1. Run or specify the exact benchmark command with {kernel_path} and {result_path} substitutions.
2. Report per-shape baseline runtime if available.
3. Identify bottlenecks from measurements, code inspection, and optional profiler output.
4. If command execution is unavailable, mark measured=false and keep performance claims conservative.

Return baseline profile evidence.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (status="done" if measured, else "error"):
{"workflow":"${WORKFLOW_NAME}","phase":"ProfileBaseline","ts":"<ts>","status":"<done|error>","technique":"baseline_profile","speedup":null,"note":"<measured? baseline runtime ms + top bottlenecks>"}`, { model: MODEL.profile,
  label: 'profile-baseline',
  phase: 'ProfileBaseline',
  schema: {
    type: 'object',
    properties: {
      measured: { type: 'boolean' },
      baseline_runtime_ms: { type: 'number' },
      per_shape_runtime: { type: 'array', items: { type: 'object' } },
      bottlenecks: { type: 'array', items: { type: 'string' } },
      profiler_notes: { type: 'string' },
    },
    required: ['measured', 'bottlenecks'],
  },
})

// =============================================================================
// Iterative Astra Loop: Plan -> Code -> Evaluate -> Record
// =============================================================================
for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
  log(`\n=== Astra iteration ${iteration + 1}/${MAX_ITERATIONS} | best=${bestResult?.speedup || 0}x ===`)

  phase('Plan')

  const plan = await agent(`You are Astra's Planning Agent. Propose the next ${langToken(LEGACY_PLAN_LANG_TOKEN)} optimization.

# Current best kernel
\`\`\`${fenceToken()}
${currentBestCode.substring(0, 12000)}
\`\`\`

# Baseline profile
\`\`\`json
${JSON.stringify(baselineProfile, null, 2).substring(0, 6000)}
\`\`\`

# Prior run log
\`\`\`json
${JSON.stringify(runLog.slice(-6), null, 2).substring(0, 8000)}
\`\`\`

# Lessons
${lessons.join('\n') || 'No lessons yet.'}

# Planning rules
1. Pick one coherent optimization direction for this iteration.
2. Ground the plan in bottlenecks or failed evidence, not generic advice.
3. Preserve generated_export_func=${GENERATED_EXPORT_FUNC}.
4. Prefer production-safe changes: loop transformations, memory access improvements, ${USE_DRIVER ? `${DRIVER_LANG_FENCE} idiomatic` : 'CUDA'} intrinsics, fast math only when correctness tolerance allows it.
5. Include explicit risk and rollback criteria.

Return a structured plan.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is loop iteration ${iteration}):
{"workflow":"${WORKFLOW_NAME}","phase":"Plan","ts":"<ts>","status":"done","candidate_id":"iter-${iteration}","technique":"<the optimization direction you chose>","speedup":null,"note":"<target regions + expected speedup reason, one line>"}`, {
    label: `plan-${iteration}`,
    phase: 'Plan',
    schema: {
      type: 'object',
      properties: {
        optimization_goal: { type: 'string' },
        target_regions: { type: 'array', items: { type: 'string' } },
        proposed_changes: { type: 'array', items: { type: 'string' } },
        correctness_risks: { type: 'array', items: { type: 'string' } },
        expected_speedup_reason: { type: 'string' },
      },
      required: ['optimization_goal', 'proposed_changes'],
    },
  })

  phase('Code')

  const code = await agent(`You are Astra's Coding Agent. Apply the planning agent's optimization to the current best ${langToken(LEGACY_CODE_LANG_TOKEN)} kernel.

# Current best kernel
\`\`\`${fenceToken()}
${currentBestCode.substring(0, 14000)}
\`\`\`

# Integration contract
${setup.integration_contract}

# Plan
\`\`\`json
${JSON.stringify(plan, null, 2)}
\`\`\`

# Hard requirements
1. Return complete ${USE_DRIVER ? `${DRIVER_LANG_FENCE}` : 'CUDA/C++'} code, not a patch.
2. Keep generated export function callable as ${GENERATED_EXPORT_FUNC}.
3. Preserve includes, ${USE_DRIVER ? 'host/launcher' : 'PyBind/export'} surface, and baseline-compatible function signature unless the integration contract allows a change.
4. Apply only the planned optimization. Do not combine unrelated rewrites.
5. Add short comments only where they clarify non-obvious ${langToken(LEGACY_CODE_LANG_TOKEN)} choices.

Return the candidate code and changed regions.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is loop iteration ${iteration}):
{"workflow":"${WORKFLOW_NAME}","phase":"Code","ts":"<ts>","status":"done","candidate_id":"iter-${iteration}","technique":"<the optimization you applied this iteration>","speedup":null,"note":"<changed regions, one line>"}`, {
    label: `code-${iteration}`,
    phase: 'Code',
    schema: {
      type: 'object',
      properties: {
        candidate_code: { type: 'string' },
        changed_regions: { type: 'array', items: { type: 'string' } },
        implementation_notes: { type: 'string' },
      },
      required: ['candidate_code'],
    },
  })

  phase('Evaluate')

  const evaluation = await agent(`You are Astra's Testing and Profiling Agents working together. Evaluate this candidate with real evidence.

# Candidate code
\`\`\`${fenceToken()}
${(code.candidate_code || '').substring(0, 16000)}
\`\`\`

# Commands
- test_command: ${TEST_CMD || '(not provided)'}
- benchmark_command: ${BENCH_CMD || '(not provided)'}

# Path convention
- kernel_path: ${USE_DRIVER ? astraIterKernelPath(iteration) : `${EXP_DIR}/astra_iter_${iteration}.cu`}
- result_path: ${EXP_DIR}/astra_iter_${iteration}.json

# Test suite
\`\`\`json
${JSON.stringify(testSuite, null, 2).substring(0, 6000)}
\`\`\`

# Required behavior
1. If commands are available, write candidate to kernel_path and run correctness and benchmark commands.
2. Parse result_path JSON.
3. If commands are unavailable, compiled=false, correct=false, speedup=0, and explain missing evidence.
4. Runtime and speedup must come from measured evidence only.
5. Include profile summary and failure logs that Planning can use next iteration.

Return evaluator evidence.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if it compiled and is correct, else "error"; speedup is the measured speedup number, or null if unavailable; this is loop iteration ${iteration}):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"iter-${iteration}","speedup":<number or null>,"technique":"candidate_evaluation","note":"<compiled? correct? runtime ms; or the failure reason>"}`, {
    label: `evaluate-${iteration}`,
    phase: 'Evaluate',
    schema: {
      type: 'object',
      properties: {
        compiled: { type: 'boolean' },
        correct: { type: 'boolean' },
        speedup: { type: 'number' },
        runtime_ms: { type: 'number' },
        baseline_runtime_ms: { type: 'number' },
        per_shape_results: { type: 'array', items: { type: 'object' } },
        error_message: { type: 'string' },
        profile_summary: { type: 'string' },
      },
      required: ['compiled', 'correct', 'speedup'],
    },
  })

  if (USE_DRIVER_STANDALONE) {
    const kPath = astraIterKernelPath(iteration)
    const buildOut = `${EXP_DIR}/astra_iter_${iteration}.artifact`
    const profOut = `${EXP_DIR}/astra_iter_${iteration}.prof.native`
    await agent(
      `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
      `Return its stdout JSON verbatim.`,
      { model: MODEL.mechanical, label: `driver-build-${iteration}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    const runOut = await agent(
      `${driverSh('run.sh', `--artifact ${buildOut} --kernel ${kPath}`)}\n` +
      `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
      { model: MODEL.profile, label: `driver-run-${iteration}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    let evidenceOut
    if (PROFILING_DECISION.method === 'native_profiler') {
      await agent(
        `${driverSh('profile.sh', `--artifact ${buildOut} --kernel ${kPath} --out ${profOut}`)}\n` +
        `Return {ok, native_path}.`,
        { model: MODEL.profile, label: `driver-profile-${iteration}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
      evidenceOut = await agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
        `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
        { model: MODEL.mechanical, label: `driver-to-evidence-${iteration}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    } else {
      // Profiling-strategist chose method='${PROFILING_DECISION.method}'; do NOT run
      // the native profiler. Keep the run.sh latency path and, when perf_heuristic,
      // normalize throughput via the strategist normalizer.
      const _normalizer = PROFILING_DECISION.normalizer || 'perf_to_evidence.py'
      evidenceOut = await agent(
        `${driverSh('run.sh', `--artifact ${buildOut} --kernel ${kPath}`)} already produced throughput above. ` +
        `If method='perf_heuristic', normalize it into canonical metrics via ` +
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${_normalizer} --baseline ${EXP_DIR}/astra_iter_${iteration}.result.json\`. ` +
        `Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
        `Also write heuristic_bclass (memory_bound|compute_bound|latency_bound) based on the throughput ratio so diagnose.py does not fall to unknown. ` +
        `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, heuristic_bclass, coverage, source_backend}.`,
        { model: MODEL.mechanical, label: `driver-to-evidence-${iteration}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    }
    const diagOut = await agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
      `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
      { model: MODEL.mechanical, label: `driver-diagnose-${iteration}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    await agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${EXP_DIR}/astra_iter_${iteration}.result.json\`.\n` +
      `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
      { model: MODEL.mechanical, label: `driver-anti-cheat-${iteration}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    evaluation.driver_envelope = {
      latency_ms: Number((runOut && runOut.latency_ms) || 0),
      metrics: (evidenceOut && evidenceOut.metrics) || {},
      bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
      backend_id: DRIVER_BACKEND_ID,
    }
  } else if (IS_EMBEDDED) {
    // --- Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch) ---
    // Serial: Astra's candidate loop is already a sequential for(iteration) loop, so there
    // is no race on the shared INITIAL_KERNEL_PATH / project build.
    const kPath = astraIterKernelPath(iteration)
    const variant = `astra_${iteration + 1}`.replace(/[^A-Za-z0-9_]/g, '_')
    let embLatency = 0, embMetrics = {}, embBclass = 'unknown'
    if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
      const embResult = await agent(
        `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project kernel: ${INITIAL_KERNEL_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
        `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${INITIAL_KERNEL_PATH}\n` +
        `2. Apply candidate: cp ${kPath} ${INITIAL_KERNEL_PATH}\n3. Build: ${BUILD_CMD}\n4. Test: ${TEST_CMD || BENCH_CMD}\n5. Benchmark: ${BENCH_CMD || TEST_CMD}\n` +
        `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${INITIAL_KERNEL_PATH}\n` +
        `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
        { model: MODEL.mechanical, label: `embedded-inplace-${iteration}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
      embLatency = Number(embResult?.latency_ms || 0)
      embBclass = embResult?.heuristic_bclass || 'unknown'
      embMetrics = embResult?.metrics || { latency_ms: embLatency }
    } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
      const _plan = typeof __embeddedEvalPlan === 'function'
        ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: TEST_CMD || BENCH_CMD, benchmarkCmd: BENCH_CMD || TEST_CMD })
        : null
      if (_plan) {
        const embResult = await agent(
          `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
          `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-dispatch-${iteration}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
        embLatency = Number(embResult?.latency_ms || 0)
        embBclass = embResult?.heuristic_bclass || 'unknown'
        embMetrics = embResult?.metrics || { latency_ms: embLatency }
      }
    }
    evaluation.driver_envelope = { latency_ms: embLatency, metrics: embMetrics, bottleneck_class: embBclass, backend_id: 'embedded' }
  }

  phase('Record')

  const record = {
    iteration,
    plan,
    changed_regions: code.changed_regions || [],
    implementation_notes: code.implementation_notes || '',
    evaluation,
  }
  runLog.push(record)

  if (isBetter(evaluation, bestResult)) {
    bestResult = evaluation
    currentBestCode = code.candidate_code || currentBestCode
  }

  const lesson = await agent(`Distill the useful lesson from this Astra iteration.

# Record
\`\`\`json
${JSON.stringify(record, null, 2).substring(0, 8000)}
\`\`\`

# Best updated?
${bestResult === evaluation}

Return one or two concise lessons for the next Planning Agent. Focus on measured evidence.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is loop iteration ${iteration}):
{"workflow":"${WORKFLOW_NAME}","phase":"Record","ts":"<ts>","status":"done","candidate_id":"iter-${iteration}","technique":"lesson_distillation","speedup":null,"note":"<the distilled lesson, one line>"}`, {
    label: `record-${iteration}`,
    phase: 'Record',
    schema: {
      type: 'object',
      properties: {
        lessons: { type: 'array', items: { type: 'string' } },
      },
      required: ['lessons'],
    },
  })

  lessons.push(...(lesson.lessons || []))
}

// =============================================================================
// Phase 8: PostProcess
// =============================================================================
phase('PostProcess')

const postProcess = await agent(`Prepare final Astra post-processing guidance.

# Integration mode
${INTEGRATION_MODE}

# Generated export function
${GENERATED_EXPORT_FUNC}

# Best kernel
\`\`\`${fenceToken()}
${currentBestCode.substring(0, 12000)}
\`\`\`

# Best result
\`\`\`json
${JSON.stringify(bestResult, null, 2)}
\`\`\`

# Tasks
1. Produce reintegration notes for standalone or SGLang mode.
2. Identify files/functions likely needing manual review.
3. Note any precision or shape limitations discovered during testing.
4. Provide a safe rollback criterion.

Return post-processing notes.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"PostProcess","ts":"<ts>","status":"done","technique":"reintegration_notes","speedup":null,"note":"<integration mode + key limitations + rollback criterion, one line>"}`, {
  label: 'post-process',
  phase: 'PostProcess',
  schema: {
    type: 'object',
    properties: {
      reintegration_notes: { type: 'string' },
      manual_review_items: { type: 'array', items: { type: 'string' } },
      limitations: { type: 'array', items: { type: 'string' } },
      rollback_criteria: { type: 'array', items: { type: 'string' } },
    },
    required: ['reintegration_notes'],
  },
})

// =============================================================================
// Phase 9: Report
// =============================================================================
phase('Report')

const finalReport = await agent(`Write a concise technical report for this Astra optimization campaign.

# Kernel
${INITIAL_KERNEL_PATH}

# Compare kind
${COMPARE_KIND}

# Baseline/profile
\`\`\`json
${JSON.stringify(baselineProfile, null, 2).substring(0, 6000)}
\`\`\`

# Best result
\`\`\`json
${JSON.stringify(bestResult, null, 2)}
\`\`\`

# Run log
\`\`\`json
${JSON.stringify(runLog, null, 2).substring(0, 12000)}
\`\`\`

# Post-process notes
\`\`\`json
${JSON.stringify(postProcess, null, 2)}
\`\`\`

Cover:
1. Which agent loop decisions mattered most.
2. Which optimizations improved measured runtime.
3. Which attempted changes failed and why.
4. Whether evidence is strong enough for production reintegration.
5. Next high-value optimization opportunities.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (speedup is the best measured speedup, or null if no candidate was verified):
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","technique":"campaign_summary","speedup":<number or null>,"note":"<best speedup + whether evidence is production-ready, one line>"}`, {
  label: 'final-report',
  phase: 'Report',
})

// embedded_inplace exit safety net: unconditionally restore pristine original.
if (ORIGINAL_BACKUP) {
  await agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${INITIAL_KERNEL_PATH}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH })
}

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  kernel_path: INITIAL_KERNEL_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  compare_kind: COMPARE_KIND,
  baseline_func: BASELINE_FUNC,
  generated_export_func: GENERATED_EXPORT_FUNC,
  best_speedup: bestResult?.speedup || 0,
  best_runtime_ms: bestResult?.runtime_ms || null,
  baseline_runtime_ms: bestResult?.baseline_runtime_ms || baselineProfile?.baseline_runtime_ms || null,
  best_kernel_code: currentBestCode,
  iterations: MAX_ITERATIONS,
  run_log: runLog,
  lessons,
  post_process: postProcess,
  report: finalReport,
}
