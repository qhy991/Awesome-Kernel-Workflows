export const meta = {
  name: 'cudallm-fsr-kernel-generation',
  description: 'CUDA-LLM Feature Search and Reinforcement workflow for evaluator-guided CUDA kernel generation',
  whenToUse: 'When generating or optimizing CUDA kernels from a task specification and you want explicit CUDA feature search rather than generic iterative prompting. Maintains feature-level scores for tiling, shared memory, vectorization, warp primitives, occupancy, and fast math, then reinforces feature choices using compile, correctness, and latency evidence.',
  phases: [
    { title: 'Setup', detail: 'Read task specification, reference implementation, target GPU, and evaluator contract' },
    { title: 'FeatureCatalog', detail: 'Build a CUDA optimization feature space tailored to the task' },
    { title: 'GenerateTests', detail: 'Create diverse correctness and boundary tests for the task' },
    { title: 'SelectFeatures', detail: 'Choose a feature combination from scored feature history' },
    { title: 'GenerateKernel', detail: 'Generate candidate CUDA code conditioned on selected features' },
    { title: 'Evaluate', detail: 'Compile, correctness-test, and benchmark the candidate' },
    { title: 'Reinforce', detail: 'Update feature scores from measured compile/correctness/speedup reward' },
    { title: 'Report', detail: 'Return best kernel, feature reward table, failures, and next feature sets' },
  ],
}

const WORKFLOW_NAME = 'cudallm-fsr-kernel-generation'


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
  supported_problem_types: ['cuda-kernel-generation', 'cuda-kernel-optimization'],
  problem_types: ['CUDA kernel generation with feature search', 'reinforcement-style feature scoring from evaluator feedback'],
  reason: 'CUDA-LLM is modeled around CUDA feature search and CUDA kernel generation rewards.',
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
// CUDA-LLM — Feature Search and Reinforcement (FSR) Workflow
// =============================================================================
//
// Source: "CUDA-LLM: LLMs Can Write Efficient CUDA Kernels"
//         arXiv:2506.09092 — Wentao Chen, Jiace Zhu, Qi Fan, Yehan Ma, An Zou
//
// Boundary:
//   This workflow implements an agent-executable Feature Search and
//   Reinforcement loop. It does not train a model. It searches CUDA optimization
//   features, generates kernels, evaluates them with real compile/correctness/
//   latency evidence, and reinforces feature choices for later iterations.
// adaptation_scope: workflow_adaptation — this is not the full CUDA-LLM training
// or model-development pipeline.
//
// Usage:
//   Workflow({name: 'cudallm-fsr-kernel-generation', args: {
//     problem_path: '/path/to/task.md',
//     reference_code_path: '/path/to/reference.py',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     target_gpu: 'H100',
//     iterations: 8,
//     feature_budget: 4,
//     samples_per_feature_set: 2,
//     rtol: 0.01,
//     atol: 0.01,
//     exp_dir: '/tmp/cudallm_fsr_exp',
//   }})
//
// Evaluator JSON contract:
//   benchmark_command should write JSON at {result_path}:
//   {
//     "compiled": true,
//     "correct": true,
//     "speedup": 1.23,
//     "latency_ms": 0.12,
//     "baseline_latency_ms": 0.15,
//     "error_message": "",
//     "passed_tests": 128,
//     "total_tests": 128
//   }
//
// =============================================================================

// --- Required Args ---
const PROBLEM_DEFINITION = args.problem_definition || ''
const TASK_SPEC_PATH = args.problem_path || ''
const REFERENCE_CODE_PATH = args.reference_code_path || ''
const EVAL_CMD = args.benchmark_command || ''

// --- Optional Args ---
const TARGET_GPU = args.target_gpu || 'H100'
const ITERATIONS = args.iterations || 8
const FEATURE_BUDGET = args.feature_budget || 4
const SAMPLES_PER_FEATURE_SET = args.samples_per_feature_set || 2
const RTOL = args.rtol ?? 0.01
const ATOL = args.atol ?? 0.01
const EXP_DIR = args.exp_dir || '/tmp/cudallm_fsr_exp'
const DRIVER_PROBLEM_PATH = args.problem_json_path || `${EXP_DIR}/driver_problem.json`
const PROFILE_SOURCE_PATH = args.profile_source_path || REFERENCE_CODE_PATH || ''
const ADAPTATION_SCOPE = 'workflow_adaptation'
const INPUT_MODE = 'generate_then_optimize'

// --- Backend driver wiring (P5c Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_SETUP_LANG_TOKEN = 'CUDA'
const LEGACY_CATALOG_LANG_TOKEN = 'CUDA'
const LEGACY_TESTS_LANG_TOKEN = 'CUDA-LLM'
const LEGACY_SELECT_LANG_TOKEN = 'CUDA'
const LEGACY_GENERATE_LANG_TOKEN = 'CUDA'
const LEGACY_EVAL_LANG_TOKEN = 'CUDA'
const LEGACY_REINFORCE_LANG_TOKEN = 'CUDA'
const LEGACY_REPORT_LANG_TOKEN = 'CUDA-LLM FSR'
const LEGACY_SOURCE_EXT = '.cu'
const LEGACY_RESULT_EXT = '.json'
const LEGACY_PURE_LANG_PHRASE = 'pure CUDA/C++ only'
const LEGACY_FENCE_TOKEN = 'cuda'
// L3 deferred (R2): triton driver has no `feature_catalog` idiom today; the
// driver path falls back to LEGACY_TRITON_FEATURE_FALLBACK below. Tightening
// is filed as a P5e/P5f L3 follow-up per P5c plan §5.2 B2 + §8 R2.
const LEGACY_TRITON_FEATURE_FALLBACK = 'Explore the standard Triton optimization idioms appropriate to this driver (block tiling, vectorized loads, masking, reductions, autotune configs).'
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}
function driverPy(script, cliArgs) {
  return `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}
function driverProfileOutPath(iter, sample) {
  // .sqlite suffix satisfies cuda/profile.sh nsys fallback; ncu still writes CSV bytes here.
  return `${EXP_DIR}/cudallm_iter_${iter}_sample_${sample}.prof.sqlite`
}
function buildProfileShArgs(buildOut, iter, sample) {
  let cli = `--artifact ${buildOut} --problem ${DRIVER_PROBLEM_PATH} --out ${driverProfileOutPath(iter, sample)}`
  if (PROFILE_SOURCE_PATH) cli += ` --source ${PROFILE_SOURCE_PATH}`
  return cli
}
function profileStepFooter() {
  if (DRIVER_BACKEND_ID === 'cuda') {
    return 'The cuda driver profile.sh may return format ncu-csv or nsys-sqlite depending on which profiler is available.\n'
  }
  return 'Pass the format field from profile.sh stdout through to to_evidence.py.\n'
}
function buildToEvidencePrompt(profilePointer) {
  if (!profilePointer || profilePointer.ok === false) {
    const why = profilePointer?.error || profilePointer?.profiler || 'profiler unavailable'
    return `Profiler unavailable (${why}). ` +
      `Return {ok:true, metrics:{latency_ms:null,dram_pct:null,sm_pct:null,occupancy:null,_vendor:"nvidia"}, ` +
      `coverage:[], source_backend:"${DRIVER_BACKEND_ID}"}.`
  }
  const native = profilePointer.native_profile
  const format = profilePointer.format || (DRIVER && DRIVER.profiler_format) || 'ncu-csv'
  return `${driverPy('to_evidence.py', `--native ${native} --format ${format}`)}\n` +
    `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`
}

let DRIVER = null
let DRIVER_LANG_FENCE = 'cuda'
let DRIVER_IMPL_REQUIREMENTS = ''
let DRIVER_SOURCE_EXT = LEGACY_SOURCE_EXT
let DRIVER_FEATURE_CATALOG = ''
let DRIVER_BACKEND_ID = RESOLVED_BACKEND || ''

function langToken(legacy) {
  return USE_DRIVER ? DRIVER_LANG_FENCE : legacy
}
function pureLangPhrase() {
  return USE_DRIVER ? `pure ${DRIVER_LANG_FENCE} only` : LEGACY_PURE_LANG_PHRASE
}
function fenceToken() {
  return USE_DRIVER ? DRIVER_LANG_FENCE : LEGACY_FENCE_TOKEN
}
function cudallmCandidatePath(iter, sample) {
  const ext = USE_DRIVER ? DRIVER_SOURCE_EXT : LEGACY_SOURCE_EXT
  return `${EXP_DIR}/cudallm_iter_${iter}_sample_${sample}${ext}`
}
function cudallmResultPath(iter, sample) {
  return `${EXP_DIR}/cudallm_iter_${iter}_sample_${sample}${LEGACY_RESULT_EXT}`
}

// --- State ---
let taskSpec = ''
let referenceCode = ''
let tests = []
let featureCatalog = []
let featureScores = {}
let candidates = []
let bestCandidate = null

function initFeatureScore(feature) {
  if (!featureScores[feature.id]) {
    featureScores[feature.id] = {
      id: feature.id,
      name: feature.name,
      attempts: 0,
      compiled: 0,
      correct: 0,
      reward: 0,
      best_speedup: 0,
      failures: [],
    }
  }
}

function candidateScore(candidate) {
  if (!candidate?.eval?.compiled) return [0, 0, 0]
  if (!candidate.eval.correct) return [1, 0, 0]
  return [1, 1, candidate.eval.speedup || 0]
}

function isBetterCandidate(candidate, incumbent) {
  if (!incumbent) return !!candidate?.eval?.correct
  const a = candidateScore(candidate)
  const b = candidateScore(incumbent)
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
    `Return {present, backend_id, source_ext, aux_ext, lang_fence, impl_requirements, methods, feature_catalog, ` +
    `hw_vendor, profiler_name, profiler_format}. ` +
    `Set profiler_name/profiler_format from manifest.profiler when present.`,
    { label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH })
  if (!DRIVER || DRIVER.present === false) {
    throw new Error(`No backend driver present at ${BACKEND_DIR}. Provide a valid backend_dir or omit it for the legacy path.`)
  }
  if (RESOLVED_BACKEND && DRIVER.backend_id && normalizeSuitabilityValue(DRIVER.backend_id) !== RESOLVED_BACKEND) {
    throw new Error(`backend_dir manifest backend_id="${DRIVER.backend_id}" conflicts with args.backend/language="${RESOLVED_BACKEND}".`)
  }
  DRIVER_LANG_FENCE = DRIVER.lang_fence || DRIVER_LANG_FENCE
  DRIVER_IMPL_REQUIREMENTS = DRIVER.impl_requirements || ''
  DRIVER_SOURCE_EXT = DRIVER.source_ext || DRIVER_SOURCE_EXT
  DRIVER_FEATURE_CATALOG = DRIVER.feature_catalog || ''
  DRIVER_BACKEND_ID = DRIVER.backend_id || DRIVER_BACKEND_ID
  DRIVER.profiler_name = DRIVER.profiler_name || (DRIVER.profiler && DRIVER.profiler.name) || null
  DRIVER.profiler_format = DRIVER.profiler_format || (DRIVER.profiler && DRIVER.profiler.format) || 'ncu-csv'
  log(`Driver loaded: ${DRIVER_BACKEND_ID} (fence=${DRIVER_LANG_FENCE}, profiler=${DRIVER.profiler_name || 'none'})`)
}

const setup = await agent(`You are a ${langToken(LEGACY_SETUP_LANG_TOKEN)} kernel generation expert. Read and structure this CUDA-LLM task.

# Inputs
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${TASK_SPEC_PATH}
- reference_code_path: ${REFERENCE_CODE_PATH}
- target_gpu: ${TARGET_GPU}
- benchmark_command: ${EVAL_CMD || '(missing; evaluator evidence required before accepting speedup)'}
- tolerances: rtol=${RTOL}, atol=${ATOL}

# Tasks
1. Read problem_path and reference implementation when provided.
2. Use problem_definition as the authoritative task specification when provided.
3. Identify operation type, tensor shapes/dtypes, layout assumptions, and expected output.
4. Identify hard constraints: ${pureLangPhrase()}, no PyTorch fallback in generated kernel, preserve numerical tolerance.
5. State the evaluator JSON contract and how {kernel_path}/{result_path} are substituted.
6. List baseline performance if available.

Return structured task information.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"task_setup","note":"<operation type + key constraints/contract, one line>"}`, {
  label: 'setup-task',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      problem_definition: { type: 'string' },
      reference_code: { type: 'string' },
      operation_type: { type: 'string' },
      input_contract: { type: 'string' },
      output_contract: { type: 'string' },
      constraints: { type: 'array', items: { type: 'string' } },
      baseline_latency_ms: { type: 'number' },
    },
    required: ['problem_definition', 'operation_type', 'constraints'],
  },
})

taskSpec = PROBLEM_DEFINITION || setup.problem_definition || ''
referenceCode = setup.reference_code || ''

// =============================================================================
// Phase 2: FeatureCatalog
// =============================================================================
phase('FeatureCatalog')

const LEGACY_FEATURE_CATALOG = `# Required feature families
- tiling and block/grid decomposition
- shared memory staging
- vectorized/global memory access
- warp-level primitives
- loop unrolling and instruction scheduling
- occupancy/register pressure tuning
- fast math or CUDA intrinsics, only when tolerance allows
- boundary handling / tail masking`

const catalog = await agent(`Build a ${langToken(LEGACY_CATALOG_LANG_TOKEN)} optimization feature catalog for Feature Search and Reinforcement.

# Task
${taskSpec.substring(0, 5000)}

# Reference
\`\`\`
${referenceCode.substring(0, 5000)}
\`\`\`

# Target GPU
${TARGET_GPU}

${USE_DRIVER ? (DRIVER_FEATURE_CATALOG || LEGACY_TRITON_FEATURE_FALLBACK) : LEGACY_FEATURE_CATALOG}

# Tasks
1. Produce feature entries with id, name, family, description, prerequisites, incompatibilities, and risk.
2. Mark features that are unsafe under the given tolerance.
3. Include a conservative baseline feature set.
4. Initialize all feature scores with neutral priors.

Return feature catalog.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"FeatureCatalog","ts":"<ts>","status":"done","technique":"feature_search_space","note":"<count + the main feature families in the catalog, one line>"}`, {
  label: 'feature-catalog',
  phase: 'FeatureCatalog',
  schema: {
    type: 'object',
    properties: {
      features: { type: 'array', items: { type: 'object' } },
      baseline_feature_ids: { type: 'array', items: { type: 'string' } },
      unsafe_feature_ids: { type: 'array', items: { type: 'string' } },
      notes: { type: 'array', items: { type: 'string' } },
    },
    required: ['features', 'baseline_feature_ids'],
  },
})

featureCatalog = catalog.features || []
for (const feature of featureCatalog) initFeatureScore(feature)

// =============================================================================
// Phase 3: GenerateTests
// =============================================================================
phase('GenerateTests')

const testPlan = await agent(`Generate diverse correctness tests for this CUDA-LLM task.

# Operation
${setup.operation_type}

# Input contract
${setup.input_contract || ''}

# Output contract
${setup.output_contract || ''}

# Tolerances
rtol=${RTOL}, atol=${ATOL}

# Requirements
1. Cover small, medium, and large shapes.
2. Cover boundary/tail cases that stress masking and vectorized loads.
3. Cover dtype/layout variations if the task allows them.
4. Include random and adversarial value distributions.
5. These tests define what the user-provided benchmark_command should verify; model self-judgment is not enough.

Return test cases.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"GenerateTests","ts":"<ts>","status":"done","technique":"correctness_test_suite","note":"<count of tests + shapes/boundary cases covered, one line>"}`, {
  label: 'generate-tests',
  phase: 'GenerateTests',
  schema: {
    type: 'object',
    properties: {
      test_cases: { type: 'array', items: { type: 'object' } },
      tolerance_policy: { type: 'string' },
      harness_notes: { type: 'string' },
    },
    required: ['test_cases', 'tolerance_policy'],
  },
})

tests = testPlan.test_cases || []

// =============================================================================
// Main FSR Loop
// =============================================================================
for (let iteration = 0; iteration < ITERATIONS; iteration++) {
  for (let sample = 0; sample < SAMPLES_PER_FEATURE_SET; sample++) {
    log(`\n=== CUDA-LLM FSR iteration ${iteration + 1}/${ITERATIONS}, sample ${sample + 1}/${SAMPLES_PER_FEATURE_SET} ===`)

    phase('SelectFeatures')

    const selection = await agent(`Select a ${langToken(LEGACY_SELECT_LANG_TOKEN)} feature combination for the next candidate.

# Feature catalog
\`\`\`json
${JSON.stringify(featureCatalog, null, 2).substring(0, 10000)}
\`\`\`

# Feature scores
\`\`\`json
${JSON.stringify(featureScores, null, 2).substring(0, 10000)}
\`\`\`

# Recent candidates
\`\`\`json
${JSON.stringify(candidates.slice(-8), null, 2).substring(0, 10000)}
\`\`\`

# Selection rules
1. Select at most ${FEATURE_BUDGET} features.
2. Combine compatible features only.
3. Use exploration early: try under-tested but plausible features.
4. Use exploitation when feature evidence shows compile/correctness/speedup reward.
5. Avoid unsafe features unless explicitly justified by tolerance and tests.

Return selected feature ids and rationale.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"SelectFeatures","ts":"<ts>","status":"done","candidate_id":"iter${iteration}-s${sample}","technique":"<the selected feature combination as a +-joined list>","note":"<exploration vs exploitation + selection rationale, one line>"}`, {
      label: `select-features-${iteration}-${sample}`,
      phase: 'SelectFeatures',
      schema: {
        type: 'object',
        properties: {
          selected_feature_ids: { type: 'array', items: { type: 'string' } },
          rationale: { type: 'string' },
          expected_risks: { type: 'array', items: { type: 'string' } },
        },
        required: ['selected_feature_ids', 'rationale'],
      },
    })

    phase('GenerateKernel')

    const generation = await agent(`Generate a ${langToken(LEGACY_GENERATE_LANG_TOKEN)} kernel using the selected CUDA-LLM FSR features.

# Task specification
${taskSpec.substring(0, 8000)}

# Reference implementation
\`\`\`
${referenceCode.substring(0, 8000)}
\`\`\`

# Selected features
\`\`\`json
${JSON.stringify(selection, null, 2)}
\`\`\`

# Hard constraints
1. Return complete ${USE_DRIVER ? `${DRIVER_LANG_FENCE} source` : 'CUDA/C++ source'}, not a patch.
2. Do not call PyTorch or reference implementation from generated kernel.
3. Preserve input/output contract and tolerances.
4. Implement selected features concretely; if a feature is skipped, explain why.
5. Keep code benchmarkable by benchmark_command.${USE_DRIVER && DRIVER_IMPL_REQUIREMENTS ? `\n6. ${DRIVER_IMPL_REQUIREMENTS}` : ''}

Return candidate code and implemented features.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"GenerateKernel","ts":"<ts>","status":"done","candidate_id":"iter${iteration}-s${sample}","technique":"<the implemented feature combination as a +-joined list>","note":"<what was implemented vs skipped and why, one line>"}`, {
      label: `generate-kernel-${iteration}-${sample}`,
      phase: 'GenerateKernel',
      schema: {
        type: 'object',
        properties: {
          candidate_code: { type: 'string' },
          implemented_feature_ids: { type: 'array', items: { type: 'string' } },
          skipped_feature_ids: { type: 'array', items: { type: 'string' } },
          implementation_notes: { type: 'string' },
        },
        required: ['candidate_code', 'implemented_feature_ids'],
      },
    })

    phase('Evaluate')

    const evaluation = await agent(`Evaluate this ${langToken(LEGACY_EVAL_LANG_TOKEN)} candidate with compile, correctness, and latency evidence.

# Candidate code
\`\`\`${fenceToken()}
${(generation.candidate_code || '').substring(0, 16000)}
\`\`\`

# Eval command
${EVAL_CMD || '(no benchmark_command provided)'}

# Paths
- kernel_path: ${cudallmCandidatePath(iteration, sample)}
- result_path: ${cudallmResultPath(iteration, sample)}

# Tests
\`\`\`json
${JSON.stringify(tests, null, 2).substring(0, 8000)}
\`\`\`

# Required behavior
1. If benchmark_command is available, materialize the kernel and run it with {kernel_path}/{result_path}.
2. Parse evaluator JSON.
3. If benchmark_command is unavailable, set compiled=false, correct=false, speedup=0 and explain missing evidence.
4. Reward must be based on compile success, functional correctness over diverse tests, and measured latency.
5. Report suspected reward-hacking signs: hardcoded shapes, skipped computation, PyTorch fallback, or ignored inputs.

Return evaluator result.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" only if compiled AND correct, else "error"; speedup is the measured speedup number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"iter${iteration}-s${sample}","speedup":<number or null>,"technique":"<the feature combination under test as a +-joined list>","note":"<compiled? correct? passed/total tests; or the failure reason>"}`, {
      label: `evaluate-${iteration}-${sample}`,
      phase: 'Evaluate',
      schema: {
        type: 'object',
        properties: {
          compiled: { type: 'boolean' },
          correct: { type: 'boolean' },
          speedup: { type: 'number' },
          latency_ms: { type: 'number' },
          baseline_latency_ms: { type: 'number' },
          passed_tests: { type: 'number' },
          total_tests: { type: 'number' },
          error_message: { type: 'string' },
          reward_hacking_flags: { type: 'array', items: { type: 'string' } },
        },
        required: ['compiled', 'correct', 'speedup'],
      },
    })

    const candidate = {
      id: `iter_${iteration}_sample_${sample}`,
      selected_feature_ids: selection.selected_feature_ids || [],
      implemented_feature_ids: generation.implemented_feature_ids || [],
      code: generation.candidate_code || '',
      eval: evaluation,
    }

    if (USE_DRIVER) {
      const kPath = cudallmCandidatePath(iteration, sample)
      const rPath = cudallmResultPath(iteration, sample)
      const buildOut = `${EXP_DIR}/cudallm_iter_${iteration}_sample_${sample}.artifact`
      await agent(
        `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
        `Return its stdout JSON verbatim.`,
        { label: `driver-build-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
      const runOut = await agent(
        `${driverSh('run.sh', `--artifact ${buildOut} --kernel ${kPath} --result ${rPath}`)}\n` +
        `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
        { label: `driver-run-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
      const profilePointer = await agent(
        `${driverSh('profile.sh', buildProfileShArgs(buildOut, iteration, sample))}\n` +
        profileStepFooter() +
        `Return stdout JSON verbatim {ok, profiler, native_profile, format, error}.`,
        { label: `driver-profile-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
      const evidenceOut = await agent(
        buildToEvidencePrompt(profilePointer),
        { label: `driver-to-evidence-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
      const diagOut = await agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
        `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
        { label: `driver-diagnose-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
      const antiCheatOut = await agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${rPath}\`.\n` +
        `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
        { label: `driver-anti-cheat-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
      candidate.driver_envelope = {
        anti_cheat: antiCheatOut || {},
        metrics: (evidenceOut && evidenceOut.metrics) || {},
        vendor: (DRIVER && DRIVER.hw_vendor) || '',
        coverage: (evidenceOut && evidenceOut.coverage) || [],
        bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
        latency_ms: Number((runOut && runOut.latency_ms) || 0),
        backend_id: DRIVER_BACKEND_ID,
        profiler: (profilePointer && profilePointer.profiler) || null,
        profiler_format: (profilePointer && profilePointer.format) || null,
        profile_ok: profilePointer ? profilePointer.ok !== false : false,
      }
    }

    candidates.push(candidate)

    if (isBetterCandidate(candidate, bestCandidate)) {
      bestCandidate = candidate
    }

    phase('Reinforce')

    const reinforce = await agent(`Update ${langToken(LEGACY_REINFORCE_LANG_TOKEN)} feature scores from this measured candidate.

# Candidate
\`\`\`json
${JSON.stringify({
  id: candidate.id,
  selected_feature_ids: candidate.selected_feature_ids,
  implemented_feature_ids: candidate.implemented_feature_ids,
  eval: candidate.eval,
}, null, 2)}
\`\`\`

# Current feature scores
\`\`\`json
${JSON.stringify(featureScores, null, 2).substring(0, 10000)}
\`\`\`

# Reward rules
1. compiled=false gives strong penalty to implemented features.
2. compiled=true but correct=false gives weak compile credit and correctness penalty.
3. correct=true gives correctness credit plus speedup reward.
4. reward_hacking_flags suppress reward even if speedup appears high.
5. Features not implemented should not receive credit.

Return updated score records for affected features.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Reinforce","ts":"<ts>","status":"done","candidate_id":"iter${iteration}-s${sample}","technique":"<the feature combination whose scores you updated as a +-joined list>","note":"<which features gained/lost reward and why, one line>"}`, {
      label: `reinforce-${iteration}-${sample}`,
      phase: 'Reinforce',
      schema: {
        type: 'object',
        properties: {
          updated_scores: { type: 'array', items: { type: 'object' } },
          reinforcement_notes: { type: 'array', items: { type: 'string' } },
        },
        required: ['updated_scores'],
      },
    })

    for (const score of reinforce.updated_scores || []) {
      if (score?.id) featureScores[score.id] = score
    }
  }
}

// =============================================================================
// Phase 8: Report
// =============================================================================
phase('Report')

const finalReport = await agent(`Write a concise CUDA-LLM FSR optimization report.

# Task
${taskSpec.substring(0, 4000)}

# Adaptation scope
${ADAPTATION_SCOPE}

# Best candidate summary
\`\`\`json
${JSON.stringify(bestCandidate ? {
  id: bestCandidate.id,
  selected_feature_ids: bestCandidate.selected_feature_ids,
  implemented_feature_ids: bestCandidate.implemented_feature_ids,
  eval: bestCandidate.eval,
} : null, null, 2)}
\`\`\`

# Feature scores
\`\`\`json
${JSON.stringify(featureScores, null, 2).substring(0, 12000)}
\`\`\`

# Candidate history
\`\`\`json
${JSON.stringify(candidates.map(c => ({
  id: c.id,
  selected_feature_ids: c.selected_feature_ids,
  implemented_feature_ids: c.implemented_feature_ids,
  eval: c.eval,
})), null, 2).substring(0, 14000)}
\`\`\`

Cover:
1. Which ${langToken(LEGACY_REINFORCE_LANG_TOKEN)} features were reinforced by measured evidence.
2. Which feature combinations failed and why.
3. Whether the best kernel is trustworthy under diverse tests.
4. Which feature sets should be tried next.
5. Any reward-hacking risks that remain.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","candidate_id":"${bestCandidate ? bestCandidate.id : 'none'}","speedup":${bestCandidate && bestCandidate.eval ? (bestCandidate.eval.speedup || 0) : 0},"technique":"<best feature combination as a +-joined list>","note":"<best result + most-reinforced features, one line>"}`, {
  label: 'final-report',
  phase: 'Report',
})

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: TASK_SPEC_PATH,
  generated_kernel_path: bestCandidate?.path || '',
  initial_candidates: candidates,
  initial_generation_result: {
    verified: candidates.some(c => c.eval?.correct),
    selected_candidate_id: bestCandidate?.id || '',
  },
  reference_code_path: REFERENCE_CODE_PATH,
  target_gpu: TARGET_GPU,
  iterations: ITERATIONS,
  feature_budget: FEATURE_BUDGET,
  samples_per_feature_set: SAMPLES_PER_FEATURE_SET,
  adaptation_scope: ADAPTATION_SCOPE,
  best_speedup: bestCandidate?.eval?.speedup || 0,
  best_latency_ms: bestCandidate?.eval?.latency_ms || null,
  best_kernel_code: bestCandidate?.code || '',
  best_candidate_id: bestCandidate?.id || '',
  feature_scores: featureScores,
  candidates: candidates.map(c => ({
    id: c.id,
    selected_feature_ids: c.selected_feature_ids,
    implemented_feature_ids: c.implemented_feature_ids,
    eval: c.eval,
  })),
  report: finalReport,
}
