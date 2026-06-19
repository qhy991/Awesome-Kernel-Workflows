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
// --- END inlined arg_guard ---
// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.

const WORKFLOW_SUITABILITY = {
  method_supported_backends: ['cuda', 'triton'],
  default_backend: 'cuda',
  requires_capability: { bottleneck_classes: [], metrics: ['dram_pct', 'sm_pct'] },
  supported_problem_types: ['cuda-kernel-optimization', 'cuda-kernel-generation'],
  problem_types: ['existing CUDA kernel optimization', 'CUDA generation from problem_definition with benchmark contract'],
  reason: 'AccelOpt is intrinsic to NCU-class profiling (dram_pct/sm_pct); runs on any NVIDIA-vendor backend (cuda, triton) with a present driver.',
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

function resolveBackend() {
  const b = args.backend ? normalizeSuitabilityValue(args.backend) : null
  const l = args.language ? normalizeSuitabilityValue(args.language) : null
  if (b && l && b !== l) {
    throw new Error(`Conflicting args: backend="${b}" vs language="${l}". Pass only one.`)
  }
  if (b) return b
  if (l) return l
  const ms = WORKFLOW_SUITABILITY.method_supported_backends
  if (Array.isArray(ms) && ms.length === 1) return normalizeSuitabilityValue(ms[0])
  return WORKFLOW_SUITABILITY.default_backend
}

function assertWorkflowSuitability() {
  const backend = resolveBackend()

  const ms = WORKFLOW_SUITABILITY.method_supported_backends
  if (ms !== 'any' && !ms.map(normalizeSuitabilityValue).includes(backend)) {
    throw new Error(
      `${WORKFLOW_NAME}'s method does not support backend="${backend}". ` +
      `Method-supported: ${ms.join(', ')}. Reason: ${WORKFLOW_SUITABILITY.reason}`
    )
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

  return backend
}

const BACKEND = assertWorkflowSuitability()

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
const OP_DESC = args.op_description || 'CUDA kernel'
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

// Fallback: non-NCU profiling commands
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || ''

if (!KERNEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

const LANGUAGE = args.language || 'cuda'
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

// State
let experienceMemory = []       // Full pool of learned patterns (grows unbounded)
let lastIterNewPatterns = []    // Patterns discovered in the most recent Learn phase
let bestLatency = null
let bestKernelCode = null
let baselineLatency = null
let baselineNcuProfile = ''
let candidateBeam = []          // [{code, latency, speedup, ncuSummary, planTitle}]
let bottleneckClass = 'unknown'

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
  return `Read the CUDA kernel file at: ${KERNEL_PATH}

Analyze it and return a JSON object with:
- kernel_code: the full source code
- op_type: operation type (e.g., "quantized_gemm", "attention", "rmsnorm", "softmax")
- key_functions: list of key function names (especially __global__ kernels)
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

  const generated = await agent(USE_DRIVER ? driverGenerateSeedPrompt() : legacyGenerateSeedPrompt(), {
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
function buildBeamSection(candidateBeam, fence) {
  if (candidateBeam.length <= 1) return ''
  return `\n\n# Candidate Beam (top-${candidateBeam.length} kernels from previous iterations)\n${candidateBeam.map((c, i) => `## Candidate ${i + 1}: "${c.planTitle}" — ${c.speedup.toFixed(2)}x, ${c.latency.toFixed(3)}ms\nNCU: ${c.ncuSummary || 'N/A'}\n\`\`\`${fence}\n${c.code.substring(0, 1500)}\n\`\`\``).join('\n\n')}`
}

// =============================================================================
// Phase 1: Setup — Read kernel, build harness, NCU profile baseline
// =============================================================================
phase('Setup')

if (USE_DRIVER) {
  const driver = await agent(
    `Load the backend driver for backend="${BACKEND}".\n` +
    `1. Run exactly: \`cat ${DRIVER_DIR}/manifest${DRIVER_EXT}\` and parse JSON.\n` +
    `2. Run exactly: \`cat ${DRIVER_DIR}/idioms${DRIVER_EXT}\` and parse JSON.\n` +
    `If either is missing, return {present:false, reason:"no driver for backend ${BACKEND}"}.\n` +
    `Also compare manifest.capabilities against the required capability floor ` +
    `${JSON.stringify(WORKFLOW_SUITABILITY.requires_capability)};\n` +
    `if a required metric/class is missing return {present:true, capability_ok:false, missing:[...]}.\n` +
    `Return {present, capability_ok, missing, backend_id, source_ext, lang_fence, hw_vendor,\n` +
    `  profiler_name|null, profiler_format, capability_metrics, supported_classes, problem_types,\n` +
    `  requires_tools, impl_requirements, read_metric_guide,\n` +
    `  plan_angles:[...], unsupported_methods:[...],\n` +
    `  idioms:{<method>:{idiom,prompt_guidance}}}.`,
    { model: MODEL.mechanical, label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH })

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

const setupResult = await agent(USE_DRIVER ? driverSetupReadPrompt() : legacySetupReadPrompt(), {
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
})

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

let ncuSetup
if (USE_DRIVER) {
  const profileResult = await agent(
    IDIOMS.profiler_name
      ? `Profile the baseline kernel via the backend driver and normalize to canonical metrics.\n` +
        `Kernel: ${KERNEL_PATH}. Experiment dir: ${EXP_DIR}/baseline.\n` +
        `1. ` + driverSh('build.sh', `--source ${KERNEL_PATH} --out ${EXP_DIR}/baseline/artifact ${HARNESS_BUILD_CMD ? `--build-cmd "${HARNESS_BUILD_CMD}"` : ''}`) + `\n` +
        `2. ` + driverSh('profile.sh', `--artifact ${EXP_DIR}/baseline/artifact --problem ${PROBLEM_PATH || PROBLEM_DEFINITION || KERNEL_PATH} --out ${EXP_DIR}/baseline/prof.native`) + `\n` +
        `3. ` + driverPy('to_evidence.py', `--native ${EXP_DIR}/baseline/prof.native --format ${IDIOMS.profiler_format}`) + `\n` +
        `Return its stdout JSON verbatim: {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy,...}, coverage:[...], source_backend}. ` +
        `If the profiler exits 4 (unavailable), return {ok:true, metrics:{latency_ms:null,dram_pct:null,sm_pct:null,occupancy:null}, coverage:[], profiler_available:false}.`
      : `Backend "${BACKEND}" declares no profiler. Do not invent one. ` +
        `Build via ` + driverSh('build.sh', `--source ${KERNEL_PATH} --out ${EXP_DIR}/baseline/artifact`) + ` then ` +
        driverSh('run.sh', `--artifact ${EXP_DIR}/baseline/artifact --problem ${PROBLEM_PATH || KERNEL_PATH} --out ${EXP_DIR}/baseline/result.json`) + ` for latency only. ` +
        `Return {ok:true, metrics:{latency_ms:<from run.sh>,dram_pct:null,sm_pct:null,occupancy:null}, coverage:["latency_ms"], profiler_available:false}.`,
    { model: MODEL.profile, label: 'ncu-baseline', phase: 'Setup', schema: JSON_PASSTHROUGH })

  const metrics = profileResult.metrics || {}
  const diag = await agent(
    `Write these metrics to ${EXP_DIR}/baseline/metrics.json:\n${JSON.stringify(metrics)}\n` +
    `${substrateInstruction('diagnose.py', `--metrics ${EXP_DIR}/baseline/metrics.json`)} Return stdout JSON verbatim {bottleneck_class, evidence}.`,
    { model: MODEL.mechanical, label: 'diagnose-baseline', phase: 'Setup', schema: JSON_PASSTHROUGH })
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
} else {
  ncuSetup = await agent(legacyNcuBaselinePrompt(baselineKernel), { model: MODEL.profile,
    label: 'ncu-baseline',
    phase: 'Setup',
    schema: LEGACY_NCU_BASELINE_SCHEMA,
  })
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
if (USE_DRIVER) {
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

for (let iter = 0; iter < ITERATIONS; iter++) {
  log(`\n=== Iteration ${iter + 1}/${ITERATIONS} | Best: ${bestLatency.toFixed(3)}ms (${(baselineLatency / bestLatency).toFixed(2)}x) | Beam: ${candidateBeam.length} | Experience: ${experienceMemory.length} patterns ===`)

  // ===========================================================================
  // Phase 2: Plan — Generate optimization plans GUIDED BY NCU DATA + BEAM
  // ===========================================================================
  phase('Plan')

  // Experience sampling (AccelOpt: construct_experience.py logic)
  const experienceSection = buildExperienceSection(experienceMemory, lastIterNewPatterns, MAX_EXPERIENCE_IN_PROMPT)

  // Candidate beam context for planner
  const beamSection = buildBeamSection(candidateBeam, IDIOMS.lang_fence)

  const planAngles = IDIOMS.plan_angles

  const planPromptBase = USE_DRIVER
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

  const planSchema = USE_DRIVER
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
      agent(`${planPromptBase}\n\n# YOUR FOCUS AREA: ${planAngles[i % planAngles.length]}\nYou are planner #${i + 1}/${BREADTH}. Focus on: ${planAngles[i % planAngles.length]}.

# Recent genome trajectory (read BEFORE planning)
Run \`tail -20 ${EXP_DIR}/genome.jsonl 2>/dev/null\` to see prior attempts this session (every Plan/Execute/Evaluate/Learn step has self-reported here). Use it to: (a) avoid retrying any technique already attempted with a regression or null speedup, (b) spot multi-round patterns the per-iteration experience summary may have lost. If the file is empty or missing, ignore this and proceed with the inputs above.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is iteration ${iter}, planner ${i}):
{"workflow":"${WORKFLOW_NAME}","phase":"Plan","ts":"<ts>","status":"done","candidate_id":"iter-${iter}-plan-${i}","technique":"<your plan title / optimization move>","speedup":null,"note":"<the profile metric you cited + expected impact, one line>"}`, {
        label: `plan-${iter}-${i}`,
        phase: 'Plan',
        schema: planSchema,
      })
    )
  )

  const validPlans = plans.filter(Boolean)
  const planEvidence = (p) => (p.profile_evidence ?? p.ncu_evidence)
  log(`Plans: ${validPlans.map(p => `${p.title} (evidence: ${(p.ncu_evidence || '').substring(0, 50)}...)`).join(' | ')}`)

  // ===========================================================================
  // Phase 3: Execute — Implement each plan
  // ===========================================================================
  phase('Execute')

  const implementations = await pipeline(
    validPlans,
    (plan) => parallel(
      Array.from({length: SAMPLES_PER_PLAN}, (_, sampleIdx) => () =>
        agent(USE_DRIVER
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
        })
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

  const evalSchema = USE_DRIVER
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
      agent(USE_DRIVER
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
      })
    )
  )

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
      speedup: r.speedup * (baselineLatency / bestLatency), // relative to original baseline
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
    if (bestResult && bestResult.evaluation.ncu_comparison) {
      baselineNcuProfile = USE_DRIVER
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
    const learnSchema = USE_DRIVER
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
        agent(USE_DRIVER
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
        })
      )
    )

    // Format experience entries aligned with AccelOpt's summarizer output format
    lastIterNewPatterns = []
    for (const s of summaries.filter(Boolean)) {
      const formatted = USE_DRIVER
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
  log(`Iteration ${iter + 1} done. ${(baselineLatency / bestLatency).toFixed(2)}x vs baseline. Beam size: ${candidateBeam.length}`)
}

// =============================================================================
// Final Report
// =============================================================================
const finalReport = await agent(USE_DRIVER
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
})

let evidenceEnvelope = null
if (USE_DRIVER) {
  const insightItems = experienceMemory.slice(0, TOPK_LEARN).map(e => ({
    kind: 'bottleneck',
    directive: 'explore',
    evidence: IDIOMS.profiler_name ? 'ncu' : 'profile_heuristic',
    confidence: 'inferred',
    claim: e,
  }))
  const built = await agent(
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
    { label: 'assemble-evidence', phase: 'Iterate', schema: JSON_PASSTHROUGH })
  evidenceEnvelope = built.valid ? (built.normalized || null) : null
  if (!built.valid) log(`WARN: Layer-A envelope failed evidence_schema validation`)
}

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  baseline_latency_ms: baselineLatency,
  best_latency_ms: bestLatency,
  overall_speedup: baselineLatency / bestLatency,
  iterations_completed: ITERATIONS,
  candidate_beam: candidateBeam.map(c => ({
    plan_title: c.planTitle,
    latency_ms: c.latency,
    speedup: c.speedup,
  })),
  experience_patterns_count: experienceMemory.length,
  experience_patterns: experienceMemory,
  best_kernel_code: bestKernelCode,
  ncu_baseline_profile: baselineNcuProfile,
  report: finalReport,
  ...(USE_DRIVER ? {
    backend: BACKEND,
    baseline_profile: baselineNcuProfile,
    bottleneck_class: bottleneckClass,
    evidence: evidenceEnvelope,
  } : {}),
}
