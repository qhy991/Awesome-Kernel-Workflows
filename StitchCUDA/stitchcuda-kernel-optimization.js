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
// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.
const EXPDIR = args.exp_dir || '.'

const WORKFLOW_SUITABILITY = {
  supported_languages: ['cuda'],
  supported_problem_types: ['cuda-kernel-generation', 'cuda-kernel-optimization'],
  problem_types: ['Planner/Coder/Verifier CUDA synthesis', 'CUDA generation and iterative replanning from problem definition'],
  reason: 'StitchCUDA is specifically a CUDA synthesis workflow and expects CUDA compile/test/benchmark contracts.',
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

// --- Backend driver wiring (P5c Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const WORKSPACE = args.workspace || '/tmp/stitchcuda'
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

async function main() {
  // ============================================================================
  // Phase 1: Setup
  // ============================================================================
  phase('Setup');

  if (USE_DRIVER) {
    DRIVER = await agent(
      `Load the backend driver at ${BACKEND_DIR} and return its manifest plus idioms verbatim.\n` +
      `1. Run exactly: \`cat ${driverPath('manifest.json')}\` and parse JSON.\n` +
      `2. Run exactly: \`cat ${driverPath('idioms.json')}\` and parse JSON.\n` +
      `Return {present, backend_id, source_ext, aux_ext, lang_fence, impl_requirements, methods}.`,
      { model: MODEL.mechanical, label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH }
    );
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

  const setupResult = await agent(
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
  );

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

      const replanResult = await agent(
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
      );

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

    const planResult = await agent(
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
    );

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

    const codeResult = await agent(
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
    );

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
    if (USE_DRIVER) {
      const kPath = attemptKernelPath(attempt);
      const buildOut = `${WORKSPACE}/attempt_${attempt}/artifact`;
      const profOut = `${WORKSPACE}/attempt_${attempt}/profile.native`;
      await agent(
        `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
        `Return its stdout JSON verbatim.`,
        { model: MODEL.mechanical, label: `driver-build-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH });
      const runOut = await agent(
        `${driverSh('run.sh', `--artifact ${buildOut} --kernel ${kPath}`)}\n` +
        `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
        { model: MODEL.profile, label: `driver-run-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH });
      // profiling-strategist gate: classify this attempt's kernel (op_class+size),
      // resolve the method via the shared strategist (cached), then branch.
      const _pd = await agent(
        `Read ${kPath}; classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then run exactly: ` +
        `\`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${driverPath('manifest.json')} --task <op_class> --size <size> --cache ${EXPDIR}/prof_cache.json --trajectory ${EXPDIR}/genome.jsonl\`.\n` +
        `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
        { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Verify', schema: JSON_PASSTHROUGH });
      if (_pd && _pd.method) PROFILING_DECISION = _pd
      let evidenceOut = null;
      if (PROFILING_DECISION.method === 'native_profiler') {
        await agent(
          `${driverSh('profile.sh', `--artifact ${buildOut} --kernel ${kPath} --out ${profOut}`)}\n` +
          `Return {ok, native_path}.`,
          { model: MODEL.profile, label: `driver-profile-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH });
        evidenceOut = await agent(
          `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
          `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
          { model: MODEL.mechanical, label: `driver-to-evidence-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH });
      } else {
        // Else path: rely on the existing run.sh/latency result; when method==='perf_heuristic',
        // normalize throughput via the strategist normalizer (substrate profiling/<normalizer>),
        // tagging bottlenecks evidence='profile_heuristic', confidence from the decision.
        const _normalizer = PROFILING_DECISION.normalizer || 'perf_to_evidence.py'
        log(`Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); skipping native profiler.`);
        if (PROFILING_DECISION.method === 'perf_heuristic') {
          evidenceOut = await agent(
            `Normalize the run.sh throughput into canonical metrics. Run exactly: ` +
            `\`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${_normalizer} --baseline ${WORKSPACE}/attempt_${attempt}/result.json --peak-gflops <device_peak_gflops> --peak-gbs <device_peak_gbs>\`.\n` +
            `First write the run.sh result to ${WORKSPACE}/attempt_${attempt}/result.json from: ${JSON.stringify(runOut || {})}\n` +
            `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}. ` +
            `Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.`,
            { model: MODEL.mechanical, label: `driver-perf-heuristic-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH });
        }
      }
      const diagOut = await agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
        `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
        { model: MODEL.mechanical, label: `driver-diagnose-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH });
      const antiCheatOut = await agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath}\`.\n` +
        `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
        { model: MODEL.mechanical, label: `driver-anti-cheat-${attempt}`, phase: 'Verify', schema: JSON_PASSTHROUGH });
      driverEnvelope = {
        anti_cheat: antiCheatOut || {},
        metrics: (evidenceOut && evidenceOut.metrics) || {},
        vendor: (DRIVER && DRIVER.hw_vendor) || '',
        coverage: (evidenceOut && evidenceOut.coverage) || [],
        bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
        latency_ms: Number((runOut && runOut.latency_ms) || 0),
        backend_id: DRIVER_BACKEND_ID,
      };
    }

    // ==========================================================================
    // Phase 4: Verify (Verifier Agent)
    // ==========================================================================
    phase('Verify');

    log('Verifier: Checking correctness and performance...');

    const verifyResult = await agent(
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
    );

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
    return { success: false, reason: 'no_successful_kernel' };
  }

  const report = await agent(
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
  );

  // ============================================================================
  // Return final results
  // ============================================================================

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
