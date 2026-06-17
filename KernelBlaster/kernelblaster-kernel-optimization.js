export const meta = {
  name: 'kernelblaster-kernel-optimization',
  description: 'Memory-augmented in-context RL loop for CUDA kernel optimization with a persistent, state-keyed optimization knowledge base (KernelBlaster / MAIC-RL methodology)',
  whenToUse: 'When you want to optimize a CUDA kernel via profile-guided RL rollouts that ACCUMULATE experience across kernels and runs. KernelBlaster classifies each kernel into a hardware performance state (memory / compute / latency bound), retrieves the best-known optimization for that state from a persistent knowledge base keyed by bottleneck pattern, applies it, measures the real reward (Elapsed Cycles delta via NCU), and updates the database so future rollouts and future kernels reuse what worked. Pass optimization_db_path to carry the knowledge base across invocations.',
  phases: [
    { title: 'Setup', detail: 'Read kernel + driver harness, load/seed the persistent optimization database, NCU-profile the baseline' },
    { title: 'ProfileState', detail: 'Profile current kernel, parse NCU metrics, classify hardware performance state (memory/compute/latency bound)' },
    { title: 'Retrieve', detail: 'Match state against the knowledge base; rank candidate optimizations by confidence x measured payoff' },
    { title: 'Plan', detail: 'LLM selects/specializes a strategy for the matched state, citing the DB and NCU evidence' },
    { title: 'Execute', detail: 'Apply the strategy-guided optimization to produce a complete compilable kernel' },
    { title: 'Evaluate', detail: 'Compile + correctness-check against the driver, re-profile, measure Elapsed Cycles speedup' },
    { title: 'Reward', detail: 'Compute RL reward, append trajectory to replay buffer, update DB entry stats (confidence/usage/measured payoff)' },
    { title: 'Iterate', detail: 'Periodic policy-update cycle over the replay buffer, then continue rollout / persist database' },
  ],
}

// --- BEGIN genome-report (auto-inserted by scripts/patch-genome-report.js) ---
// Self-reported, work-plane (forgeable) stage trace for observability + the
// recombiner. NOT a trust anchor — see _meta/genome-trajectory-schema.md.
async function __genomeReport(phaseName, wfName) {
  try {
    const __dir = (typeof args !== 'undefined' && args && args.exp_dir) ? args.exp_dir : '.'
    await agent(
      'Append exactly one line to ' + __dir + '/genome.jsonl (create it if missing; use a shell append: printf %s\\n ... >> file). ' +
      'The line must be this JSON on ONE line: {"workflow":"' + wfName + '","phase":"' + phaseName + '","ts":"<UTC>","status":"entered"}. ' +
      'Produce <UTC> by running: date -u +%Y-%m-%dT%H:%M:%SZ . Do nothing else; modify no other file. Echo the exact line you appended.',
      { label: 'genome:' + phaseName, phase: phaseName }
    )
  } catch (__e) { /* observability must never break the workflow */ }
}
// --- END genome-report ---

const WORKFLOW_SUITABILITY = {
  supported_languages: ['cuda'],
  supported_problem_types: ['cuda-kernel-optimization'],
  problem_types: ['CUDA RL-style optimization using NCU elapsed cycles', 'memory-augmented CUDA kernel refinement'],
  reason: 'KernelBlaster is anchored on CUDA kernels, NCU elapsed-cycle feedback, and CUDA optimization memory.',
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
        `${meta.name} is not suitable for language="${args.language}". ` +
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
        `${meta.name} is not suitable for problem_type="${args.problem_type}". ` +
        `Supported problem types: ${WORKFLOW_SUITABILITY.supported_problem_types.join(', ')}. ` +
        `Typical use cases: ${WORKFLOW_SUITABILITY.problem_types.join('; ')}. ` +
        `Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }
  }
}

assertWorkflowSuitability()

// =============================================================================
// KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented
//                In-Context Reinforcement Learning (MAIC-RL)
// =============================================================================
//
// Source: KernelBlaster (arXiv:2602.14293), NVlabs/KernelBlaster (Apache-2.0)
//   Kris Shengjun Dong, Sahil Modi, Dima Nikiforov, Sana Damani, Edward Lin,
//   Siva Kumar Sastry Hari, Christos Kozyrakis.
//
// Implements the paper's core loop:
//   Profile -> Classify State -> Retrieve Optimization -> Apply -> Re-Profile
//   -> Reward -> Update Knowledge Base -> Repeat   (with cross-task memory)
//
// Faithful to the upstream system:
//   1. State taxonomy from data/kernelblaster/optimization_database.json:
//        memory_bandwidth_limited / compute_throughput_limited / latency_occupancy_limited
//   2. State-keyed optimization strategies with confidence_score / usage_count /
//      predicted vs actual_improvement (measured payoff), as in agents/database.py.
//   3. Elapsed Cycles (NCU) as the scalar performance metric (get_elapsed_cycles_ncu_log).
//   4. Reward = actual_improvement/100 + accuracy_bonus +- penalty   (calculate_reward).
//   5. Replay buffer of trajectories + periodic policy_update_cycle (rl_agents.py).
//   6. PERSISTENT cross-task knowledge base: optimization_db_path is read at Setup and
//      rewritten at the end, so experience compounds across kernels and runs.
//
// Usage:
//   Workflow({name: 'kernelblaster-kernel-optimization', args: {
//     kernel_path: '/path/to/init.cu',
//     driver_path: '/path/to/driver.cpp',          // KernelBench-CUDA build/run/validate harness
//     op_description: 'Square matrix multiplication',
//     target_gpu: 'L40S',
//     optimization_db_path: '/path/to/optimization_database.json',  // persistent cross-task memory
//     exp_dir: '/path/to/experiment/output',
//     ncu_binary: '<user-provided ncu binary path>',
//     kernel_name_regex: 'matmul_kernel',
//     build_cmd: '<user-provided build command>',
//     run_cmd: '<user-provided benchmark command>',
//     rl_iterations: 3,             // number of rollouts (RL iterations)
//     rollout_steps: 4,             // optimization steps per rollout
//     breadth: 2,                   // candidate plans per step
//     update_frequency: 3,          // run policy-update cycle every N trajectories
//     run_timestamp_iso: '2026-06-02T17:00:00Z',
//   }})
//
// =============================================================================

// --- Required args ---
let KERNEL_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'

// --- Optional args ---
const DRIVER_PATH = args.driver_path || ''
const OP_DESC = args.op_description || 'CUDA kernel'
const GPU_TYPE = args.target_gpu || 'L40S'
const OPT_DB_PATH = args.optimization_db_path || ''
const EXP_DIR = args.exp_dir || '/tmp/kernelblaster_exp'
const NCU_BINARY = args.ncu_binary || ''
const KERNEL_NAME_REGEX = args.kernel_name_regex || ''
const BUILD_CMD = args.build_cmd || ''
const RUN_CMD = args.run_cmd || ''
const RL_ITERATIONS = args.rl_iterations || 3
const ROLLOUT_STEPS = args.rollout_steps || 4
const BREADTH = args.breadth || 2
const UPDATE_FREQUENCY = args.update_frequency || 3
const RUN_TS = args.run_timestamp_iso || 'unknown'

// Fallback (no NCU): custom test + benchmark commands
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || ''

if (!KERNEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

const LANGUAGE = args.language || 'cuda'
const SEED_CANDIDATES = args.seed_candidates || 3
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null

// =============================================================================
// Persistent State-Keyed Optimization Knowledge Base (MAIC-RL memory)
//
// Mirrors data/kernelblaster/optimization_database.json: three known states,
// each mapped to a list of optimization techniques carrying the running RL
// statistics the paper updates (confidence_score, usage_count, predicted vs
// actual measured speedup). This seed is used when no optimization_db_path is
// provided or the file is empty; otherwise the loaded DB takes precedence.
// =============================================================================
function seedOptimizationDatabase() {
  return {
    schema_version: 'kernelblaster-1.0',
    known_states: {
      memory_bandwidth_limited: {
        state_name: 'memory_bandwidth_limited',
        primary_bottleneck: 'memory_bound',
        secondary_characteristics: [
          'High memory throughput utilization',
          'Bandwidth saturation',
          'Potential coalescing issues',
          'Cache inefficiencies',
        ],
        performance_signature: 'Memory bandwidth is the primary limiting factor with potential for access pattern optimization',
      },
      compute_throughput_limited: {
        state_name: 'compute_throughput_limited',
        primary_bottleneck: 'compute_bound',
        secondary_characteristics: [
          'High compute unit utilization',
          'Instruction throughput bottleneck',
          'Potential for specialized units',
          'Arithmetic intensity',
        ],
        performance_signature: 'Compute units are saturated, indicating opportunity for algorithmic or instruction-level optimization',
      },
      latency_occupancy_limited: {
        state_name: 'latency_occupancy_limited',
        primary_bottleneck: 'latency_bound',
        secondary_characteristics: [
          'Low occupancy',
          'Insufficient parallelism',
          'Resource underutilization',
          'Synchronization overhead',
        ],
        performance_signature: 'Neither memory nor compute are saturated, indicating latency hiding or occupancy issues',
      },
    },
    optimization_strategies: {
      memory_bandwidth_limited: makeEntries([
        ['vectorized_memory_access', 'Use float4/int4/half2 for wider memory transactions', 'memory'],
        ['memory_coalescing_optimization', 'Align access patterns so consecutive threads touch consecutive addresses', 'memory'],
        ['shared_memory_tiling', 'Cache frequently reused data on-chip to cut global traffic', 'memory'],
        ['data_layout_transformation', 'Convert AoS to SoA layouts for coalesced streaming', 'general'],
        ['read_only_constant_cache', 'Route read-only operands through __ldg / constant memory', 'memory'],
      ]),
      compute_throughput_limited: makeEntries([
        ['tensor_core_utilization', 'Map matmul-shaped work to tensor cores (half/bf16, mma)', 'compute'],
        ['instruction_mix_optimization', 'Rebalance ALU/SFU/LSU pressure, use intrinsics', 'compute'],
        ['warp_level_reduction', 'Replace shared-memory reductions with __shfl primitives', 'compute'],
        ['loop_unrolling', '#pragma unroll hot loops to expose ILP', 'compute'],
        ['algorithmic_changes', 'Replace the algorithm with a lower-complexity variant', 'general'],
      ]),
      latency_occupancy_limited: makeEntries([
        ['occupancy_tuning', 'Tune threads/block + shared mem to raise achieved occupancy', 'launch'],
        ['register_pressure_reduction', 'Cut live registers / add __launch_bounds__ to avoid spills', 'launch'],
        ['increase_parallelism', 'Launch more blocks / persistent kernel to fill the SMs', 'launch'],
        ['memory_latency_hiding', 'Async copies + software pipelining to hide long_scoreboard stalls', 'memory'],
        ['reduce_synchronization', 'Remove unnecessary __syncthreads / use warp-sync', 'compute'],
      ]),
    },
  }
}

function makeEntries(triples) {
  return {
    optimizations: triples.map(([technique, description, category]) => ({
      technique,
      description,
      category,
      confidence_score: 0.5,   // RL prior, updated from measured results
      usage_count: 0,
      predicted_speedup: 1.0,
      actual_speedup: null,    // measured (null until tried)
      actual_improvement: null,
      last_updated: null,
    })),
  }
}

// RL reward (verbatim formula from agents/opt_ncu_rl.py::calculate_reward):
//   base = actual_improvement/100
//   accuracy_bonus = +0.2 if 0.8<=acc<=1.2 else -0.1*|acc-1|   (acc = actual/predicted, capped 2.0)
//   penalty = -0.5 if not faster else 0
function calculateReward(predictedImprovement, actualImprovement, isFaster) {
  const baseReward = actualImprovement / 100.0
  let accuracyBonus = 0.0
  const safePredicted = (predictedImprovement === null || predictedImprovement === undefined || isNaN(predictedImprovement)) ? 0.0 : predictedImprovement
  if (safePredicted > 0.0) {
    const accuracy = Math.min(actualImprovement / safePredicted, 2.0)
    accuracyBonus = (accuracy >= 0.8 && accuracy <= 1.2) ? 0.2 : -0.1 * Math.abs(accuracy - 1.0)
  }
  const penalty = isFaster ? 0.0 : -0.5
  return baseReward + accuracyBonus + penalty
}

// Rank optimizations for a state (database.py::select_best_optimization score):
//   score = predicted_speedup * confidence_score, with a mild unused-exploration boost
function rankOptimizations(stateEntry) {
  if (!stateEntry || !stateEntry.optimizations) return []
  return [...stateEntry.optimizations]
    .map((opt) => {
      const measured = (opt.actual_speedup !== null && opt.actual_speedup !== undefined) ? opt.actual_speedup : opt.predicted_speedup
      const explorationBoost = opt.usage_count === 0 ? 0.15 : 0.0
      return { opt, score: measured * (opt.confidence_score || 0.5) + explorationBoost }
    })
    .sort((a, b) => b.score - a.score)
    .map((x) => x.opt)
}

// Update a DB entry with a measured result (database.py::update_optimization_result):
//   blends confidence toward outcome, records measured speedup, bumps usage_count.
function updateOptimizationResult(stateEntry, technique, actualImprovement, actualSpeedup) {
  if (!stateEntry || !stateEntry.optimizations) return
  const opt = stateEntry.optimizations.find((o) => o.technique === technique)
  if (!opt) return
  opt.usage_count = (opt.usage_count || 0) + 1
  opt.actual_improvement = actualImprovement
  opt.actual_speedup = actualSpeedup
  opt.last_updated = RUN_TS
  // Exponential-moving confidence: success (>0 improvement) raises it, failure lowers it.
  const target = actualImprovement > 0 ? Math.min(1.0, 0.5 + actualImprovement / 100.0) : Math.max(0.05, 0.5 + actualImprovement / 100.0)
  opt.confidence_score = 0.7 * (opt.confidence_score || 0.5) + 0.3 * target
  // Predicted speedup tracks measured outcome so future ranking improves.
  if (actualSpeedup && actualSpeedup > 0) {
    opt.predicted_speedup = 0.6 * (opt.predicted_speedup || 1.0) + 0.4 * actualSpeedup
  }
}

// =============================================================================
// In-workflow state
// =============================================================================
let optDb = null               // the persistent knowledge base (loaded or seeded)
let baselineCycles = null      // Elapsed Cycles of the original kernel (set once)
let bestCycles = null          // best Elapsed Cycles achieved
let bestKernelCode = null      // source of the current best kernel
let replayBuffer = []          // [{steps:[{state,action,cycles,predicted,actual,reward}], total_reward, initial_cycles, final_cycles}]
let totalTrajectories = 0
let dbUpdateLog = []           // human-readable record of DB mutations

function dbSummaryForPrompt(db) {
  const lines = []
  for (const [stateName, entry] of Object.entries(db.optimization_strategies)) {
    const ranked = rankOptimizations(entry).slice(0, 5)
    lines.push(`## State: ${stateName} (${db.known_states[stateName] ? db.known_states[stateName].primary_bottleneck : 'unknown'})`)
    for (const o of ranked) {
      const measured = (o.actual_speedup !== null && o.actual_speedup !== undefined) ? `${o.actual_speedup.toFixed(2)}x measured` : 'unmeasured'
      lines.push(`- ${o.technique} (conf ${o.confidence_score.toFixed(2)}, used ${o.usage_count}x, ${measured}): ${o.description}`)
    }
  }
  return lines.join('\n')
}

// =============================================================================
// Phase 1: Setup — read kernel + driver, load/seed DB, NCU-profile baseline
// =============================================================================
phase('Setup'); await __genomeReport('Setup', meta.name)

if (INPUT_MODE === 'generate_then_optimize') {
  const generated = await agent(`No kernel_path was provided. Generate and verify an initial CUDA kernel before starting KernelBlaster.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- op_description: ${OP_DESC}
- language: ${LANGUAGE}
- target_gpu: ${GPU_TYPE}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- build_cmd: ${BUILD_CMD || '(not provided)'}
- run_cmd: ${RUN_CMD || '(not provided)'}
- test_command: ${TEST_CMD || '(not provided)'}
- benchmark_command: ${BENCH_CMD || '(not provided)'}

# Contract
Generate ${SEED_CANDIDATES} complete candidates under ${EXP_DIR}/generated/. Run available commands using {kernel_path}/{result_path}. Return the best verified generated kernel path.`, {
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
  KERNEL_PATH = generatedKernelPath
}

const setupResult = await agent(`Read the CUDA kernel at: ${KERNEL_PATH}
${DRIVER_PATH ? `The build/run/validate harness (KernelBench-CUDA style driver) is at: ${DRIVER_PATH}` : ''}
${OPT_DB_PATH ? `A persistent optimization knowledge base may exist at: ${OPT_DB_PATH} — if it exists and is valid JSON, read it and return its contents in loaded_db (else return null).` : ''}

Analyze the kernel and return JSON with:
- kernel_code: full source of the kernel file
- op_type: operation type (e.g. "gemm", "conv2d", "softmax", "reduction")
- key_functions: list of __global__ kernel names
- current_approach: brief description of the implementation strategy
- launch_config: grid/block dims if visible
- driver_summary: how the driver builds/runs/validates the kernel (if a driver path was given)
- loaded_db: parsed JSON of the persistent optimization database if it exists and is valid, else null

Return ONLY the JSON object.`, {
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
      driver_summary: { type: 'string' },
      loaded_db: { type: ['object', 'null'] },
    },
    required: ['kernel_code', 'op_type', 'key_functions', 'current_approach'],
  },
})

const baselineKernel = setupResult.kernel_code
const opType = setupResult.op_type
bestKernelCode = baselineKernel

// Load persistent DB if present and well-formed; otherwise seed a fresh one.
if (setupResult.loaded_db && setupResult.loaded_db.optimization_strategies) {
  optDb = setupResult.loaded_db
  log(`Loaded persistent optimization DB from ${OPT_DB_PATH} (cross-task memory active)`)
} else {
  optDb = seedOptimizationDatabase()
  log(`Seeded fresh optimization DB (3 states, ${Object.values(optDb.optimization_strategies).reduce((n, e) => n + e.optimizations.length, 0)} strategies)`)
}

log(`Baseline: ${opType}, kernels: ${setupResult.key_functions.join(', ')}`)

// NCU baseline profile -> Elapsed Cycles + Speed-of-Light metrics
const ncuBaseline = await agent(`You are a CUDA profiling expert using Nsight Compute (ncu). Profile the baseline kernel and report Elapsed Cycles plus Speed-of-Light metrics.

# Environment
- NCU binary: ${NCU_BINARY || '(not provided)'}
- Experiment directory: ${EXP_DIR}
- Kernel file: ${KERNEL_PATH}
- Driver / harness: ${DRIVER_PATH || '(not provided)'}
- Kernel name regex for ncu -k: ${KERNEL_NAME_REGEX || '(auto-detect)'}
- Build command: ${BUILD_CMD || '(not provided)'}
- Run command: ${RUN_CMD || '(not provided)'}
- GPU type: ${GPU_TYPE}

# Kernel source
\`\`\`cuda
${baselineKernel.substring(0, 4000)}
\`\`\`

# Steps
1. mkdir -p ${EXP_DIR}/baseline
2. Build the kernel only if build_cmd is provided; otherwise perform static compileability review.
3. If ncu_binary and run_cmd are provided, profile and capture **Elapsed Cycles** (the KernelBlaster scalar metric) along with:
   - Memory Throughput %, Compute (SM) Throughput %, Achieved Occupancy %
   - Top stall reason, L2 hit rate, registers/thread
   Do not invent a profiler command, harness, or benchmark binary.
4. If profiling is unavailable, fall back to user-provided test_command/benchmark_command if present; otherwise use static analysis and mark cycles as missing evidence.

Return the parsed metrics.`, {
  label: 'ncu-baseline',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      elapsed_cycles: { type: 'number' },
      memory_throughput_pct: { type: 'number' },
      compute_throughput_pct: { type: 'number' },
      achieved_occupancy_pct: { type: 'number' },
      top_stall_reason: { type: 'string' },
      l2_hit_rate_pct: { type: 'number' },
      registers_per_thread: { type: 'number' },
      profile_summary: { type: 'string' },
      ncu_available: { type: 'boolean' },
    },
    required: ['elapsed_cycles', 'profile_summary'],
  },
})

baselineCycles = ncuBaseline.elapsed_cycles
bestCycles = baselineCycles
log(`Baseline Elapsed Cycles: ${baselineCycles} | ${ncuBaseline.profile_summary.substring(0, 120)}`)

// =============================================================================
// RL Rollouts — each rollout is a multi-step optimization trajectory.
// Database + replay buffer persist across rollouts (and, via OPT_DB_PATH, runs).
// =============================================================================
for (let iter = 0; iter < RL_ITERATIONS; iter++) {
  log(`\n=== Rollout ${iter + 1}/${RL_ITERATIONS} | Best: ${bestCycles} cycles (${(baselineCycles / bestCycles).toFixed(2)}x) | Buffer: ${replayBuffer.length} trajectories ===`)

  const trajectory = { steps: [], total_reward: 0, initial_cycles: bestCycles, final_cycles: bestCycles }
  let currentCode = bestKernelCode
  let currentCycles = bestCycles
  const usedThisRollout = new Set()

  for (let step = 0; step < ROLLOUT_STEPS; step++) {
    // -------------------------------------------------------------------------
    // Phase 2: ProfileState — profile current kernel, classify hardware state
    // -------------------------------------------------------------------------
    phase('ProfileState'); await __genomeReport('ProfileState', meta.name)

    const stateResult = await agent(`You are a CUDA performance-state classifier (KernelBlaster MAIC-RL). Profile the current kernel and classify it into EXACTLY ONE hardware performance state.

# Current kernel (Elapsed Cycles so far: ${currentCycles})
\`\`\`cuda
${currentCode.substring(0, 3500)}
\`\`\`

# Baseline NCU summary
${ncuBaseline.profile_summary}
- Memory Throughput: ${ncuBaseline.memory_throughput_pct || 'N/A'}%
- Compute (SM) Throughput: ${ncuBaseline.compute_throughput_pct || 'N/A'}%
- Achieved Occupancy: ${ncuBaseline.achieved_occupancy_pct || 'N/A'}%
- Top stall: ${ncuBaseline.top_stall_reason || 'N/A'}

# The three known states (choose the best match):
- memory_bandwidth_limited: high mem throughput / bandwidth saturation / coalescing or cache issues
- compute_throughput_limited: SM/compute units saturated / instruction-throughput bound
- latency_occupancy_limited: neither mem nor compute saturated / low occupancy / sync or latency stalls

Re-profile only if ncu_binary and run_cmd were provided; otherwise reason from the metrics above and the code, and mark missing profiler evidence explicitly.
Return the classification.`, {
      label: `state-${iter}-${step}`,
      phase: 'ProfileState',
      schema: {
        type: 'object',
        properties: {
          state_name: { type: 'string', enum: ['memory_bandwidth_limited', 'compute_throughput_limited', 'latency_occupancy_limited'] },
          primary_bottleneck: { type: 'string' },
          evidence: { type: 'string' },
          current_cycles_estimate: { type: 'number' },
        },
        required: ['state_name', 'evidence'],
      },
    })

    const currentState = optDb.optimization_strategies[stateResult.state_name] ? stateResult.state_name : 'latency_occupancy_limited'
    log(`Step ${step + 1}: state=${currentState} | ${stateResult.evidence.substring(0, 90)}`)

    // -------------------------------------------------------------------------
    // Phase 3: Retrieve — rank candidate optimizations for the matched state
    // -------------------------------------------------------------------------
    phase('Retrieve'); await __genomeReport('Retrieve', meta.name)

    const ranked = rankOptimizations(optDb.optimization_strategies[currentState])
      .filter((o) => !usedThisRollout.has(o.technique))
    if (ranked.length === 0) {
      log(`No unused optimizations left for state ${currentState}; ending rollout early.`)
      break
    }
    const candidates = ranked.slice(0, BREADTH)
    log(`Retrieved top-${candidates.length}: ${candidates.map((c) => c.technique).join(', ')}`)

    // -------------------------------------------------------------------------
    // Phase 4: Plan — LLM specializes one strategy, citing DB + NCU evidence
    // -------------------------------------------------------------------------
    phase('Plan'); await __genomeReport('Plan', meta.name)

    const plans = await parallel(
      candidates.map((cand) => () =>
        agent(`You are a CUDA optimization expert guided by the KernelBlaster knowledge base. Turn this retrieved strategy into a concrete, evidence-based plan for THIS kernel.

# Operation: ${OP_DESC} (${opType})
# Performance state: ${currentState}
# Retrieved strategy: ${cand.technique}
  ${cand.description}
  DB stats: confidence=${cand.confidence_score.toFixed(2)}, usage=${cand.usage_count}, ${cand.actual_speedup ? `measured ${cand.actual_speedup.toFixed(2)}x` : 'unmeasured'}

# NCU evidence
${stateResult.evidence}

# Current kernel
\`\`\`cuda
${currentCode.substring(0, 3500)}
\`\`\`

# Knowledge base (other measured strategies, for context)
${dbSummaryForPrompt(optDb)}

Produce a plan that:
1. Names the exact code region + transformation for "${cand.technique}"
2. Cites the state evidence that justifies it
3. Predicts a speedup (predicted_improvement %) grounded in the metric you target
4. Keeps the kernel functionally correct (same outputs as baseline)`, {
          label: `plan-${iter}-${step}-${cand.technique.substring(0, 12)}`,
          phase: 'Plan',
          schema: {
            type: 'object',
            properties: {
              technique: { type: 'string' },
              title: { type: 'string' },
              plan: { type: 'string' },
              evidence: { type: 'string' },
              predicted_improvement: { type: 'number' },
              risk: { type: 'string' },
            },
            required: ['technique', 'plan', 'predicted_improvement'],
          },
        })
      )
    )

    const validPlans = plans.filter(Boolean)
    if (validPlans.length === 0) { log('No valid plans; ending rollout.'); break }

    // -------------------------------------------------------------------------
    // Phase 5: Execute — implement the selected plan(s)
    // -------------------------------------------------------------------------
    phase('Execute'); await __genomeReport('Execute', meta.name)

    const impls = await parallel(
      validPlans.map((plan) => () =>
        agent(`You are an expert CUDA developer. Apply this optimization plan to produce a COMPLETE, compilable kernel.

# Strategy: ${plan.technique} — "${plan.title || plan.technique}"
# Plan: ${plan.plan}
# Evidence: ${plan.evidence || ''}

# Current kernel (optimize THIS):
\`\`\`cuda
${currentCode.substring(0, 4000)}
\`\`\`

Requirements:
1. Output a COMPLETE .cu file (all #includes, kernel(s), and the host entry the driver expects).
2. Must be FUNCTIONALLY CORRECT vs the baseline (the driver validates outputs).
3. Apply ONLY the "${plan.technique}" strategy faithfully.
4. Keep the kernel signature compatible with the driver harness.
5. Compile-able with -lineinfo on ${GPU_TYPE}.

Return the complete CUDA code.`, {
          label: `impl-${iter}-${step}-${plan.technique.substring(0, 12)}`,
          phase: 'Execute',
          schema: {
            type: 'object',
            properties: {
              code: { type: 'string' },
              technique: { type: 'string' },
              implementation_notes: { type: 'string' },
            },
            required: ['code'],
          },
        })
      )
    )

    const variants = []
    for (let i = 0; i < validPlans.length; i++) {
      if (impls[i] && impls[i].code) {
        variants.push({ plan: validPlans[i], code: impls[i].code, technique: validPlans[i].technique })
      }
    }
    if (variants.length === 0) { log('No implementations produced; ending rollout.'); break }

    // -------------------------------------------------------------------------
    // Phase 6: Evaluate — compile, correctness-check, re-profile (Elapsed Cycles)
    // -------------------------------------------------------------------------
    phase('Evaluate'); await __genomeReport('Evaluate', meta.name)

    const evals = await parallel(
      variants.map((v) => () =>
        agent(`You are a CUDA evaluator using the KernelBench-CUDA driver and Nsight Compute. Evaluate this optimized kernel.

# Strategy applied: ${v.technique}
# Driver / harness: ${DRIVER_PATH || '(standalone harness)'}
# Build: ${BUILD_CMD || '(from driver)'} | Run: ${RUN_CMD || '(from driver)'}
# Baseline Elapsed Cycles: ${currentCycles}

# Kernel:
\`\`\`cuda
${v.code.substring(0, 4000)}
\`\`\`

Steps:
1. Static correctness: race conditions, OOB, missing __syncthreads, wrong reductions.
2. Compilability: includes present, valid CUDA, matches driver entry.
3. Build + run the driver to VALIDATE correctness against the reference.
4. Re-profile to get Elapsed Cycles only when ncu_binary plus run_cmd are provided; otherwise use test_command/benchmark_command if present and mark NCU cycles as missing evidence.
5. speedup = baseline_cycles / new_cycles; improvement% = (baseline-new)/baseline*100.

Return the evaluation.`, {
          label: `eval-${iter}-${step}-${v.technique.substring(0, 12)}`,
          phase: 'Evaluate',
          schema: {
            type: 'object',
            properties: {
              is_correct: { type: 'boolean' },
              is_compilable: { type: 'boolean' },
              elapsed_cycles: { type: 'number' },
              speedup: { type: 'number' },
              improvement_pct: { type: 'number' },
              correctness_issues: { type: 'array', items: { type: 'string' } },
              performance_analysis: { type: 'string' },
            },
            required: ['is_correct', 'is_compilable', 'elapsed_cycles', 'speedup'],
          },
        })
      )
    )

    // Pick the best correct variant for this step.
    let bestStep = null
    for (let i = 0; i < variants.length; i++) {
      const e = evals[i]
      if (!e || !e.is_correct || !e.is_compilable) continue
      if (!bestStep || e.speedup > bestStep.eval.speedup) {
        bestStep = { variant: variants[i], eval: e }
      }
    }

    // -------------------------------------------------------------------------
    // Phase 7: Reward — compute RL reward, update DB stats, extend trajectory
    // -------------------------------------------------------------------------
    phase('Reward'); await __genomeReport('Reward', meta.name)

    if (!bestStep) {
      // All variants failed correctness/compile: penalize the attempted techniques.
      for (const v of variants) {
        usedThisRollout.add(v.technique)
        updateOptimizationResult(optDb.optimization_strategies[currentState], v.technique, -100, null)
        dbUpdateLog.push(`[${currentState}] ${v.technique}: FAILED (incorrect/uncompilable) -> confidence down`)
      }
      log(`Step ${step + 1}: all variants invalid; penalized ${variants.length} techniques.`)
      continue
    }

    const e = bestStep.eval
    const newCycles = e.elapsed_cycles || (baselineCycles / (e.speedup || 1.0))
    const actualImprovement = currentCycles > 0 ? ((currentCycles - newCycles) / currentCycles) * 100 : 0
    const isFaster = newCycles < currentCycles
    const predicted = bestStep.variant.plan.predicted_improvement || 0
    const reward = calculateReward(predicted, actualImprovement, isFaster)
    const actualSpeedup = newCycles > 0 ? currentCycles / newCycles : 1.0

    usedThisRollout.add(bestStep.variant.technique)
    updateOptimizationResult(optDb.optimization_strategies[currentState], bestStep.variant.technique, actualImprovement, actualSpeedup)
    dbUpdateLog.push(`[${currentState}] ${bestStep.variant.technique}: ${actualImprovement.toFixed(1)}% (reward ${reward.toFixed(2)})`)

    trajectory.steps.push({
      state: currentState,
      action: bestStep.variant.technique,
      cycles: newCycles,
      predicted_improvement: predicted,
      actual_improvement: actualImprovement,
      reward,
    })
    trajectory.total_reward += reward
    trajectory.final_cycles = newCycles

    log(`Step ${step + 1}: ${bestStep.variant.technique} -> ${actualImprovement.toFixed(1)}% (${actualSpeedup.toFixed(2)}x), reward ${reward.toFixed(2)}`)

    // Adopt improvement as the new current code for the next step.
    if (isFaster) {
      currentCode = bestStep.variant.code
      currentCycles = newCycles
      if (newCycles < bestCycles) {
        bestCycles = newCycles
        bestKernelCode = bestStep.variant.code
        log(`  NEW GLOBAL BEST: ${bestCycles} cycles (${(baselineCycles / bestCycles).toFixed(2)}x vs baseline)`)
      }
    } else if (actualImprovement < -25) {
      // Severe degradation: stop the rollout (matches the paper's early stop).
      log(`  Severe degradation (${actualImprovement.toFixed(1)}%); stopping rollout.`)
      break
    }
  }

  // Commit the trajectory to the replay buffer.
  replayBuffer.push(trajectory)
  totalTrajectories += 1

  // ---------------------------------------------------------------------------
  // Phase 8: Iterate — periodic policy-update cycle over the replay buffer
  // ---------------------------------------------------------------------------
  phase('Iterate'); await __genomeReport('Iterate', meta.name)

  if (totalTrajectories % UPDATE_FREQUENCY === 0 && replayBuffer.length >= 2) {
    const perfData = []
    for (const traj of replayBuffer.slice(-10)) {
      for (const s of traj.steps) {
        perfData.push({ state: s.state, technique: s.action, predicted: s.predicted_improvement, actual: s.actual_improvement, reward: s.reward })
      }
    }

    const policyUpdate = await agent(`You are the KernelBlaster policy-evaluation agent. Analyze recent optimization trajectories and recommend confidence adjustments to the knowledge base.

# Recent steps (state, technique, predicted%, actual%, reward):
${JSON.stringify(perfData, null, 2)}

# Current DB (per state, top strategies):
${dbSummaryForPrompt(optDb)}

Identify:
1. Which techniques systematically over- or under-predict (prediction bias).
2. Which (state, technique) pairs reliably improve performance.
3. Recommended confidence adjustments (technique -> +/- delta in [-0.3, 0.3]).

Return structured recommendations.`, {
      label: `policy-update-${iter}`,
      phase: 'Iterate',
      schema: {
        type: 'object',
        properties: {
          analysis: { type: 'string' },
          confidence_adjustments: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                state: { type: 'string' },
                technique: { type: 'string' },
                delta: { type: 'number' },
                reason: { type: 'string' },
              },
              required: ['state', 'technique', 'delta'],
            },
          },
        },
        required: ['analysis'],
      },
    })

    // Apply bounded confidence adjustments from the policy-update cycle.
    let applied = 0
    for (const adj of (policyUpdate.confidence_adjustments || [])) {
      const entry = optDb.optimization_strategies[adj.state]
      if (!entry) continue
      const opt = entry.optimizations.find((o) => o.technique === adj.technique)
      if (!opt) continue
      const delta = Math.max(-0.3, Math.min(0.3, adj.delta || 0))
      opt.confidence_score = Math.max(0.05, Math.min(1.0, (opt.confidence_score || 0.5) + delta))
      applied += 1
    }
    log(`Policy-update cycle: applied ${applied} confidence adjustments. ${policyUpdate.analysis.substring(0, 100)}`)
  }

  log(`Rollout ${iter + 1} done. Trajectory reward: ${trajectory.total_reward.toFixed(2)}, ${trajectory.steps.length} steps.`)
}

// =============================================================================
// Persist the knowledge base (cross-task / cross-run memory) + final report
// =============================================================================
if (OPT_DB_PATH) {
  await agent(`Persist the updated KernelBlaster optimization knowledge base to disk so it carries across runs and kernels.

Write this exact JSON (pretty-printed) to: ${OPT_DB_PATH}

\`\`\`json
${JSON.stringify(optDb, null, 2).substring(0, 60000)}
\`\`\`

Use a file write. Confirm the byte count written.`, {
    label: 'persist-db',
    phase: 'Iterate',
    schema: {
      type: 'object',
      properties: { written: { type: 'boolean' }, path: { type: 'string' }, bytes: { type: 'number' } },
      required: ['written'],
    },
  })
  log(`Persisted optimization DB to ${OPT_DB_PATH}`)
}

const bufferStats = {
  num_trajectories: replayBuffer.length,
  avg_reward: replayBuffer.length ? replayBuffer.reduce((s, t) => s + t.total_reward, 0) / replayBuffer.length : 0,
  success_rate: replayBuffer.length ? replayBuffer.filter((t) => t.final_cycles < t.initial_cycles).length / replayBuffer.length : 0,
}

const finalReport = await agent(`Write a concise technical report for this KernelBlaster (MAIC-RL) optimization run.

# Results
- Operation: ${OP_DESC} (${opType})
- GPU: ${GPU_TYPE}
- Baseline Elapsed Cycles: ${baselineCycles}
- Best Elapsed Cycles: ${bestCycles}
- Overall speedup: ${(baselineCycles / bestCycles).toFixed(2)}x
- RL rollouts: ${RL_ITERATIONS}, trajectories: ${replayBuffer.length}
- Replay buffer: avg reward ${bufferStats.avg_reward.toFixed(2)}, success rate ${(bufferStats.success_rate * 100).toFixed(0)}%

# Knowledge-base mutations (chronological)
${dbUpdateLog.map((l, i) => `${i + 1}. ${l}`).join('\n') || '(none)'}

# Final per-state strategy stats
${dbSummaryForPrompt(optDb)}

# Final kernel
\`\`\`cuda
${bestKernelCode.substring(0, 3000)}
\`\`\`

Write:
1. The RL optimization journey: which states were hit, which strategies were retrieved, what rewards resulted.
2. Which (state, technique) pairs the knowledge base now favors and why (cross-task transferable insights).
3. Prediction accuracy: where predicted vs actual improvement diverged.
4. Remaining bottleneck of the final kernel and recommended next strategies.`, {
  label: 'final-report',
  phase: 'Iterate',
})

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  baseline_elapsed_cycles: baselineCycles,
  best_elapsed_cycles: bestCycles,
  overall_speedup: baselineCycles / bestCycles,
  rl_iterations: RL_ITERATIONS,
  trajectories_count: replayBuffer.length,
  buffer_avg_reward: bufferStats.avg_reward,
  buffer_success_rate: bufferStats.success_rate,
  optimization_database: optDb,
  db_mutations: dbUpdateLog,
  best_kernel_code: bestKernelCode,
  report: finalReport,
}
