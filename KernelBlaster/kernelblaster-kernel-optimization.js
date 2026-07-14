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

// --- BEGIN sol-execbench-eval substrate (auto-inlined by scripts/patch-sol-execbench-eval.js) ---
const SOL_SOLUTION_CONTRACT = [
  'SOL-EXECBENCH SOLUTION CONTRACT (this task is evaluated by the sol-execbench CLI):',
  '',
  'You are authoring a kernel that will be packaged into a solution.json and run by',
  'the sol-execbench harness, which compiles it internally. Therefore:',
  '',
  '1. Emit a COMPLETE kernel source plus a torch binding that exposes the task',
  '   entry point (run(...)). Do NOT write a standalone main()/CLI harness.',
  '2. Match the task reference signature exactly (same argument order/dtypes).',
  '3. Do NOT package, compile, or benchmark yourself — the workflow + substrate',
  '   handle pack -> sol-execbench -> parse. Return only the kernel + binding.',
].join('\n')

function __solQ(s) { return `"${String(s).replace(/"/g, '\\"')}"` }

function __solExecbenchEvalPlan(ctx) {
  const substrateDir = ctx.substrateDir            // abs path to _substrate/integration
  const kernelSource = ctx.kernelSource            // path to candidate kernel on disk
  const contractEnv = ctx.contractEnv              // path to session contract.env
  const solutionOut = ctx.solutionOut              // where to write solution.json
  const benchOut = ctx.benchOut                    // where sol-execbench writes bench.jsonl
  const solCli = ctx.solCli                        // e.g. /abs/sol-execbench/.venv/bin/sol-execbench
  const taskDir = ctx.taskDir                      // FlashInfer-Bench/<task> dir
  const benchConfig = ctx.benchConfig              // --config path
  const seedDir = ctx.seedDir                      // cd target for the run
  const cvd = ctx.cudaVisibleDevices || '0'
  const ld = ctx.ldLibraryPath ? `LD_LIBRARY_PATH=${__solQ(ctx.ldLibraryPath)}:$LD_LIBRARY_PATH ` : ''

  const pack = `python3 ${__solQ(substrateDir + '/pack_sol_candidate.py')} --kernel ${__solQ(kernelSource)} --contract ${__solQ(contractEnv)} --out ${__solQ(solutionOut)}`
  const run = `cd ${__solQ(seedDir)} && ${ld}CUDA_VISIBLE_DEVICES=${cvd} ${__solQ(solCli)} ${__solQ(taskDir)} --solution ${__solQ(solutionOut)} --config ${__solQ(benchConfig)} -o ${__solQ(benchOut)}`
  const parse = `python3 ${__solQ(substrateDir + '/parse_sol_bench.py')} ${__solQ(benchOut)}`

  return {
    pack,
    run,
    parse,
    order: ['pack', 'run', 'parse'],
    cleanupInvariant: 'solution.json + bench.jsonl are per-candidate scratch files in the run dir; overwrite freely. No project source is mutated (non-mutating method).',
  }
}
// --- END sol-execbench-eval substrate ---


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

const WORKFLOW_NAME = 'kernelblaster-kernel-optimization'


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
const BUILD_CMD = args.build_cmd || args.build_command || ''
const RUN_CMD = args.run_cmd || ''
const RL_ITERATIONS = args.rl_iterations || 3
const ROLLOUT_STEPS = args.rollout_steps || 4
const BREADTH = args.breadth || 2
const UPDATE_FREQUENCY = args.update_frequency || 3
const RUN_TS = args.run_timestamp_iso || 'unknown'

// Fallback (no NCU): custom test + benchmark commands
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || ''
const SOL_CLI = args.sol_cli || ''
const SOL_TASK_DIR = args.sol_task_dir || ''
const SOL_BENCH_CONFIG = args.sol_bench_config || ''
const SOL_SEED_DIR = args.sol_seed_dir || EXP_DIR
const SOL_CVD = args.sol_cuda_visible_devices || '0'
const SOL_LD_LIBRARY_PATH = args.sol_ld_library_path || ''

// --- Project-native integration (embedded operators via integration-strategist) ---
// For inference-engine embedded operators (e.g. llama.cpp .cuh) the candidate cannot
// compile as a standalone TU; it is built/tested/benchmarked INSIDE the host project.
// BENCH_CMD already names the legacy (standalone) benchmark, so the project-native
// benchmark uses a distinct name (PROJECT_BENCH_CMD) to avoid a duplicate `const`
// (which would break wfcheck). KernelBlaster is BESPOKE raw-ncu (no backend driver),
// so there is NO USE_DRIVER / USE_DRIVER_STANDALONE — IS_EMBEDDED alone gates the new
// embedded eval branch; the legacy benchmark path stays byte-identical when standalone.
// BUILD_CMD already exists above (args.build_cmd) — reuse it; do NOT redeclare.
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const PROJECT_BENCH_CMD = args.project_benchmark_command || BENCH_CMD || ''
const REGISTER_SCRIPT = args.register_script || ''
const SOL_SUBSTRATE_DIR = args.sol_substrate_dir || `${args.substrate_dir || '_substrate'}/integration`
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

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
phase('Setup')

if (INPUT_MODE === 'generate_then_optimize') {
  const generated = await agentRetry(() => agent(`No kernel_path was provided. Generate and verify an initial CUDA kernel before starting KernelBlaster.

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
  }), { retries: 5 })
  initialCandidates = generated.initial_candidates || []
  initialGenerationResult = generated.initial_generation_result || { verified: false }
  generatedKernelPath = generated.generated_kernel_path || ''
  if (!generatedKernelPath) throw new Error('Generation mode did not produce generated_kernel_path')
  if ((TEST_CMD || BENCH_CMD) && initialGenerationResult.verified === false) throw new Error('No generated seed passed correctness evidence')
  KERNEL_PATH = generatedKernelPath
}

const setupResult = await agentRetry(() => agent(`Read the CUDA kernel at: ${KERNEL_PATH}
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

Return ONLY the JSON object.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"baseline_analysis","speedup":null,"note":"<op_type + kernels + whether persistent DB was loaded, one line>"}`, {
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
}), { retries: 5 })

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

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it below. The agent only classifies the task (fuzzy op_class/size); the
// substrate strategist DETERMINISTICALLY picks the method and STAMPS confidence by
// method (measured/inferred/hypothesized) -- the model must NOT assign confidence.
// Falls back to native_profiler (NCU 'measured') so the happy path is unchanged.
// When method !== native_profiler, NCU elapsed-cycle/state metrics are unavailable,
// so the hardware-state classification + any DB write are stamped as inferred. ---
const SUBSTRATE = args.substrate_dir || '_substrate'
const PY = args.substrate_command_prefix || ''
const BACKEND_MANIFEST = args.backend_manifest || `${SUBSTRATE}/backends/cuda/manifest.json`
const STRATEGIST_SIZE = args.profiling_size || 'large'
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured' }
const _pd = await agentRetry(() => agent(`Read ${KERNEL_PATH}; classify its op_class (one of attention|gemm|elementwise|reduction|default; the kernel is op_type='${opType}', op_description='${OP_DESC}') and size (tiny|small|large; default '${STRATEGIST_SIZE}'). Then run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${BACKEND_MANIFEST} --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`.
Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`, {
  model: MODEL.mechanical,
  label: 'profiling-strategist',
  phase: 'Setup',
  schema: { type: 'object', additionalProperties: true },
}), { retries: 5, allowNull: true })
if (_pd && _pd.method) PROFILING_DECISION = _pd
log(`Profiling-strategist: method=${PROFILING_DECISION.method}, confidence=${PROFILING_DECISION.confidence}`)

// --- integration-strategist: route build/test mode (standalone vs embedded_*).
// For an inference-engine embedded operator (e.g. llama.cpp .cuh referenced via
// KERNEL_PATH), can_compile_standalone=no, so the candidate is built/tested/benchmarked
// INSIDE the host project rather than as an isolated TU. KernelBlaster has NO backend
// driver (it inlines ncu + benchmark), so the only thing this gates is the new embedded
// eval branch; the legacy benchmark path stays byte-identical when standalone. ---
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Read ${KERNEL_PATH || '(not provided)'}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
    `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_integ && _integ.method) INTEGRATION_DECISION = _integ
}
log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error('integration-strategist returned derive_adapter — provide project_root + build/test commands')
}
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
const IS_SOL = INTEGRATION_DECISION.method === 'sol_execbench_solution'
// The embedded operator file we swap in place is the project-referenced KERNEL_PATH.
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// A-O1 closure: native_profiler chosen but the inline NCU profiler is unreachable on
// the embedded path (no isolated TU to ncu) → downgrade to perf_heuristic so the
// project benchmark's throughput is normalized instead. Also covers no-ncu standalone.
if (PROFILING_DECISION.method === 'native_profiler' && (IS_EMBEDDED || !NCU_BINARY)) {
  log(`profiling: native_profiler unreachable (${IS_EMBEDDED ? 'embedded path' : 'no ncu_binary'}) -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'project-native-perf', rationale: 'native_profiler unreachable -> perf_heuristic' }
}

// NCU baseline profile -> Elapsed Cycles + Speed-of-Light metrics
const ncuBaseline = await agentRetry(() => agent(`You are a CUDA profiling expert using Nsight Compute (ncu). Profile the baseline kernel and report Elapsed Cycles plus Speed-of-Light metrics.

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

Return the parsed metrics.
Profiling-strategist selected method='${PROFILING_DECISION.method}', confidence='${PROFILING_DECISION.confidence}'. If method !== 'native_profiler', NCU elapsed-cycle/state metrics are unavailable — derive the hardware-performance state from throughput/perf evidence instead, mark the state classification and any DB write as confidence='${PROFILING_DECISION.confidence}' (inferred, NOT measured), and do NOT fabricate NCU counters. If method === 'native_profiler', proceed with NCU as written.`, { model: MODEL.profile,
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
}), { retries: 5 })

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
    phase('ProfileState')

    const stateResult = await agentRetry(() => agent(`You are a CUDA performance-state classifier (KernelBlaster MAIC-RL). Profile the current kernel and classify it into EXACTLY ONE hardware performance state.

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
Return the classification.
Profiling-strategist selected method='${PROFILING_DECISION.method}', confidence='${PROFILING_DECISION.confidence}'. If method !== 'native_profiler', NCU elapsed-cycle/state metrics are unavailable — derive the hardware-performance state from throughput/perf evidence instead, mark the state classification and any DB write as confidence='${PROFILING_DECISION.confidence}' (inferred, NOT measured), and do NOT fabricate NCU counters. ${PROFILING_DECISION.method === 'perf_heuristic' ? `Because method='perf_heuristic', ALSO write heuristic_bclass (memory_bound|compute_bound|latency_bound) from the throughput ratio so the diagnose contract does not fall to unknown.` : ''} If method === 'native_profiler', proceed with NCU as written.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (rollout ${iter}, step ${step}):
{"workflow":"${WORKFLOW_NAME}","phase":"ProfileState","ts":"<ts>","status":"done","candidate_id":"r${iter}-s${step}","technique":"<the classified state_name>","speedup":null,"note":"<primary_bottleneck + the evidence in one line>"}`, {
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
    }), { retries: 5 })

    const currentState = optDb.optimization_strategies[stateResult.state_name] ? stateResult.state_name : 'latency_occupancy_limited'
    log(`Step ${step + 1}: state=${currentState} | ${stateResult.evidence.substring(0, 90)}`)

    // -------------------------------------------------------------------------
    // Phase 3: Retrieve — rank candidate optimizations for the matched state
    // -------------------------------------------------------------------------
    phase('Retrieve')

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
    phase('Plan')

    const plans = await parallel(
      candidates.map((cand) => () =>
        agentRetry(() => agent(`You are a CUDA optimization expert guided by the KernelBlaster knowledge base. Turn this retrieved strategy into a concrete, evidence-based plan for THIS kernel.

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
4. Keeps the kernel functionally correct (same outputs as baseline)

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (rollout ${iter}, step ${step}):
{"workflow":"${WORKFLOW_NAME}","phase":"Plan","ts":"<ts>","status":"done","candidate_id":"r${iter}-s${step}-${cand.technique}","technique":"${cand.technique}","speedup":null,"note":"<the code region + transformation planned and the predicted_improvement, one line>"}`, {
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
        }), { retries: 5 })
      )
    )

    const validPlans = plans.filter(Boolean)
    if (validPlans.length === 0) { log('No valid plans; ending rollout.'); break }

    // -------------------------------------------------------------------------
    // Phase 5: Execute — implement the selected plan(s)
    // -------------------------------------------------------------------------
    phase('Execute')

    const impls = await parallel(
      validPlans.map((plan) => () =>
        agentRetry(() => agent(`You are an expert CUDA developer. Apply this optimization plan to produce a COMPLETE, compilable kernel.

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
${INTEGRATION_DECISION.method === 'sol_execbench_solution' ? `\n${SOL_SOLUTION_CONTRACT}` : ''}

Return the complete CUDA code.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (rollout ${iter}, step ${step}):
{"workflow":"${WORKFLOW_NAME}","phase":"Execute","ts":"<ts>","status":"done","candidate_id":"r${iter}-s${step}-${plan.technique}","technique":"${plan.technique}","speedup":null,"note":"<what changed in the kernel to apply this strategy, one line>"}`, {
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
        }), { retries: 5 })
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
    phase('Evaluate')

    let evals
    if (IS_SOL) {
      // sol-execbench candidates are independent scratch artifacts, but the
      // workflow still evaluates them serially so each parse result is tied to
      // exactly one solution.json/bench.jsonl pair.
      evals = []
      for (let i = 0; i < variants.length; i++) {
        const v = variants[i]
        const suffix = `r${iter}-s${step}-${i}`.replace(/[^A-Za-z0-9_]/g, '_')
        const kPath = `${EXP_DIR}/variants/${suffix}/candidate.cu`
        await agentRetry(() => agent(`Write the candidate kernel source to ${kPath} (mkdir -p its dir first).\n\n` +
          `\`\`\`cuda\n${(v.code || '').substring(0, 6000)}\n\`\`\`\n` +
          `Return {ok:true, path:"${kPath}"}.`,
          { model: MODEL.mechanical, label: `sol-materialize-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
        const variantName = `sol_${suffix}`
        const plan = __solExecbenchEvalPlan({
          substrateDir: SOL_SUBSTRATE_DIR,
          kernelSource: kPath,
          contractEnv: `${EXP_DIR}/contract.env`,
          solutionOut: `${EXP_DIR}/${variantName}.solution.json`,
          benchOut: `${EXP_DIR}/${variantName}.bench.jsonl`,
          solCli: SOL_CLI, taskDir: SOL_TASK_DIR, benchConfig: SOL_BENCH_CONFIG,
          seedDir: SOL_SEED_DIR, cudaVisibleDevices: SOL_CVD, ldLibraryPath: SOL_LD_LIBRARY_PATH,
        })
        const solResult = await agentRetry(() => agent(
          `Evaluate this candidate with sol-execbench. Run IN THIS EXACT ORDER:\n1. Pack: ${plan.pack}\n2. Run: ${plan.run}\n3. Parse: ${plan.parse}\n` +
          `Parse the line SPEEDUP=<geomean> STATUS=<PASS|FAIL> WORKLOADS=<passed>/<total>; do not fabricate values. Return {is_correct, is_compilable, elapsed_cycles, speedup, improvement_pct}. ${plan.cleanupInvariant}`,
          { model: MODEL.profile, label: `sol-evaluate-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
        evals.push(solResult ? {
          is_correct: !!solResult.is_correct,
          is_compilable: !!solResult.is_compilable,
          elapsed_cycles: Number(solResult.elapsed_cycles || 0),
          speedup: Number(solResult.speedup || 0),
          improvement_pct: Number(solResult.improvement_pct || 0),
          heuristic_bclass: solResult.heuristic_bclass || 'unknown',
        } : null)
      }
    } else if (IS_EMBEDDED) {
      // --- Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch) ---
      // SEPARATE SERIAL for-loop (NOT parallel): for inference-engine embedded operators
      // (e.g. llama.cpp .cuh) embedded_inplace mutates the shared project file KERNEL_PATH
      // and embedded_dispatch shares the project build, so candidates CANNOT be evaluated
      // concurrently (parallel-embedded-race bug-class). Builds/tests/benchmarks the
      // candidate INSIDE the host project instead of inline-ncu on an isolated TU.
      // Returns the same {is_correct,is_compilable,elapsed_cycles,speedup,improvement_pct}
      // shape the legacy parallel path produces, so downstream Reward logic is unchanged.
      evals = []
      for (let i = 0; i < variants.length; i++) {
        const v = variants[i]
        const suffix = `r${iter}-s${step}-${i}`.replace(/[^A-Za-z0-9_]/g, '_')
        const kPath = `${EXP_DIR}/variants/${suffix}/candidate.cu`
        const variant = `kb_${suffix}`.replace(/[^A-Za-z0-9_]/g, '_')
        // Materialize the candidate source so the embedded eval can apply/register it.
        await agentRetry(() => agent(`Write the candidate kernel source to ${kPath} (mkdir -p its dir first).\n\n` +
          `\`\`\`cuda\n${(v.code || '').substring(0, 6000)}\n\`\`\`\n` +
          `Return {ok:true, path:"${kPath}"}.`,
          { model: MODEL.mechanical, label: `embedded-materialize-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
        let embResult = null
        if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
          embResult = await agentRetry(() => agent(
            `EMBEDDED-INPLACE EVAL (serial). Strategy: ${v.technique}\n` +
            `Candidate: ${kPath} | project operator file: ${KERNEL_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
            `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
            `2. Apply candidate: cp ${kPath} ${KERNEL_PATH}\n3. Build: ${BUILD_CMD}\n4. Test: ${TEST_CMD}\n5. Benchmark: ${PROJECT_BENCH_CMD || TEST_CMD}\n` +
            `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
            `baseline_cycles=${currentCycles}; speedup = baseline_cycles / new_cycles (or measured latency ratio); ` +
            `improvement_pct = (baseline-new)/baseline*100. Parse heuristic_bclass (memory/compute/latency bound).\n` +
            `Return {is_correct, is_compilable, elapsed_cycles, speedup, improvement_pct, heuristic_bclass}.`,
            { model: MODEL.profile, label: `embedded-inplace-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
        } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
          const _plan = typeof __embeddedEvalPlan === 'function'
            ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: TEST_CMD, benchmarkCmd: PROJECT_BENCH_CMD || TEST_CMD })
            : null
          if (_plan) {
            embResult = await agentRetry(() => agent(
              `EMBEDDED-DISPATCH EVAL (serial). Strategy: ${v.technique}\n` +
              `Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
              `baseline_cycles=${currentCycles}; speedup = baseline_cycles / new_cycles (or measured latency ratio); ` +
              `improvement_pct = (baseline-new)/baseline*100. Parse heuristic_bclass.\n` +
              `Return {is_correct, is_compilable, elapsed_cycles, speedup, improvement_pct, heuristic_bclass}.`,
              { model: MODEL.profile, label: `embedded-dispatch-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
          }
        }
        evals.push(embResult ? {
          is_correct: !!embResult.is_correct,
          is_compilable: !!embResult.is_compilable,
          elapsed_cycles: Number(embResult.elapsed_cycles || 0),
          speedup: Number(embResult.speedup || 0),
          improvement_pct: Number(embResult.improvement_pct || 0),
          heuristic_bclass: embResult.heuristic_bclass || 'unknown',
        } : null)
      }
    } else {
    evals = await parallel(
      variants.map((v) => () =>
        agentRetry(() => agent(`You are a CUDA evaluator using the KernelBench-CUDA driver and Nsight Compute. Evaluate this optimized kernel.

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

Return the evaluation.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if correct AND compilable, else "error"; speedup is the measured number, or null if no profiler/test evidence was available):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"r${iter}-s${step}-${v.technique}","technique":"${v.technique}","speedup":<number or null>,"note":"<compiled? correct? elapsed_cycles + improvement_pct; or the failure reason>"}`, {
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
        }), { retries: 5 })
      )
    )
    }

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
    phase('Reward')

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
  phase('Iterate')

  if (totalTrajectories % UPDATE_FREQUENCY === 0 && replayBuffer.length >= 2) {
    const perfData = []
    for (const traj of replayBuffer.slice(-10)) {
      for (const s of traj.steps) {
        perfData.push({ state: s.state, technique: s.action, predicted: s.predicted_improvement, actual: s.actual_improvement, reward: s.reward })
      }
    }

    const policyUpdate = await agentRetry(() => agent(`You are the KernelBlaster policy-evaluation agent. Analyze recent optimization trajectories and recommend confidence adjustments to the knowledge base.

# Recent steps (state, technique, predicted%, actual%, reward):
${JSON.stringify(perfData, null, 2)}

# Current DB (per state, top strategies):
${dbSummaryForPrompt(optDb)}

Identify:
1. Which techniques systematically over- or under-predict (prediction bias).
2. Which (state, technique) pairs reliably improve performance.
3. Recommended confidence adjustments (technique -> +/- delta in [-0.3, 0.3]).

Return structured recommendations.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (after rollout ${iter}):
{"workflow":"${WORKFLOW_NAME}","phase":"Iterate","ts":"<ts>","status":"done","candidate_id":"policy-update-${iter}","technique":"policy_update_cycle","speedup":null,"note":"<prediction-bias findings + how many confidence adjustments recommended, one line>"}`, {
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
    }), { retries: 5 })

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
  await agentRetry(() => agent(`Persist the updated KernelBlaster optimization knowledge base to disk so it carries across runs and kernels.

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
  }), { retries: 5 })
  log(`Persisted optimization DB to ${OPT_DB_PATH}`)
}

const bufferStats = {
  num_trajectories: replayBuffer.length,
  avg_reward: replayBuffer.length ? replayBuffer.reduce((s, t) => s + t.total_reward, 0) / replayBuffer.length : 0,
  success_rate: replayBuffer.length ? replayBuffer.filter((t) => t.final_cycles < t.initial_cycles).length / replayBuffer.length : 0,
}

const finalReport = await agentRetry(() => agent(`Write a concise technical report for this KernelBlaster (MAIC-RL) optimization run.

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
}), { retries: 5 })

// embedded_inplace exit safety net: unconditionally restore the pristine operator file.
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${KERNEL_PATH}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Iterate', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

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
