export const meta = {
  name: 'argus-kernel-optimization',
  description: 'Agentic GPU kernel optimization guided by data-flow invariants with ICRL planning (ARGUS methodology)',
  whenToUse: 'When optimizing GPU kernels that require coordinated reasoning over tiling, shared-memory staging, software pipelining, and instruction scheduling. Uses compile-time data-flow invariants for dense structured feedback instead of sparse pass/fail signals. Best for performance-critical kernels (GEMM, attention, MoE) targeting near-library-level throughput.',
  phases: [
    { title: 'Setup', detail: 'Read kernel spec, hardware config, initialize knowledge base context' },
    { title: 'Plan', detail: 'ICRL planner proposes optimizations + data-flow invariants from knowledge base' },
    { title: 'Select', detail: 'Optimization selector samples from planner proposals, resolves dependencies' },
    { title: 'Lower', detail: 'Lowering agent implements transformations with tag functions and assertions' },
    { title: 'Validate', detail: 'Compile-time invariant checking + unit tests + runtime profiling' },
    { title: 'Learn', detail: 'ICRL update: reward from invariant violations + performance, update planner policy' },
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

const WORKFLOW_NAME = 'argus-kernel-optimization'

const SUBSTRATE = args.substrate_dir || '_substrate'
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }


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
// Per-turn wall-clock watchdog (parity with CUDAAgent #12/#14). ARGUS is a linear
// pipeline (no MAX_TURNS loop), so a single hung doer turn (Validator/Lowering
// runs build+test+bench behind the agent) stalls the whole run. Wrapping the
// eval-bearing turns bounds them; on expiry the `turn-timeout:` reject propagates
// and aborts the round cleanly (#20-style) instead of hanging.
const TURN_TIMEOUT_MS = (args.turn_timeout_min || 12) * 60 * 1000  // per-turn wall-clock cap
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


// =============================================================================
// ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants
// =============================================================================
//
// Source: "ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants"
//         Mai et al., CausalFlow / HKUST / Tsinghua / Stanford / UCAS / UC Riverside
//         arXiv:2604.18616, 2026
//
// Core insight: Data-flow invariants provide DENSE, STRUCTURED feedback for
// GPU kernel optimization. Instead of sparse pass/fail from unit tests:
//   - Tag functions annotate tensor elements with symbolic coordinates
//   - Tag assertions verify relational constraints at use sites
//   - Violations produce concrete counterexamples (thread, element, program point)
//   - This enables the agent to diagnose GLOBAL constraint violations
//
// Architecture (5 components from Section 6):
//   1. Knowledge Base: curated optimization skills (global intrusive + local + ISA)
//   2. Learnable Planner (ICRL): proposes (optimization, context, invariants)
//   3. Optimization Selector: samples from proposals, resolves coupling
//   4. Lowering Agent: implements transformations in DSL with tag functions
//   5. Validator: invariant checking + unit tests + profiling → reward signal
//
// The ICRL loop (Algorithm 1):
//   for each task:
//     for t = 0..T:
//       P_t = planner(state_t)           // ranked proposals
//       a_t = SELECT(P_t)                // sample optimization plan
//       s_{t+1} = LOWER(state_t, a_t)    // implement transformation
//       r_t = REWARD(s_{t+1}, tests)     // invariant violations + perf
//     PolicyEval → Analyze → ParameterUpdate (text gradients)
//
// Usage:
//   Workflow({name: 'argus-kernel-optimization', args: {
//     kernel_path: '/path/to/kernel.py',       // Argus DSL or CUDA/Triton
//     kernel_spec: 'Flash attention GQA, bf16, d=128, Br=256, Bc=64',
//     target_gpu: 'AMD MI300X' | 'NVIDIA H100',
//     test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     invariant_check_command: '<user-provided invariant checker command with {kernel_path}/{invariant_result_path}>',
//     invariant_result_path: '/tmp/argus_exp/invariants/result.json',
//     knowledge_base_path: '',
//     iterations: 10,
//     inner_steps: 5,
//     optimization_categories: ['global_intrusive', 'local_source', 'isa_specific'],
//     // --- Embedded-dispatch mode (optional; default 'standalone') ---
//     integration_pattern: 'embedded',     // 'standalone' (default) | 'embedded*'
//     register_script: '/abs/scripts/llamacpp_register_variant.py', // adapter (3 verbs)
//     project_root: '/abs/llama.cpp',       // or ggml_root: project root passed to adapter
//     reference_cuh: '/abs/.../fattn-mma.cuh', // or reference_file: dispatch sig to match
//     register_params: '--dkq 256 --dv 256 --cmake-build-dir /p/build', // opaque adapter params
//     build_command: '<project-native build cmd>', // required in embedded mode (plan.build)
//   }})
//   In embedded mode, test_command/benchmark_command are run VERBATIM against the
//   built project (no {kernel_path}/{result_path} substitution); the candidate is
//   registered into the project's dispatch table and unregistered after each eval.
//
// =============================================================================

// --- Required Args ---
let KERNEL_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const KERNEL_SPEC = args.kernel_spec || ''

// --- Optional Args ---
const HARDWARE_TARGET = args.target_gpu || 'NVIDIA H100'
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || ''
const BUILD_CMD = args.build_command || ''  // project-native build (embedded mode)

// --- Sol-execbench integration args (supplied by KerSor Task 8; unused on non-sol tasks) ---
const SOL_CLI = args.sol_cli || ''
const SOL_TASK_DIR = args.sol_task_dir || ''
const SOL_BENCH_CONFIG = args.sol_bench_config || ''
const SOL_SEED_DIR = args.sol_seed_dir || EXP_DIR
const SOL_CVD = args.sol_cuda_visible_devices || '0'
const SOL_LD_LIBRARY_PATH = args.sol_ld_library_path || ''
const SOL_SUBSTRATE_DIR = args.sol_substrate_dir || ''   // KerSor passes <workflow_lib>/_substrate/integration
const INVARIANT_CHECK_CMD = args.invariant_check_command || ''
const INVARIANT_RESULT_PATH = args.invariant_result_path || `${args.exp_dir || '/tmp/argus_exp'}/invariants/latest.json`
const KNOWLEDGE_BASE_PATH = args.knowledge_base_path || ''
const ITERATIONS = args.iterations || 5
const INNER_STEPS = args.inner_steps || 3
const EXP_DIR = args.exp_dir || '/tmp/argus_exp'
const OPT_CATEGORIES = args.optimization_categories || ['global_intrusive', 'local_source', 'isa_specific']
const invariantEvidenceMode = INVARIANT_CHECK_CMD ? 'executable_checker' : 'missing_invariant_evidence'

if (!KERNEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH && !KERNEL_SPEC) {
  throw new Error('Provide one of kernel_path, problem_definition, problem_path, or kernel_spec')
}

const LANGUAGE = args.language || 'cuda'

// --- Embedded-dispatch mode (fully gated; standalone path unchanged) ---
const INTEGRATION_PATTERN = (args.integration_pattern || 'standalone')
const EMBEDDED = INTEGRATION_PATTERN.startsWith('embedded')
const REGISTER_SCRIPT = args.register_script || ''
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const REFERENCE_FILE = args.reference_cuh || args.reference_file || ''
const REGISTER_PARAMS = args.register_params || ''

if (EMBEDDED) {
  const missing = []
  if (!REGISTER_SCRIPT) missing.push('register_script')
  if (!PROJECT_ROOT) missing.push('project_root (or ggml_root)')
  if (!TEST_CMD) missing.push('test_command')
  if (!BENCH_CMD) missing.push('benchmark_command')
  // The project build command is reused below as plan.build; ARGUS has no
  // dedicated build arg, so a build_command must be supplied in embedded mode.
  if (!BUILD_CMD) missing.push('build_command')
  if (missing.length) {
    throw new Error(`integration_pattern="${INTEGRATION_PATTERN}" (embedded) requires non-empty: ${missing.join(', ')}`)
  }
}

const SEED_CANDIDATES = args.seed_candidates || 3
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null

// State (ICRL)
let bestKernelCode = null
let bestThroughput = 0
let plannerPolicy = ''
let optimizationHistory = []
let invariantViolationLog = []
let candidateBeam = []

// Knowledge base categories (ARGUS Table 1)
const KNOWLEDGE_BASE = {
  global_intrusive: [
    'software_pipelining: overlap computation with data transfers using multi-stage buffering',
    'split_k: dispatch multiplications along K dimension across multiple threadgroups',
    'mfma_matmul: adopt swizzled layouts to use hardware matrix core (MFMA/WMMA) instructions',
    'stagger_k: stagger workloads along K to mitigate memory controller hot spots',
    'async_memcpy: asynchronous data loading with explicit fence/barrier management',
  ],
  local_source: [
    'bank_conflict_mitigation: pad or swizzle shared memory to eliminate bank conflicts',
    'vectorized_loads: use float4/int4/buffer_load_dwordx4 for coalesced wide memory access',
    'loop_unrolling: aggressively unroll inner loops to expose ILP',
    'workgroup_swizzling: rearrange workgroups for better L2 and chiplet locality',
  ],
  isa_specific: [
    'hw_oob_guarded_loads: use hardware out-of-bounds guards instead of explicit branches',
    'use_agprs: select accumulator register class (AGPR) to double available vector registers',
    'instruction_scheduling: overlap MFMA, ALU, and memory instructions to hide latency',
  ],
}

// =============================================================================
// Phase 1: Setup — Read kernel, hardware specs, initialize
// =============================================================================
phase('Setup')

// --- profiling-strategist: ADDITIVE extra-evidence routing. ARGUS has no native
// profiler; this lets the planner/validator OPTIONALLY consult one (native if the
// host has one, else perf_heuristic) alongside the invariant/test loop. The agent
// only classifies the task (fuzzy op_class/size); the substrate DETERMINISTICALLY
// picks the method and stamps confidence (measured/inferred/hypothesized) -- NOT
// the agent. See _substrate/profiling/README.md. Falls back to native_profiler.
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured' }
{
  const _profilerManifest = args.backend_manifest || `${SUBSTRATE}/backends/cuda/manifest.json`
  const _pd = await agentRetry(() => agent(
    `Classify this GPU kernel's op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large) from its spec/path: ` +
    `kernel_spec="${KERNEL_SPEC}", kernel_path="${KERNEL_PATH}", computation hint from hardware target ${HARDWARE_TARGET}. Then ` +
    `Run exactly: \`${SUBSTRATE}/profiling/profiling_strategist.py resolve ` +
    `--backend-manifest ${_profilerManifest} --task <op_class> --size <size> ` +
    `--cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`.\n` +
    `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
  log(`Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}')`)
}

// --- integration-strategist: DECIDE standalone vs embedded_* at runtime. This
// REPLACES the static `EMBEDDED` (from args.integration_pattern) as the authority
// for which eval path runs: the strategist classifies whether the kernel can
// compile as a single TU and routes accordingly. ARGUS has no backend driver, so
// there is no USE_DRIVER_STANDALONE; standalone runs ARGUS's native (in-prompt)
// validation, IS_EMBEDDED runs project-native register/build/test/benchmark.
// Additive: when method==='standalone' the path below is byte-identical to before.
// See _substrate/integration/README.md and _substrate/integration/ROLLOUT.md.
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Read ${KERNEL_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. a llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
    `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_integ && _integ.method) INTEGRATION_DECISION = _integ
}
log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error('integration-strategist returned derive_adapter — provide project_root + register_script + build/test/benchmark commands')
}
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
const IS_SOL = INTEGRATION_DECISION.method === 'sol_execbench_solution'
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// If the strategist routed to an embedded path, validate the required wiring now
// (the static `EMBEDDED` arg check above only fires when integration_pattern was
// passed; the strategist can decide embedded even when it was not).
if (IS_EMBEDDED) {
  const missing = []
  if (INTEGRATION_DECISION.method === 'embedded_dispatch' && !REGISTER_SCRIPT) missing.push('register_script')
  if (!PROJECT_ROOT) missing.push('project_root (or ggml_root)')
  if (!TEST_CMD) missing.push('test_command')
  if (!BENCH_CMD) missing.push('benchmark_command')
  if (!BUILD_CMD) missing.push('build_command')
  if (missing.length) {
    throw new Error(`integration-strategist routed to ${INTEGRATION_DECISION.method} but missing: ${missing.join(', ')}`)
  }
}
// A-O1 closure: ARGUS has no native profiler/driver, so a native_profiler choice
// cannot be honored without a standalone driver path -> downgrade to perf_heuristic
// (which writes heuristic_bclass in the validate prompt so diagnose stays defined).
if (PROFILING_DECISION.method === 'native_profiler' && !BENCH_CMD) {
  log(`profiling: native_profiler but no benchmark command -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'argus-perf', rationale: 'native_profiler but no benchmark_command -> perf_heuristic' }
}

if (INPUT_MODE === 'generate_then_optimize') {
  const generated = await agentRetry(() => agent(`No kernel_path was provided. Generate and verify an initial kernel before starting ARGUS ICRL optimization.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- kernel_spec: ${KERNEL_SPEC || '(not provided)'}
- language: ${LANGUAGE}
- target_gpu: ${HARDWARE_TARGET}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- test_command: ${TEST_CMD || '(not provided)'}
- benchmark_command: ${BENCH_CMD || '(not provided)'}
- invariant_check_command: ${INVARIANT_CHECK_CMD || '(not provided)'}

# Contract
Generate ${SEED_CANDIDATES} complete candidates under ${EXP_DIR}/generated/. Run available commands using {kernel_path}/{result_path}. Return the best verified generated kernel path before ARGUS learns invariants or planner policy.`, {
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

const setupResult = await agentRetry(() => agent(`You are a GPU kernel optimization expert. Read and analyze the initial kernel implementation.

# Kernel file: ${KERNEL_PATH}
# Kernel specification: ${KERNEL_SPEC}
# Hardware target: ${HARDWARE_TARGET}
# Invariant evidence contract:
- invariant_check_command: ${INVARIANT_CHECK_CMD || '(missing)'}
- invariant_result_path: ${INVARIANT_RESULT_PATH}
- invariant_evidence mode: ${invariantEvidenceMode}
- missing_invariant_evidence means ARGUS dense feedback is unavailable; do not treat LLM-written invariants as checked violations.

# Task:
1. Read the kernel source code at ${KERNEL_PATH}
2. Identify:
   - What computation it performs (GEMM, attention, MoE, etc.)
  - Current optimization level (naive, partially optimized, etc.)
  - Tiling strategy (tile sizes, how work maps to threads/blocks)
  - Memory hierarchy usage (registers, shared memory, global)
  - Any existing data-flow patterns (how data moves through the kernel)
  - Whether executable invariant checking is available
3. Determine hardware constraints:
   - For ${HARDWARE_TARGET}: register file size, shared memory, compute units
   - Arithmetic intensity requirements (ops/byte for compute-bound)
   - Memory bandwidth and latency characteristics

Return the kernel analysis.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"kernel_analysis","note":"<computation type + current opt level + top missing optimizations, one line>"}`, {
  label: 'read-kernel',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      kernel_code: { type: 'string' },
      computation_type: { type: 'string' },
      current_opt_level: { type: 'string' },
      tiling_strategy: { type: 'string' },
      memory_usage: { type: 'string' },
      data_flow_patterns: { type: 'array', items: { type: 'string' } },
      hardware_constraints: { type: 'string' },
      missing_optimizations: { type: 'array', items: { type: 'string' } },
      estimated_peak_achievable: { type: 'string' },
    },
    required: ['kernel_code', 'computation_type', 'missing_optimizations'],
  },
}), { retries: 5 })

bestKernelCode = setupResult.kernel_code
const computationType = setupResult.computation_type

// Run baseline benchmark if available
const baselineResult = await withTurnTimeout(agentRetry(() => agent(`You are a GPU kernel validator. Run the baseline kernel to establish performance.

# Kernel: ${KERNEL_PATH}
# Test command: ${TEST_CMD || '(not provided; do not infer from project structure)'}
# Benchmark command: ${BENCH_CMD || '(not provided; do not infer from project structure)'}
# Experiment directory: ${EXP_DIR}

# Steps:
1. Create experiment directory: mkdir -p ${EXP_DIR}/baseline
2. Run correctness tests (if available): ${TEST_CMD}
3. Run performance benchmark: ${BENCH_CMD}
4. Record: throughput (TFLOPS or GB/s), latency, and any profiling data

If commands are not available, do static characterization only and mark baseline performance as unmeasured.

Return baseline metrics.`, {
  label: 'baseline-perf',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      tests_pass: { type: 'boolean' },
      throughput_tflops: { type: 'number' },
      latency_us: { type: 'number' },
      peak_theoretical_tflops: { type: 'number' },
      efficiency_pct: { type: 'number' },
      profiling_summary: { type: 'string' },
    },
    required: ['throughput_tflops'],
  },
}), { retries: 5 }), 'baseline-eval')

bestThroughput = baselineResult.throughput_tflops || 0
candidateBeam = [{ code: bestKernelCode, throughput: bestThroughput, label: 'baseline' }]

log(`Setup: ${computationType} on ${HARDWARE_TARGET} | Baseline: ${bestThroughput} TFLOPS (${baselineResult.efficiency_pct || '?'}% of peak)`)
log(`Missing optimizations: ${setupResult.missing_optimizations.join(', ')}`)

// =============================================================================
// ICRL Outer Loop — Learn planner policy across iterations
// =============================================================================

for (let outerIter = 0; outerIter < ITERATIONS; outerIter++) {
  log(`\n=== ICRL Iteration ${outerIter + 1}/${ITERATIONS} | Best: ${bestThroughput.toFixed(1)} TFLOPS | History: ${optimizationHistory.length} steps ===`)

  // ===========================================================================
  // Phase 2: Plan — ICRL planner proposes optimizations + invariants
  // ===========================================================================
  phase('Plan')

  const recentHistory = optimizationHistory.slice(-10)
  const recentViolations = invariantViolationLog.slice(-5)

  const plannerResult = await agentRetry(() => agent(`You are the ARGUS Learnable Planner (ICRL, Section 6).
Your job is to propose optimization candidates with associated data-flow invariants.

# Current Kernel (${computationType}):
\`\`\`
${bestKernelCode.substring(0, 5000)}
\`\`\`

# Hardware: ${HARDWARE_TARGET}
# Current throughput: ${bestThroughput} TFLOPS
# Kernel spec: ${KERNEL_SPEC}

# Knowledge Base — Available Optimizations:
## Global Intrusive Changes (restructure the kernel):
${KNOWLEDGE_BASE.global_intrusive.map(o => `- ${o}`).join('\n')}

## Local Source Changes (small, localized patches):
${KNOWLEDGE_BASE.local_source.map(o => `- ${o}`).join('\n')}

## ISA-Specific Optimizations (hardware intrinsics):
${KNOWLEDGE_BASE.isa_specific.map(o => `- ${o}`).join('\n')}

# Optimization History (recent):
${recentHistory.length > 0 ? recentHistory.map(h => `- ${h.optimization}: ${h.outcome} (${h.throughput_delta > 0 ? '+' : ''}${h.throughput_delta?.toFixed(2) || '?'} TFLOPS)`).join('\n') : 'No history yet.'}

# Recent Invariant Violations (failures to learn from):
${recentViolations.length > 0 ? recentViolations.map(v => `- ${v}`).join('\n') : 'None.'}

# Planner Policy (learned guidance):
${plannerPolicy || 'No policy learned yet — propose based on knowledge base and missing optimizations.'}

# Categories to explore: ${OPT_CATEGORIES.join(', ')}

# Profiling evidence routing (ADDITIVE — do not let it disturb the invariant/test loop):
Profiling-strategist selected method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'). If method==='native_profiler', you MAY consult a native profiler for bottleneck evidence alongside the invariant/test evidence. If method==='perf_heuristic', derive memory-vs-compute-bound hints from benchmark throughput and tag them evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. If method==='static', reason from source only (confidence='hypothesized'). Never fabricate profiler counters.

# Task (ARGUS Section 6 — Learnable Planner):
Generate a RANKED list of 3-5 optimization proposals. For each:
1. **Optimization**: which technique from the knowledge base
2. **Context**: WHERE and HOW to apply it to this specific kernel
3. **Data-flow invariant**: what property MUST hold after the transformation
   - Express as: "assert tag(X) == tag(Y)" style — what data elements must match where
   - Example: "After staging K tiles in shared memory, elements loaded by each warp must
     correspond to the same K-slice used in the MFMA computation"
4. **Confidence/utility score**: how likely to improve performance (0-1)
5. **Dependencies**: what other optimizations this requires or enables

Rank proposals by expected impact. Prefer optimizations that:
- Address the LARGEST remaining performance gap
- Have clear invariants that can be verified
- Build on already-successful optimizations in history

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is ICRL iteration ${outerIter}):
{"workflow":"${WORKFLOW_NAME}","phase":"Plan","ts":"<ts>","status":"done","candidate_id":"iter-${outerIter}","technique":"<top-ranked proposal optimization>","note":"<how many proposals + the largest performance gap targeted>"}`, {
    label: `plan-${outerIter}`,
    phase: 'Plan',
    schema: {
      type: 'object',
      properties: {
        proposals: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              optimization: { type: 'string' },
              category: { type: 'string' },
              context: { type: 'string' },
              invariant: { type: 'string' },
              confidence: { type: 'number' },
              dependencies: { type: 'array', items: { type: 'string' } },
              expected_impact: { type: 'string' },
            },
            required: ['optimization', 'context', 'invariant', 'confidence'],
          },
        },
      },
      required: ['proposals'],
    },
  }), { retries: 5 })

  const proposals = plannerResult.proposals || []
  log(`Planner: ${proposals.length} proposals | Top: "${proposals[0]?.optimization}" (conf: ${proposals[0]?.confidence?.toFixed(2)})`)

  // ===========================================================================
  // Phase 3: Select — Sample from proposals, resolve dependencies
  // ===========================================================================
  phase('Select')

  const selectResult = await agentRetry(() => agent(`You are the ARGUS Optimization Selector (Section 6).
The planner produced a ranked list of proposals. Your job is to select and sequence
a concrete optimization plan that resolves dependencies between coupled optimizations.

# Proposals (ranked by planner):
${proposals.map((p, i) => `${i + 1}. [${p.category}] ${p.optimization} (confidence: ${p.confidence?.toFixed(2)})
   Context: ${p.context}
   Invariant: ${p.invariant}
   Dependencies: ${JSON.stringify(p.dependencies || [])}
   Expected: ${p.expected_impact || 'unknown'}`).join('\n\n')}

# Selection Rules (ARGUS Section 6 — Optimization Selector):
1. Don't always pick top-ranked — sample from distribution to maintain exploration
2. Resolve coupling: if optimization A requires B, include both in sequence
3. Output an EXECUTABLE PLAN: ordered list of transformations to apply
4. For each step, preserve the invariants from the proposal context
5. If a global intrusive change is selected, it typically must come before
   local source changes that depend on its data layout

Return the selected optimization plan.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is ICRL iteration ${outerIter}):
{"workflow":"${WORKFLOW_NAME}","phase":"Select","ts":"<ts>","status":"done","candidate_id":"iter-${outerIter}","technique":"<the ordered optimization sequence you selected>","note":"<exploration vs top-rank choice + dependency resolution, one line>"}`, {
    label: `select-${outerIter}`,
    phase: 'Select',
    schema: {
      type: 'object',
      properties: {
        selected_plan: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              step: { type: 'number' },
              optimization: { type: 'string' },
              context: { type: 'string' },
              invariant: { type: 'string' },
              rationale: { type: 'string' },
            },
            required: ['optimization', 'context', 'invariant'],
          },
        },
        exploration_note: { type: 'string' },
      },
      required: ['selected_plan'],
    },
  }), { retries: 5 })

  const selectedPlan = selectResult.selected_plan || []
  log(`Selected: ${selectedPlan.map(s => s.optimization).join(' → ')}`)

  // ===========================================================================
  // Phase 4: Lower — Implement transformations with invariants
  // ===========================================================================
  phase('Lower')

  let currentCode = bestKernelCode
  const loweringResults = []

  for (let stepIdx = 0; stepIdx < Math.min(selectedPlan.length, INNER_STEPS); stepIdx++) {
    const step = selectedPlan[stepIdx]

    const lowerResult = await agentRetry(() => agent(`You are the ARGUS Lowering Agent (Section 6).
Implement the selected optimization transformation directly in the kernel code.

# Current Kernel Code:
\`\`\`
${currentCode.substring(0, 6000)}
\`\`\`

# Optimization to Apply: "${step.optimization}"
# Context: ${step.context}
# Required Data-Flow Invariant: ${step.invariant}

# Hardware: ${HARDWARE_TARGET}
# Computation: ${computationType}

# Lowering Rules (ARGUS Section 6 — Lowering Agent):
1. Apply rewrites at the level of tiles, layouts, and thread-level control
2. Insert or update TAG FUNCTIONS to propagate symbolic annotations:
   - Tag functions assign symbolic coordinates to tensor elements
   - They propagate through control and data flow
3. Insert TAG ASSERTIONS to encode the required invariant:
   - Conformity assertions: "assert tag(A[...]) == tag(B[...])"
     (paired operands must carry compatible tags)
   - Non-conformity assertions: "assert tag(X[...]) != tag(Y[...])"
     (concurrent producers must not write same location)
4. Retrieve relevant patterns from knowledge base as few-shot examples
5. The output must be a COMPLETE, COMPILABLE kernel

# Knowledge Base Examples for "${step.optimization}":
${KNOWLEDGE_BASE[step.category || 'global_intrusive']?.filter(k => k.includes(step.optimization.split(':')[0].toLowerCase().replace(/\s+/g, '_')))?.join('\n') || 'Use general optimization patterns.'}
${IS_EMBEDDED ? `
# EMBEDDED-DISPATCH AUTHORING REQUIREMENTS (override "standalone compilable kernel"):
${EMBEDDING_CONTRACT}

Read the reference dispatch file at ${REFERENCE_FILE} and match its dispatch
signature EXACTLY (entry-point shape, template params, launch-bounds). Emit a
COMPLETE dispatch-compatible .cuh (NOT a standalone translation unit, NO main(),
no standalone harness). transformed_code must be the full .cuh file contents.` : IS_SOL ? `

${SOL_SOLUTION_CONTRACT}` : ''}

Return the transformed kernel code with invariants.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (ICRL iteration ${outerIter}, lowering step ${stepIdx}):
{"workflow":"${WORKFLOW_NAME}","phase":"Lower","ts":"<ts>","status":"done","candidate_id":"iter-${outerIter}-step-${stepIdx}","technique":"${step.optimization}","note":"<what tag functions/assertions you added + any potential issues, one line>"}`, {
      label: `lower-${outerIter}-step${stepIdx}`,
      phase: 'Lower',
      schema: {
        type: 'object',
        properties: {
          transformed_code: { type: 'string' },
          invariants_added: { type: 'array', items: { type: 'string' } },
          transformation_description: { type: 'string' },
          potential_issues: { type: 'array', items: { type: 'string' } },
        },
        required: ['transformed_code', 'invariants_added'],
      },
    }), { retries: 5, allowNull: true })

    if (lowerResult && lowerResult.transformed_code) {
      currentCode = lowerResult.transformed_code
      loweringResults.push({
        step: step.optimization,
        invariants: lowerResult.invariants_added,
        description: lowerResult.transformation_description,
      })
    }
  }

  log(`Lowered: ${loweringResults.length} transformations applied`)

  // ===========================================================================
  // Phase 5: Validate — Invariant checking + tests + profiling
  // ===========================================================================
  phase('Validate')

  // Embedded mode: write the candidate to disk and build the ordered
  // register→build→test→benchmark→unregister plan against the project adapter.
  let embeddedPlan = null
  let candidatePath = ''
  let variantName = ''
  // Gated on the integration-strategist decision (IS_EMBEDDED), NOT the static
  // args.integration_pattern. The ICRL outer loop is already SERIAL (a `for`), so
  // each embedded eval mutates the shared project one-at-a-time (no race); no
  // separate serial for-loop is needed. Standalone is byte-identical to before.
  if (IS_EMBEDDED) {
    variantName = `argus_i${outerIter}`.replace(/[^A-Za-z0-9_]/g, '_')
    candidatePath = `${EXP_DIR}/candidates/${variantName}.cuh`
    await agentRetry(() => agent(`Write the embedded candidate file to disk verbatim (no edits, no extra files).

# Target path: ${candidatePath}
# Create parent dir first: mkdir -p ${EXP_DIR}/candidates
# File contents (write EXACTLY, this is a complete dispatch-compatible .cuh):
\`\`\`
${currentCode}
\`\`\`

Return after writing the file.`, { label: `write-candidate-${outerIter}`, phase: 'Validate' }), { retries: 5 })
    if (INTEGRATION_DECISION.method === 'embedded_dispatch') {
      embeddedPlan = __embeddedEvalPlan({
        adapter: 'python "' + REGISTER_SCRIPT + '"',
        variant: variantName,
        source: candidatePath,
        projectRoot: PROJECT_ROOT,
        params: REGISTER_PARAMS,
        buildCmd: BUILD_CMD,
        testCmd: TEST_CMD,
        benchmarkCmd: BENCH_CMD,
      })
    }
  }

  let solEvalBlock = ''
  if (IS_SOL) {
    const solVariantName = `sol_i${outerIter}`.replace(/[^A-Za-z0-9_]/g, '_')
    const solCandidatePath = `${EXP_DIR}/kernels/${solVariantName}.cu`
    const solPlan = __solExecbenchEvalPlan({
      substrateDir: SOL_SUBSTRATE_DIR,
      kernelSource: solCandidatePath,
      contractEnv: `${EXP_DIR}/contract.env`,
      solutionOut: `${EXP_DIR}/${solVariantName}.solution.json`,
      benchOut: `${EXP_DIR}/${solVariantName}.bench.jsonl`,
      solCli: SOL_CLI, taskDir: SOL_TASK_DIR, benchConfig: SOL_BENCH_CONFIG,
      seedDir: SOL_SEED_DIR, cudaVisibleDevices: SOL_CVD, ldLibraryPath: SOL_LD_LIBRARY_PATH,
    })
    solEvalBlock = `
# SOL-EXECBENCH EVALUATION (overrides the standalone steps below)
This candidate is evaluated by the sol-execbench CLI, which compiles it internally. Write the transformed_code above verbatim to ${solCandidatePath}, then run these commands IN THIS EXACT ORDER:

1. Pack:  ${solPlan.pack}
2. Run:   ${solPlan.run}
3. Parse: ${solPlan.parse}

The parse step prints one line "SPEEDUP=<geomean> STATUS=<PASS|FAIL> WORKLOADS=<passed>/<total>". Parse correctness and latency STRICTLY from that line and the run output. Do NOT fabricate numbers. Map into the schema: compiled = run produced a bench.jsonl, correct = STATUS==PASS, speedup = the SPEEDUP value (kernel_time_ms may be left unavailable — sol-execbench reports speedup_factor, not absolute ms). ${solPlan.cleanupInvariant}`
  }

  const validateResult = await withTurnTimeout(agentRetry(() => agent(`You are the ARGUS Validator Agent (Section 6).
Validate the transformed kernel through invariant checking, unit tests, and profiling.

# Transformed Kernel:
\`\`\`
${currentCode.substring(0, 6000)}
\`\`\`

# Invariants to Check:
${loweringResults.map(r => r.invariants.map(inv => `- ${inv}`).join('\n')).join('\n')}
${IS_EMBEDDED && INTEGRATION_DECISION.method === 'embedded_dispatch' ? `
# EMBEDDED-DISPATCH EVALUATION (this kernel is wired into a project; it is NOT a
# standalone translation unit). Run these adapter commands IN THIS EXACT ORDER.
# Run each as a Bash command verbatim; do not paraphrase or reorder.

1. REGISTER the variant into the project dispatch table:
   ${embeddedPlan.register}
2. CONFIRM it is registered (the variant name MUST appear in the output):
   ${embeddedPlan.list}
3. BUILD the project (the project compiles the registered variant; this REPLACES
   any standalone compile — ARGUS's compile-time invariant validation below runs
   against THIS built project, not a standalone compile of the file):
   ${embeddedPlan.build}

## Step 1: Compile-Time Invariant Validation (against the built project)
After plan.build succeeds, run the ARGUS static analysis over the candidate as it
was compiled INTO the project (not a standalone translation unit):
- Track tag propagation through assignments and shared memory accesses
- Check tag assertions at all use sites
- For any violation, produce a CONCRETE COUNTEREXAMPLE:
  "Violation at [program point]: thread [T] holds element [E] with tag [X],
   but assertion requires tag [Y]"
- A build failure is a hard correctness failure (tests_pass=false).

## Step 2: Functional Correctness
4. Run correctness against the built project:
   ${embeddedPlan.test}

## Step 3: Performance Profiling
5. Run the benchmark against the built project:
   ${embeddedPlan.benchmark}
- Compare against baseline: ${bestThroughput} TFLOPS; identify bottlenecks.
${PROFILING_DECISION.method === 'perf_heuristic' ? `- Profiling-strategist chose method='perf_heuristic' (confidence='${PROFILING_DECISION.confidence}'): derive a qualitative bottleneck from the measured throughput and write heuristic_bclass (memory_bound|compute_bound|latency_bound) so diagnose does not fall to unknown. Tag it evidence='profile_heuristic'.` : ''}

## Step 4: Mandatory cleanup (HARD REQUIREMENT — run even on failure/non-improvement)
6. ALWAYS unregister the variant, even if any step above failed:
   ${embeddedPlan.unregister}
7. CONFIRM the variant is gone (the variant name MUST NOT appear):
   ${embeddedPlan.list}
CLEANUP INVARIANT: ${embeddedPlan.cleanupInvariant}
You MUST leave the project byte-exact pristine; do not skip steps 6-7.

## Step 5: Compute Reward Signal (ARGUS ICRL reward)
Reward = f(correctness, invariant_satisfaction, performance)
- If invariants violated or build fails: negative/zero reward (dense signal).
- If tests fail: zero reward.
- If correct + improved: positive reward proportional to speedup.

Apply the SAME grounding/anti-fabrication rules as always: report ONLY metrics
actually emitted by the commands above. If a command did not run or produced no
parseable result, mark it unmeasured — do not infer or invent numbers.` : (IS_EMBEDDED && INTEGRATION_DECISION.method === 'embedded_inplace') ? `
# EMBEDDED-INPLACE EVALUATION (this candidate REPLACES the project file in place;
# a pristine backup was taken at Setup). Run IN THIS EXACT ORDER, as Bash verbatim.
# project kernel: ${KERNEL_PATH} | candidate: ${candidatePath} | pristine backup: ${ORIGINAL_BACKUP}

1. RESTORE pristine first (defensive): cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}
2. APPLY candidate in place: cp ${candidatePath} ${KERNEL_PATH}
3. BUILD the project (this REPLACES any standalone compile; ARGUS invariant
   validation below runs against THIS built project):
   ${BUILD_CMD}

## Step 1: Compile-Time Invariant Validation (against the built project)
After build succeeds, run the ARGUS static analysis over the candidate as compiled
INTO the project (not a standalone TU):
- Track tag propagation; check tag assertions at all use sites
- For any violation, produce a CONCRETE COUNTEREXAMPLE:
  "Violation at [program point]: thread [T] holds element [E] with tag [X],
   but assertion requires tag [Y]"
- A build failure is a hard correctness failure (tests_pass=false).

## Step 2: Functional Correctness
4. Run correctness against the built project: ${TEST_CMD}

## Step 3: Performance Profiling
5. Run the benchmark against the built project: ${BENCH_CMD || TEST_CMD}
- Compare against baseline: ${bestThroughput} TFLOPS; identify bottlenecks.
${PROFILING_DECISION.method === 'perf_heuristic' ? `- Profiling-strategist chose method='perf_heuristic' (confidence='${PROFILING_DECISION.confidence}'): derive a qualitative bottleneck from throughput and write heuristic_bclass (memory_bound|compute_bound|latency_bound) so diagnose does not fall to unknown. Tag it evidence='profile_heuristic'.` : ''}

## Step 4: Mandatory cleanup (HARD REQUIREMENT — run even on failure/non-improvement)
6. ALWAYS restore the pristine original, even if any step above failed:
   cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}
You MUST leave the project byte-exact pristine; do not skip step 6.

## Step 5: Compute Reward Signal (ARGUS ICRL reward)
Reward = f(correctness, invariant_satisfaction, performance)
- If invariants violated or build fails: negative/zero reward (dense signal).
- If tests fail: zero reward.
- If correct + improved: positive reward proportional to speedup.

Apply the SAME grounding/anti-fabrication rules: report ONLY metrics actually
emitted by the commands above; mark anything unmeasured rather than infer.` : `
# Validation Steps (ARGUS Section 5 + Section 6):

## Step 1: Compile-Time Invariant Validation
Simulate the ARGUS static analysis:
- Track tag propagation through assignments and shared memory accesses
- Check tag assertions at all use sites
- For any violation, produce a CONCRETE COUNTEREXAMPLE:
  "Violation at [program point]: thread [T] holds element [E] with tag [X],
   but assertion requires tag [Y]"
- This is the DENSE FEEDBACK that makes ARGUS work

## Step 2: Functional Correctness
${TEST_CMD ? `Run: ${TEST_CMD}` : 'Perform static correctness analysis:'}
- Check for race conditions (concurrent writes to same shared memory)
- Check for out-of-bounds accesses
- Verify tiling covers the full problem (no elements missed)
- Check synchronization (barriers between producers and consumers)

## Step 3: Performance Profiling
${BENCH_CMD ? `Run: ${BENCH_CMD}` : 'Estimate performance:'}
- Measure or estimate throughput (TFLOPS)
- Compare against baseline: ${bestThroughput} TFLOPS
- Identify remaining bottlenecks
${PROFILING_DECISION.method === 'perf_heuristic' ? `- Profiling-strategist chose method='perf_heuristic' (confidence='${PROFILING_DECISION.confidence}'): derive a qualitative bottleneck from throughput and write heuristic_bclass (memory_bound|compute_bound|latency_bound) so diagnose does not fall to unknown. Tag it evidence='profile_heuristic'.` : ''}

## Step 4: Compute Reward Signal (ARGUS ICRL reward)
Reward = f(correctness, invariant_satisfaction, performance)
- If invariants violated: negative process reward (dense signal for what went wrong)
- If tests fail: zero reward
- If correct + improved: positive reward proportional to speedup`}${solEvalBlock}

Return validation results.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if invariants satisfied AND tests passed, else "error"; speedup is the measured speedup number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Validate","ts":"<ts>","status":"<done|error>","candidate_id":"iter-${outerIter}","speedup":<number or null>,"technique":"<technique under test>","note":"<throughput TFLOPS; invariants satisfied? tests pass? reward; or the failure/counterexample>"}`, {
    label: `validate-${outerIter}`,
    phase: 'Validate',
    schema: {
      type: 'object',
      properties: {
        invariants_satisfied: { type: 'boolean' },
        invariant_violations: { type: 'array', items: { type: 'object', properties: { assertion: { type: 'string' }, counterexample: { type: 'string' }, program_point: { type: 'string' } } } },
        tests_pass: { type: 'boolean' },
        correctness_issues: { type: 'array', items: { type: 'string' } },
        throughput_tflops: { type: 'number' },
        speedup: { type: 'number' },
        remaining_bottlenecks: { type: 'array', items: { type: 'string' } },
        heuristic_bclass: { type: 'string' },
        reward: { type: 'number' },
        reward_breakdown: { type: 'string' },
      },
      required: ['invariants_satisfied', 'tests_pass', 'throughput_tflops', 'reward'],
    },
  }), { retries: 5 })

  // Record results
  const succeeded = validateResult.invariants_satisfied && validateResult.tests_pass
  const newThroughput = validateResult.throughput_tflops || 0

  for (const step of loweringResults) {
    optimizationHistory.push({
      optimization: step.step,
      outcome: succeeded ? (newThroughput > bestThroughput ? 'improved' : 'correct_no_gain') : 'failed',
      throughput_delta: newThroughput - bestThroughput,
      invariant_violations: validateResult.invariant_violations?.length || 0,
    })
  }

  if (validateResult.invariant_violations?.length > 0) {
    for (const v of validateResult.invariant_violations) {
      invariantViolationLog.push(`[iter${outerIter}] ${v.assertion}: ${v.counterexample} at ${v.program_point}`)
    }
  }

  // Update best if improved
  if (succeeded && newThroughput > bestThroughput) {
    bestKernelCode = currentCode
    bestThroughput = newThroughput
    candidateBeam.push({ code: currentCode, throughput: newThroughput, label: `iter${outerIter}` })
    candidateBeam.sort((a, b) => b.throughput - a.throughput)
    candidateBeam = candidateBeam.slice(0, 3)
    log(`NEW BEST: ${bestThroughput.toFixed(1)} TFLOPS (${validateResult.speedup?.toFixed(2) || '?'}x) ✓`)
  } else if (!succeeded) {
    log(`FAILED: ${validateResult.invariant_violations?.length || 0} invariant violations, ${validateResult.correctness_issues?.length || 0} correctness issues`)
  } else {
    log(`Correct but no improvement: ${newThroughput.toFixed(1)} TFLOPS`)
  }

  // ===========================================================================
  // Phase 6: Learn — ICRL policy update via text gradients
  // ===========================================================================
  phase('Learn')

  const learnResult = await agentRetry(() => agent(`You are performing the ICRL policy update for the ARGUS planner (Algorithm 1, Section 6).

# This Iteration's Results:
- Optimizations attempted: ${loweringResults.map(r => r.step).join(' → ')}
- Outcome: ${succeeded ? 'SUCCESS' : 'FAILED'}
- Throughput: ${newThroughput.toFixed(1)} TFLOPS (was ${bestThroughput.toFixed(1)})
- Reward: ${validateResult.reward}
- Invariant violations: ${JSON.stringify(validateResult.invariant_violations || [])}
- Remaining bottlenecks: ${JSON.stringify(validateResult.remaining_bottlenecks || [])}

# Full Optimization History:
${optimizationHistory.map(h => `- ${h.optimization}: ${h.outcome} (Δ${h.throughput_delta?.toFixed(2) || '?'} TFLOPS, ${h.invariant_violations} violations)`).join('\n')}

# Current Planner Policy:
${plannerPolicy || '(empty — this is the initial update)'}

# ICRL Update Rules (ARGUS Algorithm 1):
1. POLICY EVALUATION: Assess which optimization choices led to improvements vs failures
2. ANALYZE: Compute text-level gradients — what should the planner do MORE of / LESS of?
3. PARAMETER UPDATE: Produce an updated planner policy (natural language guidance)

Key insights from ARGUS ICRL:
- Process rewards from invariant violations give DENSE signal (not just pass/fail)
- Learn which invariants are effective at guiding transformations
- Learn ordering dependencies (e.g., bank conflict mitigation before pipelining)
- Bind knowledge base entries to this specific kernel instance

Produce the updated planner policy — this will guide the next iteration's proposals.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is ICRL iteration ${outerIter}):
{"workflow":"${WORKFLOW_NAME}","phase":"Learn","ts":"<ts>","status":"done","candidate_id":"iter-${outerIter}","technique":"icrl_policy_update","note":"<key learnings + effective optimizations + next priority, one line>"}`, {
    label: `learn-${outerIter}`,
    phase: 'Learn',
    schema: {
      type: 'object',
      properties: {
        policy_update: { type: 'string' },
        key_learnings: { type: 'array', items: { type: 'string' } },
        effective_optimizations: { type: 'array', items: { type: 'string' } },
        anti_patterns: { type: 'array', items: { type: 'string' } },
        next_priority: { type: 'string' },
      },
      required: ['policy_update', 'key_learnings'],
    },
  }), { retries: 5 })

  plannerPolicy = learnResult.policy_update
  log(`ICRL update: ${learnResult.key_learnings.length} learnings | Next priority: ${learnResult.next_priority || 'continue'}`)
}

// =============================================================================
// Final Report
// =============================================================================
const finalReport = await agentRetry(() => agent(`Write a concise technical report on the ARGUS optimization results.

# ARGUS Optimization Results
- Computation: ${computationType}
- Hardware: ${HARDWARE_TARGET}
- Kernel spec: ${KERNEL_SPEC}
- Baseline throughput: ${baselineResult.throughput_tflops} TFLOPS
- Final best throughput: ${bestThroughput} TFLOPS
- Overall speedup: ${bestThroughput / (baselineResult.throughput_tflops || 1)}x
- ICRL iterations: ${ITERATIONS}
- Total optimization steps: ${optimizationHistory.length}

# Optimization History:
${optimizationHistory.map((h, i) => `${i + 1}. ${h.optimization}: ${h.outcome} (Δ${h.throughput_delta?.toFixed(2)} TFLOPS)`).join('\n')}

# Invariant Violation Log (dense feedback):
${invariantViolationLog.slice(0, 10).join('\n') || 'No violations recorded.'}

# Final Planner Policy (learned):
${plannerPolicy}

# Invariant Evidence Contract:
- invariant_check_command: ${INVARIANT_CHECK_CMD || '(missing)'}
- invariant_result_path: ${INVARIANT_RESULT_PATH}
- invariant_evidence: ${invariantEvidenceMode}

# Final Kernel:
\`\`\`
${bestKernelCode.substring(0, 4000)}
\`\`\`

Write:
1. Optimization journey: which transformations worked, in what order, and why
2. Role of data-flow invariants: how did violation feedback guide fixes?
3. ICRL learning: how did the planner policy evolve?
4. Remaining gap to peak theoretical performance
5. Recommendations for further optimization`, {
  label: 'final-report',
  phase: 'Learn',
}), { retries: 5 }), 'validate')

// embedded_inplace exit safety net: unconditionally restore the pristine original
// so the project is left byte-exact regardless of how the loop terminated.
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${KERNEL_PATH}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Learn', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  computation_type: computationType,
  target_gpu: HARDWARE_TARGET,
  baseline_throughput_tflops: baselineResult.throughput_tflops || 0,
  best_throughput_tflops: bestThroughput,
  overall_speedup: bestThroughput / (baselineResult.throughput_tflops || 1),
  iterations_completed: ITERATIONS,
  optimization_steps: optimizationHistory.length,
  optimization_history: optimizationHistory,
  invariant_violations_total: invariantViolationLog.length,
  planner_policy_final: plannerPolicy,
  candidate_beam: candidateBeam.map(c => ({ label: c.label, throughput: c.throughput })),
  best_kernel_code: bestKernelCode,
  invariant_check_command: INVARIANT_CHECK_CMD,
  invariant_result_path: INVARIANT_RESULT_PATH,
  invariant_evidence: invariantEvidenceMode,
  report: finalReport,
}
