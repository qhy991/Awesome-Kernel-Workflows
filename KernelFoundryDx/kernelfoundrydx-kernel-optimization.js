export const meta = {
  name: 'kernelfoundrydx-kernel-optimization',
  description: 'Diagnosis-driven multi-island evolutionary Triton kernel optimizer with expert-guided RAG initialization, a centralized hint/experience library, and anti-cheating validation (Kernel Foundry / Multi-Experts methodology, arXiv:2605.30359)',
  whenToUse: 'When you want to evolve a Triton kernel from a PyTorch reference using a population-based search rather than a single optimization path. Kernel Foundry seeds correct initial kernels with an "expert" model guided by RAG few-shots, then runs several role-specialized evolutionary islands (e.g. fusion / memory-access / parameter-tuning). Every candidate is compiled+run on real hardware (no heavy profiling), DIAGNOSED (failure mode for incorrect kernels; memory/latency/instruction-bound class for correct ones), and refined with natural-language hints drawn from a shared experience library whose entries are reinforced or down-weighted by their measured speedup. Anti-cheating validation ensures speedups come from genuine kernel work. Use for KernelBench-style operator/fusion tasks on NVIDIA GPUs.',
  phases: [
    { title: 'Setup', detail: 'Read the PyTorch reference task, establish the eager baseline, seed the experience library' },
    { title: 'Init', detail: 'Expert-guided RAG initialization: retrieve similar verified PyTorch->Triton pairs, generate correct seeds, anti-cheating validation' },
    { title: 'Evolve', detail: 'Per island, per iteration: LLM mutation conditioned on island role + retrieved hints + history' },
    { title: 'Evaluate', detail: 'Compile + run each variant on real hardware, measure correctness and speedup (lightweight signals only)' },
    { title: 'Diagnose', detail: 'Result Analyzer: classify failure mode (incorrect) or performance limiter (correct), generate hints, anti-cheating check' },
    { title: 'Evolve-Pop', detail: 'Update island populations + elite archives; probabilistic elite migration on stagnation; reinforce/down-weight hints' },
    { title: 'Report', detail: 'Best valid kernel across all islands, hint library evolution, island trajectories' },
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

const WORKFLOW_NAME = 'kernelfoundrydx-kernel-optimization'


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
// The hard "do not re-propose" constraint is owned by the cumulative transfer
// object, where a failed_strategy stops applying only when a later
// validated_win supersedes it. Deriving it from the previous round alone drops
// a strategy that failed in round 1 and simply was not retried in round 2.
// KerSor emits the cumulative ids as `failed_strategy_ids`; the per-round
// derivation stays as the fallback for a dispatch that predates that channel.
const FAILED_STRATEGY_IDS = Array.isArray(args.failed_strategy_ids)
  ? args.failed_strategy_ids.filter(id => typeof id === 'string' && id)
  : ((ATTEMPT_EVIDENCE && Array.isArray(ATTEMPT_EVIDENCE.transfer_items))
    ? ATTEMPT_EVIDENCE.transfer_items.filter(i => i && i.kind === 'failed_strategy' && i.id).map(i => i.id)
    : [])
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


// --- BEGIN inlined backend-axis (resolve) scaffolding (from _meta/scaffolding/backend-axis.js) ---
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
// --- END inlined backend-axis (resolve) scaffolding ---
// triton-only: any resolved backend axis must be triton (supported=[triton]).
// Workflow-specific guard, kept OUTSIDE the backend-axis SSOT block so the
// canonical resolveBackendAxis stays byte-identical to _meta/scaffolding/backend-axis.js.
if (RESOLVED_BACKEND && RESOLVED_BACKEND !== 'triton') {
  throw new Error(`${WORKFLOW_NAME} supports only backend="triton" (got "${RESOLVED_BACKEND}"). This workflow is the triton-only "Dx" variant.`)
}


// =============================================================================
// Kernel Foundry: A Diagnosis-driven Evolutionary Kernel Optimizer with
//                 Multi-Experts  (arXiv:2605.30359)
// =============================================================================
//
// Source: Huang, Chen, Huang, Yin, Li, Zhen, Yuan, Shao
//         "Kernel Foundry: A Diagnosis-driven Evolutionary Kernel Optimizer
//          with Multi-Experts", arXiv:2605.30359 (CUHK + Huawei Noah's Ark Lab).
//         No public code repo at preprint time — this workflow is a faithful
//         adaptation of the method as described in the paper (incl. App. A/B/C).
//
// NOTE: This is DISTINCT from the existing KernelFoundry/ in this repo
//       (arXiv:2603.12440, Intel, SYCL, MAP-Elites + meta-prompt co-evolution).
//       That one is hardware-aware QD search on SYCL; THIS one is a
//       diagnosis-driven multi-island evolutionary optimizer on Triton.
//       Directory suffix "Dx" = Diagnosis-driven.
//
// Method summary (faithful to the paper):
//
//   1. EXPERT INIT   PyTorch input -> retrieve top-k similar verified
//                    PyTorch->Triton pairs from a distilled corpus (RAG) ->
//                    "expert" model generates candidate Triton seeds with the
//                    retrieved few-shots -> anti-cheating validation (discard if
//                    cheating likelihood > 50%). Correct seeds populate islands.
//   2. EVOLUTION     Multiple role-specialized ISLANDS run in parallel. Each
//                    island keeps an independent population + local elite archive.
//                    Per iteration, per island:
//                      a. MUTATE   the optimizer model generates a new variant
//                                  conditioned on a parent candidate + the island's
//                                  role-specialized system prompt + retrieved hints
//                                  + the island's evolution history.
//                      b. EVALUATE compile + run on real hardware; collect
//                                  lightweight signals only (compile status,
//                                  runtime/speedup, launch config) — NO ncu/nsys.
//                      c. DIAGNOSE Result Analyzer:
//                                   - incorrect -> classify failure mode
//                                     (compile error / runtime fail / mismatch)
//                                     -> targeted correctness hint.
//                                   - correct   -> classify dominant limiter
//                                     (memory_bound / latency_bound / instruction_bound)
//                                     from runtime stats + config -> perf hint.
//                                  Anti-cheating validation re-applied.
//                      d. UPDATE   insert into island population/archive; reinforce
//                                  hints correlated with speedup, down-weight others.
//                    On STAGNATION, elites from other islands are probabilistically
//                    migrated in as parents/seeds.
//   3. SELECT        Output = best valid kernel across the UNION of all islands.
//
// MULTI-EXPERTS (note: NOT a gated mixture-of-experts):
//   * Expert model (small, domain-specialized) drives INITIALIZATION via RAG.
//   * A separate optimizer model (large) drives EVOLUTION.
//   * Role-specialized ISLANDS: each island gets a different system prompt + a
//     different hint subset, focusing on a distinct optimization perspective.
//
// EXPERIENCE / HINT LIBRARY (the diagnosis-driven core):
//   Centralized, shared across islands. Expert-initialized from NVIDIA best
//   practices, then refined online. Entries grouped by diagnosed error type /
//   bottleneck class. Each entry = {trigger (error types / bottleneck class),
//   context descriptors, optimization suggestion, confidence stats (success
//   frequency, avg speedup)}. Can be OVERRIDDEN / PERSISTED via hint_library_path.
//
// Feedback signal: measured SPEEDUP vs PyTorch (higher = better) + correctness,
//   from compile + execute on real GPU. Explicitly NOT profiler-based; the
//   bottleneck class is INFERRED from runtime stats + config (coarse by design).
//
// Usage:
//   Workflow({name: 'kernelfoundrydx-kernel-optimization', args: {
//     problem_path: '/path/to/KernelBench/level2/95_Matmul_Add_Swish.py',
//     op_description: 'Matmul + Add + Swish + Tanh + GELU + Hardtanh fusion',
//     target_gpu: 'RTX 5090',
//     islands: 3,
//     iterations: 6,
//     population_size: 4,
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     hint_library_path: '/path/to/hint_library.json',  // optional, persists across runs
//     rag_corpus_path: '/path/to/kernelbook_pairs/',     // optional retrieval corpus
//     exp_dir: '/tmp/kernelfoundrydx_exp',
//     run_timestamp_iso: '2026-06-02T17:00:00+08:00',
//   }})
//
// =============================================================================

// --- Required Args ---
const TASK_PATH = args.problem_path
const PROBLEM_DEFINITION = args.problem_definition || ''

// --- Optional Args ---
const OP_DESC = args.op_description || 'PyTorch operator (to Triton kernel)'
const TARGET_GPU = args.target_gpu || 'RTX 5090'
const ISLANDS = args.islands || 3
// AWK #60: a mandatory directive (e.g. "Cody-Waite exp2 FMA polynomial on the
// softmax axis") dispatched by KerSor. When set, one island per iteration is
// pinned to it as a REQUIRED mutation objective + a cheap textual acceptance
// check, so a high-leverage named move can no longer be ignored for 8 rounds.
const MANDATORY_DIRECTIVE = args.mandatory_directive || ''
const ITERATIONS = args.iterations || 6
const POPULATION_SIZE = args.population_size || 4
const SEED_CANDIDATES = args.seed_candidates || 3
const RAG_TOPK = args.rag_topk || 5
const STAGNATION_WINDOW = args.stagnation_window || 2
const CHEAT_THRESHOLD = args.cheat_threshold || 0.5  // discard if cheating likelihood > this
const BENCH_CMD = args.benchmark_command || ''
const TEST_CMD = args.test_command || ''
const HINT_LIBRARY_PATH = args.hint_library_path || ''
const RAG_CORPUS_PATH = args.rag_corpus_path || ''
const EXP_DIR = args.exp_dir || '/tmp/kernelfoundrydx_exp'
const RUN_TS = args.run_timestamp_iso || 'unknown'
const INPUT_MODE = 'generate_then_optimize'

// --- Project-native integration (embedded operators via integration-strategist) ---
// For inference-engine embedded operators (e.g. llama.cpp .cuh) the candidate cannot
// compile as a standalone TU; it is built/tested inside the host project. BENCH_CMD
// already exists for the standalone benchmark, so the project-native benchmark uses a
// distinct name (PROJECT_BENCH_CMD) to avoid a duplicate `const` (breaks wfcheck).
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const BUILD_CMD = args.build_command || ''
const PROJECT_BENCH_CMD = args.project_benchmark_command || BENCH_CMD || ''
const REGISTER_SCRIPT = args.register_script || ''

// --- Backend driver wiring (P5d Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_SETUP_LANG_TOKEN = 'Triton'
const LEGACY_SEED_LANG_TOKEN = 'Triton'
const LEGACY_VALIDATE_LANG_TOKEN = 'Triton'
const LEGACY_MUTATE_LANG_TOKEN = 'Triton'
const LEGACY_EVAL_LANG_TOKEN = 'Triton'
const LEGACY_KERNEL_FILENAME = 'kernel.py'
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

// --- BEGIN inlined backend-axis (driver) scaffolding (from _meta/scaffolding/backend-axis.js) ---
function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}
// --- END inlined backend-axis (driver) scaffolding ---

let DRIVER = null
let DRIVER_LANG_FENCE = 'triton'
let DRIVER_IMPL_REQUIREMENTS = ''
let DRIVER_SOURCE_EXT = '.py'
let DRIVER_BACKEND_ID = RESOLVED_BACKEND || ''

function langToken(legacy) {
  return USE_DRIVER ? DRIVER_LANG_FENCE : legacy
}
function kernelFilename() {
  return USE_DRIVER ? `kernel${DRIVER_SOURCE_EXT}` : LEGACY_KERNEL_FILENAME
}

// AWK #58/#59: per-candidate variant path (ABSOLUTE — EXP_DIR is outside any
// worktree), so the candidate source persists after teardown. This path is the
// single source of truth inside the workflow; `code` is display/compat only.
function kfdxVariantPath(iter, islIdx) {
  return `${EXP_DIR}/variants/iter${iter}-isl${islIdx}/${kernelFilename()}`
}

// --- Island roles (role-specialized system prompts; the "multi-experts" axis) ---
// Each island focuses on a distinct optimization perspective. Cycled if ISLANDS > roles.
const ISLAND_ROLES = [
  {
    name: 'operator_fusion',
    focus: 'Fuse adjacent elementwise / reduction operations into a single Triton kernel to eliminate intermediate global-memory traffic and kernel-launch overhead. Prefer fusing producer-consumer chains and epilogues into the main compute kernel.',
  },
  {
    name: 'memory_access',
    focus: 'Optimize memory access: coalesced/vectorized tl.load/tl.store, block/tile sizing for L2 reuse, avoiding redundant global reads, masking only at boundaries, contiguous layouts.',
  },
  {
    name: 'kernel_parameter_tuning',
    focus: 'Tune launch and tiling parameters: BLOCK_SIZE, num_warps, num_stages, grid mapping, autotune configs. Restructure the program around the most parallel decomposition.',
  },
  {
    name: 'compute_restructuring',
    focus: 'Restructure compute: reduce instruction count, use fast-math/approximate intrinsics where tolerance allows, hoist invariants, exploit tl.dot / accumulation patterns, minimize divergence.',
  },
]

// --- State ---
let hintLibrary = []          // [{id, trigger, bottleneck_class, context, suggestion, success_count, use_count, avg_speedup}]
let islands = []              // [{role, population:[{code, speedup, correct, diagnosis}], archive:[], bestSpeedup, stagnation}]
let baselineLatency = null
let bestKernel = null         // {code, speedup, correct, island, iteration}
let bestSpeedup = null
let evolutionTrajectory = []  // log of {iteration, island, speedup, correct, limiter, hint_ids}
let totalEvaluated = 0

// Helper: pick role for island index
function roleFor(idx) {
  return ISLAND_ROLES[idx % ISLAND_ROLES.length]
}

// Helper: format hint library subset relevant to a role / bottleneck for a prompt
function buildHintSection(role, bottleneckClass) {
  if (hintLibrary.length === 0) return '(experience library is empty — generate from first principles and NVIDIA best practices)'
  // Prefer hints matching this role's bottleneck affinity or a given class, then high-confidence ones
  const scored = hintLibrary.map(h => {
    let s = (h.avg_speedup || 1.0) * (h.use_count > 0 ? (h.success_count / h.use_count) : 0.5)
    if (bottleneckClass && h.bottleneck_class === bottleneckClass) s += 1.0
    return { h, s }
  })
  scored.sort((a, b) => b.s - a.s)
  const top = scored.slice(0, 6).map(({ h }, i) =>
    `${i + 1}. [${h.bottleneck_class || h.trigger || 'general'}] ${h.suggestion}  (used ${h.use_count || 0}x, avg ${Number(h.avg_speedup || 1).toFixed(2)}x)`
  )
  return top.join('\n')
}

// Helper: reinforce or down-weight a hint after observing a candidate's speedup
function updateHint(hintId, speedup, correct) {
  const h = hintLibrary.find(x => x.id === hintId)
  if (!h) return
  h.use_count = (h.use_count || 0) + 1
  const success = correct && speedup > 1.0
  if (success) h.success_count = (h.success_count || 0) + 1
  // Running average of speedup observed when this hint was applied to a correct kernel
  if (correct) {
    const prevAvg = h.avg_speedup || 1.0
    const n = h.success_count || 1
    h.avg_speedup = prevAvg + (speedup - prevAvg) / Math.max(1, n)
  }
}

// =============================================================================
// Phase: Setup — Read the reference task, baseline, seed the experience library
// =============================================================================
phase('Setup')

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
  if (DRIVER.backend_id && normalizeSuitabilityValue(DRIVER.backend_id) !== 'triton') {
    throw new Error(`backend_dir manifest backend_id="${DRIVER.backend_id}" conflicts with KernelFoundryDx supported=[triton] (this workflow is triton-only).`)
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

const setup = await agentRetry(() => agent(`Read the PyTorch reference task file at: ${TASK_PATH || '(not provided)'}
If problem_definition is provided, use it as the authoritative task description:
${PROBLEM_DEFINITION || '(not provided)'}

This file defines a reference computation (a "Model" with a forward()) that we must reproduce as a high-performance Triton kernel.

Analyze it and return JSON:
- task_text: the relevant PyTorch source (Model.forward + get_inputs / init_inputs if present)
- op_type: short operation type label (e.g. "matmul_fusion", "elementwise", "reduction", "conv", "norm")
- op_chain: ordered list of the operations performed in forward()
- input_shapes: the input tensor shapes (from get_inputs / init_inputs if available)
- fusion_opportunities: list of producer-consumer chains that could be fused into one kernel
- numerical_notes: anything affecting numerical tolerance (softmax, exp, large reductions, etc.)

Return ONLY the JSON object.`, {
  label: 'read-task',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      task_text: { type: 'string' },
      op_type: { type: 'string' },
      op_chain: { type: 'array', items: { type: 'string' } },
      input_shapes: { type: 'string' },
      fusion_opportunities: { type: 'array', items: { type: 'string' } },
      numerical_notes: { type: 'string' },
    },
    required: ['task_text', 'op_type', 'op_chain'],
  },
}), { retries: 5 })

const taskText = setup.task_text
const opType = setup.op_type
log(`Task: ${opType} | chain: ${(setup.op_chain || []).join(' -> ')}`)

// Establish eager baseline + seed the experience/hint library (from NVIDIA best practices)
const baseline = await agentRetry(() => agent(`You are setting up a ${langToken(LEGACY_SETUP_LANG_TOKEN)} kernel optimization run for a PyTorch reference task.

# Operation: ${OP_DESC} (${opType})
# Op chain: ${(setup.op_chain || []).join(' -> ')}
# Target GPU: ${TARGET_GPU}

# Step 1: Establish the PyTorch-eager baseline latency.
${BENCH_CMD ? `Run the reference under eager mode using: ${BENCH_CMD}` : '(no benchmark_command provided; baseline latency is unavailable as measured evidence)'}
${TEST_CMD ? `Correctness reference command: ${TEST_CMD}` : ''}
Experiment dir: ${EXP_DIR}

# Step 2: Seed the experience/hint library.
${HINT_LIBRARY_PATH
    ? `If a hint library exists at ${HINT_LIBRARY_PATH}, load it (JSON array of hint entries) and return its entries as initial_hints. Otherwise seed from NVIDIA best practices below.`
    : 'No persistent hint library path was given — seed from NVIDIA best practices.'}

Seed 6-10 GENERAL Triton optimization hints, each grouped by the bottleneck class it addresses. Bottleneck classes are exactly: "correctness", "memory_bound", "latency_bound", "instruction_bound".
Each hint entry must have: trigger (when it applies), bottleneck_class (one of the four), context (short descriptor), suggestion (the actionable optimization in one sentence).

Return JSON.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"baseline_and_hint_seed","speedup":null,"note":"<eager baseline ms + number of hints seeded + op chain, one line>"}`, {
  label: 'baseline-and-seed',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      baseline_latency_ms: { type: 'number' },
      baseline_available: { type: 'boolean' },
      initial_hints: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            trigger: { type: 'string' },
            bottleneck_class: { type: 'string' },
            context: { type: 'string' },
            suggestion: { type: 'string' },
          },
          required: ['bottleneck_class', 'suggestion'],
        },
      },
      notes: { type: 'string' },
    },
    required: ['baseline_latency_ms', 'initial_hints'],
  },
}), { retries: 5 })

baselineLatency = baseline.baseline_latency_ms
hintLibrary = (baseline.initial_hints || []).map((h, i) => ({
  id: `hint_seed_${i}`,
  trigger: h.trigger || '',
  bottleneck_class: h.bottleneck_class || 'general',
  context: h.context || '',
  suggestion: h.suggestion,
  success_count: 0,
  use_count: 0,
  avg_speedup: 1.0,
}))
log(`Baseline: ${baselineLatency}ms | seeded ${hintLibrary.length} hints`)

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it in the Evaluate driver loop below. The agent only classifies the task
// (fuzzy, from op_type/op_chain); the substrate stamps confidence by method
// (measured/inferred/hypothesized) -- not the agent. See
// _substrate/profiling/README.md. Falls back to native_profiler if undecided. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }
if (USE_DRIVER) {
  const _pd = await agentRetry(() => agent(
    `Classify the task's op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large) from: op_type="${opType}", op_chain=[${(setup.op_chain || []).join(', ')}], input_shapes="${setup.input_shapes || 'n/a'}". Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${driverPath('manifest.json')} --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`.\n` +
    `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}

// --- integration-strategist: route build/test mode (standalone vs embedded_*).
// For an inference-engine embedded operator (e.g. llama.cpp .cuh referenced via
// TASK_PATH), can_compile_standalone=no, so the candidate is built/tested INSIDE the
// host project rather than as an isolated TU. Standalone path stays byte-identical. ---
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _kernelFile = TASK_PATH || '(not provided)'
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Read ${_kernelFile}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${_kernelFile}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
    `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_integ && _integ.method) INTEGRATION_DECISION = _integ
}
log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error('integration-strategist returned derive_adapter — provide project_root + build/test commands')
}
const USE_DRIVER_STANDALONE = USE_DRIVER && INTEGRATION_DECISION.method === 'standalone'
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
// The embedded operator file we swap in place is the project-referenced TASK_PATH.
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${TASK_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// A-O1 closure: native_profiler chosen but no project-native profiler is reachable
// under the embedded path → downgrade to perf_heuristic (run.sh/bench gives throughput).
if (PROFILING_DECISION.method === 'native_profiler' && IS_EMBEDDED && !USE_DRIVER_STANDALONE) {
  log(`profiling: native_profiler but embedded path -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'project-native-perf', rationale: 'native_profiler unreachable on embedded path -> perf_heuristic' }
}

// =============================================================================
// Phase: Init — Expert-guided RAG initialization + anti-cheating validation
// =============================================================================
phase('Init')

const seeds = await parallel(
  Array.from({ length: SEED_CANDIDATES }, (_, i) => () =>
    agentRetry(() => agent(`You are an EXPERT GPU-programming model specializing in PyTorch -> ${langToken(LEGACY_SEED_LANG_TOKEN)} translation. Produce a CORRECT initial ${langToken(LEGACY_SEED_LANG_TOKEN)} kernel for this task. Correctness is the ONLY goal here — not speed.

# Operation: ${OP_DESC} (${opType})
# Op chain: ${(setup.op_chain || []).join(' -> ')}
# Input shapes: ${setup.input_shapes || 'see task'}
# Target GPU: ${TARGET_GPU}

# PyTorch reference:
\`\`\`python
${taskText.substring(0, 4000)}
\`\`\`

# RAG few-shots (retrieval-augmented initialization):
${RAG_CORPUS_PATH
        ? `Retrieve the top-${RAG_TOPK} most similar VERIFIED PyTorch->Triton pairs from the corpus at ${RAG_CORPUS_PATH} (e.g. via embedding similarity on the op structure). Use them as worked examples of correct, idiomatic Triton for similar ops.`
        : `No retrieval corpus path given. Recall idiomatic, VERIFIED Triton patterns for "${opType}" from first principles (correct masking, contiguous indexing, proper grid, @triton.jit, tl.load/tl.store).`}

# Requirements:
1. Output a COMPLETE runnable module: imports, the @triton.jit kernel(s), the Python wrapper that launches them, and a ModelNew class matching the reference forward() signature.
2. All compute must be in Triton (tl.* ops); do NOT call torch ops to do the actual math (that would be cheating).
3. Must be numerically equivalent to the reference within tolerance.
4. This is seed variant ${i + 1}/${SEED_CANDIDATES}; vary the decomposition slightly from a naive baseline.

Return JSON with the code and a short note on the approach.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Init","ts":"<ts>","status":"done","candidate_id":"seed-${i}","technique":"<your initialization decomposition / RAG-guided approach>","speedup":null,"note":"<the seed approach in one line>"}`, {
      label: `seed-${i}`,
      phase: 'Init',
      schema: {
        type: 'object',
        properties: {
          code: { type: 'string' },
          approach: { type: 'string' },
          retrieved_examples: { type: 'array', items: { type: 'string' } },
        },
        required: ['code'],
      },
    }), { retries: 5 })
  )
)

const seedCandidates = seeds.filter(Boolean).filter(s => s.code)
log(`Generated ${seedCandidates.length} expert seeds`)

// Anti-cheating validation of seeds (prompt-level + LLM validator)
const validatedSeeds = await parallel(
  seedCandidates.map((s, i) => () =>
    agentRetry(() => agent(`You are an anti-cheating validator for ${langToken(LEGACY_VALIDATE_LANG_TOKEN)} kernels. Determine whether this generated kernel does GENUINE kernel-level work or CHEATS (e.g. calls torch ops to do the math, returns precomputed/reference values, or omits the real computation).

# Reference op chain: ${(setup.op_chain || []).join(' -> ')}

# Candidate kernel:
\`\`\`python
${s.code.substring(0, 4000)}
\`\`\`

# Checks:
1. Is there at least one real @triton.jit kernel with tl.load / tl.store and grid indexing?
2. Is the actual math performed inside Triton (not delegated to torch.matmul / torch.nn.functional / etc.)?
3. Are all operations in the reference chain actually computed (none silently skipped)?
4. Does it avoid returning constants / reference outputs directly?

Estimate cheating_likelihood in [0,1] (1 = definitely cheating). Also do a quick static read for obvious correctness/compile risks.

Return JSON.`, {
      label: `validate-seed-${i}`,
      phase: 'Init',
      schema: {
        type: 'object',
        properties: {
          cheating_likelihood: { type: 'number' },
          is_genuine_triton: { type: 'boolean' },
          missing_ops: { type: 'array', items: { type: 'string' } },
          static_risk_notes: { type: 'string' },
        },
        required: ['cheating_likelihood', 'is_genuine_triton'],
      },
    }), { retries: 5 })
  )
)

const cleanSeeds = []
for (let i = 0; i < seedCandidates.length; i++) {
  const v = validatedSeeds[i]
  if (!v) continue
  if ((v.cheating_likelihood || 0) > CHEAT_THRESHOLD) {
    log(`Seed ${i} discarded (cheating likelihood ${v.cheating_likelihood})`)
    continue
  }
  cleanSeeds.push({ code: seedCandidates[i].code, approach: seedCandidates[i].approach })
}
log(`${cleanSeeds.length}/${seedCandidates.length} seeds passed anti-cheating`)

// Populate islands from clean seeds (round-robin), each with its role
const seedPool = cleanSeeds.length > 0 ? cleanSeeds : seedCandidates.map(s => ({ code: s.code, approach: s.approach }))
for (let isl = 0; isl < ISLANDS; isl++) {
  const role = roleFor(isl)
  const seed = seedPool[isl % seedPool.length]
  islands.push({
    role,
    population: seed ? [{ code: seed.code, speedup: null, correct: null, diagnosis: null }] : [],
    archive: [],
    bestSpeedup: null,
    stagnation: 0,
  })
}
log(`Initialized ${islands.length} islands: ${islands.map(i => i.role.name).join(', ')}`)

// AWK #60 fix 3: seed the hint library from the mandatory directive so the
// bottleneck_class-keyed retrieval can surface it to non-pinned islands too.
if (MANDATORY_DIRECTIVE) {
  hintLibrary.push({
    id: 'mandatory-directive',
    trigger: 'dispatch directive',
    bottleneck_class: '',
    context: OP_DESC,
    suggestion: MANDATORY_DIRECTIVE,
    success_count: 0, use_count: 0, avg_speedup: 1.0,
  })
  log(`Seeded hint library with mandatory directive: "${MANDATORY_DIRECTIVE.substring(0, 80)}"`)
}

// =============================================================================
// Multi-Island Diagnosis-Driven Evolution Loop
// =============================================================================

for (let iter = 0; iter < ITERATIONS; iter++) {
  log(`\n=== Iteration ${iter + 1}/${ITERATIONS} | Best: ${bestSpeedup ? bestSpeedup.toFixed(2) + 'x' : 'N/A'} | Evaluated: ${totalEvaluated} ===`)

  // AWK #60 fix 1: pin one island per iteration to the mandatory directive (rotating
  // so every island role sees it over time). That island's mutation prompt gets the
  // directive verbatim as a REQUIRED objective, not diluted inside op_description.
  const pinnedIsl = MANDATORY_DIRECTIVE ? (iter % islands.length) : -1
  if (pinnedIsl >= 0) log(`Iteration ${iter + 1}: directive-pinned island = ${islands[pinnedIsl].role.name} (AWK #60)`)

  // ===========================================================================
  // Phase: Evolve — each island mutates a parent (parallel across islands)
  // ===========================================================================
  phase('Evolve')

  const mutations = await parallel(
    islands.map((island, islIdx) => () => {
      // Choose a parent: best in population, else first seed
      const parent = island.population.length > 0
        ? [...island.population].sort((a, b) => (b.speedup || 0) - (a.speedup || 0))[0]
        : null
      const parentCode = parent ? parent.code : (seedPool[0] ? seedPool[0].code : '')
      const parentPath = parent && parent.variant_path ? parent.variant_path : null  // AWK #60 fix 4: full parent view via path when available
      const parentDiag = parent && parent.diagnosis ? parent.diagnosis : null
      const bottleneckClass = parentDiag ? parentDiag.limiter : null
      const hintSection = buildHintSection(island.role, bottleneckClass)
      const variantPath = kfdxVariantPath(iter, islIdx)  // AWK #58/#59
      // History context for this island
      const histLines = island.archive.slice(-3).map((a, k) =>
        `  - gen ${k}: ${a.correct ? 'correct' : 'INCORRECT'}, ${a.speedup ? a.speedup.toFixed(2) + 'x' : 'n/a'}${a.diagnosis ? ', ' + (a.diagnosis.limiter || a.diagnosis.failure_mode || '') : ''}`
      ).join('\n') || '  (no history yet)'

      return agentRetry(() => agent(`You are the OPTIMIZER model evolving a ${langToken(LEGACY_MUTATE_LANG_TOKEN)} kernel. You belong to a role-specialized evolutionary island.

# ISLAND ROLE: ${island.role.name}
${island.role.focus}

# Operation: ${OP_DESC} (${opType})
# Target GPU: ${TARGET_GPU}
# Iteration: ${iter + 1}/${ITERATIONS}

# Parent kernel (your starting point — produce a MUTATION/improvement of this):
${parentPath ? `\n# Full parent kernel (Read for complete context — the snippet below is orientation only, AWK #60 fix 4/#61): ${parentPath}\n` : ''}\`\`\`python
${parentCode.substring(0, 4000)}
\`\`\`

${parentDiag ? `# Diagnosis of the parent:\n- ${parent.correct === false ? 'INCORRECT: ' + (parentDiag.failure_mode || 'unknown failure') : 'limiter: ' + (parentDiag.limiter || 'unknown')}\n- ${parentDiag.rationale || ''}` : '# Parent has not been diagnosed yet.'}

# Island evolution history (recent):
${histLines}

# Relevant hints from the shared experience library (apply those that fit your role + the diagnosis):
${hintSection}

${(islIdx === pinnedIsl) ? `# REQUIRED MUTATION OBJECTIVE (AWK #60 — this island is directive-pinned for iteration ${iter + 1}):
You MUST make this mutation implement the following mandatory directive verbatim — it is the PRIMARY objective for this candidate, not optional context:
"${MANDATORY_DIRECTIVE}"
Your change_summary MUST reference this directive (name the technique you applied). A candidate that ignores the directive will be rejected before eval.

` : ''}# Your task:
Generate ONE new Triton kernel variant that improves on the parent FROM YOUR ISLAND'S PERSPECTIVE (${island.role.name}). Stay correct. All math must remain in Triton (no torch-op cheating).
Cite which hint(s) you applied (by their text) so we can track hint usefulness.
PERSIST (AWK #58/#59): Write the COMPLETE kernel to ${variantPath} (absolute path — exp_dir is outside your worktree; the file survives teardown and is the single source of truth for eval/archive). Return variant_path = this path; the \`code\` field is a display/compat payload only.

Return JSON with variant_path, the complete code, the applied hint(s), and a one-line summary of the change.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Evolve","ts":"<ts>","status":"done","candidate_id":"iter${iter}-isl${islIdx}-${island.role.name}","technique":"<the mutation you applied from your island role + hints>","speedup":null,"note":"<what changed vs the parent, one line>"}`, {
        label: `mutate-${iter}-isl${islIdx}`,
        phase: 'Evolve',
        schema: {
          type: 'object',
          properties: {
            variant_path: { type: 'string' },
            code: { type: 'string' },
            applied_hints: { type: 'array', items: { type: 'string' } },
            change_summary: { type: 'string' },
          },
          required: ['variant_path', 'code'],
        },
      }), { retries: 5 })
    })
  )

  // ===========================================================================
  // Phase: Evaluate — compile + run on real hardware (lightweight signals)
  // ===========================================================================
  phase('Evaluate')

  const variants = []
  for (let islIdx = 0; islIdx < islands.length; islIdx++) {
    const m = mutations[islIdx]
    if (!m || !m.code) continue
    // AWK #60 fix 2: acceptance check — a directive-pinned island's change_summary
    // must reference the directive (cheap textual gate). Reject before eval.
    if (islIdx === pinnedIsl && MANDATORY_DIRECTIVE) {
      const summary = (m.change_summary || '').toLowerCase()
      const directiveWords = MANDATORY_DIRECTIVE.toLowerCase()
        .split(/[^a-z0-9-]+/).filter(w => w.length > 4)
      const referenced = directiveWords.some(w => summary.includes(w))
      if (!referenced) {
        log(`[iter ${iter + 1}] REJECTED directive-pinned island ${islands[islIdx].role.name}: change_summary does not reference the mandatory directive (AWK #60).`)
        islands[islIdx].archive.push({ correct: false, speedup: null, diagnosis: { failure_mode: 'directive_ignored', rationale: `change_summary did not reference: ${MANDATORY_DIRECTIVE.substring(0, 80)}` } })
        continue
      }
    }
    variants.push({ islIdx, code: m.code, variant_path: m.variant_path, applied_hints: m.applied_hints || [], change_summary: m.change_summary || '' })
  }

  const evals = await parallel(
    variants.map((v) => () =>
      agentRetry(() => agent(`You are a ${langToken(LEGACY_EVAL_LANG_TOKEN)} kernel evaluator. Compile and run this candidate on real hardware and report LIGHTWEIGHT signals only (no ncu/nsys profiling).

# Target GPU: ${TARGET_GPU}
# Reference op chain: ${(setup.op_chain || []).join(' -> ')}
# PyTorch-eager baseline latency: ${baselineLatency}ms

# Candidate (island role: ${islands[v.islIdx].role.name}):
${v.variant_path ? `# Full candidate source (Read for complete context — AWK #61): ${v.variant_path}\n` : ''}\`\`\`python
${v.code.substring(0, 4000)}
\`\`\`

# Steps:
1. Compile-check the Triton kernel (syntax, @triton.jit signature, grid).
2. ${BENCH_CMD ? `Run with: ${BENCH_CMD}` : 'No benchmark_command provided; do static runtime characterization only and mark latency as unmeasured.'}
3. ${TEST_CMD ? `Check correctness with: ${TEST_CMD}` : 'No test_command provided; do static equivalence review only and mark correctness as unmeasured.'}
4. Report: compiles?, runs?, correct?, latency_ms only if measured, measured flag, and speedup only when both baseline and candidate latencies are measured.
5. Capture lightweight execution metadata: launch config (BLOCK_SIZE, num_warps, num_stages, grid) if visible, and a one-line runtime characterization.

Return JSON.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if it compiled and ran correctly, else "error"; speedup is the measured speedup vs eager baseline, or null if unmeasured):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"iter${iter}-isl${v.islIdx}-${islands[v.islIdx].role.name}","speedup":<number or null>,"technique":"<launch config / runtime characterization>","note":"<compiles? correct? latency; or the failure reason>"}`, {
        label: `eval-${iter}-isl${v.islIdx}`,
        phase: 'Evaluate',
        schema: {
          type: 'object',
          properties: {
            compiles: { type: 'boolean' },
            runs: { type: 'boolean' },
            is_correct: { type: 'boolean' },
            latency_ms: { type: 'number' },
            speedup: { type: 'number' },
            launch_config: { type: 'string' },
            runtime_characterization: { type: 'string' },
            error_summary: { type: 'string' },
          },
          required: ['compiles', 'is_correct', 'speedup'],
        },
      }), { retries: 5 })
    )
  )

  if (USE_DRIVER_STANDALONE) {
    for (let k = 0; k < variants.length; k++) {
      const v = variants[k]
      const suffix = `${iter}-isl${v.islIdx}`
      const kPath = `${EXP_DIR}/variants/${suffix}/${kernelFilename()}`
      const buildOut = `${EXP_DIR}/variants/${suffix}/artifact`
      const profOut = `${EXP_DIR}/variants/${suffix}/prof.native`
      const resultPath = `${EXP_DIR}/variants/${suffix}/result.json`
      await agentRetry(() => agent(
        `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
        `Return its stdout JSON verbatim.`,
        { model: MODEL.mechanical, label: `driver-build-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      const runOut = await agentRetry(() => agent(
        `${driverSh('run.sh', `--artifact ${buildOut} --problem ${TASK_PATH} --out ${resultPath}`)}\n` +
        `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
        { model: MODEL.profile, label: `driver-run-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      let evidenceOut
      if (PROFILING_DECISION.method === 'native_profiler') {
        await agentRetry(() => agent(
          `${driverSh('profile.sh', `--artifact ${buildOut} --kernel ${kPath} --out ${profOut}`)}\n` +
          `Return {ok, native_path}.`,
          { model: MODEL.profile, label: `driver-profile-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
        evidenceOut = await agentRetry(() => agent(
          `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
          `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
          { model: MODEL.mechanical, label: `driver-to-evidence-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      } else {
        // Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run the native profiler.
        // run.sh already produced throughput above; normalize it when method='perf_heuristic'.
        const _normalizer = PROFILING_DECISION.normalizer || 'perf_to_evidence.py'
        evidenceOut = await agentRetry(() => agent(
          `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${_normalizer} --baseline ${resultPath}\`.\n` +
          (PROFILING_DECISION.method === 'perf_heuristic'
            ? `Normalize the run.sh throughput into canonical metrics. Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
              `Also write heuristic_bclass (memory_bound|compute_bound|latency_bound) from the throughput ratio so diagnose.py does not fall to unknown.\n`
            : ``) +
          `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
          { model: MODEL.mechanical, label: `driver-to-evidence-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      }
      const diagOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
        `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
        { model: MODEL.mechanical, label: `driver-diagnose-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      const antiCheatOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --source ${kPath} --metrics ${resultPath}\`.\n` +
        `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
        { model: MODEL.mechanical, label: `driver-anti-cheat-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      v.driver_envelope = {
        anti_cheat: antiCheatOut || {},
        metrics: (evidenceOut && evidenceOut.metrics) || {},
        vendor: (DRIVER && DRIVER.hw_vendor) || '',
        coverage: (evidenceOut && evidenceOut.coverage) || [],
        bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
        latency_ms: Number((runOut && runOut.latency_ms) || 0),
        backend_id: DRIVER_BACKEND_ID,
      }
    }
  } else if (IS_EMBEDDED) {
    // --- Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch) ---
    // SERIAL by construction (this for-loop, NOT parallel): embedded_inplace mutates the
    // shared TASK_PATH operator file and embedded_dispatch shares the project build, so
    // candidates cannot be evaluated concurrently (parallel-embedded-race bug-class).
    for (let k = 0; k < variants.length; k++) {
      const v = variants[k]
      const suffix = `${iter}-isl${v.islIdx}`
      const kPath = `${EXP_DIR}/variants/${suffix}/${kernelFilename()}`
      const variant = `kfdx_${suffix}`.replace(/[^A-Za-z0-9_]/g, '_')
      // Materialize the candidate source so the embedded eval can apply/register it.
      await agentRetry(() => agent(`Write the candidate kernel source to ${kPath} (mkdir -p its dir first).\n\n` +
        `\`\`\`${langToken(LEGACY_EVAL_LANG_TOKEN)}\n${(v.code || '').substring(0, 6000)}\n\`\`\`\n` +
        `Return {ok:true, path:"${kPath}"}.`,
        { model: MODEL.mechanical, label: `embedded-materialize-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      let embLatency = 0, embMetrics = {}, embBclass = 'unknown'
      if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
        const embResult = await agentRetry(() => agent(
          `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project operator file: ${TASK_PATH} | pristine backup: ${ORIGINAL_BACKUP}\n` +
          `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${TASK_PATH}\n` +
          `2. Apply candidate: cp ${kPath} ${TASK_PATH}\n3. Build: ${BUILD_CMD}\n4. Test: ${TEST_CMD}\n5. Benchmark: ${PROJECT_BENCH_CMD || TEST_CMD}\n` +
          `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${TASK_PATH}\n` +
          `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-inplace-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        embLatency = Number(embResult?.latency_ms || 0)
        embBclass = embResult?.heuristic_bclass || 'unknown'
        embMetrics = embResult?.metrics || { latency_ms: embLatency }
      } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
        const _plan = typeof __embeddedEvalPlan === 'function'
          ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: TEST_CMD, benchmarkCmd: PROJECT_BENCH_CMD || TEST_CMD })
          : null
        if (_plan) {
          const embResult = await agentRetry(() => agent(
            `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
            `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
            { model: MODEL.mechanical, label: `embedded-dispatch-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
          embLatency = Number(embResult?.latency_ms || 0)
          embBclass = embResult?.heuristic_bclass || 'unknown'
          embMetrics = embResult?.metrics || { latency_ms: embLatency }
        }
      }
      v.driver_envelope = { latency_ms: embLatency, metrics: embMetrics, bottleneck_class: embBclass, backend_id: 'embedded' }
    }
  }

  // ===========================================================================
  // Phase: Diagnose — Result Analyzer (failure mode OR perf limiter) + hints
  // ===========================================================================
  phase('Diagnose')

  const diagnoses = await parallel(
    variants.map((v, k) => () => {
      const e = evals[k]
      if (!e) return Promise.resolve(null)
      const correct = !!e.is_correct && !!e.compiles
      return agentRetry(() => agent(`You are the RESULT ANALYZER for a diagnosis-driven kernel optimizer. Diagnose this evaluated candidate and produce a reusable optimization hint. Use LIGHTWEIGHT signals only (no profiler).

# Candidate island role: ${islands[v.islIdx].role.name}
# Evaluation signals:
- compiles: ${e.compiles}, runs: ${e.runs}, correct: ${correct}
- latency: ${e.latency_ms}ms, speedup vs eager: ${e.speedup}
- launch config: ${e.launch_config || 'n/a'}
- runtime characterization: ${e.runtime_characterization || 'n/a'}
- error (if any): ${e.error_summary || 'none'}

# Candidate code (head — full source at ${v.variant_path || 'n/a'}, AWK #61):
\`\`\`python
${v.code.substring(0, 2500)}
\`\`\`

# Diagnosis rules:
- If the kernel is INCORRECT (compile error / runtime fail / output mismatch): set diagnosis_type="failure", classify failure_mode as one of "compile_error" | "runtime_failure" | "output_mismatch", and produce a targeted CORRECTNESS hint (bottleneck_class="correctness").
- If the kernel is CORRECT: set diagnosis_type="performance", classify the dominant limiter as one of "memory_bound" | "latency_bound" | "instruction_bound" from the runtime stats + launch config (this is coarse/approximate by design), and produce a PERFORMANCE hint with that bottleneck_class.

The hint must be GENERAL and reusable (a one-sentence actionable suggestion with a trigger condition), not specific to this exact kernel.

Return JSON.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (status="done" for a performance diagnosis, "error" for a failure diagnosis):
{"workflow":"${WORKFLOW_NAME}","phase":"Diagnose","ts":"<ts>","status":"<done|error>","candidate_id":"iter${iter}-isl${v.islIdx}-${islands[v.islIdx].role.name}","speedup":null,"technique":"<diagnosed limiter or failure_mode>","note":"<the generated hint, one line>"}`, {
        label: `diagnose-${iter}-isl${v.islIdx}`,
        phase: 'Diagnose',
        schema: {
          type: 'object',
          properties: {
            diagnosis_type: { type: 'string' },     // failure | performance
            failure_mode: { type: 'string' },        // when failure
            limiter: { type: 'string' },             // when performance
            rationale: { type: 'string' },
            hint: {
              type: 'object',
              properties: {
                trigger: { type: 'string' },
                bottleneck_class: { type: 'string' },
                suggestion: { type: 'string' },
              },
              required: ['bottleneck_class', 'suggestion'],
            },
            cheating_likelihood: { type: 'number' },
          },
          required: ['diagnosis_type', 'hint'],
        },
      }), { retries: 5 })
    })
  )

  // ===========================================================================
  // Phase: Evolve-Pop — update populations, migrate elites, reinforce hints
  // ===========================================================================
  phase('Evolve-Pop')

  for (let k = 0; k < variants.length; k++) {
    const v = variants[k]
    const e = evals[k]
    const d = diagnoses[k]
    if (!e) continue
    totalEvaluated++

    const correct = !!e.is_correct && !!e.compiles
    const speedup = e.speedup || 0
    const island = islands[v.islIdx]

    // Anti-cheating gate at evolution time
    if (d && (d.cheating_likelihood || 0) > CHEAT_THRESHOLD) {
      log(`Island ${island.role.name}: variant discarded (cheating likelihood ${d.cheating_likelihood})`)
      island.stagnation++
      continue
    }

    const diagnosis = d ? {
      type: d.diagnosis_type,
      failure_mode: d.failure_mode,
      limiter: d.limiter,
      rationale: d.rationale,
    } : null

    const member = { code: v.code, variant_path: v.variant_path || null, speedup: correct ? speedup : null, correct, diagnosis }  // AWK #59: carry variant_path so the next round's parent has a full-view path

    // Insert into population; keep top POPULATION_SIZE by speedup (correct first)
    island.population.push(member)
    island.population.sort((a, b) => {
      if ((a.correct ? 1 : 0) !== (b.correct ? 1 : 0)) return (b.correct ? 1 : 0) - (a.correct ? 1 : 0)
      return (b.speedup || 0) - (a.speedup || 0)
    })
    island.population = island.population.slice(0, POPULATION_SIZE)

    // Archive + trajectory
    island.archive.push(member)
    evolutionTrajectory.push({
      iteration: iter + 1,
      island: island.role.name,
      correct,
      speedup: correct ? speedup : null,
      limiter: diagnosis ? (diagnosis.limiter || diagnosis.failure_mode) : null,
      change: v.change_summary,
    })

    // Reinforce / down-weight hints that were applied to this variant
    for (const appliedText of v.applied_hints) {
      const matched = hintLibrary.find(h => appliedText && (appliedText.includes(h.suggestion.substring(0, 24)) || h.suggestion.includes(appliedText.substring(0, 24))))
      if (matched) updateHint(matched.id, speedup, correct)
    }

    // Add the freshly generated hint to the shared experience library
    if (d && d.hint && d.hint.suggestion) {
      const exists = hintLibrary.find(h => h.suggestion === d.hint.suggestion)
      if (!exists) {
        hintLibrary.push({
          id: `hint_${iter}_${v.islIdx}`,
          trigger: d.hint.trigger || '',
          bottleneck_class: d.hint.bottleneck_class || 'general',
          context: island.role.name,
          suggestion: d.hint.suggestion,
          success_count: correct && speedup > 1.0 ? 1 : 0,
          use_count: 1,
          avg_speedup: correct ? speedup : 1.0,
        })
      } else {
        updateHint(exists.id, speedup, correct)
      }
    }

    // Update island + global best
    if (correct && (island.bestSpeedup === null || speedup > island.bestSpeedup)) {
      island.bestSpeedup = speedup
      island.stagnation = 0
    } else {
      island.stagnation++
    }
    if (correct && (bestSpeedup === null || speedup > bestSpeedup)) {
      bestSpeedup = speedup
      bestKernel = { code: v.code, variant_path: v.variant_path || null, speedup, correct: true, island: island.role.name, iteration: iter + 1 }
      log(`NEW GLOBAL BEST: ${speedup.toFixed(2)}x from island ${island.role.name}`)
    }
  }

  // Probabilistic elite migration on stagnation
  for (let isl = 0; isl < islands.length; isl++) {
    const island = islands[isl]
    if (island.stagnation >= STAGNATION_WINDOW) {
      // Find a correct elite from another island
      let bestOther = null
      for (let other = 0; other < islands.length; other++) {
        if (other === isl) continue
        const cand = islands[other].population.find(m => m.correct)
        if (cand && (!bestOther || (cand.speedup || 0) > (bestOther.speedup || 0))) bestOther = cand
      }
      // Deterministic "probabilistic" migration: migrate when an elite exists
      if (bestOther) {
        island.population.push({ code: bestOther.code, speedup: bestOther.speedup, correct: true, diagnosis: bestOther.diagnosis })
        island.stagnation = 0
        log(`Elite migration -> island ${island.role.name} (seeded ${bestOther.speedup ? bestOther.speedup.toFixed(2) + 'x' : 'elite'})`)
      }
    }
  }

  log(`Iteration ${iter + 1} done. Islands best: [${islands.map(i => `${i.role.name}:${i.bestSpeedup ? i.bestSpeedup.toFixed(2) + 'x' : '-'}`).join(', ')}] | hints: ${hintLibrary.length}`)
}

// =============================================================================
// Phase: Report
// =============================================================================
phase('Report')

const finalReport = await agentRetry(() => agent(`Write a concise technical report for this Kernel Foundry (diagnosis-driven, multi-island) optimization run.

# Operation: ${OP_DESC} (${opType})
# Target GPU: ${TARGET_GPU}
# Run timestamp: ${RUN_TS}
# Baseline (PyTorch eager): ${baselineLatency}ms
# Best speedup found: ${bestSpeedup ? bestSpeedup.toFixed(2) + 'x' : 'none (no correct kernel beat baseline)'}
# Islands: ${islands.map(i => `${i.role.name} (best ${i.bestSpeedup ? i.bestSpeedup.toFixed(2) + 'x' : 'n/a'})`).join(', ')}
# Total candidates evaluated: ${totalEvaluated}

# Evolution trajectory:
${evolutionTrajectory.map(t => `- iter ${t.iteration} [${t.island}] ${t.correct ? 'correct' : 'INCORRECT'} ${t.speedup ? t.speedup.toFixed(2) + 'x' : ''} ${t.limiter ? '(' + t.limiter + ')' : ''} ${t.change || ''}`).join('\n')}

# Experience/hint library (final, sorted by avg speedup):
${[...hintLibrary].sort((a, b) => (b.avg_speedup || 0) - (a.avg_speedup || 0)).map((h, i) => `${i + 1}. [${h.bottleneck_class}] ${h.suggestion} (used ${h.use_count}x, avg ${Number(h.avg_speedup || 1).toFixed(2)}x)`).join('\n')}

# Best kernel:
\`\`\`python
${bestKernel ? bestKernel.code.substring(0, 3000) : '(none)'}
\`\`\`

Write:
1. How the diagnosis-driven loop steered evolution (which limiters were hit, which hints reinforced).
2. Which island role contributed the best kernel and why.
3. How the experience library evolved (which hints proved useful vs were down-weighted).
4. Any anti-cheating discards and what they reveal.
5. Remaining bottleneck of the best kernel and what to try next.`, {
  label: 'final-report',
  phase: 'Report',
}), { retries: 5 })

// embedded_inplace exit safety net: unconditionally restore the pristine operator file.
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${TASK_PATH}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: TASK_PATH,
  best_kernel_code: bestKernel ? bestKernel.code : null,
  best_kernel_path: bestKernel ? (bestKernel.variant_path || null) : null,  // AWK #59: authoritative file path (orchestrator prefers this; best_kernel_code is the compat string that may truncate for >20KB kernels)
  generated_kernel_path: bestKernel?.path || '',
  initial_candidates: evolutionTrajectory.filter(e => e.type === 'seed'),
  initial_generation_result: {
    verified: bestSpeedup > 0,
    selected_candidate_id: bestKernel?.id || '',
  },
  baseline_latency_ms: baselineLatency,
  best_speedup: bestSpeedup,
  best_kernel_island: bestKernel ? bestKernel.island : null,
  iterations_completed: ITERATIONS,
  islands_count: islands.length,
  island_roles: islands.map(i => i.role.name),
  island_best_speedups: islands.map(i => ({ role: i.role.name, best_speedup: i.bestSpeedup })),
  candidates_evaluated: totalEvaluated,
  hint_library: hintLibrary,
  hint_library_size: hintLibrary.length,
  evolution_trajectory: evolutionTrajectory,
  report: finalReport,
}
