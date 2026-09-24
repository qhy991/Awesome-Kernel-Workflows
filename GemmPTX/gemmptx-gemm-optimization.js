export const meta = {
  name: 'gemmptx-gemm-optimization',
  description: 'GEMM-specific CUDA/CuTe/CUTLASS optimization loop driven by hardware census, PTX/SASS instruction evidence, correctness, benchmark, and optional NCU profiling.',
  whenToUse: 'Use for existing CUDA/CuTe/CUTLASS GEMM kernels when the goal is to choose and verify the best instruction path (mma.sync, wgmma.mma_async, cp.async/TMA, tcgen05/TMEM) from real hardware and disassembly evidence. Not a generic compute-bound optimizer.',
  requiredSkills: [],
  optionalSkills: ['gemmptx-instruction-evidence'],
  phases: [
    { title: 'Hardware Census', detail: 'Collect target GPU facts from a user-provided probe command and static problem context' },
    { title: 'GEMM Signature', detail: 'Classify GEMM shape/dtype/layout/roofline regime and current implementation style' },
    { title: 'Baseline Evidence', detail: 'Compile, test, benchmark, and disassemble the baseline before proposing PTX-level changes' },
    { title: 'Instruction Plan', detail: 'Select a small set of architecture-appropriate PTX/SASS hypotheses with regex evidence gates' },
    { title: 'Implement', detail: 'Materialize one candidate at a time without mutating the original kernel' },
    { title: 'Disassemble Verify', detail: 'Reject candidates whose expected PTX/SASS instruction path is not observed' },
    { title: 'Profile', detail: 'Run benchmark and optional NCU/profile evidence only after correctness and instruction verification' },
    { title: 'Decide', detail: 'Accept, reject, or record hypothesis_not_realized based on measured evidence' },
    { title: 'Report', detail: 'Write final best path, evidence, and lessons for future GEMM/PTX runs' },
  ],
}

const WORKFLOW_NAME = 'gemmptx-gemm-optimization'

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
      // Policy refusals are terminal for this request. In particular, the
      // provider's "safeguards flagged this message" response must never be
      // sent again by the generic transient-failure retry path. The message
      // check also protects direct/older Hosts that lack the typed code.
      if (e && (e.code === 'KERSOR_PROVIDER_SAFEGUARD_REFUSAL'
        || /safeguards? flagged (?:this|the) message|provider safeguard refusal/i.test(String(e.message || '')))) {
        throw e
      }
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
  let expired = false
  const guard = new Promise((_, reject) => {
    timer = setTimeout(
      () => {
        expired = true
        reject(new Error(`turn-timeout: ${label} exceeded ${Math.round(TURN_TIMEOUT_MS / 1000)}s`))
      },
      TURN_TIMEOUT_MS)
  })
  return Promise.race([promise, guard]).catch(async error => {
    if (expired) {
      // A race does not cancel agent(). Drain this Host-bounded activation so
      // no running call survives the workflow's return.
      try { await promise } catch (_) { /* preserve the guard error */ }
    }
    throw error
  }).finally(() => {
    if (typeof clearTimeout === 'function') clearTimeout(timer)
  })
}
// --- END inlined turn-timeout scaffolding ---

const MODEL = {
  mechanical: args.model_mechanical || 'haiku',
  profile: args.model_profile || 'sonnet',
  judgment: args.model_judgment || 'opus',
}

const KERNEL_PATH = args.kernel_path || ''
const PROBLEM_PATH = args.problem_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const LANGUAGE = String(args.language || 'cuda').toLowerCase()
const TARGET_GPU = args.target_gpu || ''
const EXP_DIR = args.exp_dir || '/tmp/gemmptx'
const ITERATIONS = Number(args.iterations || 3)
const HARDWARE_PROBE_COMMAND = args.hardware_probe_command || ''
const COMPILE_COMMAND = args.compile_command || ''
const TEST_COMMAND = args.test_command || ''
const BENCHMARK_COMMAND = args.benchmark_command || ''
const DISASSEMBLE_COMMAND = args.disassemble_command || args.ptx_disassemble_command || ''
const PROFILE_COMMAND = args.profile_command || args.ncu_command || ''
const MIN_SPEEDUP = Number(args.min_speedup || 1.01)

if (!KERNEL_PATH) {
  return { ok: false, error: 'missing_required_arg', missing: 'kernel_path', reason: 'GemmPTX optimizes an existing CUDA/CuTe/CUTLASS GEMM kernel.' }
}
if (!COMPILE_COMMAND || !TEST_COMMAND || !BENCHMARK_COMMAND || !DISASSEMBLE_COMMAND) {
  return {
    ok: false,
    error: 'missing_evidence_contract',
    missing: {
      compile_command: !COMPILE_COMMAND,
      test_command: !TEST_COMMAND,
      benchmark_command: !BENCHMARK_COMMAND,
      disassemble_command: !DISASSEMBLE_COMMAND,
    },
    reason: 'PTX/SASS-driven GEMM optimization requires compile, correctness, benchmark, and disassembly evidence. Do not claim an instruction path without disassembly evidence.',
  }
}
if (!['cuda', 'cu', 'cute', 'cutlass', 'cpp', 'c++'].includes(LANGUAGE)) {
  return { ok: false, error: 'unsupported_language', language: LANGUAGE, supported: ['cuda', 'cute', 'cutlass', 'cpp'] }
}

const JSON_SCHEMA = { type: 'object', additionalProperties: true }

async function __solExecbenchEvaluate(ctx) {
  // Claude's legacy Workflow host does not yet expose this optional primitive.
  // Keep the prompt-driven path as a compatibility edge, while KerSor's Host
  // owns exact source materialization and PACK/RUN/PARSE without an LLM turn.
  if (typeof evaluate !== 'function') return null
  return evaluate({
    protocol: 'sol-execbench-v1',
    label: ctx.label || 'sol-eval',
    phase: ctx.phase || 'Evaluate',
    candidatePath: ctx.kernelSource,
    candidateSource: ctx.candidateSource,
    baselineSolutionPath: ctx.baselineSolutionPath,
    remoteEvidence: true,
    disassemble: true,
    sassPattern: ctx.sassPattern || '',

    substrateDir: ctx.substrateDir,
    contractEnv: ctx.contractEnv,
    solutionOut: ctx.solutionOut,
    benchOut: ctx.benchOut,
    normalizedOut: ctx.normalizedOut || `${ctx.benchOut}.result.json`,
    solCli: ctx.solCli,
    taskDir: ctx.taskDir,
    benchConfig: ctx.benchConfig,
    seedDir: ctx.seedDir,
    cudaVisibleDevices: ctx.cudaVisibleDevices || '0',
    ldLibraryPath: ctx.ldLibraryPath || '',
    envPrefix: ctx.envPrefix || '',
    definitionPath: ctx.definitionPath || '',
    timeoutSeconds: ctx.timeoutSeconds || 0,
  }).then(__solGuardHarnessFault)
}

// A `compiled: false` from the evaluator does not always mean the candidate is
// bad.  `invalid_request` and `infrastructure_error` are the harness refusing or
// failing before the candidate was ever built, and callers that map any
// non-success onto compile_error burn refine turns and a stagnation budget on a
// misconfiguration.  Observed: a candidate staged outside the evaluation roots
// was rejected at preflight, reported three times as `compile_error`, and the run
// stopped at the stagnation limit having never compiled anything.  Surface a
// harness fault as a harness fault and stop, because retrying cannot fix it.
function __solGuardHarnessFault(result) {
  const HARNESS_FAULTS = ['invalid_request', 'infrastructure_error']
  if (result && HARNESS_FAULTS.includes(result.failure_code)) {
    const detail = result.stderr || result.stdout || ''
    throw new Error(
      `sol-execbench harness fault (${result.failure_code}) at stage `
      + `${result.stage || 'unknown'}: ${String(detail).slice(0, 400)} `
      + '- this is a harness or wiring fault, not a candidate compile error',
    )
  }
  return result
}

// --- sol-execbench wiring (Host-owned PACK/RUN/PARSE; no LLM turn) -----------
// These 15 workflows declared only the standalone path, so their benchmark was a
// command string embedded in a prompt for a read-only activation with no shell.
// The read-only allowlist deliberately excludes execution, so the agent could
// never run it.  The Host can, and this is the protocol it exposes for exactly
// that.
const SOL_CLI = args.sol_cli || ''
const SOL_TASK_DIR = args.sol_task_dir || ''
const SOL_BENCH_CONFIG = args.sol_bench_config || ''
const SOL_SEED_DIR = args.sol_seed_dir || ''
const SOL_CVD = args.sol_cuda_visible_devices || '0'
const SOL_LD_LIBRARY_PATH = args.sol_ld_library_path || ''
const SOL_ENV_PREFIX = args.sol_env_prefix || ''
const SOL_DEFINITION_PATH = args.sol_definition_path || ''
const SOL_SUBSTRATE_DIR = args.sol_substrate_dir || ''
const SOL_AVAILABLE = Boolean(SOL_CLI && SOL_TASK_DIR && SOL_SUBSTRATE_DIR)

const GEMMPTX_SKILL_HINT = 'Before making GEMM instruction-path judgments, read the workflow-local skill at GemmPTX/skills/gemmptx-instruction-evidence/SKILL.md if it exists. Apply its PTX/SASS evidence gates, architecture map, and failure checklist.'
const instructionCatalog = [
  {
    arch: 'sm80',
    instruction: 'mma.sync',
    ptx_regex: 'mma\\.sync\\.aligned',
    sass_regex: '(HMMA|MMA)',
    use_when: 'Ampere fp16/bf16 GEMM mainloops with tensor-core friendly alignment',
    pitfalls: ['register pressure from large accumulator tiles', 'cp.async stage count can reduce occupancy'],
  },
  {
    arch: 'sm90',
    instruction: 'wgmma.mma_async',
    ptx_regex: 'wgmma\\.mma_async',
    sass_regex: '(WGMMA|GMMA)',
    use_when: 'Hopper GEMM mainloops where warpgroup MMA and SMEM descriptors can feed tensor cores',
    pitfalls: ['descriptor swizzle mismatch', 'missing wgmma.fence/commit/wait ordering', 'large N accumulator register wall'],
  },
  {
    arch: 'sm90',
    instruction: 'cp.async.bulk.tensor',
    ptx_regex: 'cp\\.async\\.bulk\\.tensor',
    sass_regex: '(TMA|CP_ASYNC_BULK)',
    use_when: 'Hopper affine tile copies where TMA can replace per-thread copy loops',
    pitfalls: ['mbarrier expect_tx byte mismatch', 'missing async proxy fence before TMA stores'],
  },
  {
    arch: 'sm100',
    instruction: 'tcgen05',
    ptx_regex: 'tcgen05',
    sass_regex: '(TCGEN05|MMA)',
    use_when: 'Blackwell GEMM mainloops where TMEM accumulation can reduce register pressure',
    pitfalls: ['TMEM capacity planning', 'CTA-pair synchronization', 'small tiles may lose to wgmma/CUTLASS paths'],
  },
]

function commandContract(name, command, resultPath) {
  return [
    `# ${name} command contract`,
    `Command: ${command}`,
    `Result path: ${resultPath}`,
    'Substitute placeholders only when present: {kernel_path}, {candidate_path}, {artifact_path}, {result_path}, {exp_dir}, {target_gpu}.',
    'Run the command exactly after substitution. If it fails, return the non-zero result with stderr tail; do not invent fields.',
  ].join('\n')
}

function genomeFooter(phaseName, candidateId) {
  return [
    '',
    '# Genome self-report (REQUIRED - do this LAST; do NOT let it change your returned JSON)',
    `Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ`,
    `The line must be one JSON object with workflow="${WORKFLOW_NAME}", phase="${phaseName}", candidate_id="${candidateId || ''}", status, technique, speedup when measured, and a one-line note.`,
  ].join('\n')
}

let best = {
  candidate_id: 'baseline',
  kernel_path: KERNEL_PATH,
  speedup: 1.0,
  latency_ms: null,
  instruction_path: [],
  artifact_path: '',
}
const history = []
const HOST_SOL = SOL_AVAILABLE && typeof evaluate === 'function'
async function measureInstructionCandidate(label, candidatePath, source, baseline = false, sassPattern = '') {
  const prefix = `${EXP_DIR}/host-${label}`
  const result = await __solExecbenchEvaluate({
    label: `sol-eval-${label}`, phase: baseline ? 'Baseline Evidence' : 'Disassemble Verify',
    kernelSource: candidatePath, candidateSource: source,
    baselineSolutionPath: baseline ? `${SOL_SEED_DIR}/seed.solution.json` : undefined,
    sassPattern, substrateDir: SOL_SUBSTRATE_DIR,
    contractEnv: `${SOL_SEED_DIR}/contract.env`, solutionOut: `${prefix}.solution.json`,
    benchOut: `${prefix}.bench.jsonl`, solCli: SOL_CLI, taskDir: SOL_TASK_DIR,
    benchConfig: SOL_BENCH_CONFIG, seedDir: SOL_SEED_DIR, cudaVisibleDevices: SOL_CVD,
    ldLibraryPath: SOL_LD_LIBRARY_PATH, envPrefix: SOL_ENV_PREFIX, definitionPath: SOL_DEFINITION_PATH,
  })
  return result
}
function measuredEvidence(result) {
  return Boolean(result && result.measurement_valid !== false && result.compiled === true && result.correct === true
    && result.n_total > 0 && result.n_pass === result.n_total
    && Number.isFinite(result.speedup) && result.speedup > 0)
}
const hostBaseline = HOST_SOL ? await measureInstructionCandidate('baseline', KERNEL_PATH, undefined, true) : null
if (HOST_SOL && !measuredEvidence(hostBaseline)) {
  return { ok: false, error: 'baseline_invalid', baseline: hostBaseline }
}
if (HOST_SOL && (!hostBaseline.environment || !hostBaseline.disassembly?.ok)) {
  return { ok: false, error: 'missing_evidence_contract', baseline: hostBaseline,
    reason: 'Host baseline requires actual hardware identity and disassembly receipt.' }
}

// =============================================================================
// Phase 1: Hardware Census
// =============================================================================
phase('Hardware Census')

const hardware = HOST_SOL ? {
  measured: true, gpu_name: hostBaseline.environment.hardware,
  compute_capability: hostBaseline.environment.compute_capability,
  arch: `sm_${String(hostBaseline.environment.compute_capability || '').replace('.', '')}`,
  sm_count: hostBaseline.environment.sms, cuda_version: hostBaseline.environment.libs?.cuda || null,
  caveats: ['Hardware identity comes from the Host baseline evaluator receipt.'],
} : await agentRetry(() => agent(`You are the hardware-census agent for a GEMM/PTX optimization workflow.

# Goal
Collect target GPU facts that constrain PTX/SASS-level GEMM decisions.

# Inputs
- target_gpu: ${TARGET_GPU || '(not provided)'}
- hardware_probe_command: ${HARDWARE_PROBE_COMMAND || '(not provided)'}
- static fallback: if no command is provided, read architecture-specific facts only when they are already present in the repo (for example WarpSpeed/config/hardware-facts-sm90.md or sm100.md). Mark measured=false.

# Required fields
Return JSON with:
- measured: boolean
- gpu_name, compute_capability, arch
- sm_count, l2_bytes, shared_mem_per_sm, shared_mem_per_block_max
- regs_per_sm, max_threads_per_sm, hbm_bandwidth_gbps
- clock_state, driver_version, cuda_version
- caveats: array of strings

# Rules
- If hardware_probe_command is provided, run it exactly and parse its JSON/stdout.
- If you cannot measure a field, set it to null and explain in caveats.
- Do not invent exact SM/L2/cache numbers from memory.

${genomeFooter('Hardware Census', 'hardware')}`, {
  model: MODEL.profile,
  label: 'hardware-census',
  phase: 'Hardware Census',
  schema: JSON_SCHEMA,
}), { retries: 5 })

log(`Hardware: arch=${hardware.arch || '?'} gpu=${hardware.gpu_name || TARGET_GPU || '?'} sm=${hardware.sm_count || '?'} measured=${hardware.measured === true}`)

// =============================================================================
// Phase 2: GEMM Signature
// =============================================================================
phase('GEMM Signature')

const signature = await agentRetry(() => agent(`You are a GEMM kernel analyst. Build a structured signature for the target GEMM.

# Inputs
- kernel_path: ${KERNEL_PATH}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- language: ${LANGUAGE}
- hardware arch: ${hardware.arch || TARGET_GPU || '(unknown)'}

# Tasks
1. Read the kernel source and any provided problem file.
2. Confirm this is GEMM / batched GEMM / GEMM-like matmul. If it is not, return is_gemm=false with reason.
3. Extract or infer only from source/problem facts: dtype, accumulator dtype, M/N/K or dynamic axes, layout, transpose, epilogue, alignment assumptions.
4. Classify bottleneck prior: tensor_core_bound, grid_starved_small_m, memory_reuse_limited, register_pressure_bound, scheduler_latency_bound, or unknown.
5. Identify the current instruction path if visible from source: scalar FMA, WMMA, mma.sync, WGMMA, CUTLASS/CuTe collective, TMA, tcgen05, unknown.

Return structured JSON. Do not claim a PTX instruction is present until disassembly verifies it.

${genomeFooter('GEMM Signature', 'signature')}`, {
  model: MODEL.judgment,
  label: 'gemm-signature',
  phase: 'GEMM Signature',
  schema: {
    type: 'object',
    properties: {
      is_gemm: { type: 'boolean' },
      reason: { type: 'string' },
      op_family: { type: 'string' },
      dtype_a: { type: 'string' },
      dtype_b: { type: 'string' },
      dtype_acc: { type: 'string' },
      shape_summary: { type: 'string' },
      layout_summary: { type: 'string' },
      alignment_summary: { type: 'string' },
      bottleneck_prior: { type: 'string' },
      current_source_path: { type: 'string' },
      risks: { type: 'array', items: { type: 'string' } },
    },
    required: ['is_gemm', 'op_family', 'bottleneck_prior'],
  },
}), { retries: 5 })

if (signature.is_gemm === false) {
  return { ok: false, error: 'not_gemm', reason: signature.reason || 'GemmPTX only supports GEMM-like kernels.' }
}

// =============================================================================
// Phase 3: Baseline Evidence
// =============================================================================
phase('Baseline Evidence')

const baseline = HOST_SOL ? {
  ...hostBaseline, artifact_path: hostBaseline.solution_path,
  sass_path: hostBaseline.disassembly.sass_path,
  observed_instructions: (hostBaseline.disassembly.top_mnemonics || []).map(x => x[0]),
  instruction_summary: hostBaseline.disassembly.note,
} : await agentRetry(() => agent(`You are the baseline-evidence agent. Establish measured baseline correctness, latency, and instruction evidence before any edit.

# Baseline source
${KERNEL_PATH}

${commandContract('Compile', COMPILE_COMMAND, `${EXP_DIR}/baseline.compile.json`)}

${commandContract('Correctness', TEST_COMMAND, `${EXP_DIR}/baseline.test.json`)}

${commandContract('Benchmark', BENCHMARK_COMMAND, `${EXP_DIR}/baseline.bench.json`)}

${commandContract('Disassemble', DISASSEMBLE_COMMAND, `${EXP_DIR}/baseline.disasm.json`)}

# Required evidence
Run compile -> correctness -> benchmark -> disassemble. Return:
- compiled, correct
- artifact_path
- latency_ms, throughput, speedup (baseline speedup should be 1.0 if only absolute latency is available)
- disassembly evidence: ptx_path, sass_path, observed_instructions, registers_per_thread, shared_mem_bytes, local_mem_bytes, spill_loads, spill_stores
- instruction_summary

# Rules
- If compile or correctness fails, stop after the failure and return the evidence.
- Do not claim an instruction path without disassembly evidence.

${genomeFooter('Baseline Evidence', 'baseline')}`, {
  model: MODEL.profile,
  label: 'baseline-evidence',
  phase: 'Baseline Evidence',
  schema: JSON_SCHEMA,
}), { retries: 5 })

if (baseline.compiled === false || baseline.correct === false) {
  return { ok: false, error: 'baseline_invalid', baseline }
}

best = {
  candidate_id: 'baseline',
  kernel_path: KERNEL_PATH,
  speedup: Number(baseline.speedup || 1.0),
  latency_ms: baseline.latency_ms || null,
  throughput: baseline.throughput || null,
  artifact_path: baseline.artifact_path || '',
  instruction_path: baseline.observed_instructions || [],
  disassembly: baseline,
}
history.push({ candidate_id: 'baseline', status: 'baseline', speedup: best.speedup, latency_ms: best.latency_ms, instruction_path: best.instruction_path })

// =============================================================================
// Phase 4: Instruction Plan
// =============================================================================
phase('Instruction Plan')

const plan = await agentRetry(() => agent(`You are the GEMM/PTX instruction planner. Choose a small candidate set, each with a falsifiable instruction-level hypothesis.

# Hardware
${JSON.stringify(hardware, null, 2)}

# GEMM signature
${JSON.stringify(signature, null, 2)}

# Baseline disassembly
${JSON.stringify({
  observed_instructions: baseline.observed_instructions || [],
  registers_per_thread: baseline.registers_per_thread,
  shared_mem_bytes: baseline.shared_mem_bytes,
  spill_loads: baseline.spill_loads,
  spill_stores: baseline.spill_stores,
  instruction_summary: baseline.instruction_summary,
}, null, 2)}

# Instruction catalog
${JSON.stringify(instructionCatalog, null, 2)}

# Workflow-local skill
${GEMMPTX_SKILL_HINT}

# Requirements
Return up to ${ITERATIONS} candidates. Each candidate must include:
- candidate_id
- target_instruction
- ptx_regex
- sass_regex
- hypothesis
- expected_mechanism
- edit_strategy
- risk_level: low|medium|high
- reject_if_missing: true
- expected_gain_pct

# Planning discipline
- Prefer architecture-appropriate paths: sm80 -> mma.sync/cp.async; sm90 -> wgmma.mma_async/TMA; sm100 -> tcgen05/TMEM only when the hardware facts support it.
- Keep the first version conservative. Do not combine more than two major instruction changes in one candidate.
- Include contraindications such as register pressure, SMEM pressure, alignment, or missing tensor-core dtype.
- Every candidate must be judged by disassembly first; benchmark comes only after correctness and instruction verification.

${genomeFooter('Instruction Plan', 'plan')}`, {
  model: MODEL.judgment,
  label: 'instruction-plan',
  phase: 'Instruction Plan',
  schema: {
    type: 'object',
    properties: {
      candidates: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            candidate_id: { type: 'string' },
            target_instruction: { type: 'string' },
            ptx_regex: { type: 'string' },
            sass_regex: { type: 'string' },
            hypothesis: { type: 'string' },
            expected_mechanism: { type: 'string' },
            edit_strategy: { type: 'string' },
            risk_level: { type: 'string' },
            reject_if_missing: { type: 'boolean' },
            expected_gain_pct: { type: 'number' },
          },
          required: ['candidate_id', 'target_instruction', 'ptx_regex', 'sass_regex', 'hypothesis', 'edit_strategy'],
        },
      },
    },
    required: ['candidates'],
  },
}), { retries: 5 })

const candidates = (plan.candidates || []).slice(0, ITERATIONS)
if (candidates.length === 0) {
  return { ok: false, error: 'no_instruction_candidates', baseline, signature, hardware }
}

// =============================================================================
// Phases 5-8: Implement -> Disassemble Verify -> Profile -> Decide
// =============================================================================
for (let i = 0; i < candidates.length; i++) {
  const spec = candidates[i]
  const candidateId = spec.candidate_id || `candidate-${i + 1}`
  const candidatePath = `${EXP_DIR}/candidates/${candidateId}/kernel.cu`
  const artifactPath = `${EXP_DIR}/candidates/${candidateId}/kernel_artifact`

  phase('Implement')
  const implementation = await agentRetry(() => agent(`You are the GEMM/PTX implementor. Materialize exactly one candidate and do not mutate the original kernel.

# Candidate
${JSON.stringify(spec, null, 2)}

# Original kernel
${KERNEL_PATH}

# Workflow-local skill
${GEMMPTX_SKILL_HINT}

# Output
- candidate_path: ${candidatePath}
- artifact_path placeholder for commands: ${artifactPath}

# Procedure
1. Create the candidate directory.
2. Copy/read the original source.
3. Implement only the stated instruction-level hypothesis.
4. Keep the public API and harness interface compatible with the original.
5. Return the complete candidate source in kernel_code - every line, no
   placeholders. Preserve the original Python-visible run/forward signature and PYBIND11_MODULE binding. A read-only activation has no tool that writes files, so do not
   try to write ${candidatePath}; the Host stages what you return.
6. Do not benchmark in this phase.

# Guardrail
If you cannot honestly implement the stated PTX/SASS hypothesis, return implemented=false and explain. Do not substitute a different strategy.

${genomeFooter('Implement', candidateId)}`, {
    model: MODEL.judgment,
    label: `implement-${candidateId}`,
    phase: 'Implement',
    schema: JSON_SCHEMA,
  }), { retries: 5 })

  if (implementation.implemented === false) {
    history.push({ candidate_id: candidateId, status: 'implementation_failed', reason: implementation.reason || implementation.error || 'not implemented' })
    continue
  }

  phase('Disassemble Verify')
  const hostCandidate = HOST_SOL ? await measureInstructionCandidate(candidateId, candidatePath,
    implementation.kernel_code || '', false, spec.sass_regex || '') : null
  const hostInstructionVerified = measuredEvidence(hostCandidate)
    && hostCandidate.disassembly?.ok === true && hostCandidate.disassembly.instruction_verified === true
  const verify = HOST_SOL ? {
    compiled: hostCandidate?.compiled === true, correct: hostCandidate?.correct === true,
    status: !hostCandidate?.compiled ? 'compile_error' : !hostCandidate?.correct ? 'incorrect'
      : hostInstructionVerified ? 'verified' : 'hypothesis_not_realized',
    instruction_verified: hostInstructionVerified,
    observed_instructions: hostInstructionVerified ? [spec.target_instruction] : [],
    sass_path: hostCandidate?.disassembly?.sass_path || null,
    artifact_path: hostCandidate?.solution_path || null,
    error: hostCandidate?.stderr || hostCandidate?.disassembly?.error || '',
  } : await agentRetry(() => agent(`You are the instruction-evidence gate. Compile, test, and disassemble the candidate, then verify the expected PTX/SASS regexes.

# Candidate
- candidate_id: ${candidateId}
- candidate_path: ${candidatePath}
- artifact_path: ${artifactPath}
- target_instruction: ${spec.target_instruction}
- ptx_regex: ${spec.ptx_regex}
- sass_regex: ${spec.sass_regex}

# Workflow-local skill
${GEMMPTX_SKILL_HINT}

${commandContract('Compile', COMPILE_COMMAND, `${EXP_DIR}/candidates/${candidateId}/compile.json`)}

${commandContract('Correctness', TEST_COMMAND, `${EXP_DIR}/candidates/${candidateId}/test.json`)}

${commandContract('Disassemble', DISASSEMBLE_COMMAND, `${EXP_DIR}/candidates/${candidateId}/disasm.json`)}

# Rules
1. Run compile first. If it fails, return status="compile_error".
2. Run correctness next. If it fails, return status="incorrect".
3. Run disassemble next.
4. Search both PTX and SASS text/artifacts for ptx_regex and sass_regex.
5. If neither expected regex is observed, return status="hypothesis_not_realized". This is not a performance failure; it means the compiler did not produce the intended instruction path.
6. Do not benchmark here.

# Required return fields
status, compiled, correct, instruction_verified, observed_instructions, missing_expected_instructions, registers_per_thread, shared_mem_bytes, local_mem_bytes, spill_loads, spill_stores, ptx_path, sass_path, artifact_path.

Reminder: do not claim an instruction path without disassembly evidence.

${genomeFooter('Disassemble Verify', candidateId)}`, {
    model: MODEL.profile,
    label: `disassemble-verify-${candidateId}`,
    phase: 'Disassemble Verify',
    schema: JSON_SCHEMA,
  }), { retries: 5 })

  if (verify.status === 'compile_error' || verify.compiled === false) {
    history.push({ candidate_id: candidateId, status: 'compile_error', detail: verify.error || verify.failure_reason || '', target_instruction: spec.target_instruction })
    continue
  }
  if (verify.status === 'incorrect' || verify.correct === false) {
    history.push({ candidate_id: candidateId, status: 'incorrect', detail: verify.error || verify.failure_reason || '', target_instruction: spec.target_instruction })
    continue
  }
  if (verify.status === 'hypothesis_not_realized' || verify.instruction_verified === false) {
    history.push({
      candidate_id: candidateId,
      status: 'hypothesis_not_realized',
      target_instruction: spec.target_instruction,
      missing_expected_instructions: verify.missing_expected_instructions || [spec.ptx_regex, spec.sass_regex],
      registers_per_thread: verify.registers_per_thread,
    })
    log(`Candidate ${candidateId}: hypothesis_not_realized for ${spec.target_instruction}`)
    continue
  }

  phase('Profile')
  // This turn is told to benchmark, and its `measured` flag gates acceptance at
  // the Decide step below. A read-only activation cannot run a benchmark, so
  // measured stayed false and every candidate was rejected with speedup=n/a -
  // correct behaviour on the workflow's part, and the reason it never accepted
  // anything. The Host measures, and the turn keeps the profile reasoning.
  let __hostMeasured = hostCandidate
  if (!HOST_SOL && SOL_AVAILABLE && (implementation.kernel_code || '').trim()) {
    __hostMeasured = await __solExecbenchEvaluate({
      label: `sol-eval-${candidateId}`, phase: 'Profile',
      substrateDir: SOL_SUBSTRATE_DIR,
      kernelSource: candidatePath,
      candidateSource: implementation.kernel_code,
      contractEnv: `${SOL_SEED_DIR || '.'}/contract.env`,
      solutionOut: `${EXP_DIR}/candidates/${candidateId}/solution.json`,
      benchOut: `${EXP_DIR}/candidates/${candidateId}/bench.jsonl`,
      solCli: SOL_CLI, taskDir: SOL_TASK_DIR, benchConfig: SOL_BENCH_CONFIG,
      seedDir: SOL_SEED_DIR, cudaVisibleDevices: SOL_CVD,
      ldLibraryPath: SOL_LD_LIBRARY_PATH, envPrefix: SOL_ENV_PREFIX,
      definitionPath: SOL_DEFINITION_PATH,
    })
    if (__hostMeasured) {
      log(`Host-measured ${candidateId}: compiled=${__hostMeasured.compiled} `
        + `correct=${__hostMeasured.correct} speedup=${__hostMeasured.speedup} `
        + `workloads=${__hostMeasured.n_pass}/${__hostMeasured.n_total}`)
    }
  }
  const __measuredBlock = __hostMeasured ? `

# ALREADY MEASURED ON THE HOST - use these verbatim
measured=${__hostMeasured.compiled === true}
correct=${__hostMeasured.correct} speedup_vs_baseline=${__hostMeasured.speedup}
latency_ms=${__hostMeasured.latency_ms}
workloads_passed=${__hostMeasured.n_pass}/${__hostMeasured.n_total}
Report these rather than re-deriving them. Spend the turn on the profile and on
mechanism_moved: did ${spec.target_instruction} appear and move the number the
way the hypothesis predicted?` : ''
  const measured = await agentRetry(() => agent(`You are the measurement agent. The candidate already compiled, passed correctness, and verified its instruction path. Now benchmark and optionally profile it.

# Candidate
- candidate_id: ${candidateId}
- candidate_path: ${candidatePath}
- artifact_path: ${verify.artifact_path || artifactPath}
- target_instruction: ${spec.target_instruction}

${commandContract('Benchmark', BENCHMARK_COMMAND, `${EXP_DIR}/candidates/${candidateId}/bench.json`)}

${PROFILE_COMMAND ? commandContract('Profile', PROFILE_COMMAND, `${EXP_DIR}/candidates/${candidateId}/profile.json`) : '# Profile command: not provided. Skip NCU/profile and mark profile_available=false.'}

${__measuredBlock}

# Return
- measured: true if benchmark succeeded
- correct: true only if benchmark also confirms correctness or prior correctness remains valid
- latency_ms, throughput, speedup_vs_baseline, speedup_vs_best
- profile_available
- metrics: object with SM%, tensor pipe, L2 hit, DRAM%, occupancy, stall reasons when available
- mechanism_moved: did the predicted mechanism move in the expected direction?
- diagnosis

${genomeFooter('Profile', candidateId)}`, {
    model: MODEL.profile,
    label: `profile-${candidateId}`,
    phase: 'Profile',
    schema: JSON_SCHEMA,
  }), { retries: 5 })

  if (HOST_SOL) {
    // A narrative review cannot replace, fabricate, or downgrade Host metrics.
    Object.assign(measured, { measured: measuredEvidence(hostCandidate),
      correct: hostCandidate.correct === true, speedup_vs_baseline: hostCandidate.speedup,
      latency_ms: hostCandidate.latency_ms, profile_available: false, metrics: {} })
  }
  phase('Decide')
  // best.speedup arrived NaN and every comparison against it was false, so
  // candidates were logged as "rejected ... best=NaN" with no reason given.
  const speedup = Number(measured.speedup_vs_baseline || measured.speedup || 0)
  const bestSpeedup = Number.isFinite(best.speedup) ? best.speedup : 0
  const accepted = measured.measured !== false && measured.correct !== false && speedup >= Math.max(bestSpeedup * MIN_SPEEDUP, bestSpeedup + 0.000001)
  const record = {
    candidate_id: candidateId,
    status: accepted ? 'accepted' : 'rejected',
    target_instruction: spec.target_instruction,
    speedup,
    latency_ms: measured.latency_ms || null,
    instruction_verified: true,
    registers_per_thread: verify.registers_per_thread,
    shared_mem_bytes: verify.shared_mem_bytes,
    mechanism_moved: measured.mechanism_moved,
    diagnosis: measured.diagnosis || '',
  }
  history.push(record)

  if (accepted) {
    best = {
      candidate_id: candidateId,
      kernel_path: candidatePath,
      artifact_path: verify.artifact_path || artifactPath,
      speedup,
      latency_ms: measured.latency_ms || null,
      instruction_path: verify.observed_instructions || [spec.target_instruction],
      disassembly: verify,
      metrics: measured.metrics || {},
    }
    log(`Candidate ${candidateId}: accepted speedup=${speedup}`)
  } else {
    log(`Candidate ${candidateId}: rejected speedup=${speedup || 'n/a'} best=${best.speedup}`)
  }
}

// =============================================================================
// Phase 9: Report
// =============================================================================
phase('Report')

const report = await agentRetry(() => agent(`Write the final GEMM/PTX optimization report.

# Output directory
${EXP_DIR}

# Best
${JSON.stringify(best, null, 2)}

# History
${JSON.stringify(history, null, 2)}

# Required report contents
1. Best candidate path and speedup.
2. The verified PTX/SASS instruction path, citing PTX/SASS artifacts.
3. A table of all candidates with statuses including compile_error, incorrect, hypothesis_not_realized, rejected, accepted.
4. Lessons/dead ends: include any instruction path that failed to materialize and the likely reason.
5. Clear statement that this workflow is GEMM-specific and not a generic compute-bound optimizer.

Write:
- ${EXP_DIR}/report.md
- ${EXP_DIR}/history.json

${genomeFooter('Report', 'final')}`, {
  model: MODEL.judgment,
  label: 'report',
  phase: 'Report',
  schema: {
    type: 'object',
    properties: {
      report_path: { type: 'string' },
      history_path: { type: 'string' },
      lessons: { type: 'array', items: { type: 'string' } },
    },
    required: ['report_path', 'history_path'],
  },
}), { retries: 5 })

return {
  ok: true,
  workflow: WORKFLOW_NAME,
  best_candidate_id: best.candidate_id,
  best_kernel_path: best.kernel_path,
  best_artifact_path: best.artifact_path,
  best_speedup: best.speedup,
  best_latency_ms: best.latency_ms,
  verified_instruction_path: best.instruction_path,
  history,
  report_path: report.report_path,
  history_path: report.history_path,
  lessons: guard(report, 'lessons', []),
}
