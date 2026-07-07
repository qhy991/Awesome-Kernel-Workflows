export const meta = {
  name: 'llamacpp-metal-embedded-search',
  description: 'Multi-variant fan-out search for Metal kernels embedded in llama.cpp ggml-metal dispatch. Targets both quantized GEMM (mul_matvec, bottleneck #1) and flash attention (bottleneck #2). Each variant = new kernel function + register + project build + test + bench + unregister.',
  whenToUse: 'When the kernel under optimization is embedded in llama.cpp ggml-metal (cannot be compiled standalone) AND the bottleneck is structural enough that you want to fan out N independent kernel designs rather than iteratively patch one. Each subagent authors a complete kernel function; the workflow registers it into ggml-metal-ops.cpp and ggml-metal.metal via scripts/llamacpp_metal_register_variant.py, runs the project build/test/bench, then unregisters. Variants are evaluated SERIALLY (project build is the bottleneck) but proposals are drafted in parallel.',
  phases: [
    { title: 'Setup',     detail: 'Resolve paths, sanity-check register script, snapshot ggml-metal-ops.cpp' },
    { title: 'Baseline',  detail: 'Build + test + bench the unmodified project to set the baseline number' },
    { title: 'Propose',   detail: 'N subagents in parallel each draft one complete kernel function variant' },
    { title: 'Evaluate',  detail: 'For each variant: register -> build -> test -> bench -> unregister (serial)' },
    { title: 'Report',    detail: 'Rank by speedup, emit best variant code + history' },
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

const WORKFLOW_NAME = 'llamacpp-metal-embedded-search'

// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.

// --- BEGIN inlined arg_guard ---

// --- BEGIN inlined arg_guard (from _meta/scaffolding/arg-guard.js) ---
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
// --- END inlined arg_guard ---// --- END inlined arg_guard ---

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

// --- BEGIN inlined grounding contract ---
const GROUNDING_INSTRUCTION = [
  'GROUNDING CONTRACT (mandatory):',
  '',
  '1. Before reporting any numeric or categorical result, you MUST have executed',
  '   a real Bash command that produced the value. Citing a value from imagination,',
  '   prior knowledge, or analogy is a contract violation.',
  '',
  '2. If the workflow tells you to run a tool or script that does NOT exist on disk',
  '   or on PATH, do NOT substitute or simulate. Return:',
  '       { "grounded": false, "missing": "<exact tool or script that was absent>" }',
  '   and stop. The schema accepts this shape.',
  '',
  '3. If a Bash command fails (non-zero exit), do NOT invent the value it would',
  '   have produced. Return:',
  '       { "grounded": false, "error": "<stderr tail or short summary>" }',
  '   and stop.',
  '',
  '4. Numeric fields in the schema are REJECTED downstream when grounded === false.',
  '   Do not fill them with placeholders, zeros, or "typical" values.',
  '',
  '5. The string "fabricated", "simulated", "estimated", or "placeholder" must',
  '   never appear in a value you report as measured.',
].join('\n')

function withGroundingFields(schema) {
  schema.properties = schema.properties || {}
  if (!schema.properties.grounded) schema.properties.grounded = { type: 'boolean' }
  if (!schema.properties.missing)  schema.properties.missing  = { type: 'string' }
  if (!schema.properties.error)    schema.properties.error    = { type: 'string' }
  return schema
}

function classifyResult(r) {
  if (r == null) return { state: 'failed', error: 'agent returned null' }
  if (r.grounded === false) return { state: 'not_grounded', missing: r.missing || null, error: r.error || null }
  return { state: 'grounded', value: r }
}
// --- END inlined grounding contract ---

// =============================================================================
// Llamacpp Metal Embedded Multi-Variant Search
// =============================================================================
//
// Required args:
//   ggml_root          Absolute path to llama.cpp/ggml (parent of src/ggml-metal)
//   register_script    Absolute path to scripts/llamacpp_metal_register_variant.py
//   build_command      Project build command (run verbatim)
//   test_command       Project correctness command (KERSOR_VARIANT will be prefixed)
//   benchmark_command  Project benchmark command (KERSOR_VARIANT will be prefixed)
//   reference_metal    Path to the current best variant .metal (e.g. ggml-metal.metal).
//                      Subagents read this for inspiration and as the structural anchor.
//   kernel_family      Which kernel family to optimize: "mul_matvec" or "flash_attn"
//
// Optional args:
//   n_variants            Number of parallel proposals per round (default 3)
//   max_rounds            Number of search rounds (default 1)
//   target_speedup        Stop early if reached (default 1.05)
//   correctness_regex     Regex that must match test stdout for PASS (default: exit code only)
//   latency_regex         Regex with one capture group for latency number (default first float + us/ms)
//   latency_unit          us|ms (default us)
//   tolerance             NMSE tolerance (informational, default 5e-4)
//   op_description        Task context, prepended to proposer prompt
//   handoff_context       KerSor cross-round handoff text
//   workload_axes         Comma-separated workload labels (informational)
//   variants_dir          Where to write each variant .metal before register (default: $exp_dir/variants/)
//   exp_dir               Run scratch dir (default: process cwd)
//   backend_dir           Path to the metal backend driver (for profiling)
//
// =============================================================================

const GGML_ROOT       = args.ggml_root || ''
const REG_SCRIPT      = args.register_script || ''
const BUILD_CMD       = args.build_command || ''
const TEST_CMD        = args.test_command || ''
const BENCH_CMD       = args.benchmark_command || ''
const REFERENCE_METAL = args.reference_metal || ''
const KERNEL_FAMILY   = args.kernel_family || 'mul_matvec'
const N_VARIANTS      = Math.max(1, parseInt(args.n_variants || 3, 10))
const MAX_ROUNDS      = Math.max(1, parseInt(args.max_rounds || 1, 10))
const TARGET_SPEEDUP  = parseFloat(args.target_speedup || 1.05)
const CORRECT_REGEX   = args.correctness_regex || ''
const LATENCY_REGEX   = args.latency_regex || '([0-9]+\\.?[0-9]*)\\s*(?:us|ms)'
const LATENCY_UNIT    = (args.latency_unit || 'us').toLowerCase()
const TOLERANCE       = args.tolerance != null ? Number(args.tolerance) : 5e-4
const OP_DESC         = args.op_description || ''
const HANDOFF         = args.handoff_context || ''
const WORKLOAD_AXES   = args.workload_axes || ''
const EXP_DIR         = args.exp_dir || '.'
const VARIANTS_DIR    = args.variants_dir || `${EXP_DIR}/variants`
const BACKEND_DIR     = args.backend_dir || ''

const MISSING = []
if (!GGML_ROOT)       MISSING.push('ggml_root')
if (!REG_SCRIPT)      MISSING.push('register_script')
if (!BUILD_CMD)       MISSING.push('build_command')
if (!TEST_CMD)        MISSING.push('test_command')
if (!BENCH_CMD)       MISSING.push('benchmark_command')
if (!REFERENCE_METAL) MISSING.push('reference_metal')
if (MISSING.length) {
  throw new Error(`llamacpp-metal-embedded-search: missing required arg(s): ${MISSING.join(', ')}`)
}

if (KERNEL_FAMILY !== 'mul_matvec' && KERNEL_FAMILY !== 'flash_attn') {
  throw new Error(`llamacpp-metal-embedded-search: kernel_family must be 'mul_matvec' or 'flash_attn', got '${KERNEL_FAMILY}'`)
}

// =============================================================================
// Schemas
// =============================================================================

const PROPOSAL_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    written: { type: 'boolean', description: 'true iff the .metal file was successfully written' },
    variant_name: { type: 'string', description: 'lowercase [a-z0-9_]{1,32} - this is what KERSOR_VARIANT must be set to' },
    metal_path: { type: 'string', description: 'absolute path to the new variant .metal that was written' },
    title: { type: 'string', description: 'short label for this design' },
    rationale: { type: 'string', description: 'why this design should be faster, citing the reference kernel' },
    risk: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
  required: ['written', 'variant_name', 'metal_path', 'title', 'rationale'],
})

const BUILD_SCHEMA = withGroundingFields({
  type: 'object',
  properties: { ok: { type: 'boolean' }, stderr_tail: { type: 'string' }, duration_sec: { type: 'number' } },
  required: ['ok'],
})

const TEST_SCHEMA = withGroundingFields({
  type: 'object',
  properties: { correct: { type: 'boolean' }, detail: { type: 'string' } },
  required: ['correct'],
})

const BENCH_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    latency_per_workload: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          workload: { type: 'string' },
          latency: { type: 'number' },
          unit: { type: 'string' },
        },
        required: ['workload', 'latency'],
      },
    },
    aggregate_latency: { type: 'number' },
    aggregate_strategy: { type: 'string' },
  },
  required: ['aggregate_latency'],
})

// =============================================================================
// Helpers
// =============================================================================

async function runBuild(label, env) {
  const cmd = env ? `${env} ${BUILD_CMD}` : BUILD_CMD
  const r = await agentRetry(() => agent(
    [
      `Run the project build command and report whether it succeeded.`,
      `Command: ${cmd}`,
      `If exit is non-zero, include the last ~30 lines of stderr in stderr_tail.`,
      `Measure wall-clock duration in seconds.`,
      GROUNDING_INSTRUCTION,
    ].join('\n\n'),
    { phase: 'Evaluate', label: `build:${label}`, schema: BUILD_SCHEMA }
  ), { retries: 5 })
  return classifyResult(r)
}

async function runTest(label, env) {
  const cmd = env ? `${env} ${TEST_CMD}` : TEST_CMD
  const correctnessHint = CORRECT_REGEX
    ? `CORRECT iff /${CORRECT_REGEX}/ matches stdout AND exit is 0.`
    : `CORRECT iff exit is 0.`
  const r = await agentRetry(() => agent(
    [
      `Run the project correctness command and report PASS/FAIL.`,
      `Command: ${cmd}`,
      correctnessHint,
      `Tolerance reference (NMSE): ${TOLERANCE}.`,
      GROUNDING_INSTRUCTION,
    ].join('\n\n'),
    { phase: 'Evaluate', label: `test:${label}`, schema: TEST_SCHEMA }
  ), { retries: 5 })
  return classifyResult(r)
}

async function runBench(label, env) {
  const cmd = env ? `${env} ${BENCH_CMD}` : BENCH_CMD
  const r = await agentRetry(() => agent(
    [
      `Run the project benchmark command and extract per-workload latency.`,
      `Command: ${cmd}`,
      `Latency regex: /${LATENCY_REGEX}/ - first capture group is the number, unit is ${LATENCY_UNIT}.`,
      `If output has multiple workload rows, emit one entry per workload in latency_per_workload.`,
      `Set aggregate_latency = GEOMEAN of the per-workload latencies (single regressions cannot dominate).`,
      `Set aggregate_strategy to a short description.`,
      WORKLOAD_AXES ? `Workload axes hint: ${WORKLOAD_AXES}` : '',
      GROUNDING_INSTRUCTION,
    ].filter(Boolean).join('\n\n'),
    { phase: 'Evaluate', label: `bench:${label}`, schema: BENCH_SCHEMA }
  ), { retries: 5 })
  return classifyResult(r)
}

async function registerVariant(variantName, metalPath) {
  const r = await agentRetry(() => agent(
    [
      `Run the register script to wire this variant into ggml-metal dispatch:`,
      `  python "${REG_SCRIPT}" register --variant ${variantName} --source "${metalPath}" --project-root "${GGML_ROOT}" --kernel-family ${KERNEL_FAMILY}`,
      `Then run "python ${REG_SCRIPT} list --project-root ${GGML_ROOT}" and confirm ${variantName} appears in the list.`,
      GROUNDING_INSTRUCTION,
    ].join('\n\n'),
    {
      phase: 'Evaluate',
      label: `register:${variantName}`,
      schema: withGroundingFields({
        type: 'object',
        properties: { ok: { type: 'boolean' }, stderr_tail: { type: 'string' } },
        required: ['ok'],
      }),
    }
  ), { retries: 5 })
  return classifyResult(r)
}

async function unregisterVariant(variantName) {
  const r = await agentRetry(() => agent(
    [
      `Run the register script to remove this variant from ggml-metal dispatch (must be byte-exact reversible):`,
      `  python "${REG_SCRIPT}" unregister --variant ${variantName} --project-root "${GGML_ROOT}"`,
      `Confirm by running "python ${REG_SCRIPT} list --project-root ${GGML_ROOT}" and reporting that ${variantName} is no longer listed.`,
      GROUNDING_INSTRUCTION,
    ].join('\n\n'),
    {
      phase: 'Evaluate',
      label: `unregister:${variantName}`,
      schema: withGroundingFields({
        type: 'object',
        properties: { ok: { type: 'boolean' } },
        required: ['ok'],
      }),
    }
  ), { retries: 5 })
  return classifyResult(r)
}

// =============================================================================
// Kernel-family-specific proposal prompt guidance
// =============================================================================

const MUL_MATVEC_PROMPT = [
  `You are designing ONE complete Metal Shading Language kernel function for llama.cpp's`,
  `quantized GEMM (mul_matvec) path. This is the #1 bottleneck in single-batch decode:`,
  `for batch=1, the GPU achieves only 0.004-0.05 TFLOPS vs ~3 TFLOPS at batch=512.`,
  `The goal is to close this 60-750x gap for the M=1 vector-matrix product path.`,
  ``,
  `The kernel function signature MUST match what ggml-metal-ops.cpp expects.`,
  `Study the reference kernel in ggml-metal.metal for the exact parameter layout.`,
  `Typical mul_matvec kernel signatures:`,
  `  kernel void kernel_mul_mv_q4_0_f32(`,
  `      constant ggml_metal_kargs_mul_mv & args,`,
  `      device const char * src0,   // quantized weights (Q4_0/Q4_1/Q8_0 blocks)`,
  `      device const char * src1,   // activation vector (f32)`,
  `      device       char * dst,    // output`,
  `      threadgroup  char * shmem [[threadgroup(0)]],`,
  `      uint3  tgpig[[threadgroup_position_in_grid]],`,
  `      ushort tiisg[[thread_index_in_simdgroup]],`,
  `      ushort sgitg[[simdgroup_index_in_threadgroup]])`,
  ``,
  `Your variant kernel function should be named: kernel_<variant_name>`,
  `and use the SAME args struct type (ggml_metal_kargs_mul_mv) so the dispatch`,
  `can pass parameters byte-exact.`,
  ``,
  `Optimization strategies for single-batch quantized GEMM on Apple Silicon:`,
  `  - Wider per-thread work: process 4-8 output rows per threadgroup instead of 1-2`,
  `  - Dequantize and cache: pre-dequantize weight blocks into threadgroup memory`,
  `  - simdgroup reduction: use simd_shuffle_down for final dot-product accumulation`,
  `  - packed vectorized loads: use packed_float4 to read activation vector in 128-bit chunks`,
  `  - Threadgroup size tuning: 256/512/1024 to maximize occupancy on M4 GPU`,
  `  - Loop unrolling: [[clang::loop_unroll]] on inner quant-decode loops`,
  `  - Prefetch: stage next weight block while computing current block`,
  `  - Fuse dequant + dot product: avoid separate dequantize pass`,
].join('\n')

const FLASH_ATTN_PROMPT = [
  `You are designing ONE complete Metal Shading Language kernel function for llama.cpp's`,
  `flash attention path. This is the #2 bottleneck in single-batch decode (~25% of time).`,
  `The typical GQA config is 8 query heads / 2 KV heads, with head_dim=128.`,
  ``,
  `The kernel function MUST use the same kargs struct as the reference flash attention`,
  `kernels (ggml_metal_kargs_flash_attn_ext_blk or ggml_metal_kargs_flash_attn_ext).`,
  `Study the reference kernel in ggml-metal.metal for the exact function constant layout.`,
  ``,
  `Your variant kernel function should be named: kernel_<variant_name>`,
  ``,
  `Optimization strategies for GQA flash attention on Apple Silicon:`,
  `  - KV head sharing: 8 Q heads share 2 KV heads -- load KV once per threadgroup,`,
  `    reuse across 4 Q heads to cut memory bandwidth by ~4x`,
  `  - threadgroup memory tiling: tile Q and KV in threadgroup memory for the local`,
  `    attention window, sync with threadgroup_barrier(mem_flags::mem_threadgroup)`,
  `  - Online softmax: use the standard online softmax (m/l/delta) pattern to avoid`,
  `    storing the full attention matrix`,
  `  - simdgroup shuffle: use simd_shuffle_down for max/exp reduction within simdgroups`,
  `  - Threadgroup distribution: assign different KV chunks to different threadgroups,`,
  `    reduce partial results at the end`,
  `  - metal::fast::exp for the softmax exponential (adequate precision for attention)`,
  `  - packed_half4 for f16 KV cache loads to maximize bandwidth utilization`,
].join('\n')

// =============================================================================
// Phase: Setup + Baseline
// =============================================================================

phase('Setup')
log(`ggml_root         = ${GGML_ROOT}`)
log(`register_script   = ${REG_SCRIPT}`)
log(`reference_metal   = ${REFERENCE_METAL}`)
log(`kernel_family     = ${KERNEL_FAMILY}`)
log(`variants_dir      = ${VARIANTS_DIR}`)
log(`n_variants/round  = ${N_VARIANTS}`)
log(`max_rounds        = ${MAX_ROUNDS}`)
log(`backend           = metal (Apple Silicon)`)

await agentRetry(() => agent(
  [
    `Ensure the variants directory exists: mkdir -p "${VARIANTS_DIR}".`,
    `Also confirm the register script is executable by running:`,
    `  python "${REG_SCRIPT}" list --project-root "${GGML_ROOT}"`,
    `Report ok=true iff both succeeded.`,
    GROUNDING_INSTRUCTION,
    `# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\nAppend exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\nThen append:\n{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"workspace_sanity_check","note":"<variants dir created? register script listed ok? one line>"}`,
  ].join('\n\n'),
  {
    phase: 'Setup',
    label: 'sanity-check',
    schema: withGroundingFields({
      type: 'object',
      properties: { ok: { type: 'boolean' } },
      required: ['ok'],
    }),
  }
), { retries: 5 })

phase('Baseline')
const baseBuild = await runBuild('baseline', '')
if (baseBuild.state !== 'grounded' || !baseBuild.value.ok) {
  return { ok: false, grounded: baseBuild.state === 'grounded',
    error: `baseline build failed: ${JSON.stringify(baseBuild)}`,
    speedup: null, best_kernel_code: null, history: [] }
}
const baseTest = await runTest('baseline', '')
if (baseTest.state !== 'grounded' || !baseTest.value.correct) {
  return { ok: false, grounded: baseTest.state === 'grounded',
    error: `baseline correctness failed: ${JSON.stringify(baseTest)}`,
    speedup: null, best_kernel_code: null, history: [] }
}
const baseBench = await runBench('baseline', '')
if (baseBench.state !== 'grounded' || !baseBench.value.aggregate_latency) {
  return { ok: false, grounded: baseBench.state === 'grounded',
    error: `baseline bench could not parse latency: ${JSON.stringify(baseBench)}`,
    speedup: null, best_kernel_code: null, history: [] }
}
const BASELINE_LATENCY = baseBench.value.aggregate_latency
log(`Baseline aggregate latency: ${BASELINE_LATENCY} ${LATENCY_UNIT}`)

// =============================================================================
// Search rounds
// =============================================================================

const history = []
let bestSpeedup = 1.0
let bestLatency = BASELINE_LATENCY
let bestVariantName = '(baseline)'
let bestMetalPath = REFERENCE_METAL

for (let round = 1; round <= MAX_ROUNDS; ++round) {
  if (bestSpeedup >= TARGET_SPEEDUP) {
    log(`Target ${TARGET_SPEEDUP}x reached. Stopping after round ${round - 1}.`)
    break
  }

  phase('Propose')
  log(`\n=== Round ${round}/${MAX_ROUNDS} - proposing ${N_VARIANTS} variants in parallel ===`)

  const priorTitles = history
    .map(h => `  - ${h.variant_name} (${h.title}) -> ${h.verdict} (${h.speedup?.toFixed?.(3) ?? 'n/a'}x)`)
    .join('\n') || '  (none)'

  const familyPrompt = KERNEL_FAMILY === 'flash_attn' ? FLASH_ATTN_PROMPT : MUL_MATVEC_PROMPT

  const proposals = await parallel(
    Array.from({ length: N_VARIANTS }, (_, i) => () => {
      const slot = `r${round}v${i + 1}`
      return agentRetry(() => agent(
        [
          OP_DESC ? `Task context: ${OP_DESC}` : '',
          HANDOFF ? `Handoff from prior rounds:\n${HANDOFF}` : '',
          ``,
          familyPrompt,
          ``,
          `Reference kernel file: ${REFERENCE_METAL}`,
          `Read this file in full first. Your variant MUST use Metal Shading Language (MSL)`,
          `conventions consistent with the reference.`,
          `Do NOT create an ObjC wrapper or main() -- this kernel lives inside a project.`,
          ``,
          `Constraints:`,
          `  - Pick a variant_name matching [a-z0-9_]{1,32}; suggested: "${slot}"`,
          `  - Write the complete kernel function to: ${VARIANTS_DIR}/<variant_name>.metal`,
          `  - The file should contain ONLY the kernel function(s), no host code`,
          `  - Do NOT register the variant yourself - the workflow does that.`,
          `  - Do NOT modify ggml-metal-ops.cpp or ggml-metal.metal or any other file.`,
          `  - Make ONE design distinctly different from prior attempts:`,
          ``,
          `Prior attempts this session:\n${priorTitles}`,
          ``,
          `Workload context: ${WORKLOAD_AXES || '(none)'}`,
          `Tolerance reference (NMSE): ${TOLERANCE}`,
          ``,
          `Metal-specific optimization opportunities on Apple Silicon:`,
          `  - Unified memory (UMA): CPU and GPU share the same physical memory`,
          `  - threadgroup memory (shared) for tiling to reduce device memory traffic`,
          `  - simd_shuffle / simd_shuffle_down for intra-simdgroup reductions`,
          `  - packed_float4 / packed_half4 for 128-bit vectorized loads`,
          `  - metal::fast math intrinsics where precision tolerance allows`,
          `  - Threadgroup sizes: 256/512/1024 (multiples of simdgroup size 32)`,
          `  - [[clang::loop_unroll]] for inner loops`,
          ``,
          `Return the absolute metal_path you wrote, the variant_name, a short title,`,
          `and a rationale citing the specific design choice.`,
          GROUNDING_INSTRUCTION,
          `# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\nAppend exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\nThen append (this is round ${round}, proposal slot ${slot}):\n{"workflow":"${WORKFLOW_NAME}","phase":"Propose","ts":"<ts>","status":"done","candidate_id":"${slot}","technique":"<the main Metal optimization in this variant, e.g. simdgroup_reduction or threadgroup_tiling>","speedup":null,"note":"<one-line rationale: what makes this design distinct from prior attempts>"}`,
        ].filter(Boolean).join('\n\n'),
        { phase: 'Propose', label: `propose:${slot}`, schema: PROPOSAL_SCHEMA }
      ), { retries: 5 })
    })
  )

  const drafted = proposals
    .map(classifyResult)
    .filter(p => p.state === 'grounded' && p.value.written)
    .map(p => p.value)

  log(`Drafted ${drafted.length}/${N_VARIANTS} variants for round ${round}`)

  // ----- Serial evaluate -----
  phase('Evaluate')
  for (const v of drafted) {
    log(`\n--- Evaluating ${v.variant_name} ("${v.title}") ---`)
    const reg = await registerVariant(v.variant_name, v.metal_path)
    if (reg.state !== 'grounded' || !reg.value.ok) {
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'register_failed', speedup: null,
        detail: JSON.stringify(reg).slice(0, 400) })
      continue
    }

    const b = await runBuild(v.variant_name, '')
    if (b.state !== 'grounded' || !b.value.ok) {
      await unregisterVariant(v.variant_name)
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'build_failed', speedup: null,
        detail: b.state === 'grounded' ? (b.value.stderr_tail || '').slice(-400) : JSON.stringify(b) })
      continue
    }

    const t = await runTest(v.variant_name, `KERSOR_VARIANT=${v.variant_name}`)
    if (t.state !== 'grounded' || !t.value.correct) {
      await unregisterVariant(v.variant_name)
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'correctness_failed', speedup: null,
        detail: t.state === 'grounded' ? t.value.detail : JSON.stringify(t) })
      continue
    }

    const m = await runBench(v.variant_name, `KERSOR_VARIANT=${v.variant_name}`)
    if (m.state !== 'grounded' || !m.value.aggregate_latency) {
      await unregisterVariant(v.variant_name)
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'bench_unparseable', speedup: null,
        detail: JSON.stringify(m) })
      continue
    }

    const latency = m.value.aggregate_latency
    const speedup = BASELINE_LATENCY / latency
    log(`  ${v.variant_name}: ${latency} ${LATENCY_UNIT} -> ${speedup.toFixed(3)}x`)

    if (speedup > bestSpeedup) {
      bestSpeedup = speedup
      bestLatency = latency
      bestVariantName = v.variant_name
      bestMetalPath = v.metal_path
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'kept_best', speedup,
        detail: `latency ${latency} ${LATENCY_UNIT}`, metal_path: v.metal_path })
    } else {
      await unregisterVariant(v.variant_name)
      history.push({ round, variant_name: v.variant_name, title: v.title, verdict: 'unkept', speedup,
        detail: `latency ${latency} ${LATENCY_UNIT}` })
    }
  }
}

// =============================================================================
// Report
// =============================================================================

phase('Report')

let bestCode = null
if (bestVariantName !== '(baseline)') {
  const r = await agentRetry(() => agent(
    `Read the file at ${bestMetalPath} and return its full contents in best_kernel_code.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","candidate_id":"${bestVariantName}","technique":"best_variant_selected","speedup":${bestSpeedup},"note":"<best variant ${bestVariantName} at ${bestLatency} ${LATENCY_UNIT}; one-line summary>"}`,
    {
      phase: 'Report',
      label: 'final-read',
      schema: withGroundingFields({
        type: 'object',
        properties: { best_kernel_code: { type: 'string' } },
        required: ['best_kernel_code'],
      }),
    }
  ), { retries: 5 })
  bestCode = classifyResult(r).value?.best_kernel_code || null
  await unregisterVariant(bestVariantName)
}

return {
  ok: true,
  grounded: true,
  speedup: bestSpeedup,
  overall_speedup: bestSpeedup,
  rounds_completed: history.length,
  baseline_latency: BASELINE_LATENCY,
  best_latency: bestLatency,
  best_variant_name: bestVariantName,
  best_metal_path: bestMetalPath,
  best_kernel_code: bestCode,
  latency_unit: LATENCY_UNIT,
  kernel_family: KERNEL_FAMILY,
  variants_evaluated: history.length,
  history,
}