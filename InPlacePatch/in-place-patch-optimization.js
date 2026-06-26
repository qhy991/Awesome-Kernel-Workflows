export const meta = {
  name: 'in-place-patch-optimization',
  description: 'Iterative in-place patch -> project-native build -> test -> benchmark loop for kernels embedded in a larger codebase (no exp_dir variant materialization)',
  whenToUse: 'When the kernel under optimization cannot be compiled standalone (depends on project headers, template instantiation, or dispatch tables - e.g. llama.cpp ggml-cuda kernels). Uses the project build/test/benchmark commands verbatim. Each iteration: snapshot original, propose ONE focused patch via Edit, run project build, run project test, run project benchmark, keep if (correct AND faster) else revert.',
  phases: [
    { title: 'Snapshot',  detail: 'Read kernel + back up original bytes in-memory for revert' },
    { title: 'Baseline',  detail: 'Build + test + benchmark the unmodified kernel to set the baseline number' },
    { title: 'Propose',   detail: 'Subagent reads kernel + handoff and applies ONE focused Edit to the kernel file' },
    { title: 'Build',     detail: 'Run project build_command verbatim; on failure revert + record + continue' },
    { title: 'Test',      detail: 'Run project test_command; parse PASS/FAIL via correctness regex' },
    { title: 'Bench',     detail: 'Run project benchmark_command across N workload points; extract latency' },
    { title: 'Decide',    detail: 'Keep iff correct AND speedup>1 across the chosen aggregate metric; else revert' },
    { title: 'Report',    detail: 'Emit best speedup, history of attempts, final kernel snippet' },
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

const WORKFLOW_NAME = 'in-place-patch-optimization'

// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.

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
    const re = /(\w[\w.-]*)=("(?:\\"|[^"])*"|'(?:\\'|[^'])*'|\S+)/g
    let m
    while ((m = re.exec(trimmed)) !== null) {
      let v = m[2]
      if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
        v = v.slice(1, -1)
      }
      out[m[1]] = v
    }
    return out
  }
  return {}
}
// eslint-disable-next-line no-global-assign
args = __unwrapArgs(typeof args === 'undefined' ? undefined : args)
// --- END inlined arg_guard ---

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
  return null
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

// --- BEGIN inlined grounding contract (mirrors _substrate/grounding.js) ---
const GROUNDING_INSTRUCTION = [
  'GROUNDING CONTRACT (mandatory):',
  '',
  '1. Before reporting any numeric or categorical result, you MUST have executed',
  '   a real Bash command (compile / test / benchmark / profile) that produced',
  '   the value. Citing a value from imagination, prior knowledge, or analogy',
  '   is a contract violation.',
  '',
  '2. If the workflow tells you to run a tool or script that does NOT exist on',
  '   disk or on PATH, do NOT try to substitute, simulate, or rationalize. Return:',
  '       { "grounded": false, "missing": "<exact tool or script that was absent>" }',
  '   and stop. The schema accepts this shape.',
  '',
  '3. If a Bash command fails (non-zero exit), do NOT invent the value it would',
  '   have produced. Return:',
  '       { "grounded": false, "error": "<stderr tail or short summary>" }',
  '   and stop.',
  '',
  '4. Numeric fields in the schema (latency_ms, speedup, occupancy, etc.) are',
  '   REJECTED downstream when grounded === false. Do not work around the',
  '   schema by filling them with placeholders, zeros, or "typical" values.',
  '',
  '5. The string "fabricated", "simulated", "estimated", or "placeholder" must',
  '   never appear in a value you report as measured.',
].join('\n')

function withGroundingFields(schema) {
  schema.properties = schema.properties || {}
  if (!schema.properties.grounded) {
    schema.properties.grounded = {
      type: 'boolean',
      description: 'true iff every reported numeric value came from a real executed command (see GROUNDING_INSTRUCTION).',
    }
  }
  if (!schema.properties.missing) {
    schema.properties.missing = {
      type: 'string',
      description: 'When grounded=false: the absent tool or script that prevented grounding.',
    }
  }
  if (!schema.properties.error) {
    schema.properties.error = {
      type: 'string',
      description: 'When grounded=false: short stderr / failure summary.',
    }
  }
  return schema
}

function classifyResult(agentResult) {
  if (agentResult == null) return { state: 'failed', error: 'agent returned null' }
  if (agentResult.grounded === false) {
    return {
      state: 'not_grounded',
      missing: agentResult.missing || null,
      error: agentResult.error || null,
    }
  }
  return { state: 'grounded', value: agentResult }
}

// Propose step: no measured numerics — grounded:false must not strip applied/title/summary.
function classifyProposalResult(agentResult) {
  if (agentResult == null) return { state: 'failed', error: 'agent returned null', value: null }
  if (typeof agentResult.applied !== 'boolean') {
    return { state: 'failed', error: 'proposal missing applied (boolean)', value: agentResult }
  }
  return { state: 'ok', value: agentResult }
}
// --- END inlined grounding contract ---

// =============================================================================
// In-Place Patch Optimization
// =============================================================================
//
// Why this workflow exists:
//   The other 27 AKW workflows assume the evaluation unit is `exp_dir/variants/<n>.cu`
//   compiled by hipcc/nvcc into a single .so and dlopen'd by a python harness. That
//   contract is incompatible with kernels embedded in a larger codebase (llama.cpp's
//   fattn-rdna-apu.cuh, PyTorch's aten kernels, etc.) because the kernel depends on
//   project headers, template instantiation chains, dispatch tables, or build-time
//   defines that cannot be replicated in an isolated exp_dir.
//
//   This workflow drops the "materialize variants" contract entirely. It treats the
//   kernel file as a live source location in a real project, applies Edits in-place,
//   and delegates build/test/bench to the project's own commands - which can be
//   `cmake --build ...`, `make ...`, `bazel test //...`, anything.
//
// Required args:
//   kernel_path        Absolute path to the kernel file inside the project
//   build_command      Project build command, run verbatim (e.g. "cmake --build build-amd ...")
//   test_command       Correctness command, run verbatim
//   benchmark_command  Performance command, run verbatim. Stdout must contain a parseable latency.
//
// Optional args:
//   correctness_regex  Regex that must match test_command stdout for a PASS (default: empty -> rely on exit code)
//   latency_regex      Regex with one capture group containing the candidate latency number
//                      (default: extracts the first float on a line containing "us" or "ms")
//   latency_unit       "us" or "ms" (default "us")
//   tolerance          Correctness NMSE tolerance (informational, passed to subagent)
//   max_iterations     Maximum optimization attempts (default 5)
//   target_speedup     Stop early when reached (default 1.05)
//   op_description     Free-text context, prepended to the proposer prompt
//   handoff_context    KerSor cross-round handoff (carried in op_description if separate)
//   workload_axes      Comma-separated workload labels for the benchmark sweep (informational)
//   max_proposals_per_iter  Number of independent proposals to race per iteration (default 1)
//   backup_dir         Where to stash the original file copy (default: <kernel_path>.in_place_patch.bak)
//
// =============================================================================

const KERNEL_PATH       = args.kernel_path || ''
const BUILD_CMD         = args.build_command || ''
const TEST_CMD          = args.test_command || ''
const BENCH_CMD         = args.benchmark_command || ''
const CORRECTNESS_REGEX = args.correctness_regex || ''
const LATENCY_REGEX     = args.latency_regex || '([0-9]+\\.?[0-9]*)\\s*(?:us|ms)'
const LATENCY_UNIT      = (args.latency_unit || 'us').toLowerCase()
const TOLERANCE         = args.tolerance != null ? Number(args.tolerance) : 5e-4
const MAX_ITER          = Math.max(1, parseInt(args.max_iterations || 5, 10))
const TARGET_SPEEDUP    = parseFloat(args.target_speedup || 1.05)
const OP_DESC           = args.op_description || ''
const HANDOFF           = args.handoff_context || ''
const WORKLOAD_AXES     = args.workload_axes || ''
const MAX_PROPOSALS     = Math.max(1, parseInt(args.max_proposals_per_iter || 1, 10))
const EXPDIR            = args.exp_dir || '.'
const BACKUP_PATH       = args.backup_dir
  ? `${args.backup_dir}/in_place_patch.bak`
  : `${KERNEL_PATH}.in_place_patch.bak`

// Dirty flag: true iff an Edit has been applied to KERNEL_PATH but not yet
// validated (build/test/bench) or reverted. Every applied edit MUST be resolved
// within its iteration (revert or keep-and-resnapshot), so this is normally false
// at loop end. The exit safety net restores from BACKUP_PATH if it is ever true,
// so an unvalidated patch can never dirty the worktree on exit.
let dirty = false

// Fail-fast required-arg validation. No fallbacks; missing args mean the
// caller did not ground the workflow.
const MISSING = []
if (!KERNEL_PATH) MISSING.push('kernel_path')
if (!BUILD_CMD)   MISSING.push('build_command')
if (!TEST_CMD)    MISSING.push('test_command')
if (!BENCH_CMD)   MISSING.push('benchmark_command')
if (MISSING.length) {
  throw new Error(
    `in-place-patch-optimization: missing required arg(s): ${MISSING.join(', ')}. ` +
    `This workflow does not synthesize substitutes; provide a real project command for each.`
  )
}

// =============================================================================
// Schemas
// =============================================================================

const PROPOSAL_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    title: { type: 'string', description: 'Short label for this patch (e.g. "remove sync after KQ write")' },
    rationale: { type: 'string', description: 'Why this change should improve performance, citing bottleneck evidence' },
    risk: { type: 'string', enum: ['low', 'medium', 'high'], description: 'Author estimate of correctness risk' },
    applied: { type: 'boolean', description: 'true iff the subagent successfully applied an Edit to the kernel file' },
    summary: { type: 'string', description: 'One-line diff summary (e.g. "removed __syncthreads at line 712")' },
  },
  required: ['title', 'rationale', 'applied'],
})

const BUILD_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    stderr_tail: { type: 'string' },
    duration_sec: { type: 'number' },
  },
  required: ['ok'],
})

const TEST_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    correct: { type: 'boolean' },
    detail: { type: 'string', description: 'Pass count / failure summary / NMSE if reported' },
  },
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
    aggregate_latency: { type: 'number', description: 'Single number used to compare iterations (mean / geomean / specific point)' },
    aggregate_strategy: { type: 'string', description: 'How aggregate_latency was computed' },
  },
  required: ['aggregate_latency'],
})

// =============================================================================
// Phase 0: Snapshot the original kernel so we can always revert
// =============================================================================

phase('Snapshot')
log(`Kernel: ${KERNEL_PATH}`)
log(`Backup will live at: ${BACKUP_PATH}`)

const snapshotAgent = await agentRetry(() => agent(
  [
    `Read the file at ${KERNEL_PATH} and make a byte-exact copy at ${BACKUP_PATH} using Bash (cp on unix, copy on windows).`,
    `Then report whether the backup file now exists and is non-empty.`,
    `Do NOT modify ${KERNEL_PATH} yet.`,
    GROUNDING_INSTRUCTION,
    `\n\n# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\nAppend exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\nThen append:\n{"workflow":"${WORKFLOW_NAME}","phase":"Snapshot","ts":"<ts>","status":"done","technique":"backup_original_kernel","note":"<backup path + whether backup is non-empty>"}`,
  ].join('\n\n'),
  {
    phase: 'Snapshot',
    schema: withGroundingFields({
      type: 'object',
      properties: { backed_up: { type: 'boolean' }, size_bytes: { type: 'number' } },
      required: ['backed_up'],
    }),
  }
), { retries: 5 })
const snap = classifyResult(snapshotAgent)
if (snap.state !== 'grounded' || !snap.value.backed_up) {
  return {
    ok: false,
    grounded: snap.state === 'grounded',
    error: `failed to snapshot kernel: ${JSON.stringify(snap)}`,
    speedup: null,
    best_kernel_code: null,
    history: [],
  }
}

// =============================================================================
// profiling-strategist: this workflow has NO native profiler (it relies on the
// project's own build/test/benchmark). The strategist ADDS optional, routed
// profiling AVAILABILITY without forcing any tool. The agent only classifies the
// task (fuzzy op_class/size); the substrate strategist DETERMINISTICALLY picks
// the method and STAMPS confidence (measured/inferred/hypothesized) -- the model
// must NOT assign confidence. Default native_profiler/measured is only overwritten
// on a valid response, so the build/test/bench loop below is unchanged either way.
// =============================================================================

const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }
const SUBSTRATE = args.substrate_dir || '_substrate'
const PY = args.substrate_command_prefix || ''
const BACKEND_MANIFEST = args.backend_manifest || `${SUBSTRATE}/backends/cuda/manifest.json`
const STRATEGIST_SIZE = args.profiling_size || 'large'
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured' }
const _pd = await agentRetry(() => agent(`Read ${KERNEL_PATH}; classify its op_class (one of attention|gemm|elementwise|reduction|default; op_description='${OP_DESC}') and size (tiny|small|large; default '${STRATEGIST_SIZE}'). Then run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${BACKEND_MANIFEST} --task <op_class> --size <size> --cache ${EXPDIR}/prof_cache.json --trajectory ${EXPDIR}/genome.jsonl\`.
Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`, {
  model: MODEL.mechanical,
  label: 'profiling-strategist',
  phase: 'Snapshot',
  schema: JSON_PASSTHROUGH,
}), { retries: 5 })
if (_pd && _pd.method) PROFILING_DECISION = _pd
log(`Profiling-strategist: method=${PROFILING_DECISION.method}, confidence=${PROFILING_DECISION.confidence}`)

// =============================================================================
// Helper: build / test / bench a SINGLE candidate (the current on-disk kernel)
// =============================================================================

async function buildCurrent(label) {
  const r = await agentRetry(() => agent(
    [
      `Run the project build command and report whether it succeeded.`,
      `Command: ${BUILD_CMD}`,
      `If the command exits non-zero, include the last ~30 lines of stderr in stderr_tail.`,
      `Measure wall-clock duration in seconds.`,
      GROUNDING_INSTRUCTION,
    ].join('\n\n'),
    { phase: 'Build', label: `build:${label}`, schema: BUILD_SCHEMA }
  ), { retries: 5 })
  return classifyResult(r)
}

async function testCurrent(label) {
  const correctnessHint = CORRECTNESS_REGEX
    ? `Treat the run as CORRECT iff the regex /${CORRECTNESS_REGEX}/ matches the stdout AND the exit code is 0.`
    : `Treat the run as CORRECT iff the exit code is 0.`
  const r = await agentRetry(() => agent(
    [
      `Run the project correctness command and report PASS/FAIL.`,
      `Command: ${TEST_CMD}`,
      correctnessHint,
      `Tolerance reference (NMSE): ${TOLERANCE}.`,
      GROUNDING_INSTRUCTION,
    ].join('\n\n'),
    { phase: 'Test', label: `test:${label}`, schema: TEST_SCHEMA }
  ), { retries: 5 })
  return classifyResult(r)
}

async function benchCurrent(label) {
  const r = await agentRetry(() => agent(
    [
      `Run the project benchmark command and extract per-workload latency.`,
      `Command: ${BENCH_CMD}`,
      `Latency regex: /${LATENCY_REGEX}/ - the first capture group is the number, the unit is ${LATENCY_UNIT}.`,
      `If the output has multiple workload rows (e.g. kv=2048, kv=4096, ...), extract one latency per workload and emit them in latency_per_workload.`,
      `Compute aggregate_latency as the GEOMEAN of the per-workload latencies (so a single regression on one point cannot dominate).`,
      `Set aggregate_strategy to a short description (e.g. "geomean of 4 kv points").`,
      WORKLOAD_AXES ? `Workload axes hint: ${WORKLOAD_AXES}` : '',
      GROUNDING_INSTRUCTION,
      `\n\n# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\nAppend exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\nThen append, using the value you just measured (status="done" if a latency was parsed, else "error"):\n{"workflow":"${WORKFLOW_NAME}","phase":"Bench","ts":"<ts>","status":"<done|error>","candidate_id":"${label}","speedup":null,"technique":"benchmark_measurement","note":"<aggregate_latency + unit + aggregate_strategy, or the parse failure reason>"}`,
    ].filter(Boolean).join('\n\n'),
    { phase: 'Bench', label: `bench:${label}`, schema: BENCH_SCHEMA }
  ), { retries: 5 })
  return classifyResult(r)
}

async function revertToBackup(reason) {
  log(`Reverting kernel: ${reason}`)
  await agentRetry(() => agent(
    `Copy ${BACKUP_PATH} over ${KERNEL_PATH} (overwrite) using Bash. Then report whether the file is now byte-equal to the backup.`,
    {
      phase: 'Decide',
      label: 'revert',
      schema: withGroundingFields({
        type: 'object',
        properties: { reverted: { type: 'boolean' } },
        required: ['reverted'],
      }),
    }
  ), { retries: 5 })
  dirty = false  // a revert restores KERNEL_PATH to BACKUP_PATH -> no unvalidated edit remains
}

// =============================================================================
// Phase 1: Baseline measurement (UNMODIFIED kernel)
// =============================================================================

phase('Baseline')
const baseBuild = await buildCurrent('baseline')
if (baseBuild.state !== 'grounded' || !baseBuild.value.ok) {
  return {
    ok: false,
    grounded: baseBuild.state === 'grounded',
    error: `baseline build failed: ${JSON.stringify(baseBuild)}`,
    speedup: null,
    best_kernel_code: null,
    history: [],
  }
}
const baseTest = await testCurrent('baseline')
if (baseTest.state !== 'grounded' || !baseTest.value.correct) {
  return {
    ok: false,
    grounded: baseTest.state === 'grounded',
    error: `baseline test failed (kernel is broken before any patch): ${JSON.stringify(baseTest)}`,
    speedup: null,
    best_kernel_code: null,
    history: [],
  }
}
const baseBench = await benchCurrent('baseline')
if (baseBench.state !== 'grounded' || !baseBench.value.aggregate_latency) {
  return {
    ok: false,
    grounded: baseBench.state === 'grounded',
    error: `baseline benchmark could not extract a latency: ${JSON.stringify(baseBench)}`,
    speedup: null,
    best_kernel_code: null,
    history: [],
  }
}
const BASELINE_LATENCY = baseBench.value.aggregate_latency
log(`Baseline aggregate latency: ${BASELINE_LATENCY} ${LATENCY_UNIT} (${baseBench.value.aggregate_strategy || 'n/a'})`)

// =============================================================================
// Phase 2..N: Iterate
// =============================================================================

let bestSpeedup = 1.0
let bestLatency = BASELINE_LATENCY
let bestPatchTitle = '(unmodified baseline)'
const history = []

for (let iter = 1; iter <= MAX_ITER; ++iter) {
  if (bestSpeedup >= TARGET_SPEEDUP) {
    log(`Target speedup ${TARGET_SPEEDUP}x reached at iter ${iter - 1}. Stopping.`)
    break
  }
  log(`\n=== Iteration ${iter}/${MAX_ITER} ===`)

  const priorAttempts = history
    .map(h => `  - ${h.title} -> ${h.verdict} (speedup=${h.speedup?.toFixed?.(3) ?? 'n/a'})`)
    .join('\n') || '  (none yet)'

  const proposeContext = [
    OP_DESC ? `Task context: ${OP_DESC}` : '',
    HANDOFF ? `Handoff from prior rounds:\n${HANDOFF}` : '',
    `Kernel file: ${KERNEL_PATH}`,
    `Baseline latency: ${BASELINE_LATENCY} ${LATENCY_UNIT}`,
    `Best so far: ${bestSpeedup.toFixed(3)}x (${bestLatency} ${LATENCY_UNIT}, "${bestPatchTitle}")`,
    `Tolerance (NMSE): ${TOLERANCE}`,
    `Prior attempts:\n${priorAttempts}`,
    ``,
    `Propose ONE focused, low-risk change to the kernel file. Read the file first.`,
    `Apply it using Edit. Make exactly ONE Edit call - no shotgun changes.`,
    `Do NOT modify any other file. Do NOT run build/test/bench yourself - the workflow does that.`,
    `Avoid strategies already in the "Prior attempts" list above.`,
    `If you cannot find a productive change, return applied=false with the reason in summary.`,
    `This step ONLY applies an Edit. Report applied (true/false), title, and a one-line summary. Do NOT run build/test/bench (the workflow does) and do NOT report numeric values here — grounding of measurements is the workflow's job, not yours. Your Edit landing on disk is the only thing that matters for this step.`,
    `Profiling-strategist selected method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}', evidence='${PROFILING_DECISION.evidence_source || 'ncu'}'). If method!=='native_profiler', no native profiler is available — when reasoning about the bottleneck, derive memory-vs-compute-bound hints from the benchmark throughput the workflow measures, and tag such hints evidence='${PROFILING_DECISION.evidence_source || 'profile_heuristic'}'. Never fabricate profiler counters.`,
    `\n\n# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)\nAppend exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ\nThen append (this is optimization iteration ${iter}):\n{"workflow":"${WORKFLOW_NAME}","phase":"Propose","ts":"<ts>","status":"done","candidate_id":"iter-${iter}","technique":"<the focused change you applied, or none if applied=false>","speedup":null,"note":"<one-line diff summary of the Edit, or why no change>"}`,
  ].filter(Boolean).join('\n\n')

  const propResult = await agentRetry(() => agent(proposeContext, {
    phase: 'Propose',
    label: `propose:iter${iter}`,
    schema: PROPOSAL_SCHEMA,
  }), { retries: 5 })
  const prop = classifyProposalResult(propResult)

  // Gate ONLY on whether an Edit was applied. Grounding is irrelevant to the
  // propose step (it produces no numeric values; the workflow measures those), so
  // an Edit that landed on disk must proceed to build/test/bench — NOT be skipped
  // because the propose agent returned grounded:false. Skipping an applied Edit
  // would leave an unvalidated patch on the worktree with zero measurement.
  if (prop.state === 'failed' || !prop.value || !prop.value.applied) {
    const title = (prop.value && prop.value.title) || '(no change)'
    history.push({
      iter,
      title,
      verdict: prop.state === 'failed' ? 'proposal_failed' : 'no_change',
      speedup: null,
      detail: (prop.value && prop.value.summary) || prop.error || 'no edit applied',
    })
    log(`Iter ${iter}: no change proposed; continuing.`)
    continue
  }

  const title = prop.value.title
  dirty = true  // an Edit is now on disk; build/test/bench below must resolve it (revert or keep)

  // Build the patched kernel.
  const b = await buildCurrent(`iter${iter}`)
  if (b.state !== 'grounded' || !b.value.ok) {
    await revertToBackup(`build failed for "${title}"`)
    history.push({
      iter,
      title,
      verdict: 'build_failed',
      speedup: null,
      detail: b.state === 'grounded' ? (b.value.stderr_tail || '').slice(-400) : JSON.stringify(b),
    })
    continue
  }

  // Run correctness test.
  const t = await testCurrent(`iter${iter}`)
  if (t.state !== 'grounded' || !t.value.correct) {
    await revertToBackup(`correctness failed for "${title}"`)
    history.push({
      iter,
      title,
      verdict: 'correctness_failed',
      speedup: null,
      detail: t.state === 'grounded' ? (t.value.detail || '') : JSON.stringify(t),
    })
    continue
  }

  // Run benchmark.
  const m = await benchCurrent(`iter${iter}`)
  if (m.state !== 'grounded' || !m.value.aggregate_latency) {
    await revertToBackup(`benchmark could not be parsed for "${title}"`)
    history.push({
      iter,
      title,
      verdict: 'bench_unparseable',
      speedup: null,
      detail: JSON.stringify(m),
    })
    continue
  }

  const candidateLatency = m.value.aggregate_latency
  const speedup = BASELINE_LATENCY / candidateLatency
  log(`Iter ${iter} "${title}": ${candidateLatency} ${LATENCY_UNIT} (speedup ${speedup.toFixed(3)}x)`)

  if (speedup > bestSpeedup) {
    bestSpeedup = speedup
    bestLatency = candidateLatency
    bestPatchTitle = title
    history.push({
      iter,
      title,
      verdict: 'kept',
      speedup,
      detail: `latency ${candidateLatency} ${LATENCY_UNIT} (${m.value.aggregate_strategy || ''})`,
    })
    // Re-snapshot the now-better kernel so further iterations revert to THIS.
    await agentRetry(() => agent(
      `Copy ${KERNEL_PATH} over ${BACKUP_PATH} (overwrite) using Bash. Confirm the file copy succeeded.`,
      {
        phase: 'Decide',
        label: 'resnapshot',
        schema: withGroundingFields({
          type: 'object',
          properties: { resnapshotted: { type: 'boolean' } },
          required: ['resnapshotted'],
        }),
      }
    ), { retries: 5 })
    dirty = false  // kept: KERNEL_PATH == BACKUP_PATH (validated best), nothing unvalidated on disk
  } else {
    await revertToBackup(`no speedup ("${title}": ${speedup.toFixed(3)}x <= ${bestSpeedup.toFixed(3)}x)`)
    history.push({
      iter,
      title,
      verdict: 'reverted_no_speedup',
      speedup,
      detail: `latency ${candidateLatency} ${LATENCY_UNIT}`,
    })
  }
}

// =============================================================================
// Final report
// =============================================================================

// Exit safety net (Issue 3): if any Edit was applied but never validated/reverted
// (should not happen with the applied-only gate above, but defense-in-depth),
// restore from BACKUP_PATH so the worktree is never left with an unvalidated patch.
if (dirty) {
  await revertToBackup('unvalidated edit left on disk at exit')
}

phase('Report')

// Issue 2: do NOT inline the full kernel file. For large embedded kernels (e.g.
// llama.cpp mmq.cuh, ~4k lines / ~70K tokens) StructuredOutput cannot carry the
// full file and the final-read agent stalls retrying. The caller persists the
// best kernel from best_kernel_path (the file is already patched in place); we
// emit only a short best-patch summary instead of the full file contents.
const finalCode = bestPatchTitle && bestPatchTitle !== '(unmodified baseline)'
  ? `(in-place patched; see best_kernel_path=${KERNEL_PATH}; best patch: "${bestPatchTitle}")`
  : ''

return {
  ok: true,
  grounded: true,
  speedup: bestSpeedup,
  baseline_latency: BASELINE_LATENCY,
  best_latency: bestLatency,
  best_patch_title: bestPatchTitle,
  best_kernel_code: finalCode,
  best_kernel_path: KERNEL_PATH,
  backup_path: BACKUP_PATH,
  latency_unit: LATENCY_UNIT,
  iterations_completed: history.length,
  history,
}
