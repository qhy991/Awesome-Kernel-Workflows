export const meta = {
  name: 'automegakernel-megakernel-optimization',
  description: 'Strict adapter workflow for AutoMegaKernel: drive AMK schedule-space search over ScheduleConfig + kernel_knobs, validated and measured by the AMK harness.',
  whenToUse: 'When you already have an AutoMegaKernel checkout and want a Claude Code Workflow to drive its verified megakernel synthesis loop: amk propose -> amk eval -> amk loop/autoresearch. This is not a standalone reimplementation and not a general CUDA kernel optimizer.',
  phases: [
    { title: 'Setup', detail: 'Verify AutoMegaKernel checkout, CLI, target GPU, and harness availability' },
    { title: 'Surface', detail: 'Read incumbent ScheduleConfig and editable search_space from AMK' },
    { title: 'Baseline', detail: 'Evaluate the incumbent schedule with AMK correctness and latency gates' },
    { title: 'Search', detail: 'Run AMK loop or autoresearch over ScheduleConfig + kernel_knobs' },
    { title: 'Audit', detail: 'Check that all reported latency numbers are paired with correct=true and valid=true' },
    { title: 'Report', detail: 'Return the best AMK config/verdict and fidelity boundary' },
  ],
}

// --- BEGIN model-tier (auto-inserted by scripts/patch-model-tier.js) ---
// Tier-based model routing: mechanical shell/JSON steps use cheaper models;
// audit/report steps use stronger judgment. Tuneable via args.model_*.
const MODEL = {
  mechanical: (typeof args !== 'undefined' && args && args.model_mechanical) || 'haiku',
  profile: (typeof args !== 'undefined' && args && args.model_profile) || 'sonnet',
  judgment: (typeof args !== 'undefined' && args && args.model_judgment) || 'opus',
}
// __modelTierApplied
// --- END model-tier ---

const WORKFLOW_NAME = 'automegakernel-megakernel-optimization'

// --- genome self-report: INLINE (rich, doer-written) ---
// The "__genomeReport" mention is a sentinel so patch-genome-report.js treats
// this file as already handled. The doer prompts below ask agents to append
// result-bearing lines to <exp_dir>/genome.jsonl after each grounded command.

// --- BEGIN inlined arg_guard (Workflow runtime parses scripts as bare scripts) ---
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
    } catch (e) {
      lastError = e
    }
  }
  if (lastError) throw lastError
  if (opts && opts.allowNull === true) return null
  throw new Error(
    `agentRetry: "${(opts && opts.label) || 'agent'}" returned null after ${retries + 1} attempt(s).`,
  )
}

function expect(obj, field, ctx) {
  if (obj == null || obj[field] == null) {
    throw new Error(`agentRetry: required field "${field}" is missing${ctx ? ' from ' + ctx : ''}.`)
  }
  return obj[field]
}

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

// =============================================================================
// AutoMegaKernel Adapter Workflow
// =============================================================================
//
// Source:
//   AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting
//   Megakernel Synthesis, arXiv:2606.09682.
//   Repo: https://github.com/RightNow-AI/AutoMegaKernel
//
// Fidelity boundary:
//   strict adapter to the AutoMegaKernel harness, not a standalone
//   reimplementation. This workflow delegates the authoritative search,
//   validate-before-launch checks, correctness oracle, latency measurement,
//   roofline accounting, keep/revert decisions, and flywheel logging to AMK.
//
// Required args:
//   amk_root       Path to an existing AutoMegaKernel checkout.
//   model          AMK model id, e.g. toy, toy-2L, or a HuggingFace id supported by AMK.
//   target_gpu     AMK GpuTarget id, e.g. rtx5090, b200, h100, a100.
//
// Optional args:
//   mode           loop | autoresearch (default loop)
//   iterations     Number of AMK loop trials when mode=loop (default 8)
//   minutes        AMK autoresearch wall-clock limit when mode=autoresearch (default empty)
//   device         auto | cpu | cuda (default auto)
//   amk_command    AMK executable name/path (default amk)
//   overnight      true/false, passed to AMK autoresearch (default false)
//   cold           true/false, passed to AMK autoresearch (default false)
//   min_speedup    Required best speedup vs AMK incumbent for ok=true (default 1.01)
//   exp_dir        Scratch/report directory for adapter genome lines (default .)
//
// =============================================================================

const AMK_ROOT = args.amk_root || ''
const MODEL_ID = args.model || ''
const TARGET_GPU = args.target_gpu || ''
const MODE = (args.mode || 'loop').toLowerCase()
const ITERATIONS = Math.max(1, parseInt(args.iterations || 8, 10))
const MINUTES = args.minutes != null && args.minutes !== '' ? Math.max(1, parseInt(args.minutes, 10)) : null
const DEVICE = args.device || 'auto'
const AMK_CMD = args.amk_command || 'amk'
const OVERNIGHT = String(args.overnight || 'false').toLowerCase() === 'true'
const COLD = String(args.cold || 'false').toLowerCase() === 'true'
const MIN_SPEEDUP = args.min_speedup != null ? Number(args.min_speedup) : 1.01
const EXP_DIR = args.exp_dir || '.'

const missing = []
if (!AMK_ROOT) missing.push('amk_root')
if (!MODEL_ID) missing.push('model')
if (!TARGET_GPU) missing.push('target_gpu')
if (missing.length) {
  throw new Error(`${WORKFLOW_NAME}: missing required arg(s): ${missing.join(', ')}`)
}
if (!['loop', 'autoresearch'].includes(MODE)) {
  throw new Error(`${WORKFLOW_NAME}: mode must be "loop" or "autoresearch", got "${MODE}"`)
}

const GROUNDING_INSTRUCTION = [
  'GROUNDING CONTRACT (mandatory):',
  '',
  '1. Every numeric result you return must come from a real command executed in the AutoMegaKernel checkout.',
  '2. Do not simulate AMK output. If the AMK command is missing or fails, return grounded=false with missing or error.',
  '3. Do not report latency_us unless the same AMK verdict has valid=true and correct=true.',
  '4. Do not edit AMK raw kernel code, vm/, schedule/, instruction ABI files, or generated megakernel source.',
  '5. This adapter may only drive AMK through its CLI/MCP-equivalent surface: doctor, propose, eval, loop, autoresearch.',
].join('\n')

function withGroundingFields(schema) {
  schema.properties = schema.properties || {}
  if (!schema.properties.grounded) schema.properties.grounded = { type: 'boolean' }
  if (!schema.properties.missing) schema.properties.missing = { type: 'string' }
  if (!schema.properties.error) schema.properties.error = { type: 'string' }
  return schema
}

function classifyResult(agentResult) {
  if (agentResult == null) return { state: 'failed', error: 'agent returned null' }
  if (agentResult.grounded === false) {
    return {
      state: 'not_grounded',
      missing: agentResult.missing || null,
      error: agentResult.error || null,
      value: agentResult,
    }
  }
  return { state: 'grounded', value: agentResult }
}

function boolFlag(name, enabled) {
  return enabled ? ` --${name}` : ''
}

const SETUP_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    amk_ready: { type: 'boolean' },
    cli_version: { type: 'string' },
    cuda_available: { type: 'boolean' },
    device_name: { type: 'string' },
    targets: { type: 'array', items: { type: 'string' } },
    notes: { type: 'array', items: { type: 'string' } },
  },
  required: ['amk_ready'],
})

const SURFACE_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    schedule_config: { type: 'object', additionalProperties: true },
    schedule_id: { type: 'string' },
    search_space: { type: 'object', additionalProperties: true },
    editable_surface: { type: 'array', items: { type: 'string' } },
  },
  required: ['schedule_config', 'search_space'],
})

const VERDICT_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    valid: { type: 'boolean' },
    correct: { type: 'boolean' },
    latency_us: { type: ['number', 'null'] },
    latency_kind: { type: ['string', 'null'] },
    pct_of_roofline: { type: ['number', 'null'] },
    bound_us: { type: ['number', 'null'] },
    schedule_id: { type: 'string' },
    rejected_reason: { type: ['string', 'null'] },
    notes: { type: 'array', items: { type: 'string' } },
    raw: { type: 'object', additionalProperties: true },
  },
  required: ['valid', 'correct'],
})

const SEARCH_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    best_verdict: { type: 'object', additionalProperties: true },
    best_config: { type: 'object', additionalProperties: true },
    rows: { type: 'array', items: { type: 'object', additionalProperties: true } },
    results_tsv: { type: 'string' },
    report_path: { type: 'string' },
    raw: { type: 'object', additionalProperties: true },
  },
  required: ['best_verdict', 'best_config'],
})

const AUDIT_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    safe: { type: 'boolean' },
    speedup: { type: ['number', 'null'] },
    best_latency_us: { type: ['number', 'null'] },
    baseline_latency_us: { type: ['number', 'null'] },
    best_schedule_id: { type: 'string' },
    violations: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['safe', 'violations', 'summary'],
})

// =============================================================================
// Phase 1: Setup
// =============================================================================

phase('Setup')
log(`AMK root: ${AMK_ROOT}`)
log(`Model: ${MODEL_ID}`)
log(`Target GPU: ${TARGET_GPU}`)

const setupAgent = await agentRetry(() => agent(
  [
    `Check the AutoMegaKernel checkout and CLI without modifying source files.`,
    `Run from the checkout: cd ${AMK_ROOT} && ${AMK_CMD} doctor`,
    `Confirm whether the target GPU id "${TARGET_GPU}" appears in the reported targets or is otherwise accepted by AMK.`,
    `If the command is absent or fails, return grounded=false and include the missing tool or stderr summary.`,
    GROUNDING_INSTRUCTION,
    `Genome line: append one JSON line to ${EXP_DIR}/genome.jsonl with workflow="${WORKFLOW_NAME}", phase="Setup", status, and target_gpu.`,
  ].join('\n\n'),
  { model: MODEL.mechanical, label: 'amk-setup', phase: 'Setup', schema: SETUP_SCHEMA }
), { retries: 5, label: 'amk-setup' })
const setup = classifyResult(setupAgent)
if (setup.state !== 'grounded' || !setup.value.amk_ready) {
  return {
    ok: false,
    grounded: setup.state === 'grounded',
    error: `AutoMegaKernel setup failed: ${JSON.stringify(setup)}`,
    fidelity_boundary: 'strict adapter to AMK harness',
  }
}

// =============================================================================
// Phase 2: Surface
// =============================================================================

phase('Surface')

const surfaceAgent = await agentRetry(() => agent(
  [
    `Read the AMK editable search surface.`,
    `Run from the checkout: cd ${AMK_ROOT} && ${AMK_CMD} propose ${MODEL_ID} --gpu ${TARGET_GPU}`,
    `Return the schedule_config, schedule_id if present, and search_space exactly as AMK reports them.`,
    `The editable surface is ScheduleConfig plus optional kernel_knobs. Do not add raw CUDA edits.`,
    GROUNDING_INSTRUCTION,
    `Genome line: append one JSON line to ${EXP_DIR}/genome.jsonl with workflow="${WORKFLOW_NAME}", phase="Surface", status, schedule_id, and the editable knob names.`,
  ].join('\n\n'),
  { model: MODEL.profile, label: 'amk-surface', phase: 'Surface', schema: SURFACE_SCHEMA }
), { retries: 5, label: 'amk-surface' })
const surface = classifyResult(surfaceAgent)
if (surface.state !== 'grounded') {
  return {
    ok: false,
    grounded: false,
    error: `AMK surface discovery failed: ${JSON.stringify(surface)}`,
    fidelity_boundary: 'strict adapter to AMK harness',
  }
}
const scheduleConfig = expect(surface.value, 'schedule_config', 'AMK surface')
const editableSurface = guard(surface.value, 'editable_surface', Object.keys(guard(surface.value, 'search_space', {})))

// =============================================================================
// Phase 3: Baseline incumbent eval
// =============================================================================

phase('Baseline')

const baselineConfigPath = `${EXP_DIR}/amk_incumbent_config.json`
const baselineAgent = await agentRetry(() => agent(
  [
    `Evaluate the incumbent AMK ScheduleConfig before search.`,
    `Write this exact JSON object to ${baselineConfigPath}:`,
    JSON.stringify(scheduleConfig, null, 2),
    `Run from the checkout: cd ${AMK_ROOT} && ${AMK_CMD} eval ${MODEL_ID} --gpu ${TARGET_GPU} --config ${baselineConfigPath} --device ${DEVICE}`,
    `Return AMK's verdict. latency_us must be null unless valid=true and correct=true in that same verdict.`,
    GROUNDING_INSTRUCTION,
    `Genome line: append one JSON line to ${EXP_DIR}/genome.jsonl with workflow="${WORKFLOW_NAME}", phase="Baseline", status, valid, correct, latency_kind, and latency_us if present.`,
  ].join('\n\n'),
  { model: MODEL.profile, label: 'amk-baseline', phase: 'Baseline', schema: VERDICT_SCHEMA }
), { retries: 5, label: 'amk-baseline' })
const baseline = classifyResult(baselineAgent)
if (baseline.state !== 'grounded') {
  return {
    ok: false,
    grounded: false,
    error: `AMK baseline eval failed: ${JSON.stringify(baseline)}`,
    fidelity_boundary: 'strict adapter to AMK harness',
  }
}
if (baseline.value.latency_us != null && (!baseline.value.valid || !baseline.value.correct)) {
  return {
    ok: false,
    grounded: true,
    error: 'AMK baseline verdict violated honesty contract: latency_us reported without valid=true and correct=true.',
    baseline_verdict: baseline.value,
    fidelity_boundary: 'strict adapter to AMK harness',
  }
}

// =============================================================================
// Phase 4: Search
// =============================================================================

phase('Search')

const searchCommand = MODE === 'autoresearch'
  ? [
      `cd ${AMK_ROOT} && ${AMK_CMD} autoresearch ${MODEL_ID}`,
      `--gpu ${TARGET_GPU}`,
      MINUTES != null ? `--minutes ${MINUTES}` : `--iters ${ITERATIONS}`,
      `--device ${DEVICE}`,
      boolFlag('overnight', OVERNIGHT),
      boolFlag('cold', COLD),
    ].filter(Boolean).join(' ')
  : [
      `cd ${AMK_ROOT} && ${AMK_CMD} loop ${MODEL_ID}`,
      `--gpu ${TARGET_GPU}`,
      `--budget ${ITERATIONS}`,
      `--device ${DEVICE}`,
    ].join(' ')

const searchAgent = await agentRetry(() => agent(
  [
    `Run the AMK ${MODE} path over ScheduleConfig plus kernel_knobs. Do not edit raw kernel code.`,
    `Run exactly: ${searchCommand}`,
    `Return the parsed AMK result. Preserve best_verdict, best_config, rows, results_tsv/report_path if present.`,
    `A candidate may be kept only by AMK's own correctness-first, measured-latency keep/revert rule.`,
    GROUNDING_INSTRUCTION,
    `Genome line: append one JSON line to ${EXP_DIR}/genome.jsonl with workflow="${WORKFLOW_NAME}", phase="Search", mode="${MODE}", status, best schedule id, and best latency if present.`,
  ].join('\n\n'),
  { model: MODEL.profile, label: 'amk-search', phase: 'Search', schema: SEARCH_SCHEMA }
), { retries: 5, label: 'amk-search' })
const search = classifyResult(searchAgent)
if (search.state !== 'grounded') {
  return {
    ok: false,
    grounded: false,
    error: `AMK search failed: ${JSON.stringify(search)}`,
    baseline_verdict: baseline.value,
    fidelity_boundary: 'strict adapter to AMK harness',
  }
}

// =============================================================================
// Phase 5: Audit
// =============================================================================

phase('Audit')

const auditAgent = await agentRetry(() => agent(
  [
    `Audit the AMK search result for fidelity and honesty.`,
    `Baseline verdict JSON:`,
    JSON.stringify(baseline.value, null, 2),
    `Search result JSON:`,
    JSON.stringify(search.value, null, 2),
    `Checks:`,
    `- No latency_us field may be non-null unless the same row/verdict has valid=true and correct=true.`,
    `- best_verdict must be valid=true and correct=true to count as a success.`,
    `- speedup is baseline_latency_us / best_latency_us when both numbers are present.`,
    `- This adapter is not a standalone reimplementation; all authority belongs to AMK.`,
    GROUNDING_INSTRUCTION,
    `Genome line: append one JSON line to ${EXP_DIR}/genome.jsonl with workflow="${WORKFLOW_NAME}", phase="Audit", status, safe, speedup, and violation count.`,
  ].join('\n\n'),
  { model: MODEL.judgment, label: 'amk-audit', phase: 'Audit', schema: AUDIT_SCHEMA }
), { retries: 5, label: 'amk-audit' })
const audit = classifyResult(auditAgent)
if (audit.state !== 'grounded') {
  return {
    ok: false,
    grounded: false,
    error: `AMK audit failed: ${JSON.stringify(audit)}`,
    baseline_verdict: baseline.value,
    search_result: search.value,
    fidelity_boundary: 'strict adapter to AMK harness',
  }
}

const speedup = audit.value.speedup != null ? audit.value.speedup : null
const ok = Boolean(audit.value.safe && speedup != null && speedup >= MIN_SPEEDUP)

// =============================================================================
// Phase 6: Report
// =============================================================================

phase('Report')

return {
  ok,
  grounded: true,
  workflow: WORKFLOW_NAME,
  fidelity_boundary: 'strict adapter to AutoMegaKernel harness; not a standalone reimplementation',
  mode: MODE,
  amk_root: AMK_ROOT,
  model: MODEL_ID,
  target_gpu: TARGET_GPU,
  device: DEVICE,
  min_speedup: MIN_SPEEDUP,
  speedup,
  editable_surface: editableSurface,
  baseline_verdict: baseline.value,
  best_verdict: search.value.best_verdict,
  best_config: search.value.best_config,
  audit: audit.value,
  results_tsv: search.value.results_tsv || '',
  report_path: search.value.report_path || '',
}
