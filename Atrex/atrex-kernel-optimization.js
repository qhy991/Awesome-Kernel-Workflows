export const meta = {
  name: 'atrex-kernel-optimization',
  description: 'Strict adapter that launches and audits one Atrex Kernel Agent campaign while leaving optimization and promotion authority with the official orchestrator.',
  whenToUse: 'Use for SOL-ExecBench or native Atrex-Bench operator optimization when an existing Atrex checkout, GPU gateway, platform configuration, and complete command are available.',
  requiredSkills: [],
  optionalSkills: [],
  phases: [
    { title: 'Doctor', detail: 'Verify the Atrex checkout, sole supported entrypoint, command, platform, framework, and output boundary' },
    { title: 'Launch Campaign', detail: 'Execute one official Atrex campaign without duplicating its Long Horizon supervisor' },
    { title: 'Evidence Audit', detail: 'Audit canonical memory, journals, terminal validation, ABBA evidence, and promotion' },
    { title: 'Report', detail: 'Return the exact official promoted candidate and evidence paths with an adapter report' },
  ],
}

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
    } catch (e) {
      lastError = e
    }
  }
  if (lastError) throw lastError
  if (opts && opts.allowNull === true) return null
  throw new Error(
    `agentRetry: "${(opts && opts.label) || 'agent'}" returned null after ${retries + 1} attempt(s) ` +
    `(agent skipped or terminal API failure after retries).`,
  )
}

function guard(obj, field, fallback) {
  if (obj == null || obj[field] == null) return fallback
  return obj[field]
}
// --- END inlined agent-retry scaffolding ---

// Atrex campaigns are long-running; retain the canonical timeout helper while
// supplying this workflow's documented default before the helper reads args.
if (args.turn_timeout_min == null) args.turn_timeout_min = 720

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

const WORKFLOW_NAME = 'atrex-kernel-optimization'
const ATREX_ROOT = args.atrex_root || ''
const OPERATOR_INPUT = args.operator_input || ''
const ATREX_COMMAND = args.atrex_command || ''
const PLATFORM = args.platform || ''
const FRAMEWORK = String(args.framework || '').toLowerCase()
const MODE = String(args.mode || 'production').toLowerCase()
const MAX_ITERS = Math.max(1, Number(args.max_iters || 300))
const EXP_DIR = args.exp_dir || '/tmp/atrex-akw'
const MIN_SPEEDUP = Number(args.min_speedup || 1.0)
const MODEL = args.model || 'sonnet'

if (!ATREX_ROOT || !OPERATOR_INPUT || !ATREX_COMMAND || !PLATFORM || !FRAMEWORK) {
  return {
    ok: false,
    error: 'missing_atrex_contract',
    missing: { atrex_root: !ATREX_ROOT, operator_input: !OPERATOR_INPUT, atrex_command: !ATREX_COMMAND, platform: !PLATFORM, framework: !FRAMEWORK },
    reason: 'The strict adapter needs an existing checkout and a complete caller-owned command for the official Atrex orchestrator.',
  }
}
if (!['cuda', 'triton', 'gluon', 'flydsl', 'cutedsl', 'rocm'].includes(FRAMEWORK)) {
  return { ok: false, error: 'unsupported_framework', framework: FRAMEWORK }
}

const ANY_OBJECT = { type: 'object', additionalProperties: true }
const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    campaign_ok: { type: 'boolean' }, promoted: { type: 'boolean' }, correct: { type: 'boolean' },
    best_kernel_path: { type: 'string' }, best_speedup: { type: ['number', 'null'] },
    canonical_memory_path: { type: 'string' }, abba_evidence_path: { type: 'string' },
    journal_paths: { type: 'array', items: { type: 'string' } }, audit_notes: { type: 'array', items: { type: 'string' } },
  },
  required: ['campaign_ok', 'promoted', 'correct', 'best_kernel_path', 'best_speedup', 'canonical_memory_path', 'abba_evidence_path', 'journal_paths', 'audit_notes'],
}

function expandedCommandContract() {
  return [
    `Atrex command template: ${ATREX_COMMAND}`,
    `Atrex root: ${ATREX_ROOT}`,
    `Operator input: ${OPERATOR_INPUT}`,
    `Platform: ${PLATFORM}`,
    `Framework: ${FRAMEWORK}`,
    `Mode: ${MODE}`,
    `Max iterations: ${MAX_ITERS}`,
    `Experiment directory: ${EXP_DIR}`,
    'Substitute only placeholders that occur: {atrex_root}, {operator_input}, {platform}, {framework}, {mode}, {max_iters}, {exp_dir}.',
    'The expanded command must invoke the checkout\'s orchestrator/optimize.py, the sole supported optimization entrypoint.',
    `The command must place campaign state under ${EXP_DIR}; do not silently choose another workspace.`,
  ].join('\n')
}

log('Phase 1/4: Doctor')
const doctor = await withTurnTimeout(agentRetry(
  () => agent(`
Act as a strict Atrex Kernel Agent adapter doctor.

${expandedCommandContract()}

Inspect the checkout README, docs/design.md, docs/quickstart.md, orchestrator/optimize.py, configured gateway/platform files, and relevant schemas. Do not modify the checkout. Verify that the command targets the official entrypoint and does not call long_horizon/ as a second CLI. Verify that campaign output remains below ${EXP_DIR}. Create ${EXP_DIR}/adapter-contract.json with the expanded command and checked facts.

Do not recreate Atrex's inner loop in this workflow: its supervisor owns V0/V1 establishment, clean-session episodes, profiling, optimization dropout, GPU-wiki/reference retrieval, budgets, complete-workload terminal validation, same-allocation ABBA verification, and squash promotion.
${__experienceBlock()}${__attemptBlock()}
Return ready, expanded_command, contract_path, detected_version, and notes.
`, { label: 'atrex-doctor', phase: 'Doctor', model: MODEL, schema: ANY_OBJECT }),
  { retries: 5, label: 'atrex-doctor' },
), 'Atrex doctor')

if (!guard(doctor, 'ready', false)) return { ok: false, error: 'atrex_doctor_failed', doctor }
const expandedCommand = guard(doctor, 'expanded_command', '')
if (!expandedCommand) return { ok: false, error: 'missing_expanded_atrex_command', doctor }

log('Phase 2/4: Launch Campaign')
const launch = await withTurnTimeout(agentRetry(
  () => agent(`
Launch exactly one official Atrex Kernel Agent campaign.

Expanded command: ${expandedCommand}
Experiment directory: ${EXP_DIR}

Execute the command exactly once and let orchestrator/optimize.py own every inner decision. Do not edit candidates yourself, do not launch long_horizon/ directly, do not override its budgets or promotion gates, and do not treat memory/live.json as canonical evidence. Preserve stdout, stderr, return code, and campaign paths under ${EXP_DIR}/adapter-launch.

Return command, exit_code, campaign_root, stdout_path, stderr_path, and terminal_summary. A failed command remains a failed campaign; do not fabricate recovery evidence.
`, { label: 'atrex-launch-campaign', phase: 'Launch Campaign', model: MODEL, schema: ANY_OBJECT }),
  // Campaign launch is non-idempotent: wrap for the universal null guard, but
  // never replay the command after a terminal/null return.
  { retries: 0, label: 'atrex-launch-campaign' },
), 'Atrex official campaign')

log('Phase 3/4: Evidence Audit')
const audit = await withTurnTimeout(agentRetry(
  () => agent(`
Audit the completed Atrex campaign without changing it.

Atrex root: ${ATREX_ROOT}
Launch record: ${JSON.stringify(launch)}
Experiment directory: ${EXP_DIR}

Follow the official schemas and identify the latest canonical memory/v<N>.json, episode journals, terminal handoff/validation, same-allocation ABBA evidence, aggregation provenance, policy/protected-path verdict, and squash-promoted artifact. memory/live.json is observability only and must never be used as promotion evidence. Validate the complete workload set, not a single friendly shape. A promoted result is valid only when the official supervisor records correctness and a strict ABBA improvement.

Return exact existing paths and verdicts. Use false/null/empty strings for absent evidence; do not infer success from process exit alone.
`, { label: 'atrex-evidence-audit', phase: 'Evidence Audit', model: MODEL, schema: AUDIT_SCHEMA }),
  { retries: 5, label: 'atrex-evidence-audit' },
), 'Atrex evidence audit')

log('Phase 4/4: Report')
const report = await withTurnTimeout(agentRetry(
  () => agent(`
Write the strict-adapter report for one Atrex campaign.

Doctor: ${JSON.stringify(doctor)}
Launch: ${JSON.stringify(launch)}
Audit: ${JSON.stringify(audit)}

Write ${EXP_DIR}/adapter-report.md and append one run record to ${EXP_DIR}/genome.jsonl. Distinguish official canonical evidence from adapter observations. Document whether V0/V1, complete-workload validation, ABBA verification, and squash promotion were present. State that Atrex's profile-driven measure-revise loop, optimization dropout, knowledge base, hidden workload handling, and supervisor were delegated to the official checkout rather than reimplemented here.

Return report_path and report_complete.
`, { label: 'atrex-report', phase: 'Report', model: MODEL, schema: ANY_OBJECT }),
  { retries: 5, label: 'atrex-report' },
), 'Atrex report')

const speedup = Number(guard(audit, 'best_speedup', 0))
const bestKernelPath = guard(audit, 'best_kernel_path', '')
const officiallyPromoted = guard(audit, 'campaign_ok', false) && guard(audit, 'promoted', false) && guard(audit, 'correct', false) && bestKernelPath !== ''

return {
  ok: officiallyPromoted && speedup >= MIN_SPEEDUP && guard(report, 'report_complete', false),
  workflow: WORKFLOW_NAME,
  best_kernel_path: bestKernelPath,
  best_speedup: speedup,
  canonical_memory_path: guard(audit, 'canonical_memory_path', ''),
  abba_evidence_path: guard(audit, 'abba_evidence_path', ''),
  journal_paths: guard(audit, 'journal_paths', []),
  report_path: guard(report, 'report_path', `${EXP_DIR}/adapter-report.md`),
  official_campaign: launch,
  fidelity_boundary: 'strict_high_fidelity',
}
