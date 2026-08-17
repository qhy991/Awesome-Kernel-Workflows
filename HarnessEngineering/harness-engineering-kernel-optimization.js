export const meta = {
  name: 'harness-engineering-kernel-optimization',
  description: 'Profile-backed kernel optimization under a frozen official-style harness contract with correctness-first promotion and complete artifact retention.',
  whenToUse: 'Use for an existing CUDA, Triton, or CuTe kernel when a real harness owns compile, correctness, verification, and timing. The workflow does not invent missing evaluator commands.',
  requiredSkills: [],
  optionalSkills: [],
  phases: [
    { title: 'Freeze Contract', detail: 'Inspect the harness and materialize the immutable run contract under exp_dir' },
    { title: 'Baseline', detail: 'Establish correctness-gated baseline evidence with the supplied commands' },
    { title: 'Profile and Decide', detail: 'Select one bounded hypothesis from current evidence and optional profiles' },
    { title: 'Implement', detail: 'Create one candidate under exp_dir without mutating the source kernel or harness' },
    { title: 'Official Gate', detail: 'Run exact correctness, verification, and timing commands and promote deterministically' },
    { title: 'Audit and Report', detail: 'Audit evidence-to-artifact binding and write the final report' },
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
    `agentRetry: "${(opts && opts.label) || 'agent'}" returned null after ${retries + 1} attempt(s) ` +
    `(agent skipped or terminal API failure after retries).`,
  )
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

const WORKFLOW_NAME = 'harness-engineering-kernel-optimization'
const HARNESS_ROOT = args.harness_root || ''
const KERNEL_PATH = args.kernel_path || ''
const PROBLEM_PATH = args.problem_path || ''
const BACKEND = String(args.backend || 'cuda').toLowerCase()
const TEST_COMMAND = args.test_command || ''
const BENCHMARK_COMMAND = args.benchmark_command || ''
const PROFILE_COMMAND = args.profile_command || ''
const VERIFICATION_COMMAND = args.verification_command || ''
const VERIFICATION_PROFILE = String(args.verification_profile || 'contract-grade').toLowerCase()
const ITERATIONS = Math.max(1, Number(args.iterations || 4))
const MIN_SPEEDUP = Number(args.min_speedup || 1.01)
const EXP_DIR = args.exp_dir || '/tmp/harness-engineering'
const MODEL = args.model || 'sonnet'

if (!HARNESS_ROOT || !KERNEL_PATH || !TEST_COMMAND || !BENCHMARK_COMMAND) {
  return {
    ok: false,
    error: 'missing_harness_contract',
    missing: { harness_root: !HARNESS_ROOT, kernel_path: !KERNEL_PATH, test_command: !TEST_COMMAND, benchmark_command: !BENCHMARK_COMMAND },
    reason: 'Harness Engineering requires an existing kernel plus harness-owned correctness and timing commands.',
  }
}
if (!['cuda', 'triton', 'cute'].includes(BACKEND)) return { ok: false, error: 'unsupported_backend', backend: BACKEND }
if (!['contract-grade', 'kernelbench-verified', 'kernelgenbench', 'custom'].includes(VERIFICATION_PROFILE)) {
  return { ok: false, error: 'unsupported_verification_profile', verification_profile: VERIFICATION_PROFILE }
}

const ANY_OBJECT = { type: 'object', additionalProperties: true }
const EVIDENCE_SCHEMA = {
  type: 'object',
  properties: {
    candidate_path: { type: 'string' }, compiled: { type: 'boolean' }, correct: { type: 'boolean' },
    verified: { type: 'boolean' }, latency_ms: { type: ['number', 'null'] }, speedup: { type: ['number', 'null'] },
    evidence_path: { type: 'string' }, artifact_paths: { type: 'array', items: { type: 'string' } }, notes: { type: 'array', items: { type: 'string' } },
  },
  required: ['candidate_path', 'compiled', 'correct', 'verified', 'latency_ms', 'speedup', 'evidence_path', 'artifact_paths', 'notes'],
}

function commandContract(candidatePath, resultDir) {
  return [
    `Correctness command: ${TEST_COMMAND}`,
    `Verification command: ${VERIFICATION_COMMAND || '(not supplied)'}`,
    `Benchmark command: ${BENCHMARK_COMMAND}`,
    `Profile command: ${PROFILE_COMMAND || '(not supplied)'}`,
    `Candidate: ${candidatePath}`,
    `Result directory: ${resultDir}`,
    'Substitute only placeholders that occur: {candidate_path}, {kernel_path}, {problem_path}, {harness_root}, {result_path}, {artifact_path}, {exp_dir}.',
    'Run user-supplied commands exactly after substitution. Never replace them with guessed compiler, test, profiler, or timing commands.',
    'The benchmark command may run only after compiled=true, correct=true, and (when verification_command is supplied) verified=true.',
  ].join('\n')
}

function evidenceIsPromotable(evidence) {
  return guard(evidence, 'compiled', false) && guard(evidence, 'correct', false) && (!VERIFICATION_COMMAND || guard(evidence, 'verified', false))
}

function measuredSpeedup(baseline, candidate) {
  const baselineLatency = Number(guard(baseline, 'latency_ms', 0))
  const candidateLatency = Number(guard(candidate, 'latency_ms', 0))
  if (baselineLatency > 0 && candidateLatency > 0) return baselineLatency / candidateLatency
  const reported = Number(guard(candidate, 'speedup', 0))
  return reported > 0 ? reported : 0
}

log('Phase 1/6: Freeze Contract')
const setup = await withTurnTimeout(agentRetry(
  () => agent(`
You are freezing a Harness Engineering run contract.

Harness root: ${HARNESS_ROOT}
Baseline kernel: ${KERNEL_PATH}
Problem path: ${PROBLEM_PATH || '(none)'}
Backend: ${BACKEND}
Verification profile: ${VERIFICATION_PROFILE}
Experiment directory: ${EXP_DIR}

Inspect the harness and baseline without modifying either. Create ${EXP_DIR}, copy the baseline into ${EXP_DIR}/baseline/candidate, and write ${EXP_DIR}/contract.json containing the exact paths, commands, placeholders, verification profile, backend, and promotion rule. Reject ambiguous or missing paths. All later writes must stay under ${EXP_DIR}.
${__experienceBlock()}${__attemptBlock()}
Return the materialized baseline_candidate_path, contract_path, and any harness notes.
`, { label: 'harness-freeze-contract', phase: 'Freeze Contract', model: MODEL, schema: ANY_OBJECT }),
  { retries: 5, label: 'harness-freeze-contract' },
), 'Harness Engineering freeze contract')

const baselineCandidate = guard(setup, 'baseline_candidate_path', '')
if (!baselineCandidate) return { ok: false, error: 'contract_materialization_failed', setup }

log('Phase 2/6: Baseline')
const baseline = await withTurnTimeout(agentRetry(
  () => agent(`
Establish authoritative baseline evidence for a frozen Harness Engineering contract.

${commandContract(baselineCandidate, `${EXP_DIR}/baseline`)}
Verification profile: ${VERIFICATION_PROFILE}. Its semantics are documented in _substrate/verification/README.md; the supplied command remains the sole verdict owner.

Write canonical evidence to ${EXP_DIR}/baseline/evidence.json and retain raw stdout/stderr plus generated artifacts under ${EXP_DIR}/baseline. Report false/null on failure; never infer a pass or timing value.
`, { label: 'harness-baseline', phase: 'Baseline', model: MODEL, schema: EVIDENCE_SCHEMA }),
  { retries: 5, label: 'harness-baseline' },
), 'Harness Engineering baseline')

if (!evidenceIsPromotable(baseline)) {
  return { ok: false, error: 'invalid_baseline', best_kernel_path: baselineCandidate, baseline_evidence: baseline, verification_profile: VERIFICATION_PROFILE }
}

let bestCandidate = baselineCandidate
let bestEvidence = baseline
let bestSpeedup = 1
const history = []

for (let round = 1; round <= ITERATIONS; round++) {
  const roundDir = `${EXP_DIR}/round-${round}`
  log(`Phase 3/6: Profile and Decide — round ${round}/${ITERATIONS}`)
  const decision = await withTurnTimeout(agentRetry(
    () => agent(`
Act as the profile-backed controller for Harness Engineering round ${round}.

Current incumbent: ${bestCandidate}
Current evidence: ${JSON.stringify(bestEvidence)}
Round directory: ${roundDir}
${commandContract(bestCandidate, roundDir)}

If profile_command is supplied, execute it on the incumbent and retain raw artifacts under ${roundDir}/profile. Otherwise reason only from measured harness evidence and source inspection, explicitly marking missing profile evidence. Select exactly one bounded optimization hypothesis. Do not modify the incumbent or harness. Do not re-propose these failed strategy ids: ${FAILED_STRATEGY_IDS.join(', ') || '(none)'}.
${__experienceBlock()}${__attemptBlock()}
Return strategy_id, hypothesis, expected_bottleneck, supporting_artifacts, and implementation_constraints.
`, { label: `harness-decide-${round}`, phase: 'Profile and Decide', model: MODEL, schema: ANY_OBJECT }),
    { retries: 5, label: `harness-decide-${round}` },
  ), `Harness Engineering decision ${round}`)

  log(`Phase 4/6: Implement — round ${round}/${ITERATIONS}`)
  const implementation = await withTurnTimeout(agentRetry(
    () => agent(`
Implement exactly one Harness Engineering candidate.

Incumbent: ${bestCandidate}
Decision: ${JSON.stringify(decision)}
Destination: ${roundDir}/candidate
Backend: ${BACKEND}

Copy the incumbent and make only the selected bounded change under the destination. Never modify ${KERNEL_PATH}, ${HARNESS_ROOT}, or any path outside ${EXP_DIR}. Do not run the benchmark or claim correctness. Preserve the harness entrypoint and output contract.

Return candidate_path, strategy_id, change_summary, and modified_files.
`, { label: `harness-implement-${round}`, phase: 'Implement', model: MODEL, schema: ANY_OBJECT }),
    { retries: 5, label: `harness-implement-${round}` },
  ), `Harness Engineering implementation ${round}`)

  const candidatePath = guard(implementation, 'candidate_path', '')
  if (!candidatePath || !candidatePath.startsWith(EXP_DIR + '/')) {
    history.push({ round, accepted: false, reason: 'candidate_outside_exp_dir', decision, implementation })
    continue
  }

  log(`Phase 5/6: Official Gate — round ${round}/${ITERATIONS}`)
  const evidence = await withTurnTimeout(agentRetry(
    () => agent(`
Evaluate one Harness Engineering candidate with the frozen contract.

${commandContract(candidatePath, roundDir)}
Verification profile: ${VERIFICATION_PROFILE}

Write ${roundDir}/evidence.json, retain raw command outputs, and bind the evidence to the exact candidate path and artifact files. Never benchmark a failed candidate. Never infer fields from prose. Return false/null for absent evidence.
`, { label: `harness-gate-${round}`, phase: 'Official Gate', model: MODEL, schema: EVIDENCE_SCHEMA }),
    { retries: 5, label: `harness-gate-${round}` },
  ), `Harness Engineering official gate ${round}`)

  const candidateSpeedup = measuredSpeedup(baseline, evidence)
  const incumbentSpeedup = measuredSpeedup(baseline, bestEvidence)
  const accepted = evidenceIsPromotable(evidence) && candidateSpeedup > incumbentSpeedup
  history.push({ round, strategy_id: guard(implementation, 'strategy_id', guard(decision, 'strategy_id', '')), accepted, speedup: candidateSpeedup, candidate_path: candidatePath, evidence_path: guard(evidence, 'evidence_path', ''), reason: accepted ? 'correct_and_faster' : 'failed_gate_or_not_faster' })
  if (accepted) {
    bestCandidate = candidatePath
    bestEvidence = evidence
    bestSpeedup = candidateSpeedup
  }
  if (bestSpeedup >= MIN_SPEEDUP) break
}

log('Phase 6/6: Audit and Report')
const report = await withTurnTimeout(agentRetry(
  () => agent(`
Audit and report this Harness Engineering run.

Contract: ${guard(setup, 'contract_path', `${EXP_DIR}/contract.json`)}
Baseline: ${JSON.stringify(baseline)}
Best candidate: ${bestCandidate}
Best evidence: ${JSON.stringify(bestEvidence)}
Computed speedup: ${bestSpeedup}
History: ${JSON.stringify(history)}

Verify that every timing result is paired with compiled=true, correct=true, and any required verified=true verdict; that candidate/evidence paths stay under ${EXP_DIR}; and that rejected candidates were not silently promoted. Write ${EXP_DIR}/report.md and append one final run record to ${EXP_DIR}/genome.jsonl. State the fidelity boundary: workflow adaptation, not reproduction of contest infrastructure or hidden tests.

Return report_path, audit_ok, and audit_notes.
`, { label: 'harness-audit-report', phase: 'Audit and Report', model: MODEL, schema: ANY_OBJECT }),
  { retries: 5, label: 'harness-audit-report' },
), 'Harness Engineering audit')

return {
  ok: evidenceIsPromotable(bestEvidence) && bestSpeedup >= MIN_SPEEDUP && guard(report, 'audit_ok', false),
  workflow: WORKFLOW_NAME,
  best_kernel_path: bestCandidate,
  best_evidence: bestEvidence,
  best_speedup: bestSpeedup,
  baseline_evidence: baseline,
  verification_profile: VERIFICATION_PROFILE,
  history,
  report_path: guard(report, 'report_path', `${EXP_DIR}/report.md`),
  fidelity_boundary: 'workflow_adaptation',
}
