// typed-args.js — CANONICAL default-scaffolding snippet for the KerSor dispatch
// channels ② and ③: experience_excerpts (cross-session priors) and
// attempt_evidence + attempt_plan (typed prior-attempt context).
//
// This is NOT a runnable workflow. It is the single source of truth for the
// typed-args block that workflows inline right after arg_guard, so every solver
// can consume the KerSor dispatch channels as TYPED data (not HANDOFF prose):
//   - channel ② (EXPERIENCE_EXCERPTS): lower-authority priors from past sessions.
//   - channel ③ (ATTEMPT_EVIDENCE/PLAN): higher-authority, machine-verified prior
//     attempt context; FAILED_STRATEGY_IDS is a HARD constraint (do not re-propose).
// The block declares the consts + __experienceBlock()/__attemptBlock() helpers
// that surface the channels in agent prompts. Byte-identical across the 5
// workflows that have it (AccelOpt, Generalist, KDA, KernelFoundry, STARK); the
// patch-typed-args.js codemod wraps those + propagates the block to the rest.
//
// CONSTRAINT: the Workflow runtime forbids Date.now()/Math.random()/new Date().
// This block uses none. The consts degrade safely to null/[] when the channel is
// absent (cold-start, or KerSor didn'''t emit it) — declaring the block is
// harmless even if a workflow'''s prompts don'''t yet call the helpers.
//
// USAGE (inline right after the arg_guard block):
//   // --- BEGIN inlined typed-args (from _meta/scaffolding/typed-args.js) ---
//   <paste the block>
//   // --- END inlined typed-args ---

// --- BEGIN typed-args (channel ② experience_excerpts) ---
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
// --- END typed-args ---
