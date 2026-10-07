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
// Task requirements must reach every fresh activation independently of a
// previous agent's summary. The runtime owns file access and skill admission.
function __taskContractBlock() {
  const taskPath = args.problem_path || args.kernel_spec_path
  const inlineTask = args.problem_definition
  if (!taskPath && !inlineTask) return ''
  return '\n# Authoritative task contract\n'
    + (taskPath ? `Read the complete original task at ${taskPath} before acting, including its required skills and profiling instructions. Do not rely only on a prior agent summary.\n` : '')
    + (inlineTask ? `Complete caller-supplied problem definition:\n${typeof inlineTask === 'string' ? inlineTask : JSON.stringify(inlineTask, null, 2)}\n` : '')
    + (args.task_result_command ? `Task measurement ownership: use the declared task_result_command for all candidate tests: ${args.task_result_command}\nOnly that command creates the result file and its .artifact directory. Never pre-create, rename, delete, or write a result slot. Never run selftest.py or verify.py directly as a substitute for the declared command. Producers write candidate source files only; the task command freezes and evaluates them. Return the resulting complete contract object, including rejected candidates. Existing unknown slots require original-evidence reconciliation, not another GPU submission.\n` : '')
    + 'Retain all task constraints. Missing tools or unavailable task files must be reported explicitly; do not silently replace a required profiler.\n\n'
}
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
    parts.push('## Prior attempt evidence (last round):\n```json\n' + j + '\n```')
  }
  if (ATTEMPT_PLAN && Array.isArray(ATTEMPT_PLAN.candidate_plans)) {
    parts.push('## Routing-suggested candidate plans:\n```json\n' + JSON.stringify({phase_intent: ATTEMPT_PLAN.phase_intent, candidate_plans: ATTEMPT_PLAN.candidate_plans}, null, 2) + '\n```')
  }
  return parts.join('\n') + '\n'
}

// File-backed candidate context. Pure builders; execute persistence inside the
// existing producer/evaluator activation, not an additional model call.
function __workspaceSource(path, fallback) {
  return path ? `Read COMPLETE source from ${path}. Do not reconstruct it from a summary. Report a missing file explicitly.` : String(fallback ?? '')
}
function __workspaceResult(path, fallback) {
  if (!path) return fallback
  const measurement = {}
  for (const key of ['compiled','correct','is_valid','measurement_valid','speedup','runtime_ms','latency_ms','kernel_time_ms','n_pass','n_total','status','measured','is_correct','is_compilable','error_type','failure_origin']) {
    const value = fallback && fallback[key]
    if (value === null || ['number','boolean','string'].includes(typeof value)) measurement[key] = value
  }
  return {result_path:path,measurement,read_instruction:'Read the COMPLETE result and referenced raw traces before diagnosing. Retain the runtime measurement scalars in this index; do not replace them with an author estimate.'}
}
function __workspaceStore(path, source, root) {
  const quote = value => "'" + String(value).replaceAll("'", "'\\''") + "'"
  const program = [
    'import json,sys',
    'from pathlib import Path',
    'root=Path(sys.argv[1]).resolve()',
    'raw=Path(sys.argv[2])',
    'assert not raw.is_symlink(), "candidate path is a symlink"',
    'p=raw.resolve()',
    'assert p.is_relative_to(root) and p!=root, "candidate outside workspace"',
    'data=json.load(sys.stdin).encode("utf-8")',
    'p.parent.mkdir(parents=True,exist_ok=True)',
    'try:',
    ' with p.open("xb") as f: f.write(data)',
    'except FileExistsError:',
    ' assert p.read_bytes()==data, "candidate identity collision; retain original"',
    'assert p.read_bytes()==data, "candidate persistence incomplete"',
    'print(json.dumps({"source_path":str(p),"source_saved":True}))',
  ].join('\n')
  return '# Before testing, execute this CPU-only source persistence command once. It preserves the complete source and refuses a different existing candidate; it does not allocate a GPU. Do not recreate or overwrite a mismatched file.\n'
    + ['python3','-c',program,root,path].map(quote).join(' ')
    + " <<'KERSOR_WORKSPACE_SOURCE_JSON'\n" + JSON.stringify(String(source ?? '')) + '\nKERSOR_WORKSPACE_SOURCE_JSON'
}

// --- END typed-args ---
