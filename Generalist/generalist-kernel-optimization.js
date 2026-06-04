export const meta = {
  name: 'generalist-kernel-optimization',
  description: 'Best-of-breed single solver built on the KerSor Solver SDK substrate: deterministic diagnosis, persistent state-keyed memory, method gating, anti-cheat reward, and a beam controller. Directly callable; also the strongest default member / hardest baseline for KerSor.',
  whenToUse: 'When you want one directly-callable kernel optimizer that internalizes the universal best-of-breed components (the 7 composable axes) with a single beam search topology. For cross-topology coverage, orchestrate with KerSor instead.',
  phases: [
    { title: 'Setup', detail: 'Read kernel + baseline, init beam, locate substrate scripts' },
    { title: 'Profile', detail: 'Profile current best (ncu/benchmark) into normalized metrics' },
    { title: 'Diagnose', detail: 'diagnose.py -> shared bottleneck_class (Layer C)' },
    { title: 'Retrieve', detail: 'memory_store.py retrieve + method_gate.py allowed_methods (Layers D, E)' },
    { title: 'Plan', detail: 'Generate BREADTH plans, gated to allowed_methods, with grounded anchors' },
    { title: 'Evaluate', detail: 'Implement + eval each plan; anti_cheat.py reward; evidence_schema.py (Layers A, B)' },
    { title: 'Learn', detail: 'memory_store.py update with measured outcome; record dead-ends; beam top-K' },
    { title: 'Report', detail: 'Final report + Layer A evidence envelope' },
  ],
}

// =============================================================================
// Generalist Kernel Optimization — KerSor Solver SDK reference solver
// =============================================================================
//
// Topology: beam (the single chosen controller; topologies are exclusive — see
// _substrate/SOLVER-AUDIT.md). All best-of-breed COMPONENTS are delegated to the
// deterministic substrate scripts under _substrate/, run by agent Bash steps —
// the scripts are the fidelity anchor, not the prompts.
//
//   Layer A  evidence_schema.py   canonical attempt evidence (KerSor-native)
//   Layer B  anti_cheat.py        validity gate + robust reward (beat eager AND compile)
//   Layer C  diagnose.py          metrics -> shared bottleneck_class
//   Layer D  memory_store.py      persistent state-keyed memory (compounds across runs)
//   Layer E  method_gate.py       bottleneck_class -> allowed_methods
//   Layer F  (inline JS)          ceiling / stagnation / early-termination gates
//
// Usage:
//   Workflow({ name: 'generalist-kernel-optimization', args: {
//     kernel_path: '/path/to/kernel.cu',
//     op_description: 'Quantized GEMM Q4_0 weight x FP32 activation',
//     eval_command: 'python eval.py --kernel KERNEL_PATH --out RESULT_JSON',
//       // eval_command MUST write JSON: {compiled, correct, candidate_latency_ms,
//       //   eager_latency_ms, compile_latency_ms, speedup, metrics:{dram_pct,sm_pct,occupancy,latency_ms}}
//     ncu_command: 'ncu --set full ...',           // optional; else metrics come from eval_command
//     substrate_dir: '/path/to/Awesome-Kernel-Workflows/_substrate',
//     exp_dir: '/path/to/experiment/output',
//     memory_db: '/path/to/experiment/memory.json', // persistent; defaults to exp_dir/memory.json
//     iterations: 3, breadth: 3, topk: 3, target_speedup: 1.5,
//   }})
//
// =============================================================================

// ---- Args / contract ----
const KERNEL_PATH = args.kernel_path
const OP = args.op_description || 'kernel optimization'
const EVAL_CMD = args.eval_command
const NCU_CMD = args.ncu_command || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const EXP_DIR = args.exp_dir || '.'
const MEMORY_DB = args.memory_db || `${EXP_DIR}/memory.json`
const ITERATIONS = args.iterations || 3
const BREADTH = args.breadth || 3
const TOPK = args.topk || 3
const TARGET = args.target_speedup || 1.5
const STAGNATION_EPS = 0.02   // < 2% improvement counts as no progress (Xe-Forge/KSearch)
const STAGNATION_LIMIT = 2    // consecutive stagnant rounds -> stop

// P0 — model/intelligence routing: mechanical steps don't need Opus (token savings).
const MODEL = {
  mechanical: args.model_mechanical || 'haiku',  // run substrate scripts, parse JSON
  profile: args.model_profile || 'sonnet',       // run eval/ncu, normalize metrics
  judgment: args.model_judgment || 'opus',       // plan / implement / report (override per kernel complexity)
}
// P0 — token-budget wiring: rough per-unit output-token estimates for scaling + stopping.
const EST_PER_CANDIDATE = args.est_tokens_per_candidate || 20000
const EST_PER_ROUND = EST_PER_CANDIDATE * BREADTH + 15000

// Schemas for structured agent returns
const METRICS_SCHEMA = {
  type: 'object',
  properties: {
    compiled: { type: 'boolean' }, correct: { type: 'boolean' },
    candidate_latency_ms: { type: ['number', 'null'] },
    eager_latency_ms: { type: ['number', 'null'] },
    compile_latency_ms: { type: ['number', 'null'] },
    speedup: { type: ['number', 'null'] },
    metrics: { type: 'object' },
  },
  required: ['compiled', 'correct', 'speedup', 'metrics'],
}
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }
const ANTICHEAT_SCHEMA = {
  type: 'object',
  properties: {
    valid: { type: 'boolean' }, reward: { type: 'number' },
    recorded_speedup: { type: 'number' }, reward_reason: { type: 'string' },
    flags: { type: 'array' }, blocking_flags: { type: 'array' },
  },
  required: ['valid', 'reward', 'recorded_speedup'],
}

phase('Setup')
log(`Generalist solver | beam | breadth=${BREADTH} topk=${TOPK} iters=${ITERATIONS} target=${TARGET}x | models ${MODEL.mechanical}/${MODEL.profile}/${MODEL.judgment} | budget ${(typeof budget !== 'undefined' && budget.total) ? Math.round(budget.total / 1000) + 'k' : 'unbounded'}`)

// Baseline candidate seeds the beam.
let candidateBeam = [{ id: 'baseline', parent_id: null, code_path: KERNEL_PATH, speedup: 1.0, metrics: {}, planTitle: 'baseline' }]
let bestSpeedup = 1.0
let stagnantRounds = 0
let dryRounds = 0                 // P1.5 — consecutive rounds with no new 'measured' insight
const DRY_LIMIT = 2               // P1.5 — loop-until-dry stop threshold
const allAttempts = []
const verifiedInsights = []       // P1.4 — verified typed insights for the Layer A envelope

for (let iter = 1; iter <= ITERATIONS; iter++) {
  const best = candidateBeam[0]
  log(`\n=== Iteration ${iter}/${ITERATIONS} | best ${bestSpeedup.toFixed(3)}x | beam ${candidateBeam.length} ===`)

  // P0 — token budget: stop if a full round can't fit; scale breadth to remaining budget.
  if (typeof budget !== 'undefined' && budget.total && budget.remaining() < EST_PER_ROUND) {
    log(`token budget ~exhausted (${Math.round(budget.remaining() / 1000)}k < ${Math.round(EST_PER_ROUND / 1000)}k/round) — stop`)
    break
  }
  const effBreadth = (typeof budget !== 'undefined' && budget.total)
    ? Math.max(1, Math.min(BREADTH, Math.floor(budget.remaining() / EST_PER_CANDIDATE)))
    : BREADTH

  // ---- Profile (Layer C input) ----
  phase('Profile')
  const metrics = await agent(
    `Profile the current best kernel and produce normalized metrics.\n` +
    `Kernel: ${best.code_path}\nOp: ${OP}\n` +
    (NCU_CMD ? `Run NCU: \`${NCU_CMD}\` and the eval command \`${EVAL_CMD}\`.\n`
             : `Run the eval command \`${EVAL_CMD}\` (writes a JSON result file).\n`) +
    `Return the JSON exactly per the schema: compiled, correct, candidate_latency_ms, ` +
    `eager_latency_ms, compile_latency_ms, speedup, and metrics{dram_pct, sm_pct, occupancy, latency_ms}. ` +
    `Use null for unknown numbers. Do not fabricate; missing => null.`,
    { label: `profile-${iter}`, phase: 'Profile', schema: METRICS_SCHEMA, model: MODEL.profile })

  // ---- Diagnose (Layer C, deterministic script) ----
  phase('Diagnose')
  const diag = await agent(
    `Write these metrics to ${EXP_DIR}/run-${iter}/metrics.json:\n${JSON.stringify(metrics.metrics || {})}\n` +
    `Then run exactly: \`python3 ${SUBSTRATE}/diagnose.py --metrics ${EXP_DIR}/run-${iter}/metrics.json\` ` +
    `and return its stdout JSON verbatim ({bottleneck_class, evidence}).`,
    { label: `diagnose-${iter}`, phase: 'Diagnose', schema: JSON_PASSTHROUGH, model: MODEL.mechanical })
  const bclass = diag.bottleneck_class || 'unknown'
  log(`bottleneck_class = ${bclass}`)

  // ---- Retrieve memory (Layer D) + gate methods (Layer E) ----
  phase('Retrieve')
  const [mem, gate] = await parallel([
    () => agent(
      `Run exactly: \`python3 ${SUBSTRATE}/memory_store.py --db ${MEMORY_DB} retrieve --class ${bclass}\` ` +
      `and return its stdout JSON verbatim ({bottleneck_class, techniques, dead_ends}).`,
      { label: `retrieve-${iter}`, phase: 'Retrieve', schema: JSON_PASSTHROUGH, model: MODEL.mechanical }),
    () => agent(
      `Run exactly: \`python3 ${SUBSTRATE}/method_gate.py --class ${bclass} --metrics ${EXP_DIR}/run-${iter}/metrics.json\` ` +
      `and return its stdout JSON verbatim ({bottleneck_class, allowed_methods, rationale}).`,
      { label: `gate-${iter}`, phase: 'Retrieve', schema: JSON_PASSTHROUGH, model: MODEL.mechanical }),
  ])
  const allowed = (gate && gate.allowed_methods) || []
  const priorTech = (mem && mem.techniques) || []
  const deadEnds = (mem && mem.dead_ends) || []
  log(`allowed_methods = ${allowed.join(', ')} | prior techniques = ${priorTech.length} | dead-ends = ${deadEnds.length}`)

  // ---- Plan: BREADTH gated plans with grounded anchors (STARK borrow) ----
  phase('Plan')
  const planContext =
    `# Bottleneck: ${bclass}\n# Allowed methods (you MUST stay within these): ${allowed.join(', ')}\n` +
    `# Prior techniques by confidence: ${JSON.stringify(priorTech.slice(0, 5))}\n` +
    `# Dead-ends to AVOID (re-allowed only if revalidate_if holds): ${JSON.stringify(deadEnds)}\n` +
    `# Current best: ${best.code_path} @ ${best.speedup.toFixed(2)}x`
  const plans = (await parallel(
    Array.from({ length: effBreadth }, (_, i) => () => agent(
      `You are planner #${i + 1}/${effBreadth} for a ${OP} kernel.\n${planContext}\n\n` +
      `Propose ONE optimization plan that uses ONLY an allowed method. Mark the exact code region to change with ` +
      `grounded anchors <<<IMPROVE BEGINS>>> ... <<<IMPROVE ENDS>>>. Pick the highest-confidence prior technique ` +
      `that fits, unless a dead-end forbids it. Return {method, plan, anchors}.`,
      { label: `plan-${iter}-${i + 1}`, phase: 'Plan', schema: JSON_PASSTHROUGH, model: MODEL.judgment })))
  ).filter(Boolean)

  // ---- Evaluate each plan: implement -> eval -> anti-cheat (Layers B, A) ----
  phase('Evaluate')
  const evaluated = (await parallel(plans.map((p, i) => () => (async () => {
    const runDir = `${EXP_DIR}/run-${iter}/cand-${i + 1}`
    const m = await agent(
      `Implement this plan on a COPY of ${best.code_path} into ${runDir}/kernel, respecting the ` +
      `<<<IMPROVE BEGINS/ENDS>>> anchors. Method: ${p.method}. Plan: ${JSON.stringify(p.plan)}.\n` +
      `Then run \`${EVAL_CMD}\` and return the JSON metrics per schema.`,
      { label: `impl-${iter}-${i + 1}`, phase: 'Evaluate', schema: METRICS_SCHEMA, model: MODEL.judgment, isolation: 'worktree' })
    const ac = await agent(
      `Write these metrics to ${runDir}/metrics.json:\n${JSON.stringify({ ...m, claimed_speedup: m.speedup })}\n` +
      `Then run exactly: \`python3 ${SUBSTRATE}/anti_cheat.py --source ${runDir}/kernel --metrics ${runDir}/metrics.json\` ` +
      `and return its stdout JSON verbatim. Then run ` +
      `\`python3 ${SUBSTRATE}/evidence_schema.py validate ${runDir}/metrics.json\` to confirm it is well-formed.`,
      { label: `anticheat-${iter}-${i + 1}`, phase: 'Evaluate', schema: ANTICHEAT_SCHEMA, model: MODEL.mechanical })
    return { plan: p, metrics: m, anticheat: ac, code_path: `${runDir}/kernel`,
             recorded_speedup: ac.valid ? ac.recorded_speedup : 0 }
  })()))).filter(Boolean)

  // ---- Learn: update persistent memory (Layer D) per measured outcome ----
  phase('Learn')
  await parallel(evaluated.map((e) => () => agent(
    `Run exactly: \`python3 ${SUBSTRATE}/memory_store.py --db ${MEMORY_DB} update --class ${bclass} ` +
    `--technique ${e.plan.method} --speedup ${e.metrics.speedup || 0} --correct ${e.metrics.correct ? 1 : 0}\`` +
    (e.anticheat.valid ? '' :
      ` ; then run \`python3 ${SUBSTRATE}/memory_store.py --db ${MEMORY_DB} add-deadend ` +
      `--claim ${JSON.stringify(e.plan.method)} --why ${JSON.stringify(e.anticheat.reward_reason || (e.anticheat.blocking_flags || []).join(','))} --revalidate-if "metrics change"\``) +
    ` and return {updated:true}.`,
    { label: `learn-${iter}-${e.plan.method}`, phase: 'Learn', schema: JSON_PASSTHROUGH, model: MODEL.mechanical })))

  // ---- P1.4: adversarial insight verification (LLM refuter judgment + deterministic downgrade) ----
  const roundInsightRaw = {
    kind: 'bottleneck', directive: 'explore',
    claim: `bottleneck_class=${bclass} dominates ${OP} at this shape`,
    evidence: NCU_CMD ? 'ncu' : 'benchmark', confidence: 'measured', source_round: iter,
  }
  const refute = await agent(
    `Adversarially REFUTE this attribution against the MEASURED profile ` +
    `${JSON.stringify(metrics.metrics || {})}: "${roundInsightRaw.claim}". ` +
    `Default to refuted=true if the data does not clearly support it. Return {refuted, reason}.`,
    { label: `refute-${iter}`, phase: 'Learn', model: MODEL.judgment,
      schema: { type: 'object', properties: { refuted: { type: 'boolean' }, reason: { type: 'string' } }, required: ['refuted'] } })
  const verified = await agent(
    `Write this insight to ${EXP_DIR}/run-${iter}/insight.json:\n${JSON.stringify(roundInsightRaw)}\n` +
    `Then run exactly: \`python3 ${SUBSTRATE}/verify_insight.py --insight ${EXP_DIR}/run-${iter}/insight.json --refuted ${refute.refuted ? 1 : 0}\` ` +
    `and return its stdout JSON verbatim.`,
    { label: `verify-insight-${iter}`, phase: 'Learn', schema: JSON_PASSTHROUGH, model: MODEL.mechanical })
  verifiedInsights.push(verified)
  const producedMeasured = (verified && verified.confidence) === 'measured'
  log(`round insight confidence: ${(verified && verified.confidence) || 'n/a'}${refute.refuted ? ' (refuted -> downgraded)' : ''}`)

  // ---- Beam update (deterministic JS): merge, keep top-K by recorded speedup ----
  const newCandidates = evaluated
    .filter((e) => e.anticheat.valid && e.recorded_speedup > 0)
    .map((e, i) => ({ id: `it${iter}-c${i + 1}`, parent_id: best.id, code_path: e.code_path,
                      speedup: e.recorded_speedup, metrics: e.metrics.metrics || {}, planTitle: e.plan.method }))
  evaluated.forEach((e) => allAttempts.push({ iter, method: e.plan.method, valid: e.anticheat.valid,
                                              reward: e.anticheat.reward, speedup: e.recorded_speedup }))
  const merged = [...candidateBeam, ...newCandidates].sort((a, b) => b.speedup - a.speedup)
  candidateBeam = merged.slice(0, TOPK)

  // ---- Cost-control gates (Layer F, inline) ----
  const newBest = candidateBeam[0].speedup
  const improvement = (newBest - bestSpeedup) / Math.max(bestSpeedup, 1e-9)
  stagnantRounds = improvement < STAGNATION_EPS ? stagnantRounds + 1 : 0
  dryRounds = producedMeasured ? 0 : dryRounds + 1   // P1.5 loop-until-dry
  bestSpeedup = newBest
  log(`iter ${iter}: best now ${bestSpeedup.toFixed(3)}x (Δ ${(improvement * 100).toFixed(1)}%) | stagnant ${stagnantRounds} | dry ${dryRounds}`)
  if (bestSpeedup >= TARGET) { log(`target ${TARGET}x reached — stop (COMPLETE)`); break }
  if (stagnantRounds >= STAGNATION_LIMIT) { log(`stagnation limit reached — stop (STALLED)`); break }
  if (dryRounds >= DRY_LIMIT) { log(`loop-until-dry: ${DRY_LIMIT} rounds with no new measured insight — stop (STALLED)`); break }
}

// ---- Report + Layer A evidence envelope ----
phase('Report')
const status = bestSpeedup >= TARGET ? 'converged' : ((stagnantRounds >= STAGNATION_LIMIT || dryRounds >= DRY_LIMIT) ? 'stalled' : 'budget_exhausted')
await agent(
  `Write a final optimization report for ${OP}.\n` +
  `Best speedup: ${bestSpeedup.toFixed(3)}x | status: ${status}\n` +
  `Beam (top-${TOPK}): ${JSON.stringify(candidateBeam.map((c) => ({ id: c.id, method: c.planTitle, speedup: c.speedup })))}\n` +
  `All attempts: ${JSON.stringify(allAttempts)}\n` +
  `Cover: which methods the gate allowed, what the persistent memory recommended, ` +
  `which attempts were rejected by anti-cheat and why, and the final best kernel path.`,
  { label: 'final-report', phase: 'Report', model: MODEL.judgment })

return {
  solver: 'generalist-kernel-optimization',
  topology: 'beam',
  overall_speedup: bestSpeedup,
  best_kernel_code: candidateBeam[0].code_path,
  convergence_status: status,
  beam: candidateBeam,
  attempts: allAttempts,
  insights: verifiedInsights,
  memory_db: MEMORY_DB,
}
