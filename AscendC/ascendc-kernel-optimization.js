// ascendc-kernel-optimization.js — canonical Ascend/AscendC kernel-optimization workflow.
//
// Source contract: the proven session-local variant `generalist-ascend-optimization`
// evolved across 910b-exp sessions 20260622-160108 / 20260623-{194914,190425,191018}
// (see AKW-Exp/910b-exp/KEY-ISSUES.md §B and GitHub issue #16). Promoted here as a
// first-class, Ascend-native catalog entry so Ascend tasks no longer STALL at
// workflow selection (round 1) and do not need a re-evolved session-local variant.
//
// What makes it Ascend-native (not a CUDA workflow with swapped labels):
//   - language = ascendc; backend = ascend; profiler = msprof (CANN), NOT ncu.
//   - compile + correctness + timing are delegated to the MultiKernelBench
//     `ascendc_direct_launch` runner via `_substrate/backends/ascend/{build,run,profile}.sh`
//     (bisheng+cmake compile coupled with correctness + NPU-event timing inside
//     eval_single_runner.py). requires_ncu = false.
//   - candidates are emitted in the `ascendc_direct_launch` JSON submission format
//     (sources[] + build{} + entry{}), written ONE FILE AT A TIME via Bash
//     (issue #17 / KEY-ISSUES §3 row 9: a single multi-file JSON blob truncates).
//
// Robustness scaffolding (issue #17): every `agent()` call is wrapped in
// `agentRetry` and every dereferenced result is null-guarded, so a transient API
// 429 / agent skip no longer crashes the whole run.

export const meta = {
  name: 'ascendc-kernel-optimization',
  description: 'Canonical Ascend/AscendC kernel-optimization workflow for Ascend 910B NPUs: iterative plan→generate→evaluate→optimize→report using AscendC (ascendc_direct_launch) and msprof profiling via the substrate ascend backend.',
  whenToUse: 'When optimizing or generating a kernel on Ascend 910B2 (or compatible Ascend NPU) in AscendC. Requires the ascend substrate backend (_substrate/backends/ascend) plus bisheng, torch_npu, and the MultiKernelBench eval harness (MULTIKERNELBENCH_ROOT). Do NOT use for CUDA/Triton/ROCm targets.',
  phases: [
    { title: 'Setup', detail: 'Read reference op + substrate ascend backend manifest; assert ascendc suitability' },
    { title: 'Generate', detail: 'Generate initial AscendC kernel candidates in ascendc_direct_launch JSON (per-file Bash write)' },
    { title: 'Evaluate', detail: 'Build + correctness + timing via substrate ascend run.sh (eval_single_runner.py)' },
    { title: 'Optimize', detail: 'Iterative msprof → plan → implement (one turn = write+eval+return) → test' },
    { title: 'Report', detail: 'Final results + best kernel' },
  ],
}

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

const WORKFLOW_NAME = 'ascendc-kernel-optimization'


// --- args ---
const KERNEL_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const OP = args.op_description || 'AscendC kernel optimization'
const EXP_DIR = args.exp_dir || '.'
const ITERATIONS = args.iterations || 3
const BREADTH = args.breadth || 2
const TARGET = args.target_speedup || 1.5
const TARGET_GPU = args.target_gpu || 'Ascend910B2'
const NOTE = args.note || ''
const SUBSTRATE_DIR = args.substrate_dir || '_substrate'
const BACKEND_DIR = args.backend_dir || `${SUBSTRATE_DIR}/backends/ascend`
const MKB_ROOT = args.mkb_root || (typeof process !== 'undefined' && process.env && process.env.MULTIKERNELBENCH_ROOT) || ''
const OP_ID = args.op || ''
const EVAL_CMD_TEMPLATE = args.benchmark_command || ''
const PROFILE_CMD_TEMPLATE = args.profile_command || ''
const SEED_CANDIDATES = args.seed_candidates || BREADTH

const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const USE_DRIVER = !!BACKEND_DIR
const STAGNATION_EPS = 0.02
const STAGNATION_LIMIT = 2

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    compiled: { type: 'boolean' },
    correct: { type: 'boolean' },
    speedup: { type: ['number', 'null'] },
    candidate_latency_ms: { type: ['number', 'null'] },
    eager_latency_ms: { type: ['number', 'null'] },
    kernel_json_path: { type: ['string', 'null'] },
    error: { type: ['string', 'null'] },
  },
  required: ['compiled', 'correct', 'speedup'],
}

// Shared directive block appended to every generation / optimization prompt.
// (1) Turn boundary (issue #17 / KEY-ISSUES §3 row 2): one turn = write files,
//     run eval ONCE, read result, RETURN. (2) Per-file Bash write (row 9):
//     never emit several AscendC files as one JSON blob — write each file via
//     Bash. (3) No harness manipulation (row 3): no allocator/free-pool tricks.
const ASCENDC_DIRECTIVES = `
# HARD RULES (issue #17 default scaffolding)
- TURN BOUNDARY: this turn must (a) write the submission files, (b) run the eval
  build/run ONCE, (c) read the result JSON, (d) RETURN immediately. Do NOT loop
  compile+eval inside one turn. Cap yourself to a small number of Bash calls.
- MULTI-FILE WRITE: write each AscendC source file SEPARATELY via Bash
  (e.g. \`printf '%s' "<src>" > path/batched_matmul_kernel.cpp\`), then assemble
  the ascendc_direct_launch submission JSON that references those files. Do NOT
  try to return several full source files inside one structured-output JSON blob
  — that path truncates/times out.
- NO HARNESS MANIPULATION: improve the kernel itself. Never manipulate the test
  harness, the reference's allocator, or the NPU free pool (e.g. allocating/
  freeing zeroed tensors of specific sizes to seed the allocator so the reference
  recycles zeroed memory and spuriously matches). That is a harness-only trick.
- Do NOT fabricate metrics. If build/eval fails, report compiled=false or
  correct=false with the real error.`

// Build the eval command for a given submission JSON path.
// Prefer the substrate ascend backend (ascendc_direct_launch runner); fall back to
// a user-supplied benchmark_command template; else mark measured evidence missing.
function evalCommandFor(jsonPath, resultPath) {
  if (USE_DRIVER) {
    return `bash ${BACKEND_DIR}/build.sh --source ${jsonPath} --out ${jsonPath}.build.json && ` +
      `bash ${BACKEND_DIR}/run.sh --artifact ${jsonPath} ${OP_ID ? `--op ${OP_ID}` : ''} ${MKB_ROOT ? `--mkb-root ${MKB_ROOT}` : ''} --out ${resultPath}`
  }
  if (EVAL_CMD_TEMPLATE) {
    return EVAL_CMD_TEMPLATE
      .replace(/\{kernel_path\}|\$\{kernel_path\}|KERNEL_PATH/g, jsonPath)
      .replace(/\{result_path\}|\$\{result_path\}|RESULT/g, resultPath)
  }
  return ''
}

function profileCommandFor(jsonPath, outPath) {
  if (USE_DRIVER) {
    return `bash ${BACKEND_DIR}/profile.sh --artifact ${jsonPath} ${OP_ID ? `--op ${OP_ID}` : ''} ${MKB_ROOT ? `--mkb-root ${MKB_ROOT}` : ''} --out ${outPath}`
  }
  if (PROFILE_CMD_TEMPLATE) {
    return PROFILE_CMD_TEMPLATE
      .replace(/\{kernel_path\}|\$\{kernel_path\}|KERNEL_PATH/g, jsonPath)
      .replace(/\{out\}|\$\{out\}|RESULT/g, outPath)
  }
  return ''
}

phase('Setup')
log(`ascendc-kernel-optimization | mode=${INPUT_MODE} | breadth=${BREADTH} iters=${ITERATIONS} target=${TARGET}x | ${TARGET_GPU}`)
if (!USE_DRIVER && !EVAL_CMD_TEMPLATE) {
  log('WARNING: no substrate ascend backend_dir and no benchmark_command — measured evidence will be missing; workflow will do static analysis only.')
}
if (USE_DRIVER && !MKB_ROOT && !EVAL_CMD_TEMPLATE) {
  log('WARNING: ascend backend selected but MULTIKERNELBENCH_ROOT / mkb_root is unset — run.sh may be unable to drive the op.')
}

// Setup: read the reference (kernel_path or problem definition) + substrate manifest.
const setupResult = await agentRetry(
  () => agent(
    `Read the reference for this AscendC task and return its operation signature.
${KERNEL_PATH ? `Reference kernel: ${KERNEL_PATH}\nRead it and summarize what it computes.` : 'No kernel_path — use the problem definition below.'}
Problem definition: ${PROBLEM_DEFINITION || `(see problem_path: ${PROBLEM_PATH || '(none)'})`}
op_description: ${OP}
${NOTE ? `User guidance: ${NOTE}` : ''}
${USE_DRIVER ? `Also read the substrate ascend backend manifest at ${BACKEND_DIR}/manifest.json and idioms at ${BACKEND_DIR}/idioms.json; note the profiler is msprof and the runner is ascendc_direct_launch.` : ''}

Return {op_name, input_shapes, output_shapes, dtype, compute_description, flops_estimate, key_constraints}.`,
    {
      label: 'analyze-reference', phase: 'Setup',
      schema: {
        type: 'object',
        properties: {
          op_name: { type: 'string' },
          input_shapes: { type: 'string' },
          output_shapes: { type: 'string' },
          dtype: { type: 'string' },
          compute_description: { type: 'string' },
          flops_estimate: { type: 'string' },
          key_constraints: { type: 'string' },
        },
        required: ['op_name', 'compute_description'],
      },
    },
  ),
  { retries: 5, label: 'analyze-reference' },
)
const opInfo = {
  op_name: expect(setupResult, 'op_name', 'analyze-reference'),
  compute_description: expect(setupResult, 'compute_description', 'analyze-reference'),
  input_shapes: guard(setupResult, 'input_shapes', ''),
  output_shapes: guard(setupResult, 'output_shapes', ''),
  dtype: guard(setupResult, 'dtype', 'float32'),
  flops_estimate: guard(setupResult, 'flops_estimate', ''),
  key_constraints: guard(setupResult, 'key_constraints', ''),
}
log(`Op: ${opInfo.op_name} | ${opInfo.compute_description}`)

const SUBMISSION_FORMAT = `## ascendc_direct_launch Submission Format
The submission MUST be a JSON object with this structure (write each source file
SEPARATELY via Bash first, then assemble this JSON referencing them):
\`\`\`json
{
  "sources": [
    {"path": "kernel.cpp",      "content": "<AscendC kernel C++ source>"},
    {"path": "pybind11.cpp",    "content": "<pybind11 binding source>"},
    {"path": "ModelNew.py",     "content": "<Python model class>"}
  ],
  "build": {
    "kernel_sources": ["kernel.cpp"],
    "binding_sources": ["pybind11.cpp"]
  },
  "entry": { "model": "ModelNew.py::ModelNew" }
}
\`\`\`
- kernel.cpp is compiled by bisheng (AscendC compiler) targeting the Ascend arch.
- pybind11.cpp is compiled by g++ with pybind11 + torch headers.
- ModelNew.py is the entry; it must expose the same interface as the reference.
- Use torch_npu for NPU operations in Python.`

phase('Generate')
log(`Generating ${SEED_CANDIDATES} initial AscendC candidate(s)...`)

const genResults = (await parallel(
  Array.from({ length: SEED_CANDIDATES }, (_, i) => () => agentRetry(
    () => agent(
      `You are generating AscendC kernel candidate ${i + 1}/${SEED_CANDIDATES} for:
${opInfo.compute_description}
Op: ${opInfo.op_name}, shapes: ${opInfo.input_shapes || 'see reference'}, dtype: ${opInfo.dtype}
Target: ${TARGET_GPU}${opInfo.key_constraints ? `\nConstraints: ${opInfo.key_constraints}` : ''}

${PROBLEM_DEFINITION || OP}
${NOTE ? `\nUser guidance: ${NOTE}` : ''}

${SUBMISSION_FORMAT}

## Candidate ${i + 1} guidance
Write a correct AscendC implementation. Prefer the Cube/MTE pipeline idioms from
${BACKEND_DIR}/idioms.json when they fit (TQue double-buffered DMA/Cube pipelining,
correct tiling across AI Cores, NZ (FRACTAL_NZ) layout for Cube utilization). Do
NOT fabricate performance — correctness first.

Write the submission JSON to ${EXP_DIR}/candidates/gen_${i + 1}.json
Then run the eval (ONCE) and read the result:
  ${evalCommandFor(`${EXP_DIR}/candidates/gen_${i + 1}.json`, `${EXP_DIR}/candidates/gen_${i + 1}_result.json`) || '(no eval command provided — report compiled=false)'}

Return {compiled, correct, speedup, candidate_latency_ms, eager_latency_ms, kernel_json_path, error}.
${ASCENDC_DIRECTIVES}`,
      { label: `generate-${i + 1}`, phase: 'Generate', schema: RESULT_SCHEMA },
    ),
    { retries: 5, label: `generate-${i + 1}` },
  )),
)).filter(Boolean)

function toCandidate(r, id, fallbackPath) {
  return {
    id,
    path: guard(r, 'kernel_json_path', fallbackPath),
    speedup: (r.compiled && r.correct && r.speedup) ? r.speedup : 0,
    latency: guard(r, 'candidate_latency_ms', null),
    compiled: !!r.compiled,
    correct: !!r.correct,
  }
}

let candidates = genResults.map((r, i) => toCandidate(r, `gen-${i + 1}`, `${EXP_DIR}/candidates/gen_${i + 1}.json`))
  .sort((a, b) => b.speedup - a.speedup)
let bestSpeedup = candidates.length > 0 ? Math.max(candidates[0].speedup, 0) : 0
let bestPath = candidates.length > 0 && candidates[0].speedup > 0 ? candidates[0].path : ''
let stagnantRounds = 0
log(`Generation done: ${candidates.filter(c => c.compiled && c.correct).length}/${SEED_CANDIDATES} correct | best ${bestSpeedup.toFixed(3)}x`)

phase('Optimize')
for (let iter = 1; iter <= ITERATIONS; iter++) {
  if (bestSpeedup >= TARGET) { log(`Target ${TARGET}x reached — stop`); break }
  if (stagnantRounds >= STAGNATION_LIMIT) { log(`Stagnation limit — stop`); break }
  if (typeof budget !== 'undefined' && budget.total && budget.remaining() < 50000) {
    log(`Budget nearly exhausted — stop`); break
  }
  log(`\n=== Optimization round ${iter}/${ITERATIONS} | best ${bestSpeedup.toFixed(3)}x ===`)

  // Profile the current best with msprof before planning (Ascend-native signal).
  const profCmd = bestPath ? profileCommandFor(bestPath, `${EXP_DIR}/prof_iter${iter}.csv`) : ''
  if (profCmd) {
    await agentRetry(
      () => agent(
        `Profile the current best kernel with msprof and summarize the bottleneck.
Run: ${profCmd}
If msprof is unavailable (no NPU / CANN absent) the script exits 4 — report that
honestly and fall back to the timing result only.
Summarize: dominant bottleneck (Cube / MTE2 / MTE3 / vector / dispatch), occupancy,
and the single highest-leverage change. Return {bottleneck, summary, top_change}.`,
        { label: `msprof-${iter}`, phase: 'Optimize', schema: { type: 'object', additionalProperties: true } },
      ),
      { retries: 3, label: `msprof-${iter}` },
    )
  }

  const planResults = (await parallel(
    Array.from({ length: BREADTH }, (_, i) => () => agentRetry(
      () => agent(
        `You are AscendC optimizer ${i + 1}/${BREADTH} for ${opInfo.op_name} on ${TARGET_GPU}.
Current best: ${bestPath || '(none yet)'} at ${bestSpeedup.toFixed(3)}x speedup.

${PROBLEM_DEFINITION || OP}
${NOTE ? `\nUser guidance: ${NOTE}` : ''}

Previous candidates (id / speedup / compiled / correct):
${JSON.stringify(candidates.map(c => ({ id: c.id, speedup: c.speedup, compiled: c.compiled, correct: c.correct })), null, 2)}

## Task (ONE optimization this turn)
1. Read the current best kernel at ${bestPath || '(the reference)'}.
2. Pick ONE AscendC-specific improvement, e.g.:
   - Double-buffered TQue for MTE2/Cube pipelining
   - FRACTAL_NZ layout for better Cube utilization
   - Retile so the working set fits L1/L0A/L0B buffers
   - Reduce host-side dispatch overhead (fold Python dispatch into C++/ATen)
   - Vector vs Cube balance / instruction scheduling
3. Implement it, write the improved submission JSON to ${EXP_DIR}/opt_${iter}_${i + 1}.json
   (same ascendc_direct_launch format).
4. Run the eval ONCE and read the result:
   ${evalCommandFor(`${EXP_DIR}/opt_${iter}_${i + 1}.json`, `${EXP_DIR}/opt_${iter}_${i + 1}_result.json`) || '(no eval command)'}

${SUBMISSION_FORMAT}

Return {compiled, correct, speedup, candidate_latency_ms, eager_latency_ms, kernel_json_path, error}.
${ASCENDC_DIRECTIVES}`,
        { label: `opt-${iter}-${i + 1}`, phase: 'Optimize', schema: RESULT_SCHEMA },
      ),
      { retries: 5, label: `opt-${iter}-${i + 1}` },
    )),
  )).filter(Boolean)

  const newCandidates = planResults.map((r, i) => toCandidate(r, `opt-${iter}-${i + 1}`, `${EXP_DIR}/opt_${iter}_${i + 1}.json`))
  candidates = [...candidates, ...newCandidates].sort((a, b) => b.speedup - a.speedup)

  const newBest = candidates.length > 0 ? candidates[0].speedup : 0
  const improvement = (newBest - bestSpeedup) / Math.max(bestSpeedup, 0.001)
  stagnantRounds = improvement < STAGNATION_EPS ? stagnantRounds + 1 : 0
  if (newBest > bestSpeedup) { bestSpeedup = newBest; bestPath = candidates[0].path }
  log(`Round ${iter}: best ${bestSpeedup.toFixed(3)}x (improvement ${(improvement * 100).toFixed(1)}%) | stagnant ${stagnantRounds}`)
}

phase('Report')
const convergence = bestSpeedup >= TARGET ? 'converged' : (stagnantRounds >= STAGNATION_LIMIT ? 'stalled' : 'budget_exhausted')

await agentRetry(
  () => agent(
    `Write a final AscendC optimization report for ${opInfo.op_name} on ${TARGET_GPU}.
Best speedup: ${bestSpeedup.toFixed(3)}x | Status: ${convergence}
Best kernel: ${bestPath || '(none)'}
All candidates: ${JSON.stringify(candidates.slice(0, 8).map(c => ({ id: c.id, speedup: c.speedup, compiled: c.compiled, correct: c.correct })))}
Write the report to ${EXP_DIR}/report.md`,
    { label: 'final-report', phase: 'Report' },
  ),
  { retries: 3, label: 'final-report' },
)

let bestKernelCode = ''
if (bestPath) {
  const readResult = await agentRetry(
    () => agent(
      `Read the file at ${bestPath} and return its FULL contents as a string in the "code" field. Do not summarize or truncate.`,
      {
        label: 'read-best-kernel', phase: 'Report',
        schema: { type: 'object', properties: { code: { type: 'string' } }, required: ['code'] },
      },
    ),
    { retries: 3, label: 'read-best-kernel' },
  )
  bestKernelCode = guard(readResult, 'code', '')
}

return {
  input_mode: INPUT_MODE,
  solver: WORKFLOW_NAME,
  topology: 'iterative',
  overall_speedup: bestSpeedup,
  best_kernel_code: bestKernelCode || bestPath,
  best_kernel_path: bestPath,
  convergence_status: convergence,
  candidates: candidates.slice(0, 5),
  rounds_completed: ITERATIONS,
}
