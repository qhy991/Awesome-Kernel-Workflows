export const meta = {
  name: 'warpspeed-kernel-search',
  description: 'Rewindable parallel kernel-optimization search on one multi-GPU node: checkpoint tree in SQLite, mutex-scheduled GPUs, two-tier benchmarking (device-unbiased A/B screen -> locked-clock confirm), curated NCU diagnosis, cross-model review, append-only BitLessons, post-mortem ablation + frontier rewind',
  whenToUse: 'Use for sustained, experiment-driven CUDA kernel optimization campaigns on a dedicated multi-GPU node, where the kernel lives in its own git repo with a project-owned correctness harness, and you want a searchable checkpoint tree (rewindable) rather than a single greedy patch loop. Not for single-shot patches (use in-place-patch-optimization) or hosts without exclusive GPU access.',
  phases: [
    { title: 'Init',       detail: 'Materialize run config (paths, command strings), preflight tools, scaffold/lock harness' },
    { title: 'Calibrate',  detail: 'One-time cross-device noise measurement defining the significance margin' },
    { title: 'Seed',       detail: 'Register the baseline checkpoint with tier-2 latency and cached NCU profile' },
    { title: 'Plan',       detail: 'Recover orphans, snapshot the search tree, plan N experiment specs under the allocation policy' },
    { title: 'Generate',   detail: 'Per spec: implementor edits a git worktree; cross-model reviewer (codex) verifies independently; <=3 iterations' },
    { title: 'Screen',     detail: 'Tier-1 interleaved A/B benchmark vs parent on a pool GPU (device-unbiased relative speedup)' },
    { title: 'Confirm',    detail: 'Tier-2 absolute latency on the locked-clock canonical device when decision-relevant' },
    { title: 'Profile',    detail: 'Curated NCU sections + analyst diagnosis for every benched candidate (parents cached once ever)' },
    { title: 'Record',     detail: 'Finalize experiment rows, hard-dedup nodes, promote checkpoints, blocked counting, extract BitLessons' },
    { title: 'Postmortem', detail: 'Blame-assignment on blocked checkpoints; one ablation experiment before any rewind; replay queued after rewind' },
    { title: 'Maintain',   detail: 'Worktree cleanup, lesson compaction every COMPACT_EVERY rounds, budget accounting' },
    { title: 'Report',     detail: 'Leaderboard, lessons, rewinds, stop reason; human-readable report in the state dir' },
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

const WORKFLOW_NAME = 'warpspeed-kernel-search'

// --- genome self-report: INLINE (rich, doer-written) + entry scribe ---
// Doer agents append result-bearing lines to <genome_dir>/genome.jsonl; top-level
// phase() calls also invoke __genomeReport for live "entered" traces. genome_dir =
// args.exp_dir (KerSor session) when set, else state_dir / <project>/.warpspeed.
// The "__genomeReport" sentinel prevents scripts/patch-genome-report.js from
// double-injecting. See _meta/genome-trajectory-schema.md.

async function __genomeReport(phaseName, wfName) {
  try {
    const __dir = genomeDir()
    await agentRetry(() => agent(
      'Append exactly one line to ' + __dir + '/genome.jsonl (create it if missing; use a shell append: printf %s\\n ... >> file). ' +
      'The line must be this JSON on ONE line: {"workflow":"' + wfName + '","phase":"' + phaseName + '","ts":"<UTC>","status":"entered"}. ' +
      'Produce <UTC> by running: date -u +%Y-%m-%dT%H:%M:%SZ . Do nothing else; modify no other file. Echo the exact line you appended.',
      { label: 'genome:' + phaseName, phase: phaseName }
    ), { retries: 5 })
  } catch (__e) { /* observability must never break the workflow */ }
}


// --- BEGIN inlined arg_guard (Workflow runtime parses scripts as bare scripts,
//                              not ES modules; static imports are rejected) ---

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


// --- BEGIN inlined grounding contract (mirrors _substrate/grounding.js) ---
const GROUNDING_INSTRUCTION = [
  'GROUNDING CONTRACT (mandatory):',
  '',
  '1. Before reporting any numeric or categorical result, you MUST have executed',
  '   a real command via the Bash tool (compile / test / benchmark / profile /',
  '   wsdb query) that produced the value. Citing a value from imagination,',
  '   prior knowledge, or analogy is a contract violation.',
  '',
  '2. If the workflow tells you to run a tool or script that does NOT exist on',
  '   disk or on PATH, do NOT try to substitute, simulate, or rationalize. Return:',
  '       { "grounded": false, "missing": "<exact tool or script that was absent>" }',
  '   and stop. The schema accepts this shape.',
  '',
  '3. If a command fails (non-zero exit), do NOT invent the value it would',
  '   have produced. Return:',
  '       { "grounded": false, "error": "<stderr tail or short summary>" }',
  '   and stop.',
  '',
  '4. Numeric fields in the schema are REJECTED downstream when grounded === false.',
  '   Do not work around the schema with placeholders, zeros, or "typical" values.',
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
// --- END inlined grounding contract ---

// =============================================================================
// WarpSpeed - rewindable parallel kernel-optimization search
// =============================================================================
//
// Decoupling contract (the core design rule):
//   - This workflow directory holds ONLY code + templates. It is never written
//     to at run time.
//   - The TARGET kernel project is args.project_dir (any git repo on disk).
//   - All run state (SQLite tree, BitLessons, worktrees, review clones, NCU
//     cache, logs, results, materialized config) lives in <project>/.warpspeed/
//     (args.state_dir to override).
//   - GPU lock files are machine-global (/tmp/warpspeed/locks) so concurrent
//     searches on different projects can never double-book a device.
//   - All tool invocations come from command strings rendered at Init by
//     tools/render_config.py - this file contains no literal tool commands.
//
// Required args:
//   project_dir    Absolute path to the kernel git repo being optimized
//   build_command  Project build command, run verbatim inside a worktree;
//                  must produce the self-timing benchmark binary
//   binary_path    Repo-relative path of the built binary (see harness README
//                  for the LAT_US/--reps/--warmup/--shape binary contract)
//   kernel_paths   Comma-separated repo-relative source paths agents may edit
//                  (this is the reviewer's diff allowlist)
//   target_gpu     e.g. "H100" | "B200" | "sm90" | "sm100"
//
// Optional args:
//   warpspeed_dir      Path to the WarpSpeed workflow dir (default 'WarpSpeed',
//                      relative to the session cwd; resolved absolute at Init)
//   state_dir          Default <project_dir>/.warpspeed
//   harness_dir        Default <project_dir>/harness (scaffolded if missing)
//   iterations         Max search rounds (default 20)
//   budget_gpu_minutes GPU-minutes ceiling tracked via the gpu_run ledger
//   target_latency_us  Stop when the confirmed best reaches this
//   parallel_agents    Experiment fan-out per round (default from config: 4)
//   reviewer_cmd       Cross-model reviewer CLI (default from config: codex)
//   bench_shape        Shape name used for screen/confirm/profile (default 'default')
//   op_description     Free-text task context for the planner/implementor
//   single_spec_json   M2 mode: run exactly ONE hand-written spec (JSON object
//                      or string) through the full candidate path, then report
//   config_overrides   JSON object merged into the materialized config
//   exp_dir            KerSor session run dir (genome.jsonl + report mirror); optional standalone
//
// KerSor / standard aliases (when the WarpSpeed-specific names are absent):
//   project_dir  <- ggml_root | dirname(kernel_path)
//   build_command <- compile_command
//   kernel_paths <- kernel_path (single file)
//
// =============================================================================

function dirnamePath(p) {
  const s = String(p || '')
  const i = s.lastIndexOf('/')
  return i > 0 ? s.slice(0, i) : (s ? '.' : '')
}

let PROJECT_DIR = args.project_dir || args.ggml_root || ''
if (!PROJECT_DIR && args.kernel_path) PROJECT_DIR = dirnamePath(args.kernel_path)

const BUILD_CMD = args.build_command || args.compile_command || ''
const BINARY_PATH = args.binary_path || ''
const KERNEL_PATHS_RAW = args.kernel_paths || args.kernel_path || ''
const KERNEL_PATHS = String(KERNEL_PATHS_RAW).split(',').map(s => s.trim()).filter(Boolean)
const TARGET_GPU = args.target_gpu || ''
const WARPSPEED_DIR_ARG = args.warpspeed_dir || 'WarpSpeed'
const STATE_DIR_ARG = args.state_dir || ''
const HARNESS_DIR_ARG = args.harness_dir || ''
const EXP_DIR = args.exp_dir || ''
const ITERATIONS = Math.max(1, parseInt(args.iterations || 20, 10))
const BUDGET_GPU_MINUTES = args.budget_gpu_minutes != null ? Number(args.budget_gpu_minutes) : null
const TARGET_LATENCY_US = args.target_latency_us != null ? Number(args.target_latency_us) : null
const PARALLEL_AGENTS_ARG = args.parallel_agents != null ? parseInt(args.parallel_agents, 10) : null
const REVIEWER_CMD_ARG = args.reviewer_cmd || ''
const BENCH_SHAPE = args.bench_shape || 'default'
const OP_DESC = args.op_description || ''
const SINGLE_SPEC_RAW = Object.prototype.hasOwnProperty.call(args, 'single_spec_json') ? args.single_spec_json : null
const SINGLE_SPEC = SINGLE_SPEC_RAW
  ? (typeof SINGLE_SPEC_RAW === 'string' ? JSON.parse(SINGLE_SPEC_RAW) : SINGLE_SPEC_RAW)
  : null
const CONFIG_OVERRIDES_RAW = Object.prototype.hasOwnProperty.call(args, 'config_overrides') ? args.config_overrides : null
const CONFIG_OVERRIDES = CONFIG_OVERRIDES_RAW
  ? (typeof CONFIG_OVERRIDES_RAW === 'string' ? JSON.parse(CONFIG_OVERRIDES_RAW) : CONFIG_OVERRIDES_RAW)
  : {}

// --- Project-native integration args (embedded inference-engine operators, e.g.
// llama.cpp .cuh). ADDITIVE: when the integration-strategist routes to standalone
// (the default), NONE of these are consulted and the legacy benchmark path is
// byte-identical. PROJECT_ROOT reuses PROJECT_DIR; BUILD_CMD already exists above
// (no duplicate const); WarpSpeed has no standalone benchmark const, so the embedded
// eval uses the rendered confirm/correctness commands threaded via the task contract.
// register_script is only needed for embedded_dispatch. See _substrate/integration/ROLLOUT.md. ---
const PROJECT_ROOT = args.project_root || PROJECT_DIR
const REGISTER_SCRIPT = args.register_script || ''

function genomeDir() {
  return EXP_DIR || STATE_DIR_ARG || (PROJECT_DIR ? `${PROJECT_DIR}/.warpspeed` : '.')
}

// --- profiling-strategist wiring (additive): pick the analysis METHOD once per
// run (backend x task x host); the agent only classifies op_class+size (fuzzy),
// the substrate STAMPS confidence -- the model must NOT assign confidence. The
// existing NCU prompts honor PROFILING_DECISION below. See
// _substrate/profiling/README.md. Falls back to native_profiler if undecided. ---
const SUBSTRATE_DIR = args.substrate_dir || '_substrate'
const SUBSTRATE_PY = args.substrate_command_prefix || 'python3'
const PROFILING_MANIFEST = args.backend_manifest || `${SUBSTRATE_DIR}/backends/cuda/manifest.json`
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }
function substrateInstruction(script, cliArgs) {
  return `Run exactly: \`${SUBSTRATE_PY} ${SUBSTRATE_DIR}/${script} ${cliArgs}\`.`
}
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured' }

function genomeFooter(phaseName, extraFields = '') {
  const dir = genomeDir()
  const tail = extraFields ? `, ${extraFields}` : ''
  return [
    '',
    `Append exactly one line to ${dir}/genome.jsonl (create if missing; shell append with >>).`,
    'Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ',
    `One-line JSON: workflow="${WORKFLOW_NAME}", phase="${phaseName}", ts=<UTC>, status="done"${tail}.`,
  ].join('\n')
}

const MISSING = []
if (!PROJECT_DIR) MISSING.push('project_dir')
if (!BUILD_CMD) MISSING.push('build_command')
if (!BINARY_PATH) MISSING.push('binary_path')
if (!KERNEL_PATHS.length) MISSING.push('kernel_paths')
if (!TARGET_GPU) MISSING.push('target_gpu')
if (MISSING.length) {
  throw new Error(
    `warpspeed-kernel-search: missing required arg(s): ${MISSING.join(', ')}. ` +
    'This workflow does not synthesize substitutes; provide real values for each.'
  )
}

// =============================================================================
// Pure-JS policy & significance math (single source of truth - unit-tested by
// tests/workflow/dryrun.test.mjs; the shell scripts only report raw numbers)
// =============================================================================

function chunkArray(items, size) {
  const out = []
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size))
  return out
}

function noisePct(withinDeviceStdPct, crossDeviceSigmaPct) {
  return Math.max(Number(withinDeviceStdPct) || 0, Number(crossDeviceSigmaPct) || 0)
}

function isSignificant(relSpeedupPct, noise, minGainPct) {
  return Number(relSpeedupPct) > Math.max(Number(minGainPct), 2 * noise)
}

function shouldConfirm(relSpeedupPct, newBestClaim, confirmMarginPct) {
  return Boolean(newBestClaim) || Math.abs(Number(relSpeedupPct)) < Number(confirmMarginPct)
}

function predictionGap(predictedPct, achievedPct) {
  if (predictedPct == null || achievedPct == null) return null
  return Math.round((Number(predictedPct) - Number(achievedPct)) * 100) / 100
}

// Allocation: exploit/explore/wildcard quotas over N specs, shifting linearly
// toward exploit as the GPU-minute budget depletes; wildcards reach 0 in the
// last quartile.
function allocationQuotas(nSpecs, allocation, quartile) {
  const t = Math.min(1, Math.max(0, (Number(quartile) || 0) / 3))
  let wildcard = allocation.wildcard * (quartile >= 3 ? 0 : (1 - t))
  let explore = allocation.explore * (1 - 0.5 * t)
  let exploit = 1 - wildcard - explore
  const counts = {
    exploit: Math.max(1, Math.round(nSpecs * exploit)),
    explore: Math.round(nSpecs * explore),
    wildcard: Math.round(nSpecs * wildcard),
  }
  while (counts.exploit + counts.explore + counts.wildcard > nSpecs) {
    if (counts.wildcard > 0) counts.wildcard--
    else if (counts.explore > 0) counts.explore--
    else counts.exploit--
  }
  while (counts.exploit + counts.explore + counts.wildcard < nSpecs) counts.exploit++
  return counts
}

// FNV-1a over the sorted strategy-tag set: deterministic dedup key.
function strategySetHash(tags) {
  const canon = Array.from(new Set((tags || []).map(String))).sort().join('|')
  let h = 0x811c9dc5
  for (let i = 0; i < canon.length; i++) {
    h ^= canon.charCodeAt(i)
    h = (h >>> 0) * 0x01000193 >>> 0
  }
  return ('00000000' + h.toString(16)).slice(-8)
}

function slugify(text) {
  return String(text || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 32) || 'exp'
}

function bestLatency(snap) {
  const lats = (snap.frontier || []).map(f => f.latency_us).filter(v => v != null)
  return lats.length ? Math.min.apply(null, lats) : null
}

function frontierByCommit(snap, commit) {
  return (snap.frontier || []).find(f => f.commit === commit) || null
}

function stopReason(snap, cfg, pmAttempted) {
  const b = snap.budget || {}
  if (b.gpu_minutes_remaining != null && b.gpu_minutes_remaining <= 0) return 'budget_exhausted'
  const best = bestLatency(snap)
  if (TARGET_LATENCY_US && best != null && best <= TARGET_LATENCY_US) return 'target_latency_reached'
  const k = cfg.K_BLOCKED
  const live = snap.frontier || []
  const pmPending = (snap.postmortem_due || []).some(c => !pmAttempted.has(c))
  if (live.length && live.every(f => f.blocked_count >= k) &&
      !(snap.queued_specs || []).length && !pmPending) {
    return 'frontier_fully_blocked'
  }
  return null
}

function tokensUsed() {
  try {
    if (typeof budget === 'function') return budget()
    if (budget && typeof budget.spent === 'function') return budget.spent()
  } catch (e) { /* budget global unavailable in this runtime */ }
  return null
}

function fmt(v) {
  if (v === undefined || v === null || v === '') return '(not provided)'
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

// =============================================================================
// Schemas
// =============================================================================

const INIT_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    config: { type: 'object', description: 'The FULL materialized config JSON exactly as printed by render_config (the .config field)' },
    preflight: {
      type: 'object',
      properties: {
        gpus: { type: 'number' }, git_ok: { type: 'boolean' }, wsdb_ok: { type: 'boolean' },
        reviewer_ok: { type: 'boolean' }, ncu_ok: { type: 'boolean' }, sanitizer_ok: { type: 'boolean' },
      },
    },
    harness_ready: { type: 'boolean', description: 'true iff the harness exists and is NOT the unimplemented template' },
    harness_scaffolded: { type: 'boolean' },
    action_required: { type: 'string', description: 'Set when a human must act before the search can start' },
  },
  required: ['config', 'preflight', 'harness_ready'],
})

const CAL_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    cross_device_sigma_pct: { type: 'number' },
    per_device_means_us: { type: 'object' },
    skipped: { type: 'boolean', description: 'true iff a fresh calibration already existed' },
  },
  required: ['skipped'],
})

const SEED_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    commit: { type: 'string' }, latency_us: { type: 'number' },
    ncu_fingerprint: { type: 'string' }, skipped: { type: 'boolean' },
  },
  required: ['commit', 'skipped'],
})

const SNAP_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    snapshot: { type: 'object', description: 'The round-snapshot JSON exactly as returned by the wsdb CLI' },
    orphans_pruned: { type: 'number' },
  },
  required: ['snapshot'],
})

const PLAN_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    specs: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          type: { type: 'string', enum: ['explore', 'exploit', 'wildcard', 'replay', 'ablation'] },
          parent_commit: { type: 'string' },
          hypothesis: { type: 'string', description: 'One falsifiable sentence' },
          direction_tags: { type: 'array', items: { type: 'string' } },
          predicted_gain_pct: { type: 'number', description: 'MANDATORY: predicted relative speedup vs parent, percent' },
          predicted_mechanism: { type: 'string', description: 'MANDATORY: why, 1-2 sentences naming the bottleneck' },
          interaction_rationale: { type: 'string', description: 'Required ONLY when re-proposing an existing strategy set' },
        },
        required: ['type', 'parent_commit', 'hypothesis', 'direction_tags', 'predicted_gain_pct', 'predicted_mechanism'],
      },
    },
  },
  required: ['specs'],
})

const REG_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    registered: { type: 'array', items: { type: 'object', properties: { exp_id: { type: 'string' }, worktree: { type: 'string' }, branch: { type: 'string' } } } },
    failed: { type: 'array', items: { type: 'object' } },
  },
  required: ['registered'],
})

const IMPL_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    commit: { type: 'string', description: 'git rev-parse HEAD of the worktree after the final commit' },
    build_ok: { type: 'boolean' },
    correctness_passed: { type: 'boolean', description: 'true iff the fixed correctness invocation exited 0' },
    summary: { type: 'string', description: 'One-paragraph diff summary: what was actually changed and where' },
    self_reported_changes: { type: 'array', items: { type: 'string' } },
    gave_up: { type: 'boolean' },
    failure_reason: { type: 'string' },
  },
  required: ['build_ok', 'correctness_passed', 'gave_up'],
})

const REVIEW_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    clone_head: { type: 'string', description: 'HEAD of the fresh review clone (must equal the candidate commit)' },
    independent_correctness: { type: 'boolean', description: 'Result of the reviewer\'s OWN harness run on a fresh build' },
    sanitizer_clean: { type: 'boolean', description: 'racecheck+memcheck clean (first review iteration only; else repeat prior value)' },
    paths_ok: { type: 'boolean', description: 'git diff parent..candidate touches ONLY the allowlisted kernel paths' },
    codex_ran: { type: 'boolean' },
    diff_review_pass: { type: 'boolean', description: 'Cross-model verdict on race/barrier/fence/tolerance correctness of the diff' },
    intent_compliance: { type: 'boolean', description: 'Cross-model verdict: does the diff plausibly realize the stated hypothesis' },
    objections: { type: 'array', items: { type: 'string' } },
    accept: { type: 'boolean' },
    objection: { type: 'string', description: 'Primary objection when accept=false' },
  },
  required: ['independent_correctness', 'paths_ok', 'codex_ran', 'accept'],
})

const SCREEN_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    rel_speedup_pct: { type: 'number' }, parent_mean_us: { type: 'number' },
    cand_mean_us: { type: 'number' }, within_device_std_pct: { type: 'number' },
    device: { type: 'string' }, parent_build_cached: { type: 'boolean' },
  },
  required: ['rel_speedup_pct', 'parent_mean_us', 'cand_mean_us', 'within_device_std_pct'],
})

const CONFIRM_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    latency_us_mean: { type: 'number' }, latency_us_std: { type: 'number' },
    reps: { type: 'number' }, clocks: { type: 'object' },
  },
  required: ['latency_us_mean'],
})

const PROFILE_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    cand: { type: 'object', description: '{fingerprint, key_metrics} of the candidate, verbatim from the profile JSON' },
    parent: { type: 'object', description: '{fingerprint, key_metrics} of the parent from the ncu cache (profiled now ONLY if the cache was missing)' },
    parent_was_cached: { type: 'boolean' },
    ncu_path: { type: 'string' },
  },
  required: ['cand', 'ncu_path'],
})

const ANALYST_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    bottleneck: { type: 'string' },
    evidence: { type: 'string', description: 'Which metric deltas support the bottleneck claim' },
    hypothesis_verdict: { type: 'string', enum: ['confirmed', 'refuted', 'partial'] },
    suggested_directions: { type: 'array', items: { type: 'string' } },
    escalate_full_ncu: { type: 'boolean' },
    diagnosis: { type: 'string', description: '2-4 sentence diagnosis tying mechanism to measurement' },
  },
  required: ['bottleneck', 'hypothesis_verdict', 'diagnosis'],
})

const FINISH_SCHEMA = withGroundingFields({
  type: 'object',
  properties: { written: { type: 'boolean' }, status: { type: 'string' } },
  required: ['written'],
})

const CKPT_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    merged: { type: 'array', items: { type: 'object' } },
    new_checkpoints: { type: 'array', items: { type: 'string' } },
    blocked: { type: 'array', items: { type: 'object' } },
    postmortem_due: { type: 'array', items: { type: 'string' } },
  },
  required: ['postmortem_due'],
})

const LESSON_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    appended_ids: { type: 'array', items: { type: 'string' } },
    skipped_reason: { type: 'string', description: 'Set when the high bar was not met and nothing was appended' },
  },
  required: ['appended_ids'],
})

const PM_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    suspect_strategy: { type: 'string', description: 'The assumption TAG suspected of poisoning the subtree' },
    suspect_checkpoint: { type: 'string', description: 'The checkpoint commit where the suspect strategy was baked in' },
    suspect_parent: { type: 'string', description: 'parent commit of suspect_checkpoint (from the trajectory)' },
    mechanism: { type: 'string' },
    ablation_hypothesis: { type: 'string' },
    ablation_tags: { type: 'array', items: { type: 'string' } },
    ablation_predicted_gain_pct: { type: 'number' },
    ablation_predicted_mechanism: { type: 'string' },
  },
  required: ['suspect_strategy', 'suspect_checkpoint', 'mechanism', 'ablation_hypothesis'],
})

const REWIND_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    rewound_to: { type: 'string' }, retired: { type: 'array', items: { type: 'string' } },
    lesson_id: { type: 'string' }, replay_queued: { type: 'boolean' },
  },
  required: ['rewound_to', 'replay_queued'],
})

const COMPACT_SCHEMA = withGroundingFields({
  type: 'object',
  properties: { appended: { type: 'number' }, superseded: { type: 'number' }, digest_set: { type: 'boolean' } },
  required: ['appended', 'superseded'],
})

const CLEAN_SCHEMA = withGroundingFields({
  type: 'object',
  properties: { worktrees_removed: { type: 'number' }, review_dirs_removed: { type: 'number' } },
  required: ['worktrees_removed'],
})

const REPORT_SCHEMA = withGroundingFields({
  type: 'object',
  properties: {
    report_path: { type: 'string' }, best_commit: { type: 'string' },
    best_latency_us: { type: 'number' }, baseline_latency_us: { type: 'number' },
    lessons_total: { type: 'number' },
  },
  required: ['report_path'],
})

// =============================================================================
// Phase: Init
// =============================================================================

phase('Init'); await __genomeReport('Init', WORKFLOW_NAME)
log(`WarpSpeed: project=${PROJECT_DIR} target=${TARGET_GPU} fan-out<=${PARALLEL_AGENTS_ARG || '(config)'} max-rounds=${ITERATIONS}`)

const overrideSets = []
const baseOverrides = Object.assign({ target_gpu: TARGET_GPU }, CONFIG_OVERRIDES)
if (BUDGET_GPU_MINUTES != null) baseOverrides.BUDGET_GPU_MINUTES = BUDGET_GPU_MINUTES
if (TARGET_LATENCY_US != null) baseOverrides.TARGET_LATENCY_US = TARGET_LATENCY_US
if (PARALLEL_AGENTS_ARG != null) baseOverrides.PARALLEL_AGENTS = PARALLEL_AGENTS_ARG
if (REVIEWER_CMD_ARG) baseOverrides.REVIEWER_CMD = REVIEWER_CMD_ARG
for (const key of Object.keys(baseOverrides)) {
  overrideSets.push(`--set ${key}=${JSON.stringify(JSON.stringify(baseOverrides[key]))}`)
}

const initResult = await agentRetry(() => agent(
  [
    'Initialize a WarpSpeed kernel-search run (config materialization + preflight). Work strictly via the Bash/Read tools.',
    '',
    `WarpSpeed workflow dir (may be relative to your cwd): ${WARPSPEED_DIR_ARG}`,
    `Target project (kernel git repo): ${PROJECT_DIR}`,
    `State dir override: ${fmt(STATE_DIR_ARG)}   Harness dir override: ${fmt(HARNESS_DIR_ARG)}`,
    '',
    'Steps, in order:',
    '1. Resolve the WarpSpeed dir to an ABSOLUTE path and verify it contains tools/render_config.py and config/config.defaults.json.',
    '2. Read config/config.defaults.json and take command_templates.render_config. Substitute {warpspeed} with the absolute WarpSpeed dir and {project} with the project dir, then execute it via the Bash tool with these EXTRA flags appended:',
    `   ${STATE_DIR_ARG ? `--state-dir ${STATE_DIR_ARG}` : '(no --state-dir flag: default <project>/.warpspeed)'}`,
    `   ${HARNESS_DIR_ARG ? `--harness-dir ${HARNESS_DIR_ARG}` : '(no --harness-dir flag: default <project>/harness)'}`,
    `   ${overrideSets.join(' ')}`,
    '   The tool prints {"ok": true, "config_path": ..., "config": {...}}. Keep the config object: you will return it VERBATIM in the `config` field.',
    '3. Create the state subdirectories listed in config.paths: worktrees, review, builds, ncu_cache, logs, results.',
    `4. If the state dir is inside the project repo, ensure the project .gitignore covers it (append a line if missing).`,
    '5. Preflight, each via the Bash tool (record booleans; do not fake):',
    '   - git available in the project dir and the project is a git repo with at least one commit (git_ok)',
    '   - run config.commands.wsdb with subcommand `init` -> wsdb_ok',
    `   - the reviewer CLI responds: run \`${'$'}{config.REVIEWER_CMD} --help\` (or --version) -> reviewer_ok`,
    '   - GPU visibility: run nvidia-smi with the -L flag and COUNT the GPUs -> gpus',
    '   - the NCU profiler binary exists on PATH (check with `command -v ncu`) -> ncu_ok',
    '   - compute-sanitizer exists on PATH (`command -v compute-sanitizer`) -> sanitizer_ok',
    `6. Harness check at config.paths.harness_dir: it must contain correctness.py, problem_shapes.json, tolerances.json AND correctness.py must NOT still raise NotImplementedError (grep for it).`,
    '   - If the harness dir is MISSING or empty: copy every file from <warpspeed>/harness-template/ into it, set harness_scaffolded=true, harness_ready=false, and action_required="Complete the scaffolded harness (see harness/README.md), verify it by hand on the baseline binary, then re-run WarpSpeed."',
    '   - If it exists but still contains NotImplementedError: harness_ready=false, action_required as above (do NOT overwrite human files).',
    '   - If ready: make it read-only with: chmod -R a-w <harness_dir>  (re-apply even if already read-only), harness_ready=true.',
    '7. Store the config snapshot in the DB: run config.commands.wsdb with `config-set --json @<file>` after writing the config JSON to <state>/logs/config_snapshot.json.',
    '',
    'Return the config object verbatim plus the preflight booleans. If any preflight tool is absent that the run cannot proceed without (wsdb, git), use the grounded=false shape instead of guessing.',
    GROUNDING_INSTRUCTION,
    genomeFooter('Init'),
  ].join('\n'),
  { phase: 'Init', label: 'init:materialize', schema: INIT_SCHEMA }
), { retries: 5 })

const init = classifyResult(initResult)
if (init.state !== 'grounded') {
  return { ok: false, grounded: false, error: `init failed: ${JSON.stringify(init)}`, action_required: null }
}
if (!init.value.harness_ready) {
  log('Harness not ready - returning early for human completion.')
  return {
    ok: false,
    grounded: true,
    action_required: init.value.action_required ||
      'Complete the correctness harness in the project, then re-run WarpSpeed.',
    harness_scaffolded: Boolean(init.value.harness_scaffolded),
  }
}

const CFG = init.value.config
const C = CFG.commands || {}
const P = CFG.paths || {}
const PARALLEL_AGENTS = Number(CFG.PARALLEL_AGENTS) || 4
const MAX_REVIEW_ITERS = Number(CFG.MAX_REVIEW_ITERS) || 3
const REVIEWER_CMD = String(CFG.REVIEWER_CMD || 'codex exec')
if (!C.wsdb || !C.screen || !C.confirm || !C.ncu_profile || !C.correctness) {
  throw new Error('init returned a config without rendered command strings - aborting before any GPU work')
}
const pf = init.value.preflight || {}
if (pf.reviewer_ok === false) {
  return {
    ok: false, grounded: true,
    action_required: `Cross-model reviewer CLI is not available ("${REVIEWER_CMD}"). Install/authenticate it (or pass reviewer_cmd=...) - WarpSpeed will not run reviews with the same model that wrote the code.`,
  }
}
log(`Init ok: ${pf.gpus || '?'} GPUs, state=${P.state_dir}, arch=${CFG.arch}`)

// =============================================================================
// Task contract threaded into every prompt
// =============================================================================

function taskContract(expId) {
  return [
    '# WarpSpeed Task Contract',
    `- project_dir: ${P.project_dir}`,
    `- state_dir: ${P.state_dir}`,
    `- harness_dir: ${P.harness_dir} (READ-ONLY - never edit, chmod, or work around it)`,
    `- arch: ${CFG.arch}   hardware_facts: ${P.hardware_facts}`,
    `- build_command (run inside a worktree/clone root): ${BUILD_CMD}`,
    `- binary_path (repo-relative build output): ${BINARY_PATH}`,
    `- editable kernel paths (THE ONLY FILES YOU MAY MODIFY): ${KERNEL_PATHS.join(', ')}`,
    `- bench shape: ${BENCH_SHAPE}`,
    '',
    '# Tool commands (use VERBATIM via the Bash tool; append arguments after them)',
    `- wsdb CLI:        ${C.wsdb}`,
    `- correctness:     ${C.correctness} <abs-binary>   (fixed invocation - never compose your own harness call)`,
    `- screen bench:    ${C.screen} <parent-bin> <cand-bin> --shape ${BENCH_SHAPE}`,
    `- confirm bench:   ${C.confirm} <abs-binary> --shape ${BENCH_SHAPE}`,
    `- ncu profile:     ${C.ncu_profile} <abs-binary> <out-json> --shape ${BENCH_SHAPE}`,
    `- sanitizer race:  ${C.sanitizer_racecheck} <abs-binary> --reps 1 --shape <smallest-shape>`,
    `- sanitizer mem:   ${C.sanitizer_memcheck} <abs-binary> --reps 1 --shape <smallest-shape>`,
    '',
    '# Discipline (mandatory)',
    '1. NEVER run a GPU binary, profiler, or sanitizer except through the tool commands above (they serialize devices via machine-global locks). Builds need no GPU and run directly.',
    '2. The harness dir is read-only ground truth. If a command of yours fails because of it, report the failure - do not chmod or copy-and-edit it.',
    `3. Only the listed kernel paths may change. Any other diff will be rejected in review.`,
    `4. Prepend WARPSPEED_EXP_ID=${expId || 'none'} to every tool command above (shell env assignment) so GPU-minutes are attributed to this experiment.`,
    '5. Relay wsdb/tool JSON outputs verbatim into your structured return - never invent fields.',
    '',
    GROUNDING_INSTRUCTION,
  ].join('\n')
}

// =============================================================================
// Phase: Calibrate
// =============================================================================

phase('Calibrate'); await __genomeReport('Calibrate', WORKFLOW_NAME)
const calResult = await agentRetry(() => agent(
  [
    'Cross-device noise calibration for WarpSpeed (skip if already done).',
    '',
    `1. Run: ${C.wsdb} calibration-get`,
    '   If it returns fresh=true with a calibration object, return {skipped: true} plus its cross_device_sigma_pct.',
    `2. Otherwise: build the project at its CURRENT HEAD in ${P.project_dir} by running the build command verbatim: ${BUILD_CMD}`,
    `3. Run the calibration tool (it holds ALL device locks, so expect it to wait for a free node): ${C.calibrate} ${P.project_dir}/${BINARY_PATH} --shape ${BENCH_SHAPE}`,
    '4. Persist the result JSON: write it to a temp file and run:',
    `   ${C.wsdb} calibration-set --json @<temp-file>`,
    '5. Return cross_device_sigma_pct, per_device_means_us, skipped=false.',
    '',
    taskContract('calibrate'),
  ].join('\n'),
  { phase: 'Calibrate', label: 'calibrate', schema: CAL_SCHEMA }
), { retries: 5 })
const cal = classifyResult(calResult)
if (cal.state !== 'grounded') {
  return { ok: false, grounded: false, error: `calibration failed: ${JSON.stringify(cal)}` }
}
log(`Calibration: sigma=${cal.value.cross_device_sigma_pct != null ? cal.value.cross_device_sigma_pct + '%' : '(from db)'} skipped=${cal.value.skipped}`)

// =============================================================================
// Phase: Seed
// =============================================================================

phase('Seed'); await __genomeReport('Seed', WORKFLOW_NAME)

// Resolve the profiling METHOD ONCE, before the baseline NCU profile is taken,
// and reuse PROFILING_DECISION across Seed/Profile (don't re-resolve per round).
{
  const _genDir = genomeDir()
  const _pd = await agentRetry(() => agent(
    `Classify the kernel under optimization (op_class one of attention|gemm|elementwise|reduction|default; size one of tiny|small|large) from this context: ` +
    `editable kernel paths=${KERNEL_PATHS.join(', ')}; bench shape="${BENCH_SHAPE}"; task="${OP_DESC || '(none)'}". ` +
    `Pick the closest op_class and size (fuzzy is fine; use default/small if unsure). Then ` +
    substrateInstruction('profiling/profiling_strategist.py',
      `resolve --backend-manifest ${PROFILING_MANIFEST} --task <op_class> --size <size> --cache ${_genDir}/prof_cache.json --trajectory ${_genDir}/genome.jsonl`) +
    ` Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}. Do NOT assign confidence yourself; relay exactly what the script stamped.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Seed', schema: JSON_PASSTHROUGH }
  ), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
  log(`Profiling-strategist: method=${PROFILING_DECISION.method} confidence=${PROFILING_DECISION.confidence}`)
}

// --- integration-strategist: route eval mode (standalone vs embedded_*). For an
// inference-engine embedded operator (e.g. a llama.cpp .cuh referenced by project-only
// headers) the candidate cannot compile as a standalone TU, so the legacy A/B-screen
// benchmark path is unreachable; the strategist routes to embedded_inplace/dispatch.
// ADDITIVE: when method==='standalone' (default) the legacy candidate pipeline runs
// byte-identically — IS_EMBEDDED stays false and the new serial branch is skipped.
// WarpSpeed has no backend driver, so there is no USE_DRIVER/USE_DRIVER_STANDALONE;
// IS_EMBEDDED alone gates the new branch. The strategist reads the single editable
// kernel file (KERNEL_PATHS[0]). See _substrate/integration/ROLLOUT.md.
const PRIMARY_KERNEL_PATH = KERNEL_PATHS[0] || ''
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _genDir = genomeDir()
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Read ${PRIMARY_KERNEL_PATH}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single translation unit — e.g. a llama.cpp .cuh with project-only deps). Then ` +
    substrateInstruction('integration/integration_strategist.py',
      `resolve --kernel "${PRIMARY_KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' --cache ${_genDir}/integ_cache.json --trajectory ${_genDir}/genome.jsonl`) +
    ` Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Seed', schema: JSON_PASSTHROUGH }
  ), { retries: 5, allowNull: true })
  if (_integ && _integ.method) INTEGRATION_DECISION = _integ
}
log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error('integration-strategist returned derive_adapter — provide project_root + build_command (+ register_script for dispatch) for the embedded operator')
}
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
// embedded_inplace mutates the project operator file in place; back it up ONCE here
// so every serial candidate restores to a pristine original and the exit net can too.
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${genomeDir()}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Byte-exact backup (once): run \`cp -a "${PRIMARY_KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Seed', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// Unconditional exit-restore net for embedded_inplace: each serial candidate restores
// after itself, but this guarantees the project operator file is pristine on EVERY exit
// path (no-op when not embedded_inplace). Call __exitRestore() before each return below.
async function __exitRestore() {
  if (!ORIGINAL_BACKUP) return
  try {
    await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${P.project_dir}/${PRIMARY_KERNEL_PATH}"\` and confirm.`,
      { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
  } catch (e) { /* restore is best-effort on the way out */ }
}
// A-O1 closure: the strategist may pick native_profiler (ncu) but ncu is absent on
// this host (preflight pf.ncu_ok=false) — downgrade so Profile uses perf_heuristic
// instead of trying ncu against a missing binary.
if (PROFILING_DECISION.method === 'native_profiler' && !pf.ncu_ok) {
  log(`profiling: native_profiler but no ncu (pf.ncu_ok=false) -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'warpspeed-perf', rationale: 'native_profiler but ncu unavailable -> perf_heuristic' }
}

// Guard sentence appended verbatim to every NCU/profile prompt so the doer honors
// the strategist's method without re-resolving. Built from the resolved decision.
const PROFILING_GUARD =
  `\nProfiling-strategist selected method='${PROFILING_DECISION.method}', confidence='${PROFILING_DECISION.confidence}'. ` +
  `If method !== 'native_profiler', do NOT run ncu/compute-sanitizer-based profiling; measure throughput instead and stamp emitted bottlenecks with evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. ` +
  `If method === 'native_profiler', proceed as written.`

const seedResult = await agentRetry(() => agent(
  [
    'Seed the WarpSpeed baseline checkpoint (skip if already seeded).',
    '',
    `1. Run: ${C.wsdb} frontier`,
    '   If the frontier already contains a checkpoint, return {skipped: true, commit: <frontier_commit>}.',
    `2. Otherwise, in ${P.project_dir}:`,
    '   - require a clean working tree (git status --porcelain empty); if dirty, return grounded=false with error "project dirty - commit or stash first".',
    '   - record HEAD: git rev-parse HEAD -> <commit>; tag it: git tag -f ckpt/baseline <commit>',
    `   - build: ${BUILD_CMD}`,
    `3. Tier-2 confirm latency: ${C.confirm} ${P.project_dir}/${BINARY_PATH} --shape ${BENCH_SHAPE}  -> latency_us_mean`,
    `4. NCU profile into the checkpoint cache (profile each checkpoint ONCE, ever): ${C.ncu_profile} ${P.project_dir}/${BINARY_PATH} ${P.ncu_cache}/<commit>.json --shape ${BENCH_SHAPE}`,
    PROFILING_GUARD,
    '5. Seed the DB: write {"commit": <commit>, "latency_us": <mean>, "assumptions": [], "ncu_fingerprint": <fingerprint>, "key_metrics": <key_metrics>} to a temp file and run:',
    `   ${C.wsdb} seed-baseline --json @<temp-file>`,
    '6. Return commit, latency_us, ncu_fingerprint, skipped=false.',
    '',
    taskContract('seed'),
  ].join('\n'),
  { phase: 'Seed', label: 'seed:baseline', schema: SEED_SCHEMA }
), { retries: 5 })
const seed = classifyResult(seedResult)
if (seed.state !== 'grounded') {
  await __exitRestore()
  return { ok: false, grounded: false, error: `seeding failed: ${JSON.stringify(seed)}` }
}
log(`Baseline: ${seed.value.commit && seed.value.commit.slice(0, 10)} ${seed.value.latency_us ? seed.value.latency_us + 'us' : '(already seeded)'}`)

// =============================================================================
// Candidate pipeline: generate -> review loop -> screen -> route -> confirm ->
// profile -> analyst -> record (one call per stage; control flow lives HERE)
// =============================================================================

function wikiPagesFor(tags) {
  const seen = new Set()
  const pages = []
  for (const tag of tags || []) {
    for (const p of (CFG.wiki_index || {})[tag] || []) {
      if (!seen.has(p)) { seen.add(p); pages.push(p) }
    }
  }
  return pages.slice(0, 3)
}

function implementorPrompt(spec, iter, priorObjections) {
  const wt = `${P.worktrees}/${spec.exp_id}`
  return [
    `You are the WarpSpeed IMPLEMENTOR for experiment ${spec.exp_id} (iteration ${iter}/${MAX_REVIEW_ITERS}). You implement ONE stated hypothesis in an isolated git worktree, prove it correct, and commit. You do NOT benchmark - the orchestrator does.`,
    '',
    '# Experiment spec',
    `- type: ${spec.type}`,
    `- parent checkpoint: ${spec.parent_commit}`,
    `- hypothesis: ${spec.hypothesis}`,
    `- direction tags: ${(spec.direction_tags || []).join(', ')}`,
    `- predicted gain: ${spec.predicted_gain_pct}% via: ${spec.predicted_mechanism}`,
    OP_DESC ? `- task context: ${OP_DESC}` : '',
    '',
    '# Your knowledge inputs (read them BEFORE editing)',
    `- hardware facts (constraints you must respect): ${P.hardware_facts}`,
    `- technique pages for your tags: ${wikiPagesFor(spec.direction_tags).join(' , ') || '(none matched - rely on hardware facts)'}`,
    `- prior lessons: run \`${C.wsdb} lessons-query --tags ${(spec.direction_tags || []).join(',')} --arch ${CFG.arch} --max 15\` and treat returned claims as evidence-backed priors (they survive rewinds for a reason).`,
    '',
    '# Procedure',
    `1. Your worktree is ${wt} (branch already created from the parent checkpoint). cd there; verify with: git rev-parse --abbrev-ref HEAD`,
    `2. Implement the hypothesis by editing ONLY: ${KERNEL_PATHS.join(', ')}`,
    `3. Build inside the worktree: ${BUILD_CMD}`,
    `4. Prove correctness with the FIXED invocation (exit 0 required): WARPSPEED_EXP_ID=${spec.exp_id} ${C.correctness} ${wt}/${BINARY_PATH}`,
    '5. Iterate edit/build/correctness until it passes. If after honest attempts you cannot realize the hypothesis correctly, set gave_up=true with failure_reason - do NOT substitute a different strategy to "show something" (quiet strategy substitution is caught in review and wastes a round).',
    `6. Commit EVERYTHING (even if you gave up - failed code is post-mortem evidence): git add -A the kernel paths; commit message format: "[${spec.exp_id}] ${spec.hypothesis}" on line 1, then a blank line, then what you ACTUALLY changed.`,
    '7. Report the commit hash from: git rev-parse HEAD',
    priorObjections && priorObjections.length
      ? ['', '# Reviewer objections from the previous iteration (address ALL of them)',
         ...priorObjections.map(o => `- ${o}`)].join('\n')
      : '',
    '',
    '# Hard rules',
    '- Never touch the harness dir, build flags, or any file outside the kernel paths.',
    '- Never run the binary, profiler, or sanitizer outside the tool commands in the contract below.',
    '- You know NOTHING about sibling experiments, the leaderboard, or the search tree. Do not go looking.',
    '',
    taskContract(spec.exp_id),
  ].filter(Boolean).join('\n')
}

function reviewerPrompt(spec, implOut, iter) {
  const reviewDir = `${P.review}/${spec.exp_id}`
  return [
    `You are the WarpSpeed REVIEW RUNNER for experiment ${spec.exp_id} (iteration ${iter}). You perform mechanical verification yourself, then obtain the code-review verdict from the CROSS-MODEL reviewer CLI. You RELAY that verdict - you never substitute your own judgment for it.`,
    '',
    `Candidate commit: ${implOut.commit}`,
    `Parent checkpoint: ${spec.parent_commit}`,
    `Hypothesis: ${spec.hypothesis}`,
    `Implementor's claimed changes: ${fmt(implOut.self_reported_changes)}`,
    '',
    '# Mechanical duties (do these yourself, in order)',
    `1. FRESH CLONE (never reuse the implementor's worktree): rm -rf ${reviewDir}; git clone --no-hardlinks ${P.project_dir} ${reviewDir}; git -C ${reviewDir} checkout ${implOut.commit}; record clone_head via git rev-parse HEAD (must equal the candidate commit - if not, accept=false).`,
    `2. Diff allowlist: git -C ${reviewDir} diff --name-only ${spec.parent_commit}..${implOut.commit} - EVERY path must be one of [${KERNEL_PATHS.join(', ')}]. Any other path (harness, build files, CI) => paths_ok=false, accept=false.`,
    `3. Fresh build in ${reviewDir}: ${BUILD_CMD}`,
    `4. INDEPENDENT correctness (your own run, not the implementor's claim): WARPSPEED_EXP_ID=${spec.exp_id} ${C.correctness} ${reviewDir}/${BINARY_PATH} -> independent_correctness = (exit 0).`,
    iter === 1
      ? `5. Sanitizers (first review iteration only) on the SMALLEST shape from ${P.harness_dir}/problem_shapes.json:\n   WARPSPEED_EXP_ID=${spec.exp_id} ${C.sanitizer_racecheck} ${reviewDir}/${BINARY_PATH} --reps 1 --shape <smallest>\n   WARPSPEED_EXP_ID=${spec.exp_id} ${C.sanitizer_memcheck} ${reviewDir}/${BINARY_PATH} --reps 1 --shape <smallest>\n   sanitizer_clean = both exit 0. NOTE: racecheck does not model mbarrier phases fully - a clean run is necessary, not sufficient; the cross-model review below covers barrier pairing.`
      : '5. Sanitizers already ran in iteration 1; carry sanitizer_clean forward from that result.',
    '',
    '# Cross-model review (the verdict you must relay)',
    `6. Produce the diff: git -C ${reviewDir} diff ${spec.parent_commit}..${implOut.commit} > ${reviewDir}/candidate.diff`,
    `7. Write a review request file ${reviewDir}/review_request.md containing: the hypothesis, the predicted mechanism, the diff (inline), and these questions:`,
    '   (a) CORRECTNESS REVIEW of the diff: data races, mbarrier arrive/wait phase pairing, missing async-proxy fences before TMA stores, shared-memory lifetime, tolerance handling. List concrete objections or state none.',
    '   (b) INTENT COMPLIANCE: does this diff plausibly realize the stated hypothesis (not a near-noop, not a quietly substituted different strategy)?',
    '   (c) Verify build flags and harness invocation are untouched by the diff.',
    '   Ask it to answer ONLY with JSON: {"diff_review_pass": bool, "intent_compliance": bool, "objections": ["..."]}',
    `8. Invoke the reviewer CLI via the Bash tool (it is a DIFFERENT model on purpose): ${REVIEWER_CMD} "$(cat ${reviewDir}/review_request.md)"`,
    '   Parse the JSON out of its reply (it may wrap it in prose; extract the last JSON object). Set codex_ran=true and relay diff_review_pass / intent_compliance / objections EXACTLY as the CLI returned them.',
    '   If the CLI errors or returns nothing parseable: codex_ran=false, accept=false, objection="reviewer_unavailable" - NEVER review the diff yourself instead.',
    '',
    '# Verdict',
    'accept = clone_head matches AND paths_ok AND independent_correctness AND (sanitizer_clean !== false) AND codex_ran AND diff_review_pass AND intent_compliance.',
    'When accept=false, set objection to the single most actionable objection (and include all of them in objections[]).',
    '',
    taskContract(spec.exp_id),
  ].join('\n')
}

function screenPrompt(spec, implOut) {
  const wt = `${P.worktrees}/${spec.exp_id}`
  const parentBin = `${P.builds}/${spec.parent_commit}/${BINARY_PATH}`
  return [
    `Tier-1 SCREEN benchmark for experiment ${spec.exp_id}: candidate vs parent, interleaved on ONE pool device.`,
    '',
    `1. Parent binary cache: if ${parentBin} is missing, build it once:`,
    `   git -C ${P.project_dir} worktree add --detach ${P.builds}/wt-${spec.exp_id} ${spec.parent_commit}`,
    `   then run the build command inside it (${BUILD_CMD}), then: mkdir -p $(dirname ${parentBin}) && cp ${P.builds}/wt-${spec.exp_id}/${BINARY_PATH} ${parentBin}`,
    `   then remove the temp worktree: git -C ${P.project_dir} worktree remove --force ${P.builds}/wt-${spec.exp_id}; git -C ${P.project_dir} worktree prune`,
    `   Set parent_build_cached=true when the cache already existed.`,
    `2. Candidate binary: ${wt}/${BINARY_PATH} (rebuild in the worktree with the build command if missing).`,
    `3. Screen (one session, interleaved A/B - the tool refuses to run outside the GPU launcher): WARPSPEED_EXP_ID=${spec.exp_id} ${C.screen} ${parentBin} ${wt}/${BINARY_PATH} --shape ${BENCH_SHAPE}`,
    '4. Relay the JSON fields verbatim (rel_speedup_pct, parent_mean_us, cand_mean_us, within_device_std_pct) plus the device number from the GPU_RUN_DEVICE= line on stderr.',
    '',
    taskContract(spec.exp_id),
    genomeFooter('Screen', `candidate_id="${spec.exp_id}", technique="${(spec.hypothesis || '').replace(/"/g, '')}"`),
  ].join('\n')
}

function confirmPrompt(spec) {
  const wt = `${P.worktrees}/${spec.exp_id}`
  return [
    `Tier-2 CONFIRM benchmark for experiment ${spec.exp_id} on the canonical locked-clock device (queued - may wait for the lock).`,
    '',
    `1. Run: WARPSPEED_EXP_ID=${spec.exp_id} ${C.confirm} ${wt}/${BINARY_PATH} --shape ${BENCH_SHAPE}`,
    '2. Relay latency_us_mean, latency_us_std, reps, clocks verbatim.',
    '',
    taskContract(spec.exp_id),
    genomeFooter('Confirm', `candidate_id="${spec.exp_id}"`),
  ].join('\n')
}

function profilePrompt(spec) {
  const wt = `${P.worktrees}/${spec.exp_id}`
  const parentCache = `${P.ncu_cache}/${spec.parent_commit}.json`
  const parentBin = `${P.builds}/${spec.parent_commit}/${BINARY_PATH}`
  return [
    `NCU PROFILE for experiment ${spec.exp_id} (serialized on the profiling device; curated sections only).`,
    '',
    `1. Parent cache: if ${parentCache} exists, READ it (parent_was_cached=true) - checkpoints are profiled once, ever. If missing, profile the parent binary first: WARPSPEED_EXP_ID=${spec.exp_id} ${C.ncu_profile} ${parentBin} ${parentCache} --shape ${BENCH_SHAPE}`,
    `2. Profile the candidate: WARPSPEED_EXP_ID=${spec.exp_id} ${C.ncu_profile} ${wt}/${BINARY_PATH} ${P.ncu_cache}/exp-${spec.exp_id}.json --shape ${BENCH_SHAPE}`,
    '3. Return cand = {fingerprint, key_metrics} and parent = {fingerprint, key_metrics}, both VERBATIM from the JSON files, and ncu_path = the candidate JSON path. Compute nothing yourself.',
    PROFILING_GUARD,
    '',
    taskContract(spec.exp_id),
    genomeFooter('Profile', `candidate_id="${spec.exp_id}"`),
  ].join('\n')
}

function analystPrompt(spec, screen, prof, implOut, sig) {
  const cand = (prof.cand || {}).key_metrics || {}
  const parent = (prof.parent || {}).key_metrics || {}
  const keys = Array.from(new Set([...Object.keys(parent), ...Object.keys(cand)])).sort()
  const deltas = keys.map(k => {
    const a = parent[k], b = cand[k]
    const d = (typeof a === 'number' && typeof b === 'number') ? (Math.round((b - a) * 100) / 100) : null
    return `  ${k}: parent=${fmt(a)} cand=${fmt(b)}${d != null ? ` delta=${d > 0 ? '+' : ''}${d}` : ''}`
  }).join('\n')
  return [
    `You are the WarpSpeed ANALYST for experiment ${spec.exp_id}. Diagnose WHY the measurement came out this way, strictly from the evidence below. No tools needed; do not invent numbers.`,
    '',
    `Hypothesis: ${spec.hypothesis}`,
    `Predicted: +${spec.predicted_gain_pct}% via: ${spec.predicted_mechanism}`,
    `Measured (screen, device-unbiased A/B): ${screen.rel_speedup_pct}% (noise ${screen.within_device_std_pct}%) -> ${sig ? 'SIGNIFICANT' : 'not significant'}`,
    `Diff summary (implementor): ${implOut.summary || fmt(implOut.self_reported_changes)}`,
    '',
    'NCU key-metric deltas (parent -> candidate):',
    deltas || '  (no metrics available)',
    '',
    'Answer:',
    '- bottleneck: the dominant limiter for the CANDIDATE (one phrase, e.g. "DRAM bandwidth", "MMA issue gaps", "barrier serialization").',
    '- evidence: which deltas support it.',
    '- hypothesis_verdict: confirmed | refuted | partial - did the PREDICTED MECHANISM actually move (regardless of the headline number)?',
    '- suggested_directions: up to 3 direction tags worth trying next from this state.',
    '- escalate_full_ncu: true only if the curated sections cannot distinguish between two plausible bottlenecks you name.',
    '- diagnosis: 2-4 sentences tying mechanism to measurement. This text is the post-mortem\'s primary evidence - be precise.',
    GROUNDING_INSTRUCTION,
  ].join('\n')
}

function finishPrompt(spec, payload) {
  return [
    `Record the final experiment row for ${spec.exp_id}.`,
    '',
    `1. Write the following JSON EXACTLY to ${P.logs}/${spec.exp_id}.finish.json:`,
    JSON.stringify(payload),
    `2. Run: ${C.wsdb} exp-finish --json @${P.logs}/${spec.exp_id}.finish.json`,
    '3. Return written=true and the status echoed by the CLI.',
    '',
    taskContract(spec.exp_id),
  ].join('\n')
}

async function finishExp(spec, status, extra, iterCount) {
  const payload = Object.assign({
    exp_id: spec.exp_id,
    status,
    review_iterations: iterCount || 0,
  }, extra || {})
  const r = await agentRetry(() => agent(finishPrompt(spec, payload), {
    phase: 'Record', label: `${spec.exp_id}:finish`, schema: FINISH_SCHEMA,
  }), { retries: 5, allowNull: true })
  const c = classifyResult(r)
  if (c.state !== 'grounded') log(`WARN: exp-finish not grounded for ${spec.exp_id} (recover-orphans will reconcile next round)`)
  return c
}

async function runCandidate(spec, snap) {
  const noise = noisePct(
    null,
    (snap.calibration || {}).cross_device_sigma_pct != null
      ? snap.calibration.cross_device_sigma_pct
      : cal.value.cross_device_sigma_pct
  )
  // ---- generate <-> review loop (control flow lives here, per spec) ----
  let impl = null
  let review = null
  let objections = []
  let iters = 0
  for (let iter = 1; iter <= MAX_REVIEW_ITERS; iter++) {
    iters = iter
    const implRes = classifyResult(await agentRetry(() => agent(implementorPrompt(spec, iter, objections), {
      phase: 'Generate', label: `${spec.exp_id}:impl${iter}`, schema: IMPL_SCHEMA,
    }), { retries: 5 }))
    if (implRes.state !== 'grounded') {
      await finishExp(spec, 'gen_failed', { failure_reason: `implementor not grounded: ${implRes.missing || implRes.error || 'unknown'}` }, iter)
      return { spec, status: 'gen_failed' }
    }
    impl = implRes.value
    if (impl.gave_up || !impl.commit) {
      await finishExp(spec, 'gen_failed', { failure_reason: impl.failure_reason || 'implementor gave up', commit_hash: impl.commit || null }, iter)
      return { spec, status: 'gen_failed' }
    }
    if (!impl.build_ok) {
      await finishExp(spec, 'compile_error', { failure_reason: impl.failure_reason || 'build failed', commit_hash: impl.commit }, iter)
      return { spec, status: 'compile_error' }
    }
    if (!impl.correctness_passed) {
      await finishExp(spec, 'incorrect', { failure_reason: impl.failure_reason || 'correctness failed in implementor loop', commit_hash: impl.commit }, iter)
      return { spec, status: 'incorrect' }
    }
    const revRes = classifyResult(await agentRetry(() => agent(reviewerPrompt(spec, impl, iter), {
      phase: 'Generate', label: `${spec.exp_id}:review${iter}`, schema: REVIEW_SCHEMA,
    }), { retries: 5 }))
    if (revRes.state !== 'grounded') {
      await finishExp(spec, 'review_rejected', { failure_reason: `review runner not grounded: ${revRes.missing || revRes.error || 'unknown'}`, commit_hash: impl.commit }, iter)
      return { spec, status: 'review_rejected' }
    }
    review = revRes.value
    // Disagreement protocol: implementor green + reviewer's independent run red
    // = nondeterminism (a race), full stop. Never retried to green.
    if (!review.independent_correctness) {
      await finishExp(spec, 'incorrect', {
        failure_reason: 'reviewer_correctness_divergence: implementor passed but independent re-run failed (treated as a race, not retried)',
        commit_hash: impl.commit, sanitizer_clean: review.sanitizer_clean === true ? 1 : (review.sanitizer_clean === false ? 0 : null),
      }, iter)
      return { spec, status: 'incorrect' }
    }
    if (review.accept) break
    objections = (review.objections && review.objections.length) ? review.objections : [review.objection || 'unspecified objection']
    if (iter === MAX_REVIEW_ITERS) {
      await finishExp(spec, 'review_rejected', {
        failure_reason: `review exhausted after ${MAX_REVIEW_ITERS} iterations: ${review.objection || objections.join('; ')}`,
        commit_hash: impl.commit, sanitizer_clean: review.sanitizer_clean === true ? 1 : (review.sanitizer_clean === false ? 0 : null),
      }, iter)
      return { spec, status: 'review_rejected' }
    }
  }

  // ---- screen ----
  const screenRes = classifyResult(await agentRetry(() => agent(screenPrompt(spec, impl), {
    phase: 'Screen', label: `${spec.exp_id}:screen`, schema: SCREEN_SCHEMA,
  }), { retries: 5 }))
  if (screenRes.state !== 'grounded') {
    await finishExp(spec, 'gen_failed', { failure_reason: `screen failed: ${screenRes.missing || screenRes.error}`, commit_hash: impl.commit }, iters)
    return { spec, status: 'gen_failed' }
  }
  const screen = screenRes.value
  const screenNoise = noisePct(screen.within_device_std_pct, noise)
  const sig = isSignificant(screen.rel_speedup_pct, screenNoise, CFG.MIN_GAIN_PCT)

  // ---- route ----
  const parentCkpt = frontierByCommit(snap, spec.parent_commit)
  const parentLat = parentCkpt ? parentCkpt.latency_us : null
  const best = bestLatency(snap)
  const projectedLat = (parentLat != null) ? parentLat / (1 + screen.rel_speedup_pct / 100) : null
  const newBestClaim = sig && (best == null || projectedLat == null || projectedLat < best)
  let confirm = null
  if (shouldConfirm(screen.rel_speedup_pct, newBestClaim, CFG.CONFIRM_MARGIN_PCT)) {
    const confRes = classifyResult(await agentRetry(() => agent(confirmPrompt(spec), {
      phase: 'Confirm', label: `${spec.exp_id}:confirm`, schema: CONFIRM_SCHEMA,
    }), { retries: 5 }))
    if (confRes.state === 'grounded') confirm = confRes.value
    else log(`WARN: confirm not grounded for ${spec.exp_id}; falling back to screen numbers`)
  }

  // ---- profile + diagnose (every screened candidate, incl. correct_slower) ----
  let prof = { cand: {}, parent: {}, ncu_path: null }
  const profRes = classifyResult(await agentRetry(() => agent(profilePrompt(spec), {
    phase: 'Profile', label: `${spec.exp_id}:ncu`, schema: PROFILE_SCHEMA,
  }), { retries: 5 }))
  if (profRes.state === 'grounded') prof = profRes.value
  else log(`WARN: profile not grounded for ${spec.exp_id}`)

  let analyst = null
  const anaRes = classifyResult(await agentRetry(() => agent(analystPrompt(spec, screen, prof, impl, sig), {
    phase: 'Profile', label: `${spec.exp_id}:analyst`, schema: ANALYST_SCHEMA,
  }), { retries: 5 }))
  if (anaRes.state === 'grounded') analyst = anaRes.value

  // ---- status + record ----
  const confirmedLat = confirm ? confirm.latency_us_mean : null
  const candLat = confirmedLat != null ? confirmedLat : screen.cand_mean_us
  const confirmedNewBest = sig && newBestClaim && (
    confirmedLat != null ? (best == null || confirmedLat < best) : true
  )
  const status = !sig ? 'correct_slower' : (confirmedNewBest ? 'new_best' : 'correct_faster')
  const achieved = screen.rel_speedup_pct
  const gap = predictionGap(spec.predicted_gain_pct, achieved)

  await finishExp(spec, status, {
    commit_hash: impl.commit,
    sanitizer_clean: review.sanitizer_clean === true ? 1 : (review.sanitizer_clean === false ? 0 : null),
    device_id: String(screen.device != null ? screen.device : ''),
    bench_tier: confirm ? 'confirm' : 'screen',
    latency_us_mean: candLat,
    latency_us_std: confirm ? confirm.latency_us_std : null,
    speedup_vs_parent: screen.rel_speedup_pct,
    achieved_gain_pct: achieved,
    prediction_gap_pct: gap,
    key_metrics: (prof.cand || {}).key_metrics || null,
    ncu_path: prof.ncu_path,
    diagnosis: analyst ? `${analyst.bottleneck} | verdict=${analyst.hypothesis_verdict} | ${analyst.diagnosis}` : null,
  }, iters)

  return {
    spec, status, impl, review, screen, confirm, prof, analyst,
    sig, achieved, gap, candLat, projectedLat,
    fingerprint: (prof.cand || {}).fingerprint || null,
    keyMetrics: (prof.cand || {}).key_metrics || null,
    noise: screenNoise,
    assumptions: Array.from(new Set([...(parentCkpt ? parentCkpt.assumptions : []), ...(spec.direction_tags || [])])).sort(),
  }
}

// =============================================================================
// Embedded candidate pipeline (inference-engine operators, e.g. llama.cpp .cuh)
// -----------------------------------------------------------------------------
// Same generate<->review loop as runCandidate, but the legacy standalone A/B screen
// is unreachable (the operator does not compile as a single TU), so eval is done
// IN THE PROJECT: embedded_inplace swaps the candidate onto the project operator
// file (backup -> apply -> build -> test -> bench -> ALWAYS restore), or
// embedded_dispatch registers it via __embeddedEvalPlan. This function is only ever
// called from the SERIAL for-loop above (never await parallel) because both modes
// share project state. Returns the same result shape so the Record barrier is unchanged.
// =============================================================================
async function runCandidateEmbedded(spec, snap) {
  const noise = noisePct(
    null,
    (snap.calibration || {}).cross_device_sigma_pct != null
      ? snap.calibration.cross_device_sigma_pct
      : cal.value.cross_device_sigma_pct
  )
  // ---- generate <-> review loop (identical contract to runCandidate) ----
  let impl = null
  let review = null
  let objections = []
  let iters = 0
  for (let iter = 1; iter <= MAX_REVIEW_ITERS; iter++) {
    iters = iter
    const implRes = classifyResult(await agentRetry(() => agent(implementorPrompt(spec, iter, objections), {
      phase: 'Generate', label: `${spec.exp_id}:impl${iter}`, schema: IMPL_SCHEMA,
    }), { retries: 5 }))
    if (implRes.state !== 'grounded') {
      await finishExp(spec, 'gen_failed', { failure_reason: `implementor not grounded: ${implRes.missing || implRes.error || 'unknown'}` }, iter)
      return { spec, status: 'gen_failed' }
    }
    impl = implRes.value
    if (impl.gave_up || !impl.commit) {
      await finishExp(spec, 'gen_failed', { failure_reason: impl.failure_reason || 'implementor gave up', commit_hash: impl.commit || null }, iter)
      return { spec, status: 'gen_failed' }
    }
    if (!impl.build_ok) {
      await finishExp(spec, 'compile_error', { failure_reason: impl.failure_reason || 'build failed', commit_hash: impl.commit }, iter)
      return { spec, status: 'compile_error' }
    }
    if (!impl.correctness_passed) {
      await finishExp(spec, 'incorrect', { failure_reason: impl.failure_reason || 'correctness failed in implementor loop', commit_hash: impl.commit }, iter)
      return { spec, status: 'incorrect' }
    }
    const revRes = classifyResult(await agentRetry(() => agent(reviewerPrompt(spec, impl, iter), {
      phase: 'Generate', label: `${spec.exp_id}:review${iter}`, schema: REVIEW_SCHEMA,
    }), { retries: 5 }))
    if (revRes.state !== 'grounded') {
      await finishExp(spec, 'review_rejected', { failure_reason: `review runner not grounded: ${revRes.missing || revRes.error || 'unknown'}`, commit_hash: impl.commit }, iter)
      return { spec, status: 'review_rejected' }
    }
    review = revRes.value
    if (!review.independent_correctness) {
      await finishExp(spec, 'incorrect', {
        failure_reason: 'reviewer_correctness_divergence: implementor passed but independent re-run failed (treated as a race, not retried)',
        commit_hash: impl.commit, sanitizer_clean: review.sanitizer_clean === true ? 1 : (review.sanitizer_clean === false ? 0 : null),
      }, iter)
      return { spec, status: 'incorrect' }
    }
    if (review.accept) break
    objections = (review.objections && review.objections.length) ? review.objections : [review.objection || 'unspecified objection']
    if (iter === MAX_REVIEW_ITERS) {
      await finishExp(spec, 'review_rejected', {
        failure_reason: `review exhausted after ${MAX_REVIEW_ITERS} iterations: ${review.objection || objections.join('; ')}`,
        commit_hash: impl.commit, sanitizer_clean: review.sanitizer_clean === true ? 1 : (review.sanitizer_clean === false ? 0 : null),
      }, iter)
      return { spec, status: 'review_rejected' }
    }
  }

  // ---- embedded eval (SERIAL; shared project state) ----
  // Candidate operator source lives in the experiment worktree at the editable path.
  const candFile = `${P.worktrees}/${spec.exp_id}/${PRIMARY_KERNEL_PATH}`
  const liveFile = `${P.project_dir}/${PRIMARY_KERNEL_PATH}`
  let embLatency = null, embMetrics = {}, embBclass = 'unknown', embCorrect = false
  if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
    const embResult = await agentRetry(() => agent(
      [
        `EMBEDDED-INPLACE EVAL (serial). Candidate: ${candFile} | project operator file: ${liveFile} | pristine backup: ${ORIGINAL_BACKUP}`,
        'Run IN ORDER (this candidate mutates the shared project operator file; ALWAYS restore at the end):',
        `1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${liveFile}`,
        `2. Apply candidate:  cp ${candFile} ${liveFile}`,
        `3. Build the project: ${BUILD_CMD}`,
        `4. Correctness (fixed invocation): ${C.correctness} ${P.project_dir}/${BINARY_PATH}`,
        `5. Benchmark latency: ${C.confirm} ${P.project_dir}/${BINARY_PATH} --shape ${BENCH_SHAPE}`,
        `6. ALWAYS restore pristine (even on failure): cp -a ${ORIGINAL_BACKUP} ${liveFile}`,
        PROFILING_GUARD,
        `Parse latency_us + heuristic_bclass (memory_bound|compute_bound|latency_bound). Return {latency_us, heuristic_bclass, compiled, correct, metrics:{latency_us}}.`,
        taskContract(spec.exp_id),
      ].join('\n'),
      { model: MODEL.profile, label: `${spec.exp_id}:embedded-inplace`, phase: 'Confirm', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    embLatency = embResult && embResult.latency_us != null ? Number(embResult.latency_us) : null
    embBclass = (embResult && embResult.heuristic_bclass) || 'unknown'
    embMetrics = (embResult && embResult.metrics) || { latency_us: embLatency }
    embCorrect = !!(embResult && embResult.correct)
  } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
    const variant = `warpspeed_${spec.exp_id}`.replace(/[^A-Za-z0-9_]/g, '_')
    const _plan = typeof __embeddedEvalPlan === 'function'
      ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: candFile, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: `${C.correctness} ${P.project_dir}/${BINARY_PATH}`, benchmarkCmd: `${C.confirm} ${P.project_dir}/${BINARY_PATH} --shape ${BENCH_SHAPE}` })
      : null
    if (_plan) {
      const embResult = await agentRetry(() => agent(
        [
          'EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:',
          `1. Register: ${_plan.register}`,
          `2. Build:    ${_plan.build}`,
          `3. Test:     ${_plan.test}`,
          `4. Benchmark:${_plan.benchmark}`,
          `5. Unregister: ${_plan.unregister}`,
          _plan.cleanupInvariant,
          PROFILING_GUARD,
          `Parse latency_us + heuristic_bclass. Return {latency_us, heuristic_bclass, compiled, correct, metrics:{latency_us}}.`,
          taskContract(spec.exp_id),
        ].join('\n'),
        { model: MODEL.profile, label: `${spec.exp_id}:embedded-dispatch`, phase: 'Confirm', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      embLatency = embResult && embResult.latency_us != null ? Number(embResult.latency_us) : null
      embBclass = (embResult && embResult.heuristic_bclass) || 'unknown'
      embMetrics = (embResult && embResult.metrics) || { latency_us: embLatency }
      embCorrect = !!(embResult && embResult.correct)
    }
  }

  // ---- status + record (embedded eval yields absolute latency, not an A/B screen) ----
  const parentCkpt = frontierByCommit(snap, spec.parent_commit)
  const parentLat = parentCkpt ? parentCkpt.latency_us : null
  const best = bestLatency(snap)
  const candLat = embLatency
  const relSpeedup = (parentLat != null && candLat != null && candLat > 0)
    ? (parentLat / candLat - 1) * 100 : 0
  const sig = isSignificant(relSpeedup, noise, CFG.MIN_GAIN_PCT)
  const newBestClaim = sig && (best == null || (candLat != null && candLat < best))
  const status = (candLat == null || !embCorrect)
    ? 'incorrect'
    : (!sig ? 'correct_slower' : (newBestClaim ? 'new_best' : 'correct_faster'))
  const achieved = relSpeedup
  const gap = predictionGap(spec.predicted_gain_pct, achieved)

  await finishExp(spec, status, {
    commit_hash: impl.commit,
    sanitizer_clean: review.sanitizer_clean === true ? 1 : (review.sanitizer_clean === false ? 0 : null),
    bench_tier: 'embedded',
    latency_us_mean: candLat,
    speedup_vs_parent: relSpeedup,
    achieved_gain_pct: achieved,
    prediction_gap_pct: gap,
    key_metrics: embMetrics,
    diagnosis: `embedded:${INTEGRATION_DECISION.method} | bclass=${embBclass}`,
  }, iters)

  return {
    spec, status, impl, review, screen: null, confirm: null, prof: { cand: { key_metrics: embMetrics } }, analyst: null,
    sig, achieved, gap, candLat, projectedLat: candLat,
    fingerprint: null,
    keyMetrics: embMetrics,
    heuristic_bclass: embBclass,
    noise,
    assumptions: Array.from(new Set([...(parentCkpt ? parentCkpt.assumptions : []), ...(spec.direction_tags || [])])).sort(),
  }
}

// =============================================================================
// Round-level helpers
// =============================================================================

function renderFrontier(snap) {
  return (snap.frontier || []).map(f =>
    `- ${f.commit.slice(0, 10)}${f.is_frontier ? ' [frontier]' : ''} lat=${fmt(f.latency_us)}us blocked=${f.blocked_count} headroom=${fmt(f.headroom_pct)}% hash=${f.strategy_set_hash || '-'} assumptions=[${(f.assumptions || []).join(',')}]`
  ).join('\n') || '- (empty)'
}

function plannerPrompt(snap, quotas, nSpecs) {
  const tags = Object.keys(CFG.wiki_index || {}).sort()
  const calib = (snap.prediction_calibration || []).map(c => `- ${c.tag}: mean prediction gap ${c.mean_gap_pct}% over n=${c.n}`).join('\n')
  const lessons = (snap.recent_lessons || []).map(l => `- [${l.id}|${l.type}|conf ${l.confidence}] ${l.claim}`).join('\n')
  const existingHashes = (snap.frontier || []).map(f => `${f.strategy_set_hash}=[${(f.assumptions || []).join(',')}]`).join('  ')
  return [
    `You are the WarpSpeed PLANNER. Produce exactly ${nSpecs} experiment specs for round ${snap.round}. You see only this compact summary - never kernel source.`,
    '',
    `# Allocation quotas for this round (budget quartile ${snap.budget.quartile})`,
    `- exploit (parent = the [frontier] checkpoint): ${quotas.exploit}`,
    `- explore (parents = OTHER live checkpoints): ${quotas.explore}`,
    `- wildcard (parent = the baseline ${snap.baseline_commit ? snap.baseline_commit.slice(0, 10) : ''}, orthogonal direction): ${quotas.wildcard}`,
    '',
    '# Live frontier (non-retired checkpoints)',
    renderFrontier(snap),
    '',
    '# Leaderboard (confirmed tier-2 latencies)',
    (snap.leaderboard || []).map((e, i) => `${i + 1}. ${e.commit.slice(0, 10)} ${fmt(e.latency_us)}us [${(e.assumptions || []).join(',')}]`).join('\n') || '(empty)',
    '',
    '# Budget',
    `- gpu-minutes used ${snap.budget.gpu_minutes_used}${snap.budget.gpu_minutes_total ? ` of ${snap.budget.gpu_minutes_total}` : ''}; tokens ${snap.budget.tokens_used}`,
    '',
    '# Prediction calibration (your past gaps; large positive = you over-promise on this tag)',
    calib || '- (no history yet)',
    '',
    '# Knowledge',
    snap.lessons_digest ? `digest: ${snap.lessons_digest}` : '',
    lessons || '- (no lessons yet)',
    OP_DESC ? `task context: ${OP_DESC}` : '',
    '',
    `# Direction tags with wiki backing (prefer these; inventing a new tag is allowed when justified)`,
    tags.join(', '),
    '',
    '# Rules',
    '1. Every spec MUST state predicted_gain_pct AND predicted_mechanism naming the bottleneck it attacks. Calibrate against your gap history.',
    `2. Soft dedup: existing strategy-set hashes are: ${existingHashes || '(none)'}. A spec whose parent-assumptions + your new tags equals an existing set needs interaction_rationale explaining the expected interaction - otherwise pick a different direction.`,
    '3. Hypotheses are one falsifiable sentence each; direction_tags from the tag list (1-3 tags).',
    '4. Respect the quotas exactly (types: exploit/explore/wildcard).',
    '5. Diversity: do not propose two specs with identical tag sets on the same parent.',
    GROUNDING_INSTRUCTION,
  ].filter(Boolean).join('\n')
}

function registerPrompt(specs, round) {
  return [
    `Register ${specs.length} experiments for round ${round}: create a git worktree per spec and open its DB row.`,
    '',
    'Specs (JSON):',
    JSON.stringify(specs),
    '',
    'For EACH spec, in order:',
    `1. git -C ${P.project_dir} worktree add -b <branch> ${P.worktrees}/<exp_id> <parent_commit>`,
    '   where <branch> = exp/r<round>/<exp_id>-<slug> (slug provided in the spec).',
    `2. Open the row: write the spec JSON (fields: exp_id, round, type, parent_commit, hypothesis, direction_tags, predicted_gain_pct, predicted_mechanism, worktree_path, branch) to ${P.logs}/<exp_id>.start.json and run: ${C.wsdb} exp-start --json @${P.logs}/<exp_id>.start.json`,
    specs.some(s => s.queued_id) ? `3. Mark consumed queued specs: ${C.wsdb} queued-consume --json '{"ids": [${specs.map(s => s.queued_id).filter(Boolean).join(', ')}]}'` : '',
    'If a worktree add fails (e.g. stale dir), remove it with git worktree remove --force + prune and retry ONCE; report in failed[] if it still fails.',
    'Return registered[] and failed[].',
    '',
    taskContract('register'),
  ].filter(Boolean).join('\n')
}

function snapshotPrompt(round) {
  return [
    `WarpSpeed round bookkeeping (start of round ${round === null ? '(next)' : round}).`,
    '',
    `1. Recover orphans: ${C.wsdb} recover-orphans`,
    `   For each returned orphan with a worktree_path under ${P.worktrees}: git -C ${P.project_dir} worktree remove --force <path> (ignore errors), then git -C ${P.project_dir} worktree prune. Count them as orphans_pruned.`,
    `2. Ingest GPU-minutes: ${C.wsdb} budget-ingest`,
    `3. Advance the round counter: ${C.wsdb} next-round`,
    `4. Snapshot: ${C.wsdb} round-snapshot`,
    '5. Return the snapshot JSON VERBATIM in the `snapshot` field (it includes round, frontier, leaderboard, lessons, budget, calibration, queued_specs, postmortem_due).',
    '',
    taskContract('snapshot'),
  ].join('\n')
}

function checkpointsPrompt(results, round, noise) {
  const sigCands = results.filter(r => r.sig && r.impl && r.fingerprint !== undefined)
  const candidates = sigCands.map(r => ({
    exp_id: r.spec.exp_id, commit: r.impl.commit, latency_us: r.candLat,
    fingerprint: r.fingerprint, strategy_set_hash: strategySetHash(r.assumptions),
  }))
  const upserts = sigCands.map(r => ({
    commit: r.impl.commit, parent_commit: r.spec.parent_commit, round,
    latency_us: r.candLat, assumptions: r.assumptions,
    strategy_set_hash: strategySetHash(r.assumptions),
    ncu_fingerprint: r.fingerprint, key_metrics: r.keyMetrics,
  }))
  return [
    `Round ${round} RECORD barrier: hard-dedup, checkpoint promotion, blocked counting.`,
    '',
    `1. Hard dedup - write this JSON to ${P.logs}/r${round}.dedup.json and run ${C.wsdb} dedup-check --json @${P.logs}/r${round}.dedup.json :`,
    JSON.stringify({ noise_pct: Math.max(noise, 0.1), candidates }),
    '   The CLI returns merge[] = candidates that duplicate an existing node (same fingerprint, latency within noise). Those experiments are deduped - skip their checkpoint upsert below.',
    `2. For each NON-merged entry below, write it to ${P.logs}/r${round}.ckpt.<n>.json and run ${C.wsdb} checkpoint-upsert --json @<file> :`,
    JSON.stringify(upserts),
    `3. Tag promoted commits for humans: git -C ${P.project_dir} tag -f ckpt/<first8ofcommit> <commit> (ignore failures).`,
    `4. Blocked counting: ${C.wsdb} blocked-mark --json '{"round": ${round}}'`,
    '5. Return merged[] (from step 1), new_checkpoints[] (commits you upserted), blocked[] and postmortem_due[] (from step 4).',
    '',
    taskContract('record'),
  ].join('\n')
}

function lessonPrompt(expId) {
  return [
    `You are the WarpSpeed LESSON EXTRACTOR for experiment ${expId}.`,
    '',
    `1. Fetch the full record: ${C.wsdb} exp-get --exp-id ${expId}`,
    '2. Decide if this experiment teaches anything DURABLE. The bar is HIGH:',
    '   - a falsifiable claim (one sentence) + mechanism (why, 1-2 sentences) + honest scope {arch, problem_shape, applies_when}',
    '   - technique_works needs a significant, confirmed gain; technique_fails needs a clean refutation (correct code, no gain, diagnosed why); env_fact for infrastructure truths; assumption_invalidated is reserved for post-mortems.',
    '   - fill predicted_vs_achieved as [[predicted_gain_pct, achieved_gain_pct]] - calibration depends on it.',
    '   - tags: the experiment direction_tags (plus technique tags you can defend).',
    '   If the record is just noise (insignificant delta, infra failure without insight), emit NOTHING and set skipped_reason.',
    `3. Append (0..2 lessons max): write a JSON ARRAY of lesson objects {type, claim, mechanism, scope, tags, evidence: ["${expId}"], predicted_vs_achieved, confidence} to ${P.logs}/${expId}.lessons.json and run: ${C.wsdb} lessons-append --json @${P.logs}/${expId}.lessons.json`,
    '4. Return appended_ids (empty array if you appended nothing).',
    '',
    taskContract(expId),
  ].join('\n')
}

function postmortemPrompt(commit) {
  return [
    `You are the WarpSpeed POST-MORTEM analyst. Checkpoint ${commit} has been blocked for K consecutive rounds: descendants stopped improving. Assign blame to the most likely poisoned BAKED-IN strategy.`,
    '',
    `1. Pull the ancestor trajectory: ${C.wsdb} trajectory --commit ${commit}`,
    '   Study: the chain of hypotheses, predicted-vs-achieved gaps per hop (large positive gaps mark where the mental model broke), NCU key-metric evolution, current headroom.',
    `2. Recent failed children for context: ${C.wsdb} exp-get --exp-id <id> for the last few experiments if needed (ids appear in the trajectory).`,
    '3. Output:',
    '   - suspect_strategy: the assumption TAG most likely making downstream work futile (e.g. a layout/persistence choice that eats the headroom others need).',
    '   - suspect_checkpoint: the commit where that tag was baked in (must be an ancestor in the trajectory with that tag newly added).',
    '   - suspect_parent: that checkpoint\'s parent commit (read it from the trajectory).',
    '   - mechanism: WHY it poisons descendants (1-2 sentences tied to metrics).',
    '   - ablation spec: hypothesis for ONE experiment that rebuilds the current best capability WITHOUT the suspect strategy, from suspect_parent. Provide ablation_hypothesis, ablation_tags (validated downstream tags minus the suspect), ablation_predicted_gain_pct (vs suspect_parent), ablation_predicted_mechanism.',
    '   The orchestrator will run the ablation BEFORE any rewind; be precise about what outcome would confirm vs refute the blame.',
    GROUNDING_INSTRUCTION,
  ].join('\n')
}

function rewindPrompt(pm, ab, round) {
  const lesson = {
    type: 'assumption_invalidated',
    claim: `Strategy '${pm.suspect_strategy}' baked at ${pm.suspect_checkpoint.slice(0, 10)} poisons descendants: ablation reached comparable performance without it.`,
    mechanism: pm.mechanism,
    scope: { arch: CFG.arch, applies_when: `descendants of strategy set containing '${pm.suspect_strategy}'` },
    tags: [pm.suspect_strategy],
    evidence: [ab.spec.exp_id],
    predicted_vs_achieved: [[ab.spec.predicted_gain_pct, ab.achieved != null ? ab.achieved : 0]],
    confidence: 0.7,
  }
  const replaySpec = {
    type: 'replay',
    hypothesis: `Re-apply validated downstream strategies [${(pm.ablation_tags || []).join(',')}] onto the rewound base (knowledge merge after invalidating '${pm.suspect_strategy}')`,
    direction_tags: pm.ablation_tags || [],
    predicted_gain_pct: pm.ablation_predicted_gain_pct || 5,
    predicted_mechanism: `Downstream strategies were validated on the retired subtree; re-deriving them without '${pm.suspect_strategy}' should recover their gains on the rewound base.`,
  }
  return [
    `Execute the REWIND for confirmed post-mortem on ${pm.suspect_checkpoint} (suspect strategy '${pm.suspect_strategy}').`,
    '',
    `1. Append the invalidation lesson: write this JSON to ${P.logs}/r${round}.pm.lesson.json and run ${C.wsdb} lessons-append --json @<that file> :`,
    JSON.stringify(lesson),
    `2. Rewind the frontier (DB pointer move - git history is untouched): ${C.wsdb} rewind --json '{"suspect_checkpoint": "${pm.suspect_checkpoint}", "suspect_tag": "${pm.suspect_strategy}", "lesson_id": "<id from step 1>"}'`,
    '   It returns rewound_to + retired[].',
    `3. Queue the replay (knowledge merge, never a textual git merge) - write this spec to ${P.logs}/r${round}.replay.json with parent_commit set to the rewound_to commit from step 2, and run ${C.wsdb} queue-spec --json '{"spec": <the spec>}' :`,
    JSON.stringify(replaySpec),
    '4. Return rewound_to, retired[], lesson_id, replay_queued=true.',
    '',
    taskContract('postmortem'),
  ].join('\n')
}

function refutePrompt(pm, ab, round) {
  const lesson = {
    type: 'technique_works',
    claim: `Post-mortem blame on '${pm.suspect_strategy}' REFUTED: ablation without it could not reach the subtree's performance - the strategy is genuinely load-bearing.`,
    mechanism: `Ablation ${ab.spec.exp_id} from ${pm.suspect_parent ? pm.suspect_parent.slice(0, 10) : 'suspect parent'} achieved ${fmt(ab.achieved)}% (status ${ab.status}), below the subtree's level.`,
    scope: { arch: CFG.arch, applies_when: 'when considering rewinding this subtree again' },
    tags: [pm.suspect_strategy],
    evidence: [ab.spec.exp_id],
    predicted_vs_achieved: [[pm.ablation_predicted_gain_pct || 0, ab.achieved != null ? ab.achieved : 0]],
    confidence: 0.6,
  }
  return [
    `Record the REFUTED post-mortem for ${pm.suspect_checkpoint} (subtree kept).`,
    '',
    `1. Write this lesson JSON to ${P.logs}/r${round}.pm.refuted.json and run ${C.wsdb} lessons-append --json @<that file> :`,
    JSON.stringify(lesson),
    '2. Return appended_ids.',
    '',
    taskContract('postmortem'),
  ].join('\n')
}

function compactPrompt(round) {
  return [
    `WarpSpeed lesson COMPACTION (round ${round}). BitLessons are append-only: you may only APPEND superseding entries, never rewrite history.`,
    '',
    `1. Read ${P.bitlessons} (and the index via ${C.wsdb} lessons-query --max 100).`,
    '2. Identify: (a) near-duplicate lessons -> append ONE merged lesson with supersedes set to the weaker id (one supersedes per append; chain if needed); (b) contradicted lessons -> append the correction with supersedes; (c) repeatedly-confirmed claims -> append a higher-confidence consolidation.',
    `3. Append via: ${C.wsdb} lessons-append --json @<file>  (array of lesson objects, each may carry a "supersedes" field).`,
    `4. Write a <=10-line digest of the CURRENT state of knowledge and store it: ${C.wsdb} digest-set --json '{"digest": "..."}'`,
    '5. Return appended, superseded counts and digest_set=true.',
    '',
    taskContract('compact'),
  ].join('\n')
}

function cleanupPrompt(expIds, round) {
  return [
    `Round ${round} worktree cleanup.`,
    '',
    `1. For each of: ${expIds.map(e => `${P.worktrees}/${e}`).join(' ')}`,
    `   run: git -C ${P.project_dir} worktree remove --force <path> (ignore individual failures), then once at the end: git -C ${P.project_dir} worktree prune`,
    `2. Remove review clones: rm -rf ${expIds.map(e => `${P.review}/${e}`).join(' ')}`,
    '3. Branches and tags are KEPT (failed code is post-mortem evidence).',
    '4. Return worktrees_removed and review_dirs_removed counts.',
    '',
    taskContract('maintain'),
  ].join('\n')
}

// =============================================================================
// Main search loop
// =============================================================================

const history = []
const pmAttempted = new Set()
let lastTokens = tokensUsed()
let stop = null
let lastSnap = null

for (let r = 1; r <= ITERATIONS; r++) {
  // ---- Plan ----
  phase('Plan'); await __genomeReport('Plan', WORKFLOW_NAME)
  const snapRes = classifyResult(await agentRetry(() => agent(snapshotPrompt(null), {
    phase: 'Plan', label: `r${r}:snapshot`, schema: SNAP_SCHEMA,
  }), { retries: 5 }))
  if (snapRes.state !== 'grounded') {
    stop = `snapshot failed: ${snapRes.missing || snapRes.error}`
    break
  }
  const snap = snapRes.value.snapshot
  lastSnap = snap
  const round = snap.round
  log(`\n=== Round ${round} (loop ${r}/${ITERATIONS}) === frontier=${(snap.frontier || []).length} best=${fmt(bestLatency(snap))}us budget_used=${snap.budget.gpu_minutes_used}min`)

  stop = stopReason(snap, CFG, pmAttempted)
  if (stop) { log(`Stop condition: ${stop}`); break }

  let specs
  if (SINGLE_SPEC) {
    specs = [Object.assign({ type: 'exploit', parent_commit: snap.frontier_commit }, SINGLE_SPEC)]
  } else {
    const queued = (snap.queued_specs || []).map(q => Object.assign({ queued_id: q.queued_id }, q.spec))
    const fresh = Math.max(0, PARALLEL_AGENTS - queued.length)
    let planned = []
    if (fresh > 0) {
      const quotas = allocationQuotas(fresh, CFG.ALLOCATION, snap.budget.quartile)
      const planRes = classifyResult(await agentRetry(() => agent(plannerPrompt(snap, quotas, fresh), {
        phase: 'Plan', label: `r${round}:plan`, schema: PLAN_SCHEMA,
      }), { retries: 5 }))
      if (planRes.state !== 'grounded') { stop = 'planner failed'; break }
      planned = (planRes.value.specs || []).slice(0, fresh)
    }
    specs = [...queued, ...planned]
  }
  specs = specs.map((s, i) => Object.assign({}, s, {
    exp_id: `r${round}e${i + 1}`,
    slug: slugify(s.hypothesis),
    branch: `exp/r${round}/r${round}e${i + 1}-${slugify(s.hypothesis)}`,
    round,
  }))
  if (!specs.length) { stop = 'planner produced no specs'; break }
  log(`Round ${round}: ${specs.length} specs - ${specs.map(s => `${s.exp_id}[${s.type}:${(s.direction_tags || []).join('+') || '?'}]`).join(' ')}`)

  const regRes = classifyResult(await agentRetry(() => agent(registerPrompt(specs.map(s => ({
    exp_id: s.exp_id, round, type: s.type, parent_commit: s.parent_commit,
    hypothesis: s.hypothesis, direction_tags: s.direction_tags,
    predicted_gain_pct: s.predicted_gain_pct, predicted_mechanism: s.predicted_mechanism,
    worktree_path: `${P.worktrees}/${s.exp_id}`, branch: s.branch, slug: s.slug,
    queued_id: s.queued_id || null,
  })), round), { phase: 'Plan', label: `r${round}:register`, schema: REG_SCHEMA }), { retries: 5 }))
  if (regRes.state !== 'grounded') { stop = 'register failed'; break }
  const registeredIds = new Set((regRes.value.registered || []).map(x => x.exp_id))
  const runnable = specs.filter(s => registeredIds.has(s.exp_id))

  // ---- Generate/Screen/Confirm/Profile per candidate, fan-out capped ----
  phase('Generate'); await __genomeReport('Generate', WORKFLOW_NAME)
  const results = []
  if (IS_EMBEDDED) {
    // EMBEDDED EVAL — SERIAL by construction (this for-loop, NOT await parallel):
    // embedded_inplace mutates the shared project operator file (PRIMARY_KERNEL_PATH)
    // and embedded_dispatch shares the single project build, so candidates cannot be
    // evaluated concurrently (the parallel-embedded-race bug-class). The standalone
    // branch below keeps the legacy parallel A/B-screen path byte-identical.
    for (const spec of runnable) {
      try {
        const r = await runCandidateEmbedded(spec, snap)
        if (r) results.push(r)
      } catch (e) {
        log(`embedded eval failed for ${spec.exp_id}: ${(e && e.message) || e}`)
      }
    }
  } else {
    for (const chunk of chunkArray(runnable, PARALLEL_AGENTS)) {
      const chunkResults = await parallel(chunk.map(s => () => runCandidate(s, snap)))
      results.push(...chunkResults.filter(Boolean))
    }
  }

  // ---- Record barrier ----
  phase('Record'); await __genomeReport('Record', WORKFLOW_NAME)
  const roundNoise = Math.max(
    (snap.calibration || {}).cross_device_sigma_pct || 0,
    ...results.map(x => x.noise || 0)
  )
  const ckptRes = classifyResult(await agentRetry(() => agent(checkpointsPrompt(results, round, roundNoise), {
    phase: 'Record', label: `r${round}:checkpoints`, schema: CKPT_SCHEMA,
  }), { retries: 5 }))
  const rec = ckptRes.state === 'grounded' ? ckptRes.value : { merged: [], postmortem_due: [] }
  const mergedIds = new Set((rec.merged || []).map(m => m.exp_id))

  const lessonable = results.filter(x =>
    x.spec && !mergedIds.has(x.spec.exp_id) &&
    x.status !== 'gen_failed' // pure orchestration failures teach nothing durable
  )
  for (const chunk of chunkArray(lessonable, PARALLEL_AGENTS)) {
    await parallel(chunk.map(x => () => agentRetry(() => agent(lessonPrompt(x.spec.exp_id), {
      phase: 'Record', label: `${x.spec.exp_id}:lesson`, schema: LESSON_SCHEMA,
    }), { retries: 5 })))
  }

  const roundSummary = {
    round,
    specs: specs.length,
    statuses: results.reduce((acc, x) => { acc[x.status] = (acc[x.status] || 0) + 1; return acc }, {}),
    merged: (rec.merged || []).length,
    new_checkpoints: (rec.new_checkpoints || []).length,
  }

  // ---- Postmortem (blame -> ablation -> rewind|refute) ----
  phase('Postmortem'); await __genomeReport('Postmortem', WORKFLOW_NAME)
  const due = (rec.postmortem_due || []).filter(c => !pmAttempted.has(c))
  for (const ckCommit of due) {
    pmAttempted.add(ckCommit)
    log(`Post-mortem on blocked checkpoint ${ckCommit.slice(0, 10)}`)
    const pmRes = classifyResult(await agentRetry(() => agent(postmortemPrompt(ckCommit), {
      phase: 'Postmortem', label: `r${round}:pm:${ckCommit.slice(0, 8)}`, schema: PM_SCHEMA,
    }), { retries: 5 }))
    if (pmRes.state !== 'grounded') continue
    const pm = pmRes.value
    if (!pm.suspect_parent) { log('post-mortem returned no suspect_parent; skipping ablation'); continue }

    const abSpec = {
      exp_id: `r${round}ab${pmAttempted.size}`,
      type: 'ablation',
      parent_commit: pm.suspect_parent,
      hypothesis: pm.ablation_hypothesis,
      direction_tags: pm.ablation_tags || [],
      predicted_gain_pct: pm.ablation_predicted_gain_pct || 0,
      predicted_mechanism: pm.ablation_predicted_mechanism || pm.mechanism,
      slug: slugify(pm.ablation_hypothesis),
      branch: `exp/r${round}/ablation-${slugify(pm.suspect_strategy)}`,
      round,
    }
    const abReg = classifyResult(await agentRetry(() => agent(registerPrompt([{
      exp_id: abSpec.exp_id, round, type: 'ablation', parent_commit: abSpec.parent_commit,
      hypothesis: abSpec.hypothesis, direction_tags: abSpec.direction_tags,
      predicted_gain_pct: abSpec.predicted_gain_pct, predicted_mechanism: abSpec.predicted_mechanism,
      worktree_path: `${P.worktrees}/${abSpec.exp_id}`, branch: abSpec.branch, slug: abSpec.slug,
    }], round), { phase: 'Postmortem', label: `r${round}:ablation:register`, schema: REG_SCHEMA }), { retries: 5 }))
    if (abReg.state !== 'grounded') continue
    const ab = await runCandidate(abSpec, snap)

    // Confirmation criterion: the ablation (built WITHOUT the suspect strategy)
    // reaches the current best within one MIN_GAIN margin => the suspect was
    // not load-bearing => blame confirmed.
    const best = bestLatency(snap)
    const abLat = ab.candLat != null ? ab.candLat : ab.projectedLat
    const confirmed = Boolean(
      (ab.status === 'correct_faster' || ab.status === 'new_best' || ab.status === 'correct_slower') &&
      abLat != null && best != null && abLat <= best * (1 + CFG.MIN_GAIN_PCT / 100)
    )
    log(`Ablation ${ab.spec.exp_id}: status=${ab.status} lat=${fmt(abLat)}us vs best=${fmt(best)}us -> blame ${confirmed ? 'CONFIRMED (rewinding)' : 'refuted (subtree kept)'}`)
    if (confirmed) {
      await agentRetry(() => agent(rewindPrompt(pm, ab, round), {
        phase: 'Postmortem', label: `r${round}:rewind`, schema: REWIND_SCHEMA,
      }), { retries: 5 })
      roundSummary.rewound = pm.suspect_checkpoint
    } else {
      await agentRetry(() => agent(refutePrompt(pm, ab, round), {
        phase: 'Postmortem', label: `r${round}:pm-refuted`, schema: LESSON_SCHEMA,
      }), { retries: 5 })
    }
  }

  // ---- Maintain ----
  phase('Maintain'); await __genomeReport('Maintain', WORKFLOW_NAME)
  const doneIds = results.map(x => x.spec.exp_id)
  if (doneIds.length) {
    await agentRetry(() => agent(cleanupPrompt(doneIds, round), {
      phase: 'Maintain', label: `r${round}:cleanup`, schema: CLEAN_SCHEMA,
    }), { retries: 5 })
  }
  if (CFG.COMPACT_EVERY && round % CFG.COMPACT_EVERY === 0) {
    await agentRetry(() => agent(compactPrompt(round), {
      phase: 'Maintain', label: `r${round}:compact`, schema: COMPACT_SCHEMA,
    }), { retries: 5 })
  }
  const nowTokens = tokensUsed()
  if (nowTokens != null && lastTokens != null && nowTokens > lastTokens) {
    await agentRetry(() => agent(
      `Record token usage: run ${C.wsdb} budget-add --json '{"tokens_used": ${nowTokens - lastTokens}}' and return written=true.\n\n${taskContract('maintain')}`,
      { phase: 'Maintain', label: `r${round}:tokens`, schema: FINISH_SCHEMA }
    ), { retries: 5 })
  }
  if (nowTokens != null) lastTokens = nowTokens

  history.push(roundSummary)
  log(`Round ${round} done: ${JSON.stringify(roundSummary.statuses)}${roundSummary.rewound ? ` REWOUND ${roundSummary.rewound.slice(0, 10)}` : ''}`)

  if (SINGLE_SPEC) { stop = 'single_spec_mode'; break }
}

if (!stop) stop = 'iteration_limit_reached'

// =============================================================================
// Phase: Report
// =============================================================================

phase('Report'); await __genomeReport('Report', WORKFLOW_NAME)
const reportRes = classifyResult(await agentRetry(() => agent(
  [
    'Produce the final WarpSpeed report.',
    '',
    `1. Run: ${C.wsdb} status --full   (and ${C.wsdb} round-snapshot for the structured view)`,
    `2. Write a human report to ${P.results}/report.md: leaderboard with assumption sets, baseline vs best (absolute + speedup), rounds run, per-status experiment counts, rewinds executed, the lessons digest, and the 10 highest-confidence lessons.`,
    EXP_DIR ? `3. Mirror the report for KerSor: cp ${P.results}/report.md ${EXP_DIR}/warpspeed-report.md` : '',
    `4. Return report_path, best_commit, best_latency_us, baseline_latency_us (the latency of the baseline checkpoint), lessons_total (count of non-superseded lessons).`,
    '',
    taskContract('report'),
    genomeFooter('Report', 'note="final summary"'),
  ].filter(Boolean).join('\n'),
  { phase: 'Report', label: 'report', schema: REPORT_SCHEMA }
), { retries: 5 }))
const rep = reportRes.state === 'grounded' ? reportRes.value : {}

const baselineLat = rep.baseline_latency_us != null ? rep.baseline_latency_us : (seed.value.latency_us || null)
const bestLat = rep.best_latency_us != null ? rep.best_latency_us : (lastSnap ? bestLatency(lastSnap) : null)

// embedded_inplace exit safety net: restore the project operator file to pristine
// before returning (no-op unless INTEGRATION_DECISION.method === 'embedded_inplace').
await __exitRestore()

return {
  ok: true,
  grounded: true,
  stop_reason: stop,
  rounds_run: history.length,
  baseline_latency_us: baselineLat,
  best_latency_us: bestLat,
  best_commit: rep.best_commit || (lastSnap ? lastSnap.frontier_commit : null),
  speedup: (baselineLat && bestLat) ? Math.round(baselineLat / bestLat * 1000) / 1000 : null,
  lessons_total: rep.lessons_total != null ? rep.lessons_total : null,
  report_path: rep.report_path || `${P.results}/report.md`,
  state_dir: P.state_dir,
  exp_dir: EXP_DIR || null,
  genome_dir: genomeDir(),
  history,
}
