export const meta = {
  name: 'kda-kernel-workflow',
  description: 'Kernel Design Agents workflow: evidence-driven kernel optimization with draft-plan-implement-validate-record cycles',
  whenToUse: 'When you need to optimize a CUDA kernel (or any performance-sensitive code) following the KDA methodology: define task contract, write plan draft, implement one candidate at a time, validate, record evidence, decide to promote/revise/reject.',
  phases: [
    { title: 'Inspect', detail: 'Read workspace, understand baseline, identify validation path' },
    { title: 'Plan', detail: 'Write docs/draft.md, convert to executable docs/plan.md' },
    { title: 'Implement', detail: 'Implement one candidate at a time from the plan' },
    { title: 'Validate', detail: 'Run validation command, measure target metric' },
    { title: 'Decide', detail: 'Record evidence, promote/revise/reject candidate' },
    { title: 'Report', detail: 'Write final optimization report and candidates.jsonl' },
  ],
  requiredSkills: [],
  optionalSkills: [
    'cuda-kernel-development',
    'humanize:gen-plan',
    'ncu-report-skill',
  ],
  externalResources: [
    {
      name: 'KernelWiki',
      url: 'https://github.com/mit-han-lab/KernelWiki/fork',
      use: 'Fork or download externally when the KDA workflow needs KernelWiki architecture knowledge; do not vendor it under KDA/skills.',
    },
  ],
  skill_binding_mode: 'local_skills_plus_external_resources',
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

const WORKFLOW_NAME = 'kda-kernel-workflow'


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
// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.


function normalizeSuitabilityValue(value) {
  const raw = String(value || '').trim().toLowerCase().replace(/_/g, '-')
  const aliases = {
    'c++': 'cpp',
    cxx: 'cpp',
    cplusplus: 'cpp',
    cute: 'cute-dsl',
    hip: 'rocm',
    'intel-xpu': 'xpu',
    optimize: 'kernel-optimization',
    optimization: 'kernel-optimization',
    generate: 'kernel-generation',
    generation: 'kernel-generation',
    explain: 'performance-explanation',
    explanation: 'performance-explanation',
  }
  return aliases[raw] || raw
}


function resolveBackendAxis() {
  const b = args.backend ? normalizeSuitabilityValue(args.backend) : null
  const l = args.language ? normalizeSuitabilityValue(args.language) : null
  if (b && l && b !== l) {
    throw new Error(`Conflicting args: backend="${args.backend}" vs language="${args.language}". Pass only one.`)
  }
  if (args.backend && !args.backend_dir) {
    throw new Error(`args.backend="${args.backend}" requires args.backend_dir; driver dispatch has no implicit-resolve path.`)
  }
  return b || l || null
}
const RESOLVED_BACKEND = resolveBackendAxis()
const USE_DRIVER = !!args.backend_dir


// =============================================================================
// KDA Kernel Workflow
// =============================================================================
//
// Faithfully implements the Kernel Design Agents loop from docs/agent-flow.md:
//   1. Define the task contract
//   2. Let the agent inspect the local workspace
//   3. Make the agent write docs/draft.md
//   4. Convert the draft into an executable plan
//   5. Implement the first candidate
//   6. Validate correctness
//   7. Measure the target metric when applicable
//   8. Record evidence and decide whether to keep, revise, or reject
//   9. Repeat until promotion criteria are met or blockers are explicit
//
// Usage:
//   Workflow({name: 'kda-kernel-workflow', args: {
//     kernel_path: '/path/to/kernel.cu',
//     task_name: 'quantized-gemm-q4_0',
//     objective: 'Optimize Q4_0 GEMM kernel for H100',
//     correctness_requirements: 'Output must match baseline within 1e-5 relative error',
//     performance_target: 'Achieve < 0.5ms on M=4096, N=4096, K=4096',
//     allowed_approaches: 'CUDA C++, shared memory tiling, warp-level primitives',
//     test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     promotion_criteria: 'Speedup >= 1.2x over baseline AND passes validation',
//     max_candidates: 5,
//   }})
//
// =============================================================================

// ---- Task Contract (from args) ----
let KERNEL_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const TASK_NAME = args.task_name || 'unnamed-task'
const OBJECTIVE = args.objective || 'Optimize the target kernel'
const CORRECTNESS = args.correctness_requirements || 'Must produce correct output'
const PERF_TARGET = args.performance_target || 'Improve over baseline'
const ALLOWED = args.allowed_approaches || 'Any approach within the language'
const VALIDATION_CMD = args.test_command
const EVAL_CMD = args.benchmark_command || VALIDATION_CMD
const PROMOTION = args.promotion_criteria || 'Passes validation and improves target metric'
const MAX_CANDIDATES = args.max_candidates || 5
const LANGUAGE = args.language || 'cuda'
const TARGET_GPU = args.target_gpu || 'unknown GPU'
const SEED_CANDIDATES = args.seed_candidates || 3
const EXP_DIR = args.exp_dir || '/tmp/kda_exp'
const KDA_INPUT_MODE_OPTIMIZE = 'optimize_existing'

// --- Backend driver wiring (P5c Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_INSPECT_LANG_TOKEN = 'CUDA'
const LEGACY_DRAFT_LANG_TOKEN = 'CUDA'
const LEGACY_PLAN_LANG_TOKEN = 'CUDA'
const LEGACY_IMPL_LANG_TOKEN = 'CUDA'
const LEGACY_VALIDATE_LANG_TOKEN = 'CUDA'
const LEGACY_SOURCE_EXT = '.cu'
const LEGACY_GENERATED_KERNEL_FILENAME = 'kernel.cu'
const LEGACY_FENCE_TOKEN = 'cuda'
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}

let DRIVER = null
let DRIVER_LANG_FENCE = 'cuda'
let DRIVER_IMPL_REQUIREMENTS = ''
let DRIVER_SOURCE_EXT = LEGACY_SOURCE_EXT
let DRIVER_BACKEND_ID = RESOLVED_BACKEND || ''

function langToken(legacy) {
  return USE_DRIVER ? DRIVER_LANG_FENCE : legacy
}
function fenceToken() {
  return USE_DRIVER ? DRIVER_LANG_FENCE : LEGACY_FENCE_TOKEN
}
// Hybrid kernel-path helper combining AccelOpt and KernelAgent modes:
//   - AccelOpt-style: caller passes args.kernel_path → use it verbatim
//   - KernelAgent-style: workspace-local generated file under EXP_DIR with
//     driver-aware source extension
// Used both for the user-supplied optimize_existing path and the
// generate_then_optimize path (where KERNEL_PATH is reassigned post-generation).
function kdaKernelPath(userKernelPath) {
  if (userKernelPath) return userKernelPath
  const ext = USE_DRIVER ? DRIVER_SOURCE_EXT : LEGACY_SOURCE_EXT
  return `${EXP_DIR}/generated/kernel${ext}`
}

if (!KERNEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

// ---- State ----
let baselineCode = ''
let baselineMetrics = {}
let currentBestCode = ''
let currentBestMetrics = {}
let currentBestId = 'baseline'
let candidates = [] // candidates.jsonl equivalent
let iteration = 0
let promotionMet = false
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null

// --- profiling-strategist: pick the analysis METHOD per backend×task×host, then
// honor it in the Validate profile block. The agent only classifies the task
// (fuzzy); the substrate stamps confidence by method (measured/inferred/hypothesized)
// -- not the agent. See _substrate/profiling/README.md. Falls back to
// native_profiler if undecided. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }

// ---- Helpers ----
function recordCandidate(id, parentId, status, code, metrics, reason) {
  candidates.push({
    id,
    parent_id: parentId,
    status, // 'promoted' | 'revised' | 'rejected'
    iteration,
    metrics: metrics || {},
    reason: reason || '',
  })
  log(`  Candidate ${id}: ${status}${reason ? ' — ' + reason : ''}`)
}

// =============================================================================
// Phase 1: Inspect — Read workspace, understand baseline
//
// KDA step 2: "Let the agent inspect the local workspace."
// KDA step: "Read the repository structure, existing implementation, tests,
//            and task documentation."
// =============================================================================
phase('Inspect')

if (USE_DRIVER) {
  DRIVER = await agentRetry(() => agent(
    `Load the backend driver at ${BACKEND_DIR} and return its manifest plus idioms verbatim.\n` +
    `1. Run exactly: \`cat ${driverPath('manifest.json')}\` and parse JSON.\n` +
    `2. Run exactly: \`cat ${driverPath('idioms.json')}\` and parse JSON.\n` +
    `Return {present, backend_id, source_ext, aux_ext, lang_fence, impl_requirements, methods}.`,
    { model: MODEL.mechanical, label: 'load-driver', phase: 'Inspect', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (!DRIVER || DRIVER.present === false) {
    throw new Error(`No backend driver present at ${BACKEND_DIR}. Provide a valid backend_dir or omit it for the legacy path.`)
  }
  if (RESOLVED_BACKEND && DRIVER.backend_id && normalizeSuitabilityValue(DRIVER.backend_id) !== RESOLVED_BACKEND) {
    throw new Error(`backend_dir manifest backend_id="${DRIVER.backend_id}" conflicts with args.backend/language="${RESOLVED_BACKEND}".`)
  }
  DRIVER_LANG_FENCE = DRIVER.lang_fence || DRIVER_LANG_FENCE
  DRIVER_IMPL_REQUIREMENTS = DRIVER.impl_requirements || ''
  DRIVER_SOURCE_EXT = DRIVER.source_ext || DRIVER_SOURCE_EXT
  DRIVER_BACKEND_ID = DRIVER.backend_id || DRIVER_BACKEND_ID
  log(`Driver loaded: ${DRIVER_BACKEND_ID} (fence=${DRIVER_LANG_FENCE})`)
}

if (INPUT_MODE === 'generate_then_optimize') {
  const generated = await agentRetry(() => agent(`No kernel_path was provided. Generate and verify an initial kernel before KDA inspects and optimizes the workspace.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- objective: ${OBJECTIVE}
- correctness_requirements: ${CORRECTNESS}
- language: ${LANGUAGE}
- target_gpu: ${TARGET_GPU}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- test_command: ${VALIDATION_CMD || '(not provided)'}
- benchmark_command: ${EVAL_CMD || '(not provided)'}

# Contract
${USE_DRIVER
  ? `Generate ${SEED_CANDIDATES} complete ${DRIVER_LANG_FENCE} candidates and write the best one to ${kdaKernelPath('')}. Run available commands using {kernel_path}/{result_path}. Return the best verified generated kernel path.`
  : `Generate ${SEED_CANDIDATES} complete candidates under ./generated or the task workspace. Run available commands using {kernel_path}/{result_path}. Return the best verified generated kernel path.`}`, {
    label: 'generate-initial-kernel',
    phase: 'Inspect',
    schema: {
      type: 'object',
      properties: {
        generated_kernel_path: { type: 'string' },
        initial_candidates: { type: 'array', items: { type: 'object' } },
        initial_generation_result: { type: 'object' },
      },
      required: ['generated_kernel_path', 'initial_candidates', 'initial_generation_result'],
    },
  }), { retries: 5 })
  initialCandidates = generated.initial_candidates || []
  initialGenerationResult = generated.initial_generation_result || { verified: false }
  generatedKernelPath = generated.generated_kernel_path || ''
  if (!generatedKernelPath) throw new Error('Generation mode did not produce generated_kernel_path')
  if ((VALIDATION_CMD || EVAL_CMD) && initialGenerationResult.verified === false) throw new Error('No generated seed passed correctness evidence')
  KERNEL_PATH = generatedKernelPath
}

const inspection = await agentRetry(() => agent(`You are in a task implementation workspace. Inspect the workspace and the target kernel.

# Task Contract
- Task name: ${TASK_NAME}
- Objective: ${OBJECTIVE}
- Correctness requirements: ${CORRECTNESS}
- Performance target: ${PERF_TARGET}
- Allowed approaches: ${ALLOWED}

# Target kernel: ${KERNEL_PATH}

# External Resources
- KernelWiki is an external repository, not a bundled KDA skill. If it is available, use it for kernel optimization patterns, GPU architecture details, and performance techniques relevant to this task. Download or fork it from https://github.com/mit-han-lab/KernelWiki/fork.

# Available Skills
${USE_DRIVER
  ? `- ${DRIVER_IMPL_REQUIREMENTS || `Use driver-supplied ${DRIVER_LANG_FENCE} guidance for hardware-aware kernel development.`}`
  : `- Use \`cuda-kernel-development\` skill for hardware-aware CUDA development guidance.`}

# Instructions
1. Read the kernel file and any surrounding code (headers, utils, tests).
2. Identify the current baseline behavior and how it is validated.
3. Understand the code structure: key functions, data flow, memory patterns.
4. Look for existing tests, benchmarks, or validation scripts in the workspace.
5. Check if there are docs/draft.md or docs/plan.md already.
6. Research relevant optimization techniques for this kernel type using available domain knowledge.

Return a structured analysis.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Inspect","ts":"<ts>","status":"done","technique":"workspace_inspection","note":"<baseline approach + validation path + key risks, one line>"}`, {
  label: 'inspect-workspace',
  phase: 'Inspect',
  schema: {
    type: 'object',
    properties: {
      kernel_code: { type: 'string' },
      key_functions: { type: 'array', items: { type: 'string' } },
      current_approach: { type: 'string' },
      baseline_behavior: { type: 'string' },
      validation_path: { type: 'string' },
      existing_tests: { type: 'string' },
      risks_and_unknowns: { type: 'array', items: { type: 'string' } },
    },
    required: ['kernel_code', 'key_functions', 'current_approach'],
  },
}), { retries: 5 })

baselineCode = expect(inspection, 'kernel_code', 'inspect-workspace')
currentBestCode = baselineCode

const _inspKeyFns = guard(inspection, 'key_functions', [])
log(`Inspected: ${_inspKeyFns.length} key functions, approach: ${guard(inspection, 'current_approach', 'unknown')}`)

// =============================================================================
// Phase 2: Plan — Write draft.md, convert to executable plan
//
// KDA step 3: "Make the agent write docs/draft.md."
// KDA step 4: "Convert the draft into an executable plan."
// "Do not start implementation until the draft exists."
// =============================================================================
phase('Plan')

const draftResult = await agentRetry(() => agent(`Write a plan draft to docs/draft.md for this kernel optimization task.

# Task Contract
- Task name: ${TASK_NAME}
- Objective: ${OBJECTIVE}
- Correctness requirements: ${CORRECTNESS}
- Performance target: ${PERF_TARGET}
- Allowed approaches: ${ALLOWED}
- Validation command: ${VALIDATION_CMD || '(to be determined)'}
- Evaluation command: ${EVAL_CMD || '(same as validation)'}
- Promotion criteria: ${PROMOTION}

# Workspace Inspection Results
- Key functions: ${inspection.key_functions.join(', ')}
- Current approach: ${inspection.current_approach}
- Baseline behavior: ${inspection.baseline_behavior || 'not yet measured'}
- Risks and unknowns: ${(inspection.risks_and_unknowns || []).join('; ') || 'none identified'}

# Current kernel (first 3000 chars):
\`\`\`
${baselineCode.substring(0, 3000)}
\`\`\`

# External Resources
- KernelWiki is an external repository, not a bundled KDA skill. If it is available, use it for architecture-specific optimization knowledge and research references. Download or fork it from https://github.com/mit-han-lab/KernelWiki/fork.

# Available Skills
- Use \`humanize:gen-plan\` skill pattern for structured plan generation.
${USE_DRIVER
  ? `- Consult driver-supplied profiling guidance for ${DRIVER_LANG_FENCE} performance evidence.`
  : `- Use \`ncu-report-skill\` if NCU profiling data exists in the workspace (check profile/ directory).`}

# Draft Requirements (from prompts/basic-flow.md)
The draft MUST include:
1. The current baseline and how it is validated.
2. The main risks and unknowns.
3. Candidate implementation directions ranked by expected value and risk.
4. The first concrete implementation steps.
5. The exact validation and evaluation commands to run.
6. The evidence required to promote, revise, or reject a candidate.

Write the draft to docs/draft.md. Then return the draft content.
${__attemptBlock()}${__experienceBlock()}
# Recent genome trajectory (read BEFORE drafting)
Run \`tail -20 ${EXP_DIR}/genome.jsonl 2>/dev/null\` to see prior attempts this session. Use it to: (a) avoid proposing candidate directions that already regressed in earlier rounds, (b) spot multi-round patterns the inspection summary may have missed. If the file is empty or missing, ignore this and proceed with the inputs above.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Plan","ts":"<ts>","status":"done","technique":"plan_draft","speedup":null,"note":"<top candidate directions ranked by value/risk, one line>"}`, {
  label: 'write-draft',
  phase: 'Plan',
  schema: {
    type: 'object',
    properties: {
      draft_content: { type: 'string' },
      candidate_directions: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            title: { type: 'string' },
            approach: { type: 'string' },
            expected_value: { type: 'string' },
            risk: { type: 'string' },
          },
          required: ['title', 'approach'],
        },
      },
    },
    required: ['draft_content', 'candidate_directions'],
  },
}), { retries: 5 })

const directions = draftResult.candidate_directions
log(`Draft written. ${directions.length} candidate directions identified.`)

// Convert draft to executable plan
const planResult = await agentRetry(() => agent(`Convert the draft into an executable plan at docs/plan.md.

# Draft Content:
${draftResult.draft_content}

# Candidate Directions:
${directions.map((d, i) => `${i + 1}. ${d.title}: ${d.approach} (value: ${d.expected_value}, risk: ${d.risk})`).join('\n')}

# Requirements for docs/plan.md:
1. Number each candidate explicitly (Candidate 1, 2, ...).
2. For each candidate, specify: what to change, which functions, what the expected effect is.
3. Order candidates by expected value / risk ratio (best first).
4. Each candidate should be implementable independently.
5. Include the exact validation and evaluation commands.
6. State what evidence is needed to promote, revise, or reject each candidate.

Write docs/plan.md and return the plan structure.`, {
  label: 'write-plan',
  phase: 'Plan',
  schema: {
    type: 'object',
    properties: {
      plan_content: { type: 'string' },
      candidates: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            id: { type: 'string' },
            title: { type: 'string' },
            changes: { type: 'string' },
            expected_effect: { type: 'string' },
            priority: { type: 'number' },
          },
          required: ['id', 'title', 'changes'],
        },
      },
    },
    required: ['plan_content', 'candidates'],
  },
}), { retries: 5 })

const planCandidates = planResult.candidates
log(`Plan written. ${planCandidates.length} candidates ordered by priority.`)

// =============================================================================
// Phase 3-5: Implement → Validate → Decide (one candidate at a time)
//
// KDA: "Implement one candidate at a time."
// KDA: "Run validation after each meaningful candidate."
// KDA: "Record candidate results, parent relationships, and evidence."
// KDA: "Repeat until the promotion criteria are met or the remaining
//        blockers are explicit."
// =============================================================================

// --- profiling-strategist resolve (once, before the candidate loop): classify
// the target kernel and let the substrate pick the analysis METHOD + stamp
// confidence. The agent only classifies (fuzzy op_class/size); the strategist
// owns method+confidence. Honored in the Validate profile block below. ---
if (USE_DRIVER) {
  const _pd = await agentRetry(() => agent(
    `Read ${KERNEL_PATH}; classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${BACKEND_DIR}/manifest.json --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}

// --- integration-strategist: route build/test mode (standalone vs embedded_*). ---
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const BUILD_CMD = args.build_command || ''
const BENCH_CMD = args.benchmark_command || ''
const REGISTER_SCRIPT = args.register_script || ''
let INTEGRATION_DECISION = { method: 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _profManifest = (USE_DRIVER && BACKEND_DIR) ? `${BACKEND_DIR}/manifest.json` : `${SUBSTRATE}/backends/cuda/manifest.json`
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `Read ${KERNEL_PATH}; classify can_compile_standalone (yes|no|uncertain). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${KERNEL_PATH}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
    `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return stdout JSON verbatim {method, build_fidelity, reversible, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: { type: 'object', additionalProperties: true } }), { retries: 5, allowNull: true })
  if (_integ && _integ.method) INTEGRATION_DECISION = _integ
}
log(`integration method = ${INTEGRATION_DECISION.method}`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error('integration-strategist returned derive_adapter — provide project_root + build/test commands')
}
const USE_DRIVER_STANDALONE = USE_DRIVER && INTEGRATION_DECISION.method === 'standalone'
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${KERNEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: { type: 'object', additionalProperties: true } }), { retries: 5 })
}
// A-O1 closure: native_profiler but no driver standalone (embedded/legacy) → perf_heuristic
if (PROFILING_DECISION.method === 'native_profiler' && !USE_DRIVER_STANDALONE) {
  log(`profiling: native_profiler but no standalone driver path -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'test-harness-perf', rationale: 'native_profiler but no standalone driver -> perf_heuristic' }
}

for (iteration = 0; iteration < Math.min(planCandidates.length, MAX_CANDIDATES) && !promotionMet; iteration++) {
  const candidate = planCandidates[iteration]
  const candidateId = `candidate-${iteration + 1}`

  log(`\n=== Candidate ${iteration + 1}/${planCandidates.length}: ${candidate.title} ===`)

  // ---- Phase 3: Implement ----
  phase('Implement')

  const impl = await agentRetry(() => agent(`Implement this optimization candidate as a complete, compilable kernel.

# Task Contract
- Objective: ${OBJECTIVE}
- Correctness: ${CORRECTNESS}
- Allowed approaches: ${ALLOWED}

# Current Best Implementation:
\`\`\`
${currentBestCode.substring(0, 4000)}
\`\`\`

# Candidate to Implement
- ID: ${candidateId}
- Title: ${candidate.title}
- Changes: ${candidate.changes}
- Expected effect: ${candidate.expected_effect || 'improvement over baseline'}

# External Resources
- KernelWiki is an external repository, not a bundled KDA skill. If it is available, use it for architecture-specific optimization techniques such as Hopper/Blackwell features and tensor core usage. Download or fork it from https://github.com/mit-han-lab/KernelWiki/fork.

# Available Skills
${USE_DRIVER
  ? `- ${DRIVER_IMPL_REQUIREMENTS || `Use driver-supplied ${DRIVER_LANG_FENCE} guidance for hardware-aware kernel patterns.`}`
  : `- Use \`cuda-kernel-development\` skill for hardware-aware CUDA patterns (shared memory tiling, warp primitives, occupancy tuning).`}

# Turn boundary (issue #17)
This turn must (a) write the candidate file, (b) hand off to the Validate phase —
do NOT loop compile+eval inside this turn. Cap your Bash calls; one implement step,
then return. NO HARNESS MANIPULATION: improve the kernel itself, never manipulate
the test harness / allocator / free pool.

# Requirements
1. Output a COMPLETE file — all includes, definitions, functions.
2. Keep function signatures unchanged unless the plan explicitly requires changes.
3. Apply ONLY the changes described in this candidate. Do not combine with other optimizations.
4. Must be functionally correct: ${CORRECTNESS}

Return the complete implementation code.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Implement","ts":"<ts>","status":"done","candidate_id":"${candidateId}","technique":"<the main optimization you applied for this candidate>","note":"<what changed vs the current best, one line>"}`, {
    label: `impl-${candidateId}`,
    phase: 'Implement',
    schema: {
      type: 'object',
      properties: {
        code: { type: 'string' },
        changes_made: { type: 'string' },
      },
      required: ['code'],
    },
  }), { retries: 5 })

  const candidateCode = expect(impl, 'code', `impl-${candidateId}`)

  // ---- Phase 4: Validate ----
  phase('Validate')

  const validation = await agentRetry(() => agent(`Validate this kernel candidate: run correctness and performance tests.

# Candidate: ${candidateId} — ${candidate.title}

# Validation command: ${VALIDATION_CMD || '(not provided — do static analysis only)'}
# Evaluation command: ${EVAL_CMD || '(same as validation)'}
# Kernel file to validate: ${KERNEL_PATH}

# Candidate code (first 3000 chars):
\`\`\`
${candidateCode.substring(0, 3000)}
\`\`\`

# Correctness requirements: ${CORRECTNESS}
# Performance target: ${PERF_TARGET}

# Instructions
1. First, write the candidate code to the kernel file at ${KERNEL_PATH}.
2. STATIC ANALYSIS — check for:
${USE_DRIVER
  ? `   - Race conditions (shared/scratch memory access, synchronization placement)
   - Out-of-bounds (index/mask computations)
   - Missing synchronization
   - Correct reductions (per ${DRIVER_LANG_FENCE} idioms)`
  : `   - Race conditions (shared memory access, __syncthreads placement)
   - Out-of-bounds (index computations)
   - Missing synchronization
   - Correct reductions (warp shuffle logic)`}
3. CORRECTNESS VALIDATION — If test_command is provided:
   - Run it: \`${VALIDATION_CMD || 'N/A'}\`
   - Report whether it passes or fails, and any error output.
   - If not provided, rely on static analysis only.
4. PERFORMANCE EVALUATION — If benchmark_command is provided:
   - Run it: \`${EVAL_CMD || 'N/A'}\`
   - Extract the measured latency/throughput from the output.
   - Compare against baseline: ${currentBestMetrics.latency_ms ? currentBestMetrics.latency_ms + 'ms' : 'not yet measured'}.
   - If not provided, give your best estimate based on the optimization applied.
${USE_DRIVER
  ? `5. If driver profiling evidence is available, consult the driver-supplied bottleneck guidance for ${DRIVER_LANG_FENCE}.`
  : `5. If NCU profiling data is available or can be generated, use the \`ncu-report-skill\` to analyze bottlenecks.`}

Return the validation results with MEASURED values when available, estimates only as fallback.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if correctness passed, else "error"; speedup is the measured speedup number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Validate","ts":"<ts>","status":"<done|error>","candidate_id":"${candidateId}","speedup":<number or null>,"technique":"<technique under test>","note":"<correct? measured latency vs baseline; or the failure reason>"}`, {
    label: `validate-${candidateId}`,
    phase: 'Validate',
    schema: {
      type: 'object',
      properties: {
        is_correct: { type: 'boolean' },
        correctness_issues: { type: 'array', items: { type: 'string' } },
        validation_ran: { type: 'boolean' },
        validation_output: { type: 'string' },
        measured_latency_ms: { type: 'number' },
        estimated_latency_ms: { type: 'number' },
        estimated_speedup: { type: 'number' },
        addresses_goal: { type: 'boolean' },
        validation_notes: { type: 'string' },
      },
      required: ['is_correct', 'estimated_speedup', 'validation_ran'],
    },
  }), { retries: 5 })

  if (USE_DRIVER_STANDALONE) {
    const kPath = kdaKernelPath(KERNEL_PATH)
    const buildOut = `${EXP_DIR}/${candidateId}.artifact`
    const profOut = `${EXP_DIR}/${candidateId}.prof.native`
    await agentRetry(() => agent(
      `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
      `Return its stdout JSON verbatim.`,
      { model: MODEL.mechanical, label: `driver-build-${candidateId}`, phase: 'Validate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    const runOut = await agentRetry(() => agent(
      `${driverSh('run.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${buildOut}.run.json`)}\n` +
      `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
      { model: MODEL.profile, label: `driver-run-${candidateId}`, phase: 'Validate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    let evidenceOut = null
    let diagOut = null
    if (PROFILING_DECISION.method === 'native_profiler') {
      await agentRetry(() => agent(
        `${driverSh('profile.sh', `--artifact ${buildOut} --problem ${PROBLEM_PATH} --out ${buildOut}.run.json --out ${profOut}`)}\n` +
        `Return {ok, native_path}.`,
        { model: MODEL.profile, label: `driver-profile-${candidateId}`, phase: 'Validate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      evidenceOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
        `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
        { model: MODEL.mechanical, label: `driver-to-evidence-${candidateId}`, phase: 'Validate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      diagOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
        `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
        { model: MODEL.mechanical, label: `driver-diagnose-${candidateId}`, phase: 'Validate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    } else {
      // profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}');
      // do NOT run profile.sh/${PROFILING_DECISION.normalizer ? 'a native profiler' : 'a native profiler'}. The run.sh
      // throughput above is the measurement; when method='perf_heuristic', normalize it into canonical metrics via the
      // strategist normalizer, tagging bottlenecks evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.
      let heuristicMetrics = {}
      if (PROFILING_DECISION.method === 'perf_heuristic') {
        evidenceOut = await agentRetry(() => agent(
          `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${PROFILING_DECISION.normalizer || 'perf_to_evidence.py'} --baseline ${EXP_DIR}/${candidateId}.result.json --run-json '${JSON.stringify((runOut && { latency_ms: runOut.latency_ms }) || {})}'\`.\n` +
          `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}. Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.`,
          { model: MODEL.mechanical, label: `driver-heuristic-evidence-${candidateId}`, phase: 'Validate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        heuristicMetrics = (evidenceOut && evidenceOut.metrics) || {}
      }
      diagOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify(heuristicMetrics)}'\`.\n` +
        `Return stdout JSON verbatim {bottleneck_class, evidence}. Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.`,
        { model: MODEL.mechanical, label: `driver-diagnose-${candidateId}`, phase: 'Validate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    }
    await agentRetry(() => agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${EXP_DIR}/${candidateId}.result.json\`.\n` +
      `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
      { model: MODEL.mechanical, label: `driver-anti-cheat-${candidateId}`, phase: 'Validate', schema: JSON_PASSTHROUGH }), { retries: 5 })
    validation.driver_envelope = {
      latency_ms: Number((runOut && runOut.latency_ms) || 0),
      metrics: (evidenceOut && evidenceOut.metrics) || {},
      bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
      backend_id: DRIVER_BACKEND_ID,
    }
  } else if (IS_EMBEDDED) {
    // --- Embedded eval (serial — KDA's candidate loop is sequential) ---
    const kPath = kdaKernelPath(KERNEL_PATH)
    const variant = `kda_${iteration + 1}`.replace(/[^A-Za-z0-9_]/g, '_')
    let embLatency = 0, embMetrics = {}, embBclass = 'unknown'
    if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
      const embResult = await agentRetry(() => agent(
        `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project kernel: ${KERNEL_PATH} | pristine: ${ORIGINAL_BACKUP}\n` +
        `Run IN ORDER:\n1. cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n2. cp ${kPath} ${KERNEL_PATH}\n` +
        `3. Build: ${BUILD_CMD}\n4. Test: ${args.test_command || TEST_CMD || BENCH_CMD}\n5. Bench: ${BENCH_CMD || args.test_command || TEST_CMD}\n` +
        `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${KERNEL_PATH}\n` +
        `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
        { model: MODEL.mechanical, label: `embedded-inplace-${candidateId}`, phase: 'Validate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      embLatency = Number(embResult?.latency_ms || 0); embBclass = embResult?.heuristic_bclass || 'unknown'; embMetrics = embResult?.metrics || { latency_ms: embLatency }
    } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
      const _plan = typeof __embeddedEvalPlan === 'function'
        ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: args.test_command || TEST_CMD || BENCH_CMD, benchmarkCmd: BENCH_CMD || args.test_command || TEST_CMD })
        : null
      if (_plan) {
        const embResult = await agentRetry(() => agent(
          `EMBEDDED-DISPATCH EVAL (serial).\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Bench: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
          `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-dispatch-${candidateId}`, phase: 'Validate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        embLatency = Number(embResult?.latency_ms || 0); embBclass = embResult?.heuristic_bclass || 'unknown'; embMetrics = embResult?.metrics || { latency_ms: embLatency }
      }
    }
    validation.driver_envelope = { latency_ms: embLatency, metrics: embMetrics, bottleneck_class: embBclass, backend_id: 'embedded' }
  }

  // ---- Phase 5: Decide ----
  phase('Decide')

  const candidateMetrics = {
    latency_ms: validation.measured_latency_ms || validation.estimated_latency_ms,
    speedup: validation.estimated_speedup,
    addresses_goal: validation.addresses_goal,
    validation_ran: validation.validation_ran,
  }

  if (!validation.is_correct) {
    // Reject: failed correctness
    recordCandidate(candidateId, currentBestId, 'rejected', candidateCode, candidateMetrics,
      `Failed correctness: ${(validation.correctness_issues || []).join('; ')}`)

  } else if (validation.estimated_speedup > 1.0 && validation.addresses_goal) {
    // Promote: correct and improves
    recordCandidate(candidateId, currentBestId, 'promoted', candidateCode, candidateMetrics,
      `Speedup ${validation.estimated_speedup.toFixed(2)}x, addresses goal`)
    currentBestCode = candidateCode
    currentBestMetrics = candidateMetrics
    currentBestId = candidateId

    // Check promotion criteria
    const meetsPromotion = validation.estimated_speedup >= 1.2 // simplified check
    if (meetsPromotion) {
      log(`  Promotion criteria met. Best candidate: ${candidateId}`)
      promotionMet = true
    }

  } else if (validation.is_correct && validation.estimated_speedup <= 1.0) {
    // Reject: correct but no improvement
    recordCandidate(candidateId, currentBestId, 'rejected', candidateCode, candidateMetrics,
      `No improvement: speedup ${validation.estimated_speedup.toFixed(2)}x`)

  } else {
    // Revise: correct, addresses goal, but marginal
    recordCandidate(candidateId, currentBestId, 'rejected', candidateCode, candidateMetrics,
      `Marginal improvement, ${validation.validation_notes || 'needs revision'}`)
  }
}

// =============================================================================
// Final Report
//
// KDA: "Record candidate results, parent relationships, and evidence."
// KDA: "A future reader should be able to reconstruct what changed, what was
//        measured, and why a candidate was promoted."
// =============================================================================
phase('Report')

const report = await agentRetry(() => agent(`Write the final optimization report.

# Task: ${TASK_NAME}
# Objective: ${OBJECTIVE}
# Promotion criteria: ${PROMOTION}

# Candidates tried: ${candidates.length}
${candidates.map(c => `- ${c.id} (parent: ${c.parent_id}): ${c.status} — ${c.reason}`).join('\n')}

# Final best metrics: ${JSON.stringify(currentBestMetrics)}
# Promotion criteria met: ${promotionMet}

# Write a report that includes:
1. What was the baseline and how it was validated.
2. What candidates were tried, in what order.
3. For each candidate: what changed, what was measured, why it was promoted or rejected.
4. Which candidate was promoted and why it meets the promotion criteria.
5. What remaining blockers or future work exist.

Also write the candidates list to candidates.jsonl (one JSON object per line).

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the final outcome (status="done" if a candidate was promoted, else "error"; speedup is the promoted candidate's measured speedup number, or null if none promoted):
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"<done|error>","technique":"final_report","speedup":<number or null>,"note":"<promoted candidate id + why it met promotion criteria, or remaining blockers, one line>"}`, {
  label: 'final-report',
  phase: 'Report',
}), { retries: 5 })

// embedded_inplace exit safety net
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${KERNEL_PATH}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  baseline_id: args.fair_baseline_id || null,  // #32: echo the frozen fair baseline KerSor handed us (contract.env::baseline_id via dispatch-args.json); null when undeclared -> check-acceptance-gate.sh Check 2c skips (back-compat).
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  task_name: TASK_NAME,
  candidates_tried: candidates.length,
  promoted: candidates.filter(c => c.status === 'promoted').length,
  rejected: candidates.filter(c => c.status === 'rejected').length,
  final_best_metrics: currentBestMetrics,
  candidates: candidates,
}
