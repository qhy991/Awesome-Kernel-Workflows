export const meta = {
  name: 'cuda-agent-kernel-optimization',
  description: 'Skill-integrated multi-turn CUDA kernel optimization with profiling-driven iterative refinement (CUDA Agent methodology)',
  whenToUse: 'When optimizing CUDA kernels from PyTorch operator specifications through iterative code generation, compilation, correctness testing, and profiling feedback. Follows the CUDA Agent skill-based workflow: profile baseline → identify bottlenecks → implement kernel + bindings → compile → verify correctness → measure speedup → refine until target met.',
  phases: [
    { title: 'Setup', detail: 'Read PyTorch model, profile baseline (eager + compile), establish workspace' },
    { title: 'Profile', detail: 'Analyze native PyTorch performance, identify bottlenecks and optimization opportunities' },
    { title: 'Implement', detail: 'Generate CUDA kernel, bindings, and model_new.py with custom operators' },
    { title: 'Verify', detail: 'Compile kernel, run correctness tests against reference, measure performance' },
    { title: 'Refine', detail: 'Iteratively fix errors and optimize based on compilation/runtime/profiling feedback' },
    { title: 'Report', detail: 'Final performance comparison and optimization summary' },
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

const WORKFLOW_NAME = 'cuda-agent-kernel-optimization'

// --- shared profiling-strategist plumbing (CUDA substrate manifest; the existing
// profile/implement/refine prompts below honor the strategist's decision). The
// agent only CLASSIFIES the task (fuzzy op_class/size); the substrate
// DETERMINISTICALLY picks the method and STAMPS confidence by method
// (measured/inferred/hypothesized) -- the model must NOT assign confidence
// itself. See _substrate/profiling/README.md. Defaults to native_profiler so
// happy-path ncu behavior is unchanged if the decision is ignored. ---
const SUBSTRATE = args.substrate_dir || '_substrate'
const PY = args.substrate_command_prefix || ''
const BACKEND_MANIFEST = args.backend_manifest || `${SUBSTRATE}/backends/cuda/manifest.json`
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }
function substrateInstruction(script, cliArgs) {
  const p = `${SUBSTRATE}/${script}`
  return PY ? `Run exactly: \`${PY} ${p} ${cliArgs}\`.`
            : `No substrate_command_prefix for ${p} ${cliArgs}; do not invent an interpreter.`
}
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured' }


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
// This workflow does NOT use the generic entry scribe from
// scripts/patch-genome-report.js (__genomeReport). Instead each phase's doer
// agent appends a rich, result-bearing line to <exp_dir>/genome.jsonl as its
// FINAL action — written AFTER the work, so it carries real outcomes
// (technique / speedup / candidate_id / status) and emits once per loop
// iteration. See _meta/genome-trajectory-schema.md. The "__genomeReport"
// mention here is a sentinel so patch-genome-report.js treats this file as
// already handled and does not re-inject the entry scribe.

// --- BEGIN embedded-eval substrate (auto-inlined by scripts/patch-embedded-eval.js) ---
const EMBEDDING_CONTRACT = [
  'EMBEDDED-DISPATCH CONTRACT (this kernel is NOT standalone):',
  '',
  'You are authoring a kernel that lives INSIDE a larger project and is wired into',
  'its dispatch table. It cannot be compiled on its own. Therefore:',
  '',
  '1. Emit a COMPLETE source file (e.g. a .cuh) that matches the reference',
  '   dispatch signature exactly -- same entry-point shape, template params, and',
  '   launch-bounds conventions as the reference file. Do NOT add a main(), a',
  '   standalone harness, or top-level test code.',
  '2. Use ONLY symbols/headers the project already provides (project headers,',
  '   template instantiations, dispatch macros). Do not invent include paths.',
  '3. Do NOT register, build, or benchmark the variant yourself, and do NOT name',
  '   any symbol with the variant suffix -- the workflow + adapter handle wiring.',
  '4. Return ONLY the file contents plus a short rationale citing the concrete',
  '   design choice (tile shape, register budget, pipelining, GQA packing, etc.).',
].join('\n')

// Build the ordered evaluation commands for one candidate against a
// contract-conforming adapter. All fields are plain strings the caller already
// resolved from `args`. `params`/`unregParams` are opaque pass-through strings
// (e.g. "--dkq 256 --dv 256 --cmake-build-dir /p/build") that the substrate does
// not parse -- they belong to the project's adapter.
function __embeddedEvalPlan(ctx) {
  const adapter = ctx.adapter                       // e.g. 'python "/abs/llamacpp_register_variant.py"'
  const variant = ctx.variant                       // unique variant name for this candidate
  const source = ctx.source                         // path to the candidate source file on disk
  const root = ctx.projectRoot                       // --project-root
  const params = ctx.params || ''                    // opaque register params pass-through
  const unregParams = ctx.unregParams || ''          // opaque unregister params pass-through
  const q = (s) => `"${s}"`
  const reg = `${adapter} register --variant ${variant} --source ${q(source)} --project-root ${q(root)}${params ? ' ' + params : ''}`.trim()
  const unreg = `${adapter} unregister --variant ${variant} --project-root ${q(root)}${unregParams ? ' ' + unregParams : ''}`.trim()
  const list = `${adapter} list --project-root ${q(root)}`
  return {
    register: reg,
    list,
    // Project-native build/test/benchmark, run VERBATIM with the variant's env
    // gate set so the project binary dispatches to this candidate.
    build: ctx.buildCmd ? `KERSOR_VARIANT=${variant} ${ctx.buildCmd}` : '',
    test: ctx.testCmd ? `KERSOR_VARIANT=${variant} ${ctx.testCmd}` : '',
    benchmark: ctx.benchmarkCmd ? `KERSOR_VARIANT=${variant} ${ctx.benchmarkCmd}` : '',
    unregister: unreg,
    // Human-orderable sequence + the non-negotiable cleanup invariant.
    order: ['register', 'list', 'build', 'test', 'benchmark', 'unregister'],
    cleanupInvariant: `On ANY failure or non-improvement, run the unregister command and confirm via list that ${variant} is gone, leaving the project byte-exact pristine.`,
  }
}
// --- END embedded-eval substrate ---

const WORKFLOW_SUITABILITY = {
  supported_languages: ['cuda', 'metal'],
  supported_problem_types: ['cuda-kernel-generation', 'cuda-kernel-optimization', 'gpu-kernel-optimization'],
  problem_types: ['PyTorch model/operator to custom CUDA/Metal ops', 'GPU inference-time profile/implement/verify/refine loop'],
  reason: 'CUDA Agent in this repo targets CUDA custom operators and CUDA profiling/evaluation feedback.',
}

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

function supportsSuitabilityValue(supported, requested) {
  return supported.includes(requested) || supported.some(value => value.endsWith(`-${requested}`))
}

function assertWorkflowSuitability() {
  const requestedLanguage = normalizeSuitabilityValue(args.language)
  if (requestedLanguage && requestedLanguage !== 'auto') {
    const supported = WORKFLOW_SUITABILITY.supported_languages.map(normalizeSuitabilityValue)
    if (!supported.includes(requestedLanguage)) {
      throw new Error(
        `${WORKFLOW_NAME} is not suitable for language="${args.language}". ` +
        `Supported languages/backends: ${WORKFLOW_SUITABILITY.supported_languages.join(', ')}. ` +
        `Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }
  }

  const requestedProblemType = normalizeSuitabilityValue(args.problem_type)
  if (requestedProblemType && requestedProblemType !== 'auto') {
    const supportedProblemTypes = (WORKFLOW_SUITABILITY.supported_problem_types || []).map(normalizeSuitabilityValue)
    if (supportedProblemTypes.length && !supportsSuitabilityValue(supportedProblemTypes, requestedProblemType)) {
      throw new Error(
        `${WORKFLOW_NAME} is not suitable for problem_type="${args.problem_type}". ` +
        `Supported problem types: ${WORKFLOW_SUITABILITY.supported_problem_types.join(', ')}. ` +
        `Typical use cases: ${WORKFLOW_SUITABILITY.problem_types.join('; ')}. ` +
        `Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }
  }
}

assertWorkflowSuitability()

// =============================================================================
// CUDA Agent: Skill-Integrated Multi-Turn Kernel Optimization
// =============================================================================
//
// Source: "CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation"
//         Dai, Wu, Yu, Gao, Li, Jiang, Lou, Song, Yu, Chen, Ma, Zhang, Liu, Wang, Liu, Zhou
//         ByteDance Seed / Tsinghua AIR, arXiv:2602.24286, 2026
//         https://cuda-agent.github.io/
//
// The CUDA Agent paper introduces a training system (agentic RL with PPO), but its
// inference-time agent loop (Section 3.2, Figure 2) defines a reusable workflow:
// adaptation_scope: inference_time_adaptation — this workflow does not reproduce
// dataset construction, PPO training, or model-weight updates from the full paper.
//
// SKILL.md (CUDA Coding Skill):
//   1. Analyze native PyTorch performance using profile.py
//      → identify bottlenecks (excessive kernel launches, suboptimal memory access)
//   2. Implement custom CUDA operators in model_new.py with kernel source + bindings
//      → target performance-critical operators identified in step 1
//   3. Compile and evaluate in GPU sandbox
//      → iteratively refine until correctness AND performance requirements met
//   4. Repeat from step 2 until ≥5% speedup over torch.compile achieved
//
// Workspace structure:
//   kernels/kernel.cu        — CUDA kernel source
//   kernels/kernel_binding.cpp — pybind11 bindings
//   model_new.py             — PyTorch model using custom CUDA ops
//   model.py                 — Original PyTorch model (reference)
//   verify.py                — Correctness verification script
//   profile.py               — Performance profiling script
//
// Reward signal (robust reward schedule):
//   r = -1 if correctness fails
//   r = 3  if faster than BOTH eager AND compile baselines (>5%)
//   r = 2  if faster than eager baseline only (>5%)
//   r = 1  otherwise (correct but not faster)
//
// Usage:
//   Workflow({name: 'cuda-agent-kernel-optimization', args: {
//     kernel_path: '/path/to/model.py',
//     op_description: 'Fused SwiGLU + Linear projection',
//     test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
//     profile_command: '<user-provided profiling command with {kernel_path}/{result_path}>',
//     compile_command: '<user-provided compile command with {kernel_path}/{result_path}>',
//     target_speedup: 1.05,
//     max_turns: 20,
//     exp_dir: '/tmp/cuda_agent_exp',
//   }})
//
// Embedded-dispatch mode (kernel is wired into a larger project, not standalone):
//   Workflow({name: 'cuda-agent-kernel-optimization', args: {
//     integration_pattern: 'embedded',           // 'standalone' (default) | 'embedded'
//     register_script: '/abs/scripts/llamacpp_register_variant.py', // adapter (3 verbs)
//     project_root: '/abs/project',              // (alias: ggml_root) --project-root
//     reference_cuh: '/abs/project/.../fattn.cuh', // (alias: reference_file) signature to match
//     register_params: '--dkq 256 --dv 256',     // opaque pass-through to the adapter
//     build_command: '<project build cmd>',       // defaults to compile_command
//     test_command: '<project correctness cmd>',
//     profile_command: '<project benchmark cmd>',
//   }})
//   In embedded mode register_script, project_root, build/test/benchmark commands
//   are required; the candidate is registered into the project, built/tested/
//   benchmarked via the project's own commands, then ALWAYS unregistered to pristine.
//
// =============================================================================

// --- Required Args ---
let MODEL_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = MODEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const OP_DESC = args.op_description || 'PyTorch model'

// --- Optional Args ---
const VERIFY_CMD = args.test_command || ''
const PROFILE_CMD = args.profile_command || ''
const COMPILE_CMD = args.compile_command || ''
const TARGET_SPEEDUP = args.target_speedup || 1.05
const MAX_TURNS = args.max_turns || 15
const EXP_DIR = args.exp_dir || '/tmp/cuda_agent_exp'
const ADAPTATION_SCOPE = 'inference_time_adaptation'
const LANGUAGE = args.language || 'cuda'
const TARGET_GPU = args.target_gpu || 'unknown GPU'
const SEED_CANDIDATES = args.seed_candidates || 3
// Optional ncu binary/command. native_profiler needs a real profiler to run; when
// absent the integration/profiling gates downgrade native_profiler -> perf_heuristic
// (A-O1 closure) so the Profile phase does not try ncu it cannot run.
const NCU_CMD = args.ncu_command || args.ncu_binary || ''

// --- Embedded-dispatch mode (gated; standalone path is byte-identical when off) ---
const INTEGRATION_PATTERN = (args.integration_pattern || 'standalone')
const EMBEDDED = INTEGRATION_PATTERN.startsWith('embedded')
const REGISTER_SCRIPT = args.register_script || ''
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const REFERENCE_FILE = args.reference_cuh || args.reference_file || ''
const REGISTER_PARAMS = args.register_params || ''
// In embedded mode the project's own build command drives the build step.
const BUILD_CMD = args.build_command || COMPILE_CMD

if (!MODEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

if (EMBEDDED) {
  const missing = []
  if (!REGISTER_SCRIPT) missing.push('register_script')
  if (!PROJECT_ROOT) missing.push('project_root (or ggml_root)')
  if (!BUILD_CMD) missing.push('build_command (or compile_command)')
  if (!VERIFY_CMD) missing.push('test_command')
  if (!PROFILE_CMD) missing.push('benchmark/profile_command')
  if (missing.length) {
    throw new Error(`integration_pattern="${INTEGRATION_PATTERN}" requires non-empty: ${missing.join(', ')}`)
  }
}

// --- State ---
let modelCode = ''
let eagerTime = 0
let compileTime = 0
let bestKernelCode = ''
let bestBindingCode = ''
let bestModelNew = ''
let bestSpeedup = 0
let currentAttempt = 0
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null
let history = []  // [{turn, action, outcome, speedup, error}]

// =============================================================================
// Phase 1: Setup — Read model, establish workspace
// =============================================================================
phase('Setup')

if (INPUT_MODE === 'generate_then_optimize') {
  const generated = await agentRetry(() => agent(`No kernel_path was provided. Generate and verify an initial PyTorch model plus CUDA kernel scaffold before CUDAAgent optimization.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- op_description: ${OP_DESC}
- language: ${LANGUAGE}
- target_gpu: ${TARGET_GPU}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- compile_command: ${COMPILE_CMD || '(not provided)'}
- test_command: ${VERIFY_CMD || '(not provided)'}
- profile_command: ${PROFILE_CMD || '(not provided)'}

# Contract
Generate ${SEED_CANDIDATES} complete candidates under ${EXP_DIR}/generated/. Run available commands using {kernel_path}/{result_path}. Return the best verified generated source path.`, {
    label: 'generate-initial-kernel',
    phase: 'Setup',
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
  if ((VERIFY_CMD || PROFILE_CMD) && initialGenerationResult.verified === false) throw new Error('No generated seed passed verification evidence')
  MODEL_PATH = generatedKernelPath
}

const setupResult = await agentRetry(() => agent(`You are a CUDA kernel optimization expert. Set up the optimization workspace.

# Task:
1. Read the PyTorch model from: ${MODEL_PATH}
2. Create workspace: mkdir -p ${EXP_DIR}/{kernels,profiles}
3. Analyze the model to identify:
   - What operations it performs
   - Which operators are performance-critical
   - Data types and tensor shapes involved
   - Opportunities for kernel fusion

# Operation: ${OP_DESC}

Return the model code and analysis.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create the file if missing; shell append with >>). Get the timestamp first by running: date -u +%Y-%m-%dT%H:%M:%SZ
Then append this one-line JSON, filling the bracketed parts from your analysis:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"workspace_setup","note":"<critical operators + fusion opportunities, one line>"}`, {
  label: 'setup-workspace',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      model_code: { type: 'string' },
      operator_list: { type: 'array', items: { type: 'string' } },
      critical_operators: { type: 'array', items: { type: 'string' } },
      fusion_opportunities: { type: 'array', items: { type: 'string' } },
      data_types: { type: 'string' },
      tensor_shapes: { type: 'string' },
    },
    required: ['model_code', 'critical_operators'],
  },
}), { retries: 5 })

modelCode = setupResult.model_code

// --- profiling-strategist: classify the kernel task (fuzzy op_class/size) and
// resolve a profiling METHOD against the CUDA substrate manifest. The agent only
// CLASSIFIES; the substrate DETERMINISTICALLY picks the method and STAMPS
// confidence. Honored in the profile/refine prompt below; defaults keep the
// happy-path ncu behavior unchanged if the decision is ignored. ---
{
  const _pd = await agentRetry(() => agent(
    `Classify the kernel under optimization. Source: ` +
    (MODEL_PATH ? `read ${MODEL_PATH}` : `operation "${OP_DESC}"`) + `.\n` +
    `Pick op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    substrateInstruction('profiling/profiling_strategist.py',
      `resolve --backend-manifest ${BACKEND_MANIFEST} --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl`) +
    ` Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
}
log(`Profiling method: ${PROFILING_DECISION.method} (confidence=${PROFILING_DECISION.confidence})`)

// --- integration-strategist: route build/test mode (standalone vs embedded_*).
// The agent only CLASSIFIES can_compile_standalone; the substrate DETERMINISTICALLY
// resolves the method (standalone | embedded_inplace | embedded_dispatch | derive_adapter).
// Defaults to standalone so the legacy path stays byte-identical when the decision
// is absent. The explicit integration_pattern arg (EMBEDDED above) is honored as a
// strong host hint into can_standalone. See _substrate/integration/README.md. ---
let INTEGRATION_DECISION = { method: EMBEDDED ? INTEGRATION_PATTERN : 'standalone', build_fidelity: 'isolated', reversible: true }
{
  const _kernelForProbe = MODEL_PATH || REFERENCE_FILE || ''
  const _canStandaloneHint = EMBEDDED ? 'no' : 'uncertain'
  const _probe = JSON.stringify({ compiler: !!COMPILE_CMD, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  if (_kernelForProbe) {
    const _integ = await agentRetry(() => agent(
      `Read ${_kernelForProbe}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
      `(use no when the file cannot compile as a single TU — e.g. a project .cuh with project-only deps; ` +
      `the caller hinted can_standalone="${_canStandaloneHint}"). Then ` +
      substrateInstruction('integration/integration_strategist.py',
        `resolve --kernel "${_kernelForProbe}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
        `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl`) +
      ` Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
      { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    if (_integ && _integ.method) INTEGRATION_DECISION = _integ
  }
}
log(`Integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error('integration-strategist returned derive_adapter — provide project_root + register_script + build/test commands (integration_pattern=embedded)')
}
// CUDAAgent has no standalone driver-build envelope (the validator agent runs
// COMPILE_CMD/VERIFY_CMD/PROFILE_CMD directly), so there is nothing extra to gate on a
// USE_DRIVER_STANDALONE flag; the standalone path is simply "not IS_EMBEDDED".
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
// embedded_inplace mutates the project file in place; back it up ONCE so every
// candidate restores to a pristine original and the exit net can too.
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Byte-exact backup (once): run \`cp -a "${REFERENCE_FILE || MODEL_PATH}" "${ORIGINAL_BACKUP}"\` and confirm it exists.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// A-O1 closure: native_profiler needs a real profiler to run; CUDAAgent's native
// profiler is ncu. If no ncu command is available, downgrade native_profiler ->
// perf_heuristic so the Profile phase derives memory/compute-bound hints from
// benchmark throughput instead of trying ncu it cannot run.
if (PROFILING_DECISION.method === 'native_profiler' && !NCU_CMD) {
  log(`profiling: native_profiler but no ncu_command -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'test-harness-perf', rationale: 'native_profiler but no ncu_command -> perf_heuristic' }
}

// =============================================================================
// Phase 2: Profile — Analyze baseline performance
// =============================================================================
phase('Profile')

const profileResult = await agentRetry(() => agent(`You are a CUDA performance profiler. Profile the baseline PyTorch model.

# Model Code:
\`\`\`python
${modelCode.substring(0, 4000)}
\`\`\`

# Profiling Tasks (CUDA Agent SKILL.md Step 1):
1. Profile the NATIVE PyTorch implementation:
${PROFILE_CMD ? `   Run: ${PROFILE_CMD}` : '   Estimate performance from operator analysis.'}

2. Measure:
   - PyTorch Eager mode execution time
   - torch.compile execution time (the baseline to beat)
   - Per-operator breakdown if available

3. Identify performance bottlenecks:
   - Excessive kernel launches (multiple small kernels instead of one fused)
   - Suboptimal memory access patterns
   - Redundant data movement (intermediate materializations)
   - Opportunities for operator fusion

4. Determine optimization strategy:
   - Which operators to fuse into a single CUDA kernel?
   - What memory access pattern to use?
   - What parallelism strategy (threads/blocks mapping)?

Return profiling results and optimization plan.

Profiling-strategist selected method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'). If method==='native_profiler', you MAY run ncu${NCU_CMD ? ` (ncu command: ${NCU_CMD})` : ''} for bottleneck evidence. If method==='perf_heuristic', derive memory-vs-compute-bound hints from benchmark throughput (latency, GFLOPS/GB-s) and tag them evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'; ALSO set heuristic_bclass (memory_bound|compute_bound|latency_bound) from the throughput ratio so downstream diagnosis does not fall back to unknown. If method==='static', reason from source only (confidence='hypothesized'). Never fabricate profiler counters.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Profile","ts":"<ts>","status":"done","technique":"<chosen optimization strategy, e.g. operator_fusion>","speedup":null,"note":"<baseline eager/compile ms + main bottleneck>"}`, { model: MODEL.profile,
  label: 'profile-baseline',
  phase: 'Profile',
  schema: {
    type: 'object',
    properties: {
      eager_time_ms: { type: 'number' },
      compile_time_ms: { type: 'number' },
      per_operator_breakdown: { type: 'array', items: { type: 'object', properties: { op: { type: 'string' }, time_ms: { type: 'number' } } } },
      bottlenecks: { type: 'array', items: { type: 'string' } },
      optimization_strategy: { type: 'string' },
      fusion_plan: { type: 'string' },
      heuristic_bclass: { type: 'string' },
    },
    required: ['eager_time_ms', 'compile_time_ms', 'bottlenecks', 'optimization_strategy'],
  },
}), { retries: 5 })

eagerTime = profileResult.eager_time_ms || 1.0
compileTime = profileResult.compile_time_ms || 1.0

log(`Baseline: eager=${eagerTime}ms, compile=${compileTime}ms | Bottlenecks: ${profileResult.bottlenecks.join(', ')}`)
log(`Strategy: ${profileResult.optimization_strategy}`)

// =============================================================================
// Iterative Refinement Loop (SKILL.md Steps 2-4)
// =============================================================================

let targetMet = false

for (currentAttempt = 0; currentAttempt < MAX_TURNS && !targetMet; currentAttempt++) {

  // ===========================================================================
  // Phase 3: Implement — Generate CUDA kernel + bindings + model_new
  // ===========================================================================
  phase('Implement')

  const recentHistory = history.slice(-5)
  const historyContext = recentHistory.length > 0
    ? `\n# Previous Attempts:\n${recentHistory.map(h => `Turn ${h.turn}: ${h.action} → ${h.outcome}${h.error ? ' (' + h.error.substring(0, 100) + ')' : ''} ${h.speedup ? h.speedup.toFixed(2) + 'x' : ''}`).join('\n')}`
    : ''

  // Proactive knowledge fetch: when retrying after a failure (history non-empty),
  // tell the Implement doer to consult the on-demand Knowledge Tools block KerSor
  // injected into the semantic inputs (op_description etc.) before rewriting — so a
  // bottleneck/API it is unsure about is grounded in the local corpus rather than
  // guessed. Best-effort; the block is absent when retrieval is off.
  const proactiveKnowledgeHint = recentHistory.length > 0
    ? '\n# Proactive knowledge fetch (on retries)\nIf a previous attempt FAILED (compile/correctness/speedup) or you are unsure about an API, intrinsic, or how a known bottleneck is typically resolved, FIRST run the search command from the `## Knowledge Tools (on-demand)` block in your input (e.g. `query.py` for kernel patterns, `chub search` for API/Triton docs). Read 1-2 returned pages, extract the actionable technique, then implement. This is best-effort: if no block is present or nothing relevant returns, proceed with your own knowledge. Do not block on it.'
    : ''

  const embeddedProposalBlock = IS_EMBEDDED
    ? `\n\n${EMBEDDING_CONTRACT}\n\nMANDATORY: Read the reference dispatch file at ${REFERENCE_FILE} and match its dispatch signature EXACTLY (same entry-point shape, template params, launch-bounds conventions). Emit a COMPLETE dispatch-compatible \`.cuh\` (NOT a standalone translation unit, NO main()/harness). Put the full \`.cuh\` contents in kernel_code; binding_code and model_new_code are not used in embedded mode (return brief placeholders).`
    : ''

  const implResult = await agentRetry(() => agent(`You are a CUDA kernel developer. Implement an optimized CUDA kernel for this PyTorch model.

# Model to Optimize:
\`\`\`python
${modelCode.substring(0, 3000)}
\`\`\`

# Operation: ${OP_DESC}
# Optimization Strategy: ${profileResult.optimization_strategy}
# Fusion Plan: ${profileResult.fusion_plan || 'Fuse critical operators'}
# Bottlenecks to Address: ${profileResult.bottlenecks.join('; ')}

# Baseline Performance:
- Eager: ${eagerTime}ms
- torch.compile: ${compileTime}ms
- Target: >${TARGET_SPEEDUP}x speedup over torch.compile (=${(compileTime / TARGET_SPEEDUP).toFixed(3)}ms)
${historyContext}${proactiveKnowledgeHint}${embeddedProposalBlock}

# CUDA Agent Workspace Requirements:
Generate THREE files:

## 1. kernels/kernel.cu — CUDA kernel source
- Optimized __global__ kernel function(s)
- Proper thread/block mapping
- Memory coalescing, shared memory usage where beneficial
- Error checking

## 2. kernels/kernel_binding.cpp — pybind11 bindings
- Expose kernel launch wrapper to Python
- Proper tensor type checking
- PYBIND11_MODULE declaration

## 3. model_new.py — PyTorch model using custom CUDA ops
- Import compiled extension
- Replace performance-critical operators with custom CUDA calls
- Maintain same interface as original model.py

# Key Rules:
- Performance gains must come SOLELY from the generated CUDA kernel
- Do NOT use torch.nn.functional fallbacks
- Ensure numerical correctness (match reference within tolerance)
- This is attempt ${currentAttempt + 1}/${MAX_TURNS}

Return all three files.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is optimization attempt ${currentAttempt}):
{"workflow":"${WORKFLOW_NAME}","phase":"Implement","ts":"<ts>","status":"done","candidate_id":"attempt-${currentAttempt}","technique":"<the main optimization you applied this attempt>","note":"<what changed vs the previous attempt>"}`, {
    label: `impl-${currentAttempt}`,
    phase: 'Implement',
    schema: {
      type: 'object',
      properties: {
        kernel_code: { type: 'string' },
        binding_code: { type: 'string' },
        model_new_code: { type: 'string' },
        implementation_notes: { type: 'string' },
      },
      required: ['kernel_code', 'binding_code', 'model_new_code'],
    },
  }), { retries: 5 })

  // ===========================================================================
  // Phase 4: Verify — Compile + correctness + performance
  // ===========================================================================
  phase('Verify')

  // Embedded evaluation (integration-strategist → embedded_dispatch / embedded_inplace).
  // This for-loop is already SERIAL, so embedded eval (which mutates a shared file or a
  // shared project build) runs inline with no race — no separate serial loop needed.
  // Gated on the RUNTIME decision IS_EMBEDDED, not the static integration_pattern arg.
  let embeddedEvalBlock = ''
  if (IS_EMBEDDED) {
    const variantName = `cuda_agent_t${currentAttempt}`.replace(/[^A-Za-z0-9_]/g, '_')
    const candidatePath = `${EXP_DIR}/kernels/${variantName}.cuh`
    if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
      // embedded_inplace: copy the candidate over the project file in place, build/test/
      // benchmark with the project's own commands, then ALWAYS restore the pristine backup.
      const projectFile = REFERENCE_FILE || MODEL_PATH
      embeddedEvalBlock = `

# EMBEDDED-INPLACE EVALUATION (overrides the standalone steps below)
This candidate is NOT standalone; it replaces the project file in place. Write the kernel_code above verbatim to ${candidatePath}, then evaluate by running these commands IN THIS EXACT ORDER:

1. Restore pristine: cp -a "${ORIGINAL_BACKUP}" "${projectFile}"
2. Apply candidate:  cp "${candidatePath}" "${projectFile}"
3. Build:            ${BUILD_CMD}
4. Test:             ${VERIFY_CMD}        (correctness)
5. Benchmark:        ${PROFILE_CMD || VERIFY_CMD}   (latency)
6. ALWAYS restore:   cp -a "${ORIGINAL_BACKUP}" "${projectFile}"

HARD REQUIREMENT (cleanup invariant): You MUST run the final restore (step 6) and confirm the project file is byte-exact pristine even on compile/correctness/benchmark FAILURE or non-improvement. Never leave the project dirty.

Parse correctness (pass/fail) and latency STRICTLY from the test/benchmark command output. Do NOT fabricate numbers; if a value is not present in the output, report it as unavailable rather than guessing. Map results into the schema (compiled=build succeeded, correct=test passed, kernel_time_ms=measured latency).`
    } else {
      // embedded_dispatch (default embedded): register into the project's dispatch table,
      // build/test/benchmark via the project's own commands, then ALWAYS unregister to pristine.
      const plan = __embeddedEvalPlan({
        adapter: 'python "' + REGISTER_SCRIPT + '"',
        variant: variantName,
        source: candidatePath,
        projectRoot: PROJECT_ROOT,
        params: REGISTER_PARAMS,
        buildCmd: BUILD_CMD,
        testCmd: VERIFY_CMD,
        benchmarkCmd: PROFILE_CMD,
      })
      embeddedEvalBlock = `

# EMBEDDED-DISPATCH EVALUATION (overrides the standalone steps below)
This candidate is NOT standalone. Write the kernel_code above verbatim to ${candidatePath}, then evaluate it against the project's dispatch adapter by running these commands IN THIS EXACT ORDER:

1. Register:   ${plan.register}
2. List:       ${plan.list}   (CONFIRM ${variantName} is now listed; abort if absent)
3. Build:      ${plan.build}
4. Test:       ${plan.test}        (correctness)
5. Benchmark:  ${plan.benchmark}   (latency)
6. Unregister: ${plan.unregister}
7. List:       ${plan.list}   (CONFIRM ${variantName} is GONE)

HARD REQUIREMENT (cleanup invariant): ${plan.cleanupInvariant}
You MUST run the unregister command and confirm removal even on compile/correctness/benchmark FAILURE or non-improvement. Never leave the project dirty.

Parse correctness (pass/fail) and latency STRICTLY from the test/benchmark command output. Do NOT fabricate numbers; if a value is not present in the output, report it as unavailable rather than guessing. Map results into the schema (compiled=build succeeded, correct=test passed, kernel_time_ms=measured latency).`
    }
  }

  const verifyResult = await agentRetry(() => agent(`You are a CUDA kernel validator. Compile, verify, and benchmark this kernel implementation.${embeddedEvalBlock}

# Kernel Code (kernel.cu):
\`\`\`cuda
${implResult.kernel_code.substring(0, 4000)}
\`\`\`

# Binding Code (kernel_binding.cpp):
\`\`\`cpp
${implResult.binding_code.substring(0, 2000)}
\`\`\`

# Model New (model_new.py):
\`\`\`python
${implResult.model_new_code.substring(0, 2000)}
\`\`\`

# Validation Steps:

## Step 1: Compile
${COMPILE_CMD ? `Run: ${COMPILE_CMD}` : 'No compile_command provided; perform static compileability review only.'}
Check for compilation errors.

## Step 2: Correctness Verification
${VERIFY_CMD ? `Run: ${VERIFY_CMD}` : 'No test_command provided; do not invent a correctness test command. Describe the required reference comparison and mark measured correctness as unavailable.'}
- Use tolerance: atol=1e-3, rtol=1e-3
- Test with multiple input shapes if applicable

## Step 3: Performance Measurement
${PROFILE_CMD ? `Run exactly the user-provided profile_command: ${PROFILE_CMD}` : 'No profile_command provided; do not invent a performance measurement command.'}
- Warm-up iterations: 10
- Measurement iterations: 100
- Report: kernel_time, speedup_vs_eager, speedup_vs_compile

## Step 4: Compute Reward (CUDA Agent reward schedule)
- r = -1 if correctness fails
- r = 3 if faster than BOTH eager(${eagerTime}ms) AND compile(${compileTime}ms) by >5%
- r = 2 if faster than eager only by >5%
- r = 1 if correct but not faster

Return results.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if correctness passed, else "error"; speedup is the measured speedup_vs_compile number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Verify","ts":"<ts>","status":"<done|error>","candidate_id":"attempt-${currentAttempt}","speedup":<number or null>,"technique":"<technique under test>","note":"<compiled? correct? reward; or the failure reason>"}`, {
    label: `verify-${currentAttempt}`,
    phase: 'Verify',
    schema: {
      type: 'object',
      properties: {
        compiled: { type: 'boolean' },
        compile_error: { type: 'string' },
        correct: { type: 'boolean' },
        correctness_error: { type: 'string' },
        kernel_time_ms: { type: 'number' },
        speedup_vs_eager: { type: 'number' },
        speedup_vs_compile: { type: 'number' },
        reward: { type: 'number' },
        performance_notes: { type: 'string' },
      },
      required: ['compiled', 'correct', 'reward'],
    },
  }), { retries: 5 })

  // Record history
  let outcome = ''
  let error = ''
  if (!verifyResult.compiled) {
    outcome = 'compile_error'
    error = verifyResult.compile_error || ''
  } else if (!verifyResult.correct) {
    outcome = 'incorrect'
    error = verifyResult.correctness_error || ''
  } else {
    outcome = `correct (${verifyResult.speedup_vs_compile?.toFixed(2) || '?'}x vs compile)`
  }

  history.push({
    turn: currentAttempt,
    action: implResult.implementation_notes?.substring(0, 50) || 'kernel implementation',
    outcome: outcome,
    speedup: verifyResult.speedup_vs_compile || 0,
    error: error,
    reward: verifyResult.reward,
  })

  // Update best
  if (verifyResult.correct && (verifyResult.speedup_vs_compile || 0) > bestSpeedup) {
    bestKernelCode = implResult.kernel_code
    bestBindingCode = implResult.binding_code
    bestModelNew = implResult.model_new_code
    bestSpeedup = verifyResult.speedup_vs_compile || 0
    log(`  NEW BEST: ${bestSpeedup.toFixed(2)}x vs compile (reward=${verifyResult.reward})`)
  }

  // Check if target met
  if (verifyResult.correct && (verifyResult.speedup_vs_compile || 0) >= TARGET_SPEEDUP) {
    targetMet = true
    log(`  TARGET MET: ${verifyResult.speedup_vs_compile?.toFixed(2)}x ≥ ${TARGET_SPEEDUP}x`)
  } else {
    // ===========================================================================
    // Phase 5: Refine — Diagnose and plan fix
    // ===========================================================================
    phase('Refine')

    if (!targetMet && currentAttempt < MAX_TURNS - 1) {
      log(`  Turn ${currentAttempt + 1}: ${outcome} | Refining...`)
    }
  }
}

// =============================================================================
// Phase 6: Report
// =============================================================================
phase('Report')

const finalReport = await agentRetry(() => agent(`Write a concise optimization report.

# CUDA Agent Optimization Results
- Adaptation scope: ${ADAPTATION_SCOPE}
- Operation: ${OP_DESC}
- Baseline eager: ${eagerTime}ms
- Baseline compile: ${compileTime}ms
- Best kernel time: ${compileTime / (bestSpeedup || 1)}ms
- Best speedup vs compile: ${bestSpeedup.toFixed(2)}x
- Target: ${TARGET_SPEEDUP}x | ${targetMet ? 'ACHIEVED' : 'NOT MET'}
- Turns used: ${currentAttempt}/${MAX_TURNS}

# Optimization History:
${history.map(h => `Turn ${h.turn + 1}: ${h.outcome} (reward=${h.reward})`).join('\n')}

# Best Kernel:
\`\`\`cuda
${bestKernelCode.substring(0, 3000)}
\`\`\`

Write:
1. What optimization strategy worked
2. Key challenges encountered (compile errors, correctness issues)
3. Performance breakdown (where the speedup comes from)
4. Remaining optimization opportunities`, {
  label: 'final-report',
  phase: 'Report',
}), { retries: 5 })

// Exit restore (embedded_inplace safety net): the candidate eval restores after each
// attempt, but force one final restore so the project file is byte-exact pristine on
// exit regardless of how the loop terminated. No-op for standalone/embedded_dispatch.
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`Exit restore: run \`cp -a "${ORIGINAL_BACKUP}" "${REFERENCE_FILE || MODEL_PATH}"\` and confirm the project file is byte-exact pristine.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  operation: OP_DESC,
  eager_time_ms: eagerTime,
  compile_time_ms: compileTime,
  best_speedup_vs_compile: bestSpeedup,
  best_speedup_vs_eager: eagerTime / (compileTime / (bestSpeedup || 1)),
  target_met: targetMet,
  turns_used: currentAttempt,
  max_turns: MAX_TURNS,
  reward_history: history.map(h => h.reward),
  adaptation_scope: ADAPTATION_SCOPE,
  best_kernel_code: bestKernelCode,
  best_binding_code: bestBindingCode,
  best_model_new: bestModelNew,
  report: finalReport,
}
