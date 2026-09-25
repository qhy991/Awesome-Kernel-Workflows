export const meta = {
  name: 'fact-kernel-optimization',
  description: 'Compositional kernel synthesis with pattern discovery and realization',
  whenToUse: 'Use for generating optimized CUTLASS kernels through compositional pattern synthesis',
  phases: [
    { title: 'Setup', detail: 'Initialize CUTLASS environment and pattern registry' },
    { title: 'Pattern Discovery', detail: 'Discover optimization patterns from exemplars' },
    { title: 'Pattern Realization', detail: 'Realize patterns as code transformations' },
    { title: 'Pattern Composition', detail: 'Compose patterns into optimized kernels' },
    { title: 'Ablation', detail: 'Ablation studies to validate pattern contributions' },
    { title: 'Evaluation', detail: 'Evaluate composed kernels on target hardware' },
    { title: 'Report', detail: 'Generate synthesis report' },
  ],
};

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
async function __solExecbenchEvaluate(ctx) {
  // Claude's legacy Workflow host does not yet expose this optional primitive.
  // Keep the prompt-driven path as a compatibility edge, while KerSor's Host
  // owns exact source materialization and PACK/RUN/PARSE without an LLM turn.
  if (typeof evaluate !== 'function') return null
  return evaluate({
    protocol: 'sol-execbench-v1',
    label: ctx.label || 'sol-eval',
    phase: ctx.phase || 'Evaluate',
    candidatePath: ctx.kernelSource,
    candidateSource: ctx.candidateSource,
    substrateDir: ctx.substrateDir,
    contractEnv: ctx.contractEnv,
    solutionOut: ctx.solutionOut,
    benchOut: ctx.benchOut,
    normalizedOut: ctx.normalizedOut || `${ctx.benchOut}.result.json`,
    solCli: ctx.solCli,
    taskDir: ctx.taskDir,
    benchConfig: ctx.benchConfig,
    seedDir: ctx.seedDir,
    cudaVisibleDevices: ctx.cudaVisibleDevices || '0',
    ldLibraryPath: ctx.ldLibraryPath || '',
    envPrefix: ctx.envPrefix || '',
    definitionPath: ctx.definitionPath || '',
    timeoutSeconds: ctx.timeoutSeconds || 0,
  }).then(__solGuardHarnessFault)
}

// A `compiled: false` from the evaluator does not always mean the candidate is
// bad.  `invalid_request` and `infrastructure_error` are the harness refusing or
// failing before the candidate was ever built, and callers that map any
// non-success onto compile_error burn refine turns and a stagnation budget on a
// misconfiguration.  Observed: a candidate staged outside the evaluation roots
// was rejected at preflight, reported three times as `compile_error`, and the run
// stopped at the stagnation limit having never compiled anything.  Surface a
// harness fault as a harness fault and stop, because retrying cannot fix it.
function __solGuardHarnessFault(result) {
  const HARNESS_FAULTS = ['invalid_request', 'infrastructure_error']
  if (result && HARNESS_FAULTS.includes(result.failure_code)) {
    const detail = result.stderr || result.stdout || ''
    throw new Error(
      `sol-execbench harness fault (${result.failure_code}) at stage `
      + `${result.stage || 'unknown'}: ${String(detail).slice(0, 400)} `
      + '- this is a harness or wiring fault, not a candidate compile error',
    )
  }
  return result
}

// --- sol-execbench wiring (Host-owned PACK/RUN/PARSE; no LLM turn) -----------
// These 15 workflows declared only the standalone path, so their benchmark was a
// command string embedded in a prompt for a read-only activation with no shell.
// The read-only allowlist deliberately excludes execution, so the agent could
// never run it.  The Host can, and this is the protocol it exposes for exactly
// that.
const SOL_CLI = args.sol_cli || ''
const SOL_TASK_DIR = args.sol_task_dir || ''
const SOL_BENCH_CONFIG = args.sol_bench_config || ''
const SOL_SEED_DIR = args.sol_seed_dir || ''
const SOL_CVD = args.sol_cuda_visible_devices || '0'
const SOL_LD_LIBRARY_PATH = args.sol_ld_library_path || ''
const SOL_ENV_PREFIX = args.sol_env_prefix || ''
const SOL_DEFINITION_PATH = args.sol_definition_path || ''
const SOL_SUBSTRATE_DIR = args.sol_substrate_dir || ''
const SOL_AVAILABLE = Boolean(SOL_CLI && SOL_TASK_DIR && SOL_SUBSTRATE_DIR)
// A handoff must not fail because a turn legally omitted an optional field.
// __fmt renders a number that may be absent without throwing; the array forms
// below use `|| []` so a missing list yields an empty render instead of ending
// the run and taking every earlier result with it.
const __fmt = (v, d = 2) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : 'unreported')


// __modelTierApplied
// --- END model-tier ---

// --- profiling-strategist substrate wiring (additive; FACT has no native
// profiler, so the strategist resolves against the canonical CUDA manifest and
// ADDS strategist-routed profiling availability for the ablation stage). Task is
// fixed to 'gemm' (FACT is CUTLASS-GEMM centric); the agent only classifies size;
// the substrate stamps method+confidence. ---
const SUBSTRATE = args.substrate_dir || '_substrate'
const SUBSTRATE_PY = args.substrate_command_prefix || 'python3'
const STRATEGIST_MANIFEST = args.backend_manifest || `${SUBSTRATE}/backends/cuda/manifest.json`
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

// Top-level profiling decision; overwritten only on a valid strategist response.
// Defaults keep the happy path (discover/realize/compose loop) unchanged if ignored.
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured' }

const WORKFLOW_NAME = 'fact-kernel-optimization'


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
  ? (args.failed_strategy_ids || []).filter(id => typeof id === 'string' && id)
  : ((ATTEMPT_EVIDENCE && Array.isArray(ATTEMPT_EVIDENCE.transfer_items))
    ? (ATTEMPT_EVIDENCE.transfer_items || []).filter(i => i && i.kind === 'failed_strategy' && i.id).map(i => i.id)
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
      // Provider refusals and model-identity mismatches are terminal for this
      // request. Repeating a rejected prompt or paying for more calls on the
      // wrong model cannot repair either condition. The message check also
      // protects direct/older Hosts that lack the typed refusal code.
      if (e && (e.code === 'KERSOR_PROVIDER_SAFEGUARD_REFUSAL'
        || e.code === 'KERSOR_CLAUDE_MODEL_IDENTITY_MISMATCH'
        || /safeguards? flagged (?:this|the) message|provider safeguard refusal/i.test(String(e.message || '')))) {
        throw e
      }
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
  let expired = false
  const guard = new Promise((_, reject) => {
    timer = setTimeout(
      () => {
        expired = true
        reject(new Error(`turn-timeout: ${label} exceeded ${Math.round(TURN_TIMEOUT_MS / 1000)}s`))
      },
      TURN_TIMEOUT_MS)
  })
  return Promise.race([promise, guard]).catch(async error => {
    if (expired) {
      // A race does not cancel agent(). Drain this Host-bounded activation so
      // no running call survives the workflow's return.
      try { await promise } catch (_) { /* preserve the guard error */ }
    }
    throw error
  }).finally(() => {
    if (typeof clearTimeout === 'function') clearTimeout(timer)
  })
}
// --- END inlined turn-timeout scaffolding ---
// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.

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


// --- Embedded-dispatch mode (gated; standalone path is byte-identical when off) ---
const INTEGRATION_PATTERN = (args.integration_pattern || 'standalone')
const EMBEDDED = INTEGRATION_PATTERN.startsWith('embedded')
const REGISTER_SCRIPT = args.register_script || ''
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const REFERENCE_FILE = args.reference_cuh || args.reference_file || ''
const REGISTER_PARAMS = args.register_params || ''
// Standalone synthesis is fully agent-narrated; embedded mode drives the project's
// own build/test/benchmark commands against a contract-conforming register adapter.
const BUILD_CMD = args.build_command || ''
const TEST_CMD = args.test_command || ''
const BENCHMARK_CMD = args.benchmark_command || ''

if (EMBEDDED) {
  const missing = []
  if (!REGISTER_SCRIPT) missing.push('register_script')
  if (!PROJECT_ROOT) missing.push('project_root (or ggml_root)')
  if (!BUILD_CMD) missing.push('build_command')
  if (!TEST_CMD) missing.push('test_command')
  if (!BENCHMARK_CMD) missing.push('benchmark_command')
  if (missing.length) {
    throw new Error(`integration_pattern="${INTEGRATION_PATTERN}" (embedded dispatch) requires non-empty: ${missing.join(', ')}`)
  }
}

// FACT: Compositional kernel synthesis framework
// Based on GitHub:Project-FACT/FACT (no published paper yet)
// Discovers, realizes, and composes optimization patterns for CUTLASS

async function main() {
  // ============================================================================
  // Phase 1: Setup
  // ============================================================================
  phase('Setup');

  const setupResult = await agentRetry(() => agent(
    `Set up FACT compositional synthesis environment:

1. Initialize CUTLASS workspace:
   - CUTLASS version and headers
   - Target GPU architecture (sm_80, sm_89, sm_90, etc.)
   - Available tensor core operations
2. Set up pattern registry T(rule, dtype, architecture):
   - rule: transformation type (tiling, fusion, memory, etc.)
   - dtype: data type (fp32, fp16, bf16, int8)
   - architecture: GPU architecture (Ampere, Hopper, etc.)
3. Identify target kernel specification:
   - Operation (GEMM, Conv, attention, etc.)
   - Input shapes and dtypes
   - Target architecture
4. Load exemplar kernels:
   - High-performance reference implementations
   - Pattern sources for discovery
5. Configure synthesis parameters:
   - Pattern discovery depth
   - Composition budget
   - Ablation strategy

Return JSON:
{
  "cutlass_version": "3.x|2.x",
  "target_architecture": "sm_80|sm_89|sm_90",
  "tensor_cores_available": true/false,
  "kernel_spec": {
    "operation": "gemm|conv2d|attention|...",
    "shapes": "shape description",
    "dtypes": ["fp32", "fp16", ...],
    "target_arch": "ampere|hopper|..."
  },
  "pattern_registry_path": "path/to/registry",
  "exemplar_kernels": ["exemplar1", "exemplar2", ...],
  "discovery_depth": <int>,
  "composition_budget": <int>,
  "baseline_gflops": <float or null>
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${args.exp_dir}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"cutlass_workspace_setup","note":"<target operation + architecture + exemplar count, one line>"}`,
    {
      label: 'Setup FACT',
      phase: 'Setup',
      schema: {
        type: 'object',
        properties: {
          cutlass_version: { type: 'string' },
          target_architecture: { type: 'string' },
          tensor_cores_available: { type: 'boolean' },
          kernel_spec: { type: 'object' },
          pattern_registry_path: { type: 'string' },
          exemplar_kernels: { type: 'array', items: { type: 'string' } },
          discovery_depth: { type: 'integer' },
          composition_budget: { type: 'integer' },
          baseline_gflops: { type: ['number', 'null'] },
        },
        required: ['cutlass_version', 'target_architecture', 'kernel_spec'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!setupResult) {
    log('Setup failed');
    return { success: false, reason: 'setup_failed' };
  }
  // Optional arrays may be legally omitted by a structured response.
  setupResult.exemplar_kernels = Array.isArray(setupResult.exemplar_kernels)
    ? setupResult.exemplar_kernels : (args.kernel_path ? [args.kernel_path] : []);

  log(`Target: ${setupResult.kernel_spec.operation} on ${setupResult.target_architecture}`);
  log(`CUTLASS ${setupResult.cutlass_version}, Tensor Cores: ${setupResult.tensor_cores_available ? 'Yes' : 'No'}`);
  log(`Exemplars: ${setupResult.exemplar_kernels.length}`);

  const kernelSpec = setupResult.kernel_spec;
  const discoveryDepth = setupResult.discovery_depth || 3;
  const compositionBudget = setupResult.composition_budget || 50;

  // --- profiling-strategist: classify the GEMM problem SIZE (task fixed to
  // 'gemm'; FACT is CUTLASS-GEMM centric), then let the substrate
  // DETERMINISTICALLY pick the profiling method and STAMP confidence. The agent
  // must NOT assign confidence. Default keeps the happy path unchanged if the
  // decision is ignored. Useful for the ablation stage; does not disturb the
  // discover/realize/compose loop. ---
  const _pd = await agentRetry(() => agent(`Classify the GEMM problem SIZE for the profiling strategist (the task is fixed to 'gemm'; you classify size only — one of tiny|small|large — based on the target operation ${kernelSpec.operation}, shapes ${kernelSpec.shapes}, and dtypes ${(kernelSpec.dtypes || []).join(', ')}).
Then run exactly: \`${SUBSTRATE_PY} ${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${STRATEGIST_MANIFEST} --task gemm --size <tiny|small|large> --cache ${args.exp_dir}/prof_cache.json --trajectory ${args.exp_dir}/genome.jsonl\`
Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}. Do NOT assign confidence yourself — the substrate stamps it.`, {
    model: MODEL.mechanical,
    label: 'profiling-strategist',
    phase: 'Setup',
    schema: JSON_PASSTHROUGH,
  }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
  log(`Profiling-strategist: method=${PROFILING_DECISION.method} confidence=${PROFILING_DECISION.confidence}`)

  // --- integration-strategist: DECIDE standalone vs embedded_* at runtime. This
  // REPLACES the static `EMBEDDED` (from args.integration_pattern) as the authority
  // for which eval path runs: the strategist classifies whether the (reference)
  // kernel can compile as a single TU and routes accordingly. FACT has no backend
  // driver, so there is no USE_DRIVER_STANDALONE; standalone runs FACT's native
  // (in-prompt) CUTLASS compile/eval, IS_EMBEDDED runs project-native
  // register/build/test/benchmark. Additive: when method==='standalone' the path
  // below is byte-identical to before. The static `EMBEDDED` arg SEEDS the default
  // for backward compat (existing embedded callers keep working), then the
  // strategist may override. See _substrate/integration/README.md and ROLLOUT.md.
  // FACT has no single KERNEL_PATH; the strategist reads the dispatch reference
  // (REFERENCE_FILE) when present — that is the file whose standalone-compile
  // ability determines the embedded route.
  const _integProbeKernel = REFERENCE_FILE || (setupResult.exemplar_kernels && setupResult.exemplar_kernels[0]) || ''
  const _integSeedMethod = EMBEDDED ? 'embedded_dispatch' : 'standalone'
  let INTEGRATION_DECISION = { method: _integSeedMethod, build_fidelity: 'isolated', reversible: true }
  {
    const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
    const _integ = await agentRetry(() => agent(
      `${_integProbeKernel ? `Read ${_integProbeKernel}; ` : ''}classify can_compile_standalone as exactly one of yes|no|uncertain ` +
      `(use no when the file cannot compile as a single TU — e.g. a llama.cpp .cuh with project-only deps). Then ` +
      `Run exactly: \`${SUBSTRATE_PY} ${SUBSTRATE}/integration/integration_strategist.py resolve ` +
      `--kernel "${_integProbeKernel || args.exp_dir + '/_fact_target'}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
      `--cache ${args.exp_dir}/integ_cache.json --trajectory ${args.exp_dir}/genome.jsonl\`. ` +
      `Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
      { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    // A caller that declared `integration_pattern` has already made this decision;
  // re-deciding it from an unvalidated model reply is how an explicit instruction
  // gets silently discarded.  Measured on B300: KDA was given
  // integration_pattern=sol_execbench_solution, ran the strategist anyway, adopted
  // the reply and logged `integration method = null`, which switched off the
  // deterministic sol-execbench path for the whole run.  Adopt only when the caller
  // said nothing, and only a method from the known set.
  if (!args.integration_pattern && _integ && ['standalone', 'embedded_inplace', 'embedded_dispatch', 'sol_execbench_solution', 'derive_adapter'].includes(_integ.method)) {
    INTEGRATION_DECISION = _integ
  }
  }
  log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
  if (INTEGRATION_DECISION.method === 'derive_adapter') {
    throw new Error('integration-strategist returned derive_adapter — provide project_root + register_script + build/test/benchmark commands')
  }
  // IS_EMBEDDED is the RUNTIME authority (gates eval/composition blocks below),
  // replacing the static `EMBEDDED` arg as the source of truth. ORIGINAL_BACKUP is
  // taken once for embedded_inplace (restore-on-exit safety net).
  const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
  const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${args.exp_dir}/integ_original.backup` : ''
  if (ORIGINAL_BACKUP && _integProbeKernel) {
    await agentRetry(() => agent(`Byte-exact backup: run \`cp -a "${_integProbeKernel}" "${ORIGINAL_BACKUP}"\` and confirm.`,
      { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
  }
  // If the strategist routed to an embedded path, validate the required wiring now
  // (the static `EMBEDDED` arg check above only fires when integration_pattern was
  // passed; the strategist can decide embedded even when it was not).
  if (IS_EMBEDDED) {
    const missing = []
    if (INTEGRATION_DECISION.method === 'embedded_dispatch' && !REGISTER_SCRIPT) missing.push('register_script')
    if (!PROJECT_ROOT) missing.push('project_root (or ggml_root)')
    if (!BUILD_CMD) missing.push('build_command')
    if (!TEST_CMD) missing.push('test_command')
    if (!BENCHMARK_CMD) missing.push('benchmark_command')
    if (missing.length) {
      throw new Error(`integration-strategist routed to ${INTEGRATION_DECISION.method} but missing: ${missing.join(', ')}`)
    }
  }
  // A-O1 closure: FACT has no native profiler/ncu, so a native_profiler choice
  // cannot be honored on the embedded path without a standalone CUTLASS compile ->
  // downgrade to perf_heuristic (which writes heuristic_bclass in the ablation
  // prompt so diagnose stays defined). Gated on IS_EMBEDDED so the standalone
  // path's PROFILING_DECISION (and the byte-identical ablation prompt) is unchanged.
  if (IS_EMBEDDED && PROFILING_DECISION.method === 'native_profiler' && !BENCHMARK_CMD) {
    log(`profiling: native_profiler but no benchmark command on embedded path -> downgrade to perf_heuristic`)
    PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
      profiler_name: 'fact-perf', rationale: 'native_profiler but no benchmark_command on embedded path -> perf_heuristic' }
  }

  // Pattern registry: T(rule, dtype, architecture) -> code transformation
  const patternRegistry = [];
  const discoveredPatterns = [];
  const realizedPatterns = [];
  let composedKernels = [];

  // ============================================================================
  // Phase 2: Pattern Discovery
  // ============================================================================
  phase('Pattern Discovery');

  log(`Discovering optimization patterns from ${setupResult.exemplar_kernels.length} exemplars...`);

  const discoveryResult = await agentRetry(() => agent(
    `Discover optimization patterns from exemplar kernels:

Target operation: ${kernelSpec.operation}
Target architecture: ${setupResult.target_architecture}
Exemplar kernels: ${(setupResult.exemplar_kernels || []).join(', ')}
Discovery depth: ${discoveryDepth}

Pattern discovery process:
1. Analyze exemplar implementations:
   - Extract code structure and transformations
   - Identify optimization idioms
   - Recognize architecture-specific patterns
2. Classify patterns by type:
   - Tiling patterns (block sizes, thread mapping)
   - Memory patterns (shared memory, global coalescing)
   - Compute patterns (tensor core usage, instruction scheduling)
   - Fusion patterns (operation fusion, epilogue fusion)
   - Data layout patterns (swizzling, padding)
3. Abstract patterns to rules:
   - Input conditions (shape constraints, dtype requirements)
   - Transformation logic (code template with parameters)
   - Output properties (performance characteristics)
4. Index patterns in registry: T(rule_type, dtype, architecture)

Return JSON:
{
  "exemplars_analyzed": <int>,
  "patterns_discovered": [
    {
      "pattern_id": "unique_id",
      "pattern_name": "descriptive name",
      "pattern_type": "tiling|memory|compute|fusion|layout",
      "rule_type": "rule classification",
      "applicable_dtypes": ["fp32", "fp16", ...],
      "applicable_architectures": ["sm_80", "sm_89", ...],
      "description": "what this pattern does",
      "abstraction": "abstract rule description",
      "expected_impact": "high|medium|low"
    },
    ...
  ],
  "discovery_summary": "summary of discovered patterns"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${args.exp_dir}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Pattern Discovery","ts":"<ts>","status":"done","technique":"pattern_discovery","note":"<how many patterns discovered + dominant pattern types, one line>"}`,
    {
      label: 'Discover patterns',
      phase: 'Pattern Discovery',
      schema: {
        type: 'object',
        properties: {
          exemplars_analyzed: { type: 'integer' },
          patterns_discovered: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                pattern_id: { type: 'string' },
                pattern_name: { type: 'string' },
                pattern_type: { type: 'string' },
                rule_type: { type: 'string' },
                applicable_dtypes: { type: 'array', items: { type: 'string' } },
                applicable_architectures: { type: 'array', items: { type: 'string' } },
                description: { type: 'string' },
                abstraction: { type: 'string' },
                expected_impact: { type: 'string' },
              },
              required: ['pattern_id', 'pattern_name', 'pattern_type', 'description'],
            },
          },
          discovery_summary: { type: 'string' },
        },
        required: ['exemplars_analyzed', 'patterns_discovered', 'discovery_summary'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!Array.isArray(discoveryResult?.patterns_discovered) || discoveryResult.patterns_discovered.length === 0) {
    log('Pattern discovery failed or no patterns found');
    return { success: false, reason: 'discovery_failed' };
  }

  discoveredPatterns.push(...discoveryResult.patterns_discovered);
  log(`Discovered ${discoveredPatterns.length} patterns:`);
  for (const pattern of discoveredPatterns) {
    log(`  - ${pattern.pattern_name} (${pattern.pattern_type}, ${pattern.expected_impact} impact)`);
  }

  // ============================================================================
  // Phase 3: Pattern Realization
  // ============================================================================
  phase('Pattern Realization');

  log('Realizing patterns as CUTLASS code transformations...');

  const realizedResults = [];
  for (const patternToRealize of discoveredPatterns) {
  let realizationResult = null;
  try {
  realizationResult = await agentRetry(() => agent(
    `Realize discovered patterns as concrete code transformations:

Target: ${kernelSpec.operation}
Architecture: ${setupResult.target_architecture}
Data types: ${(kernelSpec.dtypes || []).join(', ')}

Patterns to realize:
${JSON.stringify(patternToRealize)}
Return exactly one realized pattern for this pattern_id. Other discovered patterns are dependency names only.

Realization process:
1. For each pattern:
   a. Generate CUTLASS code template
   b. Define transformation parameters (e.g., tile sizes, thread counts)
   c. Specify applicability constraints
   d. Create dependency graph (which patterns can compose)
2. Index realized patterns in registry:
   T(rule_type, dtype, architecture) -> code_transformation
3. Validate realization:
   - Syntactic correctness
   - Type safety
   - Resource constraints (registers, shared memory)

Return JSON:
{
  "patterns_realized": [
    {
      "pattern_id": "same as discovery",
      "pattern_name": "same as discovery",
      "code_template": "CUTLASS code template with placeholders",
      "parameters": [
        {"name": "param1", "type": "int", "range": [min, max]},
        ...
      ],
      "constraints": "applicability constraints",
      "dependencies": ["pattern_id1", "pattern_id2", ...],
      "estimated_resource_usage": {
        "registers_per_thread": <int>,
        "shared_memory_bytes": <int>
      }
    },
    ...
  ],
  "realization_summary": "summary of realization process",
  "dependency_graph": "description of pattern dependencies"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${args.exp_dir}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Pattern Realization","ts":"<ts>","status":"done","technique":"pattern_realization","note":"<how many patterns realized as code transformations + key dependencies, one line>"}`,
    {
      label: `Realize pattern ${patternToRealize.pattern_id}`,
      phase: 'Pattern Realization',
      schema: {
        type: 'object',
        properties: {
          patterns_realized: {
            type: 'array', minItems: 1, maxItems: 1,
            items: {
              type: 'object',
              properties: {
                pattern_id: { type: 'string' },
                pattern_name: { type: 'string' },
                code_template: { type: 'string' },
                parameters: { type: 'array' },
                constraints: { type: 'string' },
                dependencies: { type: 'array', items: { type: 'string' } },
                estimated_resource_usage: { type: 'object' },
              },
              required: ['pattern_id', 'pattern_name', 'code_template'],
            },
          },
          realization_summary: { type: 'string' },
          dependency_graph: { type: 'string' },
        },
        required: ['patterns_realized', 'realization_summary'],
      },
    }
  ), { retries: 5, allowNull: true });

  } catch (error) {
    log(`Pattern realization failed for ${patternToRealize.pattern_id}: ${String(error.message || error).slice(0, 300)}`);
  }
  if (realizationResult?.patterns_realized?.length === 1) {
    realizedPatterns.push(realizationResult.patterns_realized[0]);
    realizedResults.push(realizationResult);
  }
  }
  if (!realizedPatterns.length) return { success: false, reason: 'realization_failed' };
  const realizationResult = {
    patterns_realized: realizedPatterns,
    dependency_graph: realizedResults.map(r => r.dependency_graph || '').filter(Boolean).join('\n'),
  };
  log(`Realized ${realizedPatterns.length} patterns as code transformations`);

  // ============================================================================
  // Phase 4: Pattern Composition
  // ============================================================================
  phase('Pattern Composition');

  log(`Composing patterns to generate optimized kernels (budget: ${compositionBudget})...`);

  // In embedded mode each composed kernel must be a complete dispatch-compatible
  // .cuh that matches the project's reference dispatch signature exactly. Gated on
  // the runtime IS_EMBEDDED (integration-strategist), not the static EMBEDDED arg.
  const compositionEmbeddingBlock = IS_EMBEDDED
    ? `\n\n${EMBEDDING_CONTRACT}\n\nMANDATORY (embedded): Read the reference dispatch file at ${REFERENCE_FILE} and match its dispatch signature EXACTLY (same entry-point shape, template params, launch-bounds conventions). Each composed candidate's kernel_code MUST be a COMPLETE dispatch-compatible \`.cuh\` (NOT a standalone translation unit, NO main()/harness/top-level test code). Use ONLY symbols/headers the project already provides; do not register, build, or benchmark the variant yourself.`
    : '';

  // The standalone branch needs the same explicitness.  Without it kernel_code
  // came back as a patch description against a sibling candidate - "rest
  // byte-identical to K001" - which cannot be compiled, packed or measured.
  const SEED_KERNEL = args.kernel_path || (SOL_SEED_DIR ? `${SOL_SEED_DIR}/kernel.cu` : '')
  const compositionStandaloneBlock = IS_EMBEDDED
    ? ''
    : `\n\nMANDATORY (standalone): each candidate's kernel_code MUST be a COMPLETE, self-contained translation unit that compiles on its own.${SEED_KERNEL ? ` Read the seed kernel at ${SEED_KERNEL} and keep its entry point and bindings intact - the same Python-visible \`run(...)\` or \`forward(...)\` signature and the same \`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\` block - changing only the implementation.` : ' Preserve the seed kernel\'s entry point and its PYBIND11_MODULE binding, changing only the implementation.'}
Emit every line of every candidate. Do NOT describe a candidate as a delta against another one, do NOT write "same as", "unchanged", "rest byte-identical", "..." or any other placeholder standing in for code. Each kernel_code is compiled exactly as given, so a description of a kernel fails where the kernel itself would have been measured.`;

  // A batch of full translation units can exhaust one structured tool response.
  // Keep the requested search count, but give each candidate its own receipt and
  // retry boundary. Carry only composition metadata forward, not repeated code.
  const __measured = [];
  async function measureComposedKernel(k, idx) {
    if (!SOL_AVAILABLE) return;
    const kid = k.kernel_id || `k${idx}`;
    const variant = `fact_${kid}`.replace(/[^A-Za-z0-9_]/g, '_');
    const r = await __solExecbenchEvaluate({
      label: `sol-eval-${variant}`, phase: 'Evaluation',
      substrateDir: SOL_SUBSTRATE_DIR,
      kernelSource: `${args.exp_dir}/${variant}.cu`, candidateSource: k.kernel_code || '',
      contractEnv: `${SOL_SEED_DIR || '.'}/contract.env`,
      solutionOut: `${args.exp_dir}/${variant}.solution.json`,
      benchOut: `${args.exp_dir}/${variant}.bench.jsonl`,
      solCli: SOL_CLI, taskDir: SOL_TASK_DIR, benchConfig: SOL_BENCH_CONFIG,
      seedDir: SOL_SEED_DIR, cudaVisibleDevices: SOL_CVD,
      ldLibraryPath: SOL_LD_LIBRARY_PATH, envPrefix: SOL_ENV_PREFIX,
      definitionPath: SOL_DEFINITION_PATH,
    });
    if (r) {
      __measured.push({ kernel_id: kid, ...r });
      log(`Host-measured ${kid}: compiled=${r.compiled} correct=${r.correct} speedup=${r.speedup}`);
    }
  }

  const compositionCount = Math.min(compositionBudget, 20);
  for (let compositionIndex = 0; compositionIndex < compositionCount; compositionIndex++) {
    const candidateId = `k${compositionIndex + 1}`;
    const priorCompositions = composedKernels.map(k => ({
      kernel_id: k.kernel_id,
      applied_patterns: k.applied_patterns,
      pattern_parameters: k.pattern_parameters,
      measurement: __measured.filter(m => m.kernel_id === k.kernel_id).map(m => ({
        compiled: m.compiled, correct: m.correct, speedup: m.speedup,
        failure_code: m.failure_code, diagnostics: String(m.stderr || '').slice(-2000),
      }))[0] || null,
    }));
    let compositionResult = null;
    try {
    compositionResult = await agentRetry(() => agent(
    `Compose patterns to generate optimized CUTLASS kernels:

Target specification:
- Operation: ${kernelSpec.operation}
- Shapes: ${kernelSpec.shapes}
- Data types: ${(kernelSpec.dtypes || []).join(', ')}
- Architecture: ${setupResult.target_architecture}

Available patterns: ${realizedPatterns.length}
Realized patterns (authoritative handoff; use these directly, no registry file is required):
${JSON.stringify(realizedPatterns)}
Dependency graph: ${JSON.stringify(realizationResult.dependency_graph)}
Composition budget: ${compositionBudget} kernel candidates

Composition strategies:
1. Greedy composition:
   - Start with baseline CUTLASS implementation
   - Iteratively add compatible patterns
   - Prioritize high-impact patterns
2. Dependency-aware composition:
   - Respect pattern dependencies from graph
   - Compose compatible pattern groups
3. Search-based composition:
   - Generate pattern combinations
   - Filter by constraints (resources, compatibility)
   - Sample diverse compositions
4. Parameter tuning:
   - For each composition, tune pattern parameters
   - Use heuristics or light autotuning

Generate exactly ONE complete kernel candidate: ${candidateId}, slot ${compositionIndex + 1} of ${compositionCount}.
Do not emit the other candidates in this response. Preserve the full source and bindings of this one candidate.
Avoid repeating an already accepted composition:
${JSON.stringify(priorCompositions)}

Return JSON:
{
  "composition_strategy": "greedy|dependency-aware|search-based",
  "candidates_generated": 1,
  "composed_kernels": [
    {
      "kernel_id": "${candidateId}",
      "applied_patterns": ["pattern_id1", "pattern_id2", ...],
      "pattern_parameters": {"pattern_id": {"param": value}, ...},
      "kernel_code": "complete CUTLASS kernel code",
      "estimated_performance": "performance estimate if available",
      "composition_rationale": "why these patterns were composed"
    }
  ],
  "composition_summary": "summary of composition process"
}${compositionEmbeddingBlock}${compositionStandaloneBlock}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${args.exp_dir}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Pattern Composition","ts":"<ts>","status":"done","technique":"<composition strategy, e.g. greedy|dependency-aware|search-based>","note":"<how many kernel candidates composed + patterns most frequently applied, one line>"}`,
    {
      label: `Compose patterns ${compositionIndex + 1}`,
      phase: 'Pattern Composition',
      schema: {
        type: 'object',
        properties: {
          composition_strategy: { type: 'string' },
          candidates_generated: { type: 'integer', const: 1 },
          composed_kernels: {
            type: 'array',
            minItems: 1,
            maxItems: 1,
            items: {
              type: 'object',
              properties: {
                kernel_id: { type: 'string', const: candidateId },
                applied_patterns: { type: 'array', items: { type: 'string' } },
                pattern_parameters: { type: 'object' },
                kernel_code: { type: 'string' },
                estimated_performance: { type: 'string' },
                composition_rationale: { type: 'string' },
              },
              required: ['kernel_id', 'applied_patterns', 'kernel_code'],
            },
          },
          composition_summary: { type: 'string' },
        },
        required: ['candidates_generated', 'composed_kernels', 'composition_summary'],
      },
    }
  ), { retries: 5, allowNull: true });

    } catch (error) {
      log(`Composition failed; retaining ${__measured.length} Host receipts: ${String(error.message || error).slice(0, 300)}`);
      if (__measured.length) break;
    }
    if (!compositionResult || compositionResult.composed_kernels?.length !== 1) {
      log(`Pattern composition ${candidateId} failed; retaining earlier candidates`);
      continue;
    }
    composedKernels.push(compositionResult.composed_kernels[0]);
    // Commit this candidate's evaluator receipt before asking for the next one.
    await measureComposedKernel(compositionResult.composed_kernels[0], compositionIndex);
    phase('Pattern Composition');
  }

  if (composedKernels.length === 0) {
    log('Pattern composition failed');
    return { success: false, reason: 'composition_failed' };
  }

  log(`Generated ${composedKernels.length} composed kernel candidates`);

  // ============================================================================
  // Phase 5: Ablation Studies
  // ============================================================================
  phase('Ablation');

  log('Running ablation studies to validate pattern contributions...');

  const ablationResult = await agentRetry(() => agent(
    `Run ablation studies on top composed kernels:

Top kernels: ${Math.min(composedKernels.length, 5)}

Ablation process:
1. Select top-performing kernels (by estimated performance)
2. For each kernel, create ablation variants:
   - Remove one pattern at a time (leave-one-out)
   - Remove pattern groups
   - Baseline (no patterns)
3. Execute ablation variants on ${setupResult.target_architecture}
4. Measure performance impact of each pattern
5. Identify critical patterns vs marginal patterns

Profiling-strategist selected method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'). If method==='native_profiler', you MAY run ncu for bottleneck evidence during ablation. If method==='perf_heuristic', derive memory-vs-compute-bound hints from benchmark throughput and tag them evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. If method==='static', reason from source only (confidence='hypothesized'). Never fabricate profiler counters.

Return JSON:
{
  "kernels_ablated": <int>,
  "ablation_results": [
    {
      "kernel_id": "original kernel id",
      "baseline_gflops": <float>,
      "pattern_contributions": [
        {
          "pattern_id": "pattern_id",
          "pattern_name": "pattern_name",
          "ablated_gflops": <float>,
          "contribution_pct": <float>,
          "criticality": "critical|important|marginal"
        },
        ...
      ]
    },
    ...
  ],
  "critical_patterns": ["pattern_id1", "pattern_id2", ...],
  "ablation_summary": "summary of ablation findings"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${args.exp_dir}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (status="done" if ablation completed, else "error"):
{"workflow":"${WORKFLOW_NAME}","phase":"Ablation","ts":"<ts>","status":"<done|error>","technique":"leave_one_out_ablation","note":"<critical vs marginal patterns identified, one line>"}`,
    {
      label: 'Ablation studies',
      phase: 'Ablation',
      schema: {
        type: 'object',
        properties: {
          kernels_ablated: { type: 'integer' },
          ablation_results: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                kernel_id: { type: 'string' },
                baseline_gflops: { type: 'number' },
                pattern_contributions: { type: 'array' },
              },
            },
          },
          critical_patterns: { type: 'array', items: { type: 'string' } },
          ablation_summary: { type: 'string' },
        },
        required: ['kernels_ablated', 'ablation_results', 'critical_patterns'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!ablationResult) {
    log('Ablation studies failed');
  } else {
  if (ablationResult) ablationResult.critical_patterns = Array.isArray(ablationResult.critical_patterns)
    ? ablationResult.critical_patterns : [];
  log(`Ablation complete: identified ${ablationResult?.critical_patterns?.length || 0} critical patterns`);
  }

  // ============================================================================
  // Phase 6: Evaluation
  // ============================================================================
  phase('Evaluation');

  log('Evaluating composed kernels on target hardware...');

  // Embedded evaluation: gated on the runtime IS_EMBEDDED (integration-strategist),
  // NOT the static EMBEDDED arg. embedded_dispatch registers each candidate into the
  // project via the adapter, builds/tests/benchmarks, then ALWAYS unregisters back to
  // pristine. embedded_inplace REPLACES the project kernel file in place (a pristine
  // backup was taken at Setup) with restore→apply→build→test→bench→restore per
  // candidate. Both replace the standalone CUTLASS compile path. FACT evaluates all
  // candidates in a SINGLE agent call (no `await parallel(`), so embedded eval is
  // already serial — no separate serial for-loop is needed.
  let evaluationEmbeddingBlock = '';
  if (IS_EMBEDDED) {
    const _embBclassLine = PROFILING_DECISION.method === 'perf_heuristic'
      ? `\nFor EACH candidate, also write heuristic_bclass (memory_bound|compute_bound|latency_bound) derived from the measured throughput so diagnose does not fall to unknown. Tag it evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.`
      : '';
    if (INTEGRATION_DECISION.method === 'embedded_dispatch') {
      const planBlocks = composedKernels.map((k, idx) => {
        const variantName = `fact_${k.kernel_id || ('k' + idx)}`.replace(/[^A-Za-z0-9_]/g, '_');
        const candidatePath = `${PROJECT_ROOT}/.fact_candidates/${variantName}.cuh`;
        const plan = __embeddedEvalPlan({
          adapter: 'python "' + REGISTER_SCRIPT + '"',
          variant: variantName,
          source: candidatePath,
          projectRoot: PROJECT_ROOT,
          params: REGISTER_PARAMS,
          buildCmd: BUILD_CMD,
          testCmd: TEST_CMD,
          benchmarkCmd: BENCHMARK_CMD,
        });
        return `### Candidate kernel_id=${k.kernel_id || ('k' + idx)} (variant ${variantName})
Write this candidate's kernel_code verbatim to ${candidatePath}, then run IN THIS EXACT ORDER:
1. Register:   ${plan.register}
2. List:       ${plan.list}   (CONFIRM ${variantName} is now listed; abort this candidate if absent)
3. Build:      ${plan.build}
4. Test:       ${plan.test}        (correctness)
5. Benchmark:  ${plan.benchmark}   (latency)
6. Unregister: ${plan.unregister}
7. List:       ${plan.list}   (CONFIRM ${variantName} is GONE)
HARD REQUIREMENT (cleanup invariant): ${plan.cleanupInvariant}`;
      }).join('\n\n');
      evaluationEmbeddingBlock = `

# EMBEDDED-DISPATCH EVALUATION (overrides the standalone CUTLASS compile/execute steps below)
These kernels are NOT standalone translation units; each is a dispatch-compatible \`.cuh\` that must be wired into the project at ${PROJECT_ROOT} via the register adapter. Do NOT attempt a standalone \`nvcc\`/CUTLASS compile. For EACH candidate below, run its commands in order, and ALWAYS run the unregister command and confirm removal via list even on build/correctness/benchmark FAILURE or non-improvement — never leave the project dirty.

${planBlocks}

Map per-candidate results into evaluation_results: compilation_success=build succeeded, correctness_passed=test passed, gflops/execution_time derived from the benchmark output, heuristic_bclass from throughput. Parse correctness (pass/fail) and latency STRICTLY from the actual test/benchmark command output. Do NOT fabricate numbers; if a value is not present in the output, report it as unavailable rather than guessing.${_embBclassLine}`;
    } else { // embedded_inplace
      const inplaceBlocks = composedKernels.map((k, idx) => {
        const variantName = `fact_${k.kernel_id || ('k' + idx)}`.replace(/[^A-Za-z0-9_]/g, '_');
        const candidatePath = `${PROJECT_ROOT}/.fact_candidates/${variantName}.cuh`;
        return `### Candidate kernel_id=${k.kernel_id || ('k' + idx)} (variant ${variantName})
Write this candidate's kernel_code verbatim to ${candidatePath}, then run IN THIS EXACT ORDER (project kernel: ${REFERENCE_FILE} | pristine backup: ${ORIGINAL_BACKUP}):
1. RESTORE pristine first (defensive): cp -a ${ORIGINAL_BACKUP} ${REFERENCE_FILE}
2. APPLY candidate in place: cp ${candidatePath} ${REFERENCE_FILE}
3. Build:      ${BUILD_CMD}
4. Test:       ${TEST_CMD}        (correctness)
5. Benchmark:  ${BENCHMARK_CMD}   (latency)
6. RESTORE pristine (HARD REQUIREMENT — run even on failure/non-improvement): cp -a ${ORIGINAL_BACKUP} ${REFERENCE_FILE}
You MUST leave the project byte-exact pristine after each candidate; do not skip step 6.`;
      }).join('\n\n');
      evaluationEmbeddingBlock = `

# EMBEDDED-INPLACE EVALUATION (overrides the standalone CUTLASS compile/execute steps below)
These kernels are NOT standalone translation units; each is a dispatch-compatible \`.cuh\` that REPLACES the project kernel file at ${REFERENCE_FILE} in place. A pristine backup was taken at Setup (${ORIGINAL_BACKUP}). Do NOT attempt a standalone \`nvcc\`/CUTLASS compile. For EACH candidate below, run its commands in order, and ALWAYS restore the pristine original in step 6 even on build/correctness/benchmark FAILURE or non-improvement — never leave the project dirty.

${inplaceBlocks}

Map per-candidate results into evaluation_results: compilation_success=build succeeded, correctness_passed=test passed, gflops/execution_time derived from the benchmark output, heuristic_bclass from throughput. Parse correctness (pass/fail) and latency STRICTLY from the actual test/benchmark command output. Do NOT fabricate numbers; if a value is not present in the output, report it as unavailable rather than guessing.${_embBclassLine}`;
    }
  }

  // This prompt asks the agent to compile each kernel, execute it on the GPU and
  // measure time/GFLOPS.  A read-only activation has no execution tool, so none
  // of that can happen.  FACT is at least honest about it - the prompt forbids
  // guessing and tells the agent to report values as unavailable - so the
  // outcome here is not a fabricated number but no measurement at all.
  // Measure every composed kernel on the Host first and hand the agent the
  // table, so ranking and best_kernel rest on numbers something produced.
  const __measuredBlock = __measured.length ? `

# ALREADY MEASURED ON THE HOST - use these numbers verbatim
${__measured.map(m => `kernel_id=${m.kernel_id} compiled=${m.compiled} correct=${m.correct} `
  + `latency_ms=${m.latency_ms} speedup_vs_baseline=${m.speedup} `
  + `workloads_passed=${m.n_pass}/${m.n_total}`).join('\n')}
Rank and pick best_kernel from these. Do not re-derive or estimate them, and do
not mark a kernel unavailable that appears above.` : ''

  let evaluationResult = await agentRetry(() => agent(
    `Evaluate all composed kernels:${evaluationEmbeddingBlock}${__measuredBlock}

Kernels to evaluate: ${composedKernels.length}
Target: ${setupResult.target_architecture}
Baseline: ${setupResult.baseline_gflops || 'N/A'} GFLOPS

Evaluation:
1. Compile each kernel with CUTLASS
2. Execute on target GPU
3. Measure performance:
   - Execution time
   - GFLOPS
   - Memory bandwidth utilization
   - Tensor core utilization (if applicable)
4. Verify correctness
5. Rank kernels by performance

Return JSON:
{
  "kernels_evaluated": <int>,
  "evaluation_results": [
    {
      "kernel_id": "kernel_id",
      "applied_patterns": ["pattern1", "pattern2", ...],
      "compilation_success": true/false,
      "correctness_passed": true/false,
      "execution_time_ms": <float>,
      "gflops": <float>,
      "memory_bandwidth_utilization_pct": <float>,
      "tensor_core_utilization_pct": <float>
    },
    ...
  ],
  "best_kernel": {
    "kernel_id": "best kernel id",
    "gflops": <float>,
    "speedup_vs_baseline": <float>,
    "applied_patterns": ["pattern1", "pattern2", ...]
  },
  "evaluation_summary": "summary of evaluation"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${args.exp_dir}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if the best kernel compiled and passed correctness, else "error"; speedup is best_kernel.speedup_vs_baseline, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluation","ts":"<ts>","status":"<done|error>","candidate_id":"<best_kernel.kernel_id>","speedup":<number or null>,"technique":"<patterns applied in best kernel>","note":"<best GFLOPS + how many kernels evaluated; or the failure reason>"}`,
    {
      label: 'Evaluate kernels',
      phase: 'Evaluation',
      schema: {
        type: 'object',
        properties: {
          kernels_evaluated: { type: 'integer' },
          evaluation_results: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                kernel_id: { type: 'string' },
                applied_patterns: { type: 'array', items: { type: 'string' } },
                compilation_success: { type: 'boolean' },
                correctness_passed: { type: 'boolean' },
                gflops: { type: 'number' },
              },
            },
          },
          best_kernel: { type: 'object' },
          evaluation_summary: { type: 'string' },
        },
        required: ['kernels_evaluated', 'evaluation_results', 'best_kernel'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (SOL_AVAILABLE && __measured.length) {
    const valid = __measured.filter(m => m.measurement_valid !== false && m.compiled === true && m.correct === true && m.n_total > 0 && m.n_pass === m.n_total);
    const best = valid.reduce((a, b) => !a || b.speedup > a.speedup ? b : a, null);
    evaluationResult = { ...(evaluationResult || {}), kernels_evaluated: __measured.length,
      evaluation_results: __measured, best_kernel: best ? { ...best, speedup_vs_baseline: best.speedup } : null };
  }
  if (!evaluationResult) {
    log('Evaluation failed');
    return { success: false, reason: 'evaluation_failed' };
  }

  log(`Evaluated ${evaluationResult.kernels_evaluated} kernels`);
  // best_kernel and its gflops come from a model-shaped evaluation result, so
  // neither is guaranteed.  Observed on B300: gflops was null and the run died
  // with "Cannot read properties of null (reading 'toFixed')" after 1.0M tokens,
  // losing everything the run had already learned.
  const _bestGflops = Number(evaluationResult?.best_kernel?.gflops)
  const _bestGflopsText = Number.isFinite(_bestGflops) ? __fmt(_bestGflops, 2) : 'unmeasured'
  log(`Best: ${_bestGflopsText} GFLOPS (${evaluationResult?.best_kernel?.speedup_vs_baseline ?? 'unknown'}x speedup)`);

  // ============================================================================
  // Phase 7: Report
  // ============================================================================
  phase('Report');

  const report = await agentRetry(() => agent(
    `Generate FACT compositional synthesis report:

Summary:
- Target: ${kernelSpec.operation} on ${setupResult.target_architecture}
- Patterns discovered: ${discoveredPatterns.length}
- Patterns realized: ${realizedPatterns.length}
- Kernels composed: ${composedKernels.length}
- Kernels evaluated: ${evaluationResult.kernels_evaluated}
- Best performance: ${_bestGflopsText} GFLOPS
- Speedup: ${evaluationResult.best_kernel.speedup_vs_baseline}x

Critical patterns: ${ablationResult?.critical_patterns.join(', ') || 'N/A'}

Generate report with:
1. Executive summary
2. Pattern discovery analysis
3. Pattern realization details
4. Composition strategy
5. Ablation study results
6. Performance evaluation
7. Best kernel breakdown

Return JSON:
{
  "summary": "brief summary",
  "patterns_discovered": ${discoveredPatterns.length},
  "patterns_realized": ${realizedPatterns.length},
  "kernels_composed": ${composedKernels.length},
  "best_gflops": ${evaluationResult.best_kernel.gflops},
  "speedup": ${evaluationResult.best_kernel.speedup_vs_baseline},
  "critical_patterns": ${JSON.stringify(ablationResult?.critical_patterns || [])},
  "report_path": "path/to/report.md"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${args.exp_dir}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (speedup is the best speedup vs baseline, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","speedup":<number or null>,"technique":"compositional_pattern_synthesis","note":"<one-line headline: best GFLOPS + critical patterns>"}`,
    {
      label: 'Generate report',
      phase: 'Report',
      schema: {
        type: 'object',
        properties: {
          summary: { type: 'string' },
          patterns_discovered: { type: 'integer' },
          patterns_realized: { type: 'integer' },
          kernels_composed: { type: 'integer' },
          best_gflops: { type: 'number' },
          speedup: { type: 'number' },
          critical_patterns: { type: 'array', items: { type: 'string' } },
          report_path: { type: 'string' },
        },
        required: ['summary', 'best_gflops'],
      },
    }
  ), { retries: 5, allowNull: true });

  // ============================================================================
  // Return final results
  // ============================================================================

  // embedded_inplace exit safety net: unconditionally restore the pristine
  // original so the project is left byte-exact regardless of how the workflow
  // terminated. (embedded_dispatch is non-mutating — adapter unregister is the
  // reversibility — so it is exempt.)
  if (ORIGINAL_BACKUP && _integProbeKernel) {
    await agentRetry(() => agent(`Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${_integProbeKernel}"\` and confirm.`,
      { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
  }

  return {
    success: true,
    method: 'FACT',
    approach: 'Compositional pattern synthesis',
    kernel: kernelSpec.operation,
    target_architecture: setupResult.target_architecture,
    cutlass_version: setupResult.cutlass_version,
    patterns_discovered: discoveredPatterns.length,
    patterns_realized: realizedPatterns.length,
    kernels_composed: composedKernels.length,
    kernels_evaluated: evaluationResult.kernels_evaluated,
    baseline_gflops: setupResult.baseline_gflops,
    best_gflops: evaluationResult.best_kernel.gflops,
    speedup: evaluationResult.best_kernel.speedup_vs_baseline,
    best_kernel_id: evaluationResult.best_kernel.kernel_id,
    best_kernel_patterns: evaluationResult.best_kernel.applied_patterns,
    critical_patterns: ablationResult?.critical_patterns || [],
    pattern_registry: patternRegistry,
    report: report?.report_path,
    summary: report?.summary,
  };
}

// Execute the workflow
return await main();
