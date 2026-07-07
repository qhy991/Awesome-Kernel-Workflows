export const meta = {
  name: 'xe-forge-kernel-optimization',
  description: 'Multi-stage CoVeR optimization for Intel XPU kernels',
  whenToUse: 'Use for optimizing kernels on Intel XPUs (Data Center GPU Max) with Triton and SYCL',
  phases: [
    { title: 'Setup', detail: 'Initialize Intel XPU environment and kernel specification' },
    { title: 'Generate Initial', detail: 'Generate initial Triton implementations' },
    { title: 'Analyze', detail: 'Analyze performance bottlenecks with profiling' },
    { title: 'Plan', detail: 'Plan optimization strategies based on analysis' },
    { title: 'Optimize', detail: 'Apply optimizations and generate variants' },
    { title: 'Verify', detail: 'Verify correctness and performance gains' },
    { title: 'Refine', detail: 'Iterative refinement with CoVeR cycles' },
    { title: 'Report', detail: 'Generate optimization report' },
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
// __modelTierApplied
// --- END model-tier ---

const WORKFLOW_NAME = 'xe-forge-kernel-optimization'

// --- shared profiling-strategist plumbing (XPU substrate manifest; the existing
// generate/refine prompts below honor the strategist's decision). The agent only
// CLASSIFIES the task (fuzzy op_class/size); the substrate DETERMINISTICALLY
// picks the method and STAMPS confidence by method (measured/inferred/
// hypothesized) -- the model must NOT assign confidence itself. See
// _substrate/profiling/README.md. Defaults to native_profiler so happy-path
// VTune behavior is unchanged if the decision is ignored. ---
const SUBSTRATE = args.substrate_dir || '_substrate'
const PY = args.substrate_command_prefix || ''
const BACKEND_MANIFEST = args.backend_manifest || `${SUBSTRATE}/backends/xpu/manifest.json`
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

const EXPDIR = args.exp_dir || '.'


// Xe-Forge: Multi-stage CoVeR (Chain-of-Verification-Refinement) for Intel XPU
// Based on arXiv:2605.26118 (Intel Labs)
// Supports Triton and SYCL backends

async function main() {
  // ============================================================================
  // Phase 1: Setup
  // ============================================================================
  phase('Setup');

  const setupResult = await agentRetry(() => agent(
    `Set up Xe-Forge optimization environment for Intel XPU:

1. Verify Intel XPU availability (Data Center GPU Max)
2. Check backend support:
   - Intel Triton compiler
   - Intel SYCL compiler (DPC++)
3. Identify target kernel specification:
   - Operation type (GEMM, conv, attention, etc.)
   - Input/output shapes
   - Data types
   - Performance baseline (if available)
4. Configure CoVeR parameters:
   - Number of CoVeR cycles
   - Verification strategies
   - Refinement depth
5. Set up profiling tools:
   - onemkl-sycl-bench
   - VTune profiler
   - Custom XPU profilers

Return JSON:
{
  "xpu_available": true/false,
  "xpu_model": "Data Center GPU Max 1550|...",
  "backends": ["triton", "sycl"],
  "kernel_spec": {
    "operation": "gemm|conv2d|attention|...",
    "shapes": "shape description",
    "dtypes": ["float32", "bfloat16", ...],
    "baseline_gflops": <float or null>
  },
  "cover_cycles": <int>,
  "verification_strategies": ["correctness", "performance", "numerical"],
  "profiling_tools": ["tool1", "tool2", ...],
  "target_backend": "triton|sycl"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"xpu_env_setup","note":"<xpu model + target backend + kernel operation + cover cycles, one line>"}`,
    {
      label: 'Setup Xe-Forge',
      phase: 'Setup',
      schema: {
        type: 'object',
        properties: {
          xpu_available: { type: 'boolean' },
          xpu_model: { type: 'string' },
          backends: { type: 'array', items: { type: 'string' } },
          kernel_spec: { type: 'object' },
          cover_cycles: { type: 'integer' },
          verification_strategies: { type: 'array', items: { type: 'string' } },
          profiling_tools: { type: 'array', items: { type: 'string' } },
          target_backend: { type: 'string' },
        },
        required: ['xpu_available', 'kernel_spec', 'target_backend'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!setupResult || !setupResult.xpu_available) {
    log('Intel XPU not available or setup failed');
    return { success: false, reason: 'xpu_unavailable' };
  }

  log(`Target: ${setupResult.kernel_spec.operation} on ${setupResult.xpu_model}`);
  log(`Backend: ${setupResult.target_backend}`);
  log(`CoVeR cycles: ${setupResult.cover_cycles || 3}`);

  const targetBackend = setupResult.target_backend;
  const coverCycles = setupResult.cover_cycles || 3;
  const kernelSpec = setupResult.kernel_spec;

  // --- profiling-strategist: classify the kernel task (fuzzy op_class/size) and
  // resolve a profiling METHOD against the XPU substrate manifest (VTune native
  // when vtune is on host, else perf_heuristic inferred). The agent only
  // CLASSIFIES; the substrate DETERMINISTICALLY picks the method and STAMPS
  // confidence. Honored in the generate/refine prompt below; defaults keep the
  // happy-path VTune behavior unchanged if the decision is ignored. ---
  {
    const _pd = await agentRetry(() => agent(
      `Classify the kernel under optimization. Operation: "${kernelSpec.operation}" (shapes: ${kernelSpec.shapes}).\n` +
      `Pick op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
      substrateInstruction('profiling/profiling_strategist.py',
        `resolve --backend-manifest ${BACKEND_MANIFEST} --task <op_class> --size <size> --cache ${EXPDIR}/prof_cache.json --trajectory ${EXPDIR}/genome.jsonl`) +
      ` Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
      { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
    if (_pd && _pd.method) PROFILING_DECISION = _pd
  }
  log(`Profiling method: ${PROFILING_DECISION.method} (confidence=${PROFILING_DECISION.confidence})`)

  // Track optimization history
  const optimizationHistory = [];
  let currentImplementation = null;
  let bestImplementation = null;
  let bestPerformance = kernelSpec.baseline_gflops || 0;

  // ============================================================================
  // Phase 2: Generate Initial Implementation
  // ============================================================================
  phase('Generate Initial');

  log('Generating initial kernel implementation...');

  const initialResult = await agentRetry(() => agent(
    `Generate initial ${targetBackend} implementation:

Kernel specification:
- Operation: ${kernelSpec.operation}
- Shapes: ${kernelSpec.shapes}
- Data types: ${kernelSpec.dtypes.join(', ')}

Backend: ${targetBackend}

For Triton:
- Use Intel Triton syntax
- Start with naive tiling strategy
- Basic memory coalescing

For SYCL:
- Use Intel DPC++ syntax
- ND-range kernels with work-group sizing
- Basic sub-group operations

Generate:
1. Complete kernel implementation
2. Host code for launching
3. Correctness test

Return JSON:
{
  "backend": "${targetBackend}",
  "kernel_code": "complete kernel code",
  "host_code": "host launch code",
  "test_code": "correctness test",
  "initial_strategy": "description of initial approach"
}

Profiling-strategist selected method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}', profiler='${PROFILING_DECISION.profiler_name}'). If method==='native_profiler', you MAY run VTune for bottleneck evidence. If method==='perf_heuristic', derive memory-vs-compute-bound hints from benchmark throughput and tag them evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'. If method==='static', reason from source only (confidence='hypothesized'). Never fabricate profiler counters.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Generate Initial","ts":"<ts>","status":"done","candidate_id":"initial","technique":"<initial tiling/strategy approach>","speedup":null,"note":"<backend + initial approach summary, one line>"}`,
    {
      label: 'Generate initial impl',
      phase: 'Generate Initial',
      schema: {
        type: 'object',
        properties: {
          backend: { type: 'string' },
          kernel_code: { type: 'string' },
          host_code: { type: 'string' },
          test_code: { type: 'string' },
          initial_strategy: { type: 'string' },
        },
        required: ['backend', 'kernel_code', 'initial_strategy'],
      },
    }
  ), { retries: 5, allowNull: true });

  if (!initialResult) {
    log('Failed to generate initial implementation');
    return { success: false, reason: 'generation_failed' };
  }

  currentImplementation = initialResult;
  log(`Initial implementation: ${initialResult.initial_strategy}`);

  // ============================================================================
  // CoVeR Cycles
  // ============================================================================

  for (let cycle = 0; cycle < coverCycles; cycle++) {
    log(`\n=== CoVeR Cycle ${cycle + 1}/${coverCycles} ===`);

    // ==========================================================================
    // Phase 3: Analyze
    // ==========================================================================
    phase('Analyze');

    log('Analyzing performance bottlenecks...');

    const analysisResult = await agentRetry(() => agent(
      `Analyze current kernel implementation (Cycle ${cycle + 1}):

Backend: ${targetBackend}
Current kernel:
\`\`\`${targetBackend}
${currentImplementation.kernel_code.substring(0, 3000)}${currentImplementation.kernel_code.length > 3000 ? '\n... (truncated)' : ''}
\`\`\`

Profiling analysis:
1. Execute kernel on Intel XPU
2. Profile with available tools: ${setupResult.profiling_tools.join(', ')}
3. Measure metrics:
   - Execution time
   - GFLOPS achieved
   - Memory bandwidth utilization
   - EU (Execution Unit) occupancy
   - Cache hit rates
   - Register pressure
4. Identify bottlenecks:
   - Memory-bound vs compute-bound
   - Memory access patterns
   - Thread divergence
   - Synchronization overhead

Return JSON:
{
  "cycle": ${cycle + 1},
  "execution_time_ms": <float>,
  "gflops": <float>,
  "memory_bandwidth_utilization_pct": <float>,
  "eu_occupancy_pct": <float>,
  "cache_hit_rate_pct": <float>,
  "bottleneck_type": "memory|compute|latency|sync",
  "bottleneck_details": "detailed bottleneck description",
  "optimization_opportunities": [
    "opportunity1",
    "opportunity2",
    ...
  ]
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured:
{"workflow":"${WORKFLOW_NAME}","phase":"Analyze","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle + 1}","technique":"profiling","speedup":null,"note":"<measured gflops + bottleneck type + main optimization opportunity, one line>"}`,
      {
        label: `Analyze cycle ${cycle + 1}`,
        phase: 'Analyze',
        schema: {
          type: 'object',
          properties: {
            cycle: { type: 'integer' },
            execution_time_ms: { type: 'number' },
            gflops: { type: 'number' },
            memory_bandwidth_utilization_pct: { type: 'number' },
            eu_occupancy_pct: { type: 'number' },
            cache_hit_rate_pct: { type: 'number' },
            bottleneck_type: { type: 'string' },
            bottleneck_details: { type: 'string' },
            optimization_opportunities: { type: 'array', items: { type: 'string' } },
          },
          required: ['cycle', 'gflops', 'bottleneck_type', 'optimization_opportunities'],
        },
      }
    ), { retries: 5, allowNull: true });

    if (!analysisResult) {
      log('Analysis failed, stopping CoVeR cycles');
      break;
    }

    log(`Performance: ${analysisResult.gflops.toFixed(2)} GFLOPS`);
    log(`Bottleneck: ${analysisResult.bottleneck_type} - ${analysisResult.bottleneck_details}`);

    // Update best
    if (analysisResult.gflops > bestPerformance) {
      bestPerformance = analysisResult.gflops;
      bestImplementation = { ...currentImplementation, analysis: analysisResult };
    }

    optimizationHistory.push({
      cycle: cycle + 1,
      gflops: analysisResult.gflops,
      bottleneck: analysisResult.bottleneck_type,
    });

    // ==========================================================================
    // Phase 4: Plan
    // ==========================================================================
    phase('Plan');

    log('Planning optimization strategies...');

    const planResult = await agentRetry(() => agent(
      `Plan optimizations based on analysis (Cycle ${cycle + 1}):

Current performance: ${analysisResult.gflops.toFixed(2)} GFLOPS
Bottleneck: ${analysisResult.bottleneck_type}
Details: ${analysisResult.bottleneck_details}

Optimization opportunities:
${analysisResult.optimization_opportunities.map((opp, idx) => `${idx + 1}. ${opp}`).join('\n')}

Backend: ${targetBackend}

Intel XPU-specific optimizations:
For Triton:
- Tile size tuning (block size, warps per block)
- Memory layout optimization (swizzling, padding)
- Pipeline optimization (software pipelining, async loads)
- Xe Matrix Extensions (XMX) utilization for matrix ops
- Sub-group operations

For SYCL:
- Work-group and sub-group sizing
- Joint matrix (XMX) APIs
- Memory optimizations (SLM, sub-group shuffle)
- Vectorization (vec<T,N>)
- Kernel fusion

Select top 3 optimization strategies to apply:

Return JSON:
{
  "cycle": ${cycle + 1},
  "strategies": [
    {
      "name": "strategy name",
      "description": "detailed description",
      "expected_impact": "high|medium|low",
      "implementation_approach": "how to implement"
    },
    ...
  ],
  "rationale": "why these strategies were chosen"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Plan","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle + 1}","technique":"<top selected strategy name>","note":"<chosen strategies + rationale, one line>"}`,
      {
        label: `Plan cycle ${cycle + 1}`,
        phase: 'Plan',
        schema: {
          type: 'object',
          properties: {
            cycle: { type: 'integer' },
            strategies: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  name: { type: 'string' },
                  description: { type: 'string' },
                  expected_impact: { type: 'string' },
                  implementation_approach: { type: 'string' },
                },
                required: ['name', 'description', 'expected_impact'],
              },
            },
            rationale: { type: 'string' },
          },
          required: ['cycle', 'strategies', 'rationale'],
        },
      }
    ), { retries: 5, allowNull: true });

    if (!planResult || planResult.strategies.length === 0) {
      log('Planning failed, stopping CoVeR cycles');
      break;
    }

    log(`Planned ${planResult.strategies.length} optimization strategies:`);
    for (const strategy of planResult.strategies) {
      log(`  - ${strategy.name} (${strategy.expected_impact} impact)`);
    }

    // ==========================================================================
    // Phase 5: Optimize
    // ==========================================================================
    phase('Optimize');

    log('Applying optimizations...');

    const optimizeResult = await agentRetry(() => agent(
      `Apply planned optimizations (Cycle ${cycle + 1}):

Current implementation:
\`\`\`${targetBackend}
${currentImplementation.kernel_code.substring(0, 2000)}...
\`\`\`

Strategies to apply:
${planResult.strategies.map((s, idx) => `${idx + 1}. ${s.name}: ${s.implementation_approach}`).join('\n')}

Generate optimized implementation:
1. Apply each strategy incrementally
2. Preserve correctness
3. Add comments explaining optimizations
4. Generate multiple variants if strategies are independent

Return JSON:
{
  "cycle": ${cycle + 1},
  "optimized_kernel_code": "complete optimized kernel",
  "host_code": "updated host code if needed",
  "changes_applied": [
    "change1",
    "change2",
    ...
  ],
  "optimization_summary": "summary of applied optimizations"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Optimize","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle + 1}","technique":"<main optimization applied this cycle>","note":"<changes applied this cycle, one line>"}`,
      {
        label: `Optimize cycle ${cycle + 1}`,
        phase: 'Optimize',
        schema: {
          type: 'object',
          properties: {
            cycle: { type: 'integer' },
            optimized_kernel_code: { type: 'string' },
            host_code: { type: 'string' },
            changes_applied: { type: 'array', items: { type: 'string' } },
            optimization_summary: { type: 'string' },
          },
          required: ['cycle', 'optimized_kernel_code', 'changes_applied'],
        },
      }
    ), { retries: 5, allowNull: true });

    if (!optimizeResult) {
      log('Optimization failed, stopping CoVeR cycles');
      break;
    }

    log(`Applied ${optimizeResult.changes_applied.length} optimizations`);

    // ==========================================================================
    // Phase 6: Verify
    // ==========================================================================
    phase('Verify');

    log('Verifying optimized implementation...');

    const verifyResult = await agentRetry(() => agent(
      `Verify optimized implementation (Cycle ${cycle + 1}):

Optimized kernel:
\`\`\`${targetBackend}
${optimizeResult.optimized_kernel_code.substring(0, 2000)}...
\`\`\`

Verification:
1. Correctness check:
   - Run test cases
   - Compare outputs with reference
   - Check numerical accuracy (relative/absolute error)
2. Performance check:
   - Execute on Intel XPU
   - Measure GFLOPS
   - Compare with previous cycle
3. Constraint checks:
   - Resource usage (registers, SLM)
   - No illegal operations

Return JSON:
{
  "cycle": ${cycle + 1},
  "correctness_passed": true/false,
  "correctness_errors": ["error1", ...],
  "max_relative_error": <float>,
  "performance_gflops": <float>,
  "performance_improvement": <float>,
  "resource_usage": {
    "registers_per_thread": <int>,
    "slm_bytes": <int>
  },
  "verification_passed": true/false,
  "notes": "verification notes"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if correctness passed, else "error"; speedup is the measured performance_improvement as a multiplier, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Verify","ts":"<ts>","status":"<done|error>","candidate_id":"cycle-${cycle + 1}","speedup":<number or null>,"technique":"<technique under test>","note":"<correct? gflops + improvement pct; or the failure reason>"}`,
      {
        label: `Verify cycle ${cycle + 1}`,
        phase: 'Verify',
        schema: {
          type: 'object',
          properties: {
            cycle: { type: 'integer' },
            correctness_passed: { type: 'boolean' },
            correctness_errors: { type: 'array', items: { type: 'string' } },
            max_relative_error: { type: 'number' },
            performance_gflops: { type: 'number' },
            performance_improvement: { type: 'number' },
            resource_usage: { type: 'object' },
            verification_passed: { type: 'boolean' },
            notes: { type: 'string' },
          },
          required: ['cycle', 'correctness_passed', 'performance_gflops', 'verification_passed'],
        },
      }
    ), { retries: 5, allowNull: true });

    if (!verifyResult) {
      log('Verification failed, stopping CoVeR cycles');
      break;
    }

    if (!verifyResult.verification_passed) {
      log(`Verification failed: ${verifyResult.notes}`);
      log('Keeping previous implementation');
      continue;
    }

    log(`Verification passed: ${verifyResult.performance_gflops.toFixed(2)} GFLOPS (${verifyResult.performance_improvement > 0 ? '+' : ''}${(verifyResult.performance_improvement * 100).toFixed(1)}%)`);

    // Update current implementation
    currentImplementation = {
      backend: targetBackend,
      kernel_code: optimizeResult.optimized_kernel_code,
      host_code: optimizeResult.host_code || currentImplementation.host_code,
      verification: verifyResult,
    };

    // Update best
    if (verifyResult.performance_gflops > bestPerformance) {
      bestPerformance = verifyResult.performance_gflops;
      bestImplementation = { ...currentImplementation };
    }

    // ==========================================================================
    // Phase 7: Refine (decision to continue)
    // ==========================================================================
    phase('Refine');

    if (cycle < coverCycles - 1) {
      const refineDecision = await agentRetry(() => agent(
        `Decide whether to continue CoVeR cycles (Cycle ${cycle + 1}):

Current performance: ${verifyResult.performance_gflops.toFixed(2)} GFLOPS
Improvement this cycle: ${(verifyResult.performance_improvement * 100).toFixed(1)}%
Remaining cycles: ${coverCycles - cycle - 1}

Termination criteria:
- Improvement < 5% (diminishing returns)
- Resource limits reached
- No more optimization opportunities

Should we continue with another CoVeR cycle?

Return JSON:
{
  "continue": true/false,
  "reason": "reason for decision"
}`,
        {
          label: 'Refine decision',
          phase: 'Refine',
          schema: {
            type: 'object',
            properties: {
              continue: { type: 'boolean' },
              reason: { type: 'string' },
            },
            required: ['continue', 'reason'],
          },
        }
      ), { retries: 5, allowNull: true });

      if (refineDecision && !refineDecision.continue) {
        log(`Early termination: ${refineDecision.reason}`);
        break;
      }
    }
  }

  // ============================================================================
  // Phase 8: Report
  // ============================================================================
  phase('Report');

  const report = await agentRetry(() => agent(
    `Generate Xe-Forge optimization report:

Kernel: ${kernelSpec.operation}
Target: ${setupResult.xpu_model}
Backend: ${targetBackend}
CoVeR cycles: ${optimizationHistory.length}

Optimization trajectory:
${optimizationHistory.map(h => `  Cycle ${h.cycle}: ${h.gflops.toFixed(2)} GFLOPS (${h.bottleneck})`).join('\n')}

Final results:
- Best performance: ${bestPerformance.toFixed(2)} GFLOPS
- Baseline: ${kernelSpec.baseline_gflops || 'N/A'} GFLOPS
- Speedup: ${kernelSpec.baseline_gflops ? (bestPerformance / kernelSpec.baseline_gflops).toFixed(2) + 'x' : 'N/A'}

Generate report with:
1. Executive summary
2. CoVeR cycle analysis
3. Bottleneck evolution
4. Applied optimizations
5. Performance trajectory
6. Final kernel analysis
7. Intel XPU-specific insights

Return JSON:
{
  "summary": "brief summary",
  "cycles_completed": ${optimizationHistory.length},
  "best_gflops": ${bestPerformance},
  "baseline_gflops": ${kernelSpec.baseline_gflops || null},
  "speedup": ${kernelSpec.baseline_gflops ? bestPerformance / kernelSpec.baseline_gflops : null},
  "report_path": "path/to/report.md"
}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the final results (speedup is best_gflops over baseline_gflops, or null if no baseline):
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","technique":"final_report","speedup":<number or null>,"note":"<best gflops + cycles completed + speedup summary, one line>"}`,
    {
      label: 'Generate report',
      phase: 'Report',
      schema: {
        type: 'object',
        properties: {
          summary: { type: 'string' },
          cycles_completed: { type: 'integer' },
          best_gflops: { type: 'number' },
          baseline_gflops: { type: ['number', 'null'] },
          speedup: { type: ['number', 'null'] },
          report_path: { type: 'string' },
        },
        required: ['summary', 'cycles_completed', 'best_gflops'],
      },
    }
  ), { retries: 5, allowNull: true });

  // ============================================================================
  // Return final results
  // ============================================================================

  return {
    success: true,
    method: 'Xe-Forge',
    approach: 'CoVeR (Chain-of-Verification-Refinement)',
    kernel: kernelSpec.operation,
    xpu_target: setupResult.xpu_model,
    backend: targetBackend,
    cover_cycles: optimizationHistory.length,
    baseline_gflops: kernelSpec.baseline_gflops,
    best_gflops: bestPerformance,
    speedup: kernelSpec.baseline_gflops ? bestPerformance / kernelSpec.baseline_gflops : null,
    optimization_history: optimizationHistory,
    final_kernel: bestImplementation?.kernel_code,
    report: report?.report_path,
    summary: report?.summary,
  };
}

// Execute the workflow
return await main();
