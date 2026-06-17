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

// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.

const EXPDIR = args.exp_dir || '.'

const WORKFLOW_SUITABILITY = {
  supported_languages: ['triton', 'sycl', 'xpu'],
  supported_problem_types: ['xpu-kernel-optimization', 'triton-kernel-optimization'],
  problem_types: ['Intel XPU Triton/SYCL optimization', 'CoVeR staged refinement on Intel GPU hardware'],
  reason: 'Xe-Forge targets Intel XPU with Triton/SYCL tooling and is not meant for NVIDIA CUDA-only optimization.',
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
        `${meta.name} is not suitable for language="${args.language}". ` +
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
        `${meta.name} is not suitable for problem_type="${args.problem_type}". ` +
        `Supported problem types: ${WORKFLOW_SUITABILITY.supported_problem_types.join(', ')}. ` +
        `Typical use cases: ${WORKFLOW_SUITABILITY.problem_types.join('; ')}. ` +
        `Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }
  }
}

assertWorkflowSuitability()

// Xe-Forge: Multi-stage CoVeR (Chain-of-Verification-Refinement) for Intel XPU
// Based on arXiv:2605.26118 (Intel Labs)
// Supports Triton and SYCL backends

async function main() {
  // ============================================================================
  // Phase 1: Setup
  // ============================================================================
  phase('Setup');

  const setupResult = await agent(
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
{"workflow":"${meta.name}","phase":"Setup","ts":"<ts>","status":"done","technique":"xpu_env_setup","note":"<xpu model + target backend + kernel operation + cover cycles, one line>"}`,
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
  );

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

  const initialResult = await agent(
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

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXPDIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${meta.name}","phase":"Generate Initial","ts":"<ts>","status":"done","candidate_id":"initial","technique":"<initial tiling/strategy approach>","speedup":null,"note":"<backend + initial approach summary, one line>"}`,
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
  );

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

    const analysisResult = await agent(
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
{"workflow":"${meta.name}","phase":"Analyze","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle + 1}","technique":"profiling","speedup":null,"note":"<measured gflops + bottleneck type + main optimization opportunity, one line>"}`,
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
    );

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

    const planResult = await agent(
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
{"workflow":"${meta.name}","phase":"Plan","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle + 1}","technique":"<top selected strategy name>","note":"<chosen strategies + rationale, one line>"}`,
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
    );

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

    const optimizeResult = await agent(
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
{"workflow":"${meta.name}","phase":"Optimize","ts":"<ts>","status":"done","candidate_id":"cycle-${cycle + 1}","technique":"<main optimization applied this cycle>","note":"<changes applied this cycle, one line>"}`,
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
    );

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

    const verifyResult = await agent(
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
{"workflow":"${meta.name}","phase":"Verify","ts":"<ts>","status":"<done|error>","candidate_id":"cycle-${cycle + 1}","speedup":<number or null>,"technique":"<technique under test>","note":"<correct? gflops + improvement pct; or the failure reason>"}`,
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
    );

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
      const refineDecision = await agent(
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
      );

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

  const report = await agent(
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
{"workflow":"${meta.name}","phase":"Report","ts":"<ts>","status":"done","technique":"final_report","speedup":<number or null>,"note":"<best gflops + cycles completed + speedup summary, one line>"}`,
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
  );

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
