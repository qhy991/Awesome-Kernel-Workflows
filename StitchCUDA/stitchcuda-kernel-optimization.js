export const meta = {
  name: 'stitchcuda-kernel-optimization',
  description: 'Three-agent orchestration for CUDA kernel synthesis with adaptive replanning',
  whenToUse: 'Use for generating optimized CUDA kernels via Planner-Coder-Verifier orchestration',
  phases: [
    { title: 'Setup', detail: 'Initialize StitchCUDA environment and orchestrator' },
    { title: 'Plan', detail: 'Generate initial optimization plan' },
    { title: 'Code', detail: 'Generate CUDA kernel implementation' },
    { title: 'Verify', detail: 'Verify correctness and performance' },
    { title: 'Replan', detail: 'Adaptive replanning when needed' },
    { title: 'Report', detail: 'Generate synthesis report' },
  ],
};

// StitchCUDA: Three-agent orchestration for CUDA kernel synthesis
// Based on arXiv:2603.02637
// Planner → Coder → Verifier with adaptive replanning

async function main() {
  // ============================================================================
  // Phase 1: Setup
  // ============================================================================
  phase('Setup');

  const setupResult = await agent(
    `Set up StitchCUDA orchestration environment:

1. Initialize CUDA environment:
   - CUDA version and user-provided compiler/toolchain
   - Target GPU architecture (sm_80, sm_89, sm_90, etc.)
   - PyTorch load_inline integration
2. Configure KernelBench evaluation:
   - Benchmark suite selection
   - Performance metrics
   - Correctness test suite
3. Set up three-agent orchestration:
   - Planner: strategic optimization planning
   - Coder: kernel code generation
   - Verifier: correctness and performance verification
4. Configure adaptive replanning heuristics:
   - Replan trigger: N consecutive compile failures
   - Replan trigger: N consecutive correctness failures
   - Replan trigger: performance stagnation (M iterations)
5. Identify target kernel specification:
   - Operation type
   - Input/output shapes
   - Data types
   - Performance baseline

Return JSON:
{
  "cuda_version": "12.x|11.x",
  "target_architecture": "sm_80|sm_89|sm_90",
  "pytorch_available": true/false,
  "kernel_spec": {
    "operation": "gemm|conv2d|attention|reduce|...",
    "shapes": "shape description",
    "dtypes": ["float32", "float16", ...],
    "baseline_gflops": <float or null>
  },
  "kernelbench_config": {
    "benchmark_suite": "suite name",
    "metrics": ["correctness", "performance", ...]
  },
  "replan_heuristics": {
    "compile_failure_threshold": <int>,
    "correctness_failure_threshold": <int>,
    "stagnation_iterations": <int>
  },
  "max_attempts": <int>
}`,
    {
      label: 'Setup StitchCUDA',
      phase: 'Setup',
      schema: {
        type: 'object',
        properties: {
          cuda_version: { type: 'string' },
          target_architecture: { type: 'string' },
          pytorch_available: { type: 'boolean' },
          kernel_spec: { type: 'object' },
          kernelbench_config: { type: 'object' },
          replan_heuristics: { type: 'object' },
          max_attempts: { type: 'integer' },
        },
        required: ['cuda_version', 'target_architecture', 'kernel_spec'],
      },
    }
  );

  if (!setupResult) {
    log('Setup failed');
    return { success: false, reason: 'setup_failed' };
  }

  log(`Target: ${setupResult.kernel_spec.operation} on ${setupResult.target_architecture}`);
  log(`CUDA ${setupResult.cuda_version}, PyTorch: ${setupResult.pytorch_available ? 'Yes' : 'No'}`);
  log(`Max attempts: ${setupResult.max_attempts || 20}`);

  const kernelSpec = setupResult.kernel_spec;
  const maxAttempts = setupResult.max_attempts || 20;
  const replanHeuristics = setupResult.replan_heuristics || {
    compile_failure_threshold: 2,
    correctness_failure_threshold: 2,
    stagnation_iterations: 3,
  };

  // Orchestration state
  let currentPlan = null;
  let currentCode = null;
  let bestKernel = null;
  let bestPerformance = kernelSpec.baseline_gflops || 0;

  // Counters for replanning triggers
  let consecutiveCompileFailures = 0;
  let consecutiveCorrectnessFailures = 0;
  const performanceHistory = [];

  // Attempt loop
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    log(`\n=== Attempt ${attempt + 1}/${maxAttempts} ===`);

    // ==========================================================================
    // Replanning Decision
    // ==========================================================================

    const shouldReplan = (
      consecutiveCompileFailures >= replanHeuristics.compile_failure_threshold ||
      consecutiveCorrectnessFailures >= replanHeuristics.correctness_failure_threshold ||
      (performanceHistory.length >= replanHeuristics.stagnation_iterations &&
       isStagnant(performanceHistory, replanHeuristics.stagnation_iterations))
    );

    if (shouldReplan && attempt > 0) {
      log('Adaptive replanning triggered');
      phase('Replan');

      const replanResult = await agent(
        `Adaptive replanning triggered (Attempt ${attempt + 1}):

Current situation:
- Consecutive compile failures: ${consecutiveCompileFailures}
- Consecutive correctness failures: ${consecutiveCorrectnessFailures}
- Performance stagnation: ${isStagnant(performanceHistory, replanHeuristics.stagnation_iterations)}
- Recent performance: ${performanceHistory.slice(-3).map(p => p.toFixed(2)).join(' → ')} GFLOPS

Current plan summary:
${currentPlan?.plan_summary || 'No plan yet'}

Replanning strategy:
1. Diagnose root cause of failures:
   - Compile failures: likely syntax/API errors or invalid configurations
   - Correctness failures: likely algorithmic bugs or numerical issues
   - Stagnation: current approach hitting fundamental limits
2. Generate alternative approach:
   - Different optimization strategy
   - Different algorithm implementation
   - Different tiling/threading configuration
   - Fallback to simpler baseline if needed
3. Create new plan that avoids previous failure modes

Return JSON:
{
  "diagnosis": "root cause analysis",
  "alternative_approach": "description of new approach",
  "key_changes": ["change1", "change2", ...],
  "new_plan_summary": "summary of new plan"
}`,
        {
          label: 'Replan',
          phase: 'Replan',
          schema: {
            type: 'object',
            properties: {
              diagnosis: { type: 'string' },
              alternative_approach: { type: 'string' },
              key_changes: { type: 'array', items: { type: 'string' } },
              new_plan_summary: { type: 'string' },
            },
            required: ['diagnosis', 'alternative_approach', 'new_plan_summary'],
          },
        }
      );

      if (replanResult) {
        log(`Replanning: ${replanResult.alternative_approach}`);
        // Reset failure counters after replan
        consecutiveCompileFailures = 0;
        consecutiveCorrectnessFailures = 0;
      }
    }

    // ==========================================================================
    // Phase 2: Plan (Planner Agent)
    // ==========================================================================
    phase('Plan');

    const planContext = shouldReplan && attempt > 0
      ? `Replanned approach from previous attempt`
      : `Initial planning`;

    log(`Planner: ${planContext}...`);

    const planResult = await agent(
      `Generate optimization plan (Attempt ${attempt + 1}):

Kernel specification:
- Operation: ${kernelSpec.operation}
- Shapes: ${kernelSpec.shapes}
- Data types: ${kernelSpec.dtypes.join(', ')}
- Target: ${setupResult.target_architecture}

${shouldReplan && attempt > 0 ? `
Replanning context:
- Previous failures: compile=${consecutiveCompileFailures}, correctness=${consecutiveCorrectnessFailures}
- Performance history: ${performanceHistory.slice(-5).map(p => p.toFixed(2)).join(', ')} GFLOPS
` : ''}

Planning strategy:
1. High-level optimization approach:
   - Memory optimization strategy
   - Compute optimization strategy
   - Threading/block configuration
2. Decompose into implementation steps:
   - Data loading and layout
   - Core computation kernel
   - Memory hierarchy usage (shared memory, registers)
   - Output writing
3. Identify key optimizations:
   - Coalesced memory access
   - Shared memory usage
   - Register blocking
   - Instruction-level parallelism
   - Warp-level primitives
4. Specify constraints:
   - Resource limits (registers, shared memory)
   - Correctness requirements
   - Performance targets

Return JSON:
{
  "attempt": ${attempt + 1},
  "plan_summary": "high-level plan summary",
  "optimization_approach": "memory-bound|compute-bound|balanced",
  "key_strategies": [
    "strategy1",
    "strategy2",
    ...
  ],
  "implementation_steps": [
    {"step": 1, "description": "step description"},
    ...
  ],
  "threading_config": {
    "block_size": "block dimension",
    "grid_size": "grid dimension",
    "threads_per_block": <int>
  },
  "memory_strategy": "description of memory usage",
  "expected_bottleneck": "memory|compute|latency"
}`,
      {
        label: `Plan attempt ${attempt + 1}`,
        phase: 'Plan',
        schema: {
          type: 'object',
          properties: {
            attempt: { type: 'integer' },
            plan_summary: { type: 'string' },
            optimization_approach: { type: 'string' },
            key_strategies: { type: 'array', items: { type: 'string' } },
            implementation_steps: { type: 'array' },
            threading_config: { type: 'object' },
            memory_strategy: { type: 'string' },
            expected_bottleneck: { type: 'string' },
          },
          required: ['attempt', 'plan_summary', 'key_strategies', 'implementation_steps'],
        },
      }
    );

    if (!planResult) {
      log('Planning failed, skipping this attempt');
      continue;
    }

    currentPlan = planResult;
    log(`Plan: ${planResult.plan_summary}`);
    log(`Strategies: ${planResult.key_strategies.join(', ')}`);

    // ==========================================================================
    // Phase 3: Code (Coder Agent)
    // ==========================================================================
    phase('Code');

    log('Coder: Generating CUDA kernel...');

    const codeResult = await agent(
      `Generate CUDA kernel implementation (Attempt ${attempt + 1}):

Plan to implement:
${JSON.stringify(currentPlan, null, 2)}

Code generation:
1. Implement complete CUDA kernel following the plan
2. Use PyTorch load_inline compatible format:
   - __global__ kernel function
   - Template parameters if needed
   - Extern "C" wrapper if needed
3. Implement all steps from the plan:
${currentPlan.implementation_steps.map((s, idx) => `   ${idx + 1}. ${s.description}`).join('\n')}
4. Apply key optimizations:
${currentPlan.key_strategies.map((s, idx) => `   - ${s}`).join('\n')}
5. Include host launch code

Return JSON:
{
  "attempt": ${attempt + 1},
  "kernel_code": "complete CUDA kernel code",
  "host_code": "host launch code",
  "kernel_name": "kernel function name",
  "implementation_notes": "notes on implementation choices"
}`,
      {
        label: `Code attempt ${attempt + 1}`,
        phase: 'Code',
        schema: {
          type: 'object',
          properties: {
            attempt: { type: 'integer' },
            kernel_code: { type: 'string' },
            host_code: { type: 'string' },
            kernel_name: { type: 'string' },
            implementation_notes: { type: 'string' },
          },
          required: ['attempt', 'kernel_code', 'kernel_name'],
        },
      }
    );

    if (!codeResult) {
      log('Code generation failed, skipping this attempt');
      consecutiveCompileFailures++;
      continue;
    }

    currentCode = codeResult;
    log(`Generated kernel: ${codeResult.kernel_name}`);

    // ==========================================================================
    // Phase 4: Verify (Verifier Agent)
    // ==========================================================================
    phase('Verify');

    log('Verifier: Checking correctness and performance...');

    const verifyResult = await agent(
      `Verify CUDA kernel (Attempt ${attempt + 1}):

Kernel to verify:
\`\`\`cuda
${codeResult.kernel_code.substring(0, 2500)}${codeResult.kernel_code.length > 2500 ? '\n... (truncated)' : ''}
\`\`\`

Verification process:
1. Compile check:
   - Use the user-provided compile/build contract for ${setupResult.target_architecture}; if none is provided, perform static compileability review only.
   - Check for syntax errors, warnings
   - Verify resource usage (registers, shared memory)
2. Correctness check:
   - Run with test inputs
   - Compare with reference implementation
   - Check numerical accuracy (absolute/relative error)
   - Test edge cases
3. Performance check:
   - Benchmark on ${setupResult.target_architecture}
   - Measure execution time, GFLOPS
   - Profile with nsys/ncu if available
   - Compare with baseline: ${kernelSpec.baseline_gflops || 'N/A'} GFLOPS
4. KernelBench evaluation (if configured):
   - Run full benchmark suite
   - Aggregate scores

Return JSON:
{
  "attempt": ${attempt + 1},
  "compilation_success": true/false,
  "compilation_errors": ["error1", ...],
  "resource_usage": {
    "registers_per_thread": <int>,
    "shared_memory_bytes": <int>
  },
  "correctness_passed": true/false,
  "correctness_errors": ["error1", ...],
  "max_error": <float>,
  "performance_gflops": <float>,
  "execution_time_ms": <float>,
  "speedup_vs_baseline": <float>,
  "kernelbench_score": <float or null>,
  "verification_passed": true/false,
  "failure_reason": "compilation|correctness|performance|null"
}`,
      {
        label: `Verify attempt ${attempt + 1}`,
        phase: 'Verify',
        schema: {
          type: 'object',
          properties: {
            attempt: { type: 'integer' },
            compilation_success: { type: 'boolean' },
            compilation_errors: { type: 'array', items: { type: 'string' } },
            resource_usage: { type: 'object' },
            correctness_passed: { type: 'boolean' },
            correctness_errors: { type: 'array', items: { type: 'string' } },
            max_error: { type: 'number' },
            performance_gflops: { type: 'number' },
            execution_time_ms: { type: 'number' },
            speedup_vs_baseline: { type: 'number' },
            verification_passed: { type: 'boolean' },
            failure_reason: { type: ['string', 'null'] },
          },
          required: ['attempt', 'compilation_success', 'correctness_passed', 'verification_passed'],
        },
      }
    );

    if (!verifyResult) {
      log('Verification failed to run');
      consecutiveCompileFailures++;
      continue;
    }

    // Update counters based on verification result
    if (!verifyResult.compilation_success) {
      consecutiveCompileFailures++;
      consecutiveCorrectnessFailures = 0;
      log(`Compilation failed: ${verifyResult.compilation_errors.join(', ')}`);
      continue;
    } else {
      consecutiveCompileFailures = 0;
    }

    if (!verifyResult.correctness_passed) {
      consecutiveCorrectnessFailures++;
      log(`Correctness failed: ${verifyResult.correctness_errors.join(', ')}`);
      continue;
    } else {
      consecutiveCorrectnessFailures = 0;
    }

    // Verification fully passed
    log(`Verification passed: ${verifyResult.performance_gflops.toFixed(2)} GFLOPS (${verifyResult.speedup_vs_baseline.toFixed(2)}x)`);

    performanceHistory.push(verifyResult.performance_gflops);

    // Update best kernel
    if (verifyResult.performance_gflops > bestPerformance) {
      bestPerformance = verifyResult.performance_gflops;
      bestKernel = {
        attempt: attempt + 1,
        plan: currentPlan,
        code: currentCode,
        verification: verifyResult,
      };
      log(`New best kernel: ${bestPerformance.toFixed(2)} GFLOPS`);
    }

    // Early termination if very good performance achieved
    if (verifyResult.speedup_vs_baseline >= 2.0) {
      log('Excellent performance achieved, early termination');
      break;
    }
  }

  // ============================================================================
  // Phase 6: Report
  // ============================================================================
  phase('Report');

  if (!bestKernel) {
    log('No successful kernel found');
    return { success: false, reason: 'no_successful_kernel' };
  }

  const report = await agent(
    `Generate StitchCUDA synthesis report:

Orchestration summary:
- Target: ${kernelSpec.operation} on ${setupResult.target_architecture}
- Total attempts: ${performanceHistory.length} successful / ${maxAttempts} max
- Best performance: ${bestPerformance.toFixed(2)} GFLOPS
- Speedup: ${bestKernel.verification.speedup_vs_baseline.toFixed(2)}x
- Best attempt: ${bestKernel.attempt}

Best kernel plan:
${bestKernel.plan.plan_summary}
Key strategies: ${bestKernel.plan.key_strategies.join(', ')}

Performance trajectory:
${performanceHistory.map((p, idx) => `  Attempt ${idx + 1}: ${p.toFixed(2)} GFLOPS`).join('\n')}

Generate report with:
1. Executive summary
2. Orchestration overview (Planner-Coder-Verifier)
3. Replanning events and reasons
4. Performance progression
5. Best kernel analysis
6. Optimization breakdown

Return JSON:
{
  "summary": "brief summary",
  "total_attempts": ${performanceHistory.length},
  "successful_attempts": ${performanceHistory.length},
  "best_gflops": ${bestPerformance},
  "speedup": ${bestKernel.verification.speedup_vs_baseline},
  "best_attempt": ${bestKernel.attempt},
  "report_path": "path/to/report.md"
}`,
    {
      label: 'Generate report',
      phase: 'Report',
      schema: {
        type: 'object',
        properties: {
          summary: { type: 'string' },
          total_attempts: { type: 'integer' },
          successful_attempts: { type: 'integer' },
          best_gflops: { type: 'number' },
          speedup: { type: 'number' },
          best_attempt: { type: 'integer' },
          report_path: { type: 'string' },
        },
        required: ['summary', 'best_gflops'],
      },
    }
  );

  // ============================================================================
  // Return final results
  // ============================================================================

  return {
    success: true,
    method: 'StitchCUDA',
    approach: 'Three-agent orchestration (Planner-Coder-Verifier)',
    kernel: kernelSpec.operation,
    target_architecture: setupResult.target_architecture,
    cuda_version: setupResult.cuda_version,
    total_attempts: performanceHistory.length,
    successful_attempts: performanceHistory.length,
    baseline_gflops: kernelSpec.baseline_gflops,
    best_gflops: bestPerformance,
    speedup: bestKernel.verification.speedup_vs_baseline,
    best_attempt: bestKernel.attempt,
    best_plan: bestKernel.plan.plan_summary,
    best_strategies: bestKernel.plan.key_strategies,
    performance_trajectory: performanceHistory,
    final_kernel: bestKernel.code.kernel_code,
    report: report?.report_path,
    summary: report?.summary,
  };
}

// Helper function to detect performance stagnation
function isStagnant(history, windowSize) {
  if (history.length < windowSize) return false;
  const recentWindow = history.slice(-windowSize);
  const avgRecent = recentWindow.reduce((a, b) => a + b, 0) / windowSize;
  const variance = recentWindow.reduce((sum, val) => sum + Math.pow(val - avgRecent, 2), 0) / windowSize;
  const coefficientOfVariation = Math.sqrt(variance) / avgRecent;
  // Stagnant if coefficient of variation < 5%
  return coefficientOfVariation < 0.05;
}

// Execute the workflow
return await main();
