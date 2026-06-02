export const meta = {
  name: 'accelopt-kernel-optimization',
  description: 'Self-improving CUDA kernel optimization loop with NCU profiling (AccelOpt methodology)',
  whenToUse: 'When you need to iteratively optimize a CUDA kernel through plan-execute-profile-learn cycles. Uses Nsight Compute (ncu) for evidence-based profiling rather than guessing bottlenecks.',
  phases: [
    { title: 'Setup', detail: 'Read target kernel, compile harness, NCU profile baseline' },
    { title: 'Plan', detail: 'Generate optimization plans guided by NCU bottleneck data' },
    { title: 'Execute', detail: 'Implement optimized kernels from each plan' },
    { title: 'Evaluate', detail: 'NCU profile variants, measure real speedup' },
    { title: 'Learn', detail: 'Extract reusable insights from slow-fast kernel pairs + NCU diffs' },
    { title: 'Iterate', detail: 'Feed learnings back into next optimization round' },
  ],
}

// =============================================================================
// AccelOpt Self-Improving Kernel Optimization Workflow (NCU-Enhanced)
// =============================================================================
//
// Implements the AccelOpt paper's core loop (MLSys 2026, arXiv:2511.15915):
//   Plan → Execute → Profile → Summarize → Accumulate Experience → Repeat
//
// Enhanced with Nsight Compute (ncu) profiling at each stage:
//   - Setup: NCU --set full on baseline → identifies REAL bottlenecks
//   - Plan: Planner receives NCU metrics (stalls, memory patterns, occupancy)
//   - Evaluate: NCU profiles each variant → real latency + metric comparison
//   - Learn: Summarizer sees both code diff AND metric diff
//
// Usage:
//   Workflow({name: 'accelopt-kernel-optimization', args: {
//     kernel_path: '/path/to/kernel.cu',
//     op_description: 'Quantized GEMM Q4_0 weight * FP32 activation',
//     harness_path: '/path/to/harness.cu',         // standalone profiling harness
//     harness_build_cmd: 'nvcc -O3 -lineinfo ...',  // build command
//     harness_run_args: '',                          // runtime args for harness binary
//     kernel_name_regex: 'forward_kernel',           // ncu -k regex
//     ncu_binary: 'ncu',                             // path to ncu
//     exp_dir: '/path/to/experiment/output',         // where to save profiles
//     iterations: 3,
//     breadth: 3,
//     samples_per_plan: 2,
//   }})
//
// =============================================================================

const KERNEL_PATH = args.kernel_path
const OP_DESC = args.op_description || 'CUDA kernel'
const ITERATIONS = args.iterations || 2
const BREADTH = args.breadth || 3
const SAMPLES_PER_PLAN = args.samples_per_plan || 2

// NCU Configuration
const HARNESS_PATH = args.harness_path || ''
const HARNESS_BUILD_CMD = args.harness_build_cmd || ''
const HARNESS_RUN_ARGS = args.harness_run_args || ''
const KERNEL_NAME_REGEX = args.kernel_name_regex || ''
const NCU_BINARY = args.ncu_binary || 'ncu'
const EXP_DIR = args.exp_dir || '/tmp/accelopt_exp'

// Fallback: non-NCU profiling commands (used if NCU is not available)
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || ''

// State
let experienceMemory = []
let bestLatency = null
let bestKernelCode = null
let baselineLatency = null
let baselineNcuProfile = ''  // NCU analysis text for the baseline

// =============================================================================
// Phase 1: Setup — Read kernel, build harness, NCU profile baseline
//
// Key principle from ncu-report-skill:
//   "Profile → Diagnose → Plan, in that order. Never guess."
// =============================================================================
phase('Setup')

const setupResult = await agent(`Read the CUDA kernel file at: ${KERNEL_PATH}

Analyze it and return a JSON object with:
- kernel_code: the full source code
- op_type: operation type (e.g., "quantized_gemm", "attention", "rmsnorm", "softmax")
- key_functions: list of key function names (especially __global__ kernels)
- current_approach: brief description of the implementation strategy
- launch_config: if visible, the grid/block dimensions used
- shared_memory_usage: whether and how shared memory is used
- memory_access_patterns: description of global memory access patterns

Return ONLY the JSON object.`, {
  label: 'read-baseline',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      kernel_code: { type: 'string' },
      op_type: { type: 'string' },
      key_functions: { type: 'array', items: { type: 'string' } },
      current_approach: { type: 'string' },
      launch_config: { type: 'string' },
      shared_memory_usage: { type: 'string' },
      memory_access_patterns: { type: 'string' },
    },
    required: ['kernel_code', 'op_type', 'key_functions', 'current_approach'],
  },
})

const baselineKernel = setupResult.kernel_code
const opType = setupResult.op_type
log(`Baseline: ${opType}, kernels: ${setupResult.key_functions.join(', ')}`)

// NCU Profile the baseline
// Following the ncu-report-skill workflow:
//   1. Create run directory
//   2. Build harness (if needed)
//   3. Run ncu --set full + PmSampling
//   4. Run ncu --set source
//   5. Parse with Python API
//   6. Work through 6 analysis dimensions
const ncuSetup = await agent(`You are a CUDA profiling expert using Nsight Compute (ncu). Set up and run NCU profiling for the baseline kernel.

# Environment
- NCU binary: ${NCU_BINARY}
- Experiment directory: ${EXP_DIR}
- Kernel file: ${KERNEL_PATH}
- Kernel name regex for ncu -k: ${KERNEL_NAME_REGEX || '(auto-detect from kernel file)'}
- Harness path: ${HARNESS_PATH || '(need to build one)'}
- Harness build command: ${HARNESS_BUILD_CMD || '(need to determine)'}
- Harness run args: ${HARNESS_RUN_ARGS}

# Kernel Source:
\`\`\`cuda
${baselineKernel.substring(0, 4000)}
\`\`\`

# Instructions

## Step 1: Create run directory
\`\`\`bash
mkdir -p ${EXP_DIR}/baseline/{harness,reports,analysis}
\`\`\`

## Step 2: Build the profiling harness
If harness_path is provided and exists, use it. Otherwise, determine how to build a standalone harness that launches this kernel with representative inputs. The harness MUST be compiled with \`-lineinfo\` for source-level analysis.

Build command pattern:
\`\`\`bash
nvcc -O3 -lineinfo -arch=sm_XX -o ${EXP_DIR}/baseline/harness/bench <source> -lcudart
\`\`\`

## Step 3: Run NCU full profile
\`\`\`bash
${NCU_BINARY} --set full \\
    --section PmSampling --section PmSampling_WarpStates \\
    -k "regex:${KERNEL_NAME_REGEX || 'KERNEL_NAME'}" \\
    -c 1 \\
    -o ${EXP_DIR}/baseline/reports/full_baseline \\
    ${EXP_DIR}/baseline/harness/bench ${HARNESS_RUN_ARGS}
\`\`\`

## Step 4: Run NCU source profile
\`\`\`bash
${NCU_BINARY} --set source --section SourceCounters \\
    -k "regex:${KERNEL_NAME_REGEX || 'KERNEL_NAME'}" \\
    -c 1 \\
    -o ${EXP_DIR}/baseline/reports/source_baseline \\
    ${EXP_DIR}/baseline/harness/bench ${HARNESS_RUN_ARGS}
\`\`\`

## Step 5: Extract details page
\`\`\`bash
${NCU_BINARY} --import ${EXP_DIR}/baseline/reports/full_baseline.ncu-rep --page details \\
    > ${EXP_DIR}/baseline/analysis/details_baseline.txt
\`\`\`

## Step 6: Extract key metrics
Read the details page and the report to extract:
- gpu__time_duration.sum (kernel duration)
- sm__throughput.avg.pct_of_peak_sustained_elapsed
- dram__bytes_read.sum.pct_of_peak_sustained_elapsed
- sm__warps_active.avg.pct_of_peak_sustained_active (achieved occupancy)
- sm__maximum_warps_per_active_cycle_pct (theoretical occupancy)
- launch__waves_per_multiprocessor
- launch__registers_per_thread
- Top stall reasons (long_scoreboard, short_scoreboard, wait, barrier, etc.)
- Sectors/request for global loads
- L1/L2 hit rates
- NCU rule suggestions with Est. Speedup percentages

Execute these steps. If ncu is not available or fails, fall back to static code analysis.

Return a structured profile result.`, {
  label: 'ncu-baseline',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      latency_ms: { type: 'number' },
      sm_throughput_pct: { type: 'number' },
      dram_throughput_pct: { type: 'number' },
      achieved_occupancy_pct: { type: 'number' },
      theoretical_occupancy_pct: { type: 'number' },
      waves_per_sm: { type: 'number' },
      registers_per_thread: { type: 'number' },
      top_stall_reason: { type: 'string' },
      top_stall_pct: { type: 'number' },
      sectors_per_request: { type: 'number' },
      l1_hit_rate_pct: { type: 'number' },
      l2_hit_rate_pct: { type: 'number' },
      ncu_rule_suggestions: { type: 'array', items: { type: 'string' } },
      bottleneck_diagnosis: { type: 'string' },
      profile_summary: { type: 'string' },
      ncu_available: { type: 'boolean' },
    },
    required: ['latency_ms', 'bottleneck_diagnosis', 'profile_summary'],
  },
})

baselineLatency = ncuSetup.latency_ms
bestLatency = baselineLatency
bestKernelCode = baselineKernel

// Build the NCU profile string that will be injected into the Planner
baselineNcuProfile = `
## NCU Profile Results (Baseline)
- Latency: ${ncuSetup.latency_ms} ms
- SM Throughput: ${ncuSetup.sm_throughput_pct || 'N/A'}% of peak
- DRAM Throughput: ${ncuSetup.dram_throughput_pct || 'N/A'}% of peak
- Achieved Occupancy: ${ncuSetup.achieved_occupancy_pct || 'N/A'}%
- Theoretical Occupancy: ${ncuSetup.theoretical_occupancy_pct || 'N/A'}%
- Waves/SM: ${ncuSetup.waves_per_sm || 'N/A'}
- Registers/Thread: ${ncuSetup.registers_per_thread || 'N/A'}
- Top Stall Reason: ${ncuSetup.top_stall_reason || 'N/A'} (${ncuSetup.top_stall_pct || 'N/A'}% of samples)
- Sectors/Request (global LD): ${ncuSetup.sectors_per_request || 'N/A'} (ideal=4)
- L1 Hit Rate: ${ncuSetup.l1_hit_rate_pct || 'N/A'}%
- L2 Hit Rate: ${ncuSetup.l2_hit_rate_pct || 'N/A'}%

## Bottleneck Diagnosis:
${ncuSetup.bottleneck_diagnosis}

## NCU Rule Suggestions:
${(ncuSetup.ncu_rule_suggestions || []).map(s => `- ${s}`).join('\n') || 'N/A'}
`

log(`Baseline: ${baselineLatency}ms | ${ncuSetup.bottleneck_diagnosis}`)

// =============================================================================
// Iterative Self-Improvement Loop
// =============================================================================

for (let iter = 0; iter < ITERATIONS; iter++) {
  log(`\n=== Iteration ${iter + 1}/${ITERATIONS} | Best: ${bestLatency.toFixed(3)}ms (${(baselineLatency / bestLatency).toFixed(2)}x) | Experience: ${experienceMemory.length} patterns ===`)

  // ===========================================================================
  // Phase 2: Plan — Generate optimization plans GUIDED BY NCU DATA
  //
  // Key difference from vanilla AccelOpt: the planner receives REAL profiling
  // data (stall reasons, memory patterns, occupancy) rather than guessing from
  // code alone. This follows the ncu-report-skill principle:
  //   "Profile → Diagnose → Plan, in that order. Never guess."
  // ===========================================================================
  phase('Plan')

  const experienceSection = experienceMemory.length > 0
    ? `\n\n# Learned Optimization Patterns (from previous iterations)\n${experienceMemory.map((e, i) => `${i + 1}. ${e}`).join('\n')}`
    : ''

  // NCU-informed focus areas — derived from the diagnosis playbook
  const planAngles = [
    'memory latency hiding: address long_scoreboard stalls via ILP, prefetching, async copies, or software pipelining',
    'memory coalescing and vectorization: fix uncoalesced accesses (sectors/request > 4), use float4/int4 loads',
    'occupancy and parallelism: address SM idle time, tail effects, or low achieved occupancy',
    'compute restructuring: tensor core usage, warp-level reductions, reduced synchronization',
    'data layout and tiling: shared memory staging, bank-conflict-free layouts, double-buffering',
  ]

  const planPromptBase = `You are a CUDA kernel optimization expert. You have REAL Nsight Compute (NCU) profiling data for this kernel. Use it to generate ONE specific, evidence-based optimization plan.

# Operation: ${OP_DESC} (${opType})

# Current Best Implementation:
\`\`\`cuda
${bestKernelCode.substring(0, 4000)}
\`\`\`

# NCU PROFILING DATA (THIS IS REAL MEASURED DATA — base your plan on this):
${baselineNcuProfile}

# Current Performance:
- Latency: ${bestLatency}ms
- Speedup vs original baseline: ${(baselineLatency / bestLatency).toFixed(2)}x
${experienceSection}

# How to read NCU data for planning:
- If top stall is "long_scoreboard" (>40%): kernel is MEMORY-LATENCY-BOUND. Add ILP, async loads, or data reuse.
- If top stall is "short_scoreboard" (>30%): heavy shared-mem or dep chains. Shorten chains, add ILP.
- If top stall is "barrier" (>20%): too much __syncthreads. Use warp-level primitives.
- If top stall is "math_pipe_throttle": actually compute-bound — good! Look elsewhere.
- If DRAM throughput > 80%: bandwidth-bound. Reduce bytes read (compression, shared-mem reuse).
- If DRAM throughput < 10% AND long_scoreboard high: latency-bound on L1, not DRAM.
- If sectors/request > 5: uncoalesced access — big optimization opportunity.
- If achieved occupancy << theoretical: stalls prevent filling SM, fix stall source first.
- If waves/SM < 1: grid too small, parallelize more or use persistent kernel.
- If registers/thread > 128: likely register spill — add __launch_bounds__.
- NCU rule suggestions with "Est. Speedup: X%" are surprisingly accurate — prioritize them.

# Optimization Plan Requirements:
1. CITE the specific NCU metric(s) that justify your plan
2. Name the exact code region and transformation
3. Prefer STRUCTURAL changes over parameter tuning
4. Don't suggest lowering precision below the baseline
5. Estimate expected speedup based on the NCU data (e.g., "NCU reports sectors/request=8.2; fixing to 4.0 should cut load time ~2x on those lines")`

  const plans = await parallel(
    Array.from({length: BREADTH}, (_, i) => () =>
      agent(`${planPromptBase}\n\n# YOUR FOCUS AREA: ${planAngles[i % planAngles.length]}\nYou are planner #${i + 1}/${BREADTH}. Focus on: ${planAngles[i % planAngles.length]}.`, {
        label: `plan-${iter}-${i}`,
        phase: 'Plan',
        schema: {
          type: 'object',
          properties: {
            title: { type: 'string' },
            focus_area: { type: 'string' },
            ncu_evidence: { type: 'string' },
            analysis: { type: 'string' },
            plan: { type: 'string' },
            expected_impact: { type: 'string' },
            risk: { type: 'string' },
          },
          required: ['title', 'ncu_evidence', 'plan', 'expected_impact'],
        },
      })
    )
  )

  const validPlans = plans.filter(Boolean)
  log(`Plans: ${validPlans.map(p => `${p.title} (evidence: ${(p.ncu_evidence || '').substring(0, 50)}...)`).join(' | ')}`)

  // ===========================================================================
  // Phase 3: Execute — Implement each plan
  // ===========================================================================
  phase('Execute')

  const implementations = await pipeline(
    validPlans,
    (plan) => parallel(
      Array.from({length: SAMPLES_PER_PLAN}, (_, sampleIdx) => () =>
        agent(`You are an expert CUDA kernel developer. Implement this NCU-informed optimization plan as a complete, compilable kernel.

# Original Kernel:
\`\`\`cuda
${bestKernelCode.substring(0, 4000)}
\`\`\`

# Optimization Plan: "${plan.title}"
NCU Evidence: ${plan.ncu_evidence}
Plan: ${plan.plan}

# Requirements:
1. Output a COMPLETE .cu file: all #includes, struct definitions, __global__ kernel(s), forward() wrapper, PYBIND11_MODULE
2. Must be FUNCTIONALLY CORRECT (same output as baseline within FP tolerance)
3. Apply the plan faithfully — the plan is based on real NCU data, so the optimization targets a real bottleneck
4. Keep the forward() function signature unchanged
5. MUST compile with -lineinfo (don't use features that break debug info)
6. This is variant ${sampleIdx + 1}/${SAMPLES_PER_PLAN}

Return the complete CUDA code.`, {
          label: `impl-${iter}-${plan.title.substring(0, 15)}-v${sampleIdx}`,
          phase: 'Execute',
          schema: {
            type: 'object',
            properties: {
              code: { type: 'string' },
              implementation_notes: { type: 'string' },
            },
            required: ['code'],
          },
        })
      )
    )
  )

  const allVariants = []
  for (let planIdx = 0; planIdx < validPlans.length; planIdx++) {
    const planImpls = implementations[planIdx]
    if (!planImpls) continue
    for (let sIdx = 0; sIdx < planImpls.length; sIdx++) {
      const impl = planImpls[sIdx]
      if (impl && impl.code) {
        allVariants.push({
          plan: validPlans[planIdx],
          code: impl.code,
          id: `plan_${planIdx}_sample_${sIdx}`,
        })
      }
    }
  }

  log(`Generated ${allVariants.length} kernel variants`)

  // ===========================================================================
  // Phase 4: Evaluate — NCU profile each variant
  //
  // Uses real NCU profiling (not estimation) when available.
  // Follows the ncu-report-skill collection recipes:
  //   Recipe 5 (targeted metrics) for fast A/B comparison
  //   Recipe 6 (A/B comparison) for the best variant
  // ===========================================================================
  phase('Evaluate')

  const evaluations = await parallel(
    allVariants.map((variant, varIdx) => () =>
      agent(`You are a CUDA kernel evaluator using Nsight Compute. Evaluate this optimized kernel variant.

# Variant: ${variant.id} — Plan: "${variant.plan.title}"
# NCU Evidence for this plan: ${variant.plan.ncu_evidence}

# Kernel Code:
\`\`\`cuda
${variant.code.substring(0, 4000)}
\`\`\`

# Baseline Performance:
- Latency: ${bestLatency}ms
- Top stall: ${ncuSetup.top_stall_reason || 'unknown'}

# Evaluation Steps:

## Step 1: Static correctness check
- Race conditions? (check __syncthreads placement, shared-mem access patterns)
- Out-of-bounds? (check index computations against dimensions)
- Missing synchronization? (writes to shared followed by reads without barrier)
- Incorrect reductions? (warp shuffle masks, final-warp logic)

## Step 2: Compilability check
- All #includes present? (torch/extension.h, cuda_runtime.h, cuda_fp16.h, etc.)
- Valid CUDA syntax? (correct use of __global__, __shared__, __device__)
- PYBIND11_MODULE present?

## Step 3: Build and NCU profile (if environment allows)
\`\`\`bash
# Write kernel to file
mkdir -p ${EXP_DIR}/iter_${iter}/${variant.id}
# Write the kernel code to kernel.cu in that directory

# Build harness with new kernel
${HARNESS_BUILD_CMD ? HARNESS_BUILD_CMD.replace('KERNEL_PATH', `${EXP_DIR}/iter_${iter}/${variant.id}/kernel.cu`) : '# (no build command configured)'}

# Quick NCU metrics (Recipe 5 — targeted, fast)
${NCU_BINARY} --metrics \\
    gpu__time_duration.sum,\\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\\
    dram__bytes_read.sum.pct_of_peak_sustained_elapsed,\\
    sm__warps_active.avg.pct_of_peak_sustained_active,\\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\\
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \\
  -k "regex:${KERNEL_NAME_REGEX || 'forward'}" -c 1 \\
  ${EXP_DIR}/iter_${iter}/${variant.id}/bench ${HARNESS_RUN_ARGS}
\`\`\`

## Step 4: Compare with baseline
Calculate speedup = baseline_latency / variant_latency.
Note which NCU metrics improved and which degraded.

If NCU is not available, provide your expert static analysis:
- Did the optimization address the identified bottleneck (${ncuSetup.top_stall_reason})?
- Would you expect sectors/request to decrease?
- Would occupancy change?
- Estimate speedup based on the targeted inefficiency.

Return evaluation results.`, {
        label: `eval-${variant.id}`,
        phase: 'Evaluate',
        schema: {
          type: 'object',
          properties: {
            is_correct: { type: 'boolean' },
            is_compilable: { type: 'boolean' },
            estimated_latency_ms: { type: 'number' },
            estimated_speedup: { type: 'number' },
            correctness_issues: { type: 'array', items: { type: 'string' } },
            ncu_comparison: { type: 'string' },
            bottleneck_addressed: { type: 'boolean' },
            new_bottleneck: { type: 'string' },
            performance_analysis: { type: 'string' },
          },
          required: ['is_correct', 'is_compilable', 'estimated_speedup'],
        },
      })
    )
  )

  const results = []
  for (let i = 0; i < allVariants.length; i++) {
    const evalResult = evaluations[i]
    if (!evalResult) continue
    results.push({
      variant: allVariants[i],
      evaluation: evalResult,
      speedup: evalResult.estimated_speedup || 1.0,
    })
  }

  results.sort((a, b) => b.speedup - a.speedup)

  const improved = results.filter(r => r.speedup > 1.0 && r.evaluation.is_correct && r.evaluation.is_compilable)
  const degraded = results.filter(r => r.speedup < 1.0 && r.evaluation.is_correct && r.evaluation.is_compilable)

  log(`Results: ${improved.length} improved, ${degraded.length} degraded, ${results.length - improved.length - degraded.length} failed`)

  // Update best kernel and re-profile if improved
  if (improved.length > 0) {
    const best = improved[0]
    bestKernelCode = best.variant.code
    bestLatency = best.evaluation.estimated_latency_ms || (baselineLatency / best.speedup)

    // Update NCU profile for next iteration's planner
    if (best.evaluation.ncu_comparison) {
      baselineNcuProfile = `
## NCU Profile Results (After Iteration ${iter + 1} — Best Variant: "${best.variant.plan.title}")
- Latency: ${bestLatency}ms (${best.speedup.toFixed(2)}x speedup)
- Bottleneck addressed: ${best.evaluation.bottleneck_addressed ? 'YES' : 'NO'}
- New bottleneck: ${best.evaluation.new_bottleneck || 'unknown'}
- Comparison: ${best.evaluation.ncu_comparison}

Previous profile data for reference:
${baselineNcuProfile}`
    }

    log(`NEW BEST: "${best.variant.plan.title}" — ${best.speedup.toFixed(2)}x, ~${bestLatency.toFixed(3)}ms`)
  }

  // ===========================================================================
  // Phase 5: Learn — Extract insights from slow-fast pairs + NCU metric diffs
  //
  // Enhanced over vanilla AccelOpt: the summarizer sees BOTH the code diff AND
  // the NCU metric diff, producing richer optimization patterns.
  // ===========================================================================
  phase('Learn')

  const pairsToSummarize = []

  for (const r of improved.slice(0, 3)) {
    pairsToSummarize.push({
      slow: baselineKernel,
      fast: r.variant.code,
      speedup: r.speedup,
      plan_title: r.variant.plan.title,
      ncu_evidence: r.variant.plan.ncu_evidence,
      ncu_comparison: r.evaluation.ncu_comparison || '',
      bottleneck_addressed: r.evaluation.bottleneck_addressed,
      type: 'positive',
    })
  }

  for (const r of degraded.slice(0, 2)) {
    pairsToSummarize.push({
      slow: r.variant.code,
      fast: bestKernelCode,
      speedup: 1.0 / r.speedup,
      plan_title: r.variant.plan.title + ' [ANTI-PATTERN]',
      ncu_evidence: r.variant.plan.ncu_evidence,
      ncu_comparison: r.evaluation.ncu_comparison || '',
      bottleneck_addressed: r.evaluation.bottleneck_addressed,
      type: 'negative',
    })
  }

  if (pairsToSummarize.length > 0) {
    const summaries = await parallel(
      pairsToSummarize.map((pair) => () =>
        agent(`You are a CUDA optimization expert with NCU profiling expertise. Analyze this slow-fast kernel pair and extract a GENERAL, REUSABLE optimization insight.

# Slow Kernel:
\`\`\`cuda
${pair.slow.substring(0, 2500)}
\`\`\`

# Fast Kernel:
\`\`\`cuda
${pair.fast.substring(0, 2500)}
\`\`\`

# Speedup: ${pair.speedup.toFixed(2)}x
# This is a ${pair.type === 'positive' ? 'POSITIVE example (do this)' : 'NEGATIVE example (avoid this)'}

# NCU Evidence that motivated this optimization:
${pair.ncu_evidence || 'N/A'}

# NCU Metric Comparison (before vs after):
${pair.ncu_comparison || 'N/A'}

# Was the targeted bottleneck addressed? ${pair.bottleneck_addressed ? 'YES' : 'NO/UNKNOWN'}

## Your task:
Extract a GENERAL optimization rule. Include:
1. What NCU signal triggered this optimization (so future planners know when to apply it)
2. The code transformation pattern
3. WHY it helps (in terms of hardware — cache behavior, warp scheduling, etc.)

Format:
**{Short title}**
NCU trigger: {what metric/stall pattern signals this opportunity}
Rule: {one sentence — when you see X in NCU, do Y to the code}
Before:
\`\`\`cuda
{2-5 lines of slow pattern}
\`\`\`
After:
\`\`\`cuda
{2-5 lines of fast pattern}
\`\`\`
Why: {hardware-level explanation}`, {
          label: `learn-${pair.plan_title.substring(0, 20)}`,
          phase: 'Learn',
          schema: {
            type: 'object',
            properties: {
              summary: { type: 'string' },
              title: { type: 'string' },
              ncu_trigger: { type: 'string' },
              is_antipattern: { type: 'boolean' },
            },
            required: ['summary', 'title'],
          },
        })
      )
    )

    for (const s of summaries.filter(Boolean)) {
      experienceMemory.push(s.summary)
    }
    log(`Learned ${summaries.filter(Boolean).length} patterns (with NCU triggers). Bank: ${experienceMemory.length}`)
  } else {
    log(`No pairs to learn from.`)
  }

  phase('Iterate')
  log(`Iteration ${iter + 1} done. ${(baselineLatency / bestLatency).toFixed(2)}x vs baseline.`)
}

// =============================================================================
// Final Report
// =============================================================================
const finalReport = await agent(`Write a concise technical optimization report.

# AccelOpt + NCU Optimization Results
- Operation: ${OP_DESC} (${opType})
- Baseline Latency: ${baselineLatency}ms
- Final Best Latency: ${bestLatency}ms
- Overall Speedup: ${(baselineLatency / bestLatency).toFixed(2)}x
- Iterations: ${ITERATIONS}
- Experience Patterns: ${experienceMemory.length}

# Initial NCU Diagnosis:
${baselineNcuProfile.substring(0, 1000)}

# Learned Optimization Knowledge Base:
${experienceMemory.map((e, i) => `${i + 1}. ${e}`).join('\n\n')}

# Final Kernel:
\`\`\`cuda
${bestKernelCode.substring(0, 3000)}
\`\`\`

Write:
1. NCU-driven optimization journey (what metrics → what actions → what results)
2. Which NCU patterns reliably predicted optimization opportunities
3. Anti-patterns: what NCU data looked promising but the optimization failed
4. Remaining bottlenecks (what NCU shows for the final kernel)
5. Recommendations for further optimization with specific NCU metrics to target`, {
  label: 'final-report',
  phase: 'Iterate',
})

return {
  baseline_latency_ms: baselineLatency,
  best_latency_ms: bestLatency,
  overall_speedup: baselineLatency / bestLatency,
  iterations_completed: ITERATIONS,
  experience_patterns_count: experienceMemory.length,
  experience_patterns: experienceMemory,
  best_kernel_code: bestKernelCode,
  ncu_baseline_profile: baselineNcuProfile,
  report: finalReport,
}
