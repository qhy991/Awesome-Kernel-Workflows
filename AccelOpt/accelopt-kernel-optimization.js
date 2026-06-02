export const meta = {
  name: 'accelopt-kernel-optimization',
  description: 'Self-improving CUDA kernel optimization loop with NCU profiling (AccelOpt methodology)',
  whenToUse: 'When you need to iteratively optimize a CUDA kernel through plan-execute-profile-learn cycles. Uses Nsight Compute (ncu) for evidence-based profiling rather than guessing bottlenecks.',
  phases: [
    { title: 'Setup', detail: 'Read target kernel, compile harness, NCU profile baseline' },
    { title: 'Plan', detail: 'Generate optimization plans guided by NCU data + candidate beam context' },
    { title: 'Execute', detail: 'Implement optimized kernels from each plan' },
    { title: 'Evaluate', detail: 'NCU profile variants, per-branch dedup, update candidate beam' },
    { title: 'Learn', detail: 'Threshold-filtered slow-fast pairs → reusable patterns (AccelOpt format)' },
    { title: 'Iterate', detail: 'Feed sampled experience + beam state into next optimization round' },
  ],
}

// =============================================================================
// AccelOpt Self-Improving Kernel Optimization Workflow (NCU-Enhanced, v2)
// =============================================================================
//
// Implements the AccelOpt paper's core loop (MLSys 2026, arXiv:2511.15915):
//   Plan → Execute → Profile → Summarize → Accumulate Experience → Repeat
//
// v2 enhancements aligned with the original AccelOpt system:
//   1. Candidate beam pool (topK kernels carried forward, not just single best)
//   2. Experience pool random sampling with capacity control
//   3. Parameterized selection heuristics (threshold filtering)
//   4. Per-branch deduplication (best sample per plan)
//   5. Experience format aligned with original system
//
// Usage:
//   Workflow({name: 'accelopt-kernel-optimization', args: {
//     kernel_path: '/path/to/kernel.cu',
//     op_description: 'Quantized GEMM Q4_0 weight * FP32 activation',
//     harness_path: '/path/to/harness.cu',
//     harness_build_cmd: 'nvcc -O3 -lineinfo ...',
//     harness_run_args: '',
//     kernel_name_regex: 'forward_kernel',
//     ncu_binary: 'ncu',
//     exp_dir: '/path/to/experiment/output',
//     iterations: 3,
//     breadth: 3,
//     samples_per_plan: 2,
//     topk_candidates: 3,
//     max_experience_in_prompt: 8,
//     max_threshold: 1.05,
//     min_threshold: 1.05,
//     topk_learn: 5,
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

// Fallback: non-NCU profiling commands
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || ''

// AccelOpt-aligned parameters (v2)
const TOPK_CANDIDATES = args.topk_candidates || 3
const MAX_EXPERIENCE_IN_PROMPT = args.max_experience_in_prompt || 8
const MAX_THRESHOLD = args.max_threshold || 1.05
const MIN_THRESHOLD = args.min_threshold || 1.05
const TOPK_LEARN = args.topk_learn || 5

// State
let experienceMemory = []       // Full pool of learned patterns (grows unbounded)
let lastIterNewPatterns = []    // Patterns discovered in the most recent Learn phase
let bestLatency = null
let bestKernelCode = null
let baselineLatency = null
let baselineNcuProfile = ''
let candidateBeam = []          // [{code, latency, speedup, ncuSummary, planTitle}]

// Helper: sample n items from array without replacement (Fisher-Yates partial)
function sampleWithoutReplacement(arr, n) {
  if (n >= arr.length) return [...arr]
  const copy = [...arr]
  const result = []
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor((copy.length - i) * (i / (i + 1 + copy.length)))
    // Deterministic pseudo-shuffle using index-based swap
    const swapIdx = (i * 7 + 3) % (copy.length - i) + i
    const temp = copy[i]
    copy[i] = copy[swapIdx]
    copy[swapIdx] = temp
    result.push(copy[i])
  }
  return result
}

// Helper: construct experience section for planner prompt (AccelOpt sampling logic)
function buildExperienceSection(experienceMemory, lastIterNewPatterns, maxInPrompt) {
  if (experienceMemory.length === 0) return ''

  let selected = []
  // Priority: new patterns from last iteration (like original_rewrite_list in AccelOpt)
  const newPatterns = [...lastIterNewPatterns]
  if (newPatterns.length >= maxInPrompt) {
    selected = newPatterns.slice(0, maxInPrompt)
  } else {
    selected = [...newPatterns]
    const remaining = maxInPrompt - selected.length
    // Fill remaining slots with random samples from the full pool (excluding already-selected)
    const pool = experienceMemory.filter(e => !newPatterns.includes(e))
    const sampled = sampleWithoutReplacement(pool, remaining)
    selected = selected.concat(sampled)
  }

  return `\n\n# Learned Optimization Patterns (${selected.length}/${experienceMemory.length} sampled)\n${selected.map((e, i) => `${i + 1}. ${e}`).join('\n\n')}`
}

// Helper: format candidate beam info for planner prompt
function buildBeamSection(candidateBeam) {
  if (candidateBeam.length <= 1) return ''
  return `\n\n# Candidate Beam (top-${candidateBeam.length} kernels from previous iterations)\n${candidateBeam.map((c, i) => `## Candidate ${i + 1}: "${c.planTitle}" — ${c.speedup.toFixed(2)}x, ${c.latency.toFixed(3)}ms\nNCU: ${c.ncuSummary || 'N/A'}\n\`\`\`cuda\n${c.code.substring(0, 1500)}\n\`\`\``).join('\n\n')}`
}

// =============================================================================
// Phase 1: Setup — Read kernel, build harness, NCU profile baseline
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

// Initialize candidate beam with baseline
candidateBeam = [{
  code: baselineKernel,
  latency: baselineLatency,
  speedup: 1.0,
  ncuSummary: ncuSetup.profile_summary || ncuSetup.bottleneck_diagnosis,
  planTitle: 'baseline',
}]

// Build the NCU profile string
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
  log(`\n=== Iteration ${iter + 1}/${ITERATIONS} | Best: ${bestLatency.toFixed(3)}ms (${(baselineLatency / bestLatency).toFixed(2)}x) | Beam: ${candidateBeam.length} | Experience: ${experienceMemory.length} patterns ===`)

  // ===========================================================================
  // Phase 2: Plan — Generate optimization plans GUIDED BY NCU DATA + BEAM
  // ===========================================================================
  phase('Plan')

  // Experience sampling (AccelOpt: construct_experience.py logic)
  const experienceSection = buildExperienceSection(experienceMemory, lastIterNewPatterns, MAX_EXPERIENCE_IN_PROMPT)

  // Candidate beam context for planner
  const beamSection = buildBeamSection(candidateBeam)

  // NCU-informed focus areas
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
${beamSection}
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
5. Estimate expected speedup based on the NCU data (e.g., "NCU reports sectors/request=8.2; fixing to 4.0 should cut load time ~2x on those lines")
6. If candidate beam shows multiple approaches, consider COMBINING strengths from different candidates`

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
          planIdx: planIdx,
          code: impl.code,
          id: `plan_${planIdx}_sample_${sIdx}`,
        })
      }
    }
  }

  log(`Generated ${allVariants.length} kernel variants`)

  // ===========================================================================
  // Phase 4: Evaluate — NCU profile each variant + per-branch dedup + beam update
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
mkdir -p ${EXP_DIR}/iter_${iter}/${variant.id}

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

  // Build results with evaluation data
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

  // Per-branch deduplication: keep only the best sample per plan (AccelOpt: select_candidates.py)
  const planBestMap = new Map()
  for (const r of results) {
    if (!r.evaluation.is_correct || !r.evaluation.is_compilable) continue
    const planKey = r.variant.planIdx
    const existing = planBestMap.get(planKey)
    if (!existing || r.speedup > existing.speedup) {
      planBestMap.set(planKey, r)
    }
  }
  const dedupedResults = [...planBestMap.values()]
  dedupedResults.sort((a, b) => b.speedup - a.speedup)

  // Update candidate beam (AccelOpt: select_candidates.py topK logic)
  const newCandidates = dedupedResults
    .filter(r => r.speedup > 1.0)
    .map(r => ({
      code: r.variant.code,
      latency: r.evaluation.estimated_latency_ms || (baselineLatency / r.speedup),
      speedup: r.speedup * (baselineLatency / bestLatency), // relative to original baseline
      ncuSummary: r.evaluation.ncu_comparison || r.evaluation.performance_analysis || '',
      planTitle: r.variant.plan.title,
    }))

  // Merge new candidates into beam, re-sort, keep topK
  const mergedBeam = [...candidateBeam, ...newCandidates]
  mergedBeam.sort((a, b) => a.latency - b.latency)
  candidateBeam = mergedBeam.slice(0, TOPK_CANDIDATES)

  // Update best from beam[0]
  if (candidateBeam.length > 0 && candidateBeam[0].latency < bestLatency) {
    bestKernelCode = candidateBeam[0].code
    bestLatency = candidateBeam[0].latency

    // Update NCU profile for next iteration
    const bestResult = dedupedResults.find(r => r.variant.code === candidateBeam[0].code)
    if (bestResult && bestResult.evaluation.ncu_comparison) {
      baselineNcuProfile = `
## NCU Profile Results (After Iteration ${iter + 1} — Best: "${candidateBeam[0].planTitle}")
- Latency: ${bestLatency}ms (${(baselineLatency / bestLatency).toFixed(2)}x speedup vs original)
- Bottleneck addressed: ${bestResult.evaluation.bottleneck_addressed ? 'YES' : 'NO'}
- New bottleneck: ${bestResult.evaluation.new_bottleneck || 'unknown'}
- Comparison: ${bestResult.evaluation.ncu_comparison}

Previous profile data for reference:
${baselineNcuProfile}`
    }

    log(`NEW BEST: "${candidateBeam[0].planTitle}" — ${(baselineLatency / bestLatency).toFixed(2)}x, ~${bestLatency.toFixed(3)}ms`)
  }

  const improved = dedupedResults.filter(r => r.speedup > 1.0)
  const degraded = dedupedResults.filter(r => r.speedup < 1.0)
  log(`Results (deduped): ${improved.length} improved, ${degraded.length} degraded | Beam: [${candidateBeam.map(c => c.planTitle).join(', ')}]`)

  // ===========================================================================
  // Phase 5: Learn — Extract insights from slow-fast pairs + NCU metric diffs
  //
  // AccelOpt alignment:
  //   - Threshold filtering (max_threshold / min_threshold)
  //   - topk_learn total budget
  //   - Experience format: **title** + NCU trigger + code snippets
  // ===========================================================================
  phase('Learn')

  // Threshold-filtered selection (AccelOpt: rewrites_selection.py)
  const positiveFiltered = improved.filter(r => r.speedup > MAX_THRESHOLD)
  const negativeFiltered = degraded.filter(r => r.speedup < (1.0 / MIN_THRESHOLD))

  const maxPositive = Math.min(positiveFiltered.length, Math.ceil(TOPK_LEARN / 2))
  const selectedPositive = positiveFiltered.slice(0, maxPositive)
  const remainingSlots = Math.min(TOPK_LEARN - selectedPositive.length, negativeFiltered.length)
  const selectedNegative = negativeFiltered.slice(0, remainingSlots)

  const pairsToSummarize = []

  for (const r of selectedPositive) {
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

  for (const r of selectedNegative) {
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
Extract a GENERAL optimization rule following this EXACT format:

**{Short title}**
NCU trigger: {what metric/stall pattern signals this opportunity}
Rule: {one sentence — when you see X in NCU, do Y to the code}
Original code:
\`\`\`cuda
{2-5 lines of slow pattern}
\`\`\`
Optimized code:
\`\`\`cuda
{2-5 lines of fast pattern}
\`\`\`
Why: {hardware-level explanation}

Make the rule GENERAL enough to apply to other kernels (not specific to this one kernel).`, {
          label: `learn-${pair.plan_title.substring(0, 20)}`,
          phase: 'Learn',
          schema: {
            type: 'object',
            properties: {
              title: { type: 'string' },
              ncu_trigger: { type: 'string' },
              rule: { type: 'string' },
              original_snippet: { type: 'string' },
              optimized_snippet: { type: 'string' },
              why: { type: 'string' },
              is_antipattern: { type: 'boolean' },
            },
            required: ['title', 'ncu_trigger', 'rule', 'original_snippet', 'optimized_snippet', 'why'],
          },
        })
      )
    )

    // Format experience entries aligned with AccelOpt's summarizer output format
    lastIterNewPatterns = []
    for (const s of summaries.filter(Boolean)) {
      const formatted = `**${s.title}**\nNCU trigger: ${s.ncu_trigger}\n${s.rule}\nOriginal code:\n\`\`\`cuda\n${s.original_snippet}\n\`\`\`\nOptimized code:\n\`\`\`cuda\n${s.optimized_snippet}\n\`\`\`\nWhy: ${s.why}`
      experienceMemory.push(formatted)
      lastIterNewPatterns.push(formatted)
    }
    log(`Learned ${lastIterNewPatterns.length} patterns (threshold-filtered from ${pairsToSummarize.length} pairs). Pool: ${experienceMemory.length}`)
  } else {
    lastIterNewPatterns = []
    log(`No pairs passed threshold filters (max>${MAX_THRESHOLD}, min<${(1/MIN_THRESHOLD).toFixed(3)}).`)
  }

  phase('Iterate')
  log(`Iteration ${iter + 1} done. ${(baselineLatency / bestLatency).toFixed(2)}x vs baseline. Beam size: ${candidateBeam.length}`)
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
- Candidate Beam (final): ${candidateBeam.length} kernels
- Experience Patterns: ${experienceMemory.length}

# Initial NCU Diagnosis:
${baselineNcuProfile.substring(0, 1000)}

# Final Candidate Beam:
${candidateBeam.map((c, i) => `${i + 1}. "${c.planTitle}" — ${c.speedup.toFixed(2)}x (${c.latency.toFixed(3)}ms)`).join('\n')}

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
4. Candidate beam evolution: how the population of solutions evolved
5. Remaining bottlenecks (what NCU shows for the final kernel)
6. Recommendations for further optimization with specific NCU metrics to target`, {
  label: 'final-report',
  phase: 'Iterate',
})

return {
  baseline_latency_ms: baselineLatency,
  best_latency_ms: bestLatency,
  overall_speedup: baselineLatency / bestLatency,
  iterations_completed: ITERATIONS,
  candidate_beam: candidateBeam.map(c => ({
    plan_title: c.planTitle,
    latency_ms: c.latency,
    speedup: c.speedup,
  })),
  experience_patterns_count: experienceMemory.length,
  experience_patterns: experienceMemory,
  best_kernel_code: bestKernelCode,
  ncu_baseline_profile: baselineNcuProfile,
  report: finalReport,
}
