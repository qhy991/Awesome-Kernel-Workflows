export const meta = {
  name: 'argus-kernel-optimization',
  description: 'Agentic GPU kernel optimization guided by data-flow invariants with ICRL planning (ARGUS methodology)',
  whenToUse: 'When optimizing GPU kernels that require coordinated reasoning over tiling, shared-memory staging, software pipelining, and instruction scheduling. Uses compile-time data-flow invariants for dense structured feedback instead of sparse pass/fail signals. Best for performance-critical kernels (GEMM, attention, MoE) targeting near-library-level throughput.',
  phases: [
    { title: 'Setup', detail: 'Read kernel spec, hardware config, initialize knowledge base context' },
    { title: 'Plan', detail: 'ICRL planner proposes optimizations + data-flow invariants from knowledge base' },
    { title: 'Select', detail: 'Optimization selector samples from planner proposals, resolves dependencies' },
    { title: 'Lower', detail: 'Lowering agent implements transformations with tag functions and assertions' },
    { title: 'Validate', detail: 'Compile-time invariant checking + unit tests + runtime profiling' },
    { title: 'Learn', detail: 'ICRL update: reward from invariant violations + performance, update planner policy' },
  ],
}

// =============================================================================
// ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants
// =============================================================================
//
// Source: "ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants"
//         Mai et al., CausalFlow / HKUST / Tsinghua / Stanford / UCAS / UC Riverside
//         arXiv:2604.18616, 2026
//
// Core insight: Data-flow invariants provide DENSE, STRUCTURED feedback for
// GPU kernel optimization. Instead of sparse pass/fail from unit tests:
//   - Tag functions annotate tensor elements with symbolic coordinates
//   - Tag assertions verify relational constraints at use sites
//   - Violations produce concrete counterexamples (thread, element, program point)
//   - This enables the agent to diagnose GLOBAL constraint violations
//
// Architecture (5 components from Section 6):
//   1. Knowledge Base: curated optimization skills (global intrusive + local + ISA)
//   2. Learnable Planner (ICRL): proposes (optimization, context, invariants)
//   3. Optimization Selector: samples from proposals, resolves coupling
//   4. Lowering Agent: implements transformations in DSL with tag functions
//   5. Validator: invariant checking + unit tests + profiling → reward signal
//
// The ICRL loop (Algorithm 1):
//   for each task:
//     for t = 0..T:
//       P_t = planner(state_t)           // ranked proposals
//       a_t = SELECT(P_t)                // sample optimization plan
//       s_{t+1} = LOWER(state_t, a_t)    // implement transformation
//       r_t = REWARD(s_{t+1}, tests)     // invariant violations + perf
//     PolicyEval → Analyze → ParameterUpdate (text gradients)
//
// Usage:
//   Workflow({name: 'argus-kernel-optimization', args: {
//     kernel_path: '/path/to/kernel.py',       // Argus DSL or CUDA/Triton
//     kernel_spec: 'Flash attention GQA, bf16, d=128, Br=256, Bc=64',
//     hardware_target: 'AMD MI300X' | 'NVIDIA H100',
//     test_command: 'python test_kernel.py',
//     benchmark_command: 'python bench_kernel.py',
//     knowledge_base_path: '',
//     iterations: 10,
//     inner_steps: 5,
//     optimization_categories: ['global_intrusive', 'local_source', 'isa_specific'],
//   }})
//
// =============================================================================

// --- Required Args ---
const KERNEL_PATH = args.kernel_path
const KERNEL_SPEC = args.kernel_spec || ''

// --- Optional Args ---
const HARDWARE_TARGET = args.hardware_target || 'NVIDIA H100'
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || ''
const KNOWLEDGE_BASE_PATH = args.knowledge_base_path || ''
const ITERATIONS = args.iterations || 5
const INNER_STEPS = args.inner_steps || 3
const EXP_DIR = args.exp_dir || '/tmp/argus_exp'
const OPT_CATEGORIES = args.optimization_categories || ['global_intrusive', 'local_source', 'isa_specific']

// State (ICRL)
let bestKernelCode = null
let bestThroughput = 0
let plannerPolicy = ''
let optimizationHistory = []
let invariantViolationLog = []
let candidateBeam = []

// Knowledge base categories (ARGUS Table 1)
const KNOWLEDGE_BASE = {
  global_intrusive: [
    'software_pipelining: overlap computation with data transfers using multi-stage buffering',
    'split_k: dispatch multiplications along K dimension across multiple threadgroups',
    'mfma_matmul: adopt swizzled layouts to use hardware matrix core (MFMA/WMMA) instructions',
    'stagger_k: stagger workloads along K to mitigate memory controller hot spots',
    'async_memcpy: asynchronous data loading with explicit fence/barrier management',
  ],
  local_source: [
    'bank_conflict_mitigation: pad or swizzle shared memory to eliminate bank conflicts',
    'vectorized_loads: use float4/int4/buffer_load_dwordx4 for coalesced wide memory access',
    'loop_unrolling: aggressively unroll inner loops to expose ILP',
    'workgroup_swizzling: rearrange workgroups for better L2 and chiplet locality',
  ],
  isa_specific: [
    'hw_oob_guarded_loads: use hardware out-of-bounds guards instead of explicit branches',
    'use_agprs: select accumulator register class (AGPR) to double available vector registers',
    'instruction_scheduling: overlap MFMA, ALU, and memory instructions to hide latency',
  ],
}

// =============================================================================
// Phase 1: Setup — Read kernel, hardware specs, initialize
// =============================================================================
phase('Setup')

const setupResult = await agent(`You are a GPU kernel optimization expert. Read and analyze the initial kernel implementation.

# Kernel file: ${KERNEL_PATH}
# Kernel specification: ${KERNEL_SPEC}
# Hardware target: ${HARDWARE_TARGET}

# Task:
1. Read the kernel source code at ${KERNEL_PATH}
2. Identify:
   - What computation it performs (GEMM, attention, MoE, etc.)
   - Current optimization level (naive, partially optimized, etc.)
   - Tiling strategy (tile sizes, how work maps to threads/blocks)
   - Memory hierarchy usage (registers, shared memory, global)
   - Any existing data-flow patterns (how data moves through the kernel)
3. Determine hardware constraints:
   - For ${HARDWARE_TARGET}: register file size, shared memory, compute units
   - Arithmetic intensity requirements (ops/byte for compute-bound)
   - Memory bandwidth and latency characteristics

Return the kernel analysis.`, {
  label: 'read-kernel',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      kernel_code: { type: 'string' },
      computation_type: { type: 'string' },
      current_opt_level: { type: 'string' },
      tiling_strategy: { type: 'string' },
      memory_usage: { type: 'string' },
      data_flow_patterns: { type: 'array', items: { type: 'string' } },
      hardware_constraints: { type: 'string' },
      missing_optimizations: { type: 'array', items: { type: 'string' } },
      estimated_peak_achievable: { type: 'string' },
    },
    required: ['kernel_code', 'computation_type', 'missing_optimizations'],
  },
})

bestKernelCode = setupResult.kernel_code
const computationType = setupResult.computation_type

// Run baseline benchmark if available
const baselineResult = await agent(`You are a GPU kernel validator. Run the baseline kernel to establish performance.

# Kernel: ${KERNEL_PATH}
# Test command: ${TEST_CMD || '(determine from project structure)'}
# Benchmark command: ${BENCH_CMD || '(determine from project structure)'}
# Experiment directory: ${EXP_DIR}

# Steps:
1. Create experiment directory: mkdir -p ${EXP_DIR}/baseline
2. Run correctness tests (if available): ${TEST_CMD}
3. Run performance benchmark: ${BENCH_CMD}
4. Record: throughput (TFLOPS or GB/s), latency, and any profiling data

If commands are not available, estimate baseline performance from the code structure.

Return baseline metrics.`, {
  label: 'baseline-perf',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      tests_pass: { type: 'boolean' },
      throughput_tflops: { type: 'number' },
      latency_us: { type: 'number' },
      peak_theoretical_tflops: { type: 'number' },
      efficiency_pct: { type: 'number' },
      profiling_summary: { type: 'string' },
    },
    required: ['throughput_tflops'],
  },
})

bestThroughput = baselineResult.throughput_tflops || 0
candidateBeam = [{ code: bestKernelCode, throughput: bestThroughput, label: 'baseline' }]

log(`Setup: ${computationType} on ${HARDWARE_TARGET} | Baseline: ${bestThroughput} TFLOPS (${baselineResult.efficiency_pct || '?'}% of peak)`)
log(`Missing optimizations: ${setupResult.missing_optimizations.join(', ')}`)

// =============================================================================
// ICRL Outer Loop — Learn planner policy across iterations
// =============================================================================

for (let outerIter = 0; outerIter < ITERATIONS; outerIter++) {
  log(`\n=== ICRL Iteration ${outerIter + 1}/${ITERATIONS} | Best: ${bestThroughput.toFixed(1)} TFLOPS | History: ${optimizationHistory.length} steps ===`)

  // ===========================================================================
  // Phase 2: Plan — ICRL planner proposes optimizations + invariants
  // ===========================================================================
  phase('Plan')

  const recentHistory = optimizationHistory.slice(-10)
  const recentViolations = invariantViolationLog.slice(-5)

  const plannerResult = await agent(`You are the ARGUS Learnable Planner (ICRL, Section 6).
Your job is to propose optimization candidates with associated data-flow invariants.

# Current Kernel (${computationType}):
\`\`\`
${bestKernelCode.substring(0, 5000)}
\`\`\`

# Hardware: ${HARDWARE_TARGET}
# Current throughput: ${bestThroughput} TFLOPS
# Kernel spec: ${KERNEL_SPEC}

# Knowledge Base — Available Optimizations:
## Global Intrusive Changes (restructure the kernel):
${KNOWLEDGE_BASE.global_intrusive.map(o => `- ${o}`).join('\n')}

## Local Source Changes (small, localized patches):
${KNOWLEDGE_BASE.local_source.map(o => `- ${o}`).join('\n')}

## ISA-Specific Optimizations (hardware intrinsics):
${KNOWLEDGE_BASE.isa_specific.map(o => `- ${o}`).join('\n')}

# Optimization History (recent):
${recentHistory.length > 0 ? recentHistory.map(h => `- ${h.optimization}: ${h.outcome} (${h.throughput_delta > 0 ? '+' : ''}${h.throughput_delta?.toFixed(2) || '?'} TFLOPS)`).join('\n') : 'No history yet.'}

# Recent Invariant Violations (failures to learn from):
${recentViolations.length > 0 ? recentViolations.map(v => `- ${v}`).join('\n') : 'None.'}

# Planner Policy (learned guidance):
${plannerPolicy || 'No policy learned yet — propose based on knowledge base and missing optimizations.'}

# Categories to explore: ${OPT_CATEGORIES.join(', ')}

# Task (ARGUS Section 6 — Learnable Planner):
Generate a RANKED list of 3-5 optimization proposals. For each:
1. **Optimization**: which technique from the knowledge base
2. **Context**: WHERE and HOW to apply it to this specific kernel
3. **Data-flow invariant**: what property MUST hold after the transformation
   - Express as: "assert tag(X) == tag(Y)" style — what data elements must match where
   - Example: "After staging K tiles in shared memory, elements loaded by each warp must
     correspond to the same K-slice used in the MFMA computation"
4. **Confidence/utility score**: how likely to improve performance (0-1)
5. **Dependencies**: what other optimizations this requires or enables

Rank proposals by expected impact. Prefer optimizations that:
- Address the LARGEST remaining performance gap
- Have clear invariants that can be verified
- Build on already-successful optimizations in history`, {
    label: `plan-${outerIter}`,
    phase: 'Plan',
    schema: {
      type: 'object',
      properties: {
        proposals: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              optimization: { type: 'string' },
              category: { type: 'string' },
              context: { type: 'string' },
              invariant: { type: 'string' },
              confidence: { type: 'number' },
              dependencies: { type: 'array', items: { type: 'string' } },
              expected_impact: { type: 'string' },
            },
            required: ['optimization', 'context', 'invariant', 'confidence'],
          },
        },
      },
      required: ['proposals'],
    },
  })

  const proposals = plannerResult.proposals || []
  log(`Planner: ${proposals.length} proposals | Top: "${proposals[0]?.optimization}" (conf: ${proposals[0]?.confidence?.toFixed(2)})`)

  // ===========================================================================
  // Phase 3: Select — Sample from proposals, resolve dependencies
  // ===========================================================================
  phase('Select')

  const selectResult = await agent(`You are the ARGUS Optimization Selector (Section 6).
The planner produced a ranked list of proposals. Your job is to select and sequence
a concrete optimization plan that resolves dependencies between coupled optimizations.

# Proposals (ranked by planner):
${proposals.map((p, i) => `${i + 1}. [${p.category}] ${p.optimization} (confidence: ${p.confidence?.toFixed(2)})
   Context: ${p.context}
   Invariant: ${p.invariant}
   Dependencies: ${JSON.stringify(p.dependencies || [])}
   Expected: ${p.expected_impact || 'unknown'}`).join('\n\n')}

# Selection Rules (ARGUS Section 6 — Optimization Selector):
1. Don't always pick top-ranked — sample from distribution to maintain exploration
2. Resolve coupling: if optimization A requires B, include both in sequence
3. Output an EXECUTABLE PLAN: ordered list of transformations to apply
4. For each step, preserve the invariants from the proposal context
5. If a global intrusive change is selected, it typically must come before
   local source changes that depend on its data layout

Return the selected optimization plan.`, {
    label: `select-${outerIter}`,
    phase: 'Select',
    schema: {
      type: 'object',
      properties: {
        selected_plan: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              step: { type: 'number' },
              optimization: { type: 'string' },
              context: { type: 'string' },
              invariant: { type: 'string' },
              rationale: { type: 'string' },
            },
            required: ['optimization', 'context', 'invariant'],
          },
        },
        exploration_note: { type: 'string' },
      },
      required: ['selected_plan'],
    },
  })

  const selectedPlan = selectResult.selected_plan || []
  log(`Selected: ${selectedPlan.map(s => s.optimization).join(' → ')}`)

  // ===========================================================================
  // Phase 4: Lower — Implement transformations with invariants
  // ===========================================================================
  phase('Lower')

  let currentCode = bestKernelCode
  const loweringResults = []

  for (let stepIdx = 0; stepIdx < Math.min(selectedPlan.length, INNER_STEPS); stepIdx++) {
    const step = selectedPlan[stepIdx]

    const lowerResult = await agent(`You are the ARGUS Lowering Agent (Section 6).
Implement the selected optimization transformation directly in the kernel code.

# Current Kernel Code:
\`\`\`
${currentCode.substring(0, 6000)}
\`\`\`

# Optimization to Apply: "${step.optimization}"
# Context: ${step.context}
# Required Data-Flow Invariant: ${step.invariant}

# Hardware: ${HARDWARE_TARGET}
# Computation: ${computationType}

# Lowering Rules (ARGUS Section 6 — Lowering Agent):
1. Apply rewrites at the level of tiles, layouts, and thread-level control
2. Insert or update TAG FUNCTIONS to propagate symbolic annotations:
   - Tag functions assign symbolic coordinates to tensor elements
   - They propagate through control and data flow
3. Insert TAG ASSERTIONS to encode the required invariant:
   - Conformity assertions: "assert tag(A[...]) == tag(B[...])"
     (paired operands must carry compatible tags)
   - Non-conformity assertions: "assert tag(X[...]) != tag(Y[...])"
     (concurrent producers must not write same location)
4. Retrieve relevant patterns from knowledge base as few-shot examples
5. The output must be a COMPLETE, COMPILABLE kernel

# Knowledge Base Examples for "${step.optimization}":
${KNOWLEDGE_BASE[step.category || 'global_intrusive']?.filter(k => k.includes(step.optimization.split(':')[0].toLowerCase().replace(/\s+/g, '_')))?.join('\n') || 'Use general optimization patterns.'}

Return the transformed kernel code with invariants.`, {
      label: `lower-${outerIter}-step${stepIdx}`,
      phase: 'Lower',
      schema: {
        type: 'object',
        properties: {
          transformed_code: { type: 'string' },
          invariants_added: { type: 'array', items: { type: 'string' } },
          transformation_description: { type: 'string' },
          potential_issues: { type: 'array', items: { type: 'string' } },
        },
        required: ['transformed_code', 'invariants_added'],
      },
    })

    if (lowerResult && lowerResult.transformed_code) {
      currentCode = lowerResult.transformed_code
      loweringResults.push({
        step: step.optimization,
        invariants: lowerResult.invariants_added,
        description: lowerResult.transformation_description,
      })
    }
  }

  log(`Lowered: ${loweringResults.length} transformations applied`)

  // ===========================================================================
  // Phase 5: Validate — Invariant checking + tests + profiling
  // ===========================================================================
  phase('Validate')

  const validateResult = await agent(`You are the ARGUS Validator Agent (Section 6).
Validate the transformed kernel through invariant checking, unit tests, and profiling.

# Transformed Kernel:
\`\`\`
${currentCode.substring(0, 6000)}
\`\`\`

# Invariants to Check:
${loweringResults.map(r => r.invariants.map(inv => `- ${inv}`).join('\n')).join('\n')}

# Validation Steps (ARGUS Section 5 + Section 6):

## Step 1: Compile-Time Invariant Validation
Simulate the ARGUS static analysis:
- Track tag propagation through assignments and shared memory accesses
- Check tag assertions at all use sites
- For any violation, produce a CONCRETE COUNTEREXAMPLE:
  "Violation at [program point]: thread [T] holds element [E] with tag [X],
   but assertion requires tag [Y]"
- This is the DENSE FEEDBACK that makes ARGUS work

## Step 2: Functional Correctness
${TEST_CMD ? `Run: ${TEST_CMD}` : 'Perform static correctness analysis:'}
- Check for race conditions (concurrent writes to same shared memory)
- Check for out-of-bounds accesses
- Verify tiling covers the full problem (no elements missed)
- Check synchronization (barriers between producers and consumers)

## Step 3: Performance Profiling
${BENCH_CMD ? `Run: ${BENCH_CMD}` : 'Estimate performance:'}
- Measure or estimate throughput (TFLOPS)
- Compare against baseline: ${bestThroughput} TFLOPS
- Identify remaining bottlenecks

## Step 4: Compute Reward Signal (ARGUS ICRL reward)
Reward = f(correctness, invariant_satisfaction, performance)
- If invariants violated: negative process reward (dense signal for what went wrong)
- If tests fail: zero reward
- If correct + improved: positive reward proportional to speedup

Return validation results.`, {
    label: `validate-${outerIter}`,
    phase: 'Validate',
    schema: {
      type: 'object',
      properties: {
        invariants_satisfied: { type: 'boolean' },
        invariant_violations: { type: 'array', items: { type: 'object', properties: { assertion: { type: 'string' }, counterexample: { type: 'string' }, program_point: { type: 'string' } } } },
        tests_pass: { type: 'boolean' },
        correctness_issues: { type: 'array', items: { type: 'string' } },
        throughput_tflops: { type: 'number' },
        speedup: { type: 'number' },
        remaining_bottlenecks: { type: 'array', items: { type: 'string' } },
        reward: { type: 'number' },
        reward_breakdown: { type: 'string' },
      },
      required: ['invariants_satisfied', 'tests_pass', 'throughput_tflops', 'reward'],
    },
  })

  // Record results
  const succeeded = validateResult.invariants_satisfied && validateResult.tests_pass
  const newThroughput = validateResult.throughput_tflops || 0

  for (const step of loweringResults) {
    optimizationHistory.push({
      optimization: step.step,
      outcome: succeeded ? (newThroughput > bestThroughput ? 'improved' : 'correct_no_gain') : 'failed',
      throughput_delta: newThroughput - bestThroughput,
      invariant_violations: validateResult.invariant_violations?.length || 0,
    })
  }

  if (validateResult.invariant_violations?.length > 0) {
    for (const v of validateResult.invariant_violations) {
      invariantViolationLog.push(`[iter${outerIter}] ${v.assertion}: ${v.counterexample} at ${v.program_point}`)
    }
  }

  // Update best if improved
  if (succeeded && newThroughput > bestThroughput) {
    bestKernelCode = currentCode
    bestThroughput = newThroughput
    candidateBeam.push({ code: currentCode, throughput: newThroughput, label: `iter${outerIter}` })
    candidateBeam.sort((a, b) => b.throughput - a.throughput)
    candidateBeam = candidateBeam.slice(0, 3)
    log(`NEW BEST: ${bestThroughput.toFixed(1)} TFLOPS (${validateResult.speedup?.toFixed(2) || '?'}x) ✓`)
  } else if (!succeeded) {
    log(`FAILED: ${validateResult.invariant_violations?.length || 0} invariant violations, ${validateResult.correctness_issues?.length || 0} correctness issues`)
  } else {
    log(`Correct but no improvement: ${newThroughput.toFixed(1)} TFLOPS`)
  }

  // ===========================================================================
  // Phase 6: Learn — ICRL policy update via text gradients
  // ===========================================================================
  phase('Learn')

  const learnResult = await agent(`You are performing the ICRL policy update for the ARGUS planner (Algorithm 1, Section 6).

# This Iteration's Results:
- Optimizations attempted: ${loweringResults.map(r => r.step).join(' → ')}
- Outcome: ${succeeded ? 'SUCCESS' : 'FAILED'}
- Throughput: ${newThroughput.toFixed(1)} TFLOPS (was ${bestThroughput.toFixed(1)})
- Reward: ${validateResult.reward}
- Invariant violations: ${JSON.stringify(validateResult.invariant_violations || [])}
- Remaining bottlenecks: ${JSON.stringify(validateResult.remaining_bottlenecks || [])}

# Full Optimization History:
${optimizationHistory.map(h => `- ${h.optimization}: ${h.outcome} (Δ${h.throughput_delta?.toFixed(2) || '?'} TFLOPS, ${h.invariant_violations} violations)`).join('\n')}

# Current Planner Policy:
${plannerPolicy || '(empty — this is the initial update)'}

# ICRL Update Rules (ARGUS Algorithm 1):
1. POLICY EVALUATION: Assess which optimization choices led to improvements vs failures
2. ANALYZE: Compute text-level gradients — what should the planner do MORE of / LESS of?
3. PARAMETER UPDATE: Produce an updated planner policy (natural language guidance)

Key insights from ARGUS ICRL:
- Process rewards from invariant violations give DENSE signal (not just pass/fail)
- Learn which invariants are effective at guiding transformations
- Learn ordering dependencies (e.g., bank conflict mitigation before pipelining)
- Bind knowledge base entries to this specific kernel instance

Produce the updated planner policy — this will guide the next iteration's proposals.`, {
    label: `learn-${outerIter}`,
    phase: 'Learn',
    schema: {
      type: 'object',
      properties: {
        policy_update: { type: 'string' },
        key_learnings: { type: 'array', items: { type: 'string' } },
        effective_optimizations: { type: 'array', items: { type: 'string' } },
        anti_patterns: { type: 'array', items: { type: 'string' } },
        next_priority: { type: 'string' },
      },
      required: ['policy_update', 'key_learnings'],
    },
  })

  plannerPolicy = learnResult.policy_update
  log(`ICRL update: ${learnResult.key_learnings.length} learnings | Next priority: ${learnResult.next_priority || 'continue'}`)
}

// =============================================================================
// Final Report
// =============================================================================
const finalReport = await agent(`Write a concise technical report on the ARGUS optimization results.

# ARGUS Optimization Results
- Computation: ${computationType}
- Hardware: ${HARDWARE_TARGET}
- Kernel spec: ${KERNEL_SPEC}
- Baseline throughput: ${baselineResult.throughput_tflops} TFLOPS
- Final best throughput: ${bestThroughput} TFLOPS
- Overall speedup: ${bestThroughput / (baselineResult.throughput_tflops || 1)}x
- ICRL iterations: ${ITERATIONS}
- Total optimization steps: ${optimizationHistory.length}

# Optimization History:
${optimizationHistory.map((h, i) => `${i + 1}. ${h.optimization}: ${h.outcome} (Δ${h.throughput_delta?.toFixed(2)} TFLOPS)`).join('\n')}

# Invariant Violation Log (dense feedback):
${invariantViolationLog.slice(0, 10).join('\n') || 'No violations recorded.'}

# Final Planner Policy (learned):
${plannerPolicy}

# Final Kernel:
\`\`\`
${bestKernelCode.substring(0, 4000)}
\`\`\`

Write:
1. Optimization journey: which transformations worked, in what order, and why
2. Role of data-flow invariants: how did violation feedback guide fixes?
3. ICRL learning: how did the planner policy evolve?
4. Remaining gap to peak theoretical performance
5. Recommendations for further optimization`, {
  label: 'final-report',
  phase: 'Learn',
})

return {
  computation_type: computationType,
  hardware_target: HARDWARE_TARGET,
  baseline_throughput_tflops: baselineResult.throughput_tflops || 0,
  best_throughput_tflops: bestThroughput,
  overall_speedup: bestThroughput / (baselineResult.throughput_tflops || 1),
  iterations_completed: ITERATIONS,
  optimization_steps: optimizationHistory.length,
  optimization_history: optimizationHistory,
  invariant_violations_total: invariantViolationLog.length,
  planner_policy_final: plannerPolicy,
  candidate_beam: candidateBeam.map(c => ({ label: c.label, throughput: c.throughput })),
  best_kernel_code: bestKernelCode,
  report: finalReport,
}
