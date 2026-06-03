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
//     validation_command: 'python validate.py --kernel-path KERNEL_PATH',
//     evaluation_command: 'python benchmark.py --kernel-path KERNEL_PATH --output benchmark.csv',
//     promotion_criteria: 'Speedup >= 1.2x over baseline AND passes validation',
//     max_candidates: 5,
//   }})
//
// =============================================================================

// ---- Task Contract (from args) ----
const KERNEL_PATH = args.kernel_path
const TASK_NAME = args.task_name || 'unnamed-task'
const OBJECTIVE = args.objective || 'Optimize the target kernel'
const CORRECTNESS = args.correctness_requirements || 'Must produce correct output'
const PERF_TARGET = args.performance_target || 'Improve over baseline'
const ALLOWED = args.allowed_approaches || 'Any approach within the language'
const VALIDATION_CMD = args.validation_command
const EVAL_CMD = args.evaluation_command || VALIDATION_CMD
const PROMOTION = args.promotion_criteria || 'Passes validation and improves target metric'
const MAX_CANDIDATES = args.max_candidates || 5

// ---- State ----
let baselineCode = ''
let baselineMetrics = {}
let currentBestCode = ''
let currentBestMetrics = {}
let currentBestId = 'baseline'
let candidates = [] // candidates.jsonl equivalent
let iteration = 0
let promotionMet = false

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

const inspection = await agent(`You are in a task implementation workspace. Inspect the workspace and the target kernel.

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
- Use \`cuda-kernel-development\` skill for hardware-aware CUDA development guidance.

# Instructions
1. Read the kernel file and any surrounding code (headers, utils, tests).
2. Identify the current baseline behavior and how it is validated.
3. Understand the code structure: key functions, data flow, memory patterns.
4. Look for existing tests, benchmarks, or validation scripts in the workspace.
5. Check if there are docs/draft.md or docs/plan.md already.
6. Research relevant optimization techniques for this kernel type using available domain knowledge.

Return a structured analysis.`, {
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
})

baselineCode = inspection.kernel_code
currentBestCode = baselineCode

log(`Inspected: ${inspection.key_functions.length} key functions, approach: ${inspection.current_approach}`)

// =============================================================================
// Phase 2: Plan — Write draft.md, convert to executable plan
//
// KDA step 3: "Make the agent write docs/draft.md."
// KDA step 4: "Convert the draft into an executable plan."
// "Do not start implementation until the draft exists."
// =============================================================================
phase('Plan')

const draftResult = await agent(`Write a plan draft to docs/draft.md for this kernel optimization task.

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
- Use \`ncu-report-skill\` if NCU profiling data exists in the workspace (check profile/ directory).

# Draft Requirements (from prompts/basic-flow.md)
The draft MUST include:
1. The current baseline and how it is validated.
2. The main risks and unknowns.
3. Candidate implementation directions ranked by expected value and risk.
4. The first concrete implementation steps.
5. The exact validation and evaluation commands to run.
6. The evidence required to promote, revise, or reject a candidate.

Write the draft to docs/draft.md. Then return the draft content.`, {
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
})

const directions = draftResult.candidate_directions
log(`Draft written. ${directions.length} candidate directions identified.`)

// Convert draft to executable plan
const planResult = await agent(`Convert the draft into an executable plan at docs/plan.md.

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
})

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

for (iteration = 0; iteration < Math.min(planCandidates.length, MAX_CANDIDATES) && !promotionMet; iteration++) {
  const candidate = planCandidates[iteration]
  const candidateId = `candidate-${iteration + 1}`

  log(`\n=== Candidate ${iteration + 1}/${planCandidates.length}: ${candidate.title} ===`)

  // ---- Phase 3: Implement ----
  phase('Implement')

  const impl = await agent(`Implement this optimization candidate as a complete, compilable kernel.

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
- Use \`cuda-kernel-development\` skill for hardware-aware CUDA patterns (shared memory tiling, warp primitives, occupancy tuning).

# Requirements
1. Output a COMPLETE file — all includes, definitions, functions.
2. Keep function signatures unchanged unless the plan explicitly requires changes.
3. Apply ONLY the changes described in this candidate. Do not combine with other optimizations.
4. Must be functionally correct: ${CORRECTNESS}

Return the complete implementation code.`, {
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
  })

  const candidateCode = impl.code

  // ---- Phase 4: Validate ----
  phase('Validate')

  const validation = await agent(`Validate this kernel candidate: run correctness and performance tests.

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
   - Race conditions (shared memory access, __syncthreads placement)
   - Out-of-bounds (index computations)
   - Missing synchronization
   - Correct reductions (warp shuffle logic)
3. CORRECTNESS VALIDATION — If validation_command is provided:
   - Run it: \`${VALIDATION_CMD || 'N/A'}\`
   - Report whether it passes or fails, and any error output.
   - If not provided, rely on static analysis only.
4. PERFORMANCE EVALUATION — If evaluation_command is provided:
   - Run it: \`${EVAL_CMD || 'N/A'}\`
   - Extract the measured latency/throughput from the output.
   - Compare against baseline: ${currentBestMetrics.latency_ms ? currentBestMetrics.latency_ms + 'ms' : 'not yet measured'}.
   - If not provided, give your best estimate based on the optimization applied.
5. If NCU profiling data is available or can be generated, use the \`ncu-report-skill\` to analyze bottlenecks.

Return the validation results with MEASURED values when available, estimates only as fallback.`, {
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
  })

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

const report = await agent(`Write the final optimization report.

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

Also write the candidates list to candidates.jsonl (one JSON object per line).`, {
  label: 'final-report',
  phase: 'Report',
})

return {
  task_name: TASK_NAME,
  candidates_tried: candidates.length,
  promoted: candidates.filter(c => c.status === 'promoted').length,
  rejected: candidates.filter(c => c.status === 'rejected').length,
  final_best_metrics: currentBestMetrics,
  candidates: candidates,
}
