export const meta = {
  name: 'ako4x-kernel-optimizer',
  description: 'Multi-round closed-loop GPU kernel optimization with experience accumulation (AKO4X methodology)',
  whenToUse: 'When you need to iteratively optimize any GPU kernel (Triton/CUDA/CuTe DSL/TileLang/C++) through plan-execute-benchmark-learn cycles. Uses NCU profiling for evidence-based optimization, accumulates lessons across rounds with two-layer WHEN (narrow + broad), and archives variants with structured 5-section headers.',
  phases: [
    { title: 'Setup', detail: 'Read target kernel, detect language, create workspace, establish baseline' },
    { title: 'Round-Init', detail: 'Select parent variant with cross-round reflection, initialize round state' },
    { title: 'Iterate', detail: 'Inner iteration loop: smoke test → bench → commit → log → repeat until plateau' },
    { title: 'Archive', detail: 'Pre-archive gates, archive variant with 5-section header, update TRAPS.md' },
    { title: 'Retrospect', detail: 'Phase-2 harness retrospective (Mode 3 only), extract lessons' },
    { title: 'Report', detail: 'Final report with best variant, lessons, dead-ends, open directions' },
  ],
  requiredSkills: [],
  optionalSkills: [
    'ako4x-triton',
    'ako4x-cuda',
    'ako4x-cute-dsl',
    'ako4x-tilelang',
    'ako4x-cpp',
    'ako4x-bench',
  ],
  skill_binding_mode: 'prompt_reference_only',
}

// =============================================================================
// AKO4X Multi-Round Closed-Loop Kernel Optimization Workflow
// =============================================================================
//
// Faithfully implements the AKO4X methodology from:
//   - templates/task.md (iteration protocol)
//   - master/MASTER.md (10-step round loop)
//   - templates/agent/lessons-convention.md (5-section header, two-layer WHEN)
//   - templates/skills/bench/ (noise-aware benchmarking)
//   - templates/retrospective.md (Phase-2 harness retrospective)
//
// Key AKO4X principles:
//   1. Two-level iteration: Round (multi-round loop) + Iteration (within each round)
//   2. Pre-commit Expected: write hypothesis BEFORE benching to catch retrofitted explanations
//   3. Smoke test → Full bench: don't waste time on full bench if smoke test fails
//   4. Pre-archive gates: silent-skip detection + library-delegation check
//   5. Lessons colocated in kernel.py header, NOT in separate files
//   6. Dead-ends are expectation priors with WHY, not prohibitions
//   7. Open directions are forensic signal, not a checklist to relay
//   8. TRAPS.md accumulates cross-variant silent-bug patterns
//   9. Failed rounds archived for forensic value
//
// Usage:
//   Workflow({name: 'ako4x-kernel-optimizer', args: {
//     kernel_path: '/path/to/kernel.py',
//     op_description: 'Multi-head Latent Attention paged decode',
//     kernel_language: 'triton',                    // or 'cuda', 'cute-dsl', 'tilelang', 'cpp', 'pytorch'
//     benchmark_command: 'bash scripts/bench.sh',   // full bench command
//     smoke_test_command: '',                        // quick compile+correctness check (default: bench --first 1)
//     correctness_command: '',                       // optional: separate correctness check
//     ncu_binary: 'ncu',                             // path to ncu (optional)
//     harness_path: '',                              // standalone profiling harness (optional)
//     harness_build_cmd: '',                         // build command for harness (optional)
//     harness_run_args: '',                          // runtime args for harness (optional)
//     kernel_name_regex: '',                         // ncu -k regex (optional)
//     exp_dir: '/tmp/ako4x_exp',                     // experiment output directory
//     rounds: 5,                                     // max optimization rounds
//     iters_per_round: 5,                            // max iterations per round
//     breadth: 3,                                    // hypotheses per round
//     samples_per_hypothesis: 2,                     // variants per hypothesis
//     target_gpu: 'b200',                            // target GPU
//     mode: 2,                                       // 2 = static harness, 3 = harness co-evolution
//     use_ako4x_skills: true,                        // reference AKO4X SKILLs for DSL knowledge
//   }})
//
// =============================================================================

const KERNEL_PATH = args.kernel_path
const OP_DESC = args.op_description || 'GPU kernel'
const KERNEL_LANG = args.kernel_language || 'auto'
const BENCHMARK_CMD = args.benchmark_command || ''
const SMOKE_TEST_CMD = args.smoke_test_command || ''
const CORRECTNESS_CMD = args.correctness_command || ''
const NCU_BINARY = args.ncu_binary || 'ncu'
const HARNESS_PATH = args.harness_path || ''
const HARNESS_BUILD_CMD = args.harness_build_cmd || ''
const HARNESS_RUN_ARGS = args.harness_run_args || ''
const KERNEL_NAME_REGEX = args.kernel_name_regex || ''
const EXP_DIR = args.exp_dir || '/tmp/ako4x_exp'
const ROUNDS = args.rounds || 5
const ITERS_PER_ROUND = args.iters_per_round || 5
const BREADTH = args.breadth || 3
const SAMPLES_PER_HYPOTHESIS = args.samples_per_hypothesis || 2
const TARGET_GPU = args.target_gpu || 'b200'
const MODE = args.mode || 2  // 2 = static harness, 3 = harness co-evolution
const USE_AKO4X_SKILLS = args.use_ako4x_skills !== false

// Model routing by agent role
const MODEL = {
  mechanical: args.model_mechanical || 'haiku',  // runs shell/scripts, parses output, bookkeeping
  profile: args.model_profile || 'sonnet',       // profiling / metric analysis
  judgment: args.model_judgment || 'opus',       // planning, code gen/edit/debug, final report
}

// Token budget guard
const EST_PER_ROUND = args.est_tokens_per_round || 60000

// =============================================================================
// State: cross-round memory
// =============================================================================
let experienceMemory = []      // learned optimization patterns (for prompt injection)
let deadEnds = []              // tried-and-failed directions with WHY
let traps = []                 // cross-variant silent-bug patterns (→ TRAPS.md)
let bestScore = null
let bestKernelCode = null
let bestVariantName = null
let baselineScore = null
let baselineKernelCode = null
let roundHistory = []          // archive of all rounds
let consecutiveNoImprove = 0
let currentParentName = 'original'

// =============================================================================
// AKO4X SKILLs Reference
// =============================================================================
function getDslSkillHint(lang) {
  const skills = {
    'triton': `Read the triton SKILL (.claude/skills/triton/) for:
- num_warps/num_stages tuning (tiny tiles prefer fewer warps)
- MMA tile picker regression at small-N fp8 tiles
- PDL bindings: gdc_launch_dependents() / gdc_wait()
- Split-K reduce: prefer 2D-tile merged form over scalar tl.static_range`,
    'cuda': `Read the cuda SKILL (.claude/skills/cuda/) for:
- Shared memory tiling with bank-conflict-free layouts
- Register pressure vs occupancy tradeoff
- Warp-level primitives (shfl, vote)
- __launch_bounds__ for occupancy control`,
    'cute-dsl': `Read the cute-dsl SKILL (.claude/skills/cute-dsl/) for:
- Layout composition (swizzle, composition)
- Tensor core utilization (MMA-compatible shapes)
- Shared memory configuration`,
    'tilelang': `Read the tilelang SKILL (.claude/skills/tilelang/) for:
- Schedule primitives (split, reorder, bind)
- Memory hierarchy (shared memory, registers)
- Thread block mapping`,
    'cpp': `Read the cpp SKILL (.claude/skills/cpp/) for:
- pybind11 bindings
- CPU/GPU kernel patterns`,
    'pytorch': 'Use torch.compile or custom ops. Focus on memory access patterns and operator fusion.',
  }
  return skills[lang] || 'No DSL-specific SKILL available. Use general GPU optimization knowledge.'
}

function getBenchmarkMethodology() {
  return `Noise-aware benchmarking (from bench SKILL):
- A/B compare: run old and new back-to-back in same session to cancel drift
- Variance check: run 3-5 times to measure noise floor
- Single-run headlines can drift 5-15% without code change — NOT evidence
- For sub-1x deltas: ALWAYS use A/B compare
- For headline claims: run variance check to verify
- Subset filters (--first, --smoke) are for compile/correctness ONLY — NOT performance verdicts`
}

function formatExperience() {
  if (experienceMemory.length === 0) return ''
  return `\n\n# Learned Optimization Patterns (from previous rounds)\n${experienceMemory.map((e, i) => `${i + 1}. ${e}`).join('\n')}`
}

function formatDeadEnds() {
  if (deadEnds.length === 0) return ''
  return `\n\n# Dead-Ends (expectation priors — re-verify if toolchain shifted, do NOT blindly trust)\n${deadEnds.map((d, i) => `${i + 1}. ${d}`).join('\n')}`
}

function formatTraps() {
  if (traps.length === 0) return ''
  return `\n\n# TRAPS (cross-variant silent-bug patterns)\n${traps.map((t, i) => `${i + 1}. ${t}`).join('\n')}`
}

// =============================================================================
// Phase 1: Setup — Read kernel, detect language, create workspace, baseline
// =============================================================================
phase('Setup')

const setupResult = await agent(`Read the kernel file at: ${KERNEL_PATH}

Analyze it and return a JSON object with:
- kernel_code: the full source code
- kernel_language: detected language (triton/cuda/cute-dsl/tilelang/cpp/pytorch)
- op_type: operation type (e.g., "attention", "gemm", "rmsnorm", "softmax", "reduction")
- key_functions: list of key function names (especially __global__ or @triton.jit functions)
- current_approach: brief description of the implementation strategy
- launch_config: if visible, the grid/block dimensions used
- shared_memory_usage: whether and how shared memory is used
- memory_access_patterns: description of global memory access patterns
- potential_bottlenecks: initial observations about potential performance issues
- imports_used: list of all imports (to check for library delegation later)

Return ONLY the JSON object.`, {
  label: 'read-baseline',
  phase: 'Setup',
  model: MODEL.mechanical,
  schema: {
    type: 'object',
    properties: {
      kernel_code: { type: 'string' },
      kernel_language: { type: 'string' },
      op_type: { type: 'string' },
      key_functions: { type: 'array', items: { type: 'string' } },
      current_approach: { type: 'string' },
      launch_config: { type: 'string' },
      shared_memory_usage: { type: 'string' },
      memory_access_patterns: { type: 'string' },
      potential_bottlenecks: { type: 'string' },
      imports_used: { type: 'array', items: { type: 'string' } },
    },
    required: ['kernel_code', 'kernel_language', 'op_type', 'key_functions', 'current_approach'],
  },
})

const detectedLang = KERNEL_LANG === 'auto' ? setupResult.kernel_language : KERNEL_LANG
baselineKernelCode = setupResult.kernel_code
bestKernelCode = baselineKernelCode

log(`Kernel: ${setupResult.op_type} (${detectedLang}) | Functions: ${setupResult.key_functions.join(', ')}`)

// Create workspace
await agent(`Create the optimization workspace:

\`\`\`bash
mkdir -p ${EXP_DIR}/{variants,round-logs,failed-rounds,ncu-profiles}
\`\`\`

Create ${EXP_DIR}/TRAPS.md:
\`\`\`markdown
# TRAPS — Silent-Bug Patterns

Cross-variant toolchain and methodology facts that apply regardless of which
variant is anchor. Create only when the first such fact is found.

*No traps discovered yet.*
\`\`\`

Create ${EXP_DIR}/state.json:
\`\`\`json
{
  "round": 0,
  "best_score": null,
  "best_variant": null,
  "kernel_language": "${detectedLang}",
  "op_type": "${setupResult.op_type}",
  "benchmark_command": "${BENCHMARK_CMD}",
  "mode": ${MODE},
  "started_at": "<current ISO timestamp>",
  "status": "initializing"
}
\`\`\`

Execute these commands.`, {
  label: 'create-workspace',
  phase: 'Setup',
  model: MODEL.mechanical,
})

// Establish baseline (NCU profile if available, else benchmark)
let baselineNcuProfile = ''
let baselineLatency = null

if (HARNESS_PATH || HARNESS_BUILD_CMD) {
  const ncuSetup = await agent(`Profile the baseline kernel with Nsight Compute (ncu).

# Environment
- NCU binary: ${NCU_BINARY}
- Experiment directory: ${EXP_DIR}
- Kernel file: ${KERNEL_PATH}
- Kernel name regex: ${KERNEL_NAME_REGEX || '(auto-detect)'}
- Harness path: ${HARNESS_PATH || '(need to build one)'}
- Harness build command: ${HARNESS_BUILD_CMD || '(need to determine)'}
- Harness run args: ${HARNESS_RUN_ARGS}

# Kernel Source:
\`\`\`${detectedLang}
${baselineKernelCode.substring(0, 4000)}
\`\`\`

# Instructions
1. mkdir -p ${EXP_DIR}/baseline/{harness,reports,analysis}
2. Build harness with -lineinfo
3. Run: ${NCU_BINARY} --set full --section PmSampling --section PmSampling_WarpStates -k "regex:${KERNEL_NAME_REGEX || '.*'}" -c 1 -o ${EXP_DIR}/baseline/reports/full_baseline ${HARNESS_PATH || '<harness>'} ${HARNESS_RUN_ARGS}
4. Extract: gpu__time_duration.sum, sm__throughput, dram__throughput, achieved_occupancy, registers_per_thread, top_stall_reason, sectors_per_request, ncu_rule_suggestions

Return structured profile results.`, {
    label: 'ncu-baseline',
    phase: 'Setup',
    model: MODEL.profile,
    schema: {
      type: 'object',
      properties: {
        latency_ms: { type: 'number' },
        sm_throughput_pct: { type: 'number' },
        dram_throughput_pct: { type: 'number' },
        achieved_occupancy_pct: { type: 'number' },
        registers_per_thread: { type: 'number' },
        top_stall_reason: { type: 'string' },
        top_stall_pct: { type: 'number' },
        sectors_per_request: { type: 'number' },
        ncu_rule_suggestions: { type: 'array', items: { type: 'string' } },
        bottleneck_diagnosis: { type: 'string' },
      },
      required: ['latency_ms', 'bottleneck_diagnosis'],
    },
  })

  baselineLatency = ncuSetup.latency_ms
  bestScore = baselineLatency
  baselineScore = baselineLatency
  baselineNcuProfile = `
## NCU Profile (Baseline)
- Latency: ${ncuSetup.latency_ms} ms | SM: ${ncuSetup.sm_throughput_pct || 'N/A'}% | DRAM: ${ncuSetup.dram_throughput_pct || 'N/A'}%
- Occupancy: ${ncuSetup.achieved_occupancy_pct || 'N/A'}% | Regs: ${ncuSetup.registers_per_thread || 'N/A'}
- Top Stall: ${ncuSetup.top_stall_reason || 'N/A'} (${ncuSetup.top_stall_pct || 'N/A'}%) | Sectors/Req: ${ncuSetup.sectors_per_request || 'N/A'}
- Diagnosis: ${ncuSetup.bottleneck_diagnosis}
- NCU Rules: ${(ncuSetup.ncu_rule_suggestions || []).map(s => `- ${s}`).join('\n') || 'N/A'}`
  log(`Baseline: ${baselineLatency}ms | ${ncuSetup.bottleneck_diagnosis}`)
} else if (BENCHMARK_CMD) {
  const benchResult = await agent(`Run benchmark to establish baseline.

# Command: ${BENCHMARK_CMD}
# Kernel: ${KERNEL_PATH}

Run the benchmark and extract the performance score (latency in ms or speedup).
Return the baseline performance metric.`, {
    label: 'benchmark-baseline',
    phase: 'Setup',
    model: MODEL.mechanical,
    schema: {
      type: 'object',
      properties: {
        score: { type: 'number' },
        latency_ms: { type: 'number' },
        raw_output: { type: 'string' },
      },
      required: ['score'],
    },
  })
  baselineScore = benchResult.score
  bestScore = baselineScore
  baselineLatency = benchResult.latency_ms || benchResult.score
  log(`Baseline score: ${baselineScore}`)
} else {
  log('WARNING: No benchmark command or NCU harness. Using static analysis only.')
}

// =============================================================================
// Multi-Round Loop
// =============================================================================
for (let round = 0; round < ROUNDS; round++) {
  if (typeof budget !== 'undefined' && budget.total && budget.remaining() < EST_PER_ROUND) { log(`token budget ~exhausted — stop`); break }

  log(`\n=== Round ${round + 1}/${ROUNDS} | Best: ${bestScore} | Parent: ${currentParentName} | Experience: ${experienceMemory.length} | Dead-ends: ${deadEnds.length} ===`)

  // =========================================================================
  // Phase 2: Round-Init — Select parent with cross-round reflection
  //
  // AKO4X MASTER.md step 1: Read README, TRAPS, _failed, and variant headers.
  // Treat "Open directions" as forensic signal, NOT a checklist to relay.
  // =========================================================================
  phase('Round-Init')

  const dslHint = USE_AKO4X_SKILLS ? getDslSkillHint(detectedLang) : ''
  const benchMethodology = getBenchmarkMethodology()
  const experienceSection = formatExperience()
  const deadEndsSection = formatDeadEnds()
  const trapsSection = formatTraps()

  // =========================================================================
  // Phase 3: Iterate — Inner iteration loop
  //
  // AKO4X task.md iteration protocol:
  //   1. git log --oneline (orient)
  //   2. Modify kernel
  //   3. Optional smoke test (compile + correctness only)
  //   4. Full bench (in background; draft next hypothesis while it runs)
  //   5. Git commit: bench(<score>): <description>
  //   6. Log to ITERATIONS.md (every labeled bench leaves a row)
  //   7. Pre-commit Expected: write hypothesis BEFORE benching
  // =========================================================================
  phase('Iterate')

  // Generate hypotheses for this round
  const planPromptBase = `You are a GPU kernel optimization expert. Generate ONE specific, evidence-based optimization hypothesis.

# Operation: ${OP_DESC} (${setupResult.op_type})
# Language: ${detectedLang} | GPU: ${TARGET_GPU}
# Round: ${round + 1} | Parent: ${currentParentName}

# Current Best Kernel:
\`\`\`${detectedLang}
${bestKernelCode.substring(0, 4000)}
\`\`\`

${baselineNcuProfile ? `# NCU PROFILING DATA (REAL MEASURED DATA):\n${baselineNcuProfile}` : '# No NCU data. Use static code analysis.'}

# Performance: Best=${bestScore} | Baseline=${baselineScore} | Speedup=${baselineScore ? (baselineScore / bestScore).toFixed(2) : 'N/A'}x
${experienceSection}
${deadEndsSection}
${trapsSection}

# DSL Guidance:
${dslHint}

# Benchmark Methodology:
${benchMethodology}

# Requirements:
1. CITE the specific bottleneck (NCU metric or code pattern)
2. Name the exact code region and transformation
3. Prefer STRUCTURAL changes over parameter tuning
4. Estimate expected improvement
5. Identify risk
6. DO NOT retry dead-ends unless you have new reasoning that flips the expectation`

  const plans = await parallel(
    Array.from({length: BREADTH}, (_, i) => () =>
      agent(`${planPromptBase}\n\nYou are planner #${i + 1}/${BREADTH}. Focus on a DIFFERENT angle than other planners.

Optimization levers (pick ONE):
- Memory access: coalescing, vectorization, prefetch, cache efficiency
- Parallelism: occupancy, warp utilization, grid/block dimensions
- Data reuse: shared memory, register tiling, double-buffering
- Compute: tensor cores, warp-level primitives, reduced synchronization
- Algorithmic: split-K, fusion, tiling strategy, kernel merging
- DSL-specific: num_warps/num_stages (Triton), layout composition (CuTe), schedule (TileLang)`, {
        label: `hypothesis-${round}-${i}`,
        phase: 'Iterate',
        model: MODEL.judgment,
        schema: {
          type: 'object',
          properties: {
            title: { type: 'string' },
            bottleneck: { type: 'string' },
            ncu_evidence: { type: 'string' },
            hypothesis: { type: 'string' },
            expected_impact: { type: 'string' },
            risk: { type: 'string' },
          },
          required: ['title', 'bottleneck', 'hypothesis', 'expected_impact'],
        },
      })
    )
  )

  const validPlans = plans.filter(Boolean)
  log(`Hypotheses: ${validPlans.map(p => p.title).join(' | ')}`)

  // Inner iteration loop: for each hypothesis, implement → smoke → bench → log
  let roundBest = null
  let roundIterations = []  // ITERATIONS.md entries
  let iterCount = 0

  for (const plan of validPlans) {
    if (iterCount >= ITERS_PER_ROUND) break

    // Implement variants for this hypothesis
    const impls = await parallel(
      Array.from({length: SAMPLES_PER_HYPOTHESIS}, (_, si) => () =>
        agent(`Implement this optimization hypothesis as a complete, working kernel.

# Original Kernel:
\`\`\`${detectedLang}
${bestKernelCode.substring(0, 4000)}
\`\`\`

# Hypothesis: "${plan.title}"
Bottleneck: ${plan.bottleneck}
Evidence: ${plan.ncu_evidence || 'static analysis'}
Plan: ${plan.hypothesis}

# DSL Guidance:
${dslHint}

# Requirements:
1. COMPLETE kernel file — all imports, helpers, main kernel, launch wrapper
2. FUNCTIONALLY CORRECT (same output as baseline within tolerance)
3. Apply the hypothesis faithfully
4. Keep function signature unchanged
5. Add comment at top explaining the optimization
6. This is variant ${si + 1}/${SAMPLES_PER_HYPOTHESIS}

# Anti-patterns (dead-ends from previous rounds):
${deadEndsSection}

Return the complete kernel code.`, {
          label: `impl-${round}-${plan.title.substring(0, 10)}-v${si}`,
          phase: 'Iterate',
          model: MODEL.judgment,
          isolation: 'worktree',
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

    // For each implemented variant: smoke test → full bench → log
    for (const impl of impls.filter(Boolean)) {
      if (iterCount >= ITERS_PER_ROUND) break
      iterCount++

      const iterLabel = `r${round + 1}-iter${iterCount}-${plan.title.substring(0, 20).replace(/[^a-zA-Z0-9]/g, '-').toLowerCase()}`

      // --- Pre-commit Expected (AKO4X: write hypothesis BEFORE benching) ---
      const expectedNote = `Expected: ${plan.hypothesis}. Predicted: ${plan.expected_impact}. Risk: ${plan.risk}.`
      log(`[${iterLabel}] ${expectedNote}`)

      // --- Smoke test (AKO4X: compile + correctness check, NOT performance verdict) ---
      const smokeCmd = SMOKE_TEST_CMD || (BENCHMARK_CMD ? `${BENCHMARK_CMD} --first 1` : '')
      let smokePassed = true

      if (smokeCmd) {
        const smokeResult = await agent(`Run smoke test for this kernel variant. This is a COMPILE + CORRECTNESS check only — NOT a performance verdict.

# Kernel Code:
\`\`\`${detectedLang}
${impl.code.substring(0, 4000)}
\`\`\`

# Smoke Test Command: ${smokeCmd}

Run the command and check:
1. Does it compile? (no compile errors)
2. Does it pass correctness? (no INCORRECT_NUMERICAL, no runtime errors)

Return pass/fail with error details if failed.`, {
          label: `smoke-${iterLabel}`,
          phase: 'Iterate',
          model: MODEL.mechanical,
          schema: {
            type: 'object',
            properties: {
              passed: { type: 'boolean' },
              error_message: { type: 'string' },
              compile_error: { type: 'boolean' },
              correctness_error: { type: 'boolean' },
            },
            required: ['passed'],
          },
        })

        smokePassed = smokeResult.passed
        if (!smokePassed) {
          log(`[${iterLabel}] SMOKE FAILED: ${smokeResult.error_message}`)
          roundIterations.push({
            iter: iterLabel,
            title: plan.title,
            score: null,
            passed: '0/N',
            notes: `SMOKE FAILED: ${smokeResult.error_message}`,
            expected: expectedNote,
          })
          // Record as dead-end with WHY
          deadEnds.push(`${plan.title}: ${smokeResult.error_message} (WHY: ${smokeResult.compile_error ? 'compile error' : 'correctness failure'} — code transformation broke ${smokeResult.compile_error ? 'syntax' : 'numerical correctness'})`)
          continue
        }
      }

      // --- Full bench (AKO4X: run in background, draft next hypothesis while it runs) ---
      const benchResult = await agent(`Run the full benchmark for this kernel variant. This IS the performance verdict.

# Kernel Code:
\`\`\`${detectedLang}
${impl.code.substring(0, 4000)}
\`\`\`

# Benchmark Command: ${BENCHMARK_CMD || 'static analysis only'}
# Baseline Score: ${baselineScore}

# Noise-aware protocol:
- Run the benchmark command
- If available, also run A/B compare with the parent kernel
- Report the score and any variance information

Return benchmark results.`, {
        label: `bench-${iterLabel}`,
        phase: 'Iterate',
        model: MODEL.mechanical,
        schema: {
          type: 'object',
          properties: {
            score: { type: 'number' },
            latency_ms: { type: 'number' },
            speedup: { type: 'number' },
            passed_workloads: { type: 'string' },
            ab_compare_delta: { type: 'string' },
            variance_info: { type: 'string' },
            raw_output: { type: 'string' },
          },
          required: ['score'],
        },
      })

      const speedup = baselineScore ? baselineScore / benchResult.score : benchResult.speedup || 1.0
      const passed = benchResult.passed_workloads || 'N/N'

      log(`[${iterLabel}] Score: ${benchResult.score} | Speedup: ${speedup.toFixed(2)}x | Passed: ${passed}`)

      // Log to ITERATIONS.md (AKO4X: every labeled bench leaves a row)
      roundIterations.push({
        iter: iterLabel,
        title: plan.title,
        score: benchResult.score,
        speedup: speedup,
        passed: passed,
        notes: benchResult.ab_compare_delta || '',
        expected: expectedNote,
      })

      // Update best if improved
      if (speedup > 1.0 && passed.includes('/')) {
        // Check if ALL workloads passed (e.g., "47/47" not "45/47")
        const [passedCount, totalCount] = passed.split('/').map(Number)
        if (passedCount === totalCount) {
          if (!roundBest || benchResult.score < roundBest.score) {
            roundBest = {
              plan: plan,
              code: impl.code,
              notes: impl.implementation_notes,
              score: benchResult.score,
              speedup: speedup,
              iterLabel: iterLabel,
              benchResult: benchResult,
            }
            bestKernelCode = impl.code
            bestScore = benchResult.score
            log(`[${iterLabel}] NEW ROUND BEST: ${speedup.toFixed(2)}x`)
          }
        }
      } else if (speedup < 1.0) {
        // Record regression as dead-end with WHY
        deadEnds.push(`${plan.title} (${iterLabel}): ${speedup.toFixed(2)}x regression (WHY: ${plan.bottleneck} — transformation did not address the actual bottleneck or introduced new overhead)`)
      }
    }
  }

  // Write ITERATIONS.md for this round
  await agent(`Write the iteration log for this round.

# Round: ${round + 1}
# Iterations:
${roundIterations.map(it => `| ${it.iter} | ${it.title} | ${it.score || 'FAILED'} | ${it.passed} | ${it.notes} | Expected: ${it.expected} |`).join('\n')}

Write to ${EXP_DIR}/round-logs/round-${round + 1}-iterations.md with format:
\`\`\`markdown
# Round ${round + 1} — Iteration Log

## Summary
| Iter | Title | Score | Speedup | Passed | Notes |
|------|-------|-------|---------|--------|-------|
${roundIterations.map(it => `| ${it.iter} | ${it.title} | ${it.score || 'FAILED'} | ${it.speedup ? it.speedup.toFixed(2) + 'x' : '-'} | ${it.passed} | ${it.notes} |`).join('\n')}

## Notes
${roundIterations.map(it => `- **${it.iter}**: ${it.expected}`).join('\n')}
\`\`\`

Execute this step.`, {
    label: `iterations-${round + 1}`,
    phase: 'Iterate',
    model: MODEL.mechanical,
  })

  // =========================================================================
  // Phase 4: Archive — Pre-archive gates + variant archival
  //
  // AKO4X MASTER.md step 8:
  //   1. Pre-archive silent-skip check
  //   2. Pre-archive library-delegation check
  //   3. Archive variant with 5-section header
  //   4. Update TRAPS.md if new silent-skip pattern found
  //   5. Archive failed rounds
  // =========================================================================
  phase('Archive')

  if (roundBest) {
    // --- Pre-archive Gate 1: Silent-skip detection ---
    const silentSkipCheck = await agent(`Check this kernel variant for silent-skip patterns.

# Variant: ${roundBest.iterLabel}
# Score: ${roundBest.score} (${roundBest.speedup.toFixed(2)}x vs baseline)
# Hypothesis: ${roundBest.plan.title}
# Implementation: ${roundBest.notes || 'see kernel code'}

# Kernel Code (excerpt):
\`\`\`${detectedLang}
${roundBest.code.substring(0, 3000)}
\`\`\`

# Check for silent-skip patterns:
1. Is the speedup PLAUSIBLE for the structural change actually made?
   - A +50% speedup from a minor parameter change is suspicious
   - A +10% speedup from a major algorithmic change might be underreported
2. Did the benchmark run correctly? (all workloads passed, no errors)
3. Does the code change actually address the stated bottleneck?
4. Are there any correctness concerns that slipped through?

Return verdict: is this a legitimate improvement or suspicious?`, {
      label: `silent-skip-check-${round + 1}`,
      phase: 'Archive',
      model: MODEL.judgment,
      schema: {
        type: 'object',
        properties: {
          is_legitimate: { type: 'boolean' },
          suspicion_level: { type: 'string' },
          concerns: { type: 'array', items: { type: 'string' } },
          recommendation: { type: 'string' },
        },
        required: ['is_legitimate'],
      },
    })

    // --- Pre-archive Gate 2: Library-delegation check ---
    const libDelegationCheck = await agent(`Check this kernel for library delegation (calling pre-built kernel libraries instead of writing own code).

# Kernel Code:
\`\`\`${detectedLang}
${roundBest.code.substring(0, 5000)}
\`\`\`

# BANNED (must NOT appear as the core compute):
- flashinfer.*, torch.ops.flashinfer.*
- deepgemm.*
- cuBLAS (cublasSgemm, cublasGemmEx, etc.)
- cuDNN (cudnnConvolutionForward, etc.)
- torch.matmul / torch.mm / F.linear as the ENTIRE operator (glue around own kernel is OK)
- F.scaled_dot_product_attention as the entire operator

# ALLOWED (fine as building blocks):
- CUTLASS / CuTe headers (template instantiation, not pre-built lib)
- Triton @triton.jit kernels
- TileLang @tl.program
- CuTe DSL @cute.kernel
- torch ops as GLUE around a kernel the agent wrote

# Check:
1. Does the kernel's core compute use any BANNED library?
2. Is it a thin wrapper around a pre-built kernel library?
3. Does it write its own compute logic?

Return verdict.`, {
      label: `lib-check-${round + 1}`,
      phase: 'Archive',
      model: MODEL.judgment,
      schema: {
        type: 'object',
        properties: {
          is_own_kernel: { type: 'boolean' },
          banned_libs_found: { type: 'array', items: { type: 'string' } },
          concerns: { type: 'string' },
        },
        required: ['is_own_kernel'],
      },
    })

    // --- Decide whether to archive ---
    const shouldArchive = silentSkipCheck.is_legitimate && libDelegationCheck.is_own_kernel

    if (shouldArchive) {
      // Archive variant with 5-section header (AKO4X lessons-convention.md)
      const variantName = `iter-${round + 1}-${roundBest.plan.title.substring(0, 30).replace(/[^a-zA-Z0-9]/g, '-').toLowerCase()}`

      await agent(`Archive this kernel variant with a 5-section AKO4X header.

# Variant: ${variantName}
# Score: ${roundBest.score} (${roundBest.speedup.toFixed(2)}x vs baseline)
# Parent: ${currentParentName}
# Hypothesis: ${roundBest.plan.title}
# Evidence: ${roundBest.plan.ncu_evidence || roundBest.plan.bottleneck}
# Iter: ${roundBest.iterLabel}

# Kernel Code:
\`\`\`${detectedLang}
${roundBest.code.substring(0, 5000)}
\`\`\`

# Instructions:
1. mkdir -p ${EXP_DIR}/variants/${variantName}/
2. Write kernel to ${EXP_DIR}/variants/${variantName}/kernel.py (or appropriate extension)
3. Add 5-section header at the TOP of the kernel file:

\`\`\`python
# ${variantName} — reference kernel.py header
#
# Identity
#   ${roundBest.score} (${roundBest.speedup.toFixed(2)}x vs baseline) — Round ${round + 1}, iter ${roundBest.iterLabel}.
#   Language: ${detectedLang}, GPU: ${TARGET_GPU}
#
# Delta from ${currentParentName}
#   <one paragraph: what this variant adds at architecture level>
#
# Lessons on this variant
#
#   <lesson title>
#     How:           <what was done>
#     Why:           <mechanism explanation — in terms of hardware behavior>
#     WHEN narrow:   <exact operational condition in this kernel>
#     WHEN broad:    <principle-level condition that transfers>
#
# Dead-ends tried on this variant
#   Each is an expectation prior. Re-verify cheaply if toolchain shifted.
#
#   - <dead-end>: <what happened> (WHY: <why it failed — the mechanism>)
#
# Open directions
#   <narrative of where a future session continuing this line might go>
#   Not a priority list, not a todo. This is forensic signal about what
#   the current session considered and chose not to act on.
\`\`\`

4. Fill in ALL sections based on the hypothesis, implementation, evaluation, and iteration log.
5. For Lessons: use the exact WHAT/WHY/WHEN format shown above.
6. For Dead-ends: each MUST include WHY — the mechanism of failure.
7. For Open directions: write as forensic narrative, NOT as a checklist.

Execute these steps.`, {
        label: `archive-${variantName}`,
        phase: 'Archive',
        model: MODEL.mechanical,
      })

      currentParentName = variantName
      roundHistory.push({
        round: round + 1,
        variant: variantName,
        score: roundBest.score,
        speedup: roundBest.speedup,
        hypothesis: roundBest.plan.title,
        iter: roundBest.iterLabel,
      })
      consecutiveNoImprove = 0

      // Update TRAPS.md if new silent-skip pattern found
      if (silentSkipCheck.concerns && silentSkipCheck.concerns.length > 0) {
        traps.push(...silentSkipCheck.concerns)
        await agent(`Update ${EXP_DIR}/TRAPS.md with new silent-skip patterns discovered this round.

# New patterns to add:
${silentSkipCheck.concerns.map((c, i) => `${i + 1}. ${c}`).join('\n')}

Append these to the existing TRAPS.md file. Format each as:
## <pattern-name>
**Symptom**: <what looked wrong>
**Root cause**: <why it happened>
**Detection**: <how to catch it>

Execute this step.`, {
          label: `update-traps-${round + 1}`,
          phase: 'Archive',
          model: MODEL.mechanical,
        })
      }
    } else {
      // Pre-archive gate failed — don't archive
      const reason = !silentSkipCheck.is_legitimate
        ? `silent-skip suspected: ${silentSkipCheck.concerns?.join('; ')}`
        : `library delegation: ${libDelegationCheck.banned_libs_found?.join(', ')}`

      log(`[${roundBest.iterLabel}] NOT ARCHIVED: ${reason}`)

      // Record in TRAPS if library delegation
      if (!libDelegationCheck.is_own_kernel) {
        traps.push(`Library delegation detected: ${libDelegationCheck.banned_libs_found?.join(', ')} — kernel must write its own compute, not call pre-built libraries`)

        // Propagate to next round (AKO4X MASTER.md: "carry that forward verbatim")
        deadEnds.push(`Library delegation: using ${libDelegationCheck.banned_libs_found?.join(', ')} as core compute is NOT allowed. The operator's core compute must be hand-written (DSL/.cu/CUTLASS-instantiated kernel). (WHY: pre-built libraries don't represent the agent's optimization work)`)
      }
    }
  } else {
    consecutiveNoImprove++
    log(`No improvement this round. Consecutive no-improve: ${consecutiveNoImprove}`)
  }

  // Archive failed rounds (AKO4X MASTER.md step 9)
  const failedIters = roundIterations.filter(it => it.score === null || (it.speedup && it.speedup < 1.0))
  if (failedIters.length > 0) {
    await agent(`Archive failed iterations for forensic value.

# Failed iterations this round:
${failedIters.map(it => `- ${it.iter}: ${it.title} — ${it.notes}`).join('\n')}

Write a summary to ${EXP_DIR}/failed-rounds/round-${round + 1}-failed.md:

\`\`\`markdown
# Round ${round + 1} — Failed Iterations

${failedIters.map(it => `## ${it.iter}: ${it.title}
- **Hypothesis**: ${it.expected}
- **Result**: ${it.notes}
- **Lesson**: <extract a reusable lesson about why this failed>
`).join('\n')}
\`\`\`

Execute this step.`, {
      label: `archive-failed-${round + 1}`,
      phase: 'Archive',
      model: MODEL.mechanical,
    })
  }

  // =========================================================================
  // Phase 5: Retrospect — Phase-2 harness retrospective (Mode 3 only)
  //
  // AKO4X retrospective.md:
  //   - Sub writes PROPOSALS.md with harness improvement proposals
  //   - Master evidence-gates and applies accepted edits
  //   - Session-best handoff: package best variant for master
  // =========================================================================
  phase('Retrospect')

  if (MODE === 3 && roundBest) {
    const retrospective = await agent(`You have completed Phase-1 optimization. Now do a HARNESS RETROSPECTIVE — only based on actual evidence from this session.

# Session Summary:
- Round: ${round + 1}
- Best variant: ${roundBest.iterLabel} (${roundBest.speedup.toFixed(2)}x)
- Hypotheses tested: ${validPlans.length}
- Iterations: ${iterCount}
- Dead-ends discovered: ${deadEnds.length}

# Iteration Log:
${roundIterations.map(it => `- ${it.iter}: ${it.title} — ${it.score || 'FAILED'} (${it.passed})`).join('\n')}

# Your task:
Re-read your optimization trail. Was anything in the harness — a SKILL doc, a benchmark command, a task description — actually misleading or absent or buggy in a way that cost iterations?

Common gaps:
- SKILL doc misleading or absent
- Benchmark command missing flags or unparseable output
- Task description unclear
- TRAPS entry contradicting what you actually saw
- A recurring trap with no warning

**Deletion is a valid proposal class** — if a SKILL section was empirically refuted by your trail, propose deleting it.

**If you have no concrete evidence**: output exactly "none". Do NOT invent.

Write your proposals to ${EXP_DIR}/proposals.md with format:
\`\`\`markdown
# Proposals

## proposal-1
- **scope**: <file path to edit>
- **evidence pointer**: <ITERATIONS.md line / commit SHA / bench output>
- **patch**: <diff or new content>
- **predicted utility**: <one sentence>
- **rationale**: <prose>

---

## proposal-2
...
\`\`\`

Or if no proposals: just write "none".

Execute this step.`, {
      label: `retrospect-${round + 1}`,
      phase: 'Retrospect',
      model: MODEL.judgment,
    })

    // Master gates the proposals (simplified for workflow)
    if (retrospective && !retrospective.includes('none')) {
      log(`[Round ${round + 1}] Retrospective produced proposals — review manually for harness improvements`)
    }
  }

  // Update state
  await agent(`Update ${EXP_DIR}/state.json:
\`\`\`json
{
  "round": ${round + 1},
  "best_score": ${bestScore},
  "best_variant": "${bestVariantName || currentParentName}",
  "kernel_language": "${detectedLang}",
  "op_type": "${setupResult.op_type}",
  "benchmark_command": "${BENCHMARK_CMD}",
  "mode": ${MODE},
  "started_at": "see-round-history",
  "status": "${consecutiveNoImprove >= 2 ? 'converged' : 'optimizing'}",
  "consecutive_no_improve": ${consecutiveNoImprove},
  "experience_count": ${experienceMemory.length},
  "dead_end_count": ${deadEnds.length},
  "trap_count": ${traps.length},
  "round_history": ${JSON.stringify(roundHistory)}
}
\`\`\`
Execute.`, {
    label: `update-state-${round + 1}`,
    phase: 'Archive',
    model: MODEL.mechanical,
  })

  log(`Round ${round + 1} done. Best: ${bestScore} | Consecutive no-improve: ${consecutiveNoImprove}`)

  // Stopping conditions
  if (consecutiveNoImprove >= 2) {
    log('STOPPING: 2 consecutive rounds with no improvement.')
    break
  }
}

// =============================================================================
// Final Report
// =============================================================================
phase('Report')

const finalReport = await agent(`Write a comprehensive optimization report.

# AKO4X Kernel Optimization — Final Report

## Summary
- Operation: ${OP_DESC} (${setupResult.op_type})
- Language: ${detectedLang} | GPU: ${TARGET_GPU}
- Mode: ${MODE} (${MODE === 2 ? 'static harness' : 'harness co-evolution'})
- Rounds: ${roundHistory.length} | Baseline: ${baselineScore} | Best: ${bestScore}
- Speedup: ${baselineScore ? (baselineScore / bestScore).toFixed(2) : 'N/A'}x
- Experience: ${experienceMemory.length} patterns | Dead-ends: ${deadEnds.length} | Traps: ${traps.length}

## Round History
${roundHistory.map(r => `- Round ${r.round}: ${r.variant} (${r.speedup.toFixed(2)}x, ${r.iter}) — ${r.hypothesis}`).join('\n')}

## Learned Patterns (with two-layer WHEN)
${experienceMemory.map((e, i) => `${i + 1}. ${e}`).join('\n\n')}

## Dead-Ends (expectation priors — re-verify if toolchain shifted)
${deadEnds.map((d, i) => `${i + 1}. ${d}`).join('\n')}

## TRAPS (cross-variant silent-bug patterns)
${traps.map((t, i) => `${i + 1}. ${t}`).join('\n')}

## Final Kernel:
\`\`\`${detectedLang}
${bestKernelCode.substring(0, 3000)}
\`\`\`

${baselineNcuProfile ? `## Initial NCU Diagnosis:\n${baselineNcuProfile.substring(0, 800)}` : ''}

Write a report covering:
1. Optimization journey (what was tried, what worked, what didn't)
2. Key lessons with two-layer WHEN (narrow + broad)
3. Anti-patterns discovered (dead-ends with WHY)
4. Remaining bottlenecks and Open directions (as forensic narrative, NOT checklist)
5. Recommendations for further optimization`, {
  label: 'final-report',
  phase: 'Report',
  model: MODEL.judgment,
})

return {
  baseline_score: baselineScore,
  best_score: bestScore,
  overall_speedup: baselineScore ? baselineScore / bestScore : null,
  rounds_completed: roundHistory.length,
  experience_patterns: experienceMemory,
  dead_ends: deadEnds,
  traps: traps,
  round_history: roundHistory,
  best_kernel_code: bestKernelCode,
  report: finalReport,
}
