export const meta = {
  name: 'kernelfoundry-kernel-optimization',
  description: 'Evolutionary MAP-Elites quality-diversity search with meta-prompt evolution and templated parameter tuning (KernelFoundry methodology)',
  whenToUse: 'When generating GPU kernels (SYCL/CUDA/Triton) from PyTorch operator specs and you need diverse, high-quality solutions across multiple optimization strategies. Prevents mode collapse via behavioral-descriptor archive. Best for KernelBench-style tasks, custom operators, and cross-platform (Intel/NVIDIA) kernel generation.',
  phases: [
    { title: 'Setup', detail: 'Parse task specification, establish baseline, initialize MAP-Elites archive' },
    { title: 'Select', detail: 'Sample parent(s) from archive using gradient-informed selection strategy' },
    { title: 'Vary', detail: 'LLM generates offspring via mutation guided by meta-evolved prompts' },
    { title: 'Evaluate', detail: 'Compile, validate correctness, benchmark, classify behavioral coordinates' },
    { title: 'Insert', detail: 'Update archive if offspring improves its behavioral cell; track transitions' },
    { title: 'Evolve-Prompts', detail: 'Meta-prompter analyzes outcomes, evolves prompt sections co-operatively' },
  ],
}

const WORKFLOW_SUITABILITY = {
  supported_languages: ['sycl', 'cuda', 'triton'],
  supported_problem_types: ['gpu-kernel-optimization', 'kernel-generation', 'kernel-search'],
  problem_types: ['quality-diversity kernel generation', 'MAP-Elites search over CUDA/SYCL/Triton candidates'],
  reason: 'KernelFoundry is a quality-diversity search workflow for GPU kernels in supported backends and needs descriptor/archive feedback.',
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

// =============================================================================
// KernelFoundry: Hardware-Aware Evolutionary GPU Kernel Optimization
// =============================================================================
//
// Source: "KernelFoundry: Hardware-aware evolutionary GPU kernel optimization"
//         Wiedemann, Leboutet, Paulitsch, Wofk, Ummenhofer
//         Intel Corporation, arXiv:2603.12440, 2026
//
// Three key mechanisms:
//   1. MAP-Elites with kernel-specific behavioral descriptors:
//      - d_mem ∈ {0,1,2,3}: memory access pattern (scalar → multi-level hierarchy)
//      - d_algo ∈ {0,1,2,3}: algorithmic structure (direct translation → novel algorithm)
//      - d_sync ∈ {0,1,2,3}: parallelism coordination (none → global coordination)
//      → 4³ = 64 behavioral cells, each holding the best kernel for that strategy
//
//   2. Meta-prompt evolution: 4 evolvable prompt sections co-evolve with kernels
//      - optimization_philosophy, optimization_strategies, common_pitfalls, analysis_guidance
//      - A separate meta-prompter LLM analyzes outcomes and proposes SEARCH/REPLACE edits
//
//   3. Templated parameter optimization: kernels can declare configurable parameters
//      (tile sizes, work-group dims, unroll factors) evaluated independently per config
//
// Gradient-informed evolution:
//   ∇F (fitness gradient): which behavioral directions improve fitness
//   ∇R (improvement-rate gradient): which directions have high improvement probability
//   ∇E (exploration gradient): which empty/low-quality cells to explore
//   Combined: ∇ = α∇F + β∇R + γ∇E (default: 0.4, 0.4, 0.2)
//
// Usage:
//   Workflow({name: 'kernelfoundry-kernel-optimization', args: {
//     problem_definition: 'class Model(nn.Module): ...',
//     op_description: 'Fused softmax + dropout',
//     language: 'sycl',                    // 'sycl', 'cuda', 'triton'
//     target_gpu: 'Intel Arc B580',          // or 'NVIDIA A6000', etc.
//     test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     descriptor_result_path: '/tmp/kernelfoundry_exp/descriptors/latest.json',
//     archive_update_result_path: '/tmp/kernelfoundry_exp/archive/updates.jsonl',
//     generations: 40,
//     meta_prompt_interval: 10,
//     speedup_target: 2.0,
//     selection_strategy: 'mixed',
//   }})
//
// =============================================================================

// --- Required Args ---
const TASK_SPEC = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const OP_DESC = args.op_description || 'GPU kernel'

// --- Optional Args ---
const TARGET_LANG = args.language || 'cuda'
const TARGET_HW = args.target_gpu || 'NVIDIA GPU'
const TEST_CMD = args.test_command || ''
const BENCH_CMD = args.benchmark_command || ''
const DESCRIPTOR_RESULT_PATH = args.descriptor_result_path || `${args.exp_dir || '/tmp/kernelfoundry_exp'}/descriptors/latest.json`
const ARCHIVE_UPDATE_RESULT_PATH = args.archive_update_result_path || `${args.exp_dir || '/tmp/kernelfoundry_exp'}/archive/updates.jsonl`
const GENERATIONS = args.generations || 30
const META_PROMPT_INTERVAL = args.meta_prompt_interval || 10
const SPEEDUP_TARGET = args.speedup_target || 2.0
const SELECTION_STRATEGY = args.selection_strategy || 'mixed'
const EXP_DIR = args.exp_dir || '/tmp/kernelfoundry_exp'
const KERNEL_PATH = args.kernel_path || ''
const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const EVIDENCE_MODE = (TEST_CMD && BENCH_CMD) ? 'measured' : 'conservative_missing_evidence'

// --- State: MAP-Elites Archive ---
// 4x4x4 = 64 cells, indexed by (d_mem, d_algo, d_sync)
let archive = {}           // key="d_mem,d_algo,d_sync" → {code, fitness, speedup, id}
let transitions = []       // [{parent_cell, child_cell, delta_f, outcome, gen}]
let generation = 0
let globalBest = { code: '', fitness: 0, speedup: 0, cell: '' }

// Meta-prompt evolvable sections
let metaPrompt = {
  optimization_philosophy: 'Prioritize memory bandwidth utilization before compute optimization. Minimize global memory accesses through data reuse.',
  optimization_strategies: 'Memory: coalesced loads, shared/local memory tiling, vectorized access (vec4/vec8). Compute: loop unrolling, fused operations, register blocking. Parallelism: work-group barriers for shared memory, sub-group shuffles for reductions.',
  common_pitfalls: 'Avoid bank conflicts in shared memory. Do not assume specific work-group sizes. Guard against out-of-bounds access for non-power-of-2 dimensions.',
  analysis_guidance: 'Before writing code: identify the memory access pattern, determine arithmetic intensity, choose tiling strategy based on data reuse distance.',
}
let promptArchive = []     // [{prompt_sections, best_fitness, generation}]

// Behavioral descriptor classification
const BEHAVIORAL_DIMS = {
  d_mem: ['scalar/strided/uncoalesced', 'coalesced/vectorized (vec4, aligned)', 'shared/local memory with explicit tiling', 'multi-level hierarchy (SLM + reg blocking + prefetch)'],
  d_algo: ['direct PyTorch translation', 'fused operations (single-pass)', 'reformulated algorithm (online norm, flash pattern)', 'novel/asymptotically improved algorithm'],
  d_sync: ['no synchronization (embarrassingly parallel)', 'work-group barriers', 'sub-group primitives (shuffles, reductions, broadcast)', 'global coordination (atomics, multi-pass with sync)'],
}

// Fitness function (Section 3.2)
function computeFitness(compiled, correct, speedup) {
  if (!compiled) return 0.0
  if (!correct) return 0.1
  const sNorm = Math.min(1.0, speedup / SPEEDUP_TARGET)
  return 0.5 + 0.5 * sNorm
}

// =============================================================================
// Phase 1: Setup — Parse task, baseline, initialize archive
// =============================================================================
phase('Setup')

const setupResult = await agent(`You are a GPU kernel optimization expert setting up the KernelFoundry evolutionary search.

# Task:
${KERNEL_PATH ? `Read kernel/operator from: ${KERNEL_PATH}` : ''}
${TASK_SPEC ? `\`\`\`python\n${TASK_SPEC.substring(0, 3000)}\n\`\`\`` : '(Determine from op_description)'}

# Operation: ${OP_DESC}
# Target language: ${TARGET_LANG} (SYCL/CUDA/Triton)
# Target hardware: ${TARGET_HW}
# Evidence contract:
- descriptor_result_path: ${DESCRIPTOR_RESULT_PATH}
- archive_update_result_path: ${ARCHIVE_UPDATE_RESULT_PATH}
- evidence_mode: ${EVIDENCE_MODE}
- If evidence_mode is conservative_missing_evidence, behavioral descriptors and MAP-Elites insertion decisions are not strict paper evidence.

# Setup Tasks:
1. Parse the operator specification (inputs, outputs, shapes, dtypes)
2. Establish PyTorch baseline performance: ${BENCH_CMD || '(estimate)'}
3. Identify the optimization feature space for this operator:
   - What memory access patterns are possible? (d_mem: 0-3)
   - What algorithmic reformulations exist? (d_algo: 0-3)
   - What parallelism levels apply? (d_sync: 0-3)
4. Initialize experiment: mkdir -p ${EXP_DIR}/{kernels,archive,prompts}

Return operator analysis and baseline.`, {
  label: 'setup',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      operator_code: { type: 'string' },
      operator_type: { type: 'string' },
      input_shapes: { type: 'string' },
      baseline_time_ms: { type: 'number' },
      hardware_info: { type: 'string' },
      feasible_cells: { type: 'array', items: { type: 'string' } },
    },
    required: ['operator_code', 'baseline_time_ms'],
  },
})

const operatorCode = setupResult.operator_code
const baselineTime = setupResult.baseline_time_ms

log(`Setup: ${setupResult.operator_type} on ${TARGET_HW} | Baseline: ${baselineTime}ms | Target: ${SPEEDUP_TARGET}x | Language: ${TARGET_LANG}`)

// =============================================================================
// MAP-Elites Evolutionary Loop
// =============================================================================

for (generation = 0; generation < GENERATIONS; generation++) {

  // ===========================================================================
  // Phase 2: Select — Sample parent(s) from archive
  // ===========================================================================
  phase('Select')

  // Gradient-informed selection (Section 3.3)
  const occupiedCells = Object.keys(archive)
  let selectedParent = null
  let selectionReason = ''

  if (occupiedCells.length === 0) {
    selectionReason = 'empty archive — generate from scratch'
  } else {
    // Mixed selection: uniform(0.3) + fitness-proportionate(0.4) + curiosity(0.3)
    const selIdx = (generation * 13 + 7) % 10
    if (selIdx < 3) {
      // Uniform: random occupied cell
      const cellKey = occupiedCells[(generation * 3) % occupiedCells.length]
      selectedParent = archive[cellKey]
      selectionReason = `uniform selection from cell [${cellKey}]`
    } else if (selIdx < 7) {
      // Fitness-proportionate
      const sorted = occupiedCells.map(k => archive[k]).sort((a, b) => b.fitness - a.fitness)
      selectedParent = sorted[generation % Math.min(3, sorted.length)]
      selectionReason = `fitness-proportionate (fitness=${selectedParent.fitness.toFixed(2)})`
    } else {
      // Curiosity-driven: pick cell with highest estimated improvement potential
      const sorted = occupiedCells.map(k => archive[k]).sort((a, b) => a.fitness - b.fitness)
      selectedParent = sorted[0]
      selectionReason = `curiosity-driven (lowest fitness cell, room to improve)`
    }
  }

  // Compute gradient hints from transition history
  let gradientHints = ''
  if (transitions.length >= 5) {
    const recentImprovements = transitions.filter(t => t.outcome === 'improvement').slice(-5)
    if (recentImprovements.length > 0) {
      gradientHints = `Recent successful directions: ${recentImprovements.map(t => `${t.parent_cell}→${t.child_cell} (+${t.delta_f.toFixed(2)})`).join(', ')}`
    }
  }

  log(`Gen ${generation + 1}/${GENERATIONS} | Archive: ${occupiedCells.length}/64 cells | Selection: ${selectionReason}`)

  // ===========================================================================
  // Phase 3: Vary — LLM generates offspring with meta-evolved prompts
  // ===========================================================================
  phase('Vary')

  const parentContext = selectedParent
    ? `\n# Parent Kernel (from cell [${selectedParent.cell}], fitness=${selectedParent.fitness.toFixed(2)}, speedup=${selectedParent.speedup.toFixed(2)}x):\n\`\`\`${TARGET_LANG}\n${selectedParent.code.substring(0, 4000)}\n\`\`\``
    : ''

  const varyResult = await agent(`You are a GPU kernel generator for the KernelFoundry evolutionary framework.
Generate a ${TARGET_LANG.toUpperCase()} kernel that implements the given operator.

# Operator to Implement:
\`\`\`python
${operatorCode.substring(0, 2500)}
\`\`\`

# Operation: ${OP_DESC}
# Target: ${TARGET_LANG} on ${TARGET_HW}
# Baseline: ${baselineTime}ms | Speedup target: ${SPEEDUP_TARGET}x

# === EVOLVED OPTIMIZATION GUIDANCE (meta-prompt) ===

## Optimization Philosophy:
${metaPrompt.optimization_philosophy}

## Optimization Strategies:
${metaPrompt.optimization_strategies}

## Common Pitfalls:
${metaPrompt.common_pitfalls}

## Analysis Guidance:
${metaPrompt.analysis_guidance}

# === END EVOLVED GUIDANCE ===
${parentContext}

${gradientHints ? `# Gradient Hints (from evolutionary history):\n${gradientHints}` : ''}

# Generation Requirements:
1. Produce a COMPLETE, COMPILABLE ${TARGET_LANG} kernel
2. Include all necessary headers/imports
3. If mutating a parent: make MEANINGFUL structural changes, not just parameter tweaks
4. Try to explore a DIFFERENT optimization strategy than the parent (different memory pattern, algorithm, or parallelism level)
5. You may optionally produce a TEMPLATED kernel with configurable parameters (tile_size, work_group_size, unroll_factor) alongside a dispatch function

Return the kernel code and its optimization strategy description.`, {
    label: `vary-${generation}`,
    phase: 'Vary',
    schema: {
      type: 'object',
      properties: {
        kernel_code: { type: 'string' },
        strategy_description: { type: 'string' },
        memory_pattern: { type: 'string' },
        algorithm_type: { type: 'string' },
        parallelism_level: { type: 'string' },
        is_templated: { type: 'boolean' },
        template_params: { type: 'array', items: { type: 'string' } },
      },
      required: ['kernel_code', 'strategy_description'],
    },
  })

  const offspringCode = varyResult?.kernel_code || ''

  // ===========================================================================
  // Phase 4: Evaluate — Compile + correctness + benchmark + classify
  // ===========================================================================
  phase('Evaluate')

  const evalResult = await agent(`You are a kernel evaluator for KernelFoundry. Evaluate this ${TARGET_LANG} kernel.

# Kernel Code:
\`\`\`${TARGET_LANG}
${offspringCode.substring(0, 5000)}
\`\`\`

# Reference Operator:
\`\`\`python
${operatorCode.substring(0, 1500)}
\`\`\`

# Evaluation Steps:
1. **Compile**: Can this ${TARGET_LANG} kernel compile? Check syntax, headers, type correctness.
${TEST_CMD ? `   Run: ${TEST_CMD}` : ''}
2. **Correctness**: Does it produce numerically equivalent output?
   Tolerance: relative precision ν < 0.01 in 99% of outputs.
3. **Performance**: Measure execution time.
${BENCH_CMD ? `   Run: ${BENCH_CMD}` : `   Estimate speedup over baseline (${baselineTime}ms).`}
   speedup = ${baselineTime}ms / kernel_time_ms

4. **Behavioral Classification** (assign coordinates 0-3 for each dimension):
   - d_mem (Memory Access Pattern):
     0: Scalar, strided, or uncoalesced access
     1: Coalesced/vectorized (vec4, aligned loads)
     2: Shared/local memory with explicit tiling
     3: Multi-level hierarchy (SLM + register blocking + prefetch)
   - d_algo (Algorithmic Structure):
     0: Direct PyTorch translation
     1: Fused operations (single-pass over data)
     2: Reformulated algorithm (online norm, flash pattern)
     3: Novel/asymptotically improved algorithm
   - d_sync (Parallelism Coordination):
     0: No synchronization (embarrassingly parallel)
     1: Work-group barriers (group::barrier)
     2: Sub-group primitives (shuffles, reductions, broadcast)
     3: Global coordination (atomics, multi-pass with sync)

   Use STATIC PATTERN MATCHING on the code (not runtime behavior).

Return evaluation results.`, {
    label: `eval-${generation}`,
    phase: 'Evaluate',
    schema: {
      type: 'object',
      properties: {
        compiled: { type: 'boolean' },
        correct: { type: 'boolean' },
        speedup: { type: 'number' },
        kernel_time_ms: { type: 'number' },
        d_mem: { type: 'number' },
        d_algo: { type: 'number' },
        d_sync: { type: 'number' },
        error_message: { type: 'string' },
        performance_notes: { type: 'string' },
      },
      required: ['compiled', 'correct', 'speedup', 'd_mem', 'd_algo', 'd_sync'],
    },
  })

  const fitness = computeFitness(evalResult.compiled, evalResult.correct, evalResult.speedup || 0)
  const cellKey = `${evalResult.d_mem || 0},${evalResult.d_algo || 0},${evalResult.d_sync || 0}`

  // ===========================================================================
  // Phase 5: Insert — Update archive if offspring improves its cell
  // ===========================================================================
  phase('Insert')

  const existingElite = archive[cellKey]
  let outcome = 'neutral'

  if (!existingElite || fitness > existingElite.fitness) {
    archive[cellKey] = {
      code: offspringCode,
      fitness: fitness,
      speedup: evalResult.speedup || 0,
      cell: cellKey,
      id: `gen${generation}`,
      strategy: varyResult.strategy_description,
    }
    outcome = existingElite ? 'improvement' : 'discovery'

    if (fitness > globalBest.fitness) {
      globalBest = { code: offspringCode, fitness, speedup: evalResult.speedup || 0, cell: cellKey }
    }
  } else {
    outcome = fitness < existingElite.fitness ? 'regression' : 'neutral'
  }

  // Record transition
  const parentCell = selectedParent?.cell || 'none'
  transitions.push({
    parent_cell: parentCell,
    child_cell: cellKey,
    delta_f: existingElite ? fitness - existingElite.fitness : fitness,
    outcome: outcome,
    gen: generation,
  })

  const statusIcon = outcome === 'improvement' ? '↑' : outcome === 'discovery' ? '★' : outcome === 'regression' ? '↓' : '='
  log(`  ${statusIcon} Cell [${cellKey}] fitness=${fitness.toFixed(2)} speedup=${(evalResult.speedup || 0).toFixed(2)}x | Archive: ${Object.keys(archive).length}/64 | Best: ${globalBest.speedup.toFixed(2)}x`)

  // ===========================================================================
  // Phase 6: Evolve-Prompts — Meta-prompter updates evolvable sections
  // ===========================================================================
  if ((generation + 1) % META_PROMPT_INTERVAL === 0 && generation > 0) {
    phase('Evolve-Prompts')

    const recentOutcomes = transitions.slice(-META_PROMPT_INTERVAL)
    const improvements = recentOutcomes.filter(t => t.outcome === 'improvement' || t.outcome === 'discovery')
    const failures = recentOutcomes.filter(t => t.outcome === 'regression' || t.outcome === 'neutral')

    const metaResult = await agent(`You are the KernelFoundry Meta-Prompter (Section 3.5).
Your job is to evolve the optimization guidance prompts based on recent evolutionary outcomes.

# Current Evolvable Prompt Sections:
## optimization_philosophy:
${metaPrompt.optimization_philosophy}

## optimization_strategies:
${metaPrompt.optimization_strategies}

## common_pitfalls:
${metaPrompt.common_pitfalls}

## analysis_guidance:
${metaPrompt.analysis_guidance}

# Recent Outcomes (last ${META_PROMPT_INTERVAL} generations):
- Improvements/discoveries: ${improvements.length}
- Regressions/neutral: ${failures.length}
- Successful transitions: ${improvements.map(t => `${t.parent_cell}→${t.child_cell}`).join(', ') || 'none'}

# Top archive entries:
${Object.entries(archive).sort((a, b) => b[1].fitness - a[1].fitness).slice(0, 5).map(([k, v]) => `[${k}] fitness=${v.fitness.toFixed(2)} speedup=${v.speedup.toFixed(2)}x: ${v.strategy?.substring(0, 60)}`).join('\n')}

# Meta-Prompting Rules (Section 3.5):
1. Diagnose which guidance was MISSING, MISLEADING, or INSUFFICIENT for recent outcomes
2. Prescribe targeted SEARCH/REPLACE updates to the 4 evolvable sections
3. Successful strategies should be REINFORCED; failed advice should be PRUNED
4. Keep each section concise (2-4 sentences)

Return updated prompt sections.`, {
      label: `meta-prompt-${generation}`,
      phase: 'Evolve-Prompts',
      schema: {
        type: 'object',
        properties: {
          optimization_philosophy: { type: 'string' },
          optimization_strategies: { type: 'string' },
          common_pitfalls: { type: 'string' },
          analysis_guidance: { type: 'string' },
          evolution_rationale: { type: 'string' },
        },
        required: ['optimization_philosophy', 'optimization_strategies', 'common_pitfalls', 'analysis_guidance'],
      },
    })

    if (metaResult) {
      metaPrompt = {
        optimization_philosophy: metaResult.optimization_philosophy || metaPrompt.optimization_philosophy,
        optimization_strategies: metaResult.optimization_strategies || metaPrompt.optimization_strategies,
        common_pitfalls: metaResult.common_pitfalls || metaPrompt.common_pitfalls,
        analysis_guidance: metaResult.analysis_guidance || metaPrompt.analysis_guidance,
      }
      promptArchive.push({ prompt_sections: { ...metaPrompt }, best_fitness: globalBest.fitness, generation })
      log(`  Meta-prompt evolved (gen ${generation + 1}): ${metaResult.evolution_rationale?.substring(0, 80) || 'updated'}`)
    }
  }
}

// =============================================================================
// Final Report
// =============================================================================
phase('Evaluate')

const finalReport = await agent(`Write a concise technical report on KernelFoundry MAP-Elites optimization.

# Results
- Operation: ${OP_DESC}
- Target: ${TARGET_LANG} on ${TARGET_HW}
- Baseline: ${baselineTime}ms
- Best speedup: ${globalBest.speedup.toFixed(2)}x (cell [${globalBest.cell}])
- Generations: ${GENERATIONS}
- Archive coverage: ${Object.keys(archive).length}/64 cells
- Total improvements: ${transitions.filter(t => t.outcome === 'improvement').length}
- Total discoveries: ${transitions.filter(t => t.outcome === 'discovery').length}
- Evidence mode: ${EVIDENCE_MODE}
- Descriptor artifact: ${DESCRIPTOR_RESULT_PATH}
- Archive update artifact: ${ARCHIVE_UPDATE_RESULT_PATH}

# Archive (top cells):
${Object.entries(archive).sort((a, b) => b[1].fitness - a[1].fitness).slice(0, 10).map(([k, v]) => `[${k}] ${v.speedup.toFixed(2)}x — ${v.strategy?.substring(0, 60)}`).join('\n')}

# Best Kernel:
\`\`\`${TARGET_LANG}
${globalBest.code.substring(0, 3000)}
\`\`\`

# Final Meta-Prompt State:
${Object.entries(metaPrompt).map(([k, v]) => `${k}: ${v}`).join('\n')}

Write:
1. Quality-diversity analysis: how well did the archive cover the behavioral space?
2. Meta-prompt evolution: how did the guidance change and what impact did it have?
3. Most effective optimization strategies discovered
4. Hardware awareness: evidence of hardware-specific optimizations`, {
  label: 'final-report',
  phase: 'Evaluate',
})

return {
  input_mode: INPUT_MODE,
  problem_definition: TASK_SPEC,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: globalBest.code ? `${EXP_DIR}/best_kernel.${TARGET_LANG === 'cuda' ? 'cu' : TARGET_LANG}` : '',
  initial_candidates: [],
  initial_generation_result: {
    verified: globalBest.speedup > 0,
    selected_candidate_id: globalBest.id || '',
  },
  operation: OP_DESC,
  language: TARGET_LANG,
  target_gpu: TARGET_HW,
  baseline_time_ms: baselineTime,
  best_speedup: globalBest.speedup,
  best_cell: globalBest.cell,
  best_kernel_code: globalBest.code,
  generations: GENERATIONS,
  archive_coverage: Object.keys(archive).length,
  archive_summary: Object.entries(archive).sort((a, b) => b[1].fitness - a[1].fitness).slice(0, 10).map(([k, v]) => ({ cell: k, speedup: v.speedup, strategy: v.strategy })),
  improvements: transitions.filter(t => t.outcome === 'improvement').length,
  discoveries: transitions.filter(t => t.outcome === 'discovery').length,
  final_meta_prompt: metaPrompt,
  prompt_evolution_history: promptArchive.length,
  descriptor_result_path: DESCRIPTOR_RESULT_PATH,
  archive_update_result_path: ARCHIVE_UPDATE_RESULT_PATH,
  evidence_mode: EVIDENCE_MODE,
  report: finalReport,
}
