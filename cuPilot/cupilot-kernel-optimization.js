export const meta = {
  name: 'cupilot-kernel-optimization',
  description: 'Strategy-coordinated evolutionary multi-agent CUDA kernel optimization with roofline-guided prompting (cuPilot methodology)',
  whenToUse: 'When evolving CUDA kernels through multi-generation optimization that requires sophisticated strategies (tensor cores, tiling, pipelining, memory swizzling). Uses strategy as an intermediate semantic representation to decouple evolutionary crossover from low-level code, with roofline model guidance and RAG-based strategy initialization.',
  phases: [
    { title: 'Setup', detail: 'Read kernel spec, generate initial kernel, roofline classification, strategy pool init' },
    { title: 'Strategize', detail: 'SCE Manager generates/crosses strategies at the semantic level via RAG' },
    { title: 'Translate', detail: 'Strategy Translator applies strategies to kernel code' },
    { title: 'Revise', detail: 'Kernel Revisor: compile check → function check → NCU profiling → fix loop' },
    { title: 'Evolve', detail: 'Tournament selection + elitism preservation + strategy alignment + next generation' },
    { title: 'Report', detail: 'Best kernel, strategies applied, hardware utilization metrics' },
  ],
}

// =============================================================================
// cuPilot: Strategy-Coordinated Multi-Agent Framework for CUDA Kernel Evolution
// =============================================================================
//
// Source: "cuPilot: A Strategy-Coordinated Multi-agent Framework for CUDA Kernel Evolution"
//         Chen, Wu, Li, Ma, Si, Hu, Yin, Yang
//         Southeast University / Tsinghua / Tsing Micro / NCTE
//         arXiv:2512.16465, 2025
//         https://github.com/champloo2878/cuPilot-Kernels
//
// Key insight: Conventional crossover prompting operates at the CODE level,
// requiring LLMs to traverse strategy identification → combination → synthesis
// in one shot. This fails as kernel complexity grows. cuPilot introduces
// STRATEGY as an intermediate semantic representation, decoupling:
//   - Strategy-level crossover (SCE Manager): combine optimization ideas
//   - Strategy-to-kernel translation (Strategy Translator): apply one strategy
//
// Three contributions:
//   1. Strategy-Coordinated Evolution (SCE) algorithm
//      - Strategy crossover at semantic level, not code level
//      - Tournament selection with elitism preservation
//      - Strategy alignment after kernel optimization
//   2. Roofline-guided prompting
//      - Classify kernel as compute-bound / memory-bound / middle-zone
//      - Guide strategy generation toward relevant optimizations
//   3. Strategy-level population initialization via RAG
//      - Historical (initial_kernel, optimized_kernel, strategy) triples
//      - RAG retrieval for similar kernels → bootstrap diverse strategies
//
// Multi-agent architecture (Figure 2):
//   - Roofline Prophet: positions kernel on roofline → guides prompting
//   - SCE Manager: manages population, crossover at strategy level
//   - Strategy Translator: applies strategy to kernel (strategy→code)
//   - Kernel Revisor: syntax check → function check → NCU profiling → fix
//   - Kernel Generator: initial vanilla kernel generation
//
// Usage:
//   Workflow({name: 'cupilot-kernel-optimization', args: {
//     kernel_spec: 'PyTorch operator code or description',
//     op_description: 'Standard GEMM (M×N×K, bf16)',
//     gpu_target: 'A100',
//     compile_command: 'nvcc -O3 -arch=sm_80 ...',
//     test_command: 'python test_kernel.py',
//     ncu_command: 'ncu --metrics ... ./bench',
//     epochs: 3,
//     generations_per_epoch: 4,
//     population_size: 50,
//     strategy_pool_path: '',
//   }})
//
// =============================================================================

// --- Required Args ---
const KERNEL_SPEC = args.kernel_spec || ''
const OP_DESC = args.op_description || 'CUDA kernel'

// --- Optional Args ---
const GPU_TARGET = args.gpu_target || 'A100'
const COMPILE_CMD = args.compile_command || ''
const TEST_CMD = args.test_command || ''
const NCU_CMD = args.ncu_command || ''
const EPOCHS = args.epochs || 2
const GENERATIONS = args.generations_per_epoch || 4
const POP_SIZE = args.population_size || 30
const STRATEGY_POOL_PATH = args.strategy_pool_path || ''
const EXP_DIR = args.exp_dir || '/tmp/cupilot_exp'
const KERNEL_PATH = args.kernel_path || ''
const MAX_REVISE_LOOPS = args.max_revise_loops || 3

// --- State ---
let population = []      // [{kernel, strategy, fitness, hwUtil}]
let strategyPool = []    // historical strategies for RAG
let bestKernel = { code: '', strategy: '', fitness: 0, speedup: 0 }
let rooflineClass = ''   // 'compute-bound' | 'memory-bound' | 'middle-zone'
let epoch = 0
let generation = 0

// =============================================================================
// Phase 1: Setup — Initial kernel, roofline classification, strategy pool
// =============================================================================
phase('Setup')

const setupResults = await parallel([
  // Agent 1: Generate initial kernel + roofline classification
  () => agent(`You are the cuPilot Kernel Generator + Roofline Prophet.

# Task:
1. Read the kernel specification:
${KERNEL_PATH ? `Read from: ${KERNEL_PATH}` : ''}
${KERNEL_SPEC ? `\`\`\`python\n${KERNEL_SPEC.substring(0, 3000)}\n\`\`\`` : `Operation: ${OP_DESC}`}

2. Generate a functionally correct initial CUDA kernel (vanilla, unoptimized)
   - Include proper __global__ function, thread mapping, memory access
   - Must compile with nvcc and produce correct output

3. Perform Roofline Classification (cuPilot Section 4.3):
   Analyze the kernel's arithmetic intensity (FLOPs / bytes transferred):
   - Compute peak FLOPS for ${GPU_TARGET}
   - Compute peak memory bandwidth for ${GPU_TARGET}
   - Calculate the intersection point (arithmetic intensity threshold)

   Classify as:
   - "compute-bound": kernel is to the RIGHT of intersection
     → Prioritize: SM throughput, tensor cores, branch efficiency, instruction scheduling
   - "memory-bound": kernel is to the LEFT of intersection
     → Prioritize: memory bandwidth, L2 hit rate, vectorized loads, memory padding
   - "middle-zone": kernel is BETWEEN the two intersection points
     → Need both compute AND memory optimizations

4. Generate initial optimization guidance based on roofline position.

Return the initial kernel and roofline analysis.`, {
    label: 'setup-kernel-roofline',
    phase: 'Setup',
    schema: {
      type: 'object',
      properties: {
        initial_kernel: { type: 'string' },
        roofline_class: { type: 'string' },
        arithmetic_intensity: { type: 'number' },
        roofline_guidance: { type: 'string' },
        compute_metrics_to_focus: { type: 'array', items: { type: 'string' } },
        memory_metrics_to_focus: { type: 'array', items: { type: 'string' } },
      },
      required: ['initial_kernel', 'roofline_class', 'roofline_guidance'],
    },
  }),

  // Agent 2: Initialize strategy pool (RAG from historical data)
  () => agent(`You are the cuPilot Strategy Pool Initializer (Section 4.4).

# Task:
Generate an initial strategy pool for kernel optimization. Each strategy is a concise, reusable optimization technique description.

# Operation: ${OP_DESC}
# GPU Target: ${GPU_TARGET}
${STRATEGY_POOL_PATH ? `# Historical strategy pool: ${STRATEGY_POOL_PATH}` : ''}

# Generate 8-12 diverse strategies spanning these categories:

## Compute Optimizations:
- Invoke Tensor Core (WMMA/MMA instructions for matrix operations)
- Thread/block configuration for maximum occupancy
- Instruction-level parallelism and scheduling

## Memory Optimizations:
- Tiling and shared memory staging
- Vectorized memory access (float4, LDG.128)
- Memory padding to avoid bank conflicts
- Shared memory swizzling for conflict-free access

## Execution Pipeline:
- Double buffering (overlap compute and memory)
- Multi-stage software pipeline
- Asynchronous memory copy (cp.async)

## Low-level:
- Layout/thread block swizzling for L2 locality
- PTX-level instruction optimization
- Register blocking for data reuse

For each strategy, provide:
- name: short identifier
- description: 1-2 sentence explanation of what to do
- applicable_when: roofline condition (compute-bound, memory-bound, or both)
- expected_metrics: which NCU metrics should improve

Return the strategy pool.`, {
    label: 'setup-strategy-pool',
    phase: 'Setup',
    schema: {
      type: 'object',
      properties: {
        strategies: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              description: { type: 'string' },
              applicable_when: { type: 'string' },
              expected_metrics: { type: 'array', items: { type: 'string' } },
            },
            required: ['name', 'description'],
          },
        },
      },
      required: ['strategies'],
    },
  }),
])

const kernelSetup = setupResults[0]
const poolSetup = setupResults[1]

rooflineClass = kernelSetup?.roofline_class || 'middle-zone'
strategyPool = (poolSetup?.strategies || []).map(s => s.name + ': ' + s.description)

// Initialize population with initial kernel + no strategy
const initialKernel = kernelSetup?.initial_kernel || ''
population = [{ kernel: initialKernel, strategy: 'baseline (no optimization)', fitness: 0, speedup: 1.0 }]
bestKernel = { code: initialKernel, strategy: 'baseline', fitness: 0, speedup: 1.0 }

log(`Setup: ${OP_DESC} | Roofline: ${rooflineClass} | Strategy pool: ${strategyPool.length} strategies`)
log(`Guidance: ${kernelSetup?.roofline_guidance?.substring(0, 100)}...`)

// =============================================================================
// Evolutionary Loop: Epochs × Generations
// =============================================================================

for (epoch = 0; epoch < EPOCHS; epoch++) {
  for (generation = 0; generation < GENERATIONS; generation++) {
    log(`\n=== Epoch ${epoch + 1}/${EPOCHS}, Gen ${generation + 1}/${GENERATIONS} | Pop: ${population.length} | Best: ${bestKernel.speedup.toFixed(2)}x ===`)

    // =========================================================================
    // Phase 2: Strategize — SCE Manager generates/crosses strategies
    // =========================================================================
    phase('Strategize')

    // Select parents via tournament selection
    const sortedPop = [...population].sort((a, b) => b.fitness - a.fitness)
    const parent1 = sortedPop[0]
    const parent2 = sortedPop[Math.min(1, sortedPop.length - 1)]

    const strategizeResult = await agent(`You are the cuPilot SCE Manager (Section 4.2).
Perform STRATEGY-LEVEL crossover and generate new optimization strategies.

# Roofline Classification: ${rooflineClass}
# Roofline Guidance:
${kernelSetup?.roofline_guidance || ''}

# Parent 1 (fitness=${parent1.fitness.toFixed(2)}):
Strategy: ${parent1.strategy}

# Parent 2 (fitness=${parent2.fitness.toFixed(2)}):
Strategy: ${parent2.strategy}

# Available Strategy Pool:
${strategyPool.slice(0, 15).map((s, i) => `${i + 1}. ${s}`).join('\n')}

# SCE Crossover Rules (Section 4.2):
1. Work at the STRATEGY level, NOT the code level
2. Combine optimization ideas from both parents into new strategies
3. Generate ${Math.min(POP_SIZE, 5)} new strategy combinations
4. Each strategy should be a coherent set of optimizations that work together
5. For ${rooflineClass} kernels, prioritize:
${rooflineClass === 'compute-bound' ? '   - Tensor core usage, compute throughput, SM utilization' : rooflineClass === 'memory-bound' ? '   - Memory bandwidth, L2 hit rate, vectorized access, memory padding' : '   - Both compute AND memory optimizations together'}
6. Include at least one "exploratory" strategy that tries something novel

Return new strategies for this generation.`, {
      label: `strategize-e${epoch}-g${generation}`,
      phase: 'Strategize',
      schema: {
        type: 'object',
        properties: {
          new_strategies: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                strategy: { type: 'string' },
                rationale: { type: 'string' },
                parent_strategies_used: { type: 'array', items: { type: 'string' } },
              },
              required: ['strategy'],
            },
          },
        },
        required: ['new_strategies'],
      },
    })

    const newStrategies = strategizeResult?.new_strategies || []
    log(`  Strategies: ${newStrategies.length} new combinations`)

    // =========================================================================
    // Phase 3: Translate — Apply strategies to kernel code
    // =========================================================================
    phase('Translate')

    const translatedKernels = await parallel(
      newStrategies.slice(0, 5).map((strat, idx) => () =>
        agent(`You are the cuPilot Strategy Translator (Section 4.1).
Apply this optimization strategy to produce an optimized CUDA kernel.

# Base Kernel:
\`\`\`cuda
${bestKernel.code.substring(0, 4000)}
\`\`\`

# Strategy to Apply:
${strat.strategy}

# Roofline: ${rooflineClass}
# GPU Target: ${GPU_TARGET}

# Translation Rules:
1. Apply the strategy faithfully — this is strategy-to-kernel translation
2. Output a COMPLETE, COMPILABLE .cu file
3. Maintain functional correctness (same output as baseline)
4. Use appropriate CUDA constructs for the strategy:
   - Tensor cores → wmma/mma intrinsics or cooperative groups
   - Tiling → shared memory with __syncthreads
   - Vectorized → float4/int4 loads
   - Double buffering → ping-pong shared memory buffers
   - Swizzling → XOR-based index remapping
5. Include proper error checking and bounds handling

Return the optimized kernel.`, {
          label: `translate-e${epoch}-g${generation}-s${idx}`,
          phase: 'Translate',
          schema: {
            type: 'object',
            properties: {
              kernel_code: { type: 'string' },
              strategies_applied: { type: 'array', items: { type: 'string' } },
            },
            required: ['kernel_code'],
          },
        })
      )
    )

    // =========================================================================
    // Phase 4: Revise — Kernel Revisor: compile → function → profile → fix
    // =========================================================================
    phase('Revise')

    const revisedResults = await parallel(
      translatedKernels.filter(Boolean).map((tk, idx) => () =>
        agent(`You are the cuPilot Kernel Revisor (Section 4.1, Figure 2 right side).
Validate and refine this kernel through the revision loop.

# Kernel to Revise:
\`\`\`cuda
${(tk.kernel_code || '').substring(0, 4000)}
\`\`\`

# Strategy Applied: ${newStrategies[idx]?.strategy || 'unknown'}
# GPU: ${GPU_TARGET}

# Revision Loop (up to ${MAX_REVISE_LOOPS} iterations):

## Step 1: NVCC Compiler Check
${COMPILE_CMD ? `Run: ${COMPILE_CMD}` : 'Check: valid CUDA syntax, correct use of intrinsics, proper template parameters.'}
If syntax errors → fix them and retry.

## Step 2: Function Check
${TEST_CMD ? `Run: ${TEST_CMD}` : 'Verify numerical correctness against reference output.'}
If function errors → fix logic bugs and retry.

## Step 3: Performance Profiling
${NCU_CMD ? `Run: ${NCU_CMD}` : 'Analyze: estimate throughput, occupancy, memory efficiency.'}
Extract:
- Speedup vs PyTorch baseline
- SM Throughput utilization
- Memory bandwidth utilization
- L2 cache hit rate

## Step 4: Profiling-guided refinement
If performance is suboptimal, use NCU feedback to make ONE targeted improvement.
Then return to Step 1.

If a kernel cannot be fixed after ${MAX_REVISE_LOOPS} attempts, mark it as failed.

Return the final revised kernel and its metrics.`, {
          label: `revise-e${epoch}-g${generation}-k${idx}`,
          phase: 'Revise',
          schema: {
            type: 'object',
            properties: {
              kernel_code: { type: 'string' },
              compiled: { type: 'boolean' },
              correct: { type: 'boolean' },
              speedup: { type: 'number' },
              sm_utilization_pct: { type: 'number' },
              memory_utilization_pct: { type: 'number' },
              revision_iterations: { type: 'number' },
              strategies_confirmed: { type: 'array', items: { type: 'string' } },
            },
            required: ['kernel_code', 'compiled', 'correct', 'speedup'],
          },
        })
      )
    )

    // =========================================================================
    // Phase 5: Evolve — Tournament selection + elitism + strategy alignment
    // =========================================================================
    phase('Evolve')

    // Compute fitness and add to population
    for (let i = 0; i < revisedResults.length; i++) {
      const r = revisedResults[i]
      if (!r) continue
      const fitness = (!r.compiled || !r.correct) ? 0 : (0.5 + 0.5 * Math.min(1.0, (r.speedup || 0) / 3.0))
      const individual = {
        kernel: r.kernel_code || '',
        strategy: newStrategies[i]?.strategy || '',
        fitness: fitness,
        speedup: r.speedup || 0,
        hwUtil: { sm: r.sm_utilization_pct || 0, mem: r.memory_utilization_pct || 0 },
      }
      population.push(individual)

      // Update best
      if (r.correct && (r.speedup || 0) > bestKernel.speedup) {
        bestKernel = { code: r.kernel_code, strategy: newStrategies[i]?.strategy || '', fitness, speedup: r.speedup }
        log(`  NEW BEST: ${bestKernel.speedup.toFixed(2)}x — strategy: ${bestKernel.strategy.substring(0, 60)}`)
      }
    }

    // Tournament selection + elitism preservation
    population.sort((a, b) => b.fitness - a.fitness)
    // Keep top individuals (elitism)
    const eliteCount = Math.max(2, Math.floor(population.length * 0.2))
    population = population.slice(0, Math.min(POP_SIZE, population.length))

    // Strategy alignment: update strategies to match their optimized kernels
    // (This is the StrategyAlignment step in Algorithm 1, line 7)

    const successfulStrategies = population.filter(p => p.speedup > 1.0).map(p => p.strategy)
    if (successfulStrategies.length > 0) {
      for (const s of successfulStrategies) {
        if (!strategyPool.includes(s) && s.length > 10) {
          strategyPool.push(s)
        }
      }
    }

    log(`  Pop: ${population.length} | Top3: ${population.slice(0, 3).map(p => p.speedup.toFixed(2) + 'x').join(', ')} | Pool: ${strategyPool.length} strategies`)
  }

  // End of epoch: select top kernels as initial kernels for next epoch
  log(`\n--- Epoch ${epoch + 1} complete. Best: ${bestKernel.speedup.toFixed(2)}x ---`)
}

// =============================================================================
// Phase 6: Report
// =============================================================================
phase('Report')

const finalReport = await agent(`Write a concise technical report on cuPilot evolutionary optimization.

# cuPilot Results
- Operation: ${OP_DESC}
- GPU: ${GPU_TARGET}
- Roofline classification: ${rooflineClass}
- Best speedup: ${bestKernel.speedup.toFixed(2)}x over PyTorch
- Epochs: ${EPOCHS}, Generations/epoch: ${GENERATIONS}
- Final population: ${population.length}
- Strategy pool: ${strategyPool.length} strategies

# Best Kernel Strategy:
${bestKernel.strategy}

# Best Kernel Code:
\`\`\`cuda
${bestKernel.code.substring(0, 3000)}
\`\`\`

# Top strategies in final population:
${population.slice(0, 5).map((p, i) => `${i + 1}. ${p.strategy.substring(0, 80)} (${p.speedup.toFixed(2)}x)`).join('\n')}

Write:
1. Evolutionary trajectory: how strategies evolved across generations
2. Roofline guidance effectiveness: did the classification help?
3. Strategy-level crossover: which combinations were most productive?
4. Hardware utilization achieved vs theoretical peak
5. Remaining optimization opportunities`, {
  label: 'final-report',
  phase: 'Report',
})

return {
  operation: OP_DESC,
  gpu_target: GPU_TARGET,
  roofline_class: rooflineClass,
  best_speedup: bestKernel.speedup,
  best_strategy: bestKernel.strategy,
  best_kernel_code: bestKernel.code,
  epochs: EPOCHS,
  generations: GENERATIONS,
  final_population_size: population.length,
  strategy_pool_size: strategyPool.length,
  top_strategies: population.slice(0, 5).map(p => ({ strategy: p.strategy, speedup: p.speedup })),
  report: finalReport,
}
