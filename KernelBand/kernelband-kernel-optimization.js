export const meta = {
  name: 'kernelband-kernel-optimization',
  description: 'Multi-Armed Bandit framework for kernel optimization with hardware-aware pruning, trace-driven clustering, and Lipschitz-continuous reward estimation (KernelBand methodology)',
  whenToUse: 'When optimizing GPU kernels (especially Triton) and you want principled exploration-exploitation balance across multiple optimization strategies. Ideal when the optimization space is large and you want to avoid wasting budget on physically implausible strategies. Uses profiling-based clustering to transfer knowledge between similar kernels and UCB-based action selection to steer exploration.',
  phases: [
    { title: 'Setup', detail: 'Parse kernel, identify strategies, configure hardware target and profiling' },
    { title: 'Profile', detail: 'Extract behavioral feature vector φ(k) via NCU profiling for each candidate' },
    { title: 'Cluster', detail: 'K-Means clustering on behavioral features, update cluster centroids' },
    { title: 'Select', detail: 'Hardware-constrained Masked UCB to select (cluster, strategy) pair' },
    { title: 'Generate', detail: 'LLM applies selected strategy to selected kernel candidate' },
    { title: 'Evaluate', detail: 'Compile, verify correctness, measure latency, compute reward' },
    { title: 'Update', detail: 'Update bandit statistics, cluster assignments, and candidate pool' },
    { title: 'Report', detail: 'Final results: best kernel, strategy statistics, exploration trajectory' },
  ],
}

// =============================================================================
// KernelBand: Steering LLM-based Kernel Optimization via Hardware-Aware
//             Multi-Armed Bandits
// =============================================================================
//
// Source: "KernelBand: Steering LLM-based Kernel Optimization via
//          Hardware-Aware Multi-Armed Bandits"
//         Dezhi Ran, Shuxiao Xie, Mingfang Ji, Anmin Liu, Mengzhou Wu,
//         Yuan Cao, Yuzhe Guo, Hao Yu, Linyi Li, Yitao Hu, Wei Yang, Tao Xie
//         PKU / Tongming Lake / ECNU / Tianjin / HKUST / SFU / UTD / Fudan
//         arXiv:2511.18868, Feb 2026
//
// KernelBand formulates kernel optimization as a contextual Multi-Armed Bandit
// problem, explicitly separating code generation (LLM strength) from navigation
// of the optimization search space (bandit strength).
//
// Core Algorithm (Algorithm 1):
//   For t = 1..T:
//     1. Compute φ(k) for all k ∈ P (behavioral feature vector)
//     2. If t mod τ = 0 and |P| ≥ 2K: re-cluster via K-Means
//     3. Hardware-constrained selection:
//        - Compute mask M_{i,s} = 𝟙[h(k_c^(i))[Target(s)] < θ_sat]
//        - Select (I_t, S_t) = argmax UCB with mask
//     4. Sample k_t from cluster C_{I_t}
//     5. Generate k'_t = LLM(k_t, S_t)
//     6. If Verify(k'_t): compute reward r_t, update P, update μ̂, N
//
// Key Mechanisms:
//   1. Behavioral Feature Vector φ(k) = [T̄(k), n_reg, n_smem, d_block, η_occ]
//      - Kernels close in φ-space share similar bottlenecks (Lipschitz continuity)
//   2. Dynamic K-Means Clustering (K=3 default, re-cluster every τ=10 iters)
//      - Only profile cluster centroids (representative profiling) to save cost
//   3. Hardware Signature h(k) from NCU: throughput % for DRAM, L2, SM
//      - Pruning: strategy s valid for cluster i only if Target(s) < θ_sat (75%)
//   4. Masked UCB: argmax μ̂_{i,s} + c√(ln t / N_{i,s}) subject to M_{i,s} = 1
//
// Strategies S = {tiling, vectorization, fusion, pipeline, reordering, access_layout}
//
// Results: 1.91× geometric mean speedup on A100 (TritonBench-G, T=20)
//          35-50% higher speedup per dollar vs unguided methods
//
// Usage:
//   Workflow({name: 'kernelband-kernel-optimization', args: {
//     kernel_path: '/path/to/kernel.py',
//     op_description: 'Fused attention forward pass',
//     harness_path: '/path/to/benchmark.py',
//     compile_command: 'python -c "import kernel; kernel.test()"',
//     benchmark_command: 'python benchmark.py',
//     ncu_command: 'ncu --metrics ...',
//     feature_vector_result_path: '/tmp/kernelband_exp/features.json',
//     hardware_signature_result_path: '/tmp/kernelband_exp/hardware_signature.json',
//     gpu_target: 'A100',
//     iterations: 20,
//     num_clusters: 3,
//     recluster_period: 10,
//     strategies: ['tiling', 'vectorization', 'fusion', 'pipeline', 'reordering', 'access_layout'],
//   }})
//
// =============================================================================

// --- Required Args ---
const KERNEL_PATH = args.kernel_path || ''
const OP_DESCRIPTION = args.op_description || 'GPU kernel'

// --- Optional Args ---
const HARNESS_PATH = args.harness_path || ''
const COMPILE_CMD = args.compile_command || ''
const BENCHMARK_CMD = args.benchmark_command || ''
const NCU_CMD = args.ncu_command || ''
const NCU_BINARY = args.ncu_binary || 'ncu'
const FEATURE_VECTOR_RESULT_PATH = args.feature_vector_result_path || `${args.exp_dir || '/tmp/kernelband_exp'}/features/latest.json`
const HARDWARE_SIGNATURE_RESULT_PATH = args.hardware_signature_result_path || `${args.exp_dir || '/tmp/kernelband_exp'}/profiles/hardware_signature.json`
const GPU_TARGET = args.gpu_target || 'A100'
const ITERATIONS = args.iterations || 20
const NUM_CLUSTERS = args.num_clusters || 3
const RECLUSTER_PERIOD = args.recluster_period || 10
const UCB_C = args.ucb_exploration || 2.0
const SATURATION_THRESHOLD = args.saturation_threshold || 0.75
const EXP_DIR = args.exp_dir || '/tmp/kernelband_exp'
const EVIDENCE_MODE = (COMPILE_CMD && BENCHMARK_CMD && (NCU_CMD || NCU_BINARY))
  ? 'measured'
  : 'conservative_missing_evidence'

const STRATEGIES = args.strategies || [
  'tiling',
  'vectorization',
  'fusion',
  'pipeline',
  'reordering',
  'access_layout',
]

// --- Bandit State ---
let candidatePool = []
let clusters = []
let banditStats = {}
let bestKernel = { code: '', latency: Infinity, speedup: 0 }
let baselineLatency = 0
let totalReward = 0
let iterationLog = []

// Initialize bandit statistics: N_{i,s} = 1, μ̂_{i,s} = 0.5 for all (i, s)
for (let i = 0; i < NUM_CLUSTERS; i++) {
  for (const s of STRATEGIES) {
    const key = `${i}_${s}`
    banditStats[key] = { count: 1, mean_reward: 0.5, mask: 1 }
  }
}

// =============================================================================
// Phase 1: Setup — Parse kernel, identify hardware, establish baseline
// =============================================================================
phase('Setup')

const setupResult = await agent(`You are setting up a KernelBand optimization session.

# Task
1. Read the kernel file: ${KERNEL_PATH}
2. Create experiment directory: mkdir -p ${EXP_DIR}/{candidates,profiles,logs}
3. Record profiling evidence artifacts:
   - feature_vector_result_path: ${FEATURE_VECTOR_RESULT_PATH}
   - hardware_signature_result_path: ${HARDWARE_SIGNATURE_RESULT_PATH}
   - evidence_mode: ${EVIDENCE_MODE}
   If evidence_mode is conservative_missing_evidence, do not claim strict KernelBand execution; mark phi, masks, and rewards as estimates.
4. Establish baseline performance:
   ${COMPILE_CMD ? `Compile: ${COMPILE_CMD}` : '(compile the kernel)'}
   ${BENCHMARK_CMD ? `Benchmark: ${BENCHMARK_CMD}` : '(run the benchmark)'}
5. Run initial NCU profiling to get hardware signature:
   ${NCU_CMD || `${NCU_BINARY} --set full --target-processes all <benchmark_binary>`}
   Extract: DRAM throughput %, L2 throughput %, SM throughput %
5. Extract behavioral features φ(k₀):
   - Normalized execution time T̄(k₀) (= 1.0 for baseline)
   - Registers per thread (n_reg)
   - Shared memory per block (n_smem)
   - Block dimension (d_block)
   - Occupancy (η_occ)

# Target Hardware: ${GPU_TARGET}
# Operation: ${OP_DESCRIPTION}
# Strategies available: ${STRATEGIES.join(', ')}

Return the baseline metrics and hardware signature.`, {
  label: 'setup',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      kernel_code: { type: 'string' },
      baseline_latency_us: { type: 'number' },
      hardware_signature: {
        type: 'object',
        properties: {
          dram_throughput_pct: { type: 'number' },
          l2_throughput_pct: { type: 'number' },
          sm_throughput_pct: { type: 'number' },
          dominant_bottleneck: { type: 'string' },
        },
      },
      behavioral_features: {
        type: 'object',
        properties: {
          normalized_time: { type: 'number' },
          registers_per_thread: { type: 'number' },
          shared_mem_bytes: { type: 'number' },
          block_dimension: { type: 'number' },
          occupancy: { type: 'number' },
        },
      },
      platform_info: { type: 'string' },
    },
    required: ['baseline_latency_us'],
  },
})

baselineLatency = setupResult?.baseline_latency_us || 1000
const initialCode = setupResult?.kernel_code || ''
const hwSignature = setupResult?.hardware_signature || { dram_throughput_pct: 50, l2_throughput_pct: 50, sm_throughput_pct: 50 }
const initialFeatures = setupResult?.behavioral_features || { normalized_time: 1.0, registers_per_thread: 32, shared_mem_bytes: 0, block_dimension: 256, occupancy: 0.5 }

candidatePool.push({
  id: 0,
  code: initialCode,
  latency: baselineLatency,
  speedup: 1.0,
  features: initialFeatures,
  hw_signature: hwSignature,
  cluster: 0,
  source_strategy: 'initial',
})

clusters = [{ id: 0, centroid_features: initialFeatures, centroid_hw: hwSignature, members: [0] }]

log(`Setup: baseline ${baselineLatency}μs on ${GPU_TARGET} | ${STRATEGIES.length} strategies | K=${NUM_CLUSTERS} clusters | T=${ITERATIONS} iters`)
log(`Hardware: DRAM=${hwSignature.dram_throughput_pct}% L2=${hwSignature.l2_throughput_pct}% SM=${hwSignature.sm_throughput_pct}% → ${hwSignature.dominant_bottleneck || 'unknown'}-bound`)

// =============================================================================
// Main Bandit Loop: T iterations
// =============================================================================

for (let t = 1; t <= ITERATIONS; t++) {
  log(`\n--- Iteration ${t}/${ITERATIONS} ---`)

  // ===========================================================================
  // Periodic Re-clustering (every τ iterations when pool is large enough)
  // ===========================================================================
  const shouldRecluster = (t % RECLUSTER_PERIOD === 0) && (candidatePool.length >= 2 * NUM_CLUSTERS)

  if (shouldRecluster) {
    phase('Cluster')

    const clusterResult = await agent(`You are the KernelBand Dynamic Clustering module (Section 3.3).

# Task: Re-cluster the candidate kernel pool using K-Means on behavioral features.

# Candidate Pool (${candidatePool.length} kernels):
${candidatePool.map((c, idx) => `Kernel ${c.id}: φ = [T̄=${c.features.normalized_time.toFixed(3)}, reg=${c.features.registers_per_thread}, smem=${c.features.shared_mem_bytes}, block=${c.features.block_dimension}, occ=${c.features.occupancy.toFixed(3)}] speedup=${c.speedup.toFixed(2)}x`).join('\n')}

# Parameters:
- K = ${NUM_CLUSTERS} clusters
- Distance metric: Euclidean on normalized φ vectors

# Requirements:
1. Normalize each dimension to [0,1] before clustering
2. Assign each kernel to its nearest cluster centroid
3. Compute new centroids as the mean of cluster members
4. For each cluster centroid, identify the representative kernel (nearest to centroid)
   - This representative will be profiled for hardware signature updates

Return cluster assignments and centroids.`, {
      label: `cluster-t${t}`,
      phase: 'Cluster',
      schema: {
        type: 'object',
        properties: {
          clusters: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                id: { type: 'number' },
                centroid_features: {
                  type: 'object',
                  properties: {
                    normalized_time: { type: 'number' },
                    registers_per_thread: { type: 'number' },
                    shared_mem_bytes: { type: 'number' },
                    block_dimension: { type: 'number' },
                    occupancy: { type: 'number' },
                  },
                },
                member_ids: { type: 'array', items: { type: 'number' } },
                representative_id: { type: 'number' },
              },
            },
          },
        },
        required: ['clusters'],
      },
    })

    if (clusterResult?.clusters) {
      clusters = clusterResult.clusters.map(c => ({
        id: c.id,
        centroid_features: c.centroid_features,
        centroid_hw: hwSignature,
        members: c.member_ids || [],
      }))
      for (const c of clusterResult.clusters) {
        for (const mid of (c.member_ids || [])) {
          const cand = candidatePool.find(p => p.id === mid)
          if (cand) cand.cluster = c.id
        }
      }
      log(`Re-clustered: ${clusters.map(c => `C${c.id}(${c.members.length})`).join(', ')}`)
    }

    // Profile cluster centroids for hardware signature updates
    phase('Profile')

    const profileResult = await agent(`You are the KernelBand Representative Profiling module (Section 3.3).

# Task: Profile the representative kernel from each active cluster to update hardware signatures.

# Clusters:
${clusters.map(c => {
  const rep = candidatePool.find(p => p.id === (c.members[0] || 0))
  return `Cluster ${c.id}: representative kernel ${rep ? rep.id : 'N/A'}`
}).join('\n')}

# Profiling Command:
${NCU_CMD || `${NCU_BINARY} --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed`}

# Extract for each cluster centroid:
- DRAM throughput % (memory bandwidth utilization)
- L2 cache throughput %
- SM throughput % (compute utilization)
These determine the hardware mask M_{i,s} for strategy pruning.

A strategy s is VALID for cluster i only if: h(k_c^(i))[Target(s)] < ${SATURATION_THRESHOLD * 100}%
- tiling targets: SM (compute)
- vectorization targets: DRAM (memory bandwidth)
- fusion targets: DRAM (reduce memory traffic)
- pipeline targets: SM (increase compute overlap)
- reordering targets: L2 (cache locality)
- access_layout targets: DRAM (coalescing)

Return updated hardware signatures and masks.`, {
      label: `profile-t${t}`,
      phase: 'Profile',
      schema: {
        type: 'object',
        properties: {
          cluster_profiles: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                cluster_id: { type: 'number' },
                dram_throughput_pct: { type: 'number' },
                l2_throughput_pct: { type: 'number' },
                sm_throughput_pct: { type: 'number' },
                valid_strategies: { type: 'array', items: { type: 'string' } },
                pruned_strategies: { type: 'array', items: { type: 'string' } },
              },
            },
          },
        },
        required: ['cluster_profiles'],
      },
    })

    if (profileResult?.cluster_profiles) {
      for (const cp of profileResult.cluster_profiles) {
        const cluster = clusters.find(c => c.id === cp.cluster_id)
        if (cluster) {
          cluster.centroid_hw = {
            dram_throughput_pct: cp.dram_throughput_pct,
            l2_throughput_pct: cp.l2_throughput_pct,
            sm_throughput_pct: cp.sm_throughput_pct,
          }
        }
        for (const s of STRATEGIES) {
          const key = `${cp.cluster_id}_${s}`
          if (banditStats[key]) {
            banditStats[key].mask = (cp.valid_strategies || STRATEGIES).includes(s) ? 1 : 0
          }
        }
      }
      const prunedCount = Object.values(banditStats).filter(v => v.mask === 0).length
      log(`Profiled centroids: ${prunedCount}/${Object.keys(banditStats).length} (cluster,strategy) pairs pruned`)
    }
  }

  // ===========================================================================
  // Action Selection: Masked UCB (Section 3.4)
  // ===========================================================================
  phase('Select')

  let bestUCB = -Infinity
  let selectedCluster = 0
  let selectedStrategy = STRATEGIES[0]

  for (let i = 0; i < Math.min(NUM_CLUSTERS, clusters.length); i++) {
    for (const s of STRATEGIES) {
      const key = `${i}_${s}`
      const stat = banditStats[key]
      if (!stat || stat.mask === 0) continue
      const ucb = stat.mean_reward + UCB_C * Math.sqrt(Math.log(t) / stat.count)
      if (ucb > bestUCB) {
        bestUCB = ucb
        selectedCluster = i
        selectedStrategy = s
      }
    }
  }

  // Sample kernel from selected cluster (prefer candidates with higher local potential)
  const clusterMembers = candidatePool.filter(c => c.cluster === selectedCluster)
  const selectedKernel = clusterMembers.length > 0
    ? clusterMembers.reduce((best, c) => c.speedup > best.speedup ? c : best, clusterMembers[0])
    : candidatePool[0]

  log(`Select: cluster=${selectedCluster}, strategy=${selectedStrategy}, kernel=${selectedKernel.id} (UCB=${bestUCB.toFixed(3)})`)

  // ===========================================================================
  // Code Generation: LLM applies strategy to kernel (Section 3.1)
  // ===========================================================================
  phase('Generate')

  const generateResult = await agent(`You are the KernelBand Code Generator. Apply a specific optimization strategy to the given kernel.

# Selected Action: (Cluster ${selectedCluster}, Strategy: ${selectedStrategy})
# Target Hardware: ${GPU_TARGET}
# Operation: ${OP_DESCRIPTION}

# Source Kernel (ID ${selectedKernel.id}, current speedup: ${selectedKernel.speedup.toFixed(2)}x):
\`\`\`
${(selectedKernel.code || '').substring(0, 6000)}
\`\`\`

# Strategy: ${selectedStrategy}
Apply the "${selectedStrategy}" optimization strategy. Specific guidance:

${selectedStrategy === 'tiling' ? `TILING: Restructure computation into tiles that fit in shared memory/registers.
- Choose tile sizes that match hardware cache/register constraints
- Ensure tiles cover the full computation without overlap or gaps
- Consider multi-level tiling (thread block → warp → thread)` : ''}
${selectedStrategy === 'vectorization' ? `VECTORIZATION: Increase memory throughput via wider loads/stores.
- Use vector types (float4, int4) for coalesced memory access
- Ensure alignment constraints are met
- Batch independent computations into vector operations` : ''}
${selectedStrategy === 'fusion' ? `FUSION: Reduce memory traffic by fusing multiple operations.
- Identify producer-consumer pairs that can share registers/shared memory
- Eliminate intermediate global memory writes
- Fuse elementwise operations with preceding/following kernels` : ''}
${selectedStrategy === 'pipeline' ? `PIPELINE: Overlap computation with memory operations.
- Double-buffer shared memory loads with computation
- Use async copy (cp.async) where available
- Stage computation to hide memory latency` : ''}
${selectedStrategy === 'reordering' ? `REORDERING: Improve cache locality via access pattern changes.
- Reorder loops for better spatial/temporal locality
- Swizzle memory layouts to avoid bank conflicts
- Rearrange thread-to-data mapping for coalescing` : ''}
${selectedStrategy === 'access_layout' ? `ACCESS & LAYOUT: Optimize memory layout and access patterns.
- Ensure coalesced global memory accesses (consecutive threads → consecutive addresses)
- Pad shared memory to avoid bank conflicts
- Choose row-major vs column-major based on access pattern` : ''}

# Constraints:
- Preserve functional correctness (same outputs for same inputs)
- Target the non-saturated resource (this strategy was selected because the target resource is below ${SATURATION_THRESHOLD * 100}% utilization)
- Generate a complete, compilable kernel

# Bandit Context:
- Iteration: ${t}/${ITERATIONS}
- This (cluster, strategy) pair has been tried ${banditStats[`${selectedCluster}_${selectedStrategy}`]?.count || 0} times
- Average reward so far: ${(banditStats[`${selectedCluster}_${selectedStrategy}`]?.mean_reward || 0).toFixed(3)}

Return the optimized kernel code.`, {
    label: `generate-t${t}-${selectedStrategy}`,
    phase: 'Generate',
    schema: {
      type: 'object',
      properties: {
        optimized_kernel: { type: 'string' },
        changes_description: { type: 'string' },
        expected_improvement: { type: 'string' },
      },
      required: ['optimized_kernel'],
    },
  })

  const generatedCode = generateResult?.optimized_kernel || ''

  // ===========================================================================
  // Evaluation: Compile, verify, benchmark (Section 3.1, Algorithm 1 line 19)
  // ===========================================================================
  phase('Evaluate')

  const evalResult = await agent(`You are the KernelBand Evaluation module. Verify correctness and measure performance.

# Generated Kernel:
\`\`\`
${generatedCode.substring(0, 6000)}
\`\`\`

# Evaluation Steps (Two-stage verification from Section 4.1):
1. **Call Accuracy**: Compile and run — check for runtime errors
   ${COMPILE_CMD || '(compile the generated kernel)'}
2. **Execution Accuracy**: Verify numerical equivalence via torch.allclose
   Compare outputs against reference (CPU ATen) across 10+ input shapes
3. **Performance Measurement**: Benchmark across dominant input shapes
   ${BENCHMARK_CMD || '(run the performance benchmark)'}
   Report latency in microseconds
4. **Feature Extraction**: Extract behavioral features φ(k'):
   - Normalized time: T̄(k') = latency(k') / ${baselineLatency} (baseline latency)
   - Registers per thread, shared memory, block dim, occupancy
   ${NCU_CMD ? `Profile: ${NCU_CMD}` : ''}

# Baseline latency: ${baselineLatency} μs
# Previous best: ${bestKernel.latency === Infinity ? 'N/A' : bestKernel.latency.toFixed(1) + ' μs (' + bestKernel.speedup.toFixed(2) + 'x)'}

Return evaluation results.`, {
    label: `eval-t${t}`,
    phase: 'Evaluate',
    schema: {
      type: 'object',
      properties: {
        compiled: { type: 'boolean' },
        correct: { type: 'boolean' },
        latency_us: { type: 'number' },
        speedup: { type: 'number' },
        behavioral_features: {
          type: 'object',
          properties: {
            normalized_time: { type: 'number' },
            registers_per_thread: { type: 'number' },
            shared_mem_bytes: { type: 'number' },
            block_dimension: { type: 'number' },
            occupancy: { type: 'number' },
          },
        },
        hw_signature: {
          type: 'object',
          properties: {
            dram_throughput_pct: { type: 'number' },
            l2_throughput_pct: { type: 'number' },
            sm_throughput_pct: { type: 'number' },
          },
        },
        error_details: { type: 'string' },
      },
      required: ['compiled', 'correct'],
    },
  })

  // ===========================================================================
  // Bandit Update (Algorithm 1, lines 20-23)
  // ===========================================================================
  phase('Update')

  const compiled = evalResult?.compiled || false
  const correct = evalResult?.correct || false
  const newLatency = evalResult?.latency_us || baselineLatency
  const newSpeedup = evalResult?.speedup || (baselineLatency / newLatency)

  // Reward: r_t = max(0, (T(k_t) - T(k'_t)) / T(k_t)), zero for failures
  let reward = 0
  if (compiled && correct && newLatency < selectedKernel.latency) {
    reward = Math.max(0, Math.min(1, (selectedKernel.latency - newLatency) / selectedKernel.latency))
  }

  // Update bandit statistics: incremental mean update
  const statKey = `${selectedCluster}_${selectedStrategy}`
  if (banditStats[statKey]) {
    banditStats[statKey].count += 1
    const n = banditStats[statKey].count
    banditStats[statKey].mean_reward += (reward - banditStats[statKey].mean_reward) / n
  }

  totalReward += reward

  // Add to candidate pool if valid
  if (compiled && correct) {
    const newId = candidatePool.length
    const features = evalResult?.behavioral_features || {
      normalized_time: newLatency / baselineLatency,
      registers_per_thread: selectedKernel.features.registers_per_thread,
      shared_mem_bytes: selectedKernel.features.shared_mem_bytes,
      block_dimension: selectedKernel.features.block_dimension,
      occupancy: selectedKernel.features.occupancy,
    }

    candidatePool.push({
      id: newId,
      code: generatedCode,
      latency: newLatency,
      speedup: baselineLatency / newLatency,
      features,
      hw_signature: evalResult?.hw_signature || selectedKernel.hw_signature,
      cluster: selectedCluster,
      source_strategy: selectedStrategy,
    })

    // Update best
    if (newLatency < bestKernel.latency) {
      bestKernel = { code: generatedCode, latency: newLatency, speedup: baselineLatency / newLatency }
      log(`  NEW BEST: ${newLatency.toFixed(1)}μs (${bestKernel.speedup.toFixed(2)}x) via ${selectedStrategy}`)
    }
  }

  iterationLog.push({
    t,
    cluster: selectedCluster,
    strategy: selectedStrategy,
    kernel_id: selectedKernel.id,
    compiled,
    correct,
    reward: reward.toFixed(4),
    latency: newLatency,
    speedup: compiled && correct ? (baselineLatency / newLatency).toFixed(2) : '0',
    cumulative_reward: totalReward.toFixed(3),
  })

  const statusEmoji = compiled && correct ? (reward > 0 ? '↑' : '→') : '✗'
  log(`  ${statusEmoji} t=${t}: ${selectedStrategy}@C${selectedCluster} → ${compiled && correct ? newLatency.toFixed(1) + 'μs (' + (baselineLatency / newLatency).toFixed(2) + 'x)' : 'FAIL'} r=${reward.toFixed(3)} Σr=${totalReward.toFixed(2)}`)
}

// =============================================================================
// Phase: Report — Final summary
// =============================================================================
phase('Report')

// Compute strategy statistics
const strategyStats = {}
for (const s of STRATEGIES) {
  const entries = iterationLog.filter(e => e.strategy === s)
  const successes = entries.filter(e => e.compiled && e.correct)
  strategyStats[s] = {
    attempts: entries.length,
    successes: successes.length,
    avg_reward: entries.length > 0 ? entries.reduce((sum, e) => sum + parseFloat(e.reward), 0) / entries.length : 0,
  }
}

const finalReport = await agent(`Write a KernelBand optimization report.

# KernelBand Results
- Target: ${GPU_TARGET}
- Operation: ${OP_DESCRIPTION}
- Iterations: ${ITERATIONS}
- Clusters: ${NUM_CLUSTERS} (re-cluster period: ${RECLUSTER_PERIOD})
- Evidence mode: ${EVIDENCE_MODE}
- Feature vector artifact: ${FEATURE_VECTOR_RESULT_PATH}
- Hardware signature artifact: ${HARDWARE_SIGNATURE_RESULT_PATH}

# Performance
- Baseline: ${baselineLatency} μs
- Best: ${bestKernel.latency === Infinity ? 'No improvement' : bestKernel.latency.toFixed(1) + ' μs'}
- Best Speedup: ${bestKernel.speedup.toFixed(2)}x
- Cumulative Reward: ${totalReward.toFixed(3)}
- Candidate Pool Size: ${candidatePool.length}

# Strategy Statistics:
${STRATEGIES.map(s => {
  const st = strategyStats[s]
  return `- ${s}: ${st.attempts} attempts, ${st.successes} successes, avg_reward=${st.avg_reward.toFixed(3)}`
}).join('\n')}

# Bandit State (final μ̂ and N):
${Object.entries(banditStats).map(([k, v]) => `  ${k}: μ̂=${v.mean_reward.toFixed(3)}, N=${v.count}, mask=${v.mask}`).join('\n')}

# Iteration Log (last 10):
${iterationLog.slice(-10).map(e => `  t=${e.t}: ${e.strategy}@C${e.cluster} → ${e.speedup}x (r=${e.reward})`).join('\n')}

Analyze:
1. Which strategies were most/least effective? Why (relate to hardware bottleneck)?
2. Did the bandit converge to exploiting the best strategy or keep exploring?
3. Was hardware-aware pruning effective? How many invalid strategies were avoided?
4. Clustering effectiveness: did kernels in the same cluster respond similarly?
5. Recommendations for further optimization (more iterations, different K, etc.)`, {
  label: 'report',
  phase: 'Report',
})

return {
  gpu_target: GPU_TARGET,
  operation: OP_DESCRIPTION,
  iterations: ITERATIONS,
  baseline_latency_us: baselineLatency,
  best_latency_us: bestKernel.latency === Infinity ? null : bestKernel.latency,
  best_speedup: bestKernel.speedup,
  cumulative_reward: totalReward,
  candidate_pool_size: candidatePool.length,
  strategy_stats: strategyStats,
  bandit_stats: banditStats,
  feature_vector_result_path: FEATURE_VECTOR_RESULT_PATH,
  hardware_signature_result_path: HARDWARE_SIGNATURE_RESULT_PATH,
  evidence_mode: EVIDENCE_MODE,
  iteration_log: iterationLog,
  report: finalReport,
}
