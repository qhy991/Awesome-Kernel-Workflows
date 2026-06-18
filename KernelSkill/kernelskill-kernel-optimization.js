export const meta = {
  name: 'kernelskill-kernel-optimization',
  description: 'Multi-agent CUDA kernel optimization with a dual-level memory: a curated long-term expert "skill" library (deterministic decision policy + method knowledge) and short-term per-task optimize/repair trajectory memory. Each refinement round branches into a repair path (invalid kernel) or an optimize path (valid kernel) driven by ncu+nsys profiling (KernelSkill / KernelMem methodology).',
  whenToUse: 'When generating and optimizing custom CUDA kernels from a PyTorch reference, and you want optimization-method selection to be explicit, auditable, and grounded in profiling evidence rather than relying on the model\'s implicit heuristics. KernelSkill seeds several candidate kernels, then runs a fixed number of refinement rounds. Each round profiles the current kernel and either REPAIRS it (if it fails to compile / is incorrect) using chained repair memory to avoid cyclic fixes, or OPTIMIZES it (if valid) by extracting code features, running a deterministic gate to obtain an allowed-methods set, retrieving the rationale + implementation cues for the chosen method, planning, and applying the edit. Best for KernelBench-style operator/model tasks where multi-round, knowledge-driven refinement matters.',
  phases: [
    { title: 'Setup', detail: 'Read the PyTorch reference task, establish the eager baseline latency' },
    { title: 'Seed', detail: 'Generate several candidate CUDA kernels for correctness, pick the best valid seed' },
    { title: 'Evaluate', detail: 'Compile, verify correctness vs reference, profile with ncu+nsys' },
    { title: 'Repair', detail: 'Invalid branch: Diagnoser -> Repairer using chained repair memory' },
    { title: 'Optimize', detail: 'Valid branch: extract features -> gate (allowed methods) -> retrieve skill -> plan -> apply' },
    { title: 'Iterate', detail: 'Update short-term memory and best kernel, advance to the next round' },
    { title: 'Report', detail: 'Final report: best kernel, speedup trajectory, skills applied, repair history' },
  ],
}

const WORKFLOW_NAME = 'kernelskill-kernel-optimization'


// --- BEGIN inlined arg_guard (Workflow runtime parses scripts as bare scripts,
//                              not ES modules; static imports are rejected) ---
function __unwrapArgs(rawArgs) {
  if (rawArgs == null) return {}
  if (typeof rawArgs === 'object' && !Array.isArray(rawArgs)) return rawArgs
  if (typeof rawArgs === 'string') {
    const trimmed = rawArgs.trim()
    if (trimmed === '') return {}
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        const parsed = JSON.parse(trimmed)
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed
        throw new Error('arg_guard: parsed JSON value is not a plain object')
      } catch (e) { throw new Error(`arg_guard: invalid JSON args: ${e.message}`) }
    }
    const out = {}
    const re = /(\w[\w.-]*)=("(?:\\\\\"|[^"])*"|\'(?:\\\\\'|[^\'])*\'|\S+)/g
    let m
    while ((m = re.exec(trimmed)) !== null) {
      let v = m[2]
      if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
        v = v.slice(1, -1)
      }
      out[m[1]] = v
    }
    if (Object.keys(out).length === 0) {
      throw new Error(`arg_guard: workflow args is a non-empty string but contains no key=value pairs and is not JSON. First 160 chars: ${trimmed.slice(0, 160)}`)
    }
    return out
  }
  throw new Error(`arg_guard: workflow args has unexpected type: ${typeof rawArgs}`)
}
// eslint-disable-next-line no-global-assign
args = __unwrapArgs(typeof args === 'undefined' ? undefined : args)
// --- END inlined arg_guard ---
// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.

const WORKFLOW_SUITABILITY = {
  supported_languages: ['cuda'],
  supported_problem_types: ['cuda-kernel-optimization', 'cuda-kernel-generation'],
  problem_types: ['CUDA custom kernel generation from PyTorch reference', 'repair-or-optimize loop with memory/profiler guidance'],
  reason: 'KernelSkill is CUDA-specific and depends on CUDA compiler/verifier/profiler evidence.',
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
        `${WORKFLOW_NAME} is not suitable for language="${args.language}". ` +
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
        `${WORKFLOW_NAME} is not suitable for problem_type="${args.problem_type}". ` +
        `Supported problem types: ${WORKFLOW_SUITABILITY.supported_problem_types.join(', ')}. ` +
        `Typical use cases: ${WORKFLOW_SUITABILITY.problem_types.join('; ')}. ` +
        `Reason: ${WORKFLOW_SUITABILITY.reason}`
      )
    }
  }
}

function resolveBackendAxis() {
  const b = args.backend ? normalizeSuitabilityValue(args.backend) : null
  const l = args.language ? normalizeSuitabilityValue(args.language) : null
  if (b && l && b !== l) {
    throw new Error(`Conflicting args: backend="${args.backend}" vs language="${args.language}". Pass only one.`)
  }
  if (args.backend && !args.backend_dir) {
    throw new Error(`args.backend="${args.backend}" requires args.backend_dir; driver dispatch has no implicit-resolve path.`)
  }
  return b || l || null
}
const RESOLVED_BACKEND = resolveBackendAxis()
const USE_DRIVER = !!args.backend_dir

if (!USE_DRIVER) {
  assertWorkflowSuitability()
}

// =============================================================================
// KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization
// =============================================================================
//
// Source: KernelSkill: A Multi-Agent Framework for GPU Kernel Optimization
//         arXiv:2603.10085  (code: github.com/0satan0/KernelMem, CC-BY-4.0)
//
// Faithful adaptation of the KernelSkill / KernelMem method:
//
//   1. SEED         Generator produces a SET of seed kernels (correctness-first);
//                   Reviewer evaluates; the best valid seed becomes the working kernel.
//   2. REFINE LOOP  For up to `iterations` rounds, each round is a TWO-BRANCH control flow:
//                     - if the current kernel is INVALID (compile/correctness fail):
//                         REPAIR branch: Diagnoser -> Repairer, conditioned on a CHAINED
//                         repair memory so the model does not oscillate between the same
//                         faulty variants (cyclic repair).
//                     - if the current kernel is VALID:
//                         OPTIMIZE branch: Feature Extractor -> deterministic Gate
//                         (machine_check) -> allowed_methods -> Retrieve method knowledge
//                         -> Planner -> Optimizer. The Planner is conditioned on the
//                         short-term optimization memory (which methods were tried + their
//                         measured outcomes) so it revises plans consistently.
//   3. REPORT       Summarize the best kernel, the speedup trajectory, the skills applied,
//                   and the repair history.
//
// DUAL-LEVEL MEMORY (the paper's core idea):
//   * LONG-TERM  = a curated, cross-task "skill" library. Two stores:
//       (a) Deterministic Decision Policy: field_mapping (raw NCU -> normalized scalars),
//           derived_fields, headroom tiers, bottleneck priority rules, and a decision
//           table mapping matched evidence -> allowed_methods. This SCREENS feasible
//           methods from profiling/code evidence (machine_check layer).
//       (b) Method Knowledge (llm_assist): per-method rationale + implementation cues,
//           making the selected action interpretable and easy for the LLM to execute.
//     The skill library is embedded below (faithful to memorybank/
//     bottleneck_headroom_kernelstructure.yaml) and can be OVERRIDDEN / PERSISTED via
//     `skill_library_path`. It is curated offline and is the reusable cross-task asset.
//   * SHORT-TERM = per-task trajectory state, two modes:
//       - optimization memory: plan -> result history (method, evidence, measured speedup)
//       - repair memory: chained segments starting at the first faulty kernel, used to
//         discourage cyclic repair.
//
// FEEDBACK SIGNAL: speedup = reference (Torch Eager) latency / candidate latency, with
// ncu (Nsight Compute) + nsys (Nsight Systems) metrics steering WHICH method is selected.
// Higher speedup is better. Correctness is checked against the PyTorch reference within a
// numerical tolerance (rtol/atol) using the KernelBench validation criteria.
//
// Usage:
//   Workflow({name: 'kernelskill-kernel-optimization', args: {
//     reference_path: '/path/to/level1/19_ReLU.py',  // PyTorch reference task (required)
//     op_description: 'ReLU activation',
//     target_gpu: 'A100-80GB',
//     iterations: 15,                   // refinement rounds (paper uses 15)
//     seed_candidates: 3,               // number of seed kernels generated
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     ncu_binary: '<user-provided ncu binary path>',
//     nsys_binary: '<user-provided nsys binary path>',
//     exp_dir: '/tmp/kernelskill_exp',
//     skill_library_path: '',           // optional: load/override the long-term skill library (JSON)
//     rtol: 0.01,
//     atol: 0.01,
//     warmup: 25,
//     repeat: 100,
//     run_timestamp_iso: '2026-06-02T17:00:00+08:00',
//   }})
//
// =============================================================================

// --- Required args ---
const REFERENCE_PATH = args.problem_path
const PROBLEM_DEFINITION = args.problem_definition || ''
const INPUT_MODE = 'generate_then_optimize'

// --- Optional args ---
const OP_DESC = args.op_description || 'CUDA kernel from PyTorch reference'
const TARGET_GPU = args.target_gpu || 'A100-80GB'
const ROUNDS = args.iterations || 15
const SEED_CANDIDATES = args.seed_candidates || 3
const BENCH_CMD = args.benchmark_command || ''
const NCU_BINARY = args.ncu_binary || ''
const NSYS_BINARY = args.nsys_binary || ''
const EXP_DIR = args.exp_dir || '/tmp/kernelskill_exp'
const SKILL_LIBRARY_PATH = args.skill_library_path || ''
const RTOL = args.rtol != null ? args.rtol : 0.01
const ATOL = args.atol != null ? args.atol : 0.01
const WARMUP = args.warmup || 25
const REPEAT = args.repeat || 100
const RUN_TS = args.run_timestamp_iso || 'unknown'

// --- Backend driver wiring (P5d Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_LANG_TOKEN = 'cuda'
const LEGACY_FENCE_TOKEN = 'cuda'
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}

let DRIVER = null
let DRIVER_LANG_FENCE = LEGACY_FENCE_TOKEN
let DRIVER_IMPL_REQUIREMENTS = ''
let DRIVER_SOURCE_EXT = ''
let DRIVER_BACKEND_ID = RESOLVED_BACKEND || ''

function langToken(legacy) {
  return USE_DRIVER ? DRIVER_LANG_FENCE : legacy
}
function fenceToken() {
  return USE_DRIVER ? DRIVER_LANG_FENCE : LEGACY_FENCE_TOKEN
}
function kernelPathForRound(round) {
  const ext = USE_DRIVER ? (DRIVER_SOURCE_EXT || '.cu') : '.cu'
  return `${EXP_DIR}/round_${round}${ext}`
}

if (!REFERENCE_PATH && !PROBLEM_DEFINITION) {
  throw new Error('Provide one of problem_path or problem_definition')
}

// --- State ---
let baselineLatency = null      // Torch Eager reference latency (ms), set once
let bestKernelCode = null       // best VALID kernel found so far
let bestSpeedup = null          // speedup of bestKernelCode vs baseline
let currentKernelCode = null    // the kernel carried into the current round
let currentValid = false        // whether currentKernelCode is compile+correct
let currentLatency = null       // measured latency (ms) of currentKernelCode if valid
// Short-term OPTIMIZE memory: [{round, method, evidence, speedup_before, speedup_after, outcome}]
let optimizeMemory = []
// Short-term REPAIR memory (chained): [{round, root_cause, repair_strategy, fixed, error_excerpt}]
let repairMemory = []
// Trajectory of per-round speedups (null when invalid)
let speedupTrajectory = []
// Skills applied (method names) across the run
let skillsApplied = []

// =============================================================================
// LONG-TERM MEMORY: the curated expert "skill" library.
// Faithful to KernelMem/memorybank/bottleneck_headroom_kernelstructure.yaml:
//   - Layer 1 (machine_check): deterministic gating -> allowed_methods
//   - Layer 2 (llm_assist):    method knowledge (rationale + implementation cues)
// Override / persist via skill_library_path.
// =============================================================================
const DEFAULT_SKILL_LIBRARY = {
  version: '2.1',
  purpose: 'Two-layer optimization-rule system: (1) machine_check deterministic gating; (2) llm_assist non-binding method knowledge used only AFTER gating.',

  // --- Layer 1: deterministic decision policy -------------------------------
  machine_check: {
    // Raw NCU keys -> normalized scalar fields the gate reasons over.
    field_mapping: {
      dram_throughput_pct: 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
      l2_throughput_pct: 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
      l1_throughput_pct: 'l1tex__throughput.avg.pct_of_peak_sustained_active',
      sm_throughput_pct: 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
      achieved_occupancy_pct: 'sm__warps_active.avg.pct_of_peak_sustained_active',
      registers_per_thread: 'launch__registers_per_thread',
      kernel_duration_ns: 'gpu__time_duration.avg',
      stall_long_scoreboard_ratio: 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio',
      stall_short_scoreboard_ratio: 'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.ratio',
      branch_divergent_cnt: 'smsp__sass_branch_targets_threads_divergent.avg',
      branch_uniform_cnt: 'smsp__sass_branch_targets_threads_uniform.avg',
    },
    // Deterministic derived signals (computed from field_mapping + code features).
    derived_fields: {
      branch_divergence_pct: '100 * branch_divergent_cnt / max(1, (branch_divergent_cnt + branch_uniform_cnt))',
      memory_dependency_stall_ratio: 'stall_long_scoreboard_ratio + stall_short_scoreboard_ratio',
      primary_limiter_util_pct: 'max(dram_throughput_pct, l2_throughput_pct, sm_throughput_pct)',
      // Mechanism gate: cache/reuse speedups only plausible if reuse exists.
      reuse_possible: 'has_reuse OR is_naive_gemm OR is_gemm_kloop OR is_stencil_conv',
      // Bytes structurally unavoidable: S0 streaming, no reuse, already vectorized.
      bytes_structurally_unavoidable: '(kernel_structure_id == 0) AND streaming_no_reuse AND has_vector_load_store',
      needs_alignment_or_tail_fix: '(NOT is_aligned_vector_access) OR has_tail_handling_overhead',
    },
    // Headroom tiers gate how aggressive optimization should be.
    headroom_tiers: {
      'Tier-H': 'primary_limiter_util_pct < 60  (high headroom)',
      'Tier-M': '60 <= primary_limiter_util_pct <= 80  (medium headroom)',
      'Tier-L': 'primary_limiter_util_pct > 80 AND bytes_structurally_unavoidable  (low headroom)',
    },
    // Hard priority rules (deterministic).
    bottleneck_priority_rules: [
      'BandwidthDominatesLatency: if dram_throughput_pct>=80 OR l2_throughput_pct>=80 -> force bottleneck=memory_bandwidth (do NOT classify as memory_latency even if memory-dependency stalls exist)',
    ],
    // Decision table: matched evidence pattern -> candidate allowed_methods set.
    // The gate emits the allowed_methods for the matched case; the LLM MUST pick from it.
    decision_table: [
      {
        case_id: 'MEM_BW_SATURATED',
        when: 'dram_throughput_pct>=80 OR l2_throughput_pct>=80',
        allowed_methods: ['vectorized_memory_access', 'memory_coalescing', 'reduce_bytes_via_reuse', 'data_layout_transformation'],
      },
      {
        case_id: 'MEM_LATENCY_BOUND',
        when: 'memory_dependency_stall_ratio>=0.30 AND primary_limiter_util_pct<60',
        allowed_methods: ['shared_memory_tiling', 'register_blocking', 'software_pipelining', 'increase_ilp_async_copy'],
      },
      {
        case_id: 'COMPUTE_BOUND',
        when: 'sm_throughput_pct>=70 AND memory_not_dominant',
        allowed_methods: ['tensor_core_utilization', 'instruction_mix_optimization', 'fast_math_intrinsics', 'warp_level_reduction'],
      },
      {
        case_id: 'LOW_OCCUPANCY',
        when: 'achieved_occupancy_pct<40 AND primary_limiter_util_pct<60',
        allowed_methods: ['block_size_tuning', 'register_pressure_reduction', 'occupancy_tuning', 'thread_coarsening'],
      },
      {
        case_id: 'DIVERGENT',
        when: 'branch_divergence_pct>=5',
        allowed_methods: ['branch_reduction', 'thread_work_remapping', 'predication'],
      },
      {
        case_id: 'GEMM_REWRITE',
        when: 'tc_eligible AND (is_naive_gemm OR is_gemm_kloop OR has_k_loop)',
        allowed_methods: ['tensor_core_utilization', 'shared_memory_tiling', 'register_blocking'],
      },
      {
        case_id: 'NEEDS_ALIGN_OR_TAIL_FIX',
        when: 'needs_alignment_or_tail_fix',
        allowed_methods: ['fix_vector_alignment', 'improve_tail_handling', 'vectorized_memory_access'],
      },
      {
        case_id: 'NO_MATCH',
        when: 'no deterministic case matched',
        allowed_methods: [],  // empty -> LLM may self-generate a method, justified by evidence
      },
    ],
    // Code-structure features the Feature Extractor must provide (rule-based scan preferred,
    // LLM-assisted fallback). These participate in derived_fields and the decision table.
    code_features_schema: [
      'has_reuse', 'streaming_no_reuse', 'has_vector_load_store', 'is_aligned_vector_access',
      'has_tail_handling_overhead', 'tc_eligible', 'is_pointwise', 'uses_transcendentals',
      'is_naive_gemm', 'has_k_loop', 'is_gemm_kloop', 'is_stencil_conv',
      'has_shared_memory_tile', 'uses_vector_types', 'has_bounds_check',
      'kernel_structure_id', 'has_multiple_kernels_in_forward',
    ],
  },

  // --- Layer 2: method knowledge (rationale + implementation cues) ----------
  llm_assist: {
    vectorized_memory_access: 'Use float4/int4/half2 wide transactions to raise bytes-per-instruction. Implementation: reinterpret_cast contiguous aligned pointers to vector types; handle the tail. Trigger: bandwidth-saturated, scalar access.',
    memory_coalescing: 'Map consecutive threads to consecutive addresses so a warp issues one cache-line transaction. Implementation: index by threadIdx.x along the contiguous dimension; transpose tiles in shared memory if needed.',
    reduce_bytes_via_reuse: 'Cut DRAM traffic by reusing loaded data on-chip. Implementation: stage into __shared__ once, read many times. Only valid when reuse_possible.',
    data_layout_transformation: 'Convert AoS->SoA so accesses are contiguous/coalesceable. Implementation: split struct fields into parallel arrays.',
    shared_memory_tiling: 'Cache reused operands in shared memory to hide global-memory latency. Implementation: classic tiled GEMM/stencil; __syncthreads between load and compute; pad to avoid bank conflicts.',
    register_blocking: 'Each thread computes a micro-tile from register-resident operands to raise arithmetic intensity. Implementation: unroll inner loops, accumulate in registers.',
    software_pipelining: 'Overlap memory loads with compute across loop iterations. Implementation: double-buffer shared memory; prefetch next tile while computing current.',
    increase_ilp_async_copy: 'Raise independent in-flight memory ops to hide long_scoreboard stalls. Implementation: process multiple elements per thread; use cp.async on Ampere+.',
    tensor_core_utilization: 'Map matmul-shaped compute to tensor cores. Implementation: wmma/mma with half/bf16 inputs and aligned 16x16x16 tiles; or call cuBLASLt.',
    instruction_mix_optimization: 'Balance ALU/SFU/memory pipes. Implementation: hoist invariant math, replace divides with reciprocals, reduce redundant instructions.',
    fast_math_intrinsics: 'Use __expf/__logf/rsqrtf etc. for SFU-bound transcendental kernels when tolerance permits. Implementation: swap libm calls for intrinsics; verify within rtol/atol.',
    warp_level_reduction: 'Use __shfl_down_sync warp reductions instead of shared-memory + __syncthreads for reductions. Implementation: warp-stride reduce, then one atomic/shared write per warp.',
    block_size_tuning: 'Pick threads-per-block to raise occupancy without spilling. Implementation: sweep 128/256/512; set __launch_bounds__.',
    register_pressure_reduction: 'Lower registers/thread to raise occupancy. Implementation: shrink live ranges, recompute cheap values, use __launch_bounds__.',
    occupancy_tuning: 'Trade shared-mem/registers for more resident warps. Implementation: reduce per-block shared memory; tune block count.',
    thread_coarsening: 'Assign more work per thread to amortize launch/sync overhead. Implementation: grid-stride loop with >1 element per iteration.',
    branch_reduction: 'Remove divergent control flow within a warp. Implementation: hoist branches out of warps, use arithmetic/masking instead of if.',
    thread_work_remapping: 'Reassign work so a warp follows one control path. Implementation: sort/partition work by branch outcome.',
    predication: 'Convert short branches to predicated execution. Implementation: select()/ternary on uniform conditions.',
    fix_vector_alignment: 'Ensure vector loads are 16B-aligned to avoid split transactions. Implementation: align base pointers; pad rows to multiples of the vector width.',
    improve_tail_handling: 'Remove branchy remainder overhead. Implementation: process the bulk vectorized, handle the residual with a single masked epilogue.',
  },
}

// =============================================================================
// Helpers
// =============================================================================
function asJsonBlock(obj, limit) {
  const s = JSON.stringify(obj, null, 2)
  return limit && s.length > limit ? s.slice(0, limit) + '\n... (truncated)' : s
}

function buildOptimizeMemoryBlock(mem, maxItems) {
  if (!mem.length) return '(no prior optimization attempts this task)'
  const recent = mem.slice(-maxItems)
  return recent.map((m, i) =>
    `${i + 1}. round ${m.round}: method="${m.method}" | evidence: ${m.evidence} | ` +
    `speedup ${m.speedup_before != null ? m.speedup_before.toFixed(2) : '?'}x -> ` +
    `${m.speedup_after != null ? m.speedup_after.toFixed(2) : '?'}x | outcome: ${m.outcome}`
  ).join('\n')
}

function buildRepairMemoryBlock(mem) {
  if (!mem.length) return '(no prior repair attempts in this chain)'
  return mem.map((m, i) =>
    `${i + 1}. round ${m.round}: root_cause="${m.root_cause}" | strategy="${m.repair_strategy}" | ` +
    `fixed=${m.fixed} | error: ${(m.error_excerpt || '').slice(0, 160)}`
  ).join('\n')
}

// =============================================================================
// Phase: Setup — read the PyTorch reference, establish the eager baseline
// =============================================================================
phase('Setup')

if (USE_DRIVER) {
  DRIVER = await agent(
    `Load the backend driver at ${BACKEND_DIR} and return its manifest plus idioms verbatim.\n` +
    `1. Run exactly: \`cat ${driverPath('manifest.json')}\` and parse JSON.\n` +
    `2. Run exactly: \`cat ${driverPath('idioms.json')}\` and parse JSON.\n` +
    `Return {present, backend_id, source_ext, aux_ext, lang_fence, impl_requirements, methods}.`,
    { label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH })
  if (!DRIVER || DRIVER.present === false) {
    throw new Error(`No backend driver present at ${BACKEND_DIR}. Provide a valid backend_dir or omit it for the legacy path.`)
  }
  if (RESOLVED_BACKEND && DRIVER.backend_id && normalizeSuitabilityValue(DRIVER.backend_id) !== RESOLVED_BACKEND) {
    throw new Error(`backend_dir manifest backend_id="${DRIVER.backend_id}" conflicts with args.backend/language="${RESOLVED_BACKEND}".`)
  }
  DRIVER_LANG_FENCE = DRIVER.lang_fence || DRIVER_LANG_FENCE
  DRIVER_IMPL_REQUIREMENTS = DRIVER.impl_requirements || ''
  DRIVER_SOURCE_EXT = DRIVER.source_ext || DRIVER_SOURCE_EXT
  DRIVER_BACKEND_ID = DRIVER.backend_id || DRIVER_BACKEND_ID
  log(`Driver loaded: ${DRIVER_BACKEND_ID} (fence=${DRIVER_LANG_FENCE})`)
}

// Load (optionally) an external long-term skill library to override the default.
let skillLibrary = DEFAULT_SKILL_LIBRARY
if (SKILL_LIBRARY_PATH) {
  const loaded = await agent(`Load the KernelSkill long-term skill library (JSON) from: ${SKILL_LIBRARY_PATH}

Read the file. If it exists and is valid JSON with "machine_check" and "llm_assist" keys, return it as loaded=true with the parsed object in "library". If it does not exist or is invalid, return loaded=false (the workflow will fall back to the embedded default library) and explain why in "note".

Return ONLY the JSON object.`, {
    label: 'load-skill-library',
    phase: 'Setup',
    schema: {
      type: 'object',
      properties: {
        loaded: { type: 'boolean' },
        library: { type: 'object' },
        note: { type: 'string' },
      },
      required: ['loaded'],
    },
  })
  if (loaded && loaded.loaded && loaded.library && loaded.library.machine_check) {
    skillLibrary = loaded.library
    log(`Loaded external skill library from ${SKILL_LIBRARY_PATH}`)
  } else {
    log(`Using embedded default skill library (${loaded && loaded.note ? loaded.note : 'no external library'})`)
  }
}

const setup = await agent(`Read the PyTorch reference task at: ${REFERENCE_PATH || '(not provided)'}
If problem_definition is provided, use it as the authoritative task:
${PROBLEM_DEFINITION || '(not provided)'}

This is a KernelBench-style task: a PyTorch \`Model\` (forward pass) plus \`get_inputs()\` / \`get_init_inputs()\`. Analyze it and return:
- reference_code: the full source
- op_type: operation type (e.g., "relu", "matmul", "conv2d", "layernorm", "fused_mlp")
- forward_summary: what the forward pass computes, in 1-2 sentences
- input_shapes: the tensor shapes from get_inputs()
- dtype: the dominant dtype (e.g., float32)
- num_ops: rough count of distinct operators in forward (1 = pointwise, >1 = fusion opportunity)
- has_matmul: whether the forward contains a matmul/linear (tensor-core opportunity)

Return ONLY the JSON object.`, {
  label: 'read-reference',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      reference_code: { type: 'string' },
      op_type: { type: 'string' },
      forward_summary: { type: 'string' },
      input_shapes: { type: 'string' },
      dtype: { type: 'string' },
      num_ops: { type: 'number' },
      has_matmul: { type: 'boolean' },
    },
    required: ['reference_code', 'op_type', 'forward_summary'],
  },
})

const referenceCode = setup.reference_code
const opType = setup.op_type
log(`Reference: ${opType} — ${setup.forward_summary}`)

const baseline = await agent(`You are a GPU benchmarking expert. Establish the Torch Eager baseline latency for this PyTorch reference, which is the denominator for speedup (speedup = baseline_latency / kernel_latency).

# Reference task: ${REFERENCE_PATH}
# Operation: ${OP_DESC} (${opType})
# Target GPU: ${TARGET_GPU}
# Inputs: ${setup.input_shapes || 'see get_inputs()'}

# Instructions
1. Create ${EXP_DIR}/baseline/.
2. ${BENCH_CMD ? `Run the reference benchmark using the user-provided command: ${BENCH_CMD}` : 'No benchmark_command provided; do not construct a timing harness.'}
3. Report the mean forward latency in milliseconds only when measured.

If the GPU/torch environment or benchmark_command is unavailable, return baseline_available=false and mark baseline_latency_ms as unmeasured.

Return the baseline result.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the value you just measured (status="done" if baseline measured, else "error"):
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"<done|error>","technique":"eager_baseline","speedup":null,"note":"<baseline latency ms + whether benchmark_command was available>"}`, {
  label: 'eager-baseline',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      baseline_latency_ms: { type: 'number' },
      baseline_available: { type: 'boolean' },
      harness_notes: { type: 'string' },
    },
    required: ['baseline_latency_ms'],
  },
})

baselineLatency = baseline.baseline_latency_ms
log(`Torch Eager baseline: ${baselineLatency}ms (available=${baseline.baseline_available})`)

// =============================================================================
// Phase: Seed — generate several candidate kernels (correctness-first), pick best
// =============================================================================
phase('Seed')

const seedKernels = await parallel(
  Array.from({ length: SEED_CANDIDATES }, (_, i) => () =>
    agent(`You are the KernelSkill Generator. Translate this PyTorch reference into an EQUIVALENT implementation backed by custom CUDA kernel(s). Your ONLY goal here is CORRECTNESS — it must compile and match the reference within tolerance (rtol=${RTOL}, atol=${ATOL}). Do not over-optimize; produce a clean, materialized baseline.

# PyTorch Reference:
\`\`\`python
${referenceCode.substring(0, 4000)}
\`\`\`

# Operation: ${OP_DESC} (${opType})
# Target GPU: ${TARGET_GPU}
# Inputs: ${setup.input_shapes || 'see get_inputs()'}

# Requirements
1. Materialize as many operator-level custom CUDA kernels as is natural (this gives later optimization more surface to work with).
2. Use torch.utils.cpp_extension (load_inline) or an equivalent so the module is drop-in for the reference.
3. Keep the public forward() signature identical to the reference Model.
4. Functionally correct first; performance comes later.
5. This is seed candidate ${i + 1}/${SEED_CANDIDATES} — vary the decomposition/strategy from a naive baseline so the seeds are diverse.

Return the complete kernel source.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Seed","ts":"<ts>","status":"done","candidate_id":"seed-${i}","technique":"<your decomposition/strategy for this seed>","speedup":null,"note":"<kernels materialized + how this seed differs from a naive baseline>"}`, {
      label: `seed-${i}`,
      phase: 'Seed',
      schema: {
        type: 'object',
        properties: {
          code: { type: 'string' },
          strategy: { type: 'string' },
          kernels_materialized: { type: 'number' },
        },
        required: ['code'],
      },
    })
  )
)

const validSeeds = seedKernels.filter(Boolean).filter(s => s.code)
log(`Generated ${validSeeds.length}/${SEED_CANDIDATES} seed candidates`)

// Reviewer evaluates each seed; pick the best VALID one (fastest), else the first that compiles.
const seedEvals = await parallel(
  validSeeds.map((seed, i) => () =>
    agent(`You are the KernelSkill Reviewer (Compiler + Verifier + Profiler). Evaluate this SEED kernel against the PyTorch reference.

# Reference task: ${REFERENCE_PATH}
# Torch Eager baseline latency: ${baselineLatency}ms
# Tolerance: rtol=${RTOL}, atol=${ATOL}

# Seed kernel (candidate ${i + 1}):
\`\`\`${fenceToken()}
${seed.code.substring(0, 4000)}
\`\`\`

# Steps
1. Compiler: does it build? (check includes, pybind/load_inline, CUDA syntax)
2. Verifier: would its output match the reference within tolerance? (check shapes, indexing, reduction logic, race conditions)
3. Profiler: if it builds and is correct and benchmark_command is provided, measure latency (ms) and compute speedup = ${baselineLatency} / latency.
${BENCH_CMD ? `   Benchmark with: ${BENCH_CMD}` : ''}

If the environment cannot run code or benchmark_command is missing, do a rigorous static evaluation and mark latency/speedup as unmeasured.

Return the evaluation.`, {
      label: `seed-eval-${i}`,
      phase: 'Seed',
      schema: {
        type: 'object',
        properties: {
          is_compilable: { type: 'boolean' },
          is_correct: { type: 'boolean' },
          latency_ms: { type: 'number' },
          speedup: { type: 'number' },
          issues: { type: 'array', items: { type: 'string' } },
        },
        required: ['is_compilable', 'is_correct'],
      },
    })
  )
)

// Choose best seed: prefer valid (compilable+correct) with highest speedup; else first compilable; else first.
let chosenIdx = -1
let chosenSpeedup = -Infinity
for (let i = 0; i < validSeeds.length; i++) {
  const e = seedEvals[i]
  if (!e) continue
  if (e.is_compilable && e.is_correct) {
    const sp = e.speedup || (e.latency_ms ? baselineLatency / e.latency_ms : 0)
    if (sp > chosenSpeedup) { chosenSpeedup = sp; chosenIdx = i }
  }
}
if (chosenIdx === -1) {
  for (let i = 0; i < validSeeds.length; i++) {
    if (seedEvals[i] && seedEvals[i].is_compilable) { chosenIdx = i; break }
  }
}
if (chosenIdx === -1) chosenIdx = 0

currentKernelCode = validSeeds[chosenIdx].code
const chosenEval = seedEvals[chosenIdx] || {}
currentValid = !!(chosenEval.is_compilable && chosenEval.is_correct)
currentLatency = chosenEval.latency_ms || null
if (currentValid) {
  bestKernelCode = currentKernelCode
  bestSpeedup = chosenEval.speedup || (currentLatency ? baselineLatency / currentLatency : 1.0)
  speedupTrajectory.push(bestSpeedup)
} else {
  speedupTrajectory.push(null)
}
log(`Chosen seed #${chosenIdx + 1}: valid=${currentValid}, speedup=${currentValid ? bestSpeedup.toFixed(2) + 'x' : 'n/a'}`)

// =============================================================================
// Refinement loop — two-branch rounds (repair vs optimize)
// =============================================================================
for (let round = 0; round < ROUNDS; round++) {
  log(`\n=== Round ${round + 1}/${ROUNDS} | best: ${bestSpeedup != null ? bestSpeedup.toFixed(2) + 'x' : 'none'} | current valid: ${currentValid} ===`)

  // ===========================================================================
  // Phase: Evaluate — Reviewer (Compiler + Verifier + Profiler via ncu+nsys)
  // ===========================================================================
  phase('Evaluate')

  const review = await agent(`You are the KernelSkill Reviewer for round ${round + 1}. Produce execution feedback for the CURRENT kernel using Compiler + Verifier + Profiler (ncu + nsys).

# Reference task: ${REFERENCE_PATH}
# Torch Eager baseline: ${baselineLatency}ms
# Tolerance: rtol=${RTOL}, atol=${ATOL}
# Target GPU: ${TARGET_GPU}

# Current kernel:
\`\`\`${fenceToken()}
${currentKernelCode.substring(0, 4500)}
\`\`\`

# Steps
1. Compiler: build the kernel; capture status + first errors/warnings.
2. Verifier: compare output to the PyTorch reference within tolerance; report pass/fail + a short error excerpt if failing.
3. Profiler (only if compiles+correct): collect profiler signals into ${EXP_DIR}/round_${round}/ only through user-provided profiler binaries/commands:
   - ncu_binary: ${NCU_BINARY || '(not provided)'} for dram/l2/l1/sm throughput %, achieved occupancy %, registers/thread, kernel duration, long/short scoreboard stall ratios, branch divergent/uniform counts.
   - nsys_binary: ${NSYS_BINARY || '(not provided)'} for kernel_launch_count.
   If profiler binaries/commands are missing, do not invent them; mark profiler metrics as missing evidence.
   Then measure latency only through benchmark_command and compute speedup = ${baselineLatency} / latency when both values are measured.
${BENCH_CMD ? `   You may use: ${BENCH_CMD}` : ''}

If the environment cannot run tools or commands are missing, do a rigorous static evaluation: decide is_compilable / is_correct where possible, set unmeasured metric fields to 0, and mark missing evidence in the notes.

Return the structured review. Always include the normalized ncu_metrics object (keys: dram_throughput_pct, l2_throughput_pct, l1_throughput_pct, sm_throughput_pct, achieved_occupancy_pct, registers_per_thread, kernel_duration_ns, stall_long_scoreboard_ratio, stall_short_scoreboard_ratio, branch_divergent_cnt, branch_uniform_cnt, kernel_launch_count).

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if compiles AND correct, else "error"; speedup is the measured speedup number, or null if unmeasured):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"round-${round}","speedup":<number or null>,"technique":"evaluate_ncu_nsys","note":"<compiled? correct? primary limiter; or the failure reason>"}`, {
    label: `review-${round}`,
    phase: 'Evaluate',
    schema: {
      type: 'object',
      properties: {
        is_compilable: { type: 'boolean' },
        is_correct: { type: 'boolean' },
        latency_ms: { type: 'number' },
        speedup: { type: 'number' },
        error_excerpt: { type: 'string' },
        ncu_metrics: {
          type: 'object',
          properties: {
            dram_throughput_pct: { type: 'number' },
            l2_throughput_pct: { type: 'number' },
            l1_throughput_pct: { type: 'number' },
            sm_throughput_pct: { type: 'number' },
            achieved_occupancy_pct: { type: 'number' },
            registers_per_thread: { type: 'number' },
            kernel_duration_ns: { type: 'number' },
            stall_long_scoreboard_ratio: { type: 'number' },
            stall_short_scoreboard_ratio: { type: 'number' },
            branch_divergent_cnt: { type: 'number' },
            branch_uniform_cnt: { type: 'number' },
            kernel_launch_count: { type: 'number' },
          },
        },
        profile_summary: { type: 'string' },
      },
      required: ['is_compilable', 'is_correct'],
    },
  })

  if (USE_DRIVER) {
    const suffix = `${round}`
    const kPath = kernelPathForRound(round)
    const buildOut = `${EXP_DIR}/round_${round}.artifact`
    const profOut = `${EXP_DIR}/round_${round}.prof.native`
    const resultPath = `${EXP_DIR}/round_${round}.result.json`
    await agent(
      `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
      `Return its stdout JSON verbatim.`,
      { label: `driver-build-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    const runOut = await agent(
      `${driverSh('run.sh', `--artifact ${buildOut} --kernel ${kPath} --result ${resultPath}`)}\n` +
      `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
      { label: `driver-run-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    await agent(
      `${driverSh('profile.sh', `--artifact ${buildOut} --kernel ${kPath} --out ${profOut}`)}\n` +
      `Return {ok, native_path}.`,
      { label: `driver-profile-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    const evidenceOut = await agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
      `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
      { label: `driver-to-evidence-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    const diagOut = await agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
      `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
      { label: `driver-diagnose-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    await agent(
      `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${resultPath}\`.\n` +
      `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
      { label: `driver-anti-cheat-${suffix}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH })
    review.driver_envelope = {
      latency_ms: Number((runOut && runOut.latency_ms) || 0),
      bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
      backend_id: DRIVER_BACKEND_ID,
    }
  }

  currentValid = !!(review.is_compilable && review.is_correct)
  const roundSpeedup = currentValid
    ? (review.speedup || (review.latency_ms ? baselineLatency / review.latency_ms : (bestSpeedup || 1.0)))
    : null
  currentLatency = review.latency_ms || currentLatency

  // Update best (unconditionally tracks the fastest valid kernel)
  if (currentValid && (bestSpeedup == null || roundSpeedup > bestSpeedup)) {
    bestSpeedup = roundSpeedup
    bestKernelCode = currentKernelCode
    log(`NEW BEST: ${bestSpeedup.toFixed(2)}x`)
  }

  if (!currentValid) {
    // =========================================================================
    // Phase: Repair — Diagnoser -> Repairer, using CHAINED repair memory
    // =========================================================================
    phase('Repair')

    const diagnosis = await agent(`You are the KernelSkill Diagnoser. The current kernel is INVALID. Infer the root cause and propose a repair plan. You are given the CHAINED repair memory for this fault chain — use it to AVOID proposing a fix that was already tried and failed (no cyclic repair).

# Compiler/Verifier error excerpt:
${(review.error_excerpt || 'unknown failure').slice(0, 1500)}

# Current (faulty) kernel:
\`\`\`${fenceToken()}
${currentKernelCode.substring(0, 3500)}
\`\`\`

# Chained repair memory (prior attempts in THIS fault chain):
${buildRepairMemoryBlock(repairMemory)}

Return: root_cause (concise), repair_strategy (a concrete, DIFFERENT plan than any failed one above), and confidence_0_to_1.`, {
      label: `diagnose-${round}`,
      phase: 'Repair',
      schema: {
        type: 'object',
        properties: {
          root_cause: { type: 'string' },
          repair_strategy: { type: 'string' },
          confidence_0_to_1: { type: 'number' },
        },
        required: ['root_cause', 'repair_strategy'],
      },
    })

    const repaired = await agent(`You are the KernelSkill Repairer. Apply this repair plan to fix the kernel. Output a COMPLETE, compilable kernel that matches the PyTorch reference within tolerance (rtol=${RTOL}, atol=${ATOL}).

# Root cause: ${diagnosis.root_cause}
# Repair strategy: ${diagnosis.repair_strategy}

# Faulty kernel:
\`\`\`${fenceToken()}
${currentKernelCode.substring(0, 4000)}
\`\`\`

# Error excerpt:
${(review.error_excerpt || '').slice(0, 1000)}

Keep the forward() signature identical to the reference. Return the complete fixed kernel source.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Repair","ts":"<ts>","status":"done","candidate_id":"round-${round}","technique":"<the repair strategy you applied>","speedup":null,"note":"<root cause + what you changed to fix it>"}`, {
      label: `repair-${round}`,
      phase: 'Repair',
      schema: {
        type: 'object',
        properties: {
          code: { type: 'string' },
          changes_made: { type: 'string' },
        },
        required: ['code'],
      },
    })

    if (repaired && repaired.code) {
      currentKernelCode = repaired.code
    }
    // Record into chained repair memory (fixed status is verified next round's Evaluate)
    repairMemory.push({
      round: round + 1,
      root_cause: diagnosis.root_cause,
      repair_strategy: diagnosis.repair_strategy,
      fixed: null,
      error_excerpt: review.error_excerpt || '',
    })
    speedupTrajectory.push(null)
    log(`Repaired (round ${round + 1}): ${diagnosis.root_cause}`)

  } else {
    // =========================================================================
    // Phase: Optimize — Feature Extractor -> Gate -> Retrieve skill -> Plan -> Apply
    // A valid kernel clears the repair chain.
    // =========================================================================
    repairMemory = []
    phase('Optimize')

    // 1) Feature Extractor (hybrid: rule-based + LLM structural inference)
    const features = await agent(`You are the KernelSkill Feature Extractor. Derive the deterministic code-structure features the gate needs. Use a hybrid approach: rule-based pattern matching over the source for stable lexical/syntactic signatures, and structural inference for features that syntax alone cannot capture.

# Kernel source:
\`\`\`${fenceToken()}
${currentKernelCode.substring(0, 4000)}
\`\`\`

# Feature schema (provide every field):
${skillLibrary.machine_check.code_features_schema.join(', ')}

Definitions:
- has_reuse: exploitable data reuse exists (shared tile / register blocking / repeated reads of same data).
- streaming_no_reuse: pure load->compute->store, each element read once.
- has_vector_load_store: uses or can trivially use float4/half2 vectorized access.
- is_aligned_vector_access: vector accesses are 16B-aligned.
- has_tail_handling_overhead: branchy remainder / masked-lane handling is significant.
- tc_eligible: matmul-shaped compute with TC-compatible dtype (half/bf16) and tileable shapes.
- is_pointwise / uses_transcendentals / is_naive_gemm / has_k_loop / is_gemm_kloop / is_stencil_conv / has_shared_memory_tile / uses_vector_types / has_bounds_check: as named.
- kernel_structure_id: 0=elementwise/streaming(S0), 1=reuse-friendly tiling(S1), 2=irregular gather/scatter(S2), 3=reduction/scan(S3), 4=multi-kernel graph(S4).
- has_multiple_kernels_in_forward: forward launches >1 distinct kernel.

Return ONLY the features object (booleans + the two ints).`, {
      label: `features-${round}`,
      phase: 'Optimize',
      schema: {
        type: 'object',
        properties: {
          has_reuse: { type: 'boolean' },
          streaming_no_reuse: { type: 'boolean' },
          has_vector_load_store: { type: 'boolean' },
          is_aligned_vector_access: { type: 'boolean' },
          has_tail_handling_overhead: { type: 'boolean' },
          tc_eligible: { type: 'boolean' },
          is_pointwise: { type: 'boolean' },
          uses_transcendentals: { type: 'boolean' },
          is_naive_gemm: { type: 'boolean' },
          has_k_loop: { type: 'boolean' },
          is_gemm_kloop: { type: 'boolean' },
          is_stencil_conv: { type: 'boolean' },
          has_shared_memory_tile: { type: 'boolean' },
          uses_vector_types: { type: 'boolean' },
          has_bounds_check: { type: 'boolean' },
          kernel_structure_id: { type: 'number' },
          has_multiple_kernels_in_forward: { type: 'boolean' },
        },
        required: ['kernel_structure_id', 'has_reuse', 'streaming_no_reuse'],
      },
    })

    // 2) Deterministic Gate (machine_check) -> allowed_methods.
    //    Run the decision policy as an explicit, auditable agent over the embedded skill library.
    const gate = await agent(`You are the KernelSkill deterministic GATE (machine_check layer). You DETERMINISTICALLY evaluate the decision policy. You do NOT use judgment about which optimization is "best" — you only apply the rules to compute normalized fields, derived fields, the headroom tier, the matched bottleneck case, and the resulting allowed_methods set.

# Decision policy (long-term memory, machine_check layer):
${asJsonBlock({
  field_mapping_keys: Object.keys(skillLibrary.machine_check.field_mapping),
  derived_fields: skillLibrary.machine_check.derived_fields,
  headroom_tiers: skillLibrary.machine_check.headroom_tiers,
  bottleneck_priority_rules: skillLibrary.machine_check.bottleneck_priority_rules,
  decision_table: skillLibrary.machine_check.decision_table,
}, 6000)}

# Normalized NCU metrics (from Profiler):
${asJsonBlock(review.ncu_metrics || {}, 2000)}

# Code features (from Feature Extractor):
${asJsonBlock(features, 2000)}

# Procedure (apply in order, deterministically):
1. Compute derived_fields from the metrics + features.
2. Apply bottleneck_priority_rules (hard overrides).
3. Determine the headroom tier.
4. Find the FIRST decision_table case whose 'when' predicate is satisfied by the evidence.
5. Emit that case's allowed_methods VERBATIM as the hard-constraint set. If no case matches, use case_id NO_MATCH with an empty allowed_methods (the planner may then self-generate, justified by evidence).
6. Apply mechanism gates: if a reuse-based method (shared_memory_tiling, register_blocking, reduce_bytes_via_reuse) is in allowed_methods but reuse_possible is false, REMOVE it.

Return the gate decision. allowed_methods MUST be a subset of the matched case's list (after mechanism gating).`, {
      label: `gate-${round}`,
      phase: 'Optimize',
      schema: {
        type: 'object',
        properties: {
          tier: { type: 'string' },
          bottleneck_id: { type: 'string' },
          matched_case_id: { type: 'string' },
          allowed_methods: { type: 'array', items: { type: 'string' } },
          key_metrics: { type: 'string' },
          derived_values: { type: 'string' },
        },
        required: ['matched_case_id', 'allowed_methods'],
      },
    })

    const allowed = (gate.allowed_methods || []).filter(Boolean)
    // 3) Retrieve method knowledge (llm_assist) for the allowed methods.
    const methodKnowledge = {}
    for (const m of allowed) {
      if (skillLibrary.llm_assist[m]) methodKnowledge[m] = skillLibrary.llm_assist[m]
    }

    // 4) Planner — pick a method from allowed_methods (HARD CONSTRAINT) and write a stepwise plan,
    //    conditioned on the short-term optimization memory.
    const plan = await agent(`You are the KernelSkill Planner. The deterministic gate has ALREADY decided which optimization methods are FEASIBLE. You MUST select your method from allowed_methods (HARD CONSTRAINT). Then produce a concrete, stepwise optimization plan. Use the short-term optimization memory to avoid repeating an unproductive method and to build coupled multi-step improvements.

# Gate result:
- tier: ${gate.tier || 'Tier-M'}
- bottleneck_id: ${gate.bottleneck_id || 'unknown'}
- matched_case: ${gate.matched_case_id}
- allowed_methods (HARD CONSTRAINT): ${allowed.length ? allowed.join(', ') : '(empty — you MAY self-generate a method justified by the evidence)'}
- key metrics: ${gate.key_metrics || 'n/a'}

# Method knowledge (rationale + implementation cues) for the allowed methods:
${asJsonBlock(methodKnowledge, 4000)}

# Short-term OPTIMIZATION memory (methods tried this task + measured outcomes):
${buildOptimizeMemoryBlock(optimizeMemory, 8)}

# Current performance: ${roundSpeedup != null ? roundSpeedup.toFixed(2) + 'x vs Torch Eager' : 'unknown'}
# Profiler summary: ${(review.profile_summary || '').slice(0, 500)}

# Current kernel:
\`\`\`${fenceToken()}
${currentKernelCode.substring(0, 3500)}
\`\`\`

Rules:
- method_name MUST be exactly one of allowed_methods (unless allowed_methods is empty, in which case justify a self-generated method from the profiling evidence).
- If a method was already tried with no gain (see memory), pick a DIFFERENT allowed method or justify why a refined re-application will help.
- The plan must be complete enough that following it step-by-step fully implements the method.

Return: method_name, rationale (citing the metric evidence), and plan (numbered, concrete edits).`, {
      label: `plan-${round}`,
      phase: 'Optimize',
      schema: {
        type: 'object',
        properties: {
          method_name: { type: 'string' },
          rationale: { type: 'string' },
          plan: { type: 'string' },
        },
        required: ['method_name', 'plan'],
      },
    })

    // 5) Optimizer — apply the plan.
    const optimized = await agent(`You are the KernelSkill Optimizer. Apply the optimization plan to the current kernel and output a COMPLETE, compilable kernel that is still correct vs the PyTorch reference within tolerance (rtol=${RTOL}, atol=${ATOL}).

# Selected method: ${plan.method_name}
# Rationale: ${plan.rationale}
# Plan:
${plan.plan}

# Method implementation cues:
${skillLibrary.llm_assist[plan.method_name] || '(self-generated method — follow the plan)'}

# Current kernel:
\`\`\`${fenceToken()}
${currentKernelCode.substring(0, 4000)}
\`\`\`

Requirements:
1. Apply the plan faithfully — it is grounded in real profiling evidence.
2. Keep the forward() signature identical to the reference.
3. Output the COMPLETE kernel source (all includes, kernels, bindings).
4. Do not regress correctness.

Return the optimized kernel.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (the speedup of this edit is measured next round, so leave it null here):
{"workflow":"${WORKFLOW_NAME}","phase":"Optimize","ts":"<ts>","status":"done","candidate_id":"round-${round}","technique":"${plan.method_name}","speedup":null,"note":"<what you changed to apply ${plan.method_name} + the metric evidence it targets>"}`, {
      label: `optimize-${round}`,
      phase: 'Optimize',
      schema: {
        type: 'object',
        properties: {
          code: { type: 'string' },
          implementation_notes: { type: 'string' },
        },
        required: ['code'],
      },
    })

    const speedupBefore = roundSpeedup
    if (optimized && optimized.code) {
      currentKernelCode = optimized.code
    }
    // Record into short-term optimization memory (speedup_after verified next round).
    optimizeMemory.push({
      round: round + 1,
      method: plan.method_name,
      evidence: gate.key_metrics || gate.matched_case_id,
      speedup_before: speedupBefore,
      speedup_after: null,
      outcome: 'pending',
    })
    skillsApplied.push(plan.method_name)
    speedupTrajectory.push(roundSpeedup)
    log(`Optimize (round ${round + 1}): case=${gate.matched_case_id} -> method=${plan.method_name}`)
  }

  // ===========================================================================
  // Phase: Iterate — backfill memory outcomes from the latest measurement
  // ===========================================================================
  phase('Iterate')
  // Backfill the previous optimize-memory entry's measured result once we have a new speedup.
  if (optimizeMemory.length > 0 && currentValid) {
    const last = optimizeMemory[optimizeMemory.length - 1]
    if (last.speedup_after == null && last.outcome === 'pending') {
      // roundSpeedup reflects the kernel measured at the start of THIS round, i.e. the
      // result of the PREVIOUS round's edit — attribute it to the prior memory entry.
      // (First fill happens on the round after the edit.)
    }
  }
}

// =============================================================================
// Final report
// =============================================================================
const finalReport = await agent(`Write a concise technical report for this KernelSkill optimization run.

# KernelSkill Results
- Operation: ${OP_DESC} (${opType})
- Reference: ${REFERENCE_PATH}
- Target GPU: ${TARGET_GPU}
- Run timestamp: ${RUN_TS}
- Torch Eager baseline: ${baselineLatency}ms
- Best speedup vs Torch Eager: ${bestSpeedup != null ? bestSpeedup.toFixed(2) + 'x' : 'no valid kernel'}
- Rounds: ${ROUNDS}
- Skills applied (in order): ${skillsApplied.join(' -> ') || 'none'}
- Speedup trajectory (per round, null=invalid): ${JSON.stringify(speedupTrajectory.map(s => s == null ? null : Number(s.toFixed(3))))}

# Short-term optimization memory:
${buildOptimizeMemoryBlock(optimizeMemory, 50)}

# Best kernel:
\`\`\`${fenceToken()}
${(bestKernelCode || currentKernelCode || '').substring(0, 3000)}
\`\`\`

Write:
1. Dual-memory narrative: which long-term skills (methods) were selected, what profiling evidence the gate matched, and how short-term memory steered plan revision.
2. The speedup trajectory and which method delivered the biggest gain.
3. Repair episodes (if any) and how chained repair memory avoided cyclic fixes.
4. Remaining bottleneck for the best kernel and the next method to try.`, {
  label: 'final-report',
  phase: 'Report',
})

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: REFERENCE_PATH,
  generated_kernel_path: bestKernelCode ? `${EXP_DIR}/best_kernel.cu` : '',
  initial_candidates: [],
  initial_generation_result: {
    verified: bestSpeedup != null,
    selected_candidate_id: bestKernelCode ? 'best' : '',
  },
  baseline_latency_ms: baselineLatency,
  best_speedup: bestSpeedup,
  best_kernel_code: bestKernelCode || currentKernelCode,
  rounds_completed: ROUNDS,
  skills_applied: skillsApplied,
  speedup_trajectory: speedupTrajectory,
  optimize_memory: optimizeMemory,
  repair_attempts: optimizeMemory.length === 0 ? 0 : undefined,
  report: finalReport,
}
