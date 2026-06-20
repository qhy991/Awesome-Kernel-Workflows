export const meta = {
  name: 'keet-kernel-explanation',
  description: 'LLM-agent pipeline for interpreting NCU profiles into actionable kernel performance explanations (KEET methodology)',
  whenToUse: 'When you need to understand WHY a CUDA kernel performs the way it does. Produces a natural language explanation report from NCU profiles + source code, identifying bottlenecks and suggesting specific code changes. Use as a standalone analysis tool, or as a pre-step to feed optimization context into downstream tasks.',
  phases: [
    { title: 'Setup', detail: 'Collect NCU profiles, kernel source, and run configuration metadata' },
    { title: 'Source Inspection', detail: 'Iteratively analyze source files: describe, summarize algorithm, generate performance hypotheses' },
    { title: 'Profile Inspection', detail: 'Iteratively analyze NCU profiles: select metrics, produce grounded performance analysis' },
    { title: 'Aggregation', detail: 'Combine all analyses into a unified performance explanation report' },
    { title: 'Review', detail: 'Cross-check explanation against pre-data hypotheses, confirm/refute/inconclusive' },
  ],
}

// --- BEGIN model-tier (auto-inserted by scripts/patch-model-tier.js) ---
// Tier-based model routing: mechanical steps (run substrate scripts, parse
// JSON) use cheaper models; profile steps (run eval/ncu) use mid-tier;
// judgment steps (plan/implement/report) use the top tier. Tuneable via
// args.model_{mechanical,profile,judgment}.
const MODEL = {
  mechanical: (typeof args !== 'undefined' && args && args.model_mechanical) || 'haiku',
  profile: (typeof args !== 'undefined' && args && args.model_profile) || 'sonnet',
  judgment: (typeof args !== 'undefined' && args && args.model_judgment) || 'opus',
}
// __modelTierApplied
// --- END model-tier ---

const WORKFLOW_NAME = 'keet-kernel-explanation'


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
  supported_problem_types: ['performance-explanation'],
  problem_types: ['NCU profile explanation', 'CUDA source performance diagnosis'],
  reason: 'KEET in this repo interprets CUDA source and Nsight Compute profile artifacts; it is an explanation workflow, not a generic optimizer.',
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

assertWorkflowSuitability()

// =============================================================================
// KEET: Kernel Execution Explanation Toolkit (Workflow Implementation)
// =============================================================================
//
// Source: "KEET: Explaining Performance of GPU Kernels Using LLM Agents"
//         Davis et al., arXiv:2605.04467, 2026
//         University of Maryland / NVIDIA / Lawrence Livermore National Lab
//
// KEET decomposes NCU profile interpretation into specialized agent roles:
//   Source Code Inspection Stage:
//     - Source File Describer: describe each source file
//     - Algorithm Summarizer: summarize the algorithm from code
//     - Performance Hypothesizer: predict performance behavior from code alone
//   Profile Inspection Stage:
//     - Profile Selector: choose which profile(s) to analyze next
//     - Metric Selector: choose which metrics to focus on
//     - Profile Analyzer: produce grounded performance analysis
//     - DrGPU Evaluator: (optional) integrate DrGPU suggestions
//   Aggregation + Review:
//     - Analysis Aggregator: combine all analyses into final report
//     - Explanation Reviewer: confirm/refute hypotheses against data
//
// Key insight: hypothesis-first analysis — generate performance predictions
// from code BEFORE seeing data, then confirm/refute with measured metrics.
// This prevents confirmation bias and produces interpretable explanations.
//
// Usage:
//   Workflow({name: 'keet-kernel-explanation', args: {
//     kernel_path: '/path/to/kernel.cu',
//     ncu_report_path: '/path/to/report.ncu-rep',
//     op_description: 'Fan2 kernel from gaussian application',
//     source_paths: ['/path/to/main.cu', '/path/to/utils.cuh'],
//     ncu_binary: '<user-provided ncu binary path>',
//     exp_dir: '/tmp/keet_exp',
//     kernel_name_regex: 'kernel_name',
//     run_metadata: 'Block size: 128, Grid: (256,1,1), Registers: 64',
//     include_drgpu: false,
//     analysis_guidelines_path: '',
//     additional_profiles: [],
//   }})
//
// =============================================================================

// --- Required Args ---
const KERNEL_PATH = args.kernel_path
const NCU_REPORT_PATH = args.ncu_report_path || ''

// --- Optional Args ---
const OP_DESC = args.op_description || 'CUDA kernel'
const SOURCE_PATHS = args.source_paths || []
const NCU_BINARY = args.ncu_binary || ''
const EXP_DIR = args.exp_dir || '/tmp/keet_exp'
const KERNEL_NAME_REGEX = args.kernel_name_regex || ''
const RUN_METADATA = args.run_metadata || ''
const INCLUDE_DRGPU = args.include_drgpu || false
const ANALYSIS_GUIDELINES_PATH = args.analysis_guidelines_path || ''
const ADDITIONAL_PROFILES = args.additional_profiles || []
const HARNESS_BUILD_CMD = args.harness_build_cmd || ''
const HARNESS_RUN_ARGS = args.harness_run_args || ''

// State
let algorithmSummary = ''
let performanceHypotheses = []
let profileAnalyses = []
let sourceDescriptions = []

// Performance Analysis Guidelines (KEET Section III-E)
const DEFAULT_GUIDELINES = `# GPU Performance Analysis Guidelines

When analyzing kernel performance, consider:

1. **Memory Hierarchy**: Global memory bandwidth utilization, L1/L2 cache effectiveness,
   shared memory bank conflicts, register pressure and spilling.

2. **Compute Utilization**: SM occupancy, warp execution efficiency, instruction mix
   (FP32/FP64/INT/SFU), pipeline utilization.

3. **Latency Sources**: Memory latency (long_scoreboard stalls), instruction dependencies
   (short_scoreboard), synchronization (barrier stalls), scheduling.

4. **Parallelism**: Grid/block sizing, tail effects, wave quantization, thread divergence,
   warp scheduling efficiency.

5. **Architecture-Specific**: Tensor core usage, async copy pipeline, L2 residency controls,
   shared memory configuration (carveout vs L1).

6. **Tuning Knobs**: Relationship between block size, register usage, shared memory allocation,
   and achieved occupancy. How changing one affects others.

Always cite specific NCU metric values when making claims. Distinguish between:
- Bandwidth-bound (high DRAM throughput, low compute)
- Latency-bound (low throughput, high stall cycles, long_scoreboard)
- Compute-bound (high SM throughput, math_pipe_throttle stalls)`

// =============================================================================
// Phase 1: Setup — Collect NCU profiles, source code, metadata
// =============================================================================
phase('Setup')

// --- profiling-strategist: decide whether real NCU metrics are actually
// available on this host BEFORE we inspect a profile. KEET is a diagnostic
// (explanation) role with NO optimization loop; the strategist's value here is
// HONESTY — if no native profiler is available, KEET must NOT pretend it has NCU
// counters. The agent only classifies the op (fuzzy op_class/size from
// OP_DESC + source); the substrate strategist DETERMINISTICALLY picks the method
// and STAMPS confidence (measured/inferred/hypothesized) -- the model must NOT
// assign confidence. Defaults to native_profiler so the happy path is unchanged
// when the call returns nothing. See _substrate/profiling/README.md. ---
const SUBSTRATE_DIR = args.substrate_dir || '_substrate'
const BACKEND_MANIFEST = args.backend_manifest || `${SUBSTRATE_DIR}/backends/cuda/manifest.json`
const PROF_STRATEGIST_SCHEMA = { type: 'object', additionalProperties: true }
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured' }
const _pd = await agent(
  `You classify a CUDA op for profiler-method selection. From the operation description "${OP_DESC}" and the kernel at ${KERNEL_PATH}, ` +
  `classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (one of tiny|small|large). Then ` +
  `run exactly: \`${SUBSTRATE_DIR}/profiling/profiling_strategist.py resolve --backend-manifest ${BACKEND_MANIFEST} --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`.\n` +
  `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}. ` +
  `Do NOT assign confidence yourself — only the strategist stamps it.`,
  { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: PROF_STRATEGIST_SCHEMA })
if (_pd && _pd.method) PROFILING_DECISION = _pd

const PROFILE_METHOD_GUARD =
  `\nProfiling-strategist selected method='${PROFILING_DECISION.method}', confidence='${PROFILING_DECISION.confidence}'. ` +
  `If method !== 'native_profiler', NCU metrics are NOT available — base your explanation only on source + any throughput/perf evidence present, ` +
  `explicitly label every profile-derived claim as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}', and do NOT fabricate NCU counter values. ` +
  `If method === 'native_profiler', proceed as written.`

const setupResults = await parallel([
  // Agent 1: Read and catalog all source files
  () => agent(`You are a CUDA source code reader. Read and catalog kernel source files.

# Primary kernel file: ${KERNEL_PATH}
# Additional source files: ${JSON.stringify(SOURCE_PATHS)}

# Task:
1. Read the primary kernel file at ${KERNEL_PATH}
2. If additional source files are listed, read those too
3. For each file, extract:
   - filename and path
   - key functions (especially __global__ and __device__)
   - data structures defined
   - includes and dependencies
   - approximate lines of code

Return a catalog of all source files with their contents and metadata.`, {
    label: 'read-sources',
    phase: 'Setup',
    schema: {
      type: 'object',
      properties: {
        primary_source: { type: 'string' },
        primary_filename: { type: 'string' },
        additional_sources: { type: 'array', items: { type: 'object', properties: { path: { type: 'string' }, content: { type: 'string' } } } },
        kernel_functions: { type: 'array', items: { type: 'string' } },
        total_sloc: { type: 'number' },
      },
      required: ['primary_source', 'kernel_functions'],
    },
  }),

  // Agent 2: Extract NCU profile data
  () => agent(`You are an NCU profiling expert. Extract performance data from NCU profile(s).

# NCU Report: ${NCU_REPORT_PATH || '(need to generate)'}
# NCU Binary: ${NCU_BINARY || '(not provided)'}
# Kernel name regex: ${KERNEL_NAME_REGEX}
# Experiment directory: ${EXP_DIR}
# Run metadata: ${RUN_METADATA}
# Additional profiles to analyze: ${JSON.stringify(ADDITIONAL_PROFILES)}

# Task:
If an NCU report path is provided and ncu_binary is provided, extract metrics from it using that user-provided binary. If ncu_binary is missing, do not invent an import command; report that binary report extraction is unavailable.

If no report is provided, generate one only when harness_build_cmd, harness_run_args/run metadata, and ncu_binary are explicitly provided by the user. Do not create a default harness or choose a default profiler command.

Extract ALL available metrics organized by category:
- Duration & throughput (gpu__time_duration, sm__throughput, dram throughput)
- Occupancy (achieved, theoretical, limiting factors)
- Memory (sectors/request, L1/L2 hit rates, shared memory usage)
- Stalls (top stall reasons with percentages)
- Compute (instruction mix, warp execution efficiency)
- Launch config (grid, block, registers, shared mem)
- NCU rule suggestions with estimated speedups

Also extract metrics for any additional profiles listed.

Return structured profile data.
${PROFILE_METHOD_GUARD}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"ncu_profile_extraction","speedup":null,"note":"<profiles found + key metrics extracted, one line>"}`, {
    label: 'extract-profiles',
    phase: 'Setup',
    schema: {
      type: 'object',
      properties: {
        primary_profile: {
          type: 'object',
          properties: {
            latency_us: { type: 'number' },
            sm_throughput_pct: { type: 'number' },
            dram_throughput_pct: { type: 'number' },
            achieved_occupancy_pct: { type: 'number' },
            theoretical_occupancy_pct: { type: 'number' },
            registers_per_thread: { type: 'number' },
            shared_mem_bytes: { type: 'number' },
            block_size: { type: 'number' },
            grid_size: { type: 'string' },
            top_stalls: { type: 'array', items: { type: 'object', properties: { reason: { type: 'string' }, pct: { type: 'number' } } } },
            sectors_per_request_ld: { type: 'number' },
            l1_hit_rate_pct: { type: 'number' },
            l2_hit_rate_pct: { type: 'number' },
            ncu_suggestions: { type: 'array', items: { type: 'string' } },
            full_metrics_text: { type: 'string' },
          },
          required: ['full_metrics_text'],
        },
        additional_profiles: { type: 'array', items: { type: 'object', properties: { label: { type: 'string' }, metrics_text: { type: 'string' } } } },
        profile_count: { type: 'number' },
      },
      required: ['primary_profile', 'profile_count'],
    },
  }),
])

const sourceData = setupResults[0]
const profileData = setupResults[1]

log(`Setup: ${sourceData?.kernel_functions?.length || 0} kernel functions, ${profileData?.profile_count || 1} profile(s), ${sourceData?.total_sloc || '?'} SLoC`)

// =============================================================================
// Phase 2: Source Code Inspection Stage
// =============================================================================
phase('Source Inspection')

// KEET's Source Code Inspection: iteratively review each source file,
// build algorithm summary, generate performance hypotheses BEFORE seeing data
const sourceInspection = await agent(`You are a GPU performance expert analyzing CUDA kernel source code.
Your job is to understand the algorithm and predict performance characteristics
WITHOUT looking at any profiling data. This is the "hypothesis-first" approach from KEET.

# Operation Description: ${OP_DESC}
# Run Configuration: ${RUN_METADATA}

# Kernel Source Code:
\`\`\`cuda
${sourceData?.primary_source?.substring(0, 8000) || 'No source available'}
\`\`\`

${(sourceData?.additional_sources || []).map(s => `# Additional Source: ${s.path}\n\`\`\`cuda\n${(s.content || '').substring(0, 3000)}\n\`\`\``).join('\n\n')}

# Your Tasks (KEET Source Code Inspection roles):

## 1. Source File Description
For each source file, describe:
- What functions are defined
- What data structures are used
- Control flow patterns (loops, conditionals, divergent paths)
- Memory access patterns visible in the code

## 2. Algorithm Summary
Provide a concise summary of:
- What computation the kernel performs
- The parallelization strategy (how work is mapped to threads/blocks)
- Data dependencies and synchronization points
- Memory access patterns (coalesced?, strided?, random?)

## 3. Performance Hypotheses (CRITICAL — predict BEFORE seeing data)
Generate 3-6 specific, testable hypotheses about how this kernel should perform:
- What hardware resource is likely the bottleneck? (memory bandwidth, compute, latency)
- What stall reason should dominate? (long_scoreboard, barrier, math_pipe_throttle)
- Is memory access likely coalesced? (based on indexing patterns)
- Is occupancy likely limited? By what? (registers, shared memory, block size)
- Are there likely warp divergence issues? (conditional branches)
- Is there likely shared memory bank conflict?

Each hypothesis should be:
- Specific enough to be confirmed or refuted by NCU data
- Grounded in specific code patterns you can cite
- Expressed as: "I predict X because of code pattern Y"

This is crucial for KEET's interpretability: we generate predictions from code alone,
then verify against hardware measurements.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Source Inspection","ts":"<ts>","status":"done","technique":"hypothesis_first_source_analysis","speedup":null,"note":"<algorithm + number of perf hypotheses generated, one line>"}`, {
  label: 'source-inspection',
  phase: 'Source Inspection',
  schema: {
    type: 'object',
    properties: {
      file_descriptions: { type: 'array', items: { type: 'object', properties: { filename: { type: 'string' }, description: { type: 'string' } } } },
      algorithm_summary: { type: 'string' },
      parallelization_strategy: { type: 'string' },
      memory_access_patterns: { type: 'string' },
      performance_hypotheses: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            hypothesis: { type: 'string' },
            code_evidence: { type: 'string' },
            predicted_metric: { type: 'string' },
            confidence: { type: 'string' },
          },
          required: ['hypothesis', 'code_evidence'],
        },
      },
    },
    required: ['algorithm_summary', 'performance_hypotheses'],
  },
})

algorithmSummary = sourceInspection.algorithm_summary
performanceHypotheses = sourceInspection.performance_hypotheses || []
sourceDescriptions = sourceInspection.file_descriptions || []

log(`Source Inspection: ${performanceHypotheses.length} hypotheses generated | Strategy: ${sourceInspection.parallelization_strategy?.substring(0, 60)}...`)

// =============================================================================
// Phase 3: Profile Inspection Stage
// =============================================================================
phase('Profile Inspection')

// KEET's Profile Inspection: Metric Selector → Profile Analyzer (iterative per profile)
// For single profile: analyze directly with hypothesis-informed metric selection
// For multiple profiles: use Profile Selector to prioritize which to analyze first

const profilesToAnalyze = [
  { label: 'primary', metrics: profileData?.primary_profile?.full_metrics_text || '' },
  ...(profileData?.additional_profiles || []).map(p => ({ label: p.label, metrics: p.metrics_text })),
].filter(p => p.metrics)

const guidelines = ANALYSIS_GUIDELINES_PATH
  ? `(Read guidelines from: ${ANALYSIS_GUIDELINES_PATH})`
  : DEFAULT_GUIDELINES

// Analyze each profile — metric selection + grounded analysis
const profileAnalysisResults = await parallel(
  profilesToAnalyze.map((profile, profIdx) => () =>
    agent(`You are the KEET Profile Analyzer — an expert GPU performance analyst.
You produce data-grounded natural language explanations of kernel performance.

# Role: Profile Analyzer (KEET Section III-C)
You examine profile metrics, algorithm summary, source code, and analysis guidelines
to produce a detailed performance analysis.

# Algorithm Summary:
${algorithmSummary}

# Performance Hypotheses (generated from code BEFORE seeing this data):
${performanceHypotheses.map((h, i) => `${i + 1}. ${h.hypothesis} [evidence: ${h.code_evidence}]`).join('\n')}

# Profile Data (${profile.label}):
${profile.metrics.substring(0, 6000)}

# Key Metrics Summary:
${profileData?.primary_profile?.latency_us ? `- Latency: ${profileData.primary_profile.latency_us} us` : ''}
${profileData?.primary_profile?.sm_throughput_pct ? `- SM Throughput: ${profileData.primary_profile.sm_throughput_pct}%` : ''}
${profileData?.primary_profile?.dram_throughput_pct ? `- DRAM Throughput: ${profileData.primary_profile.dram_throughput_pct}%` : ''}
${profileData?.primary_profile?.achieved_occupancy_pct ? `- Achieved Occupancy: ${profileData.primary_profile.achieved_occupancy_pct}%` : ''}
${profileData?.primary_profile?.top_stalls ? `- Top Stalls: ${profileData.primary_profile.top_stalls.map(s => `${s.reason}(${s.pct}%)`).join(', ')}` : ''}
${profileData?.primary_profile?.sectors_per_request_ld ? `- Sectors/Request (LD): ${profileData.primary_profile.sectors_per_request_ld}` : ''}

# Source Code (for reference):
\`\`\`cuda
${sourceData?.primary_source?.substring(0, 4000) || ''}
\`\`\`

${guidelines}

# Analysis Requirements (KEET Section III-C):

## 1. Metric Selection (choose the MOST RELEVANT metrics)
Select 5-10 metrics most relevant to understanding this kernel's performance.
Prioritize metrics that can confirm or refute the performance hypotheses.
Reduce noise by skipping metrics that are clearly irrelevant.

## 2. Performance Analysis
For each bottleneck identified:
- CITE the specific NCU metric value(s) supporting your claim
- EXPLAIN how it relates to the algorithm and code structure
- CONNECT it to a specific code region when possible
- Distinguish between PRIMARY bottleneck and secondary effects

## 3. Tuning Knob Analysis
If multiple profiles are available (different configs), explain:
- How does block size affect occupancy and performance?
- How does register usage interact with occupancy?
- What is the optimal configuration and why?

## 4. Optimization Suggestions
For each bottleneck, suggest specific code changes:
- What line(s) to change
- What the change should be
- Expected impact (cite NCU metric that should improve)

## 5. Inline Citations
Reference specific metric values inline: e.g., "The kernel is memory-latency
bound (long_scoreboard = 45.2% of stall samples, dram__throughput = 12.3%)"

CRITICAL: Every claim must be supported by a cited metric value from the profile data.
Do not hallucinate metric values. If a metric is not available, say so.
${PROFILE_METHOD_GUARD}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Profile Inspection","ts":"<ts>","status":"done","technique":"metric_grounded_profile_analysis","speedup":null,"note":"<primary bottleneck + the cited metric, one line>"}`, { model: MODEL.profile,
      label: `profile-analysis-${profile.label}`,
      phase: 'Profile Inspection',
      schema: {
        type: 'object',
        properties: {
          profile_label: { type: 'string' },
          selected_metrics: { type: 'array', items: { type: 'string' } },
          primary_bottleneck: { type: 'string' },
          bottleneck_category: { type: 'string' },
          bottleneck_evidence: { type: 'string' },
          secondary_issues: { type: 'array', items: { type: 'string' } },
          tuning_analysis: { type: 'string' },
          optimization_suggestions: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                issue: { type: 'string' },
                suggestion: { type: 'string' },
                expected_impact: { type: 'string' },
                code_location: { type: 'string' },
              },
              required: ['issue', 'suggestion'],
            },
          },
          full_analysis: { type: 'string' },
        },
        required: ['primary_bottleneck', 'optimization_suggestions', 'full_analysis'],
      },
    })
  )
)

profileAnalyses = profileAnalysisResults.filter(Boolean)
log(`Profile Inspection: ${profileAnalyses.length} profile(s) analyzed | Primary bottleneck: ${profileAnalyses[0]?.primary_bottleneck || 'unknown'}`)

// Optional: DrGPU integration (KEET Section III-C)
let drgpuAnalysis = null
if (INCLUDE_DRGPU) {
  drgpuAnalysis = await agent(`You are the DrGPU Evaluator agent (KEET Section III-C).
DrGPU is a rule-based tool that decomposes stall reasons into a tree structure.
Simulate DrGPU's analysis approach:

1. Decompose the top stall reasons into categories
2. For each category, identify the most frequent leaf-node stall reason
3. Generate rule-based suggestions to address each leaf node
4. Evaluate which DrGPU suggestions are useful given the algorithm context

# Profile Data:
${profileData?.primary_profile?.full_metrics_text?.substring(0, 3000) || ''}

# Top Stalls:
${profileData?.primary_profile?.top_stalls ? JSON.stringify(profileData.primary_profile.top_stalls) : 'N/A'}

# Algorithm Context:
${algorithmSummary}

Generate DrGPU-style suggestions and evaluate their applicability.`, {
    label: 'drgpu-evaluator',
    phase: 'Profile Inspection',
    schema: {
      type: 'object',
      properties: {
        stall_tree: { type: 'string' },
        suggestions: { type: 'array', items: { type: 'object', properties: { suggestion: { type: 'string' }, applicable: { type: 'boolean' }, reason: { type: 'string' } } } },
        useful_suggestions: { type: 'array', items: { type: 'string' } },
      },
      required: ['suggestions'],
    },
  })
  log(`DrGPU: ${drgpuAnalysis?.useful_suggestions?.length || 0} useful suggestions`)
}

// =============================================================================
// Phase 4: Aggregation — Combine all analyses into final report
// =============================================================================
phase('Aggregation')

const aggregatedReport = await agent(`You are the KEET Analysis Aggregator (Section III-D).
Combine all performance analyses into a single, coherent performance explanation report.

# Operation: ${OP_DESC}
# Run Configuration: ${RUN_METADATA}

# Algorithm Summary:
${algorithmSummary}

# Source Structure:
${sourceDescriptions.map(d => `- ${d.filename}: ${d.description}`).join('\n')}

# Performance Analyses (from Profile Inspection):
${profileAnalyses.map(a => `## Profile: ${a.profile_label || 'primary'}\n${a.full_analysis}`).join('\n\n---\n\n')}

${drgpuAnalysis ? `# DrGPU Suggestions (rule-based):\n${(drgpuAnalysis.useful_suggestions || []).map(s => `- ${s}`).join('\n')}` : ''}

# Aggregation Rules (KEET Section III-D):
1. Combine analyses from all profiles into a UNIFIED narrative
2. Identify consistent patterns across profiles (if multiple)
3. Resolve contradictions between different analyses
4. Structure the report as:
   - Executive Summary (1-2 sentences: what is the kernel, what limits it)
   - Summary of Main Bottlenecks (numbered list with metric citations)
   - Detailed Analysis (one section per bottleneck)
   - Relationship to Tuning Knobs (how config parameters affect performance)
   - Optimization Suggestions (prioritized, with expected impact)
5. Maintain inline metric citations throughout
6. Keep language precise and actionable — this report may be used as context
   for an LLM performing code optimization downstream

Generate the final performance explanation report.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Aggregation","ts":"<ts>","status":"done","technique":"analysis_aggregation","speedup":null,"note":"<number of bottlenecks unified + headline finding, one line>"}`, {
  label: 'aggregate-report',
  phase: 'Aggregation',
  schema: {
    type: 'object',
    properties: {
      executive_summary: { type: 'string' },
      bottleneck_list: { type: 'array', items: { type: 'string' } },
      detailed_analysis: { type: 'string' },
      tuning_analysis: { type: 'string' },
      optimization_suggestions: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            priority: { type: 'number' },
            suggestion: { type: 'string' },
            expected_speedup: { type: 'string' },
            code_change: { type: 'string' },
          },
          required: ['suggestion'],
        },
      },
      full_report: { type: 'string' },
    },
    required: ['executive_summary', 'bottleneck_list', 'full_report'],
  },
})

log(`Aggregation: ${aggregatedReport.bottleneck_list?.length || 0} bottlenecks, ${aggregatedReport.optimization_suggestions?.length || 0} suggestions`)

// =============================================================================
// Phase 5: Review — Cross-check explanation against hypotheses
// =============================================================================
phase('Review')

const reviewResult = await agent(`You are the KEET Explanation Reviewer (Section III-D).
Your job is to cross-check the final performance explanation against the
performance hypotheses that were generated BEFORE seeing any profiling data.

# Original Performance Hypotheses (from code analysis alone):
${performanceHypotheses.map((h, i) => `${i + 1}. HYPOTHESIS: ${h.hypothesis}\n   CODE EVIDENCE: ${h.code_evidence}\n   PREDICTED METRIC: ${h.predicted_metric || 'N/A'}`).join('\n\n')}

# Final Performance Explanation:
${aggregatedReport.full_report?.substring(0, 5000) || ''}

# Profile Data (ground truth):
${profileData?.primary_profile?.full_metrics_text?.substring(0, 3000) || ''}

# Review Tasks:
For each hypothesis, determine:
1. **CONFIRMED**: The profiling data supports this hypothesis. Cite the metric.
2. **REFUTED**: The profiling data contradicts this hypothesis. Explain why.
3. **INCONCLUSIVE**: Not enough data to confirm or refute. What additional data needed?

Then assess the overall explanation:
- Does it accurately reflect the measured data?
- Are there any claims not supported by cited metrics?
- Are there important bottlenecks in the data that the explanation missed?
- Is the explanation actionable for a developer?

This review provides interpretability — readers can see which code-based
predictions were validated by hardware measurements and which were not.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Review","ts":"<ts>","status":"done","technique":"hypothesis_confirm_refute","speedup":null,"note":"<confirmed/refuted/inconclusive counts + overall quality, one line>"}`, {
  label: 'explanation-review',
  phase: 'Review',
  schema: {
    type: 'object',
    properties: {
      hypothesis_verdicts: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            hypothesis: { type: 'string' },
            verdict: { type: 'string' },
            evidence: { type: 'string' },
          },
          required: ['hypothesis', 'verdict', 'evidence'],
        },
      },
      confirmed_count: { type: 'number' },
      refuted_count: { type: 'number' },
      inconclusive_count: { type: 'number' },
      missed_bottlenecks: { type: 'array', items: { type: 'string' } },
      unsupported_claims: { type: 'array', items: { type: 'string' } },
      overall_quality: { type: 'string' },
      review_summary: { type: 'string' },
    },
    required: ['hypothesis_verdicts', 'overall_quality', 'review_summary'],
  },
})

log(`Review: ${reviewResult.confirmed_count || 0} confirmed, ${reviewResult.refuted_count || 0} refuted, ${reviewResult.inconclusive_count || 0} inconclusive | Quality: ${reviewResult.overall_quality}`)

// =============================================================================
// Return — Final KEET output
// =============================================================================
return {
  operation: OP_DESC,
  algorithm_summary: algorithmSummary,
  performance_hypotheses: performanceHypotheses,
  hypothesis_verdicts: reviewResult.hypothesis_verdicts,
  primary_bottleneck: profileAnalyses[0]?.primary_bottleneck || 'unknown',
  bottleneck_category: profileAnalyses[0]?.bottleneck_category || 'unknown',
  bottleneck_list: aggregatedReport.bottleneck_list,
  optimization_suggestions: aggregatedReport.optimization_suggestions,
  full_report: aggregatedReport.full_report,
  review_summary: reviewResult.review_summary,
  review_quality: reviewResult.overall_quality,
  drgpu_suggestions: drgpuAnalysis?.useful_suggestions || null,
  profiles_analyzed: profilesToAnalyze.length,
  source_sloc: sourceData?.total_sloc || 0,
}
