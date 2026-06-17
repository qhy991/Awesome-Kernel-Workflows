export const meta = {
  name: 'fact-kernel-optimization',
  description: 'Compositional kernel synthesis with pattern discovery and realization',
  whenToUse: 'Use for generating optimized CUTLASS kernels through compositional pattern synthesis',
  phases: [
    { title: 'Setup', detail: 'Initialize CUTLASS environment and pattern registry' },
    { title: 'Pattern Discovery', detail: 'Discover optimization patterns from exemplars' },
    { title: 'Pattern Realization', detail: 'Realize patterns as code transformations' },
    { title: 'Pattern Composition', detail: 'Compose patterns into optimized kernels' },
    { title: 'Ablation', detail: 'Ablation studies to validate pattern contributions' },
    { title: 'Evaluation', detail: 'Evaluate composed kernels on target hardware' },
    { title: 'Report', detail: 'Generate synthesis report' },
  ],
};

// --- BEGIN genome-report (auto-inserted by scripts/patch-genome-report.js) ---
// Self-reported, work-plane (forgeable) stage trace for observability + the
// recombiner. NOT a trust anchor — see _meta/genome-trajectory-schema.md.
async function __genomeReport(phaseName, wfName) {
  try {
    const __dir = (typeof args !== 'undefined' && args && args.exp_dir) ? args.exp_dir : '.'
    await agent(
      'Append exactly one line to ' + __dir + '/genome.jsonl (create it if missing; use a shell append: printf %s\\n ... >> file). ' +
      'The line must be this JSON on ONE line: {"workflow":"' + wfName + '","phase":"' + phaseName + '","ts":"<UTC>","status":"entered"}. ' +
      'Produce <UTC> by running: date -u +%Y-%m-%dT%H:%M:%SZ . Do nothing else; modify no other file. Echo the exact line you appended.',
      { label: 'genome:' + phaseName, phase: phaseName }
    )
  } catch (__e) { /* observability must never break the workflow */ }
}
// --- END genome-report ---

// --- BEGIN embedded-eval substrate (auto-inlined by scripts/patch-embedded-eval.js) ---
const EMBEDDING_CONTRACT = [
  'EMBEDDED-DISPATCH CONTRACT (this kernel is NOT standalone):',
  '',
  'You are authoring a kernel that lives INSIDE a larger project and is wired into',
  'its dispatch table. It cannot be compiled on its own. Therefore:',
  '',
  '1. Emit a COMPLETE source file (e.g. a .cuh) that matches the reference',
  '   dispatch signature exactly -- same entry-point shape, template params, and',
  '   launch-bounds conventions as the reference file. Do NOT add a main(), a',
  '   standalone harness, or top-level test code.',
  '2. Use ONLY symbols/headers the project already provides (project headers,',
  '   template instantiations, dispatch macros). Do not invent include paths.',
  '3. Do NOT register, build, or benchmark the variant yourself, and do NOT name',
  '   any symbol with the variant suffix -- the workflow + adapter handle wiring.',
  '4. Return ONLY the file contents plus a short rationale citing the concrete',
  '   design choice (tile shape, register budget, pipelining, GQA packing, etc.).',
].join('\n')

// Build the ordered evaluation commands for one candidate against a
// contract-conforming adapter. All fields are plain strings the caller already
// resolved from `args`. `params`/`unregParams` are opaque pass-through strings
// (e.g. "--dkq 256 --dv 256 --cmake-build-dir /p/build") that the substrate does
// not parse -- they belong to the project's adapter.
function __embeddedEvalPlan(ctx) {
  const adapter = ctx.adapter                       // e.g. 'python "/abs/llamacpp_register_variant.py"'
  const variant = ctx.variant                       // unique variant name for this candidate
  const source = ctx.source                         // path to the candidate source file on disk
  const root = ctx.projectRoot                       // --project-root
  const params = ctx.params || ''                    // opaque register params pass-through
  const unregParams = ctx.unregParams || ''          // opaque unregister params pass-through
  const q = (s) => `"${s}"`
  const reg = `${adapter} register --variant ${variant} --source ${q(source)} --project-root ${q(root)}${params ? ' ' + params : ''}`.trim()
  const unreg = `${adapter} unregister --variant ${variant} --project-root ${q(root)}${unregParams ? ' ' + unregParams : ''}`.trim()
  const list = `${adapter} list --project-root ${q(root)}`
  return {
    register: reg,
    list,
    // Project-native build/test/benchmark, run VERBATIM with the variant's env
    // gate set so the project binary dispatches to this candidate.
    build: ctx.buildCmd ? `KERSOR_VARIANT=${variant} ${ctx.buildCmd}` : '',
    test: ctx.testCmd ? `KERSOR_VARIANT=${variant} ${ctx.testCmd}` : '',
    benchmark: ctx.benchmarkCmd ? `KERSOR_VARIANT=${variant} ${ctx.benchmarkCmd}` : '',
    unregister: unreg,
    // Human-orderable sequence + the non-negotiable cleanup invariant.
    order: ['register', 'list', 'build', 'test', 'benchmark', 'unregister'],
    cleanupInvariant: `On ANY failure or non-improvement, run the unregister command and confirm via list that ${variant} is gone, leaving the project byte-exact pristine.`,
  }
}
// --- END embedded-eval substrate ---

const WORKFLOW_SUITABILITY = {
  supported_languages: ['cutlass', 'cuda', 'cpp', 'metal'],
  supported_problem_types: ['cutlass-pattern-synthesis', 'cutlass-gemm-optimization', 'gpu-kernel-optimization'],
  problem_types: ['CUTLASS compositional pattern synthesis', 'pattern discovery/realization/composition for CUDA C++/Metal kernels'],
  reason: 'FACT is tied to CUTLASS CUDA C++ pattern synthesis and ablation feedback.',
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

// --- Embedded-dispatch mode (gated; standalone path is byte-identical when off) ---
const INTEGRATION_PATTERN = (args.integration_pattern || 'standalone')
const EMBEDDED = INTEGRATION_PATTERN.startsWith('embedded')
const REGISTER_SCRIPT = args.register_script || ''
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const REFERENCE_FILE = args.reference_cuh || args.reference_file || ''
const REGISTER_PARAMS = args.register_params || ''
// Standalone synthesis is fully agent-narrated; embedded mode drives the project's
// own build/test/benchmark commands against a contract-conforming register adapter.
const BUILD_CMD = args.build_command || ''
const TEST_CMD = args.test_command || ''
const BENCHMARK_CMD = args.benchmark_command || ''

if (EMBEDDED) {
  const missing = []
  if (!REGISTER_SCRIPT) missing.push('register_script')
  if (!PROJECT_ROOT) missing.push('project_root (or ggml_root)')
  if (!BUILD_CMD) missing.push('build_command')
  if (!TEST_CMD) missing.push('test_command')
  if (!BENCHMARK_CMD) missing.push('benchmark_command')
  if (missing.length) {
    throw new Error(`integration_pattern="${INTEGRATION_PATTERN}" (embedded dispatch) requires non-empty: ${missing.join(', ')}`)
  }
}

// FACT: Compositional kernel synthesis framework
// Based on GitHub:Project-FACT/FACT (no published paper yet)
// Discovers, realizes, and composes optimization patterns for CUTLASS

async function main() {
  // ============================================================================
  // Phase 1: Setup
  // ============================================================================
  phase('Setup'); await __genomeReport('Setup', meta.name);

  const setupResult = await agent(
    `Set up FACT compositional synthesis environment:

1. Initialize CUTLASS workspace:
   - CUTLASS version and headers
   - Target GPU architecture (sm_80, sm_89, sm_90, etc.)
   - Available tensor core operations
2. Set up pattern registry T(rule, dtype, architecture):
   - rule: transformation type (tiling, fusion, memory, etc.)
   - dtype: data type (fp32, fp16, bf16, int8)
   - architecture: GPU architecture (Ampere, Hopper, etc.)
3. Identify target kernel specification:
   - Operation (GEMM, Conv, attention, etc.)
   - Input shapes and dtypes
   - Target architecture
4. Load exemplar kernels:
   - High-performance reference implementations
   - Pattern sources for discovery
5. Configure synthesis parameters:
   - Pattern discovery depth
   - Composition budget
   - Ablation strategy

Return JSON:
{
  "cutlass_version": "3.x|2.x",
  "target_architecture": "sm_80|sm_89|sm_90",
  "tensor_cores_available": true/false,
  "kernel_spec": {
    "operation": "gemm|conv2d|attention|...",
    "shapes": "shape description",
    "dtypes": ["fp32", "fp16", ...],
    "target_arch": "ampere|hopper|..."
  },
  "pattern_registry_path": "path/to/registry",
  "exemplar_kernels": ["exemplar1", "exemplar2", ...],
  "discovery_depth": <int>,
  "composition_budget": <int>,
  "baseline_gflops": <float or null>
}`,
    {
      label: 'Setup FACT',
      phase: 'Setup',
      schema: {
        type: 'object',
        properties: {
          cutlass_version: { type: 'string' },
          target_architecture: { type: 'string' },
          tensor_cores_available: { type: 'boolean' },
          kernel_spec: { type: 'object' },
          pattern_registry_path: { type: 'string' },
          exemplar_kernels: { type: 'array', items: { type: 'string' } },
          discovery_depth: { type: 'integer' },
          composition_budget: { type: 'integer' },
          baseline_gflops: { type: ['number', 'null'] },
        },
        required: ['cutlass_version', 'target_architecture', 'kernel_spec'],
      },
    }
  );

  if (!setupResult) {
    log('Setup failed');
    return { success: false, reason: 'setup_failed' };
  }

  log(`Target: ${setupResult.kernel_spec.operation} on ${setupResult.target_architecture}`);
  log(`CUTLASS ${setupResult.cutlass_version}, Tensor Cores: ${setupResult.tensor_cores_available ? 'Yes' : 'No'}`);
  log(`Exemplars: ${setupResult.exemplar_kernels.length}`);

  const kernelSpec = setupResult.kernel_spec;
  const discoveryDepth = setupResult.discovery_depth || 3;
  const compositionBudget = setupResult.composition_budget || 50;

  // Pattern registry: T(rule, dtype, architecture) -> code transformation
  const patternRegistry = [];
  const discoveredPatterns = [];
  const realizedPatterns = [];
  let composedKernels = [];

  // ============================================================================
  // Phase 2: Pattern Discovery
  // ============================================================================
  phase('Pattern Discovery'); await __genomeReport('Pattern Discovery', meta.name);

  log(`Discovering optimization patterns from ${setupResult.exemplar_kernels.length} exemplars...`);

  const discoveryResult = await agent(
    `Discover optimization patterns from exemplar kernels:

Target operation: ${kernelSpec.operation}
Target architecture: ${setupResult.target_architecture}
Exemplar kernels: ${setupResult.exemplar_kernels.join(', ')}
Discovery depth: ${discoveryDepth}

Pattern discovery process:
1. Analyze exemplar implementations:
   - Extract code structure and transformations
   - Identify optimization idioms
   - Recognize architecture-specific patterns
2. Classify patterns by type:
   - Tiling patterns (block sizes, thread mapping)
   - Memory patterns (shared memory, global coalescing)
   - Compute patterns (tensor core usage, instruction scheduling)
   - Fusion patterns (operation fusion, epilogue fusion)
   - Data layout patterns (swizzling, padding)
3. Abstract patterns to rules:
   - Input conditions (shape constraints, dtype requirements)
   - Transformation logic (code template with parameters)
   - Output properties (performance characteristics)
4. Index patterns in registry: T(rule_type, dtype, architecture)

Return JSON:
{
  "exemplars_analyzed": <int>,
  "patterns_discovered": [
    {
      "pattern_id": "unique_id",
      "pattern_name": "descriptive name",
      "pattern_type": "tiling|memory|compute|fusion|layout",
      "rule_type": "rule classification",
      "applicable_dtypes": ["fp32", "fp16", ...],
      "applicable_architectures": ["sm_80", "sm_89", ...],
      "description": "what this pattern does",
      "abstraction": "abstract rule description",
      "expected_impact": "high|medium|low"
    },
    ...
  ],
  "discovery_summary": "summary of discovered patterns"
}`,
    {
      label: 'Discover patterns',
      phase: 'Pattern Discovery',
      schema: {
        type: 'object',
        properties: {
          exemplars_analyzed: { type: 'integer' },
          patterns_discovered: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                pattern_id: { type: 'string' },
                pattern_name: { type: 'string' },
                pattern_type: { type: 'string' },
                rule_type: { type: 'string' },
                applicable_dtypes: { type: 'array', items: { type: 'string' } },
                applicable_architectures: { type: 'array', items: { type: 'string' } },
                description: { type: 'string' },
                abstraction: { type: 'string' },
                expected_impact: { type: 'string' },
              },
              required: ['pattern_id', 'pattern_name', 'pattern_type', 'description'],
            },
          },
          discovery_summary: { type: 'string' },
        },
        required: ['exemplars_analyzed', 'patterns_discovered', 'discovery_summary'],
      },
    }
  );

  if (!discoveryResult || discoveryResult.patterns_discovered.length === 0) {
    log('Pattern discovery failed or no patterns found');
    return { success: false, reason: 'discovery_failed' };
  }

  discoveredPatterns.push(...discoveryResult.patterns_discovered);
  log(`Discovered ${discoveredPatterns.length} patterns:`);
  for (const pattern of discoveredPatterns) {
    log(`  - ${pattern.pattern_name} (${pattern.pattern_type}, ${pattern.expected_impact} impact)`);
  }

  // ============================================================================
  // Phase 3: Pattern Realization
  // ============================================================================
  phase('Pattern Realization'); await __genomeReport('Pattern Realization', meta.name);

  log('Realizing patterns as CUTLASS code transformations...');

  const realizationResult = await agent(
    `Realize discovered patterns as concrete code transformations:

Target: ${kernelSpec.operation}
Architecture: ${setupResult.target_architecture}
Data types: ${kernelSpec.dtypes.join(', ')}

Patterns to realize:
${discoveredPatterns.map((p, idx) => `${idx + 1}. ${p.pattern_name}: ${p.description}`).join('\n')}

Realization process:
1. For each pattern:
   a. Generate CUTLASS code template
   b. Define transformation parameters (e.g., tile sizes, thread counts)
   c. Specify applicability constraints
   d. Create dependency graph (which patterns can compose)
2. Index realized patterns in registry:
   T(rule_type, dtype, architecture) -> code_transformation
3. Validate realization:
   - Syntactic correctness
   - Type safety
   - Resource constraints (registers, shared memory)

Return JSON:
{
  "patterns_realized": [
    {
      "pattern_id": "same as discovery",
      "pattern_name": "same as discovery",
      "code_template": "CUTLASS code template with placeholders",
      "parameters": [
        {"name": "param1", "type": "int", "range": [min, max]},
        ...
      ],
      "constraints": "applicability constraints",
      "dependencies": ["pattern_id1", "pattern_id2", ...],
      "estimated_resource_usage": {
        "registers_per_thread": <int>,
        "shared_memory_bytes": <int>
      }
    },
    ...
  ],
  "realization_summary": "summary of realization process",
  "dependency_graph": "description of pattern dependencies"
}`,
    {
      label: 'Realize patterns',
      phase: 'Pattern Realization',
      schema: {
        type: 'object',
        properties: {
          patterns_realized: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                pattern_id: { type: 'string' },
                pattern_name: { type: 'string' },
                code_template: { type: 'string' },
                parameters: { type: 'array' },
                constraints: { type: 'string' },
                dependencies: { type: 'array', items: { type: 'string' } },
                estimated_resource_usage: { type: 'object' },
              },
              required: ['pattern_id', 'pattern_name', 'code_template'],
            },
          },
          realization_summary: { type: 'string' },
          dependency_graph: { type: 'string' },
        },
        required: ['patterns_realized', 'realization_summary'],
      },
    }
  );

  if (!realizationResult || realizationResult.patterns_realized.length === 0) {
    log('Pattern realization failed');
    return { success: false, reason: 'realization_failed' };
  }

  realizedPatterns.push(...realizationResult.patterns_realized);
  log(`Realized ${realizedPatterns.length} patterns as code transformations`);

  // ============================================================================
  // Phase 4: Pattern Composition
  // ============================================================================
  phase('Pattern Composition'); await __genomeReport('Pattern Composition', meta.name);

  log(`Composing patterns to generate optimized kernels (budget: ${compositionBudget})...`);

  // In embedded mode each composed kernel must be a complete dispatch-compatible
  // .cuh that matches the project's reference dispatch signature exactly.
  const compositionEmbeddingBlock = EMBEDDED
    ? `\n\n${EMBEDDING_CONTRACT}\n\nMANDATORY (embedded dispatch): Read the reference dispatch file at ${REFERENCE_FILE} and match its dispatch signature EXACTLY (same entry-point shape, template params, launch-bounds conventions). Each composed candidate's kernel_code MUST be a COMPLETE dispatch-compatible \`.cuh\` (NOT a standalone translation unit, NO main()/harness/top-level test code). Use ONLY symbols/headers the project already provides; do not register, build, or benchmark the variant yourself.`
    : '';

  const compositionResult = await agent(
    `Compose patterns to generate optimized CUTLASS kernels:

Target specification:
- Operation: ${kernelSpec.operation}
- Shapes: ${kernelSpec.shapes}
- Data types: ${kernelSpec.dtypes.join(', ')}
- Architecture: ${setupResult.target_architecture}

Available patterns: ${realizedPatterns.length}
Dependency graph: ${realizationResult.dependency_graph}
Composition budget: ${compositionBudget} kernel candidates

Composition strategies:
1. Greedy composition:
   - Start with baseline CUTLASS implementation
   - Iteratively add compatible patterns
   - Prioritize high-impact patterns
2. Dependency-aware composition:
   - Respect pattern dependencies from graph
   - Compose compatible pattern groups
3. Search-based composition:
   - Generate pattern combinations
   - Filter by constraints (resources, compatibility)
   - Sample diverse compositions
4. Parameter tuning:
   - For each composition, tune pattern parameters
   - Use heuristics or light autotuning

Generate top ${Math.min(compositionBudget, 20)} kernel candidates.

Return JSON:
{
  "composition_strategy": "greedy|dependency-aware|search-based",
  "candidates_generated": <int>,
  "composed_kernels": [
    {
      "kernel_id": "unique_id",
      "applied_patterns": ["pattern_id1", "pattern_id2", ...],
      "pattern_parameters": {"pattern_id": {"param": value}, ...},
      "kernel_code": "complete CUTLASS kernel code",
      "estimated_performance": "performance estimate if available",
      "composition_rationale": "why these patterns were composed"
    },
    ...
  ],
  "composition_summary": "summary of composition process"
}${compositionEmbeddingBlock}`,
    {
      label: 'Compose patterns',
      phase: 'Pattern Composition',
      schema: {
        type: 'object',
        properties: {
          composition_strategy: { type: 'string' },
          candidates_generated: { type: 'integer' },
          composed_kernels: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                kernel_id: { type: 'string' },
                applied_patterns: { type: 'array', items: { type: 'string' } },
                pattern_parameters: { type: 'object' },
                kernel_code: { type: 'string' },
                estimated_performance: { type: 'string' },
                composition_rationale: { type: 'string' },
              },
              required: ['kernel_id', 'applied_patterns', 'kernel_code'],
            },
          },
          composition_summary: { type: 'string' },
        },
        required: ['candidates_generated', 'composed_kernels', 'composition_summary'],
      },
    }
  );

  if (!compositionResult || compositionResult.composed_kernels.length === 0) {
    log('Pattern composition failed');
    return { success: false, reason: 'composition_failed' };
  }

  composedKernels = compositionResult.composed_kernels;
  log(`Generated ${composedKernels.length} composed kernel candidates`);

  // ============================================================================
  // Phase 5: Ablation Studies
  // ============================================================================
  phase('Ablation'); await __genomeReport('Ablation', meta.name);

  log('Running ablation studies to validate pattern contributions...');

  const ablationResult = await agent(
    `Run ablation studies on top composed kernels:

Top kernels: ${Math.min(composedKernels.length, 5)}

Ablation process:
1. Select top-performing kernels (by estimated performance)
2. For each kernel, create ablation variants:
   - Remove one pattern at a time (leave-one-out)
   - Remove pattern groups
   - Baseline (no patterns)
3. Execute ablation variants on ${setupResult.target_architecture}
4. Measure performance impact of each pattern
5. Identify critical patterns vs marginal patterns

Return JSON:
{
  "kernels_ablated": <int>,
  "ablation_results": [
    {
      "kernel_id": "original kernel id",
      "baseline_gflops": <float>,
      "pattern_contributions": [
        {
          "pattern_id": "pattern_id",
          "pattern_name": "pattern_name",
          "ablated_gflops": <float>,
          "contribution_pct": <float>,
          "criticality": "critical|important|marginal"
        },
        ...
      ]
    },
    ...
  ],
  "critical_patterns": ["pattern_id1", "pattern_id2", ...],
  "ablation_summary": "summary of ablation findings"
}`,
    {
      label: 'Ablation studies',
      phase: 'Ablation',
      schema: {
        type: 'object',
        properties: {
          kernels_ablated: { type: 'integer' },
          ablation_results: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                kernel_id: { type: 'string' },
                baseline_gflops: { type: 'number' },
                pattern_contributions: { type: 'array' },
              },
            },
          },
          critical_patterns: { type: 'array', items: { type: 'string' } },
          ablation_summary: { type: 'string' },
        },
        required: ['kernels_ablated', 'ablation_results', 'critical_patterns'],
      },
    }
  );

  if (!ablationResult) {
    log('Ablation studies failed');
  } else {
    log(`Ablation complete: identified ${ablationResult.critical_patterns.length} critical patterns`);
  }

  // ============================================================================
  // Phase 6: Evaluation
  // ============================================================================
  phase('Evaluation'); await __genomeReport('Evaluation', meta.name);

  log('Evaluating composed kernels on target hardware...');

  // Embedded-dispatch evaluation: each composed kernel is registered into the
  // project, built/tested/benchmarked via the project's own commands, then ALWAYS
  // unregistered back to pristine. Replaces the standalone CUTLASS compile path.
  let evaluationEmbeddingBlock = '';
  if (EMBEDDED) {
    const planBlocks = composedKernels.map((k, idx) => {
      const variantName = `fact_${k.kernel_id || ('k' + idx)}`.replace(/[^A-Za-z0-9_]/g, '_');
      const candidatePath = `${PROJECT_ROOT}/.fact_candidates/${variantName}.cuh`;
      const plan = __embeddedEvalPlan({
        adapter: 'python "' + REGISTER_SCRIPT + '"',
        variant: variantName,
        source: candidatePath,
        projectRoot: PROJECT_ROOT,
        params: REGISTER_PARAMS,
        buildCmd: BUILD_CMD,
        testCmd: TEST_CMD,
        benchmarkCmd: BENCHMARK_CMD,
      });
      return `### Candidate kernel_id=${k.kernel_id || ('k' + idx)} (variant ${variantName})
Write this candidate's kernel_code verbatim to ${candidatePath}, then run IN THIS EXACT ORDER:
1. Register:   ${plan.register}
2. List:       ${plan.list}   (CONFIRM ${variantName} is now listed; abort this candidate if absent)
3. Build:      ${plan.build}
4. Test:       ${plan.test}        (correctness)
5. Benchmark:  ${plan.benchmark}   (latency)
6. Unregister: ${plan.unregister}
7. List:       ${plan.list}   (CONFIRM ${variantName} is GONE)
HARD REQUIREMENT (cleanup invariant): ${plan.cleanupInvariant}`;
    }).join('\n\n');
    evaluationEmbeddingBlock = `

# EMBEDDED-DISPATCH EVALUATION (overrides the standalone CUTLASS compile/execute steps below)
These kernels are NOT standalone translation units; each is a dispatch-compatible \`.cuh\` that must be wired into the project at ${PROJECT_ROOT} via the register adapter. Do NOT attempt a standalone \`nvcc\`/CUTLASS compile. For EACH candidate below, run its commands in order, and ALWAYS run the unregister command and confirm removal via list even on build/correctness/benchmark FAILURE or non-improvement — never leave the project dirty.

${planBlocks}

Map per-candidate results into evaluation_results: compilation_success=build succeeded, correctness_passed=test passed, gflops/execution_time derived from the benchmark output. Parse correctness (pass/fail) and latency STRICTLY from the actual test/benchmark command output. Do NOT fabricate numbers; if a value is not present in the output, report it as unavailable rather than guessing.`;
  }

  const evaluationResult = await agent(
    `Evaluate all composed kernels:${evaluationEmbeddingBlock}

Kernels to evaluate: ${composedKernels.length}
Target: ${setupResult.target_architecture}
Baseline: ${setupResult.baseline_gflops || 'N/A'} GFLOPS

Evaluation:
1. Compile each kernel with CUTLASS
2. Execute on target GPU
3. Measure performance:
   - Execution time
   - GFLOPS
   - Memory bandwidth utilization
   - Tensor core utilization (if applicable)
4. Verify correctness
5. Rank kernels by performance

Return JSON:
{
  "kernels_evaluated": <int>,
  "evaluation_results": [
    {
      "kernel_id": "kernel_id",
      "applied_patterns": ["pattern1", "pattern2", ...],
      "compilation_success": true/false,
      "correctness_passed": true/false,
      "execution_time_ms": <float>,
      "gflops": <float>,
      "memory_bandwidth_utilization_pct": <float>,
      "tensor_core_utilization_pct": <float>
    },
    ...
  ],
  "best_kernel": {
    "kernel_id": "best kernel id",
    "gflops": <float>,
    "speedup_vs_baseline": <float>,
    "applied_patterns": ["pattern1", "pattern2", ...]
  },
  "evaluation_summary": "summary of evaluation"
}`,
    {
      label: 'Evaluate kernels',
      phase: 'Evaluation',
      schema: {
        type: 'object',
        properties: {
          kernels_evaluated: { type: 'integer' },
          evaluation_results: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                kernel_id: { type: 'string' },
                applied_patterns: { type: 'array', items: { type: 'string' } },
                compilation_success: { type: 'boolean' },
                correctness_passed: { type: 'boolean' },
                gflops: { type: 'number' },
              },
            },
          },
          best_kernel: { type: 'object' },
          evaluation_summary: { type: 'string' },
        },
        required: ['kernels_evaluated', 'evaluation_results', 'best_kernel'],
      },
    }
  );

  if (!evaluationResult) {
    log('Evaluation failed');
    return { success: false, reason: 'evaluation_failed' };
  }

  log(`Evaluated ${evaluationResult.kernels_evaluated} kernels`);
  log(`Best: ${evaluationResult.best_kernel.gflops.toFixed(2)} GFLOPS (${evaluationResult.best_kernel.speedup_vs_baseline}x speedup)`);

  // ============================================================================
  // Phase 7: Report
  // ============================================================================
  phase('Report'); await __genomeReport('Report', meta.name);

  const report = await agent(
    `Generate FACT compositional synthesis report:

Summary:
- Target: ${kernelSpec.operation} on ${setupResult.target_architecture}
- Patterns discovered: ${discoveredPatterns.length}
- Patterns realized: ${realizedPatterns.length}
- Kernels composed: ${composedKernels.length}
- Kernels evaluated: ${evaluationResult.kernels_evaluated}
- Best performance: ${evaluationResult.best_kernel.gflops.toFixed(2)} GFLOPS
- Speedup: ${evaluationResult.best_kernel.speedup_vs_baseline}x

Critical patterns: ${ablationResult?.critical_patterns.join(', ') || 'N/A'}

Generate report with:
1. Executive summary
2. Pattern discovery analysis
3. Pattern realization details
4. Composition strategy
5. Ablation study results
6. Performance evaluation
7. Best kernel breakdown

Return JSON:
{
  "summary": "brief summary",
  "patterns_discovered": ${discoveredPatterns.length},
  "patterns_realized": ${realizedPatterns.length},
  "kernels_composed": ${composedKernels.length},
  "best_gflops": ${evaluationResult.best_kernel.gflops},
  "speedup": ${evaluationResult.best_kernel.speedup_vs_baseline},
  "critical_patterns": ${JSON.stringify(ablationResult?.critical_patterns || [])},
  "report_path": "path/to/report.md"
}`,
    {
      label: 'Generate report',
      phase: 'Report',
      schema: {
        type: 'object',
        properties: {
          summary: { type: 'string' },
          patterns_discovered: { type: 'integer' },
          patterns_realized: { type: 'integer' },
          kernels_composed: { type: 'integer' },
          best_gflops: { type: 'number' },
          speedup: { type: 'number' },
          critical_patterns: { type: 'array', items: { type: 'string' } },
          report_path: { type: 'string' },
        },
        required: ['summary', 'best_gflops'],
      },
    }
  );

  // ============================================================================
  // Return final results
  // ============================================================================

  return {
    success: true,
    method: 'FACT',
    approach: 'Compositional pattern synthesis',
    kernel: kernelSpec.operation,
    target_architecture: setupResult.target_architecture,
    cutlass_version: setupResult.cutlass_version,
    patterns_discovered: discoveredPatterns.length,
    patterns_realized: realizedPatterns.length,
    kernels_composed: composedKernels.length,
    kernels_evaluated: evaluationResult.kernels_evaluated,
    baseline_gflops: setupResult.baseline_gflops,
    best_gflops: evaluationResult.best_kernel.gflops,
    speedup: evaluationResult.best_kernel.speedup_vs_baseline,
    best_kernel_id: evaluationResult.best_kernel.kernel_id,
    best_kernel_patterns: evaluationResult.best_kernel.applied_patterns,
    critical_patterns: ablationResult?.critical_patterns || [],
    pattern_registry: patternRegistry,
    report: report?.report_path,
    summary: report?.summary,
  };
}

// Execute the workflow
return await main();
