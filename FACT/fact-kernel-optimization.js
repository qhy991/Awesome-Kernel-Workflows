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

// FACT: Compositional kernel synthesis framework
// Based on GitHub:Project-FACT/FACT (no published paper yet)
// Discovers, realizes, and composes optimization patterns for CUTLASS

async function main() {
  // ============================================================================
  // Phase 1: Setup
  // ============================================================================
  phase('Setup');

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
  phase('Pattern Discovery');

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
  phase('Pattern Realization');

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
  phase('Pattern Composition');

  log(`Composing patterns to generate optimized kernels (budget: ${compositionBudget})...`);

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
}`,
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
  phase('Ablation');

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
  phase('Evaluation');

  log('Evaluating composed kernels on target hardware...');

  const evaluationResult = await agent(
    `Evaluate all composed kernels:

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
  phase('Report');

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
