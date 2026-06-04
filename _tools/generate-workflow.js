export const meta = {
  name: 'generate-workflow',
  description: 'Generate a new kernel optimization Claude Code Workflow from a paper or repo URL',
  whenToUse: 'When adding a new academic/industrial kernel optimization method to Awesome-Kernel-Workflows. Takes a paper URL, analyzes the methodology, generates a manifest, and produces a validated .js workflow file.',
  phases: [
    { title: 'Research', detail: 'Fetch and analyze the paper/repo to understand the optimization methodology' },
    { title: 'Model', detail: 'Classify loop topology, identify phases, args, and plan angles' },
    { title: 'Assemble', detail: 'Compose complete manifest YAML from extracted structure' },
    { title: 'Generate', detail: 'Produce .js workflow from template + manifest' },
    { title: 'Validate', detail: 'Static and semantic checks on the generated workflow' },
  ],
}

// =============================================================================
// Meta-Workflow: Paper/Repo → Manifest → Validated Workflow (.js)
// =============================================================================
//
// Usage:
//   Workflow({name: 'generate-workflow', args: {
//     paper_url: 'https://arxiv.org/abs/XXXX.XXXXX',
//     repo_url: 'https://github.com/org/repo',          // optional
//     output_dir: '/path/to/Awesome-Kernel-Workflows',   // where to write
//     method_name: 'MethodName',                          // optional override
//     partial_manifest: '...',                            // optional pre-filled YAML
//   }})
//
// Returns:
//   { workflow_path, manifest_yaml, validation_report, method_name }
//
// =============================================================================

const PAPER_URL = args.paper_url || ''
const REPO_URL = args.repo_url || ''
const OUTPUT_DIR = args.output_dir || '.'
const METHOD_NAME = args.method_name || ''
const PARTIAL_MANIFEST = args.partial_manifest || ''

// =============================================================================
// Phase 1: Research — Analyze paper and repo
// =============================================================================
phase('Research')

const researchResults = await parallel([
  // Agent 1: Analyze the paper
  () => agent(`You are a research paper analyst specializing in GPU kernel optimization.

# Task
Analyze the paper at: ${PAPER_URL}
${PAPER_URL ? `Fetch and read the paper content from the URL above.` : 'No paper URL provided — skip paper analysis.'}

# Extract:
1. **Paper metadata**: title, venue, year, authors
2. **Core methodology**: What is the optimization approach? Describe the loop/pipeline.
3. **Phases/steps**: What discrete steps does the method take? List them in order.
4. **Feedback signal**: What metric or measurement drives decisions? (e.g., NCU profiling, throughput measurement, correctness testing)
5. **Profiling tool**: What tool is used to measure performance? (ncu, perf, custom benchmark, etc.)
6. **Key innovation**: What makes this method different from just "try things and benchmark"?
7. **Iteration/termination**: Does it iterate? How many times? What stops it?
8. **Experience/learning**: Does it accumulate knowledge across iterations?
9. **Parallelism opportunities**: Which steps could run in parallel?
10. **Required inputs**: What does a user need to provide to use this method?

Return a structured research brief.`, {
    label: 'research-paper',
    phase: 'Research',
    schema: {
      type: 'object',
      properties: {
        paper_title: { type: 'string' },
        paper_venue: { type: 'string' },
        core_methodology: { type: 'string' },
        phases_identified: { type: 'array', items: { type: 'string' } },
        feedback_signal: { type: 'string' },
        profiling_tool: { type: 'string' },
        key_innovation: { type: 'string' },
        iteration_strategy: { type: 'string' },
        has_experience_accumulation: { type: 'boolean' },
        parallelism_opportunities: { type: 'array', items: { type: 'string' } },
        required_inputs: { type: 'array', items: { type: 'string' } },
        method_category: { type: 'string' },
      },
      required: ['core_methodology', 'phases_identified', 'feedback_signal', 'method_category'],
    },
  }),

  // Agent 2: Analyze the repo (if provided)
  () => agent(`You are a code repository analyst specializing in GPU kernel optimization tools.

# Task
${REPO_URL ? `Analyze the repository at: ${REPO_URL}
Fetch the README and explore the repository structure.` : 'No repo URL provided. Return minimal structure.'}

# Extract (if repo available):
1. **Repository structure**: Key directories and files
2. **Configuration/args**: What parameters does the tool accept? (CLI args, config files, YAML)
3. **Usage examples**: How is the tool invoked?
4. **Dependencies**: What does it require? (CUDA version, Python packages, etc.)
5. **Output format**: What does it produce?
6. **Build instructions**: How to compile/install?
7. **License**: What license is used?

If no repo URL is provided, return empty/default values.`, {
    label: 'research-repo',
    phase: 'Research',
    schema: {
      type: 'object',
      properties: {
        repo_structure: { type: 'string' },
        config_args: { type: 'array', items: { type: 'string' } },
        usage_examples: { type: 'array', items: { type: 'string' } },
        dependencies: { type: 'array', items: { type: 'string' } },
        output_format: { type: 'string' },
        build_instructions: { type: 'string' },
        license: { type: 'string' },
      },
      required: ['repo_structure'],
    },
  }),
])

const paperResearch = researchResults[0]
const repoResearch = researchResults[1]

log(`Research: "${paperResearch?.paper_title || 'unknown'}" — ${paperResearch?.method_category || 'unclassified'}`)

// =============================================================================
// Phase 2: Model — Classify topology, identify phases, args, plan angles
// =============================================================================
phase('Model')

const modelResults = await parallel([
  // Agent 1: Topology and state variables
  () => agent(`You are a workflow architect. Based on this research brief, classify the loop topology and identify state variables.

# Paper Research:
- Core methodology: ${paperResearch?.core_methodology || 'unknown'}
- Phases identified: ${JSON.stringify(paperResearch?.phases_identified || [])}
- Iteration strategy: ${paperResearch?.iteration_strategy || 'unknown'}
- Has experience accumulation: ${paperResearch?.has_experience_accumulation || false}
- Method category: ${paperResearch?.method_category || 'unknown'}

# Topology Classification
Choose ONE:
- **iterative**: Plan→Execute→Evaluate→Learn→Repeat (has a for/while loop with fixed or convergence-based termination)
- **search**: Generate search space→Sample→Evaluate→Prune→Repeat (population-based, evolutionary, or Bayesian)
- **tree**: Branch→Score→Expand→Backtrack (tree-of-thought, beam search)
- **pipeline**: Analyze→Transform→Verify (single-pass, no iteration)

# State Variables
For the chosen topology, what mutable state is carried across iterations?
Each variable needs: name (camelCase JS), type (number|string|array|object), initial_value (JS expression), description.

Common patterns:
- Iterative: bestMetric, bestArtifact, baselineMetric, experienceMemory[]
- Search: population[], bestConfig, searchBounds
- Tree: beamSet[], visitedNodes, bestScore

Return topology classification and state variables.`, {
    label: 'model-topology',
    phase: 'Model',
    schema: {
      type: 'object',
      properties: {
        topology_type: { type: 'string' },
        max_iterations_arg: { type: 'string' },
        convergence_condition: { type: 'string' },
        state_variables: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              type: { type: 'string' },
              initial_value: { type: 'string' },
              description: { type: 'string' },
            },
            required: ['name', 'type', 'initial_value', 'description'],
          },
        },
      },
      required: ['topology_type', 'state_variables'],
    },
  }),

  // Agent 2: Phases and agents
  () => agent(`You are a workflow architect. Design the phase structure and agent calls for this kernel optimization workflow.

# Paper Research:
- Core methodology: ${paperResearch?.core_methodology || 'unknown'}
- Phases identified: ${JSON.stringify(paperResearch?.phases_identified || [])}
- Feedback signal: ${paperResearch?.feedback_signal || 'unknown'}
- Profiling tool: ${paperResearch?.profiling_tool || 'unknown'}
- Parallelism opportunities: ${JSON.stringify(paperResearch?.parallelism_opportunities || [])}
- Has experience accumulation: ${paperResearch?.has_experience_accumulation || false}

# Design Rules:
1. Each phase maps to one phase() call in the workflow
2. Each phase has position: "preamble" (before loop), "loop_body" (inside loop), or "epilogue" (after loop)
3. Each agent within a phase needs:
   - label: kebab-case unique identifier
   - description: what it does
   - parallelism: "single", "parallel_fan_out", or "pipeline_then_parallel"
   - output_schema: what fields does it return (just list field names and types)

# Standard phase patterns:
- Setup (preamble): read artifact + profile baseline → 1-2 single agents
- Plan (loop_body): generate diverse strategies → 1 parallel_fan_out agent (breadth planners)
- Execute (loop_body): implement plans → 1 pipeline_then_parallel agent
- Evaluate (loop_body): measure each variant → 1 parallel_fan_out agent
- Learn (loop_body, if experience accumulation): extract patterns → 1 parallel_fan_out agent
- Iterate (loop_body): log only, no agents

Adapt this to the specific method. Not all methods need all phases.

Return the phase design.`, {
    label: 'model-phases',
    phase: 'Model',
    schema: {
      type: 'object',
      properties: {
        phases: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              detail: { type: 'string' },
              position: { type: 'string' },
              agents: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    label: { type: 'string' },
                    description: { type: 'string' },
                    parallelism: { type: 'string' },
                    output_fields: { type: 'array', items: { type: 'string' } },
                  },
                  required: ['label', 'description', 'parallelism'],
                },
              },
            },
            required: ['name', 'detail', 'position'],
          },
        },
      },
      required: ['phases'],
    },
  }),

  // Agent 3: Args extraction
  () => agent(`You are a workflow architect. Extract all configurable parameters for this kernel optimization workflow.

# Paper Research:
- Required inputs: ${JSON.stringify(paperResearch?.required_inputs || [])}
- Profiling tool: ${paperResearch?.profiling_tool || 'unknown'}
- Iteration strategy: ${paperResearch?.iteration_strategy || 'unknown'}

# Repo Research:
- Config args found: ${JSON.stringify(repoResearch?.config_args || [])}
- Usage examples: ${JSON.stringify(repoResearch?.usage_examples || [])}

# Standard args patterns for kernel optimization workflows:
Use canonical common args:
- kernel_path: existing kernel/source file to optimize
- problem_definition: inline problem definition when no initial kernel exists
- problem_path: file containing the problem definition
- language: implementation language
- target_gpu: target accelerator string
- compile_command: compile/import command with {kernel_path}/{result_path}
- test_command: correctness command with {kernel_path}/{result_path}
- benchmark_command: performance command with {kernel_path}/{result_path}
- iterations: outer-loop budget unless a paper-specific budget name is required
- seed_candidates: number of initial generated kernels for generation mode
- breadth: parallel plans per iteration
- samples_per_plan: variants per plan
- exp_dir: artifact directory
- [profiling tool args]: paths, commands, configurations for the tool

At least one of kernel_path, problem_definition, or problem_path must be available for optimization/generation workflows.
Do not emit old aliases such as gpu_target, bench_command, eval_command, rounds, problem_description, task_spec, or task_path.
Do not emit concrete default commands such as python ..., nvcc ..., bash ..., or ncu --.... Commands must be user-provided through compile_command, test_command, benchmark_command, profile_command, ncu_command, or method-specific tool args. If a command is missing, the workflow must mark measured evidence as unavailable instead of inventing an evaluator/harness.

Method-specific args:
- Anything unique to this paper's approach

For each arg, provide: name (snake_case), type, default (JS expression for optional), description, example value.

Return required and optional args lists.`, {
    label: 'model-args',
    phase: 'Model',
    schema: {
      type: 'object',
      properties: {
        required_args: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              type: { type: 'string' },
              description: { type: 'string' },
              example: { type: 'string' },
            },
            required: ['name', 'type', 'description', 'example'],
          },
        },
        optional_args: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              type: { type: 'string' },
              default_value: { type: 'string' },
              description: { type: 'string' },
              example: { type: 'string' },
            },
            required: ['name', 'type', 'default_value', 'description'],
          },
        },
      },
      required: ['required_args', 'optional_args'],
    },
  }),
])

const topologyModel = modelResults[0]
const phasesModel = modelResults[1]
const argsModel = modelResults[2]

log(`Model: topology=${topologyModel?.topology_type}, phases=${phasesModel?.phases?.length || 0}, args=${(argsModel?.required_args?.length || 0) + (argsModel?.optional_args?.length || 0)}`)

// Plan angles — sequential, depends on topology + feedback signal
const anglesResult = await agent(`You are a kernel optimization expert. Generate the plan_angles for this workflow.

# Method:
- Category: ${paperResearch?.method_category || 'unknown'}
- Feedback signal: ${paperResearch?.feedback_signal || 'unknown'}
- Profiling tool: ${paperResearch?.profiling_tool || 'unknown'}
- Key innovation: ${paperResearch?.key_innovation || 'unknown'}

# What are plan angles?
Plan angles are orthogonal optimization dimensions. Each planner agent is assigned one angle to focus on, ensuring diverse exploration of the optimization space.

# Examples by profiling tool:
- NCU-based: memory_latency_hiding, memory_coalescing, occupancy, compute_restructuring, data_layout_tiling
- Roofline-based: compute_intensity, memory_bandwidth, parallelism_granularity, instruction_mix
- Throughput-based: batching_strategy, pipeline_depth, kernel_fusion, launch_overhead
- Custom: derive from the specific paper's optimization dimensions

Generate 3-5 plan angles specific to this method. Each needs:
- name: short snake_case label
- description: one sentence suitable for injection into a planner prompt
- trigger: what signal/metric indicates this angle is the most promising

Return the angles.`, {
  label: 'model-angles',
  phase: 'Model',
  schema: {
    type: 'object',
    properties: {
      plan_angles: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            description: { type: 'string' },
            trigger: { type: 'string' },
          },
          required: ['name', 'description'],
        },
      },
    },
    required: ['plan_angles'],
  },
})

// =============================================================================
// Phase 3: Assemble — Compose complete manifest YAML
// =============================================================================
phase('Assemble')

const methodName = METHOD_NAME || (paperResearch?.paper_title || 'unknown')
  .replace(/[^a-zA-Z0-9\s]/g, '')
  .split(/\s+/)
  .slice(0, 2)
  .join('')

const manifestResult = await agent(`You are a YAML generator. Assemble a complete manifest YAML for Awesome-Kernel-Workflows from the structured data below.

# Schema Reference
The manifest must conform to the schema at _manifests/schema.yaml. Fill ALL required fields.

# Data to assemble:

## Source
- paper_title: ${paperResearch?.paper_title || ''}
- paper_url: ${PAPER_URL}
- paper_venue: ${paperResearch?.paper_venue || ''}
- repo_url: ${REPO_URL}
- license: ${repoResearch?.license || ''}

## Workflow Identity
- name: "${methodName.toLowerCase().replace(/\s+/g, '-')}-kernel-optimization"
- description: (generate one-line description from core_methodology)
- when_to_use: (generate usage guidance)
- output_filename: "${methodName.toLowerCase().replace(/\s+/g, '-')}-kernel-optimization.js"
- directory: "${methodName}/"

## Method
- category: ${topologyModel?.topology_type === 'iterative' ? 'iterative_self_improving' : topologyModel?.topology_type || 'iterative_self_improving'}
- core_insight: ${paperResearch?.key_innovation || ''}
- feedback_signal:
  - type: profiler_metric
  - description: ${paperResearch?.feedback_signal || ''}
  - tool: ${paperResearch?.profiling_tool || 'custom_bench'}
  - higher_is_better: false

## Topology
- type: ${topologyModel?.topology_type || 'iterative'}
- state_variables: ${JSON.stringify(topologyModel?.state_variables || [])}
- max_iterations_arg: ${topologyModel?.max_iterations_arg || 'iterations'}
- convergence_condition: ${topologyModel?.convergence_condition || ''}

## Phases
${JSON.stringify(phasesModel?.phases || [], null, 2)}

## Plan Angles
${JSON.stringify(anglesResult?.plan_angles || [], null, 2)}

## Args
- required: ${JSON.stringify(argsModel?.required_args || [])}
- optional: ${JSON.stringify(argsModel?.optional_args || [])}

## Inputs
- supports_existing_kernel: true
- supports_problem_definition: true
- required_one_of: ["kernel_path", "problem_definition", "problem_path"]
- generation:
  - enabled: true
  - seed_candidates_arg: "seed_candidates"
  - output_arg: "generated_kernel_path"
  - requires_correctness_evidence: true

## Template
- base: "${topologyModel?.topology_type === 'iterative' ? 'iterative-loop' : topologyModel?.topology_type === 'search' ? 'search-based' : 'single-pass'}"
- customizations: [${paperResearch?.has_experience_accumulation ? '"experience_memory"' : ''}${paperResearch?.profiling_tool === 'ncu' ? ', "ncu_profiling"' : paperResearch?.profiling_tool ? ', "custom_profiling"' : ''}]

${PARTIAL_MANIFEST ? `# User-provided partial manifest (override above values where specified):\n${PARTIAL_MANIFEST}` : ''}

# Instructions:
Generate a complete, valid YAML manifest. Include ALL sections: source, workflow, inputs, method, topology, phases (with full agent schemas), plan_angles, args, correctness, return_fields, template.

For agent output_schema fields, design sensible JSON schemas based on what each agent needs to return.

Return the manifest as a single YAML string.`, {
  label: 'assemble-manifest',
  phase: 'Assemble',
  schema: {
    type: 'object',
    properties: {
      manifest_yaml: { type: 'string' },
      method_name: { type: 'string' },
      workflow_name: { type: 'string' },
    },
    required: ['manifest_yaml', 'method_name', 'workflow_name'],
  },
})

log(`Manifest assembled: ${manifestResult.workflow_name}`)

// =============================================================================
// Phase 4: Generate — Produce .js workflow from template + manifest
// =============================================================================
phase('Generate')

const templateBase = topologyModel?.topology_type === 'search' ? 'search-based'
  : topologyModel?.topology_type === 'pipeline' ? 'single-pass'
  : 'iterative-loop'

const workflowCode = await agent(`You are a Claude Code Workflow generator. Given this manifest, produce a complete, runnable .js workflow file.

# Manifest:
\`\`\`yaml
${manifestResult.manifest_yaml}
\`\`\`

# Template base: ${templateBase}

# Generation Rules:
1. Start with \`export const meta = {...}\` — fill from manifest workflow.* and phases[].name/detail
2. Immediately after meta, emit \`WORKFLOW_SUITABILITY\` with supported_languages, supported_problem_types, problem_types, reason, plus \`assertWorkflowSuitability()\` that hard-fails only when explicit \`args.language\` or \`args.problem_type\` is incompatible
3. Add header comment block documenting: source paper, usage example with all args, arg descriptions
4. Emit const declarations for all args (required without default, optional with || default)
5. Emit canonical input resolution: optimize args.kernel_path when present; otherwise require args.problem_definition or args.problem_path, generate seed_candidates initial kernels, verify with test_command or benchmark_command, and optimize the best verified seed
6. Emit let declarations for all state_variables
7. Emit phase('Setup') with the Setup phase agents
8. Emit the main loop (for iterative/search: for loop controlled by iterations unless the source method requires a specific budget name)
9. Inside loop: emit each loop_body phase with its agents, using correct parallelism pattern:
   - "single" → await agent(...)
   - "parallel_fan_out" → await parallel(array.map(item => () => agent(...)))
   - "pipeline_then_parallel" → await pipeline(items, item => parallel([...]))
10. Never hardcode an evaluator/compiler/profiler command in Usage examples or agent prompts. Describe the JSON/artifact contract and consume user-provided command args.
11. Emit epilogue (final report agent)
12. Emit return {...} with all return_fields, including input_mode, generated_kernel_path, initial_candidates, and initial_generation_result when generation is supported

# Critical constraints:
- Every agent() call MUST have: label, phase, schema (except final report which can omit schema)
- Every phase() call argument MUST match a title in meta.phases
- parallel() MUST receive array of arrow FUNCTIONS: () => agent(...), NOT direct agent() calls
- The workflow MUST be syntactically valid JavaScript (no TypeScript annotations!)
- Use template literals for prompt interpolation
- Keep prompts detailed and expert-level (300-600 words each) — the agent prompts are the "intelligence" of the workflow

# Agent prompt writing guidelines:
- Setup agents: instruct the agent to read/analyze the target artifact and run the profiling tool
- Plan agents: provide current best code, metrics, experience (if available), and a focus area
- Execute agents: provide the plan and current best code, ask for complete implementation
- Evaluate agents: provide the variant code and baseline metrics, ask for correctness + performance assessment
- Learn agents: provide slow/fast pairs, ask for general reusable optimization patterns
- Report agent: provide all accumulated data, ask for a concise technical summary

Return the complete .js workflow code as a single string.`, {
  label: 'generate-workflow',
  phase: 'Generate',
  schema: {
    type: 'object',
    properties: {
      workflow_code: { type: 'string' },
      filename: { type: 'string' },
      directory: { type: 'string' },
    },
    required: ['workflow_code', 'filename'],
  },
})

log(`Generated: ${workflowCode.filename} (${workflowCode.workflow_code?.length || 0} chars)`)

// =============================================================================
// Phase 5: Validate — Run validation and fix if needed
// =============================================================================
phase('Validate')

const validationResult = await agent(`You are a Claude Code Workflow validator. Perform all validation checks on this generated workflow.

# Workflow Code:
\`\`\`javascript
${workflowCode.workflow_code}
\`\`\`

# Validation Checklist:

## 1. Meta completeness
- [ ] export const meta exists with name, description, whenToUse, phases[]
- [ ] Each phase has title and detail strings
- [ ] meta.name is kebab-case

## 2. Suitability contract
- [ ] WORKFLOW_SUITABILITY exists immediately after meta
- [ ] supported_languages, supported_problem_types, problem_types, and reason are present
- [ ] assertWorkflowSuitability() checks explicit args.language and args.problem_type before work starts

## 3. Phase consistency
- [ ] Every phase() call matches a meta.phases title
- [ ] No orphan phase() calls

## 4. Agent schemas
- [ ] Every agent() has options.label (kebab-case string)
- [ ] Every agent() has options.phase matching a valid phase title
- [ ] Every agent() with schema has type:'object', properties, required
- [ ] Every required field exists in properties

## 5. Args documentation
- [ ] Header comment block exists
- [ ] All args.X references documented in header

## 6. Return envelope
- [ ] Final return {} exists outside the loop
- [ ] Return object is non-empty

## 7. Parallelism integrity
- [ ] parallel() receives array of () => agent(...) thunks
- [ ] pipeline() has correct signature

## 8. Syntax
- [ ] Valid JavaScript (no TS annotations)
- [ ] No undefined variable references
- [ ] Template literals properly closed

If violations are found, ALSO provide a corrected version of the workflow code.

Return validation results.`, {
  label: 'validate-generated',
  phase: 'Validate',
  schema: {
    type: 'object',
    properties: {
      passed: { type: 'boolean' },
      error_count: { type: 'number' },
      warning_count: { type: 'number' },
      violations: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            severity: { type: 'string' },
            message: { type: 'string' },
          },
          required: ['severity', 'message'],
        },
      },
      corrected_code: { type: 'string' },
    },
    required: ['passed', 'error_count', 'violations'],
  },
})

let finalCode = workflowCode.workflow_code

// If validation failed and a corrected version was provided, use it
if (!validationResult.passed && validationResult.corrected_code) {
  log(`Validation found ${validationResult.error_count} errors — applying corrections`)
  finalCode = validationResult.corrected_code
}

// Write the output files
const outputFilename = workflowCode.filename || `${manifestResult.workflow_name}.js`
const outputDirectory = workflowCode.directory || manifestResult.method_name || 'GeneratedMethod'
const workflowPath = `${OUTPUT_DIR}/${outputDirectory}/${outputFilename}`
const manifestPath = `${OUTPUT_DIR}/_manifests/${manifestResult.workflow_name || 'generated'}.yaml`

// The caller (Claude Code) will write these files based on our return value
log(`Output: ${workflowPath}`)
log(`Manifest: ${manifestPath}`)

return {
  workflow_code: finalCode,
  workflow_path: workflowPath,
  manifest_yaml: manifestResult.manifest_yaml,
  manifest_path: manifestPath,
  method_name: manifestResult.method_name || outputDirectory,
  workflow_name: manifestResult.workflow_name || outputFilename.replace('.js', ''),
  validation: {
    passed: validationResult.passed,
    error_count: validationResult.error_count,
    warning_count: validationResult.warning_count || 0,
    violations: validationResult.violations,
  },
  research_summary: {
    paper_title: paperResearch?.paper_title || '',
    method_category: paperResearch?.method_category || '',
    core_methodology: paperResearch?.core_methodology || '',
    topology: topologyModel?.topology_type || '',
  },
}
