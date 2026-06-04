export const meta = {
  name: 'regrapht-kernel-optimization',
  description: 'Reasoning-graph-guided CUDA optimization with Monte Carlo Graph Search (ReGraphT methodology)',
  whenToUse: 'When you want to guide a local or smaller coding model with reusable CUDA optimization trajectories instead of relying only on free-form prompting. Builds or loads a CUDA Reasoning Graph, selects graph paths with Monte Carlo Graph Search, generates candidates from retrieved optimization examples, and updates the graph from real compile/correctness/benchmark feedback.',
  phases: [
    { title: 'Setup', detail: 'Read source kernel/spec, evaluator contract, optional trace corpus, and optional existing reasoning graph' },
    { title: 'BuildGraph', detail: 'Construct or refresh the CUDA Reasoning Graph from optimization trajectories and method labels' },
    { title: 'Select', detail: 'Use Monte Carlo Graph Search to choose a promising optimization path' },
    { title: 'Generate', detail: 'Generate a CUDA candidate conditioned on selected graph examples and method sequence' },
    { title: 'Evaluate', detail: 'Compile, correctness-test, and benchmark the generated candidate' },
    { title: 'UpdateGraph', detail: 'Backpropagate measured reward, record evidence, and expand/relabel graph nodes if needed' },
    { title: 'Report', detail: 'Return best kernel, selected reasoning paths, final graph statistics, and next opportunities' },
  ],
}

const WORKFLOW_SUITABILITY = {
  supported_languages: ['cuda'],
  supported_problem_types: ['cuda-kernel-optimization', 'kernel-search'],
  problem_types: ['CUDA reasoning graph search', 'Monte Carlo graph search over CUDA optimization paths'],
  reason: 'ReGraphT currently builds CUDA reasoning graphs and consumes CUDA evaluator feedback.',
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
// ReGraphT — CUDA Reasoning Graph + Monte Carlo Graph Search Workflow
// =============================================================================
//
// Source: "From Large to Small: Transferring CUDA Optimization Expertise via
//         Reasoning Graph"
//         arXiv:2510.19873 — Junfeng Gong, Zhiyi Wei, Junying Chen,
//         Cheng Liu, Huawei Li
//
// Boundary:
//   This workflow operationalizes ReGraphT as an agent-executable inference and
//   optimization loop. It does not train or fine-tune a small language model.
//   Instead, it builds or loads a CUDA Reasoning Graph from optimization traces,
//   traverses it with Monte Carlo Graph Search, prompts a coding agent with the
//   selected method/example path, and trusts only evaluator evidence.
// adaptation_scope: training_free_inference — this covers the graph-guided
// inference/use phase, not full small-model transfer training.
//
// Usage:
//   Workflow({name: 'regrapht-kernel-optimization', args: {
//     kernel_path: '/path/to/source.cu',    // alias: kernel_path
//     op_description: 'Sequential stencil kernel to CUDA',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     trace_corpus_path: '/path/to/llm_optimization_traces.jsonl',
//     graph_path: '/path/to/regraph.json',
//     baseline_command: '<user-provided baseline command with {result_path}>',
//     iterations: 20,
//     rollouts_per_select: 12,
//     exploration_weight: 1.4,
//     max_path_length: 4,
//     target_gpu: 'H100',
//     exp_dir: '/tmp/regrapht_exp',
//   }})
//
// Evaluator JSON contract:
//   benchmark_command should write JSON at {result_path}:
//   {
//     "compiled": true,
//     "correct": true,
//     "speedup": 1.23,
//     "kernel_time_ms": 0.12,
//     "baseline_time_ms": 0.15,
//     "error_message": "",
//     "error_type": ""
//   }
//
// =============================================================================

// --- Required Args ---
let SOURCE_CODE_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = SOURCE_CODE_PATH ? 'optimize_existing' : 'generate_then_optimize'
const OP_DESC = args.op_description || 'CUDA kernel optimization task'
const EVAL_CMD = args.benchmark_command || ''

// --- Optional Args ---
const TRACE_CORPUS_PATH = args.trace_corpus_path || ''
const GRAPH_PATH = args.graph_path || ''
const BASELINE_CMD = args.baseline_command || ''
const BUDGET = args.iterations || 20
const ROLLOUTS_PER_SELECT = args.rollouts_per_select || 12
const EXPLORATION_WEIGHT = args.exploration_weight ?? 1.4
const MAX_PATH_LENGTH = args.max_path_length || 4
const TARGET_GPU = args.target_gpu || 'H100'
const EXP_DIR = args.exp_dir || '/tmp/regrapht_exp'
const ADAPTATION_SCOPE = 'training_free_inference'
const LANGUAGE = args.language || 'cuda'
const SEED_CANDIDATES = args.seed_candidates || 3

if (!SOURCE_CODE_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

// --- State ---
let sourceCode = ''
let baselineMetric = 1.0
let bestCandidate = null
let graph = {
  nodes: [
    {
      id: 'v_init',
      method: 'initial sequential/source code',
      visits: 0,
      reward: 0,
      examples: [],
    },
  ],
  edges: [],
}
let evaluatedCandidates = []
let selectedPaths = []
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null

function nodeById(id) {
  return graph.nodes.find(node => node.id === id)
}

function outgoingEdges(nodeId) {
  return graph.edges.filter(edge => edge.from === nodeId)
}

function averageReward(nodeId) {
  const node = nodeById(nodeId)
  if (!node || !node.visits) return 0
  return (node.reward || 0) / node.visits
}

function edgeScore(edge, parentVisits) {
  const child = nodeById(edge.to)
  if (!child || !child.visits) return Infinity
  const exploitation = averageReward(edge.to)
  const exploration = EXPLORATION_WEIGHT * Math.sqrt(Math.log(Math.max(parentVisits, 1)) / child.visits)
  return exploitation + exploration + (edge.prior || 0)
}

function bestMeasured() {
  const valid = evaluatedCandidates.filter(item => item.eval?.correct)
  if (!valid.length) return null
  return valid.reduce((best, item) => (item.eval.speedup || 0) > (best.eval.speedup || 0) ? item : best, valid[0])
}

function graphStats() {
  return {
    nodes: graph.nodes.length,
    edges: graph.edges.length,
    visited_nodes: graph.nodes.filter(node => node.visits > 0).length,
    evaluated_candidates: evaluatedCandidates.length,
    correct_candidates: evaluatedCandidates.filter(item => item.eval?.correct).length,
  }
}

// =============================================================================
// Phase 1: Setup
// =============================================================================
phase('Setup')

if (INPUT_MODE === 'generate_then_optimize') {
  const generated = await agent(`No kernel_path was provided. Generate and verify an initial CUDA source file before building the ReGraphT reasoning graph.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- op_description: ${OP_DESC}
- language: ${LANGUAGE}
- target_gpu: ${TARGET_GPU}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- benchmark_command: ${EVAL_CMD || '(not provided)'}
- baseline_command: ${BASELINE_CMD || '(not provided)'}

# Contract
Generate ${SEED_CANDIDATES} complete candidates under ${EXP_DIR}/generated/. Run benchmark_command if available using {kernel_path}/{result_path}. Return the best verified generated kernel path.`, {
    label: 'generate-initial-kernel',
    phase: 'Setup',
    schema: {
      type: 'object',
      properties: {
        generated_kernel_path: { type: 'string' },
        initial_candidates: { type: 'array', items: { type: 'object' } },
        initial_generation_result: { type: 'object' },
      },
      required: ['generated_kernel_path', 'initial_candidates', 'initial_generation_result'],
    },
  })
  initialCandidates = generated.initial_candidates || []
  initialGenerationResult = generated.initial_generation_result || { verified: false }
  generatedKernelPath = generated.generated_kernel_path || ''
  if (!generatedKernelPath) throw new Error('Generation mode did not produce generated_kernel_path')
  if (EVAL_CMD && initialGenerationResult.verified === false) throw new Error('No generated seed passed benchmark evidence')
  SOURCE_CODE_PATH = generatedKernelPath
}

const setupResult = await agent(`Read the CUDA optimization task and evaluator contract.

# Inputs
- kernel_path: ${SOURCE_CODE_PATH}
- operation: ${OP_DESC}
- target_gpu: ${TARGET_GPU}
- benchmark_command: ${EVAL_CMD || '(missing; do not create an evaluator command; measured evidence is unavailable)'}
- baseline_command: ${BASELINE_CMD || '(optional baseline command not provided)'}
- exp_dir: ${EXP_DIR}

# Tasks
1. Read the source code or task spec from kernel_path.
2. Identify the operation, input/output contract, and target kernel entry points.
3. If baseline_command is present, run it or describe the exact command to run and parse its JSON result.
4. State the required evaluator JSON contract. Evaluation evidence must come from benchmark_command; if it is missing, mark evidence as unavailable.
5. Identify CUDA optimization dimensions relevant to this task.

Return structured setup data.`, {
  label: 'setup-task',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      source_code: { type: 'string' },
      operation_type: { type: 'string' },
      input_contract: { type: 'string' },
      output_contract: { type: 'string' },
      kernel_entry_points: { type: 'array', items: { type: 'string' } },
      baseline_metric: { type: 'number' },
      baseline_time_ms: { type: 'number' },
      evaluator_contract: { type: 'string' },
      optimization_dimensions: { type: 'array', items: { type: 'string' } },
    },
    required: ['source_code', 'operation_type', 'evaluator_contract'],
  },
})

sourceCode = setupResult.source_code || ''
baselineMetric = setupResult.baseline_metric || 1.0

// =============================================================================
// Phase 2: BuildGraph
// =============================================================================
phase('BuildGraph')

const graphResult = await agent(`Build or refresh a CUDA Reasoning Graph for ReGraphT.

# Paper-derived graph contract
- A node represents a CUDA optimization method or intermediate optimization state.
- A directed edge represents a transition between two methods.
- The graph may contain cycles.
- Each node and edge should retain examples extracted from optimization trajectories.
- Method labels must be normalized so equivalent LLM phrases map to one method.

# Inputs
- graph_path: ${GRAPH_PATH || '(none; construct from trace corpus and task dimensions)'}
- trace_corpus_path: ${TRACE_CORPUS_PATH || '(none; synthesize a minimal seed graph from task dimensions)'}
- operation: ${OP_DESC}
- target_gpu: ${TARGET_GPU}
- setup optimization dimensions: ${(setupResult.optimization_dimensions || []).join(', ') || 'unknown'}

# Source code excerpt
\`\`\`cuda
${sourceCode.substring(0, 5000)}
\`\`\`

# Tasks
1. If graph_path exists, read the graph and normalize it to {nodes, edges}.
2. If trace_corpus_path exists, extract optimization trajectories, method labels, examples, and transitions.
3. If no corpus exists, create a conservative seed graph using standard CUDA method nodes relevant to this operation.
4. Include v_init as the start node.
5. For each edge, include a prior score and one or more example snippets when available.

Return a graph suitable for Monte Carlo Graph Search.`, {
  label: 'build-regraph',
  phase: 'BuildGraph',
  schema: {
    type: 'object',
    properties: {
      graph: { type: 'object' },
      method_labels: { type: 'array', items: { type: 'string' } },
      trace_count: { type: 'number' },
      normalization_notes: { type: 'array', items: { type: 'string' } },
    },
    required: ['graph', 'method_labels'],
  },
})

if (graphResult?.graph?.nodes && graphResult?.graph?.edges) {
  graph = graphResult.graph
}

// =============================================================================
// Main Loop: Select -> Generate -> Evaluate -> UpdateGraph
// =============================================================================
for (let attempt = 0; attempt < BUDGET; attempt++) {
  log(`\n=== ReGraphT attempt ${attempt + 1}/${BUDGET} | best=${bestCandidate?.eval?.speedup || 0}x | graph=${graph.nodes.length} nodes/${graph.edges.length} edges ===`)

  phase('Select')

  const selection = await agent(`Select a promising CUDA optimization path with Monte Carlo Graph Search.

# ReGraph state
\`\`\`json
${JSON.stringify(graph, null, 2).substring(0, 10000)}
\`\`\`

# Selection constraints
- Start from v_init.
- Run ${ROLLOUTS_PER_SELECT} conceptual MCGS rollouts.
- Use UCT-style selection: average_reward + exploration + edge prior.
- Limit path length to ${MAX_PATH_LENGTH}.
- Prefer paths whose examples fit the current operation and target GPU.
- Do not select methods contradicted by evaluator failures in prior attempts.

# Prior evaluated candidates
\`\`\`json
${JSON.stringify(evaluatedCandidates.slice(-8), null, 2).substring(0, 8000)}
\`\`\`

Return the selected method path and the examples that should condition generation.`, {
    label: `select-${attempt}`,
    phase: 'Select',
    schema: {
      type: 'object',
      properties: {
        path_node_ids: { type: 'array', items: { type: 'string' } },
        method_sequence: { type: 'array', items: { type: 'string' } },
        selected_examples: { type: 'array', items: { type: 'object' } },
        selection_rationale: { type: 'string' },
        expected_speedup: { type: 'number' },
      },
      required: ['path_node_ids', 'method_sequence', 'selection_rationale'],
    },
  })

  selectedPaths.push(selection)

  phase('Generate')

  const generation = await agent(`Generate a CUDA optimization candidate using the selected ReGraphT path.

# Operation
${OP_DESC}

# Target GPU
${TARGET_GPU}

# Original/source code
\`\`\`cuda
${sourceCode.substring(0, 8000)}
\`\`\`

# Selected CUDA optimization method sequence
${(selection.method_sequence || []).map((item, i) => `${i + 1}. ${item}`).join('\n')}

# Retrieved optimization examples
\`\`\`json
${JSON.stringify(selection.selected_examples || [], null, 2).substring(0, 10000)}
\`\`\`

# Evaluator contract
${setupResult.evaluator_contract}

# Hard requirements
1. Use the selected method sequence only when it is suitable for the current code.
2. Produce complete compilable CUDA/C++ code, not a patch fragment.
3. Preserve correctness according to the input/output contract.
4. If a selected method is unsuitable, explain why and apply the next suitable method in the path.
5. Do not claim speedup without evaluator evidence.

Return candidate code and suitability decisions for each method.`, {
    label: `generate-${attempt}`,
    phase: 'Generate',
    schema: {
      type: 'object',
      properties: {
        candidate_code: { type: 'string' },
        applied_methods: { type: 'array', items: { type: 'string' } },
        skipped_methods: { type: 'array', items: { type: 'string' } },
        suitability_notes: { type: 'string' },
        implementation_notes: { type: 'string' },
      },
      required: ['candidate_code', 'applied_methods'],
    },
  })

  phase('Evaluate')

  const evaluation = await agent(`Evaluate the generated CUDA candidate with real evidence.

# Candidate code
\`\`\`cuda
${(generation.candidate_code || '').substring(0, 12000)}
\`\`\`

# Evaluation command
${EVAL_CMD || '(no benchmark_command provided)'}

# Result path convention
Use ${EXP_DIR}/regrapht_attempt_${attempt}.json as {result_path}.
Use ${EXP_DIR}/regrapht_attempt_${attempt}.cu as {kernel_path}.

# Required behavior
1. Materialize the candidate at the kernel path if command execution is available.
2. Run benchmark_command with {kernel_path} and {result_path} substitutions if provided.
3. Parse evaluator JSON. If no command is provided, mark correct=false and explain missing evidence.
4. Correctness and speedup must be based on evaluator output only.

Return evaluator evidence.`, {
    label: `evaluate-${attempt}`,
    phase: 'Evaluate',
    schema: {
      type: 'object',
      properties: {
        compiled: { type: 'boolean' },
        correct: { type: 'boolean' },
        speedup: { type: 'number' },
        kernel_time_ms: { type: 'number' },
        baseline_time_ms: { type: 'number' },
        result_path: { type: 'string' },
        error_message: { type: 'string' },
        error_type: { type: 'string' },
        evidence_summary: { type: 'string' },
      },
      required: ['compiled', 'correct', 'speedup'],
    },
  })

  const candidateRecord = {
    id: `attempt_${attempt}`,
    path_node_ids: selection.path_node_ids || [],
    methods: generation.applied_methods || [],
    code: generation.candidate_code || '',
    eval: evaluation,
  }
  evaluatedCandidates.push(candidateRecord)

  const measuredBest = bestMeasured()
  if (measuredBest && (!bestCandidate || (measuredBest.eval.speedup || 0) > (bestCandidate.eval.speedup || 0))) {
    bestCandidate = measuredBest
  }

  phase('UpdateGraph')

  const update = await agent(`Update the CUDA Reasoning Graph from measured evaluator feedback.

# Selected path
\`\`\`json
${JSON.stringify(selection, null, 2).substring(0, 6000)}
\`\`\`

# Generated methods
${(generation.applied_methods || []).join(' -> ') || '(none)'}

# Evaluator result
\`\`\`json
${JSON.stringify(evaluation, null, 2)}
\`\`\`

# Current graph
\`\`\`json
${JSON.stringify(graph, null, 2).substring(0, 10000)}
\`\`\`

# Update rules
1. Increment visits along selected nodes.
2. Add reward equal to speedup for correct candidates, small positive reward for compile-only progress, and zero/negative evidence for failures.
3. Attach the concrete optimization example to the traversed edge or final node.
4. If the candidate used a new method label, add or relabel a node instead of duplicating equivalent methods.
5. Add 1-3 follow-up edges from the best promising node when evidence suggests a next method.
6. If graph_path is provided, write the updated graph there.

Return the updated graph and update summary.`, {
    label: `update-graph-${attempt}`,
    phase: 'UpdateGraph',
    schema: {
      type: 'object',
      properties: {
        updated_graph: { type: 'object' },
        reward: { type: 'number' },
        added_nodes: { type: 'array', items: { type: 'string' } },
        added_edges: { type: 'array', items: { type: 'string' } },
        relabel_notes: { type: 'array', items: { type: 'string' } },
        wrote_graph_path: { type: 'string' },
      },
      required: ['updated_graph', 'reward'],
    },
  })

  if (update?.updated_graph?.nodes && update?.updated_graph?.edges) {
    graph = update.updated_graph
  }
}

// =============================================================================
// Phase 7: Report
// =============================================================================
phase('Report')

const finalGraphStats = graphStats()
const finalReport = await agent(`Write a concise technical report for this ReGraphT optimization run.

# Operation
${OP_DESC}

# Boundary
This workflow used ReGraphT as a training-free inference loop. It built or loaded a CUDA Reasoning Graph, selected paths with Monte Carlo Graph Search, generated candidates from graph-conditioned examples, and used evaluator evidence for reward.

# Adaptation scope
${ADAPTATION_SCOPE}

# Final graph stats
\`\`\`json
${JSON.stringify(finalGraphStats, null, 2)}
\`\`\`

# Best candidate
\`\`\`json
${JSON.stringify(bestCandidate ? {
  id: bestCandidate.id,
  methods: bestCandidate.methods,
  eval: bestCandidate.eval,
} : null, null, 2)}
\`\`\`

# Best code excerpt
\`\`\`cuda
${(bestCandidate?.code || '').substring(0, 5000)}
\`\`\`

# Selected paths
\`\`\`json
${JSON.stringify(selectedPaths.slice(-10), null, 2).substring(0, 10000)}
\`\`\`

Cover:
1. Which reasoning-graph methods were most useful.
2. Which methods were unsuitable or failed, based on evaluator evidence.
3. Whether the best candidate is trustworthy.
4. Which open graph paths should be tried next.`, {
  label: 'final-report',
  phase: 'Report',
})

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  kernel_path: SOURCE_CODE_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  operation: OP_DESC,
  baseline_metric: baselineMetric,
  best_speedup: bestCandidate?.eval?.speedup || 0,
  best_kernel_code: bestCandidate?.code || '',
  best_candidate_id: bestCandidate?.id || '',
  attempts: BUDGET,
  evaluated_candidates: evaluatedCandidates.length,
  correct_candidates: evaluatedCandidates.filter(item => item.eval?.correct).length,
  adaptation_scope: ADAPTATION_SCOPE,
  selected_paths: selectedPaths,
  graph,
  graph_stats: finalGraphStats,
  report: finalReport,
}
