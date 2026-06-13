export const meta = {
  name: 'gpuforecasters-kernel-optimization',
  description: 'Kernel optimization with learned speedup forecasting and PUCT search',
  whenToUse: 'Use for kernel optimization guided by learned performance forecasting models',
  phases: [
    { title: 'Setup', detail: 'Initialize forecasting models and search parameters' },
    { title: 'Train Forecasters', detail: 'Train surrogate models for speedup prediction' },
    { title: 'Calibration', detail: 'Calibrate abstention thresholds for forecasters' },
    { title: 'PUCT Search', detail: 'Tree search with forecaster-guided exploration' },
    { title: 'Refinement', detail: 'Refine promising candidates with focused search' },
    { title: 'Validation', detail: 'Validate final candidates on target hardware' },
    { title: 'Report', detail: 'Generate optimization report' },
  ],
  // Embedded-dispatch args (optional; default behavior is standalone). When
  // integration_pattern starts with "embedded", candidates are evaluated against
  // a project's register adapter instead of {kernel_path}/{result_path} substitution.
  args: [
    { name: 'integration_pattern', detail: 'standalone (default) | embedded[-*]. "embedded" routes evaluation through a register adapter.' },
    { name: 'register_script', detail: 'Path to a contract-conforming adapter (e.g. scripts/llamacpp_register_variant.py). Required when embedded.' },
    { name: 'project_root', detail: 'Project root the adapter wires the variant into. Alias: ggml_root. Required when embedded.' },
    { name: 'reference_cuh', detail: 'Reference dispatch .cuh whose signature the candidate must match exactly. Alias: reference_file.' },
    { name: 'register_params', detail: 'Opaque pass-through to the adapter (e.g. "--dkq 256 --dv 256 --cmake-build-dir /p/build").' },
  ],
};

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
  supported_languages: ['cuda', 'metal'],
  supported_problem_types: ['cuda-kernel-optimization', 'gpu-kernel-optimization', 'kernel-search'],
  problem_types: ['CUDA/GPU/Metal kernel search with speedup forecaster', 'PUCT optimization with execute-or-abstain feedback'],
  reason: 'GPU Forecasters expects GPU speedup evaluator feedback and a CUDA/Metal-oriented search/evaluation loop.',
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

const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const KERNEL_PATH = args.kernel_path || ''
const OP_DESC = args.op_description || args.operation || 'CUDA kernel'
const LANGUAGE = args.language || 'cuda'
const TARGET_GPU = args.target_gpu || ''
const TEST_CMD = args.test_command || ''
const BENCHMARK_CMD = args.benchmark_command || ''
const BUILD_CMD = args.build_command || ''
const USER_NOTE = args.note || args.notes || ''
// Embedded-dispatch mode: gate everything below behind EMBEDDED so the
// standalone path is byte-identical to before when integration_pattern is absent.
const INTEGRATION_PATTERN = (args.integration_pattern || 'standalone')
const EMBEDDED = INTEGRATION_PATTERN.startsWith('embedded')
const REGISTER_SCRIPT = args.register_script || ''
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const REFERENCE_FILE = args.reference_cuh || args.reference_file || ''
const REGISTER_PARAMS = args.register_params || ''  // opaque pass-through e.g. "--dkq 256 --dv 256 --cmake-build-dir /p/build"
const BASELINE_DESCRIPTION = args.baseline || args.baseline_description || args.baseline_notes || ''
const BASELINE_LATENCY_MS = args.baseline_latency_ms ?? args.baseline_perf_ms ?? args.baseline_time_ms ?? args.baseline_perf ?? ''
const BASELINE_RESULT_PATH = args.baseline_result_path || ''
const EXP_DIR = args.exp_dir || '/tmp/gpuforecasters_exp'
const REQUESTED_TRAINING_BUDGET = args.curriculum_size || args.training_budget || ''
const REQUESTED_GPU_BUDGET = args.gpu_budget || args.iterations || ''
const REQUESTED_PUCT_C = args.puct_c || args.puct_exploration_constant || ''
const REQUESTED_TREE_DEPTH = args.tree_depth || args.tree_depth_limit || ''
function assertEmbeddedArgs() {
  if (!EMBEDDED) return
  const missing = []
  if (!REGISTER_SCRIPT) missing.push('register_script')
  if (!PROJECT_ROOT) missing.push('project_root (or ggml_root)')
  if (!BUILD_CMD) missing.push('build_command')
  if (!TEST_CMD) missing.push('test_command')
  if (!BENCHMARK_CMD) missing.push('benchmark_command')
  if (missing.length) {
    throw new Error(
      `${meta.name}: integration_pattern="${INTEGRATION_PATTERN}" (embedded dispatch) requires ` +
      `the following non-empty args: ${missing.join(', ')}. ` +
      `Provide a contract-conforming register_script and the project's build/test/benchmark commands ` +
      `(see _substrate/embedded/ADAPTER_CONTRACT.md), or use integration_pattern="standalone".`
    )
  }
}

assertEmbeddedArgs()

const INPUT_MODE = KERNEL_PATH ? 'optimize_existing' : (PROBLEM_DEFINITION || PROBLEM_PATH || USER_NOTE ? 'generate_then_optimize' : 'unspecified_task')
const EVIDENCE_MODE = (TEST_CMD && BENCHMARK_CMD)
  ? 'measured_correctness_and_performance'
  : (TEST_CMD ? 'correctness_only' : (BENCHMARK_CMD ? 'benchmark_only' : 'conservative_missing_evidence'))

function formatContractValue(value) {
  if (value === undefined || value === null || value === '') return '(not provided)'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function positiveNumber(value) {
  const n = Number(value)
  return Number.isFinite(n) && n > 0 ? n : null
}

const PROVIDED_BASELINE_LATENCY_MS = positiveNumber(BASELINE_LATENCY_MS)

function taskContract() {
  return `# Task Contract
- input_mode: ${INPUT_MODE}
- problem_definition: ${formatContractValue(PROBLEM_DEFINITION)}
- problem_path: ${formatContractValue(PROBLEM_PATH)}
- kernel_path: ${formatContractValue(KERNEL_PATH)}
- op_description: ${formatContractValue(OP_DESC)}
- language: ${formatContractValue(LANGUAGE)}
- target_gpu: ${formatContractValue(TARGET_GPU)}
- exp_dir: ${EXP_DIR}
- user_note: ${formatContractValue(USER_NOTE)}

# Baseline Contract
- baseline: ${formatContractValue(BASELINE_DESCRIPTION)}
- baseline_latency_ms: ${formatContractValue(BASELINE_LATENCY_MS)}
- baseline_result_path: ${formatContractValue(BASELINE_RESULT_PATH)}

# Evidence Commands
- test_command: ${formatContractValue(TEST_CMD)}
- benchmark_command: ${formatContractValue(BENCHMARK_CMD)}
- evidence_mode: ${EVIDENCE_MODE}

# Evidence Rules
1. Treat user_note as authoritative task context. If it contains validation commands, baseline details, tolerances, or constraints, preserve and follow them.
2. When test_command is provided, run it exactly with {kernel_path} and {result_path} substitutions for every candidate that claims correctness.
3. When benchmark_command is provided, run it exactly with {kernel_path} and {result_path} substitutions before reporting measured latency or speedup.
4. Compute speedup against baseline_latency_ms when provided; otherwise obtain a measured baseline through benchmark_command before claiming measured speedup.
5. If a required command is missing or cannot run, mark measured evidence unavailable. Do not invent measured correctness, latency, or speedup.
6. Materialize generated candidates under exp_dir so evidence artifacts can be inspected.${EMBEDDED ? '\n\n' + embeddedEvidenceContract() : ''}`
}

// Sanitize a candidate/round identity into a valid adapter variant name.
function sanitizeVariant(name) {
  return String(name || 'cand').replace(/[^A-Za-z0-9_]/g, '_')
}

// Proposal-prompt appendix: how to AUTHOR an embedded-dispatch candidate.
function embeddedProposalContract() {
  return [
    EMBEDDING_CONTRACT,
    '',
    'REFERENCE DISPATCH FILE: ' + (REFERENCE_FILE || '(not provided)'),
    REFERENCE_FILE
      ? 'Read ' + REFERENCE_FILE + ' and match its dispatch signature EXACTLY ' +
        '(entry-point shape, template params, launch-bounds conventions). Emit a ' +
        'COMPLETE dispatch-compatible .cuh -- never a standalone translation unit, ' +
        'main(), or test harness.'
      : 'No reference file provided; still emit a COMPLETE dispatch-compatible .cuh ' +
        'matching the project dispatch signature, not a standalone translation unit.',
  ].join('\n')
}

// Evidence-prompt appendix: how to EVALUATE one embedded candidate via the adapter.
function embeddedEvidenceContract() {
  return [
    '# Embedded-Dispatch Evidence (integration_pattern=' + INTEGRATION_PATTERN + ')',
    'This kernel is NOT standalone. Do NOT use {kernel_path}/{result_path} substitution.',
    'Each candidate is a complete dispatch-compatible .cuh evaluated against the project',
    'register adapter using the workflow-provided embedded eval plan. For every candidate:',
    '1. Derive a unique variant name from the candidate/round identity (sanitized to [A-Za-z0-9_]).',
    '2. Build the plan via the workflow helper (adapter: python "' + REGISTER_SCRIPT + '", project-root: ' + PROJECT_ROOT + ').',
    '3. Run IN ORDER: plan.register, plan.list (confirm the variant is registered),',
    '   plan.build, plan.test (correctness), plan.benchmark (latency),',
    '   then ALWAYS plan.unregister (even on failure/non-improvement) and confirm via',
    '   plan.list that the variant is gone.',
    '4. HARD REQUIREMENT (cleanup invariant): on ANY failure or non-improvement you MUST',
    '   unregister and confirm removal, leaving the project byte-exact pristine.',
    '5. Parse correctness and latency ONLY from the command stdout, under the SAME',
    '   grounding/anti-fabrication rules above. Never invent measured correctness, latency,',
    '   or speedup; if a command is missing or fails, mark measured evidence unavailable.',
  ].join('\n')
}

// Build the ordered eval plan for one embedded candidate (reuses the inlined substrate).
// Used to render concrete, copy-runnable plan commands into the eval-phase prompts so the
// evaluating subagent runs the exact register/list/build/test/benchmark/unregister sequence.
function embeddedPlanFor(candidatePath, variantSeed) {
  return __embeddedEvalPlan({
    adapter: 'python "' + REGISTER_SCRIPT + '"',
    variant: sanitizeVariant(variantSeed),
    source: candidatePath,
    projectRoot: PROJECT_ROOT,
    params: REGISTER_PARAMS,
    buildCmd: BUILD_CMD,
    testCmd: TEST_CMD,
    benchmarkCmd: BENCHMARK_CMD,
  })
}

// GPU Forecasters: Kernel optimization with learned performance prediction
// Based on arXiv:2605.31464 (MIT)
// Implements surrogate models with abstention + PUCT tree search

async function main() {
  // ============================================================================
  // Phase 1: Setup
  // ============================================================================
  phase('Setup');

  const setupResult = await agent(
    `Set up GPU Forecasters optimization environment:

${taskContract()}

1. Identify target kernel and optimization space
2. Configure surrogate forecasting models:
   - Model types (MLP, Transformer, etc.)
   - Training budget
   - Abstention strategy (native vs calibrated)
3. Set up PUCT search parameters:
   - Exploration constant
   - Simulation budget
   - Tree depth limit
4. Prepare baseline implementation from kernel_path or generate an initial kernel if only problem context is provided
5. Configure execution backend (Modal, local GPU, etc.)
6. Preserve the evidence commands and baseline contract for all later phases${EMBEDDED ? `

# Embedded-Dispatch Authoring Contract
When you author or modify any kernel candidate (here or in later phases), follow this:
${embeddedProposalContract()}` : ''}

Return JSON:
{
  "kernel_name": "kernel name",
  "optimization_space": {
    "parameters": ["param1", "param2", ...],
    "search_space_size": <int>
  },
  "forecaster_models": ["model1", "model2", ...],
  "training_budget": <int>,
  "puct_exploration_constant": <float>,
  "puct_simulation_budget": <int>,
  "tree_depth_limit": <int>,
  "baseline_perf": <float>,
  "backend": "modal|local",
  "target_gpu": "A100|H100|V100|..."
}`,
    {
      label: 'Setup GPUForecasters',
      phase: 'Setup',
      schema: {
        type: 'object',
        properties: {
          kernel_name: { type: 'string' },
          optimization_space: { type: 'object' },
          forecaster_models: { type: 'array', items: { type: 'string' } },
          training_budget: { type: 'integer' },
          puct_exploration_constant: { type: 'number' },
          puct_simulation_budget: { type: 'integer' },
          tree_depth_limit: { type: 'integer' },
          baseline_perf: { type: 'number' },
          backend: { type: 'string' },
          target_gpu: { type: 'string' },
        },
        required: ['kernel_name', 'optimization_space', 'forecaster_models', 'baseline_perf'],
      },
    }
  );

  if (!setupResult) {
    log('Setup failed');
    return { success: false, reason: 'setup_failed' };
  }

  if (PROVIDED_BASELINE_LATENCY_MS !== null) {
    setupResult.baseline_perf = PROVIDED_BASELINE_LATENCY_MS
  }
  setupResult.target_gpu = TARGET_GPU || setupResult.target_gpu || 'NVIDIA GPU'

  log(`Optimizing ${setupResult.kernel_name} on ${setupResult.target_gpu}`);
  log(`Search space: ${setupResult.optimization_space.search_space_size} configurations`);
  log(`Forecaster models: ${setupResult.forecaster_models.join(', ')}`);

  const trainingBudget = REQUESTED_TRAINING_BUDGET || setupResult.training_budget || 100;
  const puctExploration = REQUESTED_PUCT_C || setupResult.puct_exploration_constant || 1.0;
  const puctSimulations = REQUESTED_GPU_BUDGET || setupResult.puct_simulation_budget || 500;
  const treeDepthLimit = REQUESTED_TREE_DEPTH || setupResult.tree_depth_limit || 10;

  // Track optimization history
  const executionLog = [];
  let bestConfig = null;
  let bestSpeedup = 1.0;

  // ============================================================================
  // Phase 2: Train Forecasters
  // ============================================================================
  phase('Train Forecasters');

  log(`Training surrogate models with budget ${trainingBudget} evaluations...`);

  const trainingResult = await agent(
    `Train surrogate forecasting models:

${taskContract()}

Target: ${setupResult.kernel_name}
Training budget: ${trainingBudget} kernel executions
Forecaster models: ${setupResult.forecaster_models.join(', ')}

Training process:
1. Sample initial configurations (random, LHS, Sobol)
2. Materialize each config under ${EXP_DIR}/training/${EMBEDDED ? `
   When authoring each candidate, follow the Embedded-Dispatch Authoring Contract above.` : ''}
3. ${EMBEDDED
    ? `For each candidate written at <candidatePath> with a unique sanitized <variantName>, build the eval plan and run, IN ORDER:
   - plan.register, then plan.list (confirm <variantName> is registered)
   - plan.build, then plan.test (correctness), then plan.benchmark (latency)
   - then ALWAYS plan.unregister and confirm via plan.list that <variantName> is gone (cleanup invariant).
   Concrete example for candidate "${EXP_DIR}/training/cand_001.cuh", variant "cand_001"
   (these exact command strings are produced by the workflow's embedded eval plan):
${(() => { const p = embeddedPlanFor(`${EXP_DIR}/training/cand_001.cuh`, 'cand_001'); return [
     '     register:   ' + p.register,
     '     list:       ' + p.list,
     '     build:      ' + p.build,
     '     test:       ' + p.test,
     '     benchmark:  ' + p.benchmark,
     '     unregister: ' + p.unregister,
   ].join('\n'); })()}
   Parse correctness from plan.test stdout and latency from plan.benchmark stdout, then measure speedup against the baseline contract.`
    : 'Run test_command before accepting correctness when provided'}
4. ${EMBEDDED
    ? 'On ANY failure or non-improvement still run plan.unregister and confirm removal so the project stays byte-exact pristine.'
    : `Run benchmark_command on ${setupResult.target_gpu} when provided and measure speedup against the baseline contract`}
5. Store evaluator JSON artifacts beside each candidate
6. Reject or label candidates with missing/failed correctness evidence
7. Collect training dataset: (config, speedup) pairs only from measured or explicitly labeled forecast-only evidence
8. Train each forecaster model:
   - Input: configuration vector plus kernel/code context
   - Output: predicted speedup + uncertainty estimate
9. Implement abstention mechanism:
   - Native abstention: model-intrinsic uncertainty (e.g., dropout variance)
   - Calibrated abstention: learned threshold based on prediction error
10. Validate on hold-out set

Return JSON:
{
  "training_samples": <int>,
  "training_dataset_size": <int>,
  "trained_models": [
    {
      "model_name": "model1",
      "architecture": "MLP|Transformer|...",
      "train_mae": <float>,
      "val_mae": <float>,
      "abstention_rate": <float>,
      "abstention_strategy": "native|calibrated"
    },
    ...
  ],
  "best_training_speedup": <float>,
  "best_training_config": "config description"
}`,
    {
      label: 'Train forecasters',
      phase: 'Train Forecasters',
      schema: {
        type: 'object',
        properties: {
          training_samples: { type: 'integer' },
          training_dataset_size: { type: 'integer' },
          trained_models: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                model_name: { type: 'string' },
                architecture: { type: 'string' },
                train_mae: { type: 'number' },
                val_mae: { type: 'number' },
                abstention_rate: { type: 'number' },
                abstention_strategy: { type: 'string' },
              },
              required: ['model_name', 'train_mae', 'abstention_rate'],
            },
          },
          best_training_speedup: { type: 'number' },
          best_training_config: { type: 'string' },
        },
        required: ['training_samples', 'trained_models', 'best_training_speedup'],
      },
    }
  );

  if (!trainingResult || trainingResult.trained_models.length === 0) {
    log('Forecaster training failed');
    return { success: false, reason: 'training_failed' };
  }

  log(`Trained ${trainingResult.trained_models.length} forecaster models`);
  for (const model of trainingResult.trained_models) {
    log(`  ${model.model_name}: MAE=${model.train_mae.toFixed(3)}, abstain=${(model.abstention_rate * 100).toFixed(1)}%`);
  }

  // Update best from training
  if (trainingResult.best_training_speedup > bestSpeedup) {
    bestSpeedup = trainingResult.best_training_speedup;
    bestConfig = trainingResult.best_training_config;
  }

  // ============================================================================
  // Phase 3: Calibration
  // ============================================================================
  phase('Calibration');

  log('Calibrating abstention thresholds...');

  const calibrationResult = await agent(
    `Calibrate abstention thresholds for forecasters:

${taskContract()}

Trained models: ${trainingResult.trained_models.map(m => m.model_name).join(', ')}

Calibration process:
1. Collect uncertainty estimates on validation set
2. Correlate uncertainty with prediction error
3. Find optimal abstention threshold:
   - Minimize: prediction error on non-abstained samples
   - Subject to: abstention rate ≤ target (e.g., 20%)
4. For ensemble: combine predictions when all agree, abstain if any abstains
5. Measure calibrated performance:
   - MAE on non-abstained predictions
   - Abstention rate
   - Coverage (1 - abstention_rate)

Return JSON:
{
  "calibrated_models": [
    {
      "model_name": "model1",
      "calibrated_threshold": <float>,
      "calibrated_mae": <float>,
      "calibrated_abstention_rate": <float>,
      "coverage": <float>
    },
    ...
  ],
  "ensemble_strategy": "unanimous|majority|weighted",
  "ensemble_mae": <float>,
  "ensemble_abstention_rate": <float>,
  "ensemble_coverage": <float>
}`,
    {
      label: 'Calibrate forecasters',
      phase: 'Calibration',
      schema: {
        type: 'object',
        properties: {
          calibrated_models: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                model_name: { type: 'string' },
                calibrated_threshold: { type: 'number' },
                calibrated_mae: { type: 'number' },
                calibrated_abstention_rate: { type: 'number' },
                coverage: { type: 'number' },
              },
              required: ['model_name', 'calibrated_mae', 'coverage'],
            },
          },
          ensemble_strategy: { type: 'string' },
          ensemble_mae: { type: 'number' },
          ensemble_abstention_rate: { type: 'number' },
          ensemble_coverage: { type: 'number' },
        },
        required: ['calibrated_models', 'ensemble_mae'],
      },
    }
  );

  if (!calibrationResult) {
    log('Calibration failed');
    return { success: false, reason: 'calibration_failed' };
  }

  log(`Ensemble calibrated: MAE=${calibrationResult.ensemble_mae.toFixed(3)}, coverage=${(calibrationResult.ensemble_coverage * 100).toFixed(1)}%`);

  // ============================================================================
  // Phase 4: PUCT Search
  // ============================================================================
  phase('PUCT Search');

  log(`Running PUCT tree search with ${puctSimulations} simulations...`);

  const puctResult = await agent(
    `Perform PUCT (Polynomial Upper Confidence Trees) search:

${taskContract()}

Search parameters:
- Exploration constant (c_puct): ${puctExploration}
- Simulation budget: ${puctSimulations}
- Tree depth limit: ${treeDepthLimit}

Forecaster ensemble:
- Models: ${calibrationResult.calibrated_models.map(m => m.model_name).join(', ')}
- Ensemble strategy: ${calibrationResult.ensemble_strategy}
- Abstention rate: ${(calibrationResult.ensemble_abstention_rate * 100).toFixed(1)}%

PUCT algorithm:
1. Initialize root node with baseline config
2. For each simulation:
   a. Selection: traverse tree using PUCT formula
      PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
      where:
      - Q(s,a) = mean reward (speedup) from (s,a)
      - P(s,a) = prior from forecaster
      - N(s) = visit count of state s
      - N(s,a) = visit count of (s,a)
   b. Expansion: expand node with forecaster-predicted actions
   c. Simulation:
${EMBEDDED
  ? `      - If forecaster abstains: materialize the config under ${EXP_DIR}/puct/ and execute the embedded eval plan:
        author a dispatch .cuh per the Authoring Contract, then run IN ORDER plan.register, plan.list, plan.build,
        plan.test, plan.benchmark, then ALWAYS plan.unregister + confirm removal via plan.list (cleanup invariant).
        Parse correctness from plan.test stdout and latency from plan.benchmark stdout.
      - Else: use forecaster prediction
      - If plan.test fails, treat the candidate as invalid regardless of predicted speedup
      - Use the measured plan.benchmark latency as the authoritative measured speedup`
  : `      - If forecaster abstains: materialize the config under ${EXP_DIR}/puct/ and execute user-provided evidence commands for ground truth
      - Else: use forecaster prediction
      - If test_command fails, treat the candidate as invalid regardless of predicted speedup
      - If benchmark_command is available, use its evaluator JSON as the authoritative measured speedup`}
   d. Backpropagation: update Q values along path
3. Return best config from tree (highest Q value)

Track:
- Total GPU executions (should be << simulation budget due to abstention)
- Tree statistics (depth, breadth, nodes explored)
- Best config found

Return JSON:
{
  "simulations": ${puctSimulations},
  "total_executions": <int>,
  "abstention_saved_executions": <int>,
  "tree_nodes_explored": <int>,
  "tree_max_depth": <int>,
  "best_config": "config description",
  "best_speedup": <float>,
  "search_trajectory": [
    {"step": <int>, "speedup": <float>, "executed": true/false},
    ...
  ]
}`,
    {
      label: 'PUCT search',
      phase: 'PUCT Search',
      schema: {
        type: 'object',
        properties: {
          simulations: { type: 'integer' },
          total_executions: { type: 'integer' },
          abstention_saved_executions: { type: 'integer' },
          tree_nodes_explored: { type: 'integer' },
          tree_max_depth: { type: 'integer' },
          best_config: { type: 'string' },
          best_speedup: { type: 'number' },
          search_trajectory: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                step: { type: 'integer' },
                speedup: { type: 'number' },
                executed: { type: 'boolean' },
              },
            },
          },
        },
        required: ['simulations', 'total_executions', 'best_speedup', 'best_config'],
      },
    }
  );

  if (!puctResult) {
    log('PUCT search failed');
    return { success: false, reason: 'puct_failed' };
  }

  log(`PUCT search complete: ${puctResult.total_executions} GPU executions (saved ${puctResult.abstention_saved_executions} via forecasters)`);
  log(`Best speedup: ${puctResult.best_speedup.toFixed(3)}x`);

  // Update best
  if (puctResult.best_speedup > bestSpeedup) {
    bestSpeedup = puctResult.best_speedup;
    bestConfig = puctResult.best_config;
  }

  executionLog.push(...(puctResult.search_trajectory || []));

  // ============================================================================
  // Phase 5: Refinement
  // ============================================================================
  phase('Refinement');

  log('Refining top candidates with local search...');

  const refinementResult = await agent(
    `Refine best configuration found:

${taskContract()}

Best config from PUCT: ${puctResult.best_config}
Best speedup: ${puctResult.best_speedup.toFixed(3)}x

Refinement strategies:
1. Local search around best config:
   - Grid search in neighborhood
   - Gradient-based refinement (if applicable)
2. Ablation studies:
   - Test impact of individual optimizations
   - Identify critical parameters
3. Multi-start local search from top-k PUCT nodes
4. Fine-grained parameter tuning

Execute promising refinements on GPU (use forecasters to filter).
Materialize each refinement under ${EXP_DIR}/refinement/ and use the evidence commands exactly when provided.${EMBEDDED ? `
For each refinement, author a dispatch .cuh per the Authoring Contract and evaluate it via the embedded eval
plan: run IN ORDER plan.register, plan.list, plan.build, plan.test, plan.benchmark, then ALWAYS plan.unregister
and confirm removal via plan.list (cleanup invariant). Parse correctness/latency from command stdout only.` : ''}
Only promote a refinement as measured if correctness passes and benchmark evidence is available.

Return JSON:
{
  "refinement_candidates": <int>,
  "refinement_executions": <int>,
  "best_refined_config": "config description",
  "best_refined_speedup": <float>,
  "improvement_over_puct": <float>,
  "ablation_insights": "brief insights"
}`,
    {
      label: 'Refine candidates',
      phase: 'Refinement',
      schema: {
        type: 'object',
        properties: {
          refinement_candidates: { type: 'integer' },
          refinement_executions: { type: 'integer' },
          best_refined_config: { type: 'string' },
          best_refined_speedup: { type: 'number' },
          improvement_over_puct: { type: 'number' },
          ablation_insights: { type: 'string' },
        },
        required: ['refinement_executions', 'best_refined_speedup', 'best_refined_config'],
      },
    }
  );

  if (!refinementResult) {
    log('Refinement failed, using PUCT result');
  } else {
    log(`Refinement complete: ${refinementResult.refinement_executions} additional executions`);
    log(`Best refined speedup: ${refinementResult.best_refined_speedup.toFixed(3)}x (${refinementResult.improvement_over_puct > 0 ? '+' : ''}${(refinementResult.improvement_over_puct * 100).toFixed(1)}%)`);

    // Update best
    if (refinementResult.best_refined_speedup > bestSpeedup) {
      bestSpeedup = refinementResult.best_refined_speedup;
      bestConfig = refinementResult.best_refined_config;
    }
  }

  // ============================================================================
  // Phase 6: Validation
  // ============================================================================
  phase('Validation');

  log('Validating best configuration...');

  const validationResult = await agent(
    `Validate best configuration:

${taskContract()}

Best config: ${bestConfig}
Best speedup: ${bestSpeedup.toFixed(3)}x

Validation:
1. Materialize the final candidate under ${EXP_DIR}/final/${EMBEDDED ? `
   Author it as a dispatch .cuh per the Authoring Contract, then evaluate via the embedded eval plan:
   run IN ORDER plan.register, plan.list, plan.build, plan.test, plan.benchmark, then ALWAYS plan.unregister
   and confirm removal via plan.list (cleanup invariant). Use plan.test for correctness and plan.benchmark
   stdout for latency; do NOT use {kernel_path}/{result_path} substitution.` : ''}
2. Run test_command exactly if provided and fail validation if correctness fails
3. Run benchmark_command exactly if provided; parse its JSON artifact as authoritative measured performance
4. Execute on target hardware (${setupResult.target_gpu}) multiple times when benchmark_command supports repeated runs
5. Measure performance statistics:
   - Mean speedup
   - Std dev
   - Min/max
6. Verify correctness (output matches baseline)
7. Profile hardware utilization:
   - SM occupancy
   - Memory bandwidth
   - Compute throughput
8. Test on different input sizes (if applicable)
9. Compare with baseline and other methods
10. If evidence_mode is conservative_missing_evidence, return validation_passed=false unless the user note explicitly authorizes static-only validation

Return JSON:
{
  "config": "${bestConfig}",
  "validation_runs": <int>,
  "mean_speedup": <float>,
  "std_speedup": <float>,
  "min_speedup": <float>,
  "max_speedup": <float>,
  "correctness_passed": true/false,
  "hardware_utilization": {
    "sm_occupancy_pct": <float>,
    "memory_bandwidth_pct": <float>,
    "compute_throughput_pct": <float>
  },
  "validation_passed": true/false
}`,
    {
      label: 'Validate best config',
      phase: 'Validation',
      schema: {
        type: 'object',
        properties: {
          config: { type: 'string' },
          validation_runs: { type: 'integer' },
          mean_speedup: { type: 'number' },
          std_speedup: { type: 'number' },
          min_speedup: { type: 'number' },
          max_speedup: { type: 'number' },
          correctness_passed: { type: 'boolean' },
          hardware_utilization: { type: 'object' },
          validation_passed: { type: 'boolean' },
        },
        required: ['mean_speedup', 'correctness_passed', 'validation_passed'],
      },
    }
  );

  if (!validationResult || !validationResult.validation_passed) {
    log('Validation failed');
    return {
      success: false,
      reason: 'validation_failed',
      best_config: bestConfig,
      best_speedup: bestSpeedup,
      input_mode: INPUT_MODE,
      evidence_mode: EVIDENCE_MODE,
      test_command: TEST_CMD,
      benchmark_command: BENCHMARK_CMD,
      baseline_latency_ms: setupResult.baseline_perf,
      exp_dir: EXP_DIR,
    };
  }

  log(`Validation passed: ${validationResult.mean_speedup.toFixed(3)}x ± ${validationResult.std_speedup.toFixed(3)}x`);

  // ============================================================================
  // Phase 7: Report
  // ============================================================================
  phase('Report');

  const report = await agent(
    `Generate GPU Forecasters optimization report:

${taskContract()}

Summary:
- Kernel: ${setupResult.kernel_name}
- Target GPU: ${setupResult.target_gpu}
- Search space: ${setupResult.optimization_space.search_space_size} configs
- Training budget: ${trainingResult.training_samples} samples
- PUCT simulations: ${puctResult.simulations}
- Total GPU executions: ${trainingResult.training_samples + puctResult.total_executions + (refinementResult?.refinement_executions || 0)}
- Executions saved by forecasters: ${puctResult.abstention_saved_executions}

Results:
- Baseline: ${setupResult.baseline_perf.toFixed(3)} ms
- Best speedup: ${validationResult.mean_speedup.toFixed(3)}x ± ${validationResult.std_speedup.toFixed(3)}x
- Best config: ${validationResult.config}

Forecaster performance:
${calibrationResult.calibrated_models.map(m => `  ${m.model_name}: MAE=${m.calibrated_mae.toFixed(3)}, coverage=${(m.coverage * 100).toFixed(1)}%`).join('\n')}

Generate report with:
1. Executive summary
2. Search efficiency analysis (executions saved)
3. Forecaster accuracy and calibration
4. PUCT search trajectory visualization
5. Best configuration analysis
6. Hardware utilization breakdown
7. Comparison with baselines

Return JSON:
{
  "summary": "brief summary",
  "best_speedup": ${validationResult.mean_speedup},
  "total_executions": <int>,
  "executions_saved": ${puctResult.abstention_saved_executions},
  "search_efficiency": <float>,
  "report_path": "path/to/report.md"
}`,
    {
      label: 'Generate report',
      phase: 'Report',
      schema: {
        type: 'object',
        properties: {
          summary: { type: 'string' },
          best_speedup: { type: 'number' },
          total_executions: { type: 'integer' },
          executions_saved: { type: 'integer' },
          search_efficiency: { type: 'number' },
          report_path: { type: 'string' },
        },
        required: ['summary', 'best_speedup', 'total_executions'],
      },
    }
  );

  // ============================================================================
  // Return final results
  // ============================================================================

  return {
    success: true,
    method: 'GPU Forecasters',
    approach: 'Learned speedup forecasting + PUCT search',
    input_mode: INPUT_MODE,
    evidence_mode: EVIDENCE_MODE,
    kernel: setupResult.kernel_name,
    target_gpu: setupResult.target_gpu,
    exp_dir: EXP_DIR,
    test_command: TEST_CMD,
    benchmark_command: BENCHMARK_CMD,
    user_note_present: !!USER_NOTE,
    search_space_size: setupResult.optimization_space.search_space_size,
    training_budget: trainingResult.training_samples,
    forecaster_models: trainingResult.trained_models.map(m => m.model_name),
    ensemble_mae: calibrationResult.ensemble_mae,
    ensemble_coverage: calibrationResult.ensemble_coverage,
    puct_simulations: puctResult.simulations,
    total_executions: trainingResult.training_samples + puctResult.total_executions + (refinementResult?.refinement_executions || 0),
    executions_saved: puctResult.abstention_saved_executions,
    baseline_perf: setupResult.baseline_perf,
    best_speedup: validationResult.mean_speedup,
    speedup_std: validationResult.std_speedup,
    best_config: validationResult.config,
    hardware_utilization: validationResult.hardware_utilization,
    validation_passed: validationResult.validation_passed,
    report: report?.report_path,
    summary: report?.summary,
  };
}

// Execute the workflow
return await main();
