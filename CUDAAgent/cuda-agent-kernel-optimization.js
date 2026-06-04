export const meta = {
  name: 'cuda-agent-kernel-optimization',
  description: 'Skill-integrated multi-turn CUDA kernel optimization with profiling-driven iterative refinement (CUDA Agent methodology)',
  whenToUse: 'When optimizing CUDA kernels from PyTorch operator specifications through iterative code generation, compilation, correctness testing, and profiling feedback. Follows the CUDA Agent skill-based workflow: profile baseline → identify bottlenecks → implement kernel + bindings → compile → verify correctness → measure speedup → refine until target met.',
  phases: [
    { title: 'Setup', detail: 'Read PyTorch model, profile baseline (eager + compile), establish workspace' },
    { title: 'Profile', detail: 'Analyze native PyTorch performance, identify bottlenecks and optimization opportunities' },
    { title: 'Implement', detail: 'Generate CUDA kernel, bindings, and model_new.py with custom operators' },
    { title: 'Verify', detail: 'Compile kernel, run correctness tests against reference, measure performance' },
    { title: 'Refine', detail: 'Iteratively fix errors and optimize based on compilation/runtime/profiling feedback' },
    { title: 'Report', detail: 'Final performance comparison and optimization summary' },
  ],
}

const WORKFLOW_SUITABILITY = {
  supported_languages: ['cuda'],
  supported_problem_types: ['cuda-kernel-generation', 'cuda-kernel-optimization'],
  problem_types: ['PyTorch model/operator to custom CUDA ops', 'CUDA inference-time profile/implement/verify/refine loop'],
  reason: 'CUDA Agent in this repo targets CUDA custom operators and CUDA profiling/evaluation feedback.',
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
// CUDA Agent: Skill-Integrated Multi-Turn Kernel Optimization
// =============================================================================
//
// Source: "CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation"
//         Dai, Wu, Yu, Gao, Li, Jiang, Lou, Song, Yu, Chen, Ma, Zhang, Liu, Wang, Liu, Zhou
//         ByteDance Seed / Tsinghua AIR, arXiv:2602.24286, 2026
//         https://cuda-agent.github.io/
//
// The CUDA Agent paper introduces a training system (agentic RL with PPO), but its
// inference-time agent loop (Section 3.2, Figure 2) defines a reusable workflow:
// adaptation_scope: inference_time_adaptation — this workflow does not reproduce
// dataset construction, PPO training, or model-weight updates from the full paper.
//
// SKILL.md (CUDA Coding Skill):
//   1. Analyze native PyTorch performance using profile.py
//      → identify bottlenecks (excessive kernel launches, suboptimal memory access)
//   2. Implement custom CUDA operators in model_new.py with kernel source + bindings
//      → target performance-critical operators identified in step 1
//   3. Compile and evaluate in GPU sandbox
//      → iteratively refine until correctness AND performance requirements met
//   4. Repeat from step 2 until ≥5% speedup over torch.compile achieved
//
// Workspace structure:
//   kernels/kernel.cu        — CUDA kernel source
//   kernels/kernel_binding.cpp — pybind11 bindings
//   model_new.py             — PyTorch model using custom CUDA ops
//   model.py                 — Original PyTorch model (reference)
//   verify.py                — Correctness verification script
//   profile.py               — Performance profiling script
//
// Reward signal (robust reward schedule):
//   r = -1 if correctness fails
//   r = 3  if faster than BOTH eager AND compile baselines (>5%)
//   r = 2  if faster than eager baseline only (>5%)
//   r = 1  otherwise (correct but not faster)
//
// Usage:
//   Workflow({name: 'cuda-agent-kernel-optimization', args: {
//     kernel_path: '/path/to/model.py',
//     op_description: 'Fused SwiGLU + Linear projection',
//     test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
//     profile_command: '<user-provided profiling command with {kernel_path}/{result_path}>',
//     compile_command: '<user-provided compile command with {kernel_path}/{result_path}>',
//     target_speedup: 1.05,
//     max_turns: 20,
//     exp_dir: '/tmp/cuda_agent_exp',
//   }})
//
// =============================================================================

// --- Required Args ---
let MODEL_PATH = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const INPUT_MODE = MODEL_PATH ? 'optimize_existing' : 'generate_then_optimize'
const OP_DESC = args.op_description || 'PyTorch model'

// --- Optional Args ---
const VERIFY_CMD = args.test_command || ''
const PROFILE_CMD = args.profile_command || ''
const COMPILE_CMD = args.compile_command || ''
const TARGET_SPEEDUP = args.target_speedup || 1.05
const MAX_TURNS = args.max_turns || 15
const EXP_DIR = args.exp_dir || '/tmp/cuda_agent_exp'
const ADAPTATION_SCOPE = 'inference_time_adaptation'
const LANGUAGE = args.language || 'cuda'
const TARGET_GPU = args.target_gpu || 'unknown GPU'
const SEED_CANDIDATES = args.seed_candidates || 3

if (!MODEL_PATH && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

// --- State ---
let modelCode = ''
let eagerTime = 0
let compileTime = 0
let bestKernelCode = ''
let bestBindingCode = ''
let bestModelNew = ''
let bestSpeedup = 0
let currentAttempt = 0
let generatedKernelPath = ''
let initialCandidates = []
let initialGenerationResult = null
let history = []  // [{turn, action, outcome, speedup, error}]

// =============================================================================
// Phase 1: Setup — Read model, establish workspace
// =============================================================================
phase('Setup')

if (INPUT_MODE === 'generate_then_optimize') {
  const generated = await agent(`No kernel_path was provided. Generate and verify an initial PyTorch model plus CUDA kernel scaffold before CUDAAgent optimization.

# Problem Input
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- op_description: ${OP_DESC}
- language: ${LANGUAGE}
- target_gpu: ${TARGET_GPU}
- seed_candidates: ${SEED_CANDIDATES}

# Evidence Commands
- compile_command: ${COMPILE_CMD || '(not provided)'}
- test_command: ${VERIFY_CMD || '(not provided)'}
- profile_command: ${PROFILE_CMD || '(not provided)'}

# Contract
Generate ${SEED_CANDIDATES} complete candidates under ${EXP_DIR}/generated/. Run available commands using {kernel_path}/{result_path}. Return the best verified generated source path.`, {
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
  if ((VERIFY_CMD || PROFILE_CMD) && initialGenerationResult.verified === false) throw new Error('No generated seed passed verification evidence')
  MODEL_PATH = generatedKernelPath
}

const setupResult = await agent(`You are a CUDA kernel optimization expert. Set up the optimization workspace.

# Task:
1. Read the PyTorch model from: ${MODEL_PATH}
2. Create workspace: mkdir -p ${EXP_DIR}/{kernels,profiles}
3. Analyze the model to identify:
   - What operations it performs
   - Which operators are performance-critical
   - Data types and tensor shapes involved
   - Opportunities for kernel fusion

# Operation: ${OP_DESC}

Return the model code and analysis.`, {
  label: 'setup-workspace',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      model_code: { type: 'string' },
      operator_list: { type: 'array', items: { type: 'string' } },
      critical_operators: { type: 'array', items: { type: 'string' } },
      fusion_opportunities: { type: 'array', items: { type: 'string' } },
      data_types: { type: 'string' },
      tensor_shapes: { type: 'string' },
    },
    required: ['model_code', 'critical_operators'],
  },
})

modelCode = setupResult.model_code

// =============================================================================
// Phase 2: Profile — Analyze baseline performance
// =============================================================================
phase('Profile')

const profileResult = await agent(`You are a CUDA performance profiler. Profile the baseline PyTorch model.

# Model Code:
\`\`\`python
${modelCode.substring(0, 4000)}
\`\`\`

# Profiling Tasks (CUDA Agent SKILL.md Step 1):
1. Profile the NATIVE PyTorch implementation:
${PROFILE_CMD ? `   Run: ${PROFILE_CMD}` : '   Estimate performance from operator analysis.'}

2. Measure:
   - PyTorch Eager mode execution time
   - torch.compile execution time (the baseline to beat)
   - Per-operator breakdown if available

3. Identify performance bottlenecks:
   - Excessive kernel launches (multiple small kernels instead of one fused)
   - Suboptimal memory access patterns
   - Redundant data movement (intermediate materializations)
   - Opportunities for operator fusion

4. Determine optimization strategy:
   - Which operators to fuse into a single CUDA kernel?
   - What memory access pattern to use?
   - What parallelism strategy (threads/blocks mapping)?

Return profiling results and optimization plan.`, {
  label: 'profile-baseline',
  phase: 'Profile',
  schema: {
    type: 'object',
    properties: {
      eager_time_ms: { type: 'number' },
      compile_time_ms: { type: 'number' },
      per_operator_breakdown: { type: 'array', items: { type: 'object', properties: { op: { type: 'string' }, time_ms: { type: 'number' } } } },
      bottlenecks: { type: 'array', items: { type: 'string' } },
      optimization_strategy: { type: 'string' },
      fusion_plan: { type: 'string' },
    },
    required: ['eager_time_ms', 'compile_time_ms', 'bottlenecks', 'optimization_strategy'],
  },
})

eagerTime = profileResult.eager_time_ms || 1.0
compileTime = profileResult.compile_time_ms || 1.0

log(`Baseline: eager=${eagerTime}ms, compile=${compileTime}ms | Bottlenecks: ${profileResult.bottlenecks.join(', ')}`)
log(`Strategy: ${profileResult.optimization_strategy}`)

// =============================================================================
// Iterative Refinement Loop (SKILL.md Steps 2-4)
// =============================================================================

let targetMet = false

for (currentAttempt = 0; currentAttempt < MAX_TURNS && !targetMet; currentAttempt++) {

  // ===========================================================================
  // Phase 3: Implement — Generate CUDA kernel + bindings + model_new
  // ===========================================================================
  phase('Implement')

  const recentHistory = history.slice(-5)
  const historyContext = recentHistory.length > 0
    ? `\n# Previous Attempts:\n${recentHistory.map(h => `Turn ${h.turn}: ${h.action} → ${h.outcome}${h.error ? ' (' + h.error.substring(0, 100) + ')' : ''} ${h.speedup ? h.speedup.toFixed(2) + 'x' : ''}`).join('\n')}`
    : ''

  const implResult = await agent(`You are a CUDA kernel developer. Implement an optimized CUDA kernel for this PyTorch model.

# Model to Optimize:
\`\`\`python
${modelCode.substring(0, 3000)}
\`\`\`

# Operation: ${OP_DESC}
# Optimization Strategy: ${profileResult.optimization_strategy}
# Fusion Plan: ${profileResult.fusion_plan || 'Fuse critical operators'}
# Bottlenecks to Address: ${profileResult.bottlenecks.join('; ')}

# Baseline Performance:
- Eager: ${eagerTime}ms
- torch.compile: ${compileTime}ms
- Target: >${TARGET_SPEEDUP}x speedup over torch.compile (=${(compileTime / TARGET_SPEEDUP).toFixed(3)}ms)
${historyContext}

# CUDA Agent Workspace Requirements:
Generate THREE files:

## 1. kernels/kernel.cu — CUDA kernel source
- Optimized __global__ kernel function(s)
- Proper thread/block mapping
- Memory coalescing, shared memory usage where beneficial
- Error checking

## 2. kernels/kernel_binding.cpp — pybind11 bindings
- Expose kernel launch wrapper to Python
- Proper tensor type checking
- PYBIND11_MODULE declaration

## 3. model_new.py — PyTorch model using custom CUDA ops
- Import compiled extension
- Replace performance-critical operators with custom CUDA calls
- Maintain same interface as original model.py

# Key Rules:
- Performance gains must come SOLELY from the generated CUDA kernel
- Do NOT use torch.nn.functional fallbacks
- Ensure numerical correctness (match reference within tolerance)
- This is attempt ${currentAttempt + 1}/${MAX_TURNS}

Return all three files.`, {
    label: `impl-${currentAttempt}`,
    phase: 'Implement',
    schema: {
      type: 'object',
      properties: {
        kernel_code: { type: 'string' },
        binding_code: { type: 'string' },
        model_new_code: { type: 'string' },
        implementation_notes: { type: 'string' },
      },
      required: ['kernel_code', 'binding_code', 'model_new_code'],
    },
  })

  // ===========================================================================
  // Phase 4: Verify — Compile + correctness + performance
  // ===========================================================================
  phase('Verify')

  const verifyResult = await agent(`You are a CUDA kernel validator. Compile, verify, and benchmark this kernel implementation.

# Kernel Code (kernel.cu):
\`\`\`cuda
${implResult.kernel_code.substring(0, 4000)}
\`\`\`

# Binding Code (kernel_binding.cpp):
\`\`\`cpp
${implResult.binding_code.substring(0, 2000)}
\`\`\`

# Model New (model_new.py):
\`\`\`python
${implResult.model_new_code.substring(0, 2000)}
\`\`\`

# Validation Steps:

## Step 1: Compile
${COMPILE_CMD ? `Run: ${COMPILE_CMD}` : 'No compile_command provided; perform static compileability review only.'}
Check for compilation errors.

## Step 2: Correctness Verification
${VERIFY_CMD ? `Run: ${VERIFY_CMD}` : 'No test_command provided; do not invent a correctness test command. Describe the required reference comparison and mark measured correctness as unavailable.'}
- Use tolerance: atol=1e-3, rtol=1e-3
- Test with multiple input shapes if applicable

## Step 3: Performance Measurement
${PROFILE_CMD ? `Run exactly the user-provided profile_command: ${PROFILE_CMD}` : 'No profile_command provided; do not invent a performance measurement command.'}
- Warm-up iterations: 10
- Measurement iterations: 100
- Report: kernel_time, speedup_vs_eager, speedup_vs_compile

## Step 4: Compute Reward (CUDA Agent reward schedule)
- r = -1 if correctness fails
- r = 3 if faster than BOTH eager(${eagerTime}ms) AND compile(${compileTime}ms) by >5%
- r = 2 if faster than eager only by >5%
- r = 1 if correct but not faster

Return results.`, {
    label: `verify-${currentAttempt}`,
    phase: 'Verify',
    schema: {
      type: 'object',
      properties: {
        compiled: { type: 'boolean' },
        compile_error: { type: 'string' },
        correct: { type: 'boolean' },
        correctness_error: { type: 'string' },
        kernel_time_ms: { type: 'number' },
        speedup_vs_eager: { type: 'number' },
        speedup_vs_compile: { type: 'number' },
        reward: { type: 'number' },
        performance_notes: { type: 'string' },
      },
      required: ['compiled', 'correct', 'reward'],
    },
  })

  // Record history
  let outcome = ''
  let error = ''
  if (!verifyResult.compiled) {
    outcome = 'compile_error'
    error = verifyResult.compile_error || ''
  } else if (!verifyResult.correct) {
    outcome = 'incorrect'
    error = verifyResult.correctness_error || ''
  } else {
    outcome = `correct (${verifyResult.speedup_vs_compile?.toFixed(2) || '?'}x vs compile)`
  }

  history.push({
    turn: currentAttempt,
    action: implResult.implementation_notes?.substring(0, 50) || 'kernel implementation',
    outcome: outcome,
    speedup: verifyResult.speedup_vs_compile || 0,
    error: error,
    reward: verifyResult.reward,
  })

  // Update best
  if (verifyResult.correct && (verifyResult.speedup_vs_compile || 0) > bestSpeedup) {
    bestKernelCode = implResult.kernel_code
    bestBindingCode = implResult.binding_code
    bestModelNew = implResult.model_new_code
    bestSpeedup = verifyResult.speedup_vs_compile || 0
    log(`  NEW BEST: ${bestSpeedup.toFixed(2)}x vs compile (reward=${verifyResult.reward})`)
  }

  // Check if target met
  if (verifyResult.correct && (verifyResult.speedup_vs_compile || 0) >= TARGET_SPEEDUP) {
    targetMet = true
    log(`  TARGET MET: ${verifyResult.speedup_vs_compile?.toFixed(2)}x ≥ ${TARGET_SPEEDUP}x`)
  } else {
    // ===========================================================================
    // Phase 5: Refine — Diagnose and plan fix
    // ===========================================================================
    phase('Refine')

    if (!targetMet && currentAttempt < MAX_TURNS - 1) {
      log(`  Turn ${currentAttempt + 1}: ${outcome} | Refining...`)
    }
  }
}

// =============================================================================
// Phase 6: Report
// =============================================================================
phase('Report')

const finalReport = await agent(`Write a concise optimization report.

# CUDA Agent Optimization Results
- Adaptation scope: ${ADAPTATION_SCOPE}
- Operation: ${OP_DESC}
- Baseline eager: ${eagerTime}ms
- Baseline compile: ${compileTime}ms
- Best kernel time: ${compileTime / (bestSpeedup || 1)}ms
- Best speedup vs compile: ${bestSpeedup.toFixed(2)}x
- Target: ${TARGET_SPEEDUP}x | ${targetMet ? 'ACHIEVED' : 'NOT MET'}
- Turns used: ${currentAttempt}/${MAX_TURNS}

# Optimization History:
${history.map(h => `Turn ${h.turn + 1}: ${h.outcome} (reward=${h.reward})`).join('\n')}

# Best Kernel:
\`\`\`cuda
${bestKernelCode.substring(0, 3000)}
\`\`\`

Write:
1. What optimization strategy worked
2. Key challenges encountered (compile errors, correctness issues)
3. Performance breakdown (where the speedup comes from)
4. Remaining optimization opportunities`, {
  label: 'final-report',
  phase: 'Report',
})

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: generatedKernelPath,
  initial_candidates: initialCandidates,
  initial_generation_result: initialGenerationResult,
  operation: OP_DESC,
  eager_time_ms: eagerTime,
  compile_time_ms: compileTime,
  best_speedup_vs_compile: bestSpeedup,
  best_speedup_vs_eager: eagerTime / (compileTime / (bestSpeedup || 1)),
  target_met: targetMet,
  turns_used: currentAttempt,
  max_turns: MAX_TURNS,
  reward_history: history.map(h => h.reward),
  adaptation_scope: ADAPTATION_SCOPE,
  best_kernel_code: bestKernelCode,
  best_binding_code: bestBindingCode,
  best_model_new: bestModelNew,
  report: finalReport,
}
