export const meta = {
  name: 'kernelagent-triton-synthesis',
  description: 'Multi-agent Triton kernel synthesis with parallel seed generation, iterative refinement, and strict verification (KernelAgent methodology)',
  whenToUse: 'When you need to generate verified Triton kernels from a PyTorch problem description or code. Uses parallel workers for diverse seed generation, sandboxed test execution for verification, and iterative LLM-guided refinement. Supports both direct single-kernel and Fuser pipeline (multi-subgraph) paths.',
  phases: [
    { title: 'Setup', detail: 'Parse problem, generate test harness, initialize workspace' },
    { title: 'Route', detail: 'Static analysis to choose direct path vs Fuser pipeline' },
    { title: 'Generate', detail: 'Parallel seed generation: N agents each produce a candidate Triton kernel' },
    { title: 'Verify', detail: 'Sandboxed test execution for each candidate in parallel' },
    { title: 'Refine', detail: 'Iterative refinement: failed candidates get LLM-guided fixes' },
    { title: 'Compose', detail: 'Stitch verified subgraph kernels into final Triton program (pipeline path only)' },
    { title: 'Report', detail: 'Final summary with best kernel, verification status, artifacts' },
  ],
}

// __modelTierApplied (declaration pre-existing)

const WORKFLOW_NAME = 'kernelagent-triton-synthesis'


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

// --- BEGIN inlined agent-retry scaffolding (from _meta/scaffolding/agent-retry.js) ---
async function agentRetry(fn, opts) {
  const retries = (opts && opts.retries != null) ? opts.retries : 5
  let lastError = null
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const result = await fn()
      if (result != null) return result
      // null = agent skipped mid-run OR terminal subagent failure (e.g. transient 429) — retry.
    } catch (e) {
      lastError = e
    }
  }
  if (lastError) throw lastError
  // All attempts returned null (agent skipped mid-run OR a terminal subagent
  // failure such as a sustained 429). FAIL-SAFE DEFAULT: throw an attributable
  // error instead of returning null. A null return would later hit an unguarded
  // deref (`diag.bottleneck_class`, `impl.code`, ...) and crash the run with a
  // cryptic TypeError — issue #20. Throwing here makes the round abort cleanly
  // with a recorded reason, and inside `parallel()` a throwing thunk simply
  // resolves to a null slot that `.filter(Boolean)` drops (graceful). Callers
  // that INTENTIONALLY degrade on a missing result opt out with `{ allowNull: true }`.
  if (opts && opts.allowNull === true) return null
  throw new Error(
    `agentRetry: "${(opts && opts.label) || 'agent'}" returned null after ${retries + 1} attempt(s) ` +
    `(agent skipped or terminal API failure after retries).`,
  )
}

/**
 * Null-guard a REQUIRED structured field. Throws a clear, attributable error
 * (instead of a cryptic TypeError) when an agent returned null/malformed output,
 * so the run fails loudly at the dereference rather than producing garbage.
 */
function expect(obj, field, ctx) {
  if (obj == null || obj[field] == null) {
    throw new Error(
      `agentRetry: required field "${field}" is missing${ctx ? ' from ' + ctx : ''} ` +
      `(agent returned null or a malformed result after retries).`,
    )
  }
  return obj[field]
}

/**
 * Null-guard an OPTIONAL structured field with a fallback (no throw).
 * Use for deref points that have a sensible default (e.g. `[]`, `''`, `0`).
 */
function guard(obj, field, fallback) {
  if (obj == null || obj[field] == null) return fallback
  return obj[field]
}
// --- END inlined agent-retry scaffolding ---
// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.

const WORKFLOW_SUITABILITY = {
  supported_languages: ['triton'],
  supported_problem_types: ['triton-kernel-generation', 'operator-generation'],
  problem_types: ['Triton kernel synthesis from PyTorch-style operator tasks', 'parallel verification and refinement'],
  reason: 'KernelAgent targets Triton synthesis and verification; it is not a CUDA/CUTLASS/SYCL workflow.',
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
// KernelAgent — Multi-Agent Triton Kernel Synthesis Workflow
// =============================================================================
//
// Implements the KernelAgent methodology (PyTorch Labs, Apache-2.0):
//   Static Analysis → Route → Parallel Generate → Verify → Refine → Compose
//
// Based on the PyTorch KernelFalcon blog and KernelAgent codebase:
//   https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/
//
// Key innovations:
//   1. Auto-routing: static complexity analysis chooses direct vs pipeline path
//   2. Parallel seed generation: N diverse initializations via temperature variation
//   3. Strict verification: sandboxed subprocess execution, no PyTorch fallbacks
//   4. Iterative refinement: error-driven LLM repair up to iterations
//   5. Composable pipeline: extract subgraphs → dispatch → compose
//
// Usage:
//   Workflow({name: 'kernelagent-triton-synthesis', args: {
//     problem_path: '/path/to/problem.py',
//     problem_definition: 'Implement ReLU over a contiguous 1D tensor of length 1024',
//     num_workers: 4,
//     iterations: 10,
//     model_name: 'gpt-5',
//     verify: true,
//     compose: true,
//     exp_dir: '/path/to/experiment/output',
//   }})
//
// =============================================================================

// --- Required Args ---
const PROBLEM_PATH = args.problem_path || ''
const PROBLEM_DESC = args.problem_definition || ''

// --- Optional Args ---
const NUM_WORKERS = args.num_workers || 4
const MAX_ROUNDS = args.iterations || 10
const MODEL_NAME = args.model_name || 'claude-sonnet-4-20250514'
const VERIFY = args.verify !== false
const COMPOSE = args.compose !== false
const EXP_DIR = args.exp_dir || '/tmp/kernelagent_exp'
const TEMPERATURE_BASE = args.temperature_base || 0.8
const MAX_SEEDS = args.seed_candidates || NUM_WORKERS
const FUSER_EXTRACT_MODEL = args.fuser_extract_model || MODEL_NAME
const FUSER_DISPATCH_MODEL = args.fuser_dispatch_model || MODEL_NAME
const FUSER_COMPOSE_MODEL = args.fuser_compose_model || MODEL_NAME

// --- Model routing (additive harness wiring) ---
const MODEL = {
  mechanical: args.model_mechanical || 'haiku',  // runs shell/scripts, sandboxed test exec, parses output, bookkeeping
  profile: args.model_profile || 'sonnet',       // routing/complexity analysis
  judgment: args.model_judgment || 'opus',       // generating or editing Triton kernel code, refinement/debug, composition, report
}
const EST_PER_ROUND = args.est_tokens_per_round || 60000

// --- Backend driver wiring (P5b Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_ROUTE_LANG_TOKEN = 'Triton'
const LEGACY_HARNESS_LANG_TOKEN = 'Triton'
const LEGACY_SYNTH_LANG_TOKEN = 'Triton'
const LEGACY_AGGREGATE_LANG_TOKEN = 'Triton'
const LEGACY_VERIFY_LANG_TOKEN = 'Triton'
const LEGACY_REFINE_LANG_TOKEN = 'Triton'
const LEGACY_COMPOSE_LANG_TOKEN = 'Triton'
const LEGACY_KERNEL_FILENAME = 'kernel.py'
const LEGACY_TEST_FILENAME = 'test_kernel.py'
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}

let DRIVER = null
let DRIVER_LANG_FENCE = 'triton'
let DRIVER_IMPL_REQUIREMENTS = ''
let DRIVER_SOURCE_EXT = '.py'
let DRIVER_AUX_EXT = '.py'
let DRIVER_BACKEND_ID = RESOLVED_BACKEND || ''

function langToken(legacy) {
  return USE_DRIVER ? DRIVER_LANG_FENCE : legacy
}
function kernelFilename() {
  return USE_DRIVER ? `kernel${DRIVER_SOURCE_EXT}` : (args.kernel_filename || LEGACY_KERNEL_FILENAME)
}
function testFilename() {
  return USE_DRIVER ? `test_kernel${DRIVER_AUX_EXT}` : (args.test_filename || LEGACY_TEST_FILENAME)
}

// --- State ---
let routingDecision = null        // { path: 'direct' | 'pipeline', reason: string }
let problemDescription = ''       // Full problem description (from file or arg)
let testCode = ''                 // Generated test harness
let candidates = []               // [{id, code, seed_index, status, error, round}]
let verifiedKernels = []          // [{id, code, test_output, verification_result}]
let subgraphs = []                // For pipeline path: extracted subgraphs
let subgraphResults = []          // Per-subgraph verified kernels
let composedKernel = null         // Final composed Triton program
let refinementHistory = []        // [{candidate_id, round, prev_error, new_code, fixed}]
let sessionDir = ''               // Artifact directory

// --- verification-strategist decisions (additive; defaults keep happy path) ---
// VERIFICATION_PRUNE gates the parallel Generate screening depth; VERIFICATION_CONFIRM
// gates the Verify/Compose acceptance depth. Populated in Setup; the actual correctness
// verdict still belongs to the PyTorch test harness below.
let VERIFICATION_PRUNE = { method: 'compile_lint', confidence: 'inferred', evidence_source: 'compile' }
let VERIFICATION_CONFIRM = { method: 'reference_test', confidence: 'measured', evidence_source: 'correctness' }

// =============================================================================
// Helper: Disallowed PyTorch patterns (from KernelAgent worker.py)
// =============================================================================
const DISALLOWED_PATTERNS = [
  { pattern: /\bimport\s+torch\.nn/, reason: 'torch.nn modules not allowed — must use pure Triton' },
  { pattern: /\btorch\.nn\./, reason: 'torch.nn usage not allowed' },
  { pattern: /\btorch\.nn\.functional\b/, reason: 'torch.nn.functional not allowed' },
  { pattern: /\bF\.[A-Za-z_]+\(/, reason: 'F.* alias calls not allowed' },
  { pattern: /\btorch\.(relu|sigmoid|tanh|softmax|gelu|mish)\(/, reason: 'PyTorch activation helpers not allowed' },
]

function checkDisallowedPatterns(code) {
  const violations = []
  for (const { pattern, reason } of DISALLOWED_PATTERNS) {
    if (pattern.test(code)) {
      violations.push(reason)
    }
  }
  return violations
}

// =============================================================================
// Helper: Static complexity analysis for routing (from Fuser/auto_agent.py)
// =============================================================================
const HARD_OP_TOKENS = [
  'conv_transpose2d', 'multiheadattention', 'scaled_dot_product_attention',
  'attention', 'softmax', 'einsum', 'group_norm',
]

const CONV_OP_TOKENS = ['conv2d', 'conv1d', 'conv3d']

function analyzeComplexity(problemCode) {
  const code = problemCode.toLowerCase()
  const signals = []

  // Check for hard-to-fuse ops
  for (const token of HARD_OP_TOKENS) {
    if (code.includes(token)) {
      signals.push(`hard_op:${token}`)
    }
  }

  // Check for conv ops
  for (const token of CONV_OP_TOKENS) {
    if (code.includes(token)) {
      signals.push(`conv_op:${token}`)
    }
  }

  // Check for control flow in forward
  const controlFlowPatterns = [/\bif\b.*:/, /\bfor\b.*:/, /\bwhile\b.*:/]
  let hasControlFlow = false
  for (const p of controlFlowPatterns) {
    if (p.test(code)) { hasControlFlow = true; break }
  }
  if (hasControlFlow) signals.push('control_flow')

  // Estimate op chain length (sequential transforms)
  const opChain = (code.match(/\b(relu|sigmoid|tanh|softmax|gelu|conv|linear|batchnorm|layernorm|dropout|pool|add|mul|matmul|bmm)\b/g) || [])
  if (opChain.length >= 4) signals.push(`long_chain:${opChain.length}`)

  return { signals, opCount: opChain.length, hasControlFlow }
}

function shouldUsePipeline(analysis) {
  // Route to pipeline if any of:
  // - attention-like patterns
  // - conv_transpose2d
  // - group_norm with conv or long chains
  // - explicit control flow
  for (const s of analysis.signals) {
    if (s.startsWith('hard_op:attention') || s.startsWith('hard_op:multihead') ||
        s.startsWith('hard_op:softmax') || s.startsWith('hard_op:conv_transpose') ||
        s.startsWith('hard_op:group_norm') || s === 'control_flow') {
      return true
    }
  }
  if (analysis.signals.some(s => s.startsWith('hard_op:group_norm')) && analysis.opCount >= 4) {
    return true
  }
  return false
}

// =============================================================================
// Phase 1: Setup — Parse problem, generate test, initialize workspace
// =============================================================================
phase('Setup')

if (USE_DRIVER) {
  DRIVER = await agentRetry(() => agent(
    `Load the backend driver at ${BACKEND_DIR} and return its manifest plus idioms verbatim.\n` +
    `1. Run exactly: \`cat ${driverPath('manifest.json')}\` and parse JSON.\n` +
    `2. Run exactly: \`cat ${driverPath('idioms.json')}\` and parse JSON.\n` +
    `Return {present, backend_id, source_ext, aux_ext, lang_fence, impl_requirements, methods}.`,
    { model: MODEL.mechanical, label: 'load-driver', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (!DRIVER || DRIVER.present === false) {
    throw new Error(`No backend driver present at ${BACKEND_DIR}. Provide a valid backend_dir or omit it for the legacy path.`)
  }
  if (RESOLVED_BACKEND && DRIVER.backend_id && normalizeSuitabilityValue(DRIVER.backend_id) !== RESOLVED_BACKEND) {
    throw new Error(`backend_dir manifest backend_id="${DRIVER.backend_id}" conflicts with args.backend/language="${RESOLVED_BACKEND}".`)
  }
  DRIVER_LANG_FENCE = DRIVER.lang_fence || DRIVER_LANG_FENCE
  DRIVER_IMPL_REQUIREMENTS = DRIVER.impl_requirements || ''
  DRIVER_SOURCE_EXT = DRIVER.source_ext || DRIVER_SOURCE_EXT
  DRIVER_AUX_EXT = DRIVER.aux_ext || DRIVER.source_ext || DRIVER_AUX_EXT
  DRIVER_BACKEND_ID = DRIVER.backend_id || DRIVER_BACKEND_ID
  log(`Driver loaded: ${DRIVER_BACKEND_ID} (fence=${DRIVER_LANG_FENCE})`)
}

// Read problem from file or use description directly
const setupResult = await agentRetry(() => agent(`You are a ${langToken(LEGACY_ROUTE_LANG_TOKEN)} kernel synthesis expert. Analyze the problem and produce a structured description.

# Problem Source
${PROBLEM_PATH ? `File: ${PROBLEM_PATH} — read and extract the problem description from the Python code.` : ''}
${PROBLEM_DESC ? `Description: ${PROBLEM_DESC}` : ''}

# Instructions
1. If a file path is given, read the file and extract the PyTorch module/problem definition
2. Produce a clear, detailed problem description suitable for ${langToken(LEGACY_ROUTE_LANG_TOKEN)} kernel generation
3. Identify: input tensors (shapes, dtypes), operations, expected output shape/dtype
4. Note any special requirements (numerical precision, edge cases, memory layout)

Return a JSON object with:
- problem_definition: detailed text description
- input_tensors: list of {name, shape, dtype} for each input
- operations: list of operation names in order
- output_spec: {shape, dtype} of the output
- complexity_signals: any signals that suggest this needs multi-subgraph decomposition

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"problem_parse","note":"<op count + input tensor shapes + any complexity signals, one line>"}`, {
  label: 'setup-problem',
  phase: 'Setup',
  model: MODEL.profile,
  schema: {
    type: 'object',
    properties: {
      problem_definition: { type: 'string' },
      input_tensors: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, shape: { type: 'string' }, dtype: { type: 'string' } } } },
      operations: { type: 'array', items: { type: 'string' } },
      output_spec: { type: 'object', properties: { shape: { type: 'string' }, dtype: { type: 'string' } } },
      complexity_signals: { type: 'array', items: { type: 'string' } },
    },
    required: ['problem_definition', 'operations'],
  },
}), { retries: 5 })

problemDescription = setupResult.problem_definition
log(`Problem parsed: ${setupResult.operations.length} ops detected`)

// Generate test harness
const testResult = await agentRetry(() => agent(`You are a ${langToken(LEGACY_HARNESS_LANG_TOKEN)} kernel test engineer. Generate a Python test harness for the following problem.

# Problem Description
${problemDescription}

# Input Tensors
${JSON.stringify(setupResult.input_tensors || [], null, 2)}

# Operations
${JSON.stringify(setupResult.operations || [], null, 2)}

# Requirements
The test harness must:
1. Import the kernel function: \`from kernel import kernel_function\`
2. Create realistic test inputs with appropriate shapes and dtypes
3. Run the kernel function and compare output against a PyTorch reference
4. Print "PASS" if output matches within tolerance, "FAIL" otherwise
5. Exit with code 0 on success, 1 on failure
6. Use torch.allclose with rtol=1e-3, atol=1e-3 for float comparison
7. Test with at least 2 different input sizes if applicable

Return ONLY the Python test code (no markdown, no explanation).`, {
  label: 'setup-test',
  phase: 'Setup',
  model: MODEL.mechanical,
  schema: {
    type: 'object',
    properties: {
      test_code: { type: 'string' },
    },
    required: ['test_code'],
  },
}), { retries: 5 })

testCode = testResult.test_code
log(`Test harness generated (${testCode.length} chars)`)

// verification-strategist: classify the verification NEED per phase (prune for
// cheap screening of the parallel Generate candidates; confirm for deep Verify/Compose
// acceptance) and let the substrate DETERMINISTICALLY map (need x host) -> method +
// stamp confidence. Computed once per session; the vars gate the Generate/Verify
// guidance sentences below. The model picks the fuzzy need; it must NOT assign
// confidence itself. The actual correctness verdict still belongs to the test harness.
const _vp = await agentRetry(() => agent(
  `Classify the verification need for the parallel Generate screening phase as 'prune' (cheap early filter of many diverse seed candidates). Then ` +
  `run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/verification/verification_strategist.py resolve --need prune --host-probe '{"compiler":true,"harness_runnable":true,"reference_available":false}' --cache ${EXP_DIR}/verify_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`.\n` +
  `Return its stdout JSON verbatim {method, evidence_source, confidence, requires, abstained_from, rationale}.`,
  { model: MODEL.mechanical, label: 'verification-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
if (_vp && _vp.method) VERIFICATION_PRUNE = _vp
const _vc = await agentRetry(() => agent(
  `Classify the verification need for the final Verify/Compose acceptance phase as 'confirm' (deep acceptance of a surviving candidate). Then ` +
  `run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/verification/verification_strategist.py resolve --need confirm --host-probe '{"compiler":true,"harness_runnable":true,"reference_available":false}' --cache ${EXP_DIR}/verify_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`.\n` +
  `Return its stdout JSON verbatim {method, evidence_source, confidence, requires, abstained_from, rationale}.`,
  { model: MODEL.mechanical, label: 'verification-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
if (_vc && _vc.method) VERIFICATION_CONFIRM = _vc
log(`Verification-strategist: prune=${VERIFICATION_PRUNE.method}/${VERIFICATION_PRUNE.confidence} confirm=${VERIFICATION_CONFIRM.method}/${VERIFICATION_CONFIRM.confidence}`)

// Initialize experiment directory
sessionDir = `${EXP_DIR}/session_${Date.now()}`
log(`Session directory: ${sessionDir}`)

// =============================================================================
// Phase 2: Route — Static analysis to choose path
// =============================================================================
phase('Route')

const routeResult = await agentRetry(() => agent(`Analyze this problem to determine the optimal synthesis path.

# Problem Description
${problemDescription}

# Operations
${JSON.stringify(setupResult.operations)}

# Complexity Signals
${JSON.stringify(setupResult.complexity_signals || [])}

# Routing Rules
- DIRECT path: Simple, single-kernel problems (element-wise ops, reductions, simple matmul)
  - Fewer than 4 sequential ops
  - No attention patterns, no conv_transpose2d, no group_norm chains
  - No control flow in forward pass

- PIPELINE path: Complex multi-op problems requiring decomposition
  - Attention-like patterns (softmax over QK, multihead attention, einsum with bmm)
  - conv_transpose2d present
  - group_norm with conv or long chains (>=4 steps)
  - Explicit control flow in forward
  - Multiple distinct computational stages

Return a JSON object with:
- path: 'direct' or 'pipeline'
- reason: explanation of the routing decision
- subgraph_count: estimated number of subgraphs (1 for direct)
- estimated_difficulty: 'easy', 'medium', 'hard'

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Route","ts":"<ts>","status":"done","technique":"<chosen path: direct or pipeline>","note":"<routing reason + estimated difficulty + subgraph count>"}`, {
  label: 'route-analysis',
  phase: 'Route',
  model: MODEL.profile,
  schema: {
    type: 'object',
    properties: {
      path: { type: 'string', enum: ['direct', 'pipeline'] },
      reason: { type: 'string' },
      subgraph_count: { type: 'number' },
      estimated_difficulty: { type: 'string' },
    },
    required: ['path', 'reason'],
  },
}), { retries: 5 })

routingDecision = routeResult
log(`Routing: ${routeResult.path} — ${routeResult.reason}`)

// If pipeline path, extract subgraphs
if (routeResult.path === 'pipeline') {
  const subgraphResult = await agentRetry(() => agent(`Extract subgraphs from this problem for parallel kernel synthesis.

# Problem Description
${problemDescription}

# Operations
${JSON.stringify(setupResult.operations)}

# Instructions
1. Identify distinct computational subgraphs that can be independently synthesized
2. For each subgraph, provide: operations, input/output specs, dependencies
3. Ensure subgraphs cover the full computation when composed
4. Deduplicate subgraphs with identical shape signatures

Return a JSON object with:
- subgraphs: array of {id, operations, inputs, outputs, shape_signature, description}`, {
    label: 'route-subgraphs',
    phase: 'Route',
    model: MODEL.profile,
    schema: {
      type: 'object',
      properties: {
        subgraphs: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              operations: { type: 'array', items: { type: 'string' } },
              inputs: { type: 'array', items: { type: 'object' } },
              outputs: { type: 'array', items: { type: 'object' } },
              shape_signature: { type: 'string' },
              description: { type: 'string' },
            },
            required: ['id', 'operations', 'description'],
          },
        },
      },
      required: ['subgraphs'],
    },
  }), { retries: 5 })

  subgraphs = subgraphResult.subgraphs
  log(`Extracted ${subgraphs.length} subgraphs for parallel synthesis`)
}

// =============================================================================
// Phase 3: Generate — Parallel seed generation
// =============================================================================
phase('Generate')

const synthesisTargets = routingDecision.path === 'pipeline' && subgraphs.length > 0
  ? subgraphs.map(sg => ({
      id: sg.id,
      description: `Subgraph ${sg.id}: ${sg.description}\nOperations: ${sg.operations.join(' → ')}`,
      operations: sg.operations,
    }))
  : [{ id: 'main', description: problemDescription, operations: setupResult.operations }]

// For each target, generate MAX_SEEDS diverse kernel candidates in parallel
const seedPromises = []
for (const target of synthesisTargets) {
  for (let seedIdx = 0; seedIdx < MAX_SEEDS; seedIdx++) {
    const temperature = TEMPERATURE_BASE + (seedIdx * 0.1)
    seedPromises.push(() =>
      agentRetry(() => agent(`You are an expert ${langToken(LEGACY_SYNTH_LANG_TOKEN)} kernel developer. Generate a complete, working ${langToken(LEGACY_SYNTH_LANG_TOKEN)} kernel.

# Problem
${target.description}

# Test Harness (the kernel MUST pass this test)
\`\`\`python
${testCode.substring(0, 3000)}
\`\`\`

# ${USE_DRIVER ? `${DRIVER_LANG_FENCE} Guidelines` : 'Triton Guidelines'}
${USE_DRIVER
  ? (DRIVER_IMPL_REQUIREMENTS || `1. Follow the ${DRIVER_LANG_FENCE} kernel idiom for this backend.\n2. Provide a callable host wrapper.\n3. DO NOT use torch.nn, torch.nn.functional, or PyTorch activation helpers.`)
  : `1. Use @triton.jit decorator for kernel functions
2. Use tl.program_id() for block indexing
3. Use tl.load/tl.store for memory access with proper masking
4. Use tl.arange() for offset computation
5. Ensure all operations are within Triton's supported set
6. The kernel_function() wrapper must handle tensor allocation and kernel launch
7. DO NOT use torch.nn, torch.nn.functional, or PyTorch activation helpers
8. Use tl.constexpr for compile-time constants where appropriate
9. Handle edge cases: non-divisible sizes, boundary conditions`}

# Seed Variant ${seedIdx + 1}/${MAX_SEEDS}
Generate a unique implementation approach. Temperature: ${temperature.toFixed(1)}
${seedIdx === 0 ? 'Start with the most straightforward implementation.' : ''}
${seedIdx === 1 ? 'Try a different tiling strategy or block size.' : ''}
${seedIdx === 2 ? (USE_DRIVER ? 'Consider using vectorized memory access or a different memory layout.' : 'Consider using vectorized loads (tl.load with mask) or different memory layout.') : ''}
${seedIdx >= 3 ? 'Explore an alternative algorithmic approach.' : ''}

# Verification Depth (verification-strategist, prune screening)
Verification-strategist selected method='${VERIFICATION_PRUNE.method}' (confidence='${VERIFICATION_PRUNE.confidence}', evidence='${VERIFICATION_PRUNE.evidence_source}'). Honor it: if method==='reference_test', run the full PyTorch reference numerical comparison and tag the verdict evidence='correctness'. If 'smoke_test', compile+run for shape/finiteness only, tag evidence='runtime'. If 'compile_lint', compile+lint only (no execution), tag evidence='compile'. If 'static', reason from source only (evidence='llm_inferred'). Never fabricate a pass verdict you did not actually run.

# Output
Return a JSON object with:
- kernel_code: ${USE_DRIVER ? `complete source file with ${DRIVER_LANG_FENCE} kernel(s) and a callable wrapper` : 'complete Python file with @triton.jit kernel and kernel_function wrapper'}
- approach: brief description of the implementation strategy
- potential_issues: any concerns about correctness or performance

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is seed variant ${seedIdx} for target ${target.id}):
{"workflow":"${WORKFLOW_NAME}","phase":"Generate","ts":"<ts>","status":"done","candidate_id":"${target.id}-seed${seedIdx}","technique":"<your implementation strategy>","note":"<key design choice + any potential issues>"}`, {
        label: `gen-${target.id}-seed${seedIdx}`,
        phase: 'Generate',
        model: MODEL.judgment,
        isolation: 'worktree',
        schema: {
          type: 'object',
          properties: {
            kernel_code: { type: 'string' },
            approach: { type: 'string' },
            potential_issues: { type: 'string' },
          },
          required: ['kernel_code', 'approach'],
        },
      }), { retries: 5 })
    )
  }
}

const seedResults = await parallel(seedPromises)

// Collect candidates with validation
for (let i = 0; i < seedResults.length; i++) {
  const result = seedResults[i]
  if (!result || !result.kernel_code) continue

  const violations = checkDisallowedPatterns(result.kernel_code)
  const candidateId = `candidate_${i}`
  const targetIdx = Math.floor(i / MAX_SEEDS)
  const seedIdx = i % MAX_SEEDS

  candidates.push({
    id: candidateId,
    code: result.kernel_code,
    target_id: synthesisTargets[targetIdx]?.id || 'main',
    seed_index: seedIdx,
    approach: result.approach,
    status: violations.length > 0 ? 'rejected' : 'pending',
    error: violations.length > 0 ? `Policy violations: ${violations.join('; ')}` : null,
    round: 0,
  })
}

const validCandidates = candidates.filter(c => c.status !== 'rejected')
log(`Generated ${candidates.length} candidates (${validCandidates.length} valid, ${candidates.length - validCandidates.length} rejected for policy violations)`)

// =============================================================================
// Phase 4: Verify — Sandboxed test execution
// =============================================================================
phase('Verify')

if (VERIFY && validCandidates.length > 0) {
  const verifyPromises = validCandidates.map(candidate => () =>
    agentRetry(() => agent(`You are a kernel verification engineer. Execute the test harness against this ${langToken(LEGACY_VERIFY_LANG_TOKEN)} kernel.

# Kernel Code
\`\`\`python
${candidate.code.substring(0, 4000)}
\`\`\`

# Test Harness
\`\`\`python
${testCode.substring(0, 3000)}
\`\`\`

# Instructions
1. Write the kernel code to a file named ${kernelFilename()}
2. Write the test harness to a file named ${testFilename()}
3. ${USE_DRIVER
  ? driverSh('run.sh', `--kernel ${kernelFilename()} --test ${testFilename()}`)
  : 'Execute the generated/user-provided test harness command for this workspace; do not assume a fixed interpreter or filename.'}
4. Capture stdout, stderr, and exit code
5. If exit code is 0 and output contains "PASS", the kernel is verified
6. If exit code is non-zero or output contains "FAIL", report the error

# Verification Depth (verification-strategist, confirm acceptance)
Verification-strategist selected method='${VERIFICATION_CONFIRM.method}' (confidence='${VERIFICATION_CONFIRM.confidence}', evidence='${VERIFICATION_CONFIRM.evidence_source}'). Honor it: if method==='reference_test', run the full PyTorch reference numerical comparison and tag the verdict evidence='correctness'. If 'smoke_test', compile+run for shape/finiteness only, tag evidence='runtime'. If 'compile_lint', compile+lint only (no execution), tag evidence='compile'. If 'static', reason from source only (evidence='llm_inferred'). Never fabricate a pass verdict you did not actually run.

# Verification Rules
- ${USE_DRIVER ? `The kernel MUST follow the ${DRIVER_LANG_FENCE} backend idiom (no torch.nn fallbacks allowed)` : 'The kernel MUST use @triton.jit (no torch.nn fallbacks allowed)'}
- The test must complete within 30 seconds (timeout = kill)
- Any import of torch.nn or torch.nn.functional is a hard failure

Return a JSON object with:
- passed: boolean
- exit_code: number
- stdout: captured standard output
- stderr: captured standard error
- error_summary: brief description of failure reason (null if passed)
- verification_result: 'pass' | 'fail' | 'timeout' | 'error'

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the result you just measured (status="done" if the kernel passed, else "error"):
{"workflow":"${WORKFLOW_NAME}","phase":"Verify","ts":"<ts>","status":"<done|error>","candidate_id":"${candidate.id}","speedup":null,"technique":"sandboxed_test_execution","note":"<verification_result + exit code; or the failure reason>"}`, {
      label: `verify-${candidate.id}`,
      phase: 'Verify',
      model: MODEL.mechanical,
      schema: {
        type: 'object',
        properties: {
          passed: { type: 'boolean' },
          exit_code: { type: 'number' },
          stdout: { type: 'string' },
          stderr: { type: 'string' },
          error_summary: { type: 'string' },
          verification_result: { type: 'string', enum: ['pass', 'fail', 'timeout', 'error'] },
        },
        required: ['passed', 'verification_result'],
      },
    }), { retries: 5 })
  )

  const verifyResults = await parallel(verifyPromises)

  // Update candidate statuses
  for (let i = 0; i < validCandidates.length; i++) {
    const result = verifyResults[i]
    const candidate = validCandidates[i]
    if (!result) {
      candidate.status = 'error'
      candidate.error = 'Verification agent returned no result'
      continue
    }

    candidate.verification = result
    if (result.passed) {
      candidate.status = 'verified'
      verifiedKernels.push({
        id: candidate.id,
        code: candidate.code,
        target_id: candidate.target_id,
        test_output: result.stdout,
        verification_result: result.verification_result,
        approach: candidate.approach,
      })
    } else {
      candidate.status = 'failed'
      candidate.error = result.error_summary || result.stderr?.substring(0, 500) || 'Unknown failure'
    }
  }

  if (USE_DRIVER) {
    for (let i = 0; i < validCandidates.length; i++) {
      const candidate = validCandidates[i]
      const kPath = `${EXP_DIR}/candidates/${candidate.id}/${kernelFilename()}`
      const tPath = `${EXP_DIR}/candidates/${candidate.id}/${testFilename()}`
      const buildOut = `${EXP_DIR}/candidates/${candidate.id}/artifact`
      const profOut = `${EXP_DIR}/candidates/${candidate.id}/prof.native`
      const resultPath = `${EXP_DIR}/candidates/${candidate.id}/result.json`
      await agentRetry(() => agent(
        `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
        `Return its stdout JSON verbatim.`,
        { model: MODEL.mechanical, label: `driver-build-${candidate.id}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5 })
      const runOut = await agentRetry(() => agent(
        `${driverSh('run.sh', `--artifact ${buildOut} --kernel ${kPath} --test ${tPath} --result ${resultPath}`)}\n` +
        `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
        { model: MODEL.profile, label: `driver-run-${candidate.id}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      await agentRetry(() => agent(
        `${driverSh('profile.sh', `--artifact ${buildOut} --kernel ${kPath} --out ${profOut}`)}\n` +
        `Return {ok, native_path}.`,
        { model: MODEL.profile, label: `driver-profile-${candidate.id}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5 })
      const evidenceOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/to_evidence.py --native ${profOut}\`.\n` +
        `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`,
        { model: MODEL.mechanical, label: `driver-to-evidence-${candidate.id}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      const diagOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
        `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
        { model: MODEL.mechanical, label: `driver-diagnose-${candidate.id}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      const antiCheatOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --kernel ${kPath} --result ${resultPath}\`.\n` +
        `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
        { model: MODEL.mechanical, label: `driver-anti-cheat-${candidate.id}`, phase: 'Verify', schema: JSON_PASSTHROUGH }), { retries: 5 })
      candidate.driver_envelope = {
        anti_cheat: antiCheatOut || {},
        metrics: (evidenceOut && evidenceOut.metrics) || {},
        vendor: (DRIVER && DRIVER.hw_vendor) || '',
        coverage: (evidenceOut && evidenceOut.coverage) || [],
        bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
        latency_ms: Number((runOut && runOut.latency_ms) || 0),
        backend_id: DRIVER_BACKEND_ID,
      }
    }
  }

  const passedCount = validCandidates.filter(c => c.status === 'verified').length
  log(`Verification: ${passedCount}/${validCandidates.length} passed`)
} else {
  // No verification — mark all as unverified
  for (const c of validCandidates) {
    c.status = 'unverified'
    verifiedKernels.push({
      id: c.id,
      code: c.code,
      target_id: c.target_id,
      test_output: '',
      verification_result: 'skipped',
      approach: c.approach,
    })
  }
  log('Verification skipped (verify=false)')
}

// =============================================================================
// Phase 5: Refine — Iterative refinement for failed candidates
// =============================================================================
phase('Refine')

const failedCandidates = candidates.filter(c => c.status === 'failed')
let currentRound = 0

while (failedCandidates.length > 0 && currentRound < MAX_ROUNDS && verifiedKernels.length === 0) {
  if (typeof budget !== 'undefined' && budget.total && budget.remaining() < EST_PER_ROUND) { log(`token budget ~exhausted — stop`); break }
  currentRound++
  log(`Refinement round ${currentRound}/${MAX_ROUNDS} — ${failedCandidates.length} candidates to fix`)

  const refinePromises = failedCandidates.map(candidate => () =>
    agentRetry(() => agent(`You are a ${langToken(LEGACY_REFINE_LANG_TOKEN)} kernel debugging expert. Fix this failing kernel.

# Problem Description
${problemDescription}

# Current Kernel Code
\`\`\`python
${candidate.code.substring(0, 4000)}
\`\`\`

# Test Output (stdout)
\`\`\`
${(candidate.verification?.stdout || '').substring(0, 1500)}
\`\`\`

# Error Output (stderr)
\`\`\`
${(candidate.verification?.stderr || '').substring(0, 1500)}
\`\`\`

# Error Summary
${candidate.error}

# Previous Refinement Attempts
${refinementHistory.filter(h => h.candidate_id === candidate.id).map(h =>
  `Round ${h.round}: ${h.prev_error} → ${h.fixed ? 'FIXED' : 'STILL FAILING'}`
).join('\n') || 'None'}

# Debugging Guidelines
1. Analyze the error message carefully — it often points to the exact issue
2. Common fixes:
${USE_DRIVER
  ? `   - Shape mismatch: check load/store mask dimensions
   - Index out of bounds: verify offset calculations and masking
   - Compilation error: check ${DRIVER_LANG_FENCE} API usage
   - Wrong output: verify reduction logic, accumulation, synchronization
   - torch.nn fallback: remove any PyTorch nn/functional usage, rewrite in pure ${DRIVER_LANG_FENCE}`
  : `   - Shape mismatch: check tl.load mask dimensions
   - Index out of bounds: verify offset calculations and masking
   - Compilation error: check Triton API usage (tl.constexpr, tl.load signature)
   - Wrong output: verify reduction logic, accumulation, synchronization
   - torch.nn fallback: remove any PyTorch nn/functional usage, rewrite in pure Triton`}
3. Make MINIMAL changes — don't rewrite from scratch unless necessary
4. Preserve the overall approach, fix only the bug

Return a JSON object with:
- kernel_code: the fixed complete kernel code
- changes_made: list of specific changes made
- confidence: 'high' | 'medium' | 'low' that this fix is correct
- fix_explanation: brief explanation of what was wrong and how it was fixed

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (this is refinement round ${currentRound} for ${candidate.id}):
{"workflow":"${WORKFLOW_NAME}","phase":"Refine","ts":"<ts>","status":"done","candidate_id":"${candidate.id}","technique":"<the bug class you fixed>","note":"<what was wrong + the change you made + your confidence>"}`, {
      label: `refine-${candidate.id}-r${currentRound}`,
      phase: 'Refine',
      model: MODEL.judgment,
      isolation: 'worktree',
      schema: {
        type: 'object',
        properties: {
          kernel_code: { type: 'string' },
          changes_made: { type: 'array', items: { type: 'string' } },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
          fix_explanation: { type: 'string' },
        },
        required: ['kernel_code', 'fix_explanation'],
      },
    }), { retries: 5 })
  )

  const refineResults = await parallel(refinePromises)

  // Update candidates with refined code
  for (let i = 0; i < failedCandidates.length; i++) {
    const result = refineResults[i]
    const candidate = failedCandidates[i]
    if (!result || !result.kernel_code) continue

    refinementHistory.push({
      candidate_id: candidate.id,
      round: currentRound,
      prev_error: candidate.error,
      new_code: result.kernel_code,
      fixed: false,
      changes: result.changes_made,
    })

    // Check policy violations on refined code
    const violations = checkDisallowedPatterns(result.kernel_code)
    if (violations.length > 0) {
      candidate.status = 'rejected'
      candidate.error = `Refined code has policy violations: ${violations.join('; ')}`
      continue
    }

    candidate.code = result.kernel_code
    candidate.round = currentRound
    candidate.status = 'pending' // Reset for re-verification
  }

  // Re-verify refined candidates
  const toReVerify = failedCandidates.filter(c => c.status === 'pending')
  if (toReVerify.length > 0 && VERIFY) {
    const reVerifyPromises = toReVerify.map(candidate => () =>
      agentRetry(() => agent(`Verify this refined ${langToken(LEGACY_VERIFY_LANG_TOKEN)} kernel against the test harness.

# Kernel Code
\`\`\`python
${candidate.code.substring(0, 4000)}
\`\`\`

# Test Harness
\`\`\`python
${testCode.substring(0, 3000)}
\`\`\`

${USE_DRIVER ? driverSh('run.sh', `--kernel ${kernelFilename()} --test ${testFilename()}`) + '\n' : ''}Execute the test and report results. Return JSON with:
- passed: boolean
- exit_code: number
- stdout: string
- stderr: string
- error_summary: string or null
- verification_result: 'pass' | 'fail' | 'timeout' | 'error'`, {
        label: `reverify-${candidate.id}-r${currentRound}`,
        phase: 'Refine',
        model: MODEL.mechanical,
        schema: {
          type: 'object',
          properties: {
            passed: { type: 'boolean' },
            exit_code: { type: 'number' },
            stdout: { type: 'string' },
            stderr: { type: 'string' },
            error_summary: { type: 'string' },
            verification_result: { type: 'string', enum: ['pass', 'fail', 'timeout', 'error'] },
          },
          required: ['passed', 'verification_result'],
        },
      }), { retries: 5 })
    )

    const reVerifyResults = await parallel(reVerifyPromises)

    for (let i = 0; i < toReVerify.length; i++) {
      const result = reVerifyResults[i]
      const candidate = toReVerify[i]
      if (!result) continue

      candidate.verification = result
      if (result.passed) {
        candidate.status = 'verified'
        // Mark the last refinement as fixed
        const lastRef = refinementHistory.filter(h => h.candidate_id === candidate.id).pop()
        if (lastRef) lastRef.fixed = true
        verifiedKernels.push({
          id: candidate.id,
          code: candidate.code,
          target_id: candidate.target_id,
          test_output: result.stdout,
          verification_result: result.verification_result,
          approach: candidate.appach,
        })
      } else {
        candidate.status = 'failed'
        candidate.error = result.error_summary || result.stderr?.substring(0, 500) || 'Still failing'
      }
    }
  }

  // Update failedCandidates for next round
  failedCandidates.length = 0
  failedCandidates.push(...candidates.filter(c => c.status === 'failed'))

  const newPassed = verifiedKernels.length
  log(`Round ${currentRound} complete: ${newPassed} verified total`)
}

if (currentRound > 0) {
  log(`Refinement finished after ${currentRound} rounds: ${verifiedKernels.length} verified, ${failedCandidates.length} still failing`)
}

// =============================================================================
// Phase 6: Compose — Stitch subgraph kernels (pipeline path only)
// =============================================================================
phase('Compose')

if (routingDecision.path === 'pipeline' && subgraphs.length > 1 && COMPOSE && verifiedKernels.length > 0) {
  const composeResult = await agentRetry(() => agent(`You are a ${langToken(LEGACY_COMPOSE_LANG_TOKEN)} kernel composition expert. Stitch these verified subgraph kernels into a single, cohesive ${langToken(LEGACY_COMPOSE_LANG_TOKEN)} program.

# Problem Description
${problemDescription}

# Verified Subgraph Kernels
${verifiedKernels.map((k, i) => `
## Subgraph ${k.target_id} (approach: ${k.approach})
\`\`\`python
${k.code.substring(0, 2500)}
\`\`\`
`).join('\n')}

# Subgraph Dependencies
${JSON.stringify(subgraphs.map(sg => ({ id: sg.id, operations: sg.operations })), null, 2)}

# Composition Requirements
1. Create a single Python file containing all @triton.jit kernels
2. Implement a kernel_function() that orchestrates the full computation:
   - Call subgraph kernels in dependency order
   - Pass intermediate results between kernels
   - Handle tensor allocation for intermediates
3. The composed kernel_function must have the same interface as the original problem
4. Include a self-test at the bottom that verifies against the original problem
5. Ensure no PyTorch fallbacks — all computation must go through Triton kernels

Return a JSON object with:
- composed_code: the complete composed Python file
- composition_notes: explanation of how subgraphs were stitched
- intermediate_tensors: list of intermediate tensor allocations needed

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Compose","ts":"<ts>","status":"done","technique":"subgraph_composition","note":"<how subgraphs were stitched + intermediate tensors count>"}`, {
    label: 'compose-stitch',
    phase: 'Compose',
    model: MODEL.judgment,
    schema: {
      type: 'object',
      properties: {
        composed_code: { type: 'string' },
        composition_notes: { type: 'string' },
        intermediate_tensors: { type: 'array', items: { type: 'string' } },
      },
      required: ['composed_code', 'composition_notes'],
    },
  }), { retries: 5 })

  composedKernel = composeResult.composed_code
  log(`Composed kernel: ${composeResult.composition_notes}`)

  // Verify composed kernel
  if (VERIFY) {
    const composeVerify = await agentRetry(() => agent(`Verify this composed ${langToken(LEGACY_VERIFY_LANG_TOKEN)} kernel against the original problem.

# Composed Kernel
\`\`\`python
${composedKernel.substring(0, 5000)}
\`\`\`

# Test Harness
\`\`\`python
${testCode.substring(0, 3000)}
\`\`\`

${USE_DRIVER ? driverSh('run.sh', `--kernel ${kernelFilename()} --test ${testFilename()}`) + '\n' : ''}Execute the test and report results. Return JSON with:
- passed: boolean
- exit_code: number
- stdout: string
- stderr: string
- verification_result: 'pass' | 'fail' | 'timeout' | 'error'`, {
      label: 'compose-verify',
      phase: 'Compose',
      model: MODEL.mechanical,
      schema: {
        type: 'object',
        properties: {
          passed: { type: 'boolean' },
          exit_code: { type: 'number' },
          stdout: { type: 'string' },
          stderr: { type: 'string' },
          verification_result: { type: 'string', enum: ['pass', 'fail', 'timeout', 'error'] },
        },
        required: ['passed', 'verification_result'],
      },
    }), { retries: 5, allowNull: true })

    if (composeVerify?.passed) {
      log('Composed kernel verified successfully')
    } else {
      log(`Composed kernel verification failed: ${composeVerify?.stderr?.substring(0, 200) || 'unknown error'}`)
    }
  }
} else if (routingDecision.path === 'direct' || subgraphs.length <= 1) {
  log('Composition skipped (direct path or single subgraph)')
} else {
  log('Composition skipped (no verified kernels)')
}

// =============================================================================
// Phase 7: Report — Final summary
// =============================================================================
phase('Report')

const bestKernel = verifiedKernels.length > 0 ? verifiedKernels[0] : null
const finalCode = composedKernel || (bestKernel ? bestKernel.code : null)

const reportResult = await agentRetry(() => agent(`Generate a comprehensive synthesis report for this KernelAgent session.

# Problem
${problemDescription}

# Routing Decision
Path: ${routingDecision.path}
Reason: ${routingDecision.reason}
Estimated Difficulty: ${routeResult.estimated_difficulty || 'unknown'}

# Generation Summary
- Total candidates generated: ${candidates.length}
- Policy-rejected: ${candidates.filter(c => c.status === 'rejected').length}
- Verification passed: ${verifiedKernels.length}
- Verification failed: ${candidates.filter(c => c.status === 'failed').length}
- Refinement rounds: ${currentRound}
- Refinement attempts: ${refinementHistory.length}

# Best Kernel Approach
${bestKernel ? bestKernel.approach : 'No verified kernel found'}

# Composition
${composedKernel ? 'Multi-subgraph kernels composed into single program' : 'Single-kernel solution (no composition needed)'}

# Report Format
Provide:
1. Executive summary (1-2 sentences)
2. Synthesis outcome: success/partial/failed
3. Best kernel details: approach, key implementation choices
4. Verification status with any caveats
5. Performance considerations (if inferable from code structure)
6. Recommendations for further optimization
7. Full path to session artifacts

Return a JSON object with:
- outcome: 'success' | 'partial' | 'failed'
- summary: string
- best_kernel_approach: string
- verification_status: string
- recommendations: array of strings
- artifacts_path: string

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"<done|error>","technique":"synthesis_report","note":"<outcome + verified count + best kernel approach, one line>"}`, {
  label: 'report-summary',
  phase: 'Report',
  model: MODEL.judgment,
  schema: {
    type: 'object',
    properties: {
      outcome: { type: 'string', enum: ['success', 'partial', 'failed'] },
      summary: { type: 'string' },
      best_kernel_approach: { type: 'string' },
      verification_status: { type: 'string' },
      recommendations: { type: 'array', items: { type: 'string' } },
      artifacts_path: { type: 'string' },
    },
    required: ['outcome', 'summary'],
  },
}), { retries: 5, allowNull: true })

log(`Synthesis ${reportResult?.outcome || 'unknown'}: ${reportResult?.summary || ''}`)

return {
  input_mode: 'generate_then_optimize',
  problem_definition: PROBLEM_DESC || problemDescription,
  problem_path: PROBLEM_PATH,
  generated_kernel_path: sessionDir ? `${sessionDir}/best_kernel.py` : '',
  initial_candidates: candidates,
  initial_generation_result: {
    verified: verifiedKernels.length > 0,
    selected_candidate_id: bestKernel?.id || '',
  },
  outcome: reportResult?.outcome || 'unknown',
  summary: reportResult?.summary || '',
  routing_path: routingDecision.path,
  routing_reason: routingDecision.reason,
  problem_definition_resolved: problemDescription,
  total_candidates: candidates.length,
  verified_count: verifiedKernels.length,
  refinement_rounds: currentRound,
  refinement_attempts: refinementHistory.length,
  best_kernel_code: finalCode || '',
  best_kernel_approach: bestKernel?.approach || '',
  composed_kernel: composedKernel || '',
  verification_status: reportResult?.verification_status || 'unknown',
  recommendations: reportResult?.recommendations || [],
  session_directory: sessionDir,
  test_code: testCode,
  subgraph_count: subgraphs.length,
  candidates_summary: candidates.map(c => ({
    id: c.id,
    target: c.target_id,
    status: c.status,
    round: c.round,
    approach: c.approach,
  })),
}
