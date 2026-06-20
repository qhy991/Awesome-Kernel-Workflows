export const meta = {
  name: 'tritorx-operator-generation',
  description: 'FSM-based agentic generation of Triton ATen kernels for emerging accelerator platforms with linter-driven feedback (TritorX methodology)',
  whenToUse: 'When generating functionally correct Triton kernel + wrapper pairs for an entire PyTorch ATen operator set on a new accelerator platform (ASIC/NPU). Prioritizes correctness and coverage over performance. Uses a finite state machine with custom linter, JIT compilation, and OpInfo test harness for iterative refinement.',
  phases: [
    { title: 'Setup', detail: 'Parse operator list, load docstrings, prepare test harness and hardware config' },
    { title: 'Generate', detail: 'LLM generates initial Triton kernel + Python wrapper from operator docstring' },
    { title: 'Lint', detail: 'Custom linter checks for illegal dispatches, invalid intrinsics, format issues' },
    { title: 'Compile-Test', detail: 'JIT compile on target hardware, run OpInfo tests across dtypes/shapes' },
    { title: 'Debug', detail: 'Analyze failures (compile/runtime/accuracy), build feedback prompt, retry' },
    { title: 'Report', detail: 'Coverage statistics, operator-level pass/fail, production readiness summary' },
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

const WORKFLOW_NAME = 'tritorx-operator-generation'

// --- verification-strategist wiring (GENERATOR: no performance signal; routes
// verification DEPTH, not a profiler). The agent classifies the verification NEED
// (prune for early screening at Lint, confirm for final acceptance at Compile-Test);
// the substrate DETERMINISTICALLY maps (need × host) to a method and STAMPS
// confidence by method -- not the agent. See _substrate/verification/README.md. ---
const SUBSTRATE = args.substrate_dir || '_substrate'
const PY = args.substrate_command_prefix || ''
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }


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
  supported_languages: ['triton'],
  supported_problem_types: ['aten-triton-operator-generation', 'operator-generation'],
  problem_types: ['ATen/Triton operator coverage generation', 'coverage-first compile/lint/test/debug loops'],
  reason: 'TritorX targets Triton dialect operator generation and coverage, not performance-first CUDA tuning.',
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
// TritorX: Agentic Operator Generation for ML ASICs
// =============================================================================
//
// Source: "Agentic Operator Generation for ML ASICs"
//         Hammond, Markosyan, Dontula, Mahns, Fisches, Pedchenko, Muzumdar,
//         Supper, Saroufim, Isaacson, Wang, Hunt, Gondkar, Levenstein,
//         Synnaeve, Li, Kahn, Mathews
//         Meta (FAIR / Meta Superintelligence Labs), arXiv:2512.10977, 2025
//
// TritorX generates functionally correct Triton PyTorch ATen kernels at scale
// for emerging accelerator platforms (e.g., Meta MTIA). Unlike performance-focused
// kernel optimization, TritorX prioritizes COVERAGE — generating correct kernels
// for the entire operator set (481 ops, 20,000+ tests).
//
// FSM Architecture (Figure 3):
//   Init → Generate Kernel → Triton Linter → Compile/Execute/Test
//     → Process Results → Debug → (feedback loop back to Generate)
//     → Success (all OpInfo tests pass) or Failure (max retries)
//
// Key mechanisms:
//   1. Custom Triton Linter: prevents "cheating" (dispatching to host PyTorch ops),
//      enforces valid Triton dialect syntax, catches unauthorized intrinsics
//   2. In-context distillation: learns hardware-specific Triton semantics iteratively
//      from compiler/runtime error feedback (not from upfront documentation)
//   3. OpInfo test harness: tests across all supported dtypes, shapes, and argument
//      patterns — orders of magnitude more tests than KernelBench
//   4. Feedback summarization: secondary LLM summarizes long compiler logs to fit
//      in context window
//   5. LLDB debugger integration: for runtime crashes, extracts backtrace + registers
//
// Usage:
//   Workflow({name: 'tritorx-operator-generation', args: {
//     operator_name: 'aten::softmax',
//     operator_docstring: 'Applies softmax over dim ...',
//     target_platform: 'MTIA',
//     triton_dialect: 'triton-mtia',
//     compile_command: '<user-provided compile/import command>',
//     test_command: '<user-provided OpInfo/equivalent test command>',
//     lint_command: '<user-provided Triton dialect linter command>',
//     strict_harness: true,
//     max_attempts: 3,
//     max_llm_calls_per_attempt: 15,
//     few_shot_examples: [],
//   }})
//
// =============================================================================

// --- Required Args ---
const OP_NAME = args.operator_name || ''
const OP_DOCSTRING = args.operator_docstring || ''

// --- Optional Args ---
const TARGET_PLATFORM = args.target_platform || 'GPU'
const TRITON_DIALECT = args.triton_dialect || 'triton'
const COMPILE_CMD = args.compile_command || ''
const TEST_CMD = args.test_command || ''
const LINT_CMD = args.lint_command || ''
const STRICT_HARNESS = args.strict_harness === true
const MAX_ATTEMPTS = args.max_attempts || 3
const MAX_LLM_CALLS = args.max_llm_calls_per_attempt || 15
const FEW_SHOT_EXAMPLES = args.few_shot_examples || []
const EXP_DIR = args.exp_dir || '/tmp/tritorx_exp'
const SUPPORTED_DTYPES = args.supported_dtypes || ['bfloat16', 'float16', 'float32', 'int32', 'int64']
const ENABLE_SUMMARIZATION = args.enable_summarization !== false
const OPERATOR_LIST = args.operator_list || []
const HARNESS_EVIDENCE = (LINT_CMD && TEST_CMD && COMPILE_CMD)
  ? 'strict_linter_opinfo'
  : (STRICT_HARNESS ? 'missing_required_harness' : 'TritorX-style FSM')

// --- State ---
let currentKernel = ''
let currentWrapper = ''
let attempt = 0
let llmCallsThisAttempt = 0
let allResults = []
let operatorsPassed = []
let operatorsFailed = []

// Operators to process (single or batch)
const operators = OPERATOR_LIST.length > 0
  ? OPERATOR_LIST
  : [{ name: OP_NAME, docstring: OP_DOCSTRING }]

// --- verification-strategist decisions (resolved once, before the operator loop).
// Lint = early screening (prune); Compile-Test = final acceptance (confirm).
// Defaults keep the happy path unchanged if the resolve is ignored. ---
let VERIFICATION_PRUNE = { method: 'compile_lint', confidence: 'inferred', evidence_source: 'compile' }
let VERIFICATION_CONFIRM = { method: 'compile_lint', confidence: 'inferred', evidence_source: 'compile' }

// =============================================================================
// Phase 1: Setup — Parse operators, prepare harness
// =============================================================================
phase('Setup')

const setupResult = await agent(`You are setting up a TritorX kernel generation session.

# Target Platform: ${TARGET_PLATFORM}
# Triton Dialect: ${TRITON_DIALECT}
# Operators to generate: ${operators.length === 1 ? operators[0].name : `${operators.length} operators`}
# Harness contract:
- strict_harness: ${STRICT_HARNESS}
- harness_evidence: ${HARNESS_EVIDENCE}
- Strict TritorX fidelity requires lint_command, compile_command, and an OpInfo/equivalent test_command. Without them, run only as TritorX-style FSM.

# Tasks:
1. Create experiment directory: mkdir -p ${EXP_DIR}/{kernels,wrappers,logs,tests}
2. Verify the target platform toolchain is available:
   ${COMPILE_CMD ? `Compile check: ${COMPILE_CMD}` : '(verify Triton JIT is accessible)'}
3. Verify test harness:
   ${TEST_CMD ? `Test harness: ${TEST_CMD}` : '(verify OpInfo or equivalent test framework)'}
4. Parse operator docstrings and identify:
   - Input/output signatures
   - Supported dtypes: ${SUPPORTED_DTYPES.join(', ')}
   - Nested operator dependencies (e.g., argmax references max)
   - Expected output format

${operators.length === 1 ? `# Operator: ${operators[0].name}\n# Docstring:\n${operators[0].docstring?.substring(0, 2000) || '(to be loaded)'}` : ''}

Return setup status.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"harness_setup","note":"<platform/dialect, operator count, harness_evidence, key signatures/constraints, one line>"}`, {
  label: 'setup',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      platform_ready: { type: 'boolean' },
      test_harness_ready: { type: 'boolean' },
      operator_signatures: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, signature: { type: 'string' } } } },
      nested_dependencies: { type: 'array', items: { type: 'string' } },
      platform_constraints: { type: 'string' },
    },
    required: ['platform_ready'],
  },
})

log(`Setup: ${TARGET_PLATFORM} | ${operators.length} operator(s) | Dtypes: ${SUPPORTED_DTYPES.join(', ')}`)

// --- verification-strategist resolve (once, at Setup): classify the verification
// NEED per phase (prune for Lint screening, confirm for Compile-Test acceptance)
// and let the substrate pick the method + stamp confidence. The agent only
// classifies (fuzzy need × host probe); the strategist owns method+confidence.
// Honored in the Lint and Compile-Test prompts below. ---
const _vsp = await agent(
  `Classify the host verification capability for this TritorX session on ${TARGET_PLATFORM} (${TRITON_DIALECT}). ` +
  `Probe: is a Triton compiler available (compiler=true)? is the test harness runnable (harness_runnable=${!!TEST_CMD || STRICT_HARNESS})? is a numerical reference available (reference_available=${!!TEST_CMD})? ` +
  `Then run the strategist TWICE with the same host probe and write both results to the trajectory:\n` +
  `1. Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/verification/verification_strategist.py resolve --need prune --host-probe '<probe json>' --cache ${EXP_DIR}/verify_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`\n` +
  `2. Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/verification/verification_strategist.py resolve --need confirm --host-probe '<probe json>' --cache ${EXP_DIR}/verify_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`\n` +
  `Return JSON {prune: <stdout1 verbatim>, confirm: <stdout2 verbatim>} where each value has {method, confidence, evidence_source, requires, abstained_from, rationale}.`,
  { model: MODEL.mechanical, label: 'verification-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH })
if (_vsp && _vsp.prune && _vsp.prune.method) VERIFICATION_PRUNE = _vsp.prune
if (_vsp && _vsp.confirm && _vsp.confirm.method) VERIFICATION_CONFIRM = _vsp.confirm

// =============================================================================
// Per-Operator Generation Loop
// =============================================================================

for (const op of operators) {
  const opName = op.name || OP_NAME
  const opDoc = op.docstring || OP_DOCSTRING
  let success = false

  log(`\n=== Generating: ${opName} ===`)

  for (attempt = 0; attempt < MAX_ATTEMPTS && !success; attempt++) {
    llmCallsThisAttempt = 0
    currentKernel = ''
    currentWrapper = ''
    let feedbackPrompt = ''

    // =========================================================================
    // Phase 2: Generate — LLM produces Triton kernel + wrapper
    // =========================================================================
    phase('Generate')

    const generateResult = await agent(`You are a TritorX kernel generator. Generate a Triton ${TRITON_DIALECT} kernel and Python wrapper for this PyTorch ATen operator.

# Operator: ${opName}
# Docstring:
${opDoc.substring(0, 3000) || `(Generate implementation for ${opName})`}

# Target Platform: ${TARGET_PLATFORM}
# Triton Dialect: ${TRITON_DIALECT}
# Supported dtypes: ${SUPPORTED_DTYPES.join(', ')}

${FEW_SHOT_EXAMPLES.length > 0 ? `# Few-shot Examples:\n${FEW_SHOT_EXAMPLES.slice(0, 3).map((ex, i) => `## Example ${i + 1}: ${ex.name}\nKernel:\n\`\`\`python\n${ex.kernel?.substring(0, 1000)}\n\`\`\`\nWrapper:\n\`\`\`python\n${ex.wrapper?.substring(0, 500)}\n\`\`\``).join('\n\n')}` : ''}

${feedbackPrompt ? `# Feedback from previous attempt:\n${feedbackPrompt}` : ''}

# Output Requirements:
1. **Triton Kernel** (@triton.jit decorated function):
   - Implement the core computation using Triton block-level operations
   - Handle masking for non-aligned dimensions
   - Support all required dtypes via proper casting
   ${TARGET_PLATFORM === 'MTIA' ? '- Use 32-byte aligned memory access patterns\n   - Use MTIA-compatible Triton intrinsics only' : ''}

2. **Python Wrapper** (matching ATen operator signature):
   - Match the exact PyTorch operator signature from the docstring
   - Compute grid dimensions and launch the kernel
   - Handle dtype dispatch for multiple input types
   - Do NOT call any other torch/aten operators (this is "cheating")
   - Do NOT use torch.nn.functional or torch.ops fallbacks

Return the kernel and wrapper code.

Attempt ${attempt + 1}/${MAX_ATTEMPTS}, LLM call ${llmCallsThisAttempt + 1}/${MAX_LLM_CALLS}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (operator ${opName}, attempt ${attempt}):
{"workflow":"${WORKFLOW_NAME}","phase":"Generate","ts":"<ts>","status":"done","candidate_id":"${opName}","technique":"<core kernel approach you generated, e.g. block_reduction>","speedup":null,"note":"<implementation approach, masking/dtype handling, one line>"}`, {
      label: `generate-${opName}-a${attempt}`,
      phase: 'Generate',
      schema: {
        type: 'object',
        properties: {
          kernel_code: { type: 'string' },
          wrapper_code: { type: 'string' },
          implementation_notes: { type: 'string' },
        },
        required: ['kernel_code', 'wrapper_code'],
      },
    })

    currentKernel = generateResult?.kernel_code || ''
    currentWrapper = generateResult?.wrapper_code || ''
    llmCallsThisAttempt++

    // Inner FSM loop: Lint → Compile/Test → Debug → retry
    let fsmState = 'lint'

    while (fsmState !== 'success' && fsmState !== 'failure' && llmCallsThisAttempt < MAX_LLM_CALLS) {

      if (fsmState === 'lint') {
        // =====================================================================
        // Phase 3: Lint — Custom linter checks
        // =====================================================================
        phase('Lint')

        const lintResult = await agent(`You are the TritorX Custom Linter (Section 3.2).
Check this Triton kernel + wrapper for violations.

# Kernel:
\`\`\`python
${currentKernel.substring(0, 4000)}
\`\`\`

# Wrapper:
\`\`\`python
${currentWrapper.substring(0, 2000)}
\`\`\`

# Lint Rules:
1. Wrapper must NOT call other ATen operators (no torch.*, no F.*, no aten::*)
   EXCEPT: tensor allocation (torch.empty, torch.zeros) is allowed
2. Kernel must use valid ${TRITON_DIALECT} syntax and intrinsics only
3. Kernel must have @triton.jit decorator
4. Wrapper must match the operator signature for ${opName}
5. No unauthorized use of host-side PyTorch functions in the kernel path
${TARGET_PLATFORM === 'MTIA' ? '6. Memory access must be 32-byte aligned\n7. Only MTIA-supported Triton intrinsics' : ''}

${LINT_CMD ? `Also run: ${LINT_CMD}` : ''}

Verification-strategist selected method='${VERIFICATION_PRUNE.method}' (confidence='${VERIFICATION_PRUNE.confidence}', evidence='${VERIFICATION_PRUNE.evidence_source}'). Honor it: if method==='reference_test', run the full reference numerical comparison and tag the verdict evidence='correctness'. If 'smoke_test', compile+run for shape/finiteness only, tag evidence='runtime'. If 'compile_lint', compile+lint only (no execution), tag evidence='compile'. If 'static', reason from source only (evidence='llm_inferred'). Never fabricate a pass verdict you did not actually run.

Return lint results.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (operator ${opName}, status="done" if lint passed else "error"):
{"workflow":"${WORKFLOW_NAME}","phase":"Lint","ts":"<ts>","status":"<done|error>","candidate_id":"${opName}","technique":"custom_triton_linter","speedup":null,"note":"<passed? else top violations + severity, one line>"}`, {
          label: `lint-${opName}-a${attempt}-c${llmCallsThisAttempt}`,
          phase: 'Lint',
          schema: {
            type: 'object',
            properties: {
              passed: { type: 'boolean' },
              violations: { type: 'array', items: { type: 'string' } },
              severity: { type: 'string' },
            },
            required: ['passed'],
          },
        })
        llmCallsThisAttempt++

        if (lintResult?.passed) {
          fsmState = 'compile_test'
        } else {
          feedbackPrompt = `Lint violations:\n${(lintResult?.violations || []).join('\n')}\nFix these issues.`
          fsmState = 'debug'
        }

      } else if (fsmState === 'compile_test') {
        // =====================================================================
        // Phase 4: Compile-Test — JIT compile + OpInfo tests
        // =====================================================================
        phase('Compile-Test')

        const testResult = await agent(`You are the TritorX Compile/Test executor.
Compile and test this kernel on ${TARGET_PLATFORM}.

# Kernel:
\`\`\`python
${currentKernel.substring(0, 3000)}
\`\`\`

# Wrapper:
\`\`\`python
${currentWrapper.substring(0, 2000)}
\`\`\`

# Steps:
1. JIT Compile: ${COMPILE_CMD || `Import and compile the Triton kernel via JIT`}
2. Run tests: ${TEST_CMD || `Run OpInfo tests for ${opName} across dtypes: ${SUPPORTED_DTYPES.join(', ')}`}
3. For each test:
   - Compile kernel (may need recompilation for different dtypes)
   - Execute on ${TARGET_PLATFORM}
   - Compare output against PyTorch reference (CPU ATen)
   - Check numerical tolerance (dtype-dependent)

Verification-strategist selected method='${VERIFICATION_CONFIRM.method}' (confidence='${VERIFICATION_CONFIRM.confidence}', evidence='${VERIFICATION_CONFIRM.evidence_source}'). Honor it: if method==='reference_test', run the full reference numerical comparison and tag the verdict evidence='correctness'. If 'smoke_test', compile+run for shape/finiteness only, tag evidence='runtime'. If 'compile_lint', compile+lint only (no execution), tag evidence='compile'. If 'static', reason from source only (evidence='llm_inferred'). Never fabricate a pass verdict you did not actually run.

Report: compilation status, tests passed/failed, error details.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" if all_passed, else "error"):
{"workflow":"${WORKFLOW_NAME}","phase":"Compile-Test","ts":"<ts>","status":"<done|error>","candidate_id":"${opName}","technique":"jit_compile_opinfo","speedup":null,"note":"<compiled? tests_passed/tests_total; failure_type + failing_dtypes if any, one line>"}`, {
          label: `test-${opName}-a${attempt}-c${llmCallsThisAttempt}`,
          phase: 'Compile-Test',
          schema: {
            type: 'object',
            properties: {
              compiled: { type: 'boolean' },
              compile_error: { type: 'string' },
              tests_total: { type: 'number' },
              tests_passed: { type: 'number' },
              tests_failed: { type: 'number' },
              all_passed: { type: 'boolean' },
              failure_type: { type: 'string' },
              failure_details: { type: 'string' },
              failing_dtypes: { type: 'array', items: { type: 'string' } },
            },
            required: ['compiled', 'all_passed'],
          },
        })
        llmCallsThisAttempt++

        if (testResult?.all_passed) {
          fsmState = 'success'
        } else {
          // Build feedback for debug phase
          if (!testResult?.compiled) {
            feedbackPrompt = `Compilation failed:\n${testResult?.compile_error || 'unknown error'}`
          } else if (testResult?.failure_type === 'runtime') {
            feedbackPrompt = `Runtime error. Failing dtypes: ${(testResult?.failing_dtypes || []).join(', ')}\n${testResult?.failure_details || ''}`
          } else {
            feedbackPrompt = `Accuracy error: ${testResult?.tests_passed}/${testResult?.tests_total} tests passed.\nFailing dtypes: ${(testResult?.failing_dtypes || []).join(', ')}\n${testResult?.failure_details || ''}`
          }
          fsmState = 'debug'
        }

      } else if (fsmState === 'debug') {
        // =====================================================================
        // Phase 5: Debug — Analyze failure, regenerate
        // =====================================================================
        phase('Debug')

        const debugResult = await agent(`You are the TritorX Debug agent. Fix this kernel based on the error feedback.

# Current Kernel:
\`\`\`python
${currentKernel.substring(0, 3000)}
\`\`\`

# Current Wrapper:
\`\`\`python
${currentWrapper.substring(0, 1500)}
\`\`\`

# Error Feedback:
${feedbackPrompt}

# Operator: ${opName}
# Platform: ${TARGET_PLATFORM}

# Debug Guidelines (TritorX Section 3.2):
- For compile errors: check Triton syntax, intrinsic compatibility, type mismatches
- For runtime crashes: check memory access alignment, bounds, synchronization
- For accuracy errors: check dtype handling, reduction semantics, broadcasting
- Do NOT introduce calls to other ATen operators as a "fix"

Return the fixed kernel and wrapper.
LLM call ${llmCallsThisAttempt + 1}/${MAX_LLM_CALLS}

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append (operator ${opName}, attempt ${attempt}):
{"workflow":"${WORKFLOW_NAME}","phase":"Debug","ts":"<ts>","status":"done","candidate_id":"${opName}","technique":"<the fix you applied, e.g. fix_memory_alignment>","speedup":null,"note":"<root cause from the error feedback + what you changed, one line>"}`, {
          label: `debug-${opName}-a${attempt}-c${llmCallsThisAttempt}`,
          phase: 'Debug',
          schema: {
            type: 'object',
            properties: {
              kernel_code: { type: 'string' },
              wrapper_code: { type: 'string' },
              fix_description: { type: 'string' },
            },
            required: ['kernel_code', 'wrapper_code'],
          },
        })
        llmCallsThisAttempt++

        if (debugResult?.kernel_code) {
          currentKernel = debugResult.kernel_code
          currentWrapper = debugResult.wrapper_code || currentWrapper
        }
        fsmState = 'lint'
      }
    }

    if (fsmState === 'success') {
      success = true
      operatorsPassed.push({ name: opName, attempt: attempt + 1, llm_calls: llmCallsThisAttempt })
      log(`  ${opName}: PASS (attempt ${attempt + 1}, ${llmCallsThisAttempt} LLM calls)`)
    } else if (fsmState === 'failure' || llmCallsThisAttempt >= MAX_LLM_CALLS) {
      log(`  ${opName}: attempt ${attempt + 1} FAILED (${llmCallsThisAttempt} LLM calls)`)
    }
  }

  if (!success) {
    operatorsFailed.push({ name: opName, attempts: MAX_ATTEMPTS, last_error: feedbackPrompt?.substring(0, 100) || '' })
    log(`  ${opName}: FAILED after ${MAX_ATTEMPTS} attempts`)
  }

  allResults.push({ name: opName, success, kernel: success ? currentKernel : '', wrapper: success ? currentWrapper : '' })
}

// =============================================================================
// Phase 6: Report
// =============================================================================
phase('Report')

const coverage = operators.length > 0 ? (operatorsPassed.length / operators.length * 100).toFixed(1) : '0'

const finalReport = await agent(`Write a TritorX generation report.

# TritorX Results
- Target: ${TARGET_PLATFORM} (${TRITON_DIALECT})
- Operators attempted: ${operators.length}
- Operators passed: ${operatorsPassed.length} (${coverage}% coverage)
- Operators failed: ${operatorsFailed.length}
- Max attempts per operator: ${MAX_ATTEMPTS}
- Max LLM calls per attempt: ${MAX_LLM_CALLS}
- Harness evidence: ${HARNESS_EVIDENCE}

# Passed Operators:
${operatorsPassed.map(o => `- ${o.name}: attempt ${o.attempt}, ${o.llm_calls} LLM calls`).join('\n') || 'None'}

# Failed Operators:
${operatorsFailed.map(o => `- ${o.name}: ${o.last_error}`).join('\n') || 'None'}

Write:
1. Coverage analysis by operator category
2. Common failure patterns and root causes
3. In-context learning effectiveness (did later calls succeed faster?)
4. Platform-specific challenges encountered
5. Recommendations for improving coverage

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","technique":"coverage_report","speedup":null,"note":"<operators passed/attempted, coverage %, dominant failure pattern, one line>"}`, {
  label: 'final-report',
  phase: 'Report',
})

return {
  target_platform: TARGET_PLATFORM,
  operators_attempted: operators.length,
  operators_passed: operatorsPassed.length,
  operators_failed: operatorsFailed.length,
  coverage_pct: parseFloat(coverage),
  passed_details: operatorsPassed,
  failed_details: operatorsFailed,
  all_results: allResults,
  strict_harness: STRICT_HARNESS,
  harness_evidence: HARNESS_EVIDENCE,
  report: finalReport,
}
