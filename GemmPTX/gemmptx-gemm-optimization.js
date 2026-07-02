export const meta = {
  name: 'gemmptx-gemm-optimization',
  description: 'GEMM-specific CUDA/CuTe/CUTLASS optimization loop driven by hardware census, PTX/SASS instruction evidence, correctness, benchmark, and optional NCU profiling.',
  whenToUse: 'Use for existing CUDA/CuTe/CUTLASS GEMM kernels when the goal is to choose and verify the best instruction path (mma.sync, wgmma.mma_async, cp.async/TMA, tcgen05/TMEM) from real hardware and disassembly evidence. Not a generic compute-bound optimizer.',
  requiredSkills: [],
  optionalSkills: ['gemmptx-instruction-evidence'],
  phases: [
    { title: 'Hardware Census', detail: 'Collect target GPU facts from a user-provided probe command and static problem context' },
    { title: 'GEMM Signature', detail: 'Classify GEMM shape/dtype/layout/roofline regime and current implementation style' },
    { title: 'Baseline Evidence', detail: 'Compile, test, benchmark, and disassemble the baseline before proposing PTX-level changes' },
    { title: 'Instruction Plan', detail: 'Select a small set of architecture-appropriate PTX/SASS hypotheses with regex evidence gates' },
    { title: 'Implement', detail: 'Materialize one candidate at a time without mutating the original kernel' },
    { title: 'Disassemble Verify', detail: 'Reject candidates whose expected PTX/SASS instruction path is not observed' },
    { title: 'Profile', detail: 'Run benchmark and optional NCU/profile evidence only after correctness and instruction verification' },
    { title: 'Decide', detail: 'Accept, reject, or record hypothesis_not_realized based on measured evidence' },
    { title: 'Report', detail: 'Write final best path, evidence, and lessons for future GEMM/PTX runs' },
  ],
}

const WORKFLOW_NAME = 'gemmptx-gemm-optimization'

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
    } catch (e) {
      lastError = e
    }
  }
  if (lastError) throw lastError
  if (opts && opts.allowNull === true) return null
  throw new Error(
    `agentRetry: "${(opts && opts.label) || 'agent'}" returned null after ${retries + 1} attempt(s) ` +
    `(agent skipped or terminal API failure after retries).`,
  )
}

function guard(obj, field, fallback) {
  if (obj == null || obj[field] == null) return fallback
  return obj[field]
}
// --- END inlined agent-retry scaffolding ---

const MODEL = {
  mechanical: args.model_mechanical || 'haiku',
  profile: args.model_profile || 'sonnet',
  judgment: args.model_judgment || 'opus',
}

const KERNEL_PATH = args.kernel_path || ''
const PROBLEM_PATH = args.problem_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const LANGUAGE = String(args.language || 'cuda').toLowerCase()
const TARGET_GPU = args.target_gpu || ''
const EXP_DIR = args.exp_dir || '/tmp/gemmptx'
const ITERATIONS = Number(args.iterations || 3)
const HARDWARE_PROBE_COMMAND = args.hardware_probe_command || ''
const COMPILE_COMMAND = args.compile_command || ''
const TEST_COMMAND = args.test_command || ''
const BENCHMARK_COMMAND = args.benchmark_command || ''
const DISASSEMBLE_COMMAND = args.disassemble_command || args.ptx_disassemble_command || ''
const PROFILE_COMMAND = args.profile_command || args.ncu_command || ''
const MIN_SPEEDUP = Number(args.min_speedup || 1.01)

if (!KERNEL_PATH) {
  return { ok: false, error: 'missing_required_arg', missing: 'kernel_path', reason: 'GemmPTX optimizes an existing CUDA/CuTe/CUTLASS GEMM kernel.' }
}
if (!COMPILE_COMMAND || !TEST_COMMAND || !BENCHMARK_COMMAND || !DISASSEMBLE_COMMAND) {
  return {
    ok: false,
    error: 'missing_evidence_contract',
    missing: {
      compile_command: !COMPILE_COMMAND,
      test_command: !TEST_COMMAND,
      benchmark_command: !BENCHMARK_COMMAND,
      disassemble_command: !DISASSEMBLE_COMMAND,
    },
    reason: 'PTX/SASS-driven GEMM optimization requires compile, correctness, benchmark, and disassembly evidence. Do not claim an instruction path without disassembly evidence.',
  }
}
if (!['cuda', 'cu', 'cute', 'cutlass', 'cpp', 'c++'].includes(LANGUAGE)) {
  return { ok: false, error: 'unsupported_language', language: LANGUAGE, supported: ['cuda', 'cute', 'cutlass', 'cpp'] }
}

const JSON_SCHEMA = { type: 'object', additionalProperties: true }
const GEMMPTX_SKILL_HINT = 'Before making GEMM instruction-path judgments, read the workflow-local skill at GemmPTX/skills/gemmptx-instruction-evidence/SKILL.md if it exists. Apply its PTX/SASS evidence gates, architecture map, and failure checklist.'
const instructionCatalog = [
  {
    arch: 'sm80',
    instruction: 'mma.sync',
    ptx_regex: 'mma\\.sync\\.aligned',
    sass_regex: '(HMMA|MMA)',
    use_when: 'Ampere fp16/bf16 GEMM mainloops with tensor-core friendly alignment',
    pitfalls: ['register pressure from large accumulator tiles', 'cp.async stage count can reduce occupancy'],
  },
  {
    arch: 'sm90',
    instruction: 'wgmma.mma_async',
    ptx_regex: 'wgmma\\.mma_async',
    sass_regex: '(WGMMA|GMMA)',
    use_when: 'Hopper GEMM mainloops where warpgroup MMA and SMEM descriptors can feed tensor cores',
    pitfalls: ['descriptor swizzle mismatch', 'missing wgmma.fence/commit/wait ordering', 'large N accumulator register wall'],
  },
  {
    arch: 'sm90',
    instruction: 'cp.async.bulk.tensor',
    ptx_regex: 'cp\\.async\\.bulk\\.tensor',
    sass_regex: '(TMA|CP_ASYNC_BULK)',
    use_when: 'Hopper affine tile copies where TMA can replace per-thread copy loops',
    pitfalls: ['mbarrier expect_tx byte mismatch', 'missing async proxy fence before TMA stores'],
  },
  {
    arch: 'sm100',
    instruction: 'tcgen05',
    ptx_regex: 'tcgen05',
    sass_regex: '(TCGEN05|MMA)',
    use_when: 'Blackwell GEMM mainloops where TMEM accumulation can reduce register pressure',
    pitfalls: ['TMEM capacity planning', 'CTA-pair synchronization', 'small tiles may lose to wgmma/CUTLASS paths'],
  },
]

function commandContract(name, command, resultPath) {
  return [
    `# ${name} command contract`,
    `Command: ${command}`,
    `Result path: ${resultPath}`,
    'Substitute placeholders only when present: {kernel_path}, {candidate_path}, {artifact_path}, {result_path}, {exp_dir}, {target_gpu}.',
    'Run the command exactly after substitution. If it fails, return the non-zero result with stderr tail; do not invent fields.',
  ].join('\n')
}

function genomeFooter(phaseName, candidateId) {
  return [
    '',
    '# Genome self-report (REQUIRED - do this LAST; do NOT let it change your returned JSON)',
    `Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ`,
    `The line must be one JSON object with workflow="${WORKFLOW_NAME}", phase="${phaseName}", candidate_id="${candidateId || ''}", status, technique, speedup when measured, and a one-line note.`,
  ].join('\n')
}

let best = {
  candidate_id: 'baseline',
  kernel_path: KERNEL_PATH,
  speedup: 1.0,
  latency_ms: null,
  instruction_path: [],
  artifact_path: '',
}
const history = []

// =============================================================================
// Phase 1: Hardware Census
// =============================================================================
phase('Hardware Census')

const hardware = await agentRetry(() => agent(`You are the hardware-census agent for a GEMM/PTX optimization workflow.

# Goal
Collect target GPU facts that constrain PTX/SASS-level GEMM decisions.

# Inputs
- target_gpu: ${TARGET_GPU || '(not provided)'}
- hardware_probe_command: ${HARDWARE_PROBE_COMMAND || '(not provided)'}
- static fallback: if no command is provided, read architecture-specific facts only when they are already present in the repo (for example WarpSpeed/config/hardware-facts-sm90.md or sm100.md). Mark measured=false.

# Required fields
Return JSON with:
- measured: boolean
- gpu_name, compute_capability, arch
- sm_count, l2_bytes, shared_mem_per_sm, shared_mem_per_block_max
- regs_per_sm, max_threads_per_sm, hbm_bandwidth_gbps
- clock_state, driver_version, cuda_version
- caveats: array of strings

# Rules
- If hardware_probe_command is provided, run it exactly and parse its JSON/stdout.
- If you cannot measure a field, set it to null and explain in caveats.
- Do not invent exact SM/L2/cache numbers from memory.

${genomeFooter('Hardware Census', 'hardware')}`, {
  model: MODEL.profile,
  label: 'hardware-census',
  phase: 'Hardware Census',
  schema: JSON_SCHEMA,
}), { retries: 5 })

log(`Hardware: arch=${hardware.arch || '?'} gpu=${hardware.gpu_name || TARGET_GPU || '?'} sm=${hardware.sm_count || '?'} measured=${hardware.measured === true}`)

// =============================================================================
// Phase 2: GEMM Signature
// =============================================================================
phase('GEMM Signature')

const signature = await agentRetry(() => agent(`You are a GEMM kernel analyst. Build a structured signature for the target GEMM.

# Inputs
- kernel_path: ${KERNEL_PATH}
- problem_path: ${PROBLEM_PATH || '(not provided)'}
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- language: ${LANGUAGE}
- hardware arch: ${hardware.arch || TARGET_GPU || '(unknown)'}

# Tasks
1. Read the kernel source and any provided problem file.
2. Confirm this is GEMM / batched GEMM / GEMM-like matmul. If it is not, return is_gemm=false with reason.
3. Extract or infer only from source/problem facts: dtype, accumulator dtype, M/N/K or dynamic axes, layout, transpose, epilogue, alignment assumptions.
4. Classify bottleneck prior: tensor_core_bound, grid_starved_small_m, memory_reuse_limited, register_pressure_bound, scheduler_latency_bound, or unknown.
5. Identify the current instruction path if visible from source: scalar FMA, WMMA, mma.sync, WGMMA, CUTLASS/CuTe collective, TMA, tcgen05, unknown.

Return structured JSON. Do not claim a PTX instruction is present until disassembly verifies it.

${genomeFooter('GEMM Signature', 'signature')}`, {
  model: MODEL.judgment,
  label: 'gemm-signature',
  phase: 'GEMM Signature',
  schema: {
    type: 'object',
    properties: {
      is_gemm: { type: 'boolean' },
      reason: { type: 'string' },
      op_family: { type: 'string' },
      dtype_a: { type: 'string' },
      dtype_b: { type: 'string' },
      dtype_acc: { type: 'string' },
      shape_summary: { type: 'string' },
      layout_summary: { type: 'string' },
      alignment_summary: { type: 'string' },
      bottleneck_prior: { type: 'string' },
      current_source_path: { type: 'string' },
      risks: { type: 'array', items: { type: 'string' } },
    },
    required: ['is_gemm', 'op_family', 'bottleneck_prior'],
  },
}), { retries: 5 })

if (signature.is_gemm === false) {
  return { ok: false, error: 'not_gemm', reason: signature.reason || 'GemmPTX only supports GEMM-like kernels.' }
}

// =============================================================================
// Phase 3: Baseline Evidence
// =============================================================================
phase('Baseline Evidence')

const baseline = await agentRetry(() => agent(`You are the baseline-evidence agent. Establish measured baseline correctness, latency, and instruction evidence before any edit.

# Baseline source
${KERNEL_PATH}

${commandContract('Compile', COMPILE_COMMAND, `${EXP_DIR}/baseline.compile.json`)}

${commandContract('Correctness', TEST_COMMAND, `${EXP_DIR}/baseline.test.json`)}

${commandContract('Benchmark', BENCHMARK_COMMAND, `${EXP_DIR}/baseline.bench.json`)}

${commandContract('Disassemble', DISASSEMBLE_COMMAND, `${EXP_DIR}/baseline.disasm.json`)}

# Required evidence
Run compile -> correctness -> benchmark -> disassemble. Return:
- compiled, correct
- artifact_path
- latency_ms, throughput, speedup (baseline speedup should be 1.0 if only absolute latency is available)
- disassembly evidence: ptx_path, sass_path, observed_instructions, registers_per_thread, shared_mem_bytes, local_mem_bytes, spill_loads, spill_stores
- instruction_summary

# Rules
- If compile or correctness fails, stop after the failure and return the evidence.
- Do not claim an instruction path without disassembly evidence.

${genomeFooter('Baseline Evidence', 'baseline')}`, {
  model: MODEL.profile,
  label: 'baseline-evidence',
  phase: 'Baseline Evidence',
  schema: JSON_SCHEMA,
}), { retries: 5 })

if (baseline.compiled === false || baseline.correct === false) {
  return { ok: false, error: 'baseline_invalid', baseline }
}

best = {
  candidate_id: 'baseline',
  kernel_path: KERNEL_PATH,
  speedup: Number(baseline.speedup || 1.0),
  latency_ms: baseline.latency_ms || null,
  throughput: baseline.throughput || null,
  artifact_path: baseline.artifact_path || '',
  instruction_path: baseline.observed_instructions || [],
  disassembly: baseline,
}
history.push({ candidate_id: 'baseline', status: 'baseline', speedup: best.speedup, latency_ms: best.latency_ms, instruction_path: best.instruction_path })

// =============================================================================
// Phase 4: Instruction Plan
// =============================================================================
phase('Instruction Plan')

const plan = await agentRetry(() => agent(`You are the GEMM/PTX instruction planner. Choose a small candidate set, each with a falsifiable instruction-level hypothesis.

# Hardware
${JSON.stringify(hardware, null, 2)}

# GEMM signature
${JSON.stringify(signature, null, 2)}

# Baseline disassembly
${JSON.stringify({
  observed_instructions: baseline.observed_instructions || [],
  registers_per_thread: baseline.registers_per_thread,
  shared_mem_bytes: baseline.shared_mem_bytes,
  spill_loads: baseline.spill_loads,
  spill_stores: baseline.spill_stores,
  instruction_summary: baseline.instruction_summary,
}, null, 2)}

# Instruction catalog
${JSON.stringify(instructionCatalog, null, 2)}

# Workflow-local skill
${GEMMPTX_SKILL_HINT}

# Requirements
Return up to ${ITERATIONS} candidates. Each candidate must include:
- candidate_id
- target_instruction
- ptx_regex
- sass_regex
- hypothesis
- expected_mechanism
- edit_strategy
- risk_level: low|medium|high
- reject_if_missing: true
- expected_gain_pct

# Planning discipline
- Prefer architecture-appropriate paths: sm80 -> mma.sync/cp.async; sm90 -> wgmma.mma_async/TMA; sm100 -> tcgen05/TMEM only when the hardware facts support it.
- Keep the first version conservative. Do not combine more than two major instruction changes in one candidate.
- Include contraindications such as register pressure, SMEM pressure, alignment, or missing tensor-core dtype.
- Every candidate must be judged by disassembly first; benchmark comes only after correctness and instruction verification.

${genomeFooter('Instruction Plan', 'plan')}`, {
  model: MODEL.judgment,
  label: 'instruction-plan',
  phase: 'Instruction Plan',
  schema: {
    type: 'object',
    properties: {
      candidates: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            candidate_id: { type: 'string' },
            target_instruction: { type: 'string' },
            ptx_regex: { type: 'string' },
            sass_regex: { type: 'string' },
            hypothesis: { type: 'string' },
            expected_mechanism: { type: 'string' },
            edit_strategy: { type: 'string' },
            risk_level: { type: 'string' },
            reject_if_missing: { type: 'boolean' },
            expected_gain_pct: { type: 'number' },
          },
          required: ['candidate_id', 'target_instruction', 'ptx_regex', 'sass_regex', 'hypothesis', 'edit_strategy'],
        },
      },
    },
    required: ['candidates'],
  },
}), { retries: 5 })

const candidates = (plan.candidates || []).slice(0, ITERATIONS)
if (candidates.length === 0) {
  return { ok: false, error: 'no_instruction_candidates', baseline, signature, hardware }
}

// =============================================================================
// Phases 5-8: Implement -> Disassemble Verify -> Profile -> Decide
// =============================================================================
for (let i = 0; i < candidates.length; i++) {
  const spec = candidates[i]
  const candidateId = spec.candidate_id || `candidate-${i + 1}`
  const candidatePath = `${EXP_DIR}/candidates/${candidateId}/kernel.cu`
  const artifactPath = `${EXP_DIR}/candidates/${candidateId}/kernel_artifact`

  phase('Implement')
  const implementation = await agentRetry(() => agent(`You are the GEMM/PTX implementor. Materialize exactly one candidate and do not mutate the original kernel.

# Candidate
${JSON.stringify(spec, null, 2)}

# Original kernel
${KERNEL_PATH}

# Workflow-local skill
${GEMMPTX_SKILL_HINT}

# Output
- candidate_path: ${candidatePath}
- artifact_path placeholder for commands: ${artifactPath}

# Procedure
1. Create the candidate directory.
2. Copy/read the original source.
3. Implement only the stated instruction-level hypothesis.
4. Keep the public API and harness interface compatible with the original.
5. Write the candidate source to ${candidatePath}.
6. Do not benchmark in this phase.

# Guardrail
If you cannot honestly implement the stated PTX/SASS hypothesis, return implemented=false and explain. Do not substitute a different strategy.

${genomeFooter('Implement', candidateId)}`, {
    model: MODEL.judgment,
    label: `implement-${candidateId}`,
    phase: 'Implement',
    schema: JSON_SCHEMA,
  }), { retries: 5 })

  if (implementation.implemented === false) {
    history.push({ candidate_id: candidateId, status: 'implementation_failed', reason: implementation.reason || implementation.error || 'not implemented' })
    continue
  }

  phase('Disassemble Verify')
  const verify = await agentRetry(() => agent(`You are the instruction-evidence gate. Compile, test, and disassemble the candidate, then verify the expected PTX/SASS regexes.

# Candidate
- candidate_id: ${candidateId}
- candidate_path: ${candidatePath}
- artifact_path: ${artifactPath}
- target_instruction: ${spec.target_instruction}
- ptx_regex: ${spec.ptx_regex}
- sass_regex: ${spec.sass_regex}

# Workflow-local skill
${GEMMPTX_SKILL_HINT}

${commandContract('Compile', COMPILE_COMMAND, `${EXP_DIR}/candidates/${candidateId}/compile.json`)}

${commandContract('Correctness', TEST_COMMAND, `${EXP_DIR}/candidates/${candidateId}/test.json`)}

${commandContract('Disassemble', DISASSEMBLE_COMMAND, `${EXP_DIR}/candidates/${candidateId}/disasm.json`)}

# Rules
1. Run compile first. If it fails, return status="compile_error".
2. Run correctness next. If it fails, return status="incorrect".
3. Run disassemble next.
4. Search both PTX and SASS text/artifacts for ptx_regex and sass_regex.
5. If neither expected regex is observed, return status="hypothesis_not_realized". This is not a performance failure; it means the compiler did not produce the intended instruction path.
6. Do not benchmark here.

# Required return fields
status, compiled, correct, instruction_verified, observed_instructions, missing_expected_instructions, registers_per_thread, shared_mem_bytes, local_mem_bytes, spill_loads, spill_stores, ptx_path, sass_path, artifact_path.

Reminder: do not claim an instruction path without disassembly evidence.

${genomeFooter('Disassemble Verify', candidateId)}`, {
    model: MODEL.profile,
    label: `disassemble-verify-${candidateId}`,
    phase: 'Disassemble Verify',
    schema: JSON_SCHEMA,
  }), { retries: 5 })

  if (verify.status === 'compile_error' || verify.compiled === false) {
    history.push({ candidate_id: candidateId, status: 'compile_error', detail: verify.error || verify.failure_reason || '', target_instruction: spec.target_instruction })
    continue
  }
  if (verify.status === 'incorrect' || verify.correct === false) {
    history.push({ candidate_id: candidateId, status: 'incorrect', detail: verify.error || verify.failure_reason || '', target_instruction: spec.target_instruction })
    continue
  }
  if (verify.status === 'hypothesis_not_realized' || verify.instruction_verified === false) {
    history.push({
      candidate_id: candidateId,
      status: 'hypothesis_not_realized',
      target_instruction: spec.target_instruction,
      missing_expected_instructions: verify.missing_expected_instructions || [spec.ptx_regex, spec.sass_regex],
      registers_per_thread: verify.registers_per_thread,
    })
    log(`Candidate ${candidateId}: hypothesis_not_realized for ${spec.target_instruction}`)
    continue
  }

  phase('Profile')
  const measured = await agentRetry(() => agent(`You are the measurement agent. The candidate already compiled, passed correctness, and verified its instruction path. Now benchmark and optionally profile it.

# Candidate
- candidate_id: ${candidateId}
- candidate_path: ${candidatePath}
- artifact_path: ${verify.artifact_path || artifactPath}
- target_instruction: ${spec.target_instruction}

${commandContract('Benchmark', BENCHMARK_COMMAND, `${EXP_DIR}/candidates/${candidateId}/bench.json`)}

${PROFILE_COMMAND ? commandContract('Profile', PROFILE_COMMAND, `${EXP_DIR}/candidates/${candidateId}/profile.json`) : '# Profile command: not provided. Skip NCU/profile and mark profile_available=false.'}

# Return
- measured: true if benchmark succeeded
- correct: true only if benchmark also confirms correctness or prior correctness remains valid
- latency_ms, throughput, speedup_vs_baseline, speedup_vs_best
- profile_available
- metrics: object with SM%, tensor pipe, L2 hit, DRAM%, occupancy, stall reasons when available
- mechanism_moved: did the predicted mechanism move in the expected direction?
- diagnosis

${genomeFooter('Profile', candidateId)}`, {
    model: MODEL.profile,
    label: `profile-${candidateId}`,
    phase: 'Profile',
    schema: JSON_SCHEMA,
  }), { retries: 5 })

  phase('Decide')
  const speedup = Number(measured.speedup_vs_baseline || measured.speedup || 0)
  const accepted = measured.measured !== false && measured.correct !== false && speedup >= Math.max(best.speedup * MIN_SPEEDUP, best.speedup + 0.000001)
  const record = {
    candidate_id: candidateId,
    status: accepted ? 'accepted' : 'rejected',
    target_instruction: spec.target_instruction,
    speedup,
    latency_ms: measured.latency_ms || null,
    instruction_verified: true,
    registers_per_thread: verify.registers_per_thread,
    shared_mem_bytes: verify.shared_mem_bytes,
    mechanism_moved: measured.mechanism_moved,
    diagnosis: measured.diagnosis || '',
  }
  history.push(record)

  if (accepted) {
    best = {
      candidate_id: candidateId,
      kernel_path: candidatePath,
      artifact_path: verify.artifact_path || artifactPath,
      speedup,
      latency_ms: measured.latency_ms || null,
      instruction_path: verify.observed_instructions || [spec.target_instruction],
      disassembly: verify,
      metrics: measured.metrics || {},
    }
    log(`Candidate ${candidateId}: accepted speedup=${speedup}`)
  } else {
    log(`Candidate ${candidateId}: rejected speedup=${speedup || 'n/a'} best=${best.speedup}`)
  }
}

// =============================================================================
// Phase 9: Report
// =============================================================================
phase('Report')

const report = await agentRetry(() => agent(`Write the final GEMM/PTX optimization report.

# Output directory
${EXP_DIR}

# Best
${JSON.stringify(best, null, 2)}

# History
${JSON.stringify(history, null, 2)}

# Required report contents
1. Best candidate path and speedup.
2. The verified PTX/SASS instruction path, citing PTX/SASS artifacts.
3. A table of all candidates with statuses including compile_error, incorrect, hypothesis_not_realized, rejected, accepted.
4. Lessons/dead ends: include any instruction path that failed to materialize and the likely reason.
5. Clear statement that this workflow is GEMM-specific and not a generic compute-bound optimizer.

Write:
- ${EXP_DIR}/report.md
- ${EXP_DIR}/history.json

${genomeFooter('Report', 'final')}`, {
  model: MODEL.judgment,
  label: 'report',
  phase: 'Report',
  schema: {
    type: 'object',
    properties: {
      report_path: { type: 'string' },
      history_path: { type: 'string' },
      lessons: { type: 'array', items: { type: 'string' } },
    },
    required: ['report_path', 'history_path'],
  },
}), { retries: 5 })

return {
  ok: true,
  workflow: WORKFLOW_NAME,
  best_candidate_id: best.candidate_id,
  best_kernel_path: best.kernel_path,
  best_artifact_path: best.artifact_path,
  best_speedup: best.speedup,
  best_latency_ms: best.latency_ms,
  verified_instruction_path: best.instruction_path,
  history,
  report_path: report.report_path,
  history_path: report.history_path,
  lessons: guard(report, 'lessons', []),
}
