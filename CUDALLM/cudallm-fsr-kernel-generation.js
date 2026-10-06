export const meta = {
  name: 'cudallm-fsr-kernel-generation',
  description: 'CUDA-LLM Feature Search and Reinforcement workflow for evaluator-guided CUDA kernel generation',
  whenToUse: 'When generating or optimizing CUDA kernels from a task specification and you want explicit CUDA feature search rather than generic iterative prompting. Maintains feature-level scores for tiling, shared memory, vectorization, warp primitives, occupancy, and fast math, then reinforces feature choices using compile, correctness, and latency evidence.',
  phases: [
    { title: 'Setup', detail: 'Read task specification, reference implementation, target GPU, and evaluator contract' },
    { title: 'FeatureCatalog', detail: 'Build a CUDA optimization feature space tailored to the task' },
    { title: 'GenerateTests', detail: 'Create diverse correctness and boundary tests for the task' },
    { title: 'SelectFeatures', detail: 'Choose a feature combination from scored feature history' },
    { title: 'GenerateKernel', detail: 'Generate candidate CUDA code conditioned on selected features' },
    { title: 'Evaluate', detail: 'Compile, correctness-test, and benchmark the candidate' },
    { title: 'Reinforce', detail: 'Update feature scores from measured compile/correctness/speedup reward' },
    { title: 'Report', detail: 'Return best kernel, feature reward table, failures, and next feature sets' },
  ],
}
// --- BEGIN inlined task-result scaffolding (from _meta/scaffolding/task-result.js) ---
// Native task-result compatibility: task command is the measurement authority.
function __taskHold(reason, taskResult = null) {
  // The supervising agent must receive the complete diagnostic evidence even
  // when the method cannot safely submit another GPU/model request. Native
  // runtimes may serialize only the message, so retain it there as well.
  const error = new Error('TASK_RESULT_HOLD: ' + reason
    + (taskResult ? '\nRaw task evidence for diagnosis (not an accepted score):\n' + JSON.stringify(taskResult) : ''))
  error.code = 'TASK_RESULT_HOLD'
  error.retryable = false
  error.outcome_state = 'unknown'
  if (taskResult) error.task_result = taskResult
  throw error
}
function __parseTaskResultJSON(raw) {
  try { return JSON.parse(raw) } catch (original) {
    // Native legacy replies double-encode JSON. A model may decode an ANSI
    // escape inside a quoted diagnostic while copying that string. Quote only
    // literal controls inside strings: retain every character, never trim or
    // infer missing syntax, fields, values or evidence.
    let quoted = false, escaped = false, text = ''
    for (const character of raw) {
      if (quoted && !escaped && character.charCodeAt(0) < 32) {
        text += JSON.stringify(character).slice(1, -1)
        continue
      }
      text += character
      if (escaped) escaped = false
      else if (quoted && character === '\\') escaped = true
      else if (character === '"') quoted = !quoted
    }
    if (text === raw) throw original
    return JSON.parse(text)
  }
}
function __taskResult(output, expectedPath, expectedCount) {
  let result
  try {
    const raw = output?.test_result_json
    result = typeof raw === 'string' ? __parseTaskResultJSON(raw)
      : raw && typeof raw === 'object' && !Array.isArray(raw) ? JSON.parse(JSON.stringify(raw))
      : __taskHold('missing/malformed raw task result')
  } catch { return __taskHold('missing/malformed raw task result') }
  // The caller owns the slot. Duplicated delivery paths are descriptive, not
  // candidate identity: source binding and canonical candidate_path below remain mandatory.
  if (result?.contract_version !== 'kersor-task-result-v1') return __taskHold('task result contract mismatch', result)
  const deliveryPathObservation = output?.test_result_path !== expectedPath || result?.test_result_path !== expectedPath
    ? {returned:output?.test_result_path ?? null,recorded:result?.test_result_path ?? null,canonical:expectedPath} : null
  if (output.model_observation === 'unknown' || result.model_observation === 'unknown' || result.outcome_state === 'unknown' || (result.failure_origin && result.failure_origin !== 'candidate')) return __taskHold('infrastructure, evidence, or model observation unknown', result)
  const valid = result.compiled === true && result.correct === true
    && result.full_workload_set === true && result.measurement_valid === true
    && result.source_binding?.verified === true && result.n_pass === result.n_total
    && Number.isInteger(result.n_total) && result.n_total === expectedCount
    && /^[0-9a-f]{64}$/i.test(result.source_binding?.source_sha256 || '')
    && result.candidate_path === expectedPath + '.artifact/candidate.py'
    && result.n_total > 0 && Number.isFinite(result.candidate_latency_aggregate_ms)
    && result.candidate_latency_aggregate_ms > 0 && Number.isFinite(result.speedup_vs_reference)
    && result.speedup_vs_reference > 0
  const candidateFailure = result.outcome_state === 'candidate_failure' && result.failure_origin === 'candidate'
    && (result.full_workload_set === true || (result.candidate_runtime_fault_proven === true
      && /^[0-9a-f]{64}$/i.test(result.runtime_fault_proof_sha256 || '') && result.full_workload_set === false)) && result.n_total === expectedCount
    && Number.isInteger(result.n_pass) && result.n_pass >= 0 && result.n_pass < expectedCount
    && result.source_binding?.verified === true
    && /^[0-9a-f]{64}$/i.test(result.source_binding?.source_sha256 || '')
    && result.candidate_path === expectedPath + '.artifact/candidate.py'
  if (!valid && !candidateFailure) return __taskHold('incomplete workload/measurement/source proof', result)
  return {is_valid:valid, measurement_valid:valid, compiled:result.compiled, correct:valid,
    metric_value:valid ? result.speedup_vs_reference : 0,
    speedup:valid ? result.speedup_vs_reference : 0,
    latency_ms:valid ? result.candidate_latency_aggregate_ms : null,
    n_pass:result.n_pass, n_total:result.n_total,
    pass_rate:String(result.n_pass)+'/'+String(result.n_total),
    source_binding:result.source_binding, host_candidate_path:result.candidate_path, test_result_path:expectedPath,
    ...(deliveryPathObservation ? {delivery_path_observation:deliveryPathObservation} : {}),
    outcome_state:valid ? 'passed' : 'candidate_failure',
    diagnostics:result.diagnostics || [], evidence:result.evidence || null,
    error_log:valid ? '' : JSON.stringify({error:result.error || 'candidate rejected',
      diagnostics:result.diagnostics || [], candidate_path:result.candidate_path, evidence:result.evidence || null})}
}

function __taskCommand(command, candidatePath, resultPath) {
  for (const [placeholder, value] of [['{kernel_path}', candidatePath], ['{result_path}', resultPath]]) {
    // The documented placeholders represent whole command arguments. Accept
    // either bare or already quoted placeholders without evaluating path text.
    const quoted = "'" + String(value).replaceAll("'", "'\\''") + "'"
    command = command.replaceAll('"'+placeholder+'"', placeholder).replaceAll("'"+placeholder+"'", placeholder).replaceAll(placeholder, quoted)
  }
  return command
}

async function __taskResultWithReadback(ctx, output) {
  try { return __taskResult(output, ctx.resultPath, ctx.workloadCount) }
  catch (error) {
    const readCommand = ctx.readCommand || (typeof args !== 'undefined' && args.task_result_read_command)
    const raw = error?.task_result
    if (error?.code !== 'TASK_RESULT_HOLD' || !readCommand
        || output?.model_observation === 'unknown' || raw?.model_observation === 'unknown'
        || (raw?.outcome_state && !['passed','candidate_failure'].includes(raw.outcome_state))
        || (raw?.failure_origin && raw.failure_origin !== 'candidate')) throw error
    // Repair result delivery once. The declared reader must reconcile an
    // existing slot and must never create a slot or submit another GPU job.
    const command = __taskCommand(readCommand, ctx.resultPath+'.artifact/candidate.py', ctx.resultPath)
    let reread, readExit = null
    if (typeof evaluate === 'function') {
      const execution = await evaluate({protocol:'command-v1',label:(ctx.label || 'task')+'-read-result',
        phase:'ReadResult',filesystem_policy:'read-only',
        argv:['/bin/sh','-c',command]})
      if (execution?.timed_out || execution?.stage !== 'complete' || ![0,1].includes(execution?.exit_code))
        return __taskHold('read-only task reconciliation failed', execution?.stdout_json || raw)
      reread = {test_result_path:ctx.resultPath,test_result_json:execution.stdout_json}
      readExit = execution.exit_code
    } else {
      try { reread = await agentRetry(() => agent(`Repair only the result delivery for ${ctx.resultPath}.
The previous returned JSON failed validation: ${error.message}
Run the declared CPU-only read command exactly once: ${command}
It reconciles the existing frozen candidate, complete trace and broker receipt. Do not submit a GPU job, alter any file, regenerate source, or repeat an unknown/refused request.
Return test_result_path and test_result_json copied exactly from this command's output. test_result_json is a JSON object, NOT a JSON-encoded string. Preserve every diagnostic and evidence field. Do not reconstruct paths or measurements from memory.`, {
        label:(ctx.label || 'task')+'-read-result',phase:'ReadResult',
        schema:{type:'object',properties:{test_result_path:{type:'string'},test_result_json:{anyOf:[{type:'object',additionalProperties:true},{type:'string'}]}},required:['test_result_path','test_result_json']},
      }), {retries:0}) } catch (readError) {
        return __taskHold('read-only result delivery unavailable: '+(readError?.message || String(readError)), raw)
      }
    }
    // A second invalid delivery remains HOLD. This is not an unbounded retry.
    const measured = __taskResult(reread,ctx.resultPath,ctx.workloadCount)
    if (readExit !== null && readExit !== (measured.is_valid ? 0 : 1)) return __taskHold('read-only command exit differs from raw outcome', raw)
    return {...measured,result_delivery_recovered:true}
  }
}

async function __nativeTaskEvaluate(ctx) {
  const command = __taskCommand(ctx.command, ctx.candidatePath, ctx.resultPath)
  // Host uses the existing generic command runner. Read the designated file;
  // command chatter and an agent-selected alternative are not the return owner.
  if (typeof evaluate === 'function') {
    let execution
    try {
      execution = await evaluate({protocol:'command-v1', label:ctx.label, phase:'Evaluate', filesystem_policy:'workspace-write',
        argv:['/bin/sh', '-c', `(\n${command}\n) >&2\nstatus=$?\ncat -- "$1"\nexit "$status"`, 'kersor-task-result', ctx.resultPath]})
    } catch (error) { return __taskHold('command result unavailable: ' + (error?.message || String(error))) }
    if (execution?.timed_out || execution?.stage !== 'complete' || ![0, 1].includes(execution?.exit_code)) return __taskHold('command execution incomplete or unknown', execution?.stdout_json || null)
    const measured = __taskResult({test_result_path:ctx.resultPath, test_result_json:execution.stdout_json}, ctx.resultPath, ctx.workloadCount)
    if (execution.exit_code !== (measured.is_valid ? 0 : 1)) return __taskHold('command exit differs from raw task outcome')
    return measured
  }
  let output
  try { output = await agentRetry(() => agent(`Use the explicitly declared candidate file at ${ctx.candidatePath}. If it is absent, write the COMPLETE returned source below to that exact path. Never rewrite an existing declared file or select another directory entry.
${ctx.candidateSource || ''}
Run the trusted task command once in the foreground: ${command}
If Bash returns a running session, wait for that session to terminate. Do not detach with nohup or &, return a pending summary, or launch another command for this slot. Never rename/delete its artifact directory, change the result path, or optimize the declared source during evaluation. An unknown or failed result must be returned unchanged.
Read ${ctx.resultPath}. Return test_result_path and test_result_json copied verbatim from that exact file. test_result_json is a JSON object, NOT a JSON-encoded string. Preserve every diagnostic and evidence field. No estimates or rewritten source in this reply.`, {
    label:ctx.label, phase:'Evaluate',
    schema:{type:'object', properties:{test_result_path:{type:'string'},test_result_json:{anyOf:[{type:'object',additionalProperties:true},{type:'string'}]}}, required:['test_result_path','test_result_json']},
  }), {retries:0}) } catch (error) { return __taskHold('agent/transport result unavailable: ' + (error?.message || String(error))) }
  return await __taskResultWithReadback(ctx, output)
    || __taskHold('missing task result')
}

async function __nativeTaskAcceptedParent(ctx) {
  const measured = await __nativeTaskEvaluate(ctx)
  if (!measured?.is_valid) return __taskHold('accepted parent failed full official task measurement')
  return measured
}

// Prompt-only projection. Keep the raw result in runtime state and on disk.
// A missing file reference must never be replaced by an invented path.
function __taskEvidencePrompt(result) {
  if (!result || !result.test_result_path) return result
  return {test_result_path:result.test_result_path,source_path:result.host_candidate_path,
    evidence_paths:result.evidence || null,
    measurement:{compiled:result.compiled,correct:result.correct,is_valid:result.is_valid,
      measurement_valid:result.measurement_valid,speedup:result.speedup,metric_value:result.metric_value,
      latency_ms:result.latency_ms,n_pass:result.n_pass,n_total:result.n_total,outcome_state:result.outcome_state},
    read_instruction:'Read the complete result file and referenced source/trace files for diagnostics before repairing or interpreting this candidate. This index does not replace the original evidence.'}
}
// --- END inlined task-result scaffolding ---
// --- BEGIN sol-execbench-eval substrate (auto-inlined by scripts/patch-sol-execbench-eval.js) ---
const SOL_SOLUTION_CONTRACT = [
  'SOL-EXECBENCH SOLUTION CONTRACT (this task is evaluated by the sol-execbench CLI):',
  '',
  'You are authoring a kernel that will be packaged into a solution.json and run by',
  'the sol-execbench harness, which compiles it internally. Therefore:',
  '',
  '1. Emit a COMPLETE candidate with the task entry point run(...). CUDA C++',
  '   requires a torch PYBIND11_MODULE binding; Python/Triton requires a',
  '   module-level def run(...). Do NOT write a standalone main()/CLI harness.',
  '2. Match the task reference signature exactly (same argument order/dtypes).',
  '3. Do NOT package, compile, or benchmark yourself — the workflow + substrate',
  '   handle pack -> sol-execbench -> parse. Return only the runnable source.',
].join('\n')

function __solQ(s) { return `"${String(s).replace(/"/g, '\\"')}"` }

function __solExecbenchEvalPlan(ctx) {
  const substrateDir = ctx.substrateDir            // abs path to _substrate/integration
  const kernelSource = ctx.kernelSource            // path to candidate kernel on disk
  const contractEnv = ctx.contractEnv              // path to session contract.env
  const solutionOut = ctx.solutionOut              // where to write solution.json
  const benchOut = ctx.benchOut                    // where sol-execbench writes bench.jsonl
  const normalizedOut = ctx.normalizedOut || ''    // optional canonical measurement JSON
  const solCli = ctx.solCli                        // e.g. /abs/sol-execbench/.venv/bin/sol-execbench
  const taskDir = ctx.taskDir                      // FlashInfer-Bench/<task> dir
  const benchConfig = ctx.benchConfig              // --config path
  const seedDir = ctx.seedDir                      // cd target for the run
  const cvd = ctx.cudaVisibleDevices || '0'
  const ld = ctx.ldLibraryPath ? `LD_LIBRARY_PATH=${__solQ(ctx.ldLibraryPath)}:$LD_LIBRARY_PATH ` : ''
  const env = ctx.envPrefix ? `${String(ctx.envPrefix).trim()} ` : ''
  const definition = ctx.definitionPath ? ` --definition ${__solQ(ctx.definitionPath)}` : ''

  const pack = `rm -f -- ${__solQ(solutionOut)} && python3 ${__solQ(substrateDir + '/pack_sol_candidate.py')} --kernel ${__solQ(kernelSource)} --contract ${__solQ(contractEnv)} --out ${__solQ(solutionOut)}`
  const clearRunOutputs = [benchOut, normalizedOut].filter(Boolean).map(__solQ).join(' ')
  const run = `rm -f -- ${clearRunOutputs} && test -s ${__solQ(solutionOut)} && cd ${__solQ(seedDir)} && ${env}${ld}CUDA_VISIBLE_DEVICES=${cvd} ${__solQ(solCli)} ${__solQ(taskDir)}${definition} --solution ${__solQ(solutionOut)} --config ${__solQ(benchConfig)} -o ${__solQ(benchOut)}`
  const parse = `test -s ${__solQ(benchOut)} && python3 ${__solQ(substrateDir + '/parse_sol_bench.py')} ${__solQ(benchOut)} --contract ${__solQ(contractEnv)}${normalizedOut ? ` --out ${__solQ(normalizedOut)}` : ''}`

  return {
    pack,
    run,
    parse,
    order: ['pack', 'run', 'parse'],
    cleanupInvariant: 'solution.json + bench.jsonl are per-candidate scratch files in the run dir; each stage clears its own stale outputs and requires the preceding artifact. No project source is mutated (non-mutating method).',
  }
}

async function __solExecbenchEvaluate(ctx) {
  // Claude's legacy Workflow host does not yet expose this optional primitive.
  // Keep the prompt-driven path as a compatibility edge, while KerSor's Host
  // owns exact source materialization and PACK/RUN/PARSE without an LLM turn.
  if (typeof evaluate !== 'function') return null
  return evaluate({
    protocol: 'sol-execbench-v1',
    label: ctx.label || 'sol-eval',
    phase: ctx.phase || 'Evaluate',
    candidatePath: ctx.kernelSource,
    candidateSource: ctx.candidateSource,
    candidateLanguage: ctx.candidateLanguage || '',
    baselineSolutionPath: ctx.baselineSolutionPath || '',
    baselineEvaluationPath: ctx.baselineEvaluationPath || '',
    parentSolutionPath: ctx.parentSolutionPath || '',
    substrateDir: ctx.substrateDir,
    contractEnv: ctx.contractEnv,
    solutionOut: ctx.solutionOut,
    benchOut: ctx.benchOut,
    normalizedOut: ctx.normalizedOut || `${ctx.benchOut}.result.json`,
    solCli: ctx.solCli,
    taskDir: ctx.taskDir,
    benchConfig: ctx.benchConfig,
    seedDir: ctx.seedDir,
    cudaVisibleDevices: ctx.cudaVisibleDevices || '0',
    ldLibraryPath: ctx.ldLibraryPath || '',
    envPrefix: ctx.envPrefix || '',
    definitionPath: ctx.definitionPath || '',
    bindingOut: ctx.bindingOut || ctx.bindingPath || '',
    bindingWorkflow: ctx.bindingWorkflow || '',
    candidateId: ctx.candidateId || '',
    timeoutSeconds: ctx.timeoutSeconds || 0,
  }).then(__solGuardHarnessFault)
}

// A harness refusal occurs before a candidate is built. Preserve that boundary
// instead of spending a solver refine turn on a nonexistent compile failure.
function __solGuardHarnessFault(result) {
  const HARNESS_FAULTS = ['invalid_request', 'infrastructure_error']
  if (result && HARNESS_FAULTS.includes(result.failure_code)) {
    const detail = result.stderr || result.stdout || ''
    throw new Error(
      `sol-execbench harness fault (${result.failure_code}) at stage `
      + `${result.stage || 'unknown'}: ${String(detail).slice(0, 400)} `
      + '- this is a harness or wiring fault, not a candidate compile error',
    )
  }
  return result
}
// Deterministic measurement authority for native SOL consumers. Agent replies
// may explain evidence but cannot supply correctness, workload coverage or score.
function __solHostFeedback(host, explanation = {}) {
  const latency = host?.candidate_latency_aggregate_ms
  const valid = host?.compiled === true && host?.correct === true
    && host?.measurement_valid === true && host?.full_workload_set === true
    && host?.output_contract_valid === true && host?.artifact_binding?.verified === true
    && Number.isInteger(host?.n_total) && host.n_total > 0 && host.n_pass === host.n_total
    && Number.isFinite(latency) && latency > 0
    && Number.isFinite(host?.speedup_vs_seed) && host.speedup_vs_seed > 0
  return {
    compiled: host?.compiled === true, correct: valid,
    full_workload_set: host?.full_workload_set === true,
    measurement_valid: host?.measurement_valid === true,
    output_contract_valid: host?.output_contract_valid === true,
    speedup: valid ? host.speedup_vs_seed : 0,
    latency_ms: valid ? latency : null,
    passed_tests: Number.isInteger(host?.n_pass) ? host.n_pass : null,
    total_tests: Number.isInteger(host?.n_total) ? host.n_total : null,
    source_binding: valid ? host.artifact_binding : null,
    reference_speedup: Number.isFinite(host?.speedup) ? host.speedup : null,
    explanation: {error_message: explanation?.error_message || '',
      reward_hacking_flags: explanation?.reward_hacking_flags || []},
    measurement_owner: 'Host',
  }
}
async function __solOfficialSeedBaseline(ctx) {
  const seed = await __solExecbenchEvaluate(ctx)
  const latency = seed?.candidate_latency_aggregate_ms
  if (seed?.compiled !== true || seed?.correct !== true
      || seed?.full_workload_set !== true || seed?.measurement_valid !== true
      || seed?.output_contract_valid !== true || !Number.isFinite(latency) || latency <= 0) {
    throw new Error('Host could not establish a complete official Sol seed baseline')
  }
  return seed
}
// --- END sol-execbench-eval substrate ---


// --- sol-execbench wiring (Host-owned PACK/RUN/PARSE; no LLM turn) -----------
// These 15 workflows declared only the standalone path, so their benchmark was a
// command string embedded in a prompt for a read-only activation with no shell.
// The read-only allowlist deliberately excludes execution, so the agent could
// never run it.  The Host can, and this is the protocol it exposes for exactly
// that.
const SOL_CLI = args.sol_cli || ''
const SOL_TASK_DIR = args.sol_task_dir || ''
const SOL_BENCH_CONFIG = args.sol_bench_config || ''
const SOL_SEED_DIR = args.sol_seed_dir || ''
const SOL_CVD = args.sol_cuda_visible_devices || '0'
const SOL_LD_LIBRARY_PATH = args.sol_ld_library_path || ''
const SOL_ENV_PREFIX = args.sol_env_prefix || ''
const SOL_DEFINITION_PATH = args.sol_definition_path || ''
const SOL_SUBSTRATE_DIR = args.sol_substrate_dir || ''
const SOL_AVAILABLE = Boolean(SOL_CLI && SOL_TASK_DIR && SOL_SUBSTRATE_DIR)
// pack_sol_candidate requires a CUDA/C++ candidate to expose run() through
// PYBIND11_MODULE and rejects anything else with "has no PYBIND11_MODULE
// binding".  Workflows that state this produce packable candidates; those that
// only ask for "complete kernel code" comply by luck - stitchcuda packed on
// attempt 1 and failed on attempt 2 under the same prompt.  Say it once here.
const SOL_CANDIDATE_CONTRACT = SOL_AVAILABLE ? `
MANDATORY candidate shape: emit a COMPLETE, self-contained translation unit that
compiles on its own. Keep the seed kernel's entry point and bindings intact - the
same \`run(...)\` signature and the same \`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\`
block - and change only the implementation. Emit every line: no "same as above",
"unchanged", "rest byte-identical", or "..." standing in for code. The candidate
is compiled and packed exactly as given, so a description of a kernel fails where
the kernel itself would have been measured.` : ''

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

const WORKFLOW_NAME = 'cudallm-fsr-kernel-generation'


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

// --- BEGIN inlined typed-args (from _meta/scaffolding/typed-args.js) ---
// Cross-session priors travel here as a typed array (see KerSor
// agents/dispatch-arg-synthesizer.md), independent of op_description so the
// solver can treat them as distinct lower-authority signals.
const EXPERIENCE_EXCERPTS = Array.isArray(args.experience_excerpts) ? args.experience_excerpts : []
// Task requirements must reach every fresh activation independently of a
// previous agent's summary. The runtime owns file access and skill admission.
function __taskContractBlock() {
  const taskPath = args.problem_path || args.kernel_spec_path
  const inlineTask = args.problem_definition
  if (!taskPath && !inlineTask) return ''
  return '\n# Authoritative task contract\n'
    + (taskPath ? `Read the complete original task at ${taskPath} before acting, including its required skills and profiling instructions. Do not rely only on a prior agent summary.\n` : '')
    + (inlineTask ? `Complete caller-supplied problem definition:\n${typeof inlineTask === 'string' ? inlineTask : JSON.stringify(inlineTask, null, 2)}\n` : '')
    + (args.task_result_command ? `Task measurement ownership: use the declared task_result_command for all candidate tests: ${args.task_result_command}\nOnly that command creates the result file and its .artifact directory. Never pre-create, rename, delete, or write a result slot. Never run selftest.py or verify.py directly as a substitute for the declared command. Producers write candidate source files only; the task command freezes and evaluates them. Return the resulting complete contract object, including rejected candidates. Existing unknown slots require original-evidence reconciliation, not another GPU submission.\n` : '')
    + 'Retain all task constraints. Missing tools or unavailable task files must be reported explicitly; do not silently replace a required profiler.\n\n'
}
function __experienceBlock() {
  if (!EXPERIENCE_EXCERPTS.length) return ''
  const lines = EXPERIENCE_EXCERPTS.map(e => {
    const kind = (e && e.kind) || 'note'
    const directive = (e && e.directive) || 'inform'
    const claim = (e && e.claim) || (typeof e === 'string' ? e : JSON.stringify(e))
    return `- [${kind}/${directive}] ${claim}`
  })
  return `\n# Cross-session experience excerpts (channel ② — priors from past sessions; LOWER authority than current-round evidence):\n${lines.join('\n')}\n`
}

// Channel ③: typed prior-attempt context (attempt_evidence + attempt_plan).
// KerSor's dispatch-arg-synthesizer reads run-{N-1}/analysis.json and
// round-{N}-selection.json and emits both as typed JSON objects on args.
// Solvers consume them as a HIGHER-authority signal than HANDOFF prose.
const ATTEMPT_EVIDENCE = (args.attempt_evidence && typeof args.attempt_evidence === 'object') ? args.attempt_evidence : null
const ATTEMPT_PLAN = (args.attempt_plan && typeof args.attempt_plan === 'object') ? args.attempt_plan : null
// The hard "do not re-propose" constraint is owned by the cumulative transfer
// object, where a failed_strategy stops applying only when a later
// validated_win supersedes it. Deriving it from the previous round alone drops
// a strategy that failed in round 1 and simply was not retried in round 2.
// KerSor emits the cumulative ids as `failed_strategy_ids`; the per-round
// derivation stays as the fallback for a dispatch that predates that channel.
const FAILED_STRATEGY_IDS = Array.isArray(args.failed_strategy_ids)
  ? args.failed_strategy_ids.filter(id => typeof id === 'string' && id)
  : ((ATTEMPT_EVIDENCE && Array.isArray(ATTEMPT_EVIDENCE.transfer_items))
    ? ATTEMPT_EVIDENCE.transfer_items.filter(i => i && i.kind === 'failed_strategy' && i.id).map(i => i.id)
    : [])
function __attemptBlock() {
  if (!ATTEMPT_EVIDENCE && !ATTEMPT_PLAN) return ''
  const parts = ['\n# Prior attempt context (channel ③ — TYPED, machine-verified; HIGHER authority than handoff prose):']
  if (FAILED_STRATEGY_IDS.length > 0) {
    parts.push(`## HARD CONSTRAINT — do NOT re-propose any of these failed-strategy ids: ${FAILED_STRATEGY_IDS.join(', ')}`)
  }
  if (ATTEMPT_EVIDENCE) {
    const j = JSON.stringify(ATTEMPT_EVIDENCE, null, 2)
    parts.push('## Prior attempt evidence (last round):\n```json\n' + j + '\n```')
  }
  if (ATTEMPT_PLAN && Array.isArray(ATTEMPT_PLAN.candidate_plans)) {
    parts.push('## Routing-suggested candidate plans:\n```json\n' + JSON.stringify({phase_intent: ATTEMPT_PLAN.phase_intent, candidate_plans: ATTEMPT_PLAN.candidate_plans}, null, 2) + '\n```')
  }
  return parts.join('\n') + '\n'
}
// --- END inlined typed-args ---

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
      // Retrying cannot grant permission, repair a contract, replenish a
      // budget, or undo cancellation. Preserve the original Host diagnostic.
      if (e && (e.retryable === false || e.name === 'AbortError'
        || /(?:PERMISSION|CONFIG_INVALID|REQUEST_INVALID|UNSUPPORTED|DISABLED|BUDGET|QUOTA|AUTH|FILESYSTEM_SANDBOX|WORKFLOW_INTERRUPTED|ABORT_ERR)/.test(String(e.code || '')))) throw e
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

// --- BEGIN inlined turn-timeout scaffolding (from _meta/scaffolding/turn-timeout.js) ---
const TURN_TIMEOUT_MS = (args.turn_timeout_min || 12) * 60 * 1000  // per-turn wall-clock cap

/**
 * Wrap a doer-turn promise with a wall-clock cap. On expiry the returned
 * promise rejects with `turn-timeout: <label> exceeded Ns`. Degrades to a
 * passthrough when the runtime has no timers or TURN_TIMEOUT_MS <= 0.
 */
function withTurnTimeout(promise, label) {
  if (typeof setTimeout !== 'function' || !(TURN_TIMEOUT_MS > 0)) return promise
  let timer
  const guard = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`turn-timeout: ${label} exceeded ${Math.round(TURN_TIMEOUT_MS / 1000)}s`)),
      TURN_TIMEOUT_MS)
  })
  return Promise.race([promise, guard]).finally(() => {
    if (typeof clearTimeout === 'function') clearTimeout(timer)
  })
}
// --- END inlined turn-timeout scaffolding ---
// --- genome self-report: INLINE (rich, doer-written) ---
// Each phase's doer appends a rich line to <exp_dir>/genome.jsonl as its final
// action. The "__genomeReport" mention is a sentinel so patch-genome-report.js
// treats this file as already handled. See _meta/genome-trajectory-schema.md.


// --- BEGIN inlined backend-axis (resolve) scaffolding (from _meta/scaffolding/backend-axis.js) ---
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
// --- END inlined backend-axis (resolve) scaffolding ---


// =============================================================================
// CUDA-LLM — Feature Search and Reinforcement (FSR) Workflow
// =============================================================================
//
// Source: "CUDA-LLM: LLMs Can Write Efficient CUDA Kernels"
//         arXiv:2506.09092 — Wentao Chen, Jiace Zhu, Qi Fan, Yehan Ma, An Zou
//
// Boundary:
//   This workflow implements an agent-executable Feature Search and
//   Reinforcement loop. It does not train a model. It searches CUDA optimization
//   features, generates kernels, evaluates them with real compile/correctness/
//   latency evidence, and reinforces feature choices for later iterations.
// adaptation_scope: workflow_adaptation — this is not the full CUDA-LLM training
// or model-development pipeline.
//
// Usage:
//   Workflow({name: 'cudallm-fsr-kernel-generation', args: {
//     problem_path: '/path/to/task.md',
//     reference_code_path: '/path/to/reference.py',
//     benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
//     target_gpu: 'H100',
//     iterations: 8,
//     feature_budget: 4,
//     samples_per_feature_set: 2,
//     rtol: 0.01,
//     atol: 0.01,
//     exp_dir: '/tmp/cudallm_fsr_exp',
//   }})
//
// Evaluator JSON contract:
//   benchmark_command should write JSON at {result_path}:
//   {
//     "compiled": true,
//     "correct": true,
//     "speedup": 1.23,
//     "latency_ms": 0.12,
//     "baseline_latency_ms": 0.15,
//     "error_message": "",
//     "passed_tests": 128,
//     "total_tests": 128
//   }
//
// =============================================================================

// --- Required Args ---
const PROBLEM_DEFINITION = args.problem_definition || ''
const TASK_SPEC_PATH = args.problem_path || ''
const REFERENCE_CODE_PATH = args.reference_code_path || ''
const EVAL_CMD = args.benchmark_command || ''

// --- Optional Args ---
const TARGET_GPU = args.target_gpu || 'H100'
const ITERATIONS = args.iterations || 8
const FEATURE_BUDGET = args.feature_budget || 4
const SAMPLES_PER_FEATURE_SET = args.samples_per_feature_set || 2
const RTOL = args.rtol ?? 0.01
const ATOL = args.atol ?? 0.01
const EXP_DIR = args.exp_dir || '/tmp/cudallm_fsr_exp'
const DRIVER_PROBLEM_PATH = args.problem_json_path || `${EXP_DIR}/driver_problem.json`
const PROFILE_SOURCE_PATH = args.profile_source_path || REFERENCE_CODE_PATH || ''
const ADAPTATION_SCOPE = 'workflow_adaptation'
const INPUT_MODE = args.task_result_command && args.kernel_path ? 'optimize_existing' : 'generate_then_optimize'

// --- Backend driver wiring (P5c Stage B; off-by-default; legacy path byte-identical) ---
const BACKEND_DIR = args.backend_dir || ''
const SUBSTRATE = args.substrate_dir || '_substrate'
const SH = args.driver_shell_prefix || ''
const PY = args.substrate_command_prefix || ''
const LEGACY_SETUP_LANG_TOKEN = 'CUDA'
const LEGACY_CATALOG_LANG_TOKEN = 'CUDA'
const LEGACY_TESTS_LANG_TOKEN = 'CUDA-LLM'
const LEGACY_SELECT_LANG_TOKEN = 'CUDA'
const LEGACY_GENERATE_LANG_TOKEN = 'CUDA'
const LEGACY_EVAL_LANG_TOKEN = 'CUDA'
const LEGACY_REINFORCE_LANG_TOKEN = 'CUDA'
const LEGACY_REPORT_LANG_TOKEN = 'CUDA-LLM FSR'
const LEGACY_SOURCE_EXT = '.cu'
const LEGACY_RESULT_EXT = '.json'
const LEGACY_PURE_LANG_PHRASE = 'pure CUDA/C++ only'
const LEGACY_FENCE_TOKEN = 'cuda'
// L3 deferred (R2): triton driver has no `feature_catalog` idiom today; the
// driver path falls back to LEGACY_TRITON_FEATURE_FALLBACK below. Tightening
// is filed as a P5e/P5f L3 follow-up per P5c plan §5.2 B2 + §8 R2.
const LEGACY_TRITON_FEATURE_FALLBACK = 'Explore the standard Triton optimization idioms appropriate to this driver (block tiling, vectorized loads, masking, reductions, autotune configs).'
const JSON_PASSTHROUGH = { type: 'object', additionalProperties: true }

// --- Project-native integration (embedded operators via integration-strategist) ---
// For inference-engine embedded operators (e.g. llama.cpp .cuh) that cannot
// compile as a standalone TU; built/tested/benchmarked inside the host project.
// EVAL_CMD is the existing standalone evaluator; PROJECT_BENCH_CMD is the
// project-native benchmark (falls back to EVAL_CMD when not provided).
const PROJECT_ROOT = args.project_root || args.ggml_root || ''
const BUILD_CMD = args.build_command || ''
const PROJECT_BENCH_CMD = args.project_benchmark_command || EVAL_CMD || ''
const REGISTER_SCRIPT = args.register_script || ''

// --- BEGIN inlined backend-axis (driver) scaffolding (from _meta/scaffolding/backend-axis.js) ---
function driverPath(rel) { return `${BACKEND_DIR}/${rel}` }
function driverSh(script, cliArgs) {
  return `Run exactly: \`${SH ? SH + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}
// --- END inlined backend-axis (driver) scaffolding ---
function driverPy(script, cliArgs) {
  return `Run exactly: \`${PY ? PY + ' ' : ''}${BACKEND_DIR}/${script} ${cliArgs}\`.`
}
function driverProfileOutPath(iter, sample) {
  // .sqlite suffix satisfies cuda/profile.sh nsys fallback; ncu still writes CSV bytes here.
  return `${EXP_DIR}/cudallm_iter_${iter}_sample_${sample}.prof.sqlite`
}
function buildProfileShArgs(buildOut, iter, sample) {
  let cli = `--artifact ${buildOut} --problem ${DRIVER_PROBLEM_PATH} --out ${driverProfileOutPath(iter, sample)}`
  if (PROFILE_SOURCE_PATH) cli += ` --source ${PROFILE_SOURCE_PATH}`
  return cli
}
function profileStepFooter() {
  if (DRIVER_BACKEND_ID === 'cuda') {
    return 'The cuda driver profile.sh may return format ncu-csv or nsys-sqlite depending on which profiler is available.\n'
  }
  return 'Pass the format field from profile.sh stdout through to to_evidence.py.\n'
}
function buildToEvidencePrompt(profilePointer) {
  if (!profilePointer || profilePointer.ok === false) {
    const why = profilePointer?.error || profilePointer?.profiler || 'profiler unavailable'
    return `Profiler unavailable (${why}). ` +
      `Return {ok:true, metrics:{latency_ms:null,dram_pct:null,sm_pct:null,occupancy:null,_vendor:"nvidia"}, ` +
      `coverage:[], source_backend:"${DRIVER_BACKEND_ID}"}.`
  }
  const native = profilePointer.native_profile
  const format = profilePointer.format || (DRIVER && DRIVER.profiler_format) || 'ncu-csv'
  return `${driverPy('to_evidence.py', `--native ${native} --format ${format}`)}\n` +
    `Return stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}.`
}

let DRIVER = null
let DRIVER_LANG_FENCE = 'cuda'
let DRIVER_IMPL_REQUIREMENTS = ''
let DRIVER_SOURCE_EXT = LEGACY_SOURCE_EXT
let DRIVER_FEATURE_CATALOG = ''
let DRIVER_BACKEND_ID = RESOLVED_BACKEND || ''
// --- profiling-strategist: pick the analysis METHOD per backend x task x host,
// then honor it at the profile step below. The agent only classifies the task
// (fuzzy op_class/size); the substrate stamps confidence by method
// (measured/inferred/hypothesized) -- not the agent. See
// _substrate/profiling/README.md. Falls back to native_profiler if undecided. ---
let PROFILING_DECISION = { method: 'native_profiler', confidence: 'measured', normalizer: 'to_evidence.py' }

function langToken(legacy) {
  return args.task_result_command ? (args.language || legacy) : (USE_DRIVER ? DRIVER_LANG_FENCE : legacy)
}
function pureLangPhrase() {
  if (args.task_result_command) return `pure ${args.language || 'CuTe DSL'} Python only`
  return USE_DRIVER ? `pure ${DRIVER_LANG_FENCE} only` : LEGACY_PURE_LANG_PHRASE
}
function fenceToken() {
  return USE_DRIVER ? DRIVER_LANG_FENCE : LEGACY_FENCE_TOKEN
}
function cudallmCandidatePath(iter, sample) {
  const ext = args.task_result_command ? '.py' : (USE_DRIVER ? DRIVER_SOURCE_EXT : LEGACY_SOURCE_EXT)
  return `${EXP_DIR}/cudallm_iter_${iter}_sample_${sample}${ext}`
}
function cudallmResultPath(iter, sample) {
  return `${EXP_DIR}/cudallm_iter_${iter}_sample_${sample}${LEGACY_RESULT_EXT}`
}

// --- State ---
let taskSpec = ''
let referenceCode = ''
let tests = []
let featureCatalog = []
let featureScores = {}
let candidates = []
let bestCandidate = null

// Task-owned files retain complete source and evidence. Selection/reinforcement
// carry an index, not repeated source bodies or compiler output. No byte cap.
function candidatePromptRecord(candidate) {
  if (!args.task_result_command) return candidate
  const e = candidate.eval || {}
  return {
    id: candidate.id,
    selected_feature_ids: candidate.selected_feature_ids,
    implemented_feature_ids: candidate.implemented_feature_ids,
    source_path: e.host_candidate_path || candidate.path,
    task_result_path: e.test_result_path,
    evidence_paths: e.evidence || null,
    measurement: {compiled:e.compiled,correct:e.correct,is_valid:e.is_valid,
      measurement_valid:e.measurement_valid,speedup:e.speedup,latency_ms:e.latency_ms,
      n_pass:e.n_pass,n_total:e.n_total,outcome_state:e.outcome_state},
  }
}
const CANDIDATE_FILE_GUIDANCE = args.task_result_command
  ? 'The following records index authoritative workspace files. Read task_result_path for COMPLETE diagnostics and source_path for COMPLETE code when comparing or changing a candidate. Follow evidence_paths to original per-workload traces. Do not infer missing data from this index. No original file has been truncated. Keep feature-score records as feature statistics and link evidence by path; do not copy source or raw diagnostics into score notes.'
  : ''


function initFeatureScore(feature) {
  if (!featureScores[feature.id]) {
    featureScores[feature.id] = {
      id: feature.id,
      name: feature.name,
      attempts: 0,
      compiled: 0,
      correct: 0,
      reward: 0,
      best_speedup: 0,
      failures: [],
    }
  }
}

function candidateScore(candidate) {
  if (!candidate?.eval?.compiled) return [0, 0, 0]
  if (!candidate.eval.correct) return [1, 0, 0]
  return [1, 1, candidate.eval.speedup || 0]
}

function isBetterCandidate(candidate, incumbent) {
  if (!incumbent) return !!candidate?.eval?.correct
  const a = candidateScore(candidate)
  const b = candidateScore(incumbent)
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return a[i] > b[i]
  }
  return false
}

// =============================================================================
// Phase 1: Setup
// =============================================================================
phase('Setup')

if (USE_DRIVER) {
  DRIVER = await agentRetry(() => agent(
    `${__taskContractBlock()}Load the backend driver at ${BACKEND_DIR} and return its manifest plus idioms verbatim.\n` +
    `1. Run exactly: \`cat ${driverPath('manifest.json')}\` and parse JSON.\n` +
    `2. Run exactly: \`cat ${driverPath('idioms.json')}\` and parse JSON.\n` +
    `Return {present, backend_id, source_ext, aux_ext, lang_fence, impl_requirements, methods, feature_catalog, ` +
    `hw_vendor, profiler_name, profiler_format}. ` +
    `Set profiler_name/profiler_format from manifest.profiler when present.`,
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
  DRIVER_FEATURE_CATALOG = DRIVER.feature_catalog || ''
  DRIVER_BACKEND_ID = DRIVER.backend_id || DRIVER_BACKEND_ID
  DRIVER.profiler_name = DRIVER.profiler_name || (DRIVER.profiler && DRIVER.profiler.name) || null
  DRIVER.profiler_format = DRIVER.profiler_format || (DRIVER.profiler && DRIVER.profiler.format) || 'ncu-csv'
  log(`Driver loaded: ${DRIVER_BACKEND_ID} (fence=${DRIVER_LANG_FENCE}, profiler=${DRIVER.profiler_name || 'none'})`)

  // profiling-strategist: classify the task (fuzzy op_class/size from the
  // reference/source kernel) and let the substrate pick method + stamp
  // confidence. Computed once per task; PROFILING_DECISION gates the
  // profile.sh / ncu branch in the Evaluate loop below.
  const _pd = await agentRetry(() => agent(
    `${__taskContractBlock()}Read ${PROFILE_SOURCE_PATH || REFERENCE_CODE_PATH || TASK_SPEC_PATH}; classify its op_class (one of attention|gemm|elementwise|reduction|default) and size (tiny|small|large). Then ` +
    `run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/profiling_strategist.py resolve --backend-manifest ${BACKEND_DIR}/manifest.json --task <op_class> --size <size> --cache ${EXP_DIR}/prof_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`.\n` +
    `Return its stdout JSON verbatim {method, confidence, normalizer, profiler_name, rationale}.`,
    { model: MODEL.mechanical, label: 'profiling-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  if (_pd && _pd.method) PROFILING_DECISION = _pd
  log(`Profiling-strategist: method=${PROFILING_DECISION.method} confidence=${PROFILING_DECISION.confidence} normalizer=${PROFILING_DECISION.normalizer || 'none'}`)
}

// --- integration-strategist: route build/test mode (standalone vs embedded_*).
// Generalist-level embedded treatment: an inference-engine operator (e.g.
// llama.cpp .cuh) cannot compile as a standalone TU, so the strategist may
// route to embedded_inplace / embedded_dispatch. Default is standalone, so the
// legacy candidate-eval path is byte-identical when method=standalone. ---
// Honour an explicit caller declaration.  Without this the guard below only stops
// the model from changing the decision; the declaration itself still had no effect,
// so a caller asking for sol_execbench_solution silently got standalone.
let INTEGRATION_DECISION = {
  method: args.integration_pattern || 'standalone',
  build_fidelity: args.integration_pattern === 'sol_execbench_solution' ? 'production' : 'isolated',
  reversible: true,
}
if (!args.integration_pattern) {

  const _kernelForInteg = REFERENCE_CODE_PATH || PROFILE_SOURCE_PATH || TASK_SPEC_PATH
  const _probe = JSON.stringify({ compiler: true, project_build: !!BUILD_CMD, register_script: !!REGISTER_SCRIPT, runtime_registry: false, reversibility_net: true })
  const _integ = await agentRetry(() => agent(
    `${__taskContractBlock()}Read ${_kernelForInteg}; classify can_compile_standalone as exactly one of yes|no|uncertain ` +
    `(use no when the file cannot compile as a single TU — e.g. llama.cpp .cuh with project-only deps). Then ` +
    `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/integration/integration_strategist.py resolve ` +
    `--kernel "${_kernelForInteg}" --can-standalone <yes|no|uncertain> --host-probe '${_probe}' ` +
    `--cache ${EXP_DIR}/integ_cache.json --trajectory ${EXP_DIR}/genome.jsonl\`. ` +
    `Return its stdout JSON verbatim {method, build_fidelity, reversible, eval_mechanism, rationale}.`,
    { model: MODEL.mechanical, label: 'integration-strategist', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
  // Preserve an explicit integration choice; adopt only a supported discovery.
  if (_integ && ['standalone', 'embedded_inplace', 'embedded_dispatch', 'sol_execbench_solution', 'derive_adapter'].includes(_integ.method)) {
    INTEGRATION_DECISION = _integ
  }
}
log(`integration method = ${INTEGRATION_DECISION.method} (fidelity=${INTEGRATION_DECISION.build_fidelity || 'n/a'})`)
if (INTEGRATION_DECISION.method === 'derive_adapter') {
  throw new Error('integration-strategist returned derive_adapter — provide project_root + build/test commands')
}
const USE_DRIVER_STANDALONE = USE_DRIVER && INTEGRATION_DECISION.method === 'standalone'
const IS_EMBEDDED = INTEGRATION_DECISION.method === 'embedded_inplace' || INTEGRATION_DECISION.method === 'embedded_dispatch'
const ORIGINAL_BACKUP = INTEGRATION_DECISION.method === 'embedded_inplace' ? `${EXP_DIR}/integ_original.backup` : ''
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`${__taskContractBlock()}Byte-exact backup: run \`cp -a "${REFERENCE_CODE_PATH || PROFILE_SOURCE_PATH || TASK_SPEC_PATH}" "${ORIGINAL_BACKUP}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-backup-original', phase: 'Setup', schema: JSON_PASSTHROUGH }), { retries: 5 })
}
// native_profiler downgrade keyed on !USE_DRIVER_STANDALONE: when running the
// embedded path there is no standalone artifact to feed a native profiler, so
// fall back to perf_heuristic on project-native throughput.
if (PROFILING_DECISION.method === 'native_profiler' && !USE_DRIVER_STANDALONE) {
  log(`profiling: native_profiler but not standalone driver path -> downgrade to perf_heuristic`)
  PROFILING_DECISION = { method: 'perf_heuristic', confidence: 'inferred', normalizer: 'perf_to_evidence.py',
    profiler_name: 'project-native-perf', rationale: 'native_profiler but embedded/non-standalone path -> perf_heuristic' }
}

const setup = await agentRetry(() => agent(`${__taskContractBlock()}You are a ${langToken(LEGACY_SETUP_LANG_TOKEN)} kernel generation expert. Read and structure this CUDA-LLM task.

# Inputs
- problem_definition: ${PROBLEM_DEFINITION || '(not provided)'}
- problem_path: ${TASK_SPEC_PATH}
- reference_code_path: ${REFERENCE_CODE_PATH}
- target_gpu: ${TARGET_GPU}
- benchmark_command: ${EVAL_CMD || '(missing; evaluator evidence required before accepting speedup)'}
- tolerances: rtol=${RTOL}, atol=${ATOL}

# Tasks
1. Read problem_path and reference implementation when provided.
2. Use problem_definition as the authoritative task specification when provided.
3. Identify operation type, tensor shapes/dtypes, layout assumptions, and expected output.
4. Identify hard constraints: ${pureLangPhrase()}, no PyTorch fallback in generated kernel, preserve numerical tolerance.
5. State the evaluator JSON contract and how {kernel_path}/{result_path} are substituted.
6. List baseline performance if available.

Return structured task information.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Setup","ts":"<ts>","status":"done","technique":"task_setup","note":"<operation type + key constraints/contract, one line>"}`, {
  label: 'setup-task',
  phase: 'Setup',
  schema: {
    type: 'object',
    properties: {
      problem_definition: { type: 'string' },
      reference_code: { type: 'string' },
      operation_type: { type: 'string' },
      input_contract: { type: 'string' },
      output_contract: { type: 'string' },
      constraints: { type: 'array', items: { type: 'string' } },
      baseline_latency_ms: { type: 'number' },
    },
    required: ['problem_definition', 'operation_type', 'constraints'],
  },
}), { retries: 5 })

taskSpec = PROBLEM_DEFINITION || setup.problem_definition || ''
referenceCode = setup.reference_code || ''

let solSeedMeasurement = null
if (SOL_AVAILABLE) {
  solSeedMeasurement = await __solOfficialSeedBaseline({
    label: 'sol-seed-baseline', phase: 'Setup', substrateDir: SOL_SUBSTRATE_DIR,
    kernelSource: `${EXP_DIR}/host_seed.cu`, baselineSolutionPath: `${SOL_SEED_DIR}/seed.solution.json`,
    contractEnv: `${SOL_SEED_DIR}/contract.env`, solutionOut: `${EXP_DIR}/host_seed.solution.json`,
    benchOut: `${EXP_DIR}/host_seed.bench.jsonl`, solCli: SOL_CLI,
    taskDir: SOL_TASK_DIR, benchConfig: SOL_BENCH_CONFIG, seedDir: SOL_SEED_DIR,
    cudaVisibleDevices: SOL_CVD, ldLibraryPath: SOL_LD_LIBRARY_PATH,
    envPrefix: SOL_ENV_PREFIX, definitionPath: SOL_DEFINITION_PATH,
  })
}

// =============================================================================
// Phase 2: FeatureCatalog
// =============================================================================
phase('FeatureCatalog')

const LEGACY_FEATURE_CATALOG = `# Required feature families
- tiling and block/grid decomposition
- shared memory staging
- vectorized/global memory access
- warp-level primitives
- loop unrolling and instruction scheduling
- occupancy/register pressure tuning
- fast math or CUDA intrinsics, only when tolerance allows
- boundary handling / tail masking`

const catalog = await agentRetry(() => agent(`${__taskContractBlock()}Build a ${langToken(LEGACY_CATALOG_LANG_TOKEN)} optimization feature catalog for Feature Search and Reinforcement.

# Task
${taskSpec}

# Reference
\`\`\`
${referenceCode}
\`\`\`

# Target GPU
${TARGET_GPU}

${USE_DRIVER ? (DRIVER_FEATURE_CATALOG || LEGACY_TRITON_FEATURE_FALLBACK) : LEGACY_FEATURE_CATALOG}

# Tasks
1. Produce feature entries with id, name, family, description, prerequisites, incompatibilities, and risk.
2. Mark features that are unsafe under the given tolerance.
3. Include a conservative baseline feature set.
4. Initialize all feature scores with neutral priors.

Return feature catalog.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"FeatureCatalog","ts":"<ts>","status":"done","technique":"feature_search_space","note":"<count + the main feature families in the catalog, one line>"}`, {
  label: 'feature-catalog',
  phase: 'FeatureCatalog',
  schema: {
    type: 'object',
    properties: {
      features: { type: 'array', items: { type: 'object' } },
      baseline_feature_ids: { type: 'array', items: { type: 'string' } },
      unsafe_feature_ids: { type: 'array', items: { type: 'string' } },
      notes: { type: 'array', items: { type: 'string' } },
    },
    required: ['features', 'baseline_feature_ids'],
  },
}), { retries: 5 })

featureCatalog = catalog.features || []
for (const feature of featureCatalog) initFeatureScore(feature)

let taskParentMeasurement = null
if (args.task_result_command && args.kernel_path) {
  taskParentMeasurement = await __nativeTaskAcceptedParent({candidatePath:args.kernel_path,
    resultPath:`${EXP_DIR}/accepted_parent.task.json`,command:args.task_result_command,
    workloadCount:args.task_workload_count,label:'accepted-parent-task'})
  bestCandidate = {id:'accepted-parent',path:taskParentMeasurement.host_candidate_path,
    selected_feature_ids:[],implemented_feature_ids:[],code:'',eval:taskParentMeasurement}
}

// =============================================================================
// Phase 3: GenerateTests
// =============================================================================
phase('GenerateTests')

const testPlan = await agentRetry(() => agent(`${__taskContractBlock()}Generate diverse correctness tests for this CUDA-LLM task.

# Operation
${setup.operation_type}

# Input contract
${setup.input_contract || ''}

# Output contract
${setup.output_contract || ''}

# Tolerances
rtol=${RTOL}, atol=${ATOL}

# Requirements
1. Cover small, medium, and large shapes.
2. Cover boundary/tail cases that stress masking and vectorized loads.
3. Cover dtype/layout variations if the task allows them.
4. Include random and adversarial value distributions.
5. These tests define what the user-provided benchmark_command should verify; model self-judgment is not enough.

Return test cases.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"GenerateTests","ts":"<ts>","status":"done","technique":"correctness_test_suite","note":"<count of tests + shapes/boundary cases covered, one line>"}`, {
  label: 'generate-tests',
  phase: 'GenerateTests',
  schema: {
    type: 'object',
    properties: {
      test_cases: { type: 'array', items: { type: 'object' } },
      tolerance_policy: { type: 'string' },
      harness_notes: { type: 'string' },
    },
    required: ['test_cases', 'tolerance_policy'],
  },
}), { retries: 5 })

tests = testPlan.test_cases || []

// =============================================================================
// Main FSR Loop
// =============================================================================
for (let iteration = 0; iteration < ITERATIONS; iteration++) {
  for (let sample = 0; sample < SAMPLES_PER_FEATURE_SET; sample++) {
    log(`\n=== CUDA-LLM FSR iteration ${iteration + 1}/${ITERATIONS}, sample ${sample + 1}/${SAMPLES_PER_FEATURE_SET} ===`)

    phase('SelectFeatures')

    const selection = await agentRetry(() => agent(`${__taskContractBlock()}Select a ${langToken(LEGACY_SELECT_LANG_TOKEN)} feature combination for the next candidate.

# Feature catalog
\`\`\`json
${JSON.stringify(featureCatalog, null, 2)}
\`\`\`

# Feature scores
\`\`\`json
${JSON.stringify(featureScores, null, 2)}
\`\`\`

# Recent candidates${CANDIDATE_FILE_GUIDANCE ? '\n' + CANDIDATE_FILE_GUIDANCE : ''}
\`\`\`json
${JSON.stringify(candidates.slice(-8).map(candidatePromptRecord), null, 2)}
\`\`\`

# Selection rules
1. Select at most ${FEATURE_BUDGET} features.
2. Combine compatible features only.
3. Use exploration early: try under-tested but plausible features.
4. Use exploitation when feature evidence shows compile/correctness/speedup reward.
5. Avoid unsafe features unless explicitly justified by tolerance and tests.

Return selected feature ids and rationale.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"SelectFeatures","ts":"<ts>","status":"done","candidate_id":"iter${iteration}-s${sample}","technique":"<the selected feature combination as a +-joined list>","note":"<exploration vs exploitation + selection rationale, one line>"}`, {
      label: `select-features-${iteration}-${sample}`,
      phase: 'SelectFeatures',
      schema: {
        type: 'object',
        properties: {
          selected_feature_ids: { type: 'array', items: { type: 'string' } },
          rationale: { type: 'string' },
          expected_risks: { type: 'array', items: { type: 'string' } },
        },
        required: ['selected_feature_ids', 'rationale'],
      },
    }), { retries: 5 })

    phase('GenerateKernel')

    const generation = await agentRetry(() => agent(`${__taskContractBlock()}Generate a ${langToken(LEGACY_GENERATE_LANG_TOKEN)} kernel using the selected CUDA-LLM FSR features.

# Task specification
${taskSpec}

# Reference implementation
\`\`\`
${referenceCode}
\`\`\`

${taskParentMeasurement ? `# Independently accepted parent\nRead the complete measured parent at ${taskParentMeasurement.host_candidate_path}. Use it as the starting implementation when applying the selected FSR features. Reference-relative score: ${taskParentMeasurement.speedup}; retain this parent unless an officially measured candidate improves it.` : ''}

# Selected features
\`\`\`json
${JSON.stringify(selection, null, 2)}
\`\`\`

# Hard constraints
1. Return complete ${args.task_result_command ? (args.language + ' Python source') : (USE_DRIVER ? `${DRIVER_LANG_FENCE} source` : 'CUDA/C++ source')}, not a patch.
${args.task_result_command ? `Write the COMPLETE source to ${cudallmCandidatePath(iteration,sample)} and return variant_path. The file is authoritative; candidate_code is display only. Preserve the frozen task Python run entry point and use CuTe DSL when requested.` : ''}
2. Do not call PyTorch or reference implementation from generated kernel.
3. Preserve input/output contract and tolerances.
4. Implement selected features concretely; if a feature is skipped, explain why.
5. Keep code benchmarkable by benchmark_command.${USE_DRIVER && DRIVER_IMPL_REQUIREMENTS ? `\n6. ${DRIVER_IMPL_REQUIREMENTS}` : ''}

${SOL_CANDIDATE_CONTRACT}

Return candidate code and implemented features.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"GenerateKernel","ts":"<ts>","status":"done","candidate_id":"iter${iteration}-s${sample}","technique":"<the implemented feature combination as a +-joined list>","note":"<what was implemented vs skipped and why, one line>"}`, {
      label: `generate-kernel-${iteration}-${sample}`,
      phase: 'GenerateKernel',
      schema: {
        type: 'object',
        properties: {
          candidate_code: { type: 'string' },
          variant_path: { type: 'string' },
          implemented_feature_ids: { type: 'array', items: { type: 'string' } },
          skipped_feature_ids: { type: 'array', items: { type: 'string' } },
          implementation_notes: { type: 'string' },
        },
        required: ['candidate_code', 'implemented_feature_ids'],
      },
    }), { retries: 5 })

    phase('Evaluate')

    // Rule 3 below tells the agent to report compiled=false, speedup=0 and explain
    // the missing evidence when it cannot benchmark, which is the honest outcome
    // but still leaves the candidate unmeasured: a read-only activation has no
    // execution tool, so "materialize the kernel and run it" cannot happen.
    // Measure on the Host first and hand the agent the evidence it was asked for.
    const __taskMeasured = args.task_result_command
      ? await __nativeTaskEvaluate({candidatePath:generation.variant_path || cudallmCandidatePath(iteration,sample),candidateSource:generation.candidate_code,resultPath:cudallmResultPath(iteration,sample),command:args.task_result_command,workloadCount:args.task_workload_count,label:`task-eval-${iteration}-${sample}`}) : null
    let __hostMeasured = null
    if (SOL_AVAILABLE && (generation.candidate_code || '').trim()) {
      __hostMeasured = await __solExecbenchEvaluate({
        label: `sol-eval-${iteration}-${sample}`, phase: 'Evaluate',
        substrateDir: SOL_SUBSTRATE_DIR,
        kernelSource: cudallmCandidatePath(iteration, sample),
        candidateSource: generation.candidate_code,
        baselineEvaluationPath: `${EXP_DIR}/host_seed.bench.jsonl.result.json`,
        parentSolutionPath: `${SOL_SEED_DIR}/seed.solution.json`,
        bindingOut: `${EXP_DIR}/cudallm_iter_${iteration}_sample_${sample}.binding.json`,
        bindingWorkflow: WORKFLOW_NAME, candidateId: `iter_${iteration}_sample_${sample}`,
        contractEnv: `${SOL_SEED_DIR || '.'}/contract.env`,
        solutionOut: `${EXP_DIR}/cudallm_iter_${iteration}_sample_${sample}.solution.json`,
        benchOut: `${EXP_DIR}/cudallm_iter_${iteration}_sample_${sample}.bench.jsonl`,
        solCli: SOL_CLI, taskDir: SOL_TASK_DIR, benchConfig: SOL_BENCH_CONFIG,
        seedDir: SOL_SEED_DIR, cudaVisibleDevices: SOL_CVD,
        ldLibraryPath: SOL_LD_LIBRARY_PATH, envPrefix: SOL_ENV_PREFIX,
        definitionPath: SOL_DEFINITION_PATH,
      })
      if (__hostMeasured) {
        log(`Host-measured ${iteration}-${sample}: compiled=${__hostMeasured.compiled} `
          + `correct=${__hostMeasured.correct} speedup=${__hostMeasured.speedup} `
          + `workloads=${__hostMeasured.n_pass}/${__hostMeasured.n_total}`)
      }
    }
    const __measuredBlock = __hostMeasured ? `

# ALREADY MEASURED ON THE HOST - this is the evidence, use it verbatim
compiled=${__hostMeasured.compiled} correct=${__hostMeasured.correct}
speedup=${__hostMeasured.speedup} latency_ms=${__hostMeasured.latency_ms}
workloads_passed=${__hostMeasured.n_pass}/${__hostMeasured.n_total}
${__hostMeasured.failure_code ? `failure_code=${__hostMeasured.failure_code}` : ''}
Rule 3 does not apply: the evidence exists. Report these numbers and base the
reward on them. Still report any reward-hacking signs you see in the code.` : ''

    const evaluation = args.task_result_command
      ? await agentRetry(() => agent(`${__taskContractBlock()}Interpret this already-executed task result: ${JSON.stringify(__taskMeasured)}. Do not run another benchmark, edit the artifact, or report replacement numeric measurements. Explain failure/diagnostics and any reward-hacking signs.`, {
        label:`evaluate-${iteration}-${sample}`,phase:'Evaluate',
        schema:{type:'object',properties:{error_message:{type:'string'},reward_hacking_flags:{type:'array',items:{type:'string'}}},required:['error_message','reward_hacking_flags']},
      }),{retries:0})
      : await agentRetry(() => agent(`${__taskContractBlock()}Evaluate this ${langToken(LEGACY_EVAL_LANG_TOKEN)} candidate with compile, correctness, and latency evidence.

# Candidate code
\`\`\`${fenceToken()}
${(generation.candidate_code || '')}
\`\`\`

# Eval command
${EVAL_CMD || '(no benchmark_command provided)'}${__measuredBlock}

# Paths
- kernel_path: ${cudallmCandidatePath(iteration, sample)}
- result_path: ${cudallmResultPath(iteration, sample)}

# Tests
\`\`\`json
${JSON.stringify(tests, null, 2)}
\`\`\`

# Required behavior
1. If benchmark_command is available, materialize the kernel and run it with {kernel_path}/{result_path}.
2. Parse evaluator JSON.
3. If benchmark_command is unavailable, set compiled=false, correct=false, speedup=0 and explain missing evidence.
4. Reward must be based on compile success, functional correctness over diverse tests, and measured latency.
5. Report suspected reward-hacking signs: hardcoded shapes, skipped computation, PyTorch fallback, or ignored inputs.

Return evaluator result.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append, using the values you just measured (status="done" only if compiled AND correct, else "error"; speedup is the measured speedup number, or null if unavailable):
{"workflow":"${WORKFLOW_NAME}","phase":"Evaluate","ts":"<ts>","status":"<done|error>","candidate_id":"iter${iteration}-s${sample}","speedup":<number or null>,"technique":"<the feature combination under test as a +-joined list>","note":"<compiled? correct? passed/total tests; or the failure reason>"}`, {
      label: `evaluate-${iteration}-${sample}`,
      phase: 'Evaluate',
      schema: {
        type: 'object',
        properties: {
          compiled: { type: 'boolean' },
          correct: { type: 'boolean' },
          speedup: { type: 'number' },
          latency_ms: { type: 'number' },
          baseline_latency_ms: { type: 'number' },
          passed_tests: { type: 'number' },
          total_tests: { type: 'number' },
          error_message: { type: 'string' },
          reward_hacking_flags: { type: 'array', items: { type: 'string' } },
        },
        required: ['compiled', 'correct', 'speedup'],
      },
    }), { retries: 5 })

    const candidate = {
      id: `iter_${iteration}_sample_${sample}`,
      path: args.task_result_command ? (__taskMeasured.host_candidate_path || '') : cudallmCandidatePath(iteration, sample),
      selected_feature_ids: selection.selected_feature_ids || [],
      implemented_feature_ids: generation.implemented_feature_ids || [],
      code: generation.candidate_code || '',
      eval: args.task_result_command ? {...__taskMeasured,passed_tests:__taskMeasured.n_pass,total_tests:__taskMeasured.n_total,explanation:evaluation.error_message || '',reward_hacking_flags:evaluation.reward_hacking_flags || []} : (SOL_AVAILABLE ? __solHostFeedback(__hostMeasured, evaluation) : evaluation),
    }

    if (USE_DRIVER_STANDALONE) {
      const kPath = cudallmCandidatePath(iteration, sample)
      const rPath = cudallmResultPath(iteration, sample)
      const buildOut = `${EXP_DIR}/cudallm_iter_${iteration}_sample_${sample}.artifact`
      await agentRetry(() => agent(
        `${driverSh('build.sh', `--source ${kPath} --out ${buildOut}`)}\n` +
        `Return its stdout JSON verbatim.`,
        { model: MODEL.mechanical, label: `driver-build-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      const runOut = await agentRetry(() => agent(
        `${driverSh('run.sh', `--artifact ${buildOut} --problem ${DRIVER_PROBLEM_PATH} --out ${rPath}`)}\n` +
        `Return its stdout JSON verbatim {ok, latency_ms, compiled, correct, log}.`,
        { model: MODEL.profile, label: `driver-run-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      let profilePointer = null
      let evidenceOut = null
      if (PROFILING_DECISION.method === 'native_profiler') {
        profilePointer = await agentRetry(() => agent(
          `${driverSh('profile.sh', buildProfileShArgs(buildOut, iteration, sample))}\n` +
          profileStepFooter() +
          `Return stdout JSON verbatim {ok, profiler, native_profile, format, error}.`,
          { model: MODEL.profile, label: `driver-profile-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        evidenceOut = await agentRetry(() => agent(
          buildToEvidencePrompt(profilePointer),
          { model: MODEL.mechanical, label: `driver-to-evidence-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      } else {
        // profiling-strategist chose a non-native method (e.g. perf_heuristic);
        // do NOT run profile.sh / ncu. run.sh already produced throughput
        // (latency_ms; GFLOPS/GB-s if reported). When method==='perf_heuristic',
        // normalize that throughput into canonical metrics via the strategist
        // normalizer (substrate profiling/<normalizer>), tagging bottlenecks
        // evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.
        const _norm = PROFILING_DECISION.normalizer || 'perf_to_evidence.py'
        evidenceOut = await agentRetry(() => agent(
          `Profiling-strategist chose method='${PROFILING_DECISION.method}' (confidence='${PROFILING_DECISION.confidence}'); do NOT run ${DRIVER.profiler_name || 'a native profiler'}.\n` +
          `run.sh already wrote throughput to ${rPath}. ` +
          (PROFILING_DECISION.method === 'perf_heuristic'
            ? `Normalize that throughput into canonical metrics via ` +
              `run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/profiling/${_norm} --baseline ${rPath} --peak-gflops <device_peak_gflops> --peak-gbs <device_peak_gbs>\`.\n` +
              `Return its stdout JSON verbatim {ok, metrics:{latency_ms,dram_pct,sm_pct,occupancy}, coverage, source_backend}. ` +
              `Also write heuristic_bclass (memory_bound|compute_bound|latency_bound) based on the throughput ratio so diagnose.py does not fall to unknown. ` +
              `Tag every emitted bottleneck as evidence='profile_heuristic', confidence='${PROFILING_DECISION.confidence}'.`
            : `Return {ok:true, metrics:{latency_ms:<from ${rPath}>,dram_pct:null,sm_pct:null,occupancy:null}, coverage:[], source_backend:"${DRIVER_BACKEND_ID}"}.`) +
          `\nReturn {ok, metrics, coverage, source_backend}.`,
          { model: MODEL.mechanical, label: `driver-to-evidence-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      }
      const diagOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/diagnose.py --metrics-json '${JSON.stringify((evidenceOut && evidenceOut.metrics) || {})}'\`.\n` +
        `Return stdout JSON verbatim {bottleneck_class, evidence}.`,
        { model: MODEL.mechanical, label: `driver-diagnose-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
      const antiCheatOut = await agentRetry(() => agent(
        `Run exactly: \`${PY ? PY + ' ' : ''}${SUBSTRATE}/anti_cheat.py --source ${kPath} --metrics ${rPath}\`.\n` +
        `Return stdout JSON verbatim {ok, suspicious, reasons}.`,
        { model: MODEL.mechanical, label: `driver-anti-cheat-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5 })
      candidate.driver_envelope = {
        anti_cheat: antiCheatOut || {},
        metrics: (evidenceOut && evidenceOut.metrics) || {},
        vendor: (DRIVER && DRIVER.hw_vendor) || '',
        coverage: (evidenceOut && evidenceOut.coverage) || [],
        bottleneck_class: (diagOut && diagOut.bottleneck_class) || 'unknown',
        latency_ms: Number((runOut && runOut.latency_ms) || 0),
        backend_id: DRIVER_BACKEND_ID,
        profiler: (profilePointer && profilePointer.profiler) || null,
        profiler_format: (profilePointer && profilePointer.format) || null,
        profile_ok: profilePointer ? profilePointer.ok !== false : false,
      }
    } else if (IS_EMBEDDED) {
      // --- Embedded eval (integration-strategist → embedded_inplace / embedded_dispatch) ---
      // Generalist-level treatment for inference-engine embedded operators
      // (e.g. llama.cpp .cuh) that cannot compile standalone. This runs inside
      // the existing serial sample loop (the FSR candidate eval is sequential —
      // no `await parallel(` — so there is no race on the shared project file).
      const kPath = cudallmCandidatePath(iteration, sample)
      const projectKernel = REFERENCE_CODE_PATH || PROFILE_SOURCE_PATH || TASK_SPEC_PATH
      const variant = `cudallm_${iteration}_${sample}`.replace(/[^A-Za-z0-9_]/g, '_')
      let embLatency = 0, embMetrics = {}, embBclass = 'unknown'
      if (INTEGRATION_DECISION.method === 'embedded_inplace' && ORIGINAL_BACKUP) {
        const embResult = await agentRetry(() => agent(
          `EMBEDDED-INPLACE EVAL (serial). Candidate: ${kPath} | project operator: ${projectKernel} | pristine backup: ${ORIGINAL_BACKUP}\n` +
          `Run IN ORDER:\n1. Restore pristine: cp -a ${ORIGINAL_BACKUP} ${projectKernel}\n` +
          `2. Apply candidate: cp ${kPath} ${projectKernel}\n3. Build: ${BUILD_CMD}\n4. Test: ${EVAL_CMD || PROJECT_BENCH_CMD}\n5. Benchmark: ${PROJECT_BENCH_CMD || EVAL_CMD}\n` +
          `6. ALWAYS restore: cp -a ${ORIGINAL_BACKUP} ${projectKernel}\n` +
          `Parse latency_ms + heuristic_bclass (memory/compute/latency bound). Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
          { model: MODEL.mechanical, label: `embedded-inplace-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
        embLatency = Number(embResult?.latency_ms || 0)
        embBclass = embResult?.heuristic_bclass || 'unknown'
        embMetrics = embResult?.metrics || { latency_ms: embLatency }
      } else if (INTEGRATION_DECISION.method === 'embedded_dispatch' && REGISTER_SCRIPT && PROJECT_ROOT) {
        const _plan = typeof __embeddedEvalPlan === 'function'
          ? __embeddedEvalPlan({ adapter: `python3 "${REGISTER_SCRIPT}"`, variant, source: kPath, projectRoot: PROJECT_ROOT, buildCmd: BUILD_CMD, testCmd: EVAL_CMD || PROJECT_BENCH_CMD, benchmarkCmd: PROJECT_BENCH_CMD || EVAL_CMD })
          : null
        if (_plan) {
          const embResult = await agentRetry(() => agent(
            `EMBEDDED-DISPATCH EVAL (serial). Run IN ORDER:\n1. Register: ${_plan.register}\n2. Build: ${_plan.build}\n3. Test: ${_plan.test}\n4. Benchmark: ${_plan.benchmark}\n5. Unregister: ${_plan.unregister}\n${_plan.cleanupInvariant}\n` +
            `Parse latency_ms + heuristic_bclass. Return {latency_ms, heuristic_bclass, compiled, correct, metrics:{latency_ms}}.`,
            { model: MODEL.mechanical, label: `embedded-dispatch-${iteration}-${sample}`, phase: 'Evaluate', schema: JSON_PASSTHROUGH }), { retries: 5, allowNull: true })
          embLatency = Number(embResult?.latency_ms || 0)
          embBclass = embResult?.heuristic_bclass || 'unknown'
          embMetrics = embResult?.metrics || { latency_ms: embLatency }
        }
      }
      candidate.driver_envelope = { latency_ms: embLatency, metrics: embMetrics, bottleneck_class: embBclass, backend_id: 'embedded' }
    }

    candidates.push(candidate)

    if (isBetterCandidate(candidate, bestCandidate)) {
      bestCandidate = candidate
    }

    phase('Reinforce')

    const reinforce = await agentRetry(() => agent(`${__taskContractBlock()}Update ${langToken(LEGACY_REINFORCE_LANG_TOKEN)} feature scores from this measured candidate.

# Candidate${CANDIDATE_FILE_GUIDANCE ? '\n' + CANDIDATE_FILE_GUIDANCE : ''}
\`\`\`json
${JSON.stringify(args.task_result_command ? candidatePromptRecord(candidate) : {
  id: candidate.id,
  selected_feature_ids: candidate.selected_feature_ids,
  implemented_feature_ids: candidate.implemented_feature_ids,
  eval: candidate.eval,
}, null, 2)}
\`\`\`

# Current feature scores
\`\`\`json
${JSON.stringify(featureScores, null, 2)}
\`\`\`

# Reward rules
1. compiled=false gives strong penalty to implemented features.
2. compiled=true but correct=false gives weak compile credit and correctness penalty.
3. correct=true gives correctness credit plus speedup reward.
4. reward_hacking_flags suppress reward even if speedup appears high.
5. Features not implemented should not receive credit.

Return updated score records for affected features.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Reinforce","ts":"<ts>","status":"done","candidate_id":"iter${iteration}-s${sample}","technique":"<the feature combination whose scores you updated as a +-joined list>","note":"<which features gained/lost reward and why, one line>"}`, {
      label: `reinforce-${iteration}-${sample}`,
      phase: 'Reinforce',
      schema: {
        type: 'object',
        properties: {
          updated_scores: { type: 'array', items: { type: 'object' } },
          reinforcement_notes: { type: 'array', items: { type: 'string' } },
        },
        required: ['updated_scores'],
      },
    }), { retries: 5 })

    for (const score of reinforce.updated_scores || []) {
      if (score?.id) featureScores[score.id] = score
    }
  }
}

// =============================================================================
// Phase 8: Report
// =============================================================================
phase('Report')

const finalReport = await agentRetry(() => agent(`${__taskContractBlock()}Write a concise CUDA-LLM FSR optimization report.

# Task
${taskSpec}

# Adaptation scope
${ADAPTATION_SCOPE}

# Best candidate summary
\`\`\`json
${JSON.stringify(bestCandidate ? (args.task_result_command ? candidatePromptRecord(bestCandidate) : {
  id: bestCandidate.id,
  selected_feature_ids: bestCandidate.selected_feature_ids,
  implemented_feature_ids: bestCandidate.implemented_feature_ids,
  eval: bestCandidate.eval,
}) : null, null, 2)}
\`\`\`

# Feature scores
\`\`\`json
${JSON.stringify(featureScores, null, 2)}
\`\`\`

# Candidate history${CANDIDATE_FILE_GUIDANCE ? '\n' + CANDIDATE_FILE_GUIDANCE : ''}
\`\`\`json
${JSON.stringify(candidates.map(c => args.task_result_command ? candidatePromptRecord(c) : ({
  id: c.id,
  selected_feature_ids: c.selected_feature_ids,
  implemented_feature_ids: c.implemented_feature_ids,
  eval: c.eval,
})), null, 2)}
\`\`\`

Cover:
1. Which ${langToken(LEGACY_REINFORCE_LANG_TOKEN)} features were reinforced by measured evidence.
2. Which feature combinations failed and why.
3. Whether the best kernel is trustworthy under diverse tests.
4. Which feature sets should be tried next.
5. Any reward-hacking risks that remain.

# Genome self-report (REQUIRED — do this LAST; do NOT let it change your returned JSON)
Append exactly one line to ${EXP_DIR}/genome.jsonl (create if missing; shell append with >>). Timestamp first: date -u +%Y-%m-%dT%H:%M:%SZ
Then append:
{"workflow":"${WORKFLOW_NAME}","phase":"Report","ts":"<ts>","status":"done","candidate_id":"${bestCandidate ? bestCandidate.id : 'none'}","speedup":${bestCandidate && bestCandidate.eval ? (bestCandidate.eval.speedup || 0) : 0},"technique":"<best feature combination as a +-joined list>","note":"<best result + most-reinforced features, one line>"}`, {
  label: 'final-report',
  phase: 'Report',
}), { retries: 5 })

// embedded_inplace exit safety net: unconditionally restore the pristine project
// operator so the host project is left byte-exact regardless of how the loop ended.
if (ORIGINAL_BACKUP) {
  await agentRetry(() => agent(`${__taskContractBlock()}Exit restore (unconditional): run \`cp -a "${ORIGINAL_BACKUP}" "${REFERENCE_CODE_PATH || PROFILE_SOURCE_PATH || TASK_SPEC_PATH}"\` and confirm.`,
    { model: MODEL.mechanical, label: 'integration-exit-restore', phase: 'Report', schema: JSON_PASSTHROUGH }), { retries: 5 })
}

return {
  input_mode: INPUT_MODE,
  problem_definition: PROBLEM_DEFINITION,
  problem_path: TASK_SPEC_PATH,
  generated_kernel_path: bestCandidate?.path || '',
  initial_candidates: candidates,
  initial_generation_result: {
    verified: candidates.some(c => c.eval?.correct),
    selected_candidate_id: bestCandidate?.id || '',
  },
  reference_code_path: REFERENCE_CODE_PATH,
  target_gpu: TARGET_GPU,
  iterations: ITERATIONS,
  feature_budget: FEATURE_BUDGET,
  samples_per_feature_set: SAMPLES_PER_FEATURE_SET,
  adaptation_scope: ADAPTATION_SCOPE,
  best_speedup: bestCandidate?.eval?.speedup || 0,
  best_latency_ms: bestCandidate?.eval?.latency_ms || null,
  best_kernel_code: args.task_result_command ? '' : (bestCandidate?.code || ''),
  ...(args.task_result_command ? {canonical_metric:{name:'speedup_vs_reference',value:bestCandidate?.eval?.speedup || 0},task_result_path:bestCandidate?.eval?.test_result_path || null} : {}),
  best_candidate_binding: bestCandidate?.eval?.source_binding || null,
  seed_measurement: solSeedMeasurement,
  ...(taskParentMeasurement ? {accepted_parent_measurement:taskParentMeasurement,speedup_vs_accepted_parent:(bestCandidate?.eval?.speedup || 0)/taskParentMeasurement.speedup} : {}),
  performance_domain: args.task_result_command ? 'official_full_workload_reference_relative' : (SOL_AVAILABLE ? 'official_full_workload_seed_relative' : 'legacy'),
  best_candidate_id: bestCandidate?.id || '',
  feature_scores: featureScores,
  candidates: candidates.map(c => ({
    id: c.id,
    selected_feature_ids: c.selected_feature_ids,
    implemented_feature_ids: c.implemented_feature_ids,
    eval: c.eval,
  })),
  report: finalReport,
}
