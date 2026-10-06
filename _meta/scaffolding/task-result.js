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
  if (output?.test_result_path !== expectedPath || result?.test_result_path !== expectedPath || result?.contract_version !== 'kersor-task-result-v1') return __taskHold(`task result contract/path mismatch: expected ${expectedPath}; returned ${output?.test_result_path}; recorded ${result?.test_result_path}; contract ${result?.contract_version}`, result)
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
    source_binding:result.source_binding, host_candidate_path:result.candidate_path, test_result_path:output.test_result_path,
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
        schema:{type:'object',properties:{test_result_path:{type:'string'},test_result_json:{type:'object',additionalProperties:true}},required:['test_result_path','test_result_json']},
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
    schema:{type:'object', properties:{test_result_path:{type:'string'},test_result_json:{type:'object',additionalProperties:true}}, required:['test_result_path','test_result_json']},
  }), {retries:0}) } catch (error) { return __taskHold('agent/transport result unavailable: ' + (error?.message || String(error))) }
  return await __taskResultWithReadback(ctx, output)
    || __taskHold('missing task result')
}

async function __nativeTaskAcceptedParent(ctx) {
  const measured = await __nativeTaskEvaluate(ctx)
  if (!measured?.is_valid) return __taskHold('accepted parent failed full official task measurement')
  return measured
}
