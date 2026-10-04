// Native task-result compatibility: task command is the measurement authority.
function __taskHold(reason) {
  const error = new Error('TASK_RESULT_HOLD: ' + reason)
  error.code = 'TASK_RESULT_HOLD'
  error.retryable = false
  error.outcome_state = 'unknown'
  throw error
}
function __taskResult(output, expectedPath, expectedCount) {
  let result
  try {
    const raw = output?.test_result_json
    result = typeof raw === 'string' ? JSON.parse(raw)
      : raw && typeof raw === 'object' && !Array.isArray(raw) ? JSON.parse(JSON.stringify(raw))
      : __taskHold('missing/malformed raw task result')
  } catch { return __taskHold('missing/malformed raw task result') }
  if (output?.test_result_path !== expectedPath || result?.test_result_path !== expectedPath || result?.contract_version !== 'kersor-task-result-v1') return __taskHold('task result contract/path mismatch')
  if (output.model_observation === 'unknown' || result.model_observation === 'unknown' || result.outcome_state === 'unknown' || (result.failure_origin && result.failure_origin !== 'candidate')) return __taskHold('infrastructure, evidence, or model observation unknown')
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
    && result.full_workload_set === true && result.n_total === expectedCount
    && Number.isInteger(result.n_pass) && result.n_pass >= 0 && result.n_pass < expectedCount
    && result.source_binding?.verified === true
    && /^[0-9a-f]{64}$/i.test(result.source_binding?.source_sha256 || '')
    && result.candidate_path === expectedPath + '.artifact/candidate.py'
  if (!valid && !candidateFailure) return __taskHold('incomplete workload/measurement/source proof')
  return {is_valid:valid, measurement_valid:valid, compiled:result.compiled, correct:valid,
    metric_value:valid ? result.speedup_vs_reference : 0,
    speedup:valid ? result.speedup_vs_reference : 0,
    latency_ms:valid ? result.candidate_latency_aggregate_ms : null,
    n_pass:result.n_pass, n_total:result.n_total,
    pass_rate:String(result.n_pass)+'/'+String(result.n_total),
    source_binding:result.source_binding, host_candidate_path:result.candidate_path, test_result_path:output.test_result_path,
    outcome_state:valid ? 'passed' : 'candidate_failure', error_log:valid ? '' : 'explicit candidate failure'}
}

async function __nativeTaskEvaluate(ctx) {
  let output
  try { output = await agentRetry(() => agent(`Use the explicitly declared candidate file at ${ctx.candidatePath}. If it is absent, write the COMPLETE returned source below to that exact path. Never rewrite an existing declared file or select another directory entry.
${ctx.candidateSource || ''}
Run the trusted task command once: ${ctx.command.replaceAll('{kernel_path}', ctx.candidatePath).replaceAll('{result_path}', ctx.resultPath)}
Wait for its terminal result and read ${ctx.resultPath}. Return test_result_path and test_result_json copied verbatim from that file. No estimates or rewritten source in this reply.`, {
    label:ctx.label, phase:'Evaluate',
    schema:{type:'object', properties:{test_result_path:{type:'string'},test_result_json:{type:'string'}}, required:['test_result_path','test_result_json']},
  }), {retries:0}) } catch (error) { return __taskHold('agent/transport result unavailable: ' + (error?.message || String(error))) }
  return __taskResult(output, ctx.resultPath, ctx.workloadCount)
    || __taskHold('missing task result')
}
