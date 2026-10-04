// Native task-result compatibility: task command is the measurement authority.
function __taskResult(output, expectedPath, expectedCount) {
  let result
  try { result = JSON.parse(output?.test_result_json || '') } catch { return null }
  if (output?.test_result_path !== expectedPath || result?.test_result_path !== expectedPath || result?.contract_version !== 'kersor-task-result-v1') return null
  const valid = result.compiled === true && result.correct === true
    && result.full_workload_set === true && result.measurement_valid === true
    && result.source_binding?.verified === true && result.n_pass === result.n_total
    && Number.isInteger(result.n_total) && result.n_total === expectedCount
    && /^[0-9a-f]{64}$/i.test(result.source_binding?.source_sha256 || '')
    && result.candidate_path === expectedPath + '.artifact/candidate.py'
    && result.n_total > 0 && Number.isFinite(result.candidate_latency_aggregate_ms)
    && result.candidate_latency_aggregate_ms > 0 && Number.isFinite(result.speedup_vs_reference)
    && result.speedup_vs_reference > 0
  return {is_valid:valid, measurement_valid:valid, compiled:result.compiled, correct:valid,
    metric_value:valid ? result.speedup_vs_reference : 0,
    speedup:valid ? result.speedup_vs_reference : 0,
    latency_ms:valid ? result.candidate_latency_aggregate_ms : null,
    n_pass:result.n_pass, n_total:result.n_total,
    pass_rate:String(result.n_pass)+'/'+String(result.n_total),
    source_binding:result.source_binding, host_candidate_path:result.candidate_path, test_result_path:output.test_result_path,
    error_log:valid ? '' : 'task correctness/measurement/binding gate failed'}
}

async function __nativeTaskEvaluate(ctx) {
  const output = await agentRetry(() => agent(`Use the explicitly declared candidate file at ${ctx.candidatePath}. If it is absent, write the COMPLETE returned source below to that exact path. Never rewrite an existing declared file or select another directory entry.
${ctx.candidateSource || ''}
Run the trusted task command once: ${ctx.command.replaceAll('{kernel_path}', ctx.candidatePath).replaceAll('{result_path}', ctx.resultPath)}
Wait for its terminal result and read ${ctx.resultPath}. Return test_result_path and test_result_json copied verbatim from that file. No estimates or rewritten source in this reply.`, {
    label:ctx.label, phase:'Evaluate',
    schema:{type:'object', properties:{test_result_path:{type:'string'},test_result_json:{type:'string'}}, required:['test_result_path','test_result_json']},
  }), {retries:0})
  return __taskResult(output, ctx.resultPath, ctx.workloadCount)
    || {is_valid:false,measurement_valid:false,compiled:false,correct:false,metric_value:0,speedup:0,latency_ms:null,error_log:'missing task result'}
}
