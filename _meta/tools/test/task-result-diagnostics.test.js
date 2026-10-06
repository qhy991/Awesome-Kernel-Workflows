'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path')
const source=fs.readFileSync(path.resolve(__dirname,'../../scaffolding/task-result.js'),'utf8')
const p='/workspace/result.json'
const failure={contract_version:'kersor-task-result-v1',test_result_path:p,compiled:null,correct:false,
  full_workload_set:true,measurement_valid:false,n_pass:0,n_total:17,outcome_state:'candidate_failure',failure_origin:'candidate',
  candidate_path:p+'.artifact/candidate.py',source_binding:{verified:true,source_sha256:'a'.repeat(64)},
  diagnostics:[{workload_uuid:'w',evaluation:{status:'ERROR',error:'full traceback '+ 'x'.repeat(20000)}}],
  evidence:{trace:'/workspace/raw/trace.jsonl',admission:'/workspace/raw/admission.json',terminal:'/workspace/raw/terminal.json'}}
test('candidate failure sends full raw feedback to the repairing agent',()=>{
 const parse=vm.runInNewContext(source+';__taskResult')
 const value=parse({test_result_path:p,test_result_json:failure},p,17)
 assert.equal(value.is_valid,false);assert.equal(value.outcome_state,'candidate_failure')
 assert.equal(JSON.stringify(value.diagnostics),JSON.stringify(failure.diagnostics))
 assert.ok(value.error_log.includes(failure.diagnostics[0].evaluation.error))
 assert.equal(value.evidence.trace,failure.evidence.trace)
})
test('unknown HOLD preserves the raw task evidence for the supervising agent',()=>{
 const parse=vm.runInNewContext(source+';__taskResult');const raw={...failure,outcome_state:'unknown',failure_origin:'unknown'}
 assert.throws(()=>parse({test_result_path:p,test_result_json:raw},p,17),error=>{
  assert.equal(error.code,'TASK_RESULT_HOLD');assert.equal(error.retryable,false)
  assert.equal(JSON.stringify(error.task_result),JSON.stringify(raw))
  assert.ok(error.message.includes(raw.diagnostics[0].evaluation.error))
  return true
 })
})
test('Host exit2 carries diagnostic evidence and never calls an extra agent or reexecutes',async()=>{
 const raw={...failure,outcome_state:'unknown',failure_origin:'unknown'};let calls=0
 const execute=vm.runInNewContext(source+';__nativeTaskEvaluate',{
  evaluate:async()=>{calls++;return {stage:'complete',exit_code:2,stdout_json:raw}},
  agent:()=>assert.fail('no resubmission or provider diagnostic on an unknown command')})
 await assert.rejects(execute({candidatePath:'/candidate.py',resultPath:p,command:'trusted',workloadCount:17}),e=>{
  assert.equal(e.code,'TASK_RESULT_HOLD');assert.equal(e.task_result.evidence.trace,raw.evidence.trace);return true
 })
 assert.equal(calls,1)
})

test('prompt projection references full evidence without mutating retained diagnostics',()=>{
 const project=vm.runInNewContext(fs.readFileSync(path.resolve(__dirname,'../../scaffolding/task-result.js'),'utf8')+';__taskEvidencePrompt')
 const raw={test_result_path:'/full/result.json',host_candidate_path:'/full/candidate.py',error_log:'LONG'.repeat(100000),diagnostics:[{message:'KEEP_COMPLETE'}],is_valid:false,evidence:{trace:'/full/trace.jsonl'}}
 const before=JSON.stringify(raw),out=project(raw)
 assert.equal(out.test_result_path,raw.test_result_path);assert.equal(out.evidence_paths.trace,raw.evidence.trace)
 assert.ok(!JSON.stringify(out).includes('LONG'));assert.equal(JSON.stringify(raw),before)
 const noPath={error_log:'do not discard me'};assert.equal(project(noPath),noPath)
})
