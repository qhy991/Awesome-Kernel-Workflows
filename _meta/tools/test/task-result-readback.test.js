'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path')
const source=fs.readFileSync(path.resolve(__dirname,'../../scaffolding/task-result.js'),'utf8')
const p='/workspace/result.json'
const raw={contract_version:'kersor-task-result-v1',outcome_state:'passed',test_result_path:p,compiled:true,correct:true,
 full_workload_set:true,measurement_valid:true,n_pass:17,n_total:17,candidate_latency_aggregate_ms:.02,speedup_vs_reference:2,
 candidate_path:p+'.artifact/candidate.py',source_binding:{verified:true,source_sha256:'a'.repeat(64)}}
const ctx={resultPath:p,workloadCount:17,readCommand:'trusted-result-reader --read-only --candidate {kernel_path} --result {result_path}',label:'initial2'}
test('native result transcription is repaired once by read-only reconciliation, without a GPU command',async()=>{
 let calls=0,prompt=''
 const readback=vm.runInNewContext(source+';__taskResultWithReadback',{agentRetry:fn=>fn(),agent:async text=>{
  calls++;prompt=text;return {test_result_path:p,test_result_json:raw}
 }})
 const result=await readback(ctx,{test_result_path:p,test_result_json:{...raw,candidate_path:'/truncated/candidate.py'}})
 assert.equal(result.is_valid,true);assert.equal(calls,1)
 assert.match(prompt,/--read-only/);assert.ok(prompt.includes(p+'.artifact/candidate.py'))
 assert.match(prompt,/Do not.*GPU/)
})
test('an intact result needs no extra agent or command',async()=>{
 const readback=vm.runInNewContext(source+';__taskResultWithReadback',{agent:()=>assert.fail('unnecessary readback')})
 assert.equal((await readback(ctx,{test_result_path:p,test_result_json:raw})).is_valid,true)
})
for(const extra of [{outcome_state:'unknown'},{model_observation:'unknown'}])test('genuine unknown is not retried '+JSON.stringify(extra),async()=>{
 const readback=vm.runInNewContext(source+';__taskResultWithReadback',{agent:()=>assert.fail('unknown retry')})
 await assert.rejects(readback(ctx,{test_result_path:p,test_result_json:{...raw,...extra}}),e=>e.code==='TASK_RESULT_HOLD')
})
test('a second transcription error remains HOLD, with no loop or replacement result',async()=>{
 let calls=0;const bad={...raw,candidate_path:'/truncated/candidate.py'}
 const readback=vm.runInNewContext(source+';__taskResultWithReadback',{agentRetry:fn=>fn(),agent:async()=>{calls++;return {test_result_path:p,test_result_json:bad}}})
 await assert.rejects(readback(ctx,{test_result_path:p,test_result_json:bad}),e=>e.code==='TASK_RESULT_HOLD')
 assert.equal(calls,1)
})
test('Host readback uses the generic command owner and exact designated result',async()=>{
 let calls=0;const readback=vm.runInNewContext(source+';__taskResultWithReadback',{
  evaluate:async r=>{calls++;assert.match(r.argv[2],/--read-only/);return {stage:'complete',exit_code:0,stdout_json:raw}},
  agent:()=>assert.fail('Host should not ask model to transcribe data')})
 assert.equal((await readback(ctx,{test_result_path:p,test_result_json:{...raw,candidate_path:'/wrong'}})).is_valid,true)
 assert.equal(calls,1)
})
test('duplicate result-location transcription is advisory; canonical source binding still governs',async()=>{
 const readback=vm.runInNewContext(source+';__taskResultWithReadback',{agent:()=>assert.fail('redundant path must not consume a model turn')})
 const result=await readback(ctx,{test_result_path:'/typo',test_result_json:{...raw,test_result_path:'/CUDAAAgent/result.json'}})
 assert.equal(result.is_valid,true);assert.equal(result.test_result_path,p)
 assert.equal(result.delivery_path_observation.recorded,'/CUDAAAgent/result.json')
 await assert.rejects(readback({...ctx,readCommand:null},{test_result_path:'/typo',test_result_json:{...raw,candidate_path:'/other/candidate.py'}}),e=>e.code==='TASK_RESULT_HOLD')
})
test('qualified file handoff requests only a reference while preserving canonical gates',async()=>{
 let prompt=''
 const evaluate=vm.runInNewContext(source+';__nativeTaskEvaluate',{args:{native_task_result_file_handoff:true},agentRetry:fn=>fn(),agent:async text=>{prompt=text;return {test_result_path:p,test_result_json:raw}}})
 assert.equal((await evaluate({...ctx,candidatePath:'/input.py',command:'task {kernel_path} {result_path}'})).is_valid,true)
 assert.ok(prompt.includes(JSON.stringify({test_result_path:p,test_result_json:{}})));assert.doesNotMatch(prompt,/copied verbatim/)
})

const missingStructured = 'agent({schema}): subagent completed without calling StructuredOutput (after in-conversation nudge)'
test('qualified native missing StructuredOutput reads the existing slot once without replaying evaluation',async()=>{
 const calls=[]
 const evaluate=vm.runInNewContext(source+';__nativeTaskEvaluate',{
  args:{native_task_result_file_handoff:true},agentRetry:fn=>fn(),agent:async(text,options)=>{
   calls.push({text,options});if(calls.length===1)throw new Error(missingStructured)
   assert.equal(options.phase,'ReadResult');assert.match(text,/--read-only/)
   return {test_result_path:p,test_result_json:raw}
  }})
 const result=await evaluate({...ctx,candidatePath:'/source.py',command:'trusted-evaluation {kernel_path} {result_path}'})
 assert.equal(result.is_valid,true);assert.equal(result.result_delivery_recovered,true)
 assert.equal(calls.filter(c=>c.options.phase==='Evaluate').length,1)
 assert.equal(calls.length,2)
})
for(const variant of ['transport','policy','cause','model-unknown','refusal-flag','unqualified','no-reader'])test('missing-output repair does not widen unknown or refused execution: '+variant,async()=>{
 let calls=0;const error=new Error(variant==='transport'?'HTTP 524':missingStructured)
 if(variant==='policy')error.code='KERSOR_PROVIDER_SAFEGUARD_REFUSAL'
 if(variant==='cause')error.cause=new Error('model identity unknown')
 if(variant==='model-unknown')error.model_observation='unknown'
 if(variant==='refusal-flag')error.policy_refusal=true
 const evaluate=vm.runInNewContext(source+';__nativeTaskEvaluate',{
  args:{native_task_result_file_handoff:variant!=='unqualified'},agentRetry:fn=>fn(),agent:async()=>{calls++;throw error}})
 await assert.rejects(evaluate({...ctx,...(variant==='no-reader'?{readCommand:null}:{}),candidatePath:'/source.py',command:'trusted-evaluation {kernel_path} {result_path}'}),e=>e.code==='TASK_RESULT_HOLD')
 assert.equal(calls,1)
})
test('failed read-only delivery stops after one repair, with no third agent or GPU retry',async()=>{
 let calls=0
 const evaluate=vm.runInNewContext(source+';__nativeTaskEvaluate',{args:{native_task_result_file_handoff:true},agentRetry:fn=>fn(),agent:async()=>{calls++;throw new Error(missingStructured)}})
 await assert.rejects(evaluate({...ctx,candidatePath:'/source.py',command:'trusted-evaluation {kernel_path} {result_path}'}),e=>e.code==='TASK_RESULT_HOLD')
 assert.equal(calls,2)
})

test('readback prompt carries the reason and path while the full diagnostic remains in the error',async()=>{
 let prompt='';const diagnostic='FULL_RAW_DIAGNOSTIC_KEEP_IN_FILE'.repeat(10000)
 const bad={...raw,candidate_path:'/wrong.py',diagnostics:[{error:diagnostic}]}
 const readback=vm.runInNewContext(source+';__taskResultWithReadback',{args:{native_task_result_file_handoff:true},agentRetry:fn=>fn(),agent:async text=>{prompt=text;return {test_result_path:p,test_result_json:raw}}})
 assert.equal((await readback(ctx,{test_result_path:p,test_result_json:bad})).is_valid,true)
 assert.ok(prompt.includes(p));assert.ok(!prompt.includes('FULL_RAW_DIAGNOSTIC_KEEP_IN_FILE'))
 const parse=vm.runInNewContext(source+';__taskResult')
 assert.throws(()=>parse({test_result_path:p,test_result_json:bad},p,17),e=>e.message.includes(diagnostic) && e.task_result_reason==='incomplete workload/measurement/source proof')
})
