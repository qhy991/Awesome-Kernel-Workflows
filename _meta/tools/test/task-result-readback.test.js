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
