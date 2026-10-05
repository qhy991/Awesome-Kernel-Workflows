'use strict'
const {test}=require('node:test')
const assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),cp=require('node:child_process')
const source=fs.readFileSync(path.resolve(__dirname,'../../scaffolding/task-result.js'),'utf8')
const shape=p=>({contract_version:'kersor-task-result-v1',test_result_path:p,
  compiled:true,correct:true,full_workload_set:true,measurement_valid:true,n_pass:2,n_total:2,
  candidate_latency_aggregate_ms:.02,speedup_vs_reference:2,
  candidate_path:p+'.artifact/candidate.py',source_binding:{verified:true,source_sha256:'a'.repeat(64)}})
const quote=s=>"'"+s.replaceAll("'","'\\''")+"'"

test('Host executes one command and reads the designated file despite misleading stdout',async()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'task-command-'))
 try{
  const resultPath=path.join(dir,"result 'one' $(echo inert).json"),writer=path.join(dir,'writer.js'),candidatePath="/selected '$(echo inert)'.py"
  fs.writeFileSync(writer,`require('fs').writeFileSync(${JSON.stringify(resultPath)},${JSON.stringify(JSON.stringify(shape(resultPath)))});console.log('unrelated command chatter');console.log(JSON.stringify(process.argv.slice(2)));console.log(${JSON.stringify(JSON.stringify(shape('/wrong.json')))})`)
  let calls=0
  const execute=vm.runInNewContext(source+';__nativeTaskEvaluate',{
   agent:()=>assert.fail('no agent may choose another evaluation source/path'),
   evaluate:async request=>{
    calls++;assert.equal(request.protocol,'command-v1')
    const r=cp.spawnSync(request.argv[0],Array.from(request.argv.slice(1)),{encoding:'utf8'})
    assert.match(r.stderr,/unrelated command chatter/)
    assert.ok(r.stderr.includes(JSON.stringify([candidatePath,resultPath])))
    return {stage:'complete',exit_code:r.status,stdout_json:JSON.parse(r.stdout)}
   }})
  const result=await execute({candidatePath,resultPath,command:quote(process.execPath)+' '+quote(writer)+' "{kernel_path}" \'{result_path}\'',workloadCount:2,label:'eval-one'})
  assert.equal(calls,1);assert.equal(result.test_result_path,resultPath);assert.equal(result.is_valid,true)
 }finally{fs.rmSync(dir,{recursive:true,force:true})}
})

for(const state of [{stage:'timeout',timed_out:true,exit_code:124},{stage:'complete',exit_code:2}])
 test('unfinished/unknown command remains HOLD with no replay '+JSON.stringify(state),async()=>{
  let calls=0
  const execute=vm.runInNewContext(source+';__nativeTaskEvaluate',{evaluate:async()=>{calls++;return state},agent:()=>assert.fail('no transport fallback')})
  await assert.rejects(execute({candidatePath:'/chosen.py',resultPath:'/result.json',command:'trusted {kernel_path} {result_path}',workloadCount:2}),e=>e.code==='TASK_RESULT_HOLD'&&e.retryable===false)
  assert.equal(calls,1)
 })

test('a confirmed candidate failure still enters the method as zero reward',async()=>{
 const p='/result.json',raw={...shape(p),compiled:false,correct:false,measurement_valid:false,n_pass:0,outcome_state:'candidate_failure',failure_origin:'candidate'}
 const execute=vm.runInNewContext(source+';__nativeTaskEvaluate',{evaluate:async()=>({stage:'complete',exit_code:1,stdout_json:raw})})
 const result=await execute({candidatePath:'/chosen.py',resultPath:p,command:'trusted',workloadCount:2})
 assert.equal(result.outcome_state,'candidate_failure');assert.equal(result.metric_value,0)
})

test('native eval waits for terminal and preserves the exact slot',async()=>{
 let prompt
 const p='/result.json'
 const execute=vm.runInNewContext(source+';__nativeTaskEvaluate',{agentRetry:fn=>fn(),agent:async text=>{prompt=text;return {test_result_path:p,test_result_json:JSON.stringify(shape(p))}}})
 assert.equal((await execute({candidatePath:'/chosen.py',resultPath:p,command:'trusted',workloadCount:2})).is_valid,true)
 assert.match(prompt,/running session/);assert.match(prompt,/Never rename\/delete/);assert.match(prompt,/unknown or failed result must be returned unchanged/)
})
