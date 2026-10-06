'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path')
const source=fs.readFileSync(path.resolve(__dirname,'../../scaffolding/task-result.js'),'utf8')
const p='/workspace/failed.task.json'
const raw={contract_version:'kersor-task-result-v1',outcome_state:'candidate_failure',failure_origin:'candidate',
 test_result_path:p,compiled:false,correct:false,full_workload_set:true,measurement_valid:false,n_pass:2,n_total:17,
 candidate_path:p+'.artifact/candidate.py',source_binding:{verified:true,source_sha256:'a'.repeat(64)},
 diagnostics:[{error:'\u001b[91mSCOPE_CLOSURE_CAPTURE\u001b[0m\nKeep \\ and "quoted" diagnostics intact.'}]}
const ctx={resultPath:p,workloadCount:17,readCommand:'reader --read-only {result_path}'}
test('native literal ANSI escape inside a JSON string retains candidate rejection without another call',async()=>{
 const read=vm.runInNewContext(source+';__taskResultWithReadback',{agent:()=>assert.fail('no new model request')})
 const delivery=JSON.stringify(raw).replaceAll('\\u001b','\u001b')
 const result=await read(ctx,{test_result_path:p,test_result_json:delivery})
 assert.equal(result.outcome_state,'candidate_failure');assert.equal(result.is_valid,false)
 assert.deepEqual(JSON.parse(JSON.stringify(result.diagnostics)),raw.diagnostics)
})
test('object delivery preserves a large full diagnostic without double serialization',async()=>{
 let schema
 const evaluate=vm.runInNewContext(source+';__nativeTaskEvaluate',{agentRetry:fn=>fn(),agent:async(_,o)=>{
  schema=o.schema;return {test_result_path:p,test_result_json:{...raw,diagnostics:[{error:'x'.repeat(100000)+'\u001b[91mend'}]}}
 }})
 const result=await evaluate({...ctx,command:'reader {kernel_path} {result_path}',candidatePath:'/workspace/candidate.py'})
 assert.equal(schema.properties.test_result_json.type,'object')
 assert.equal(result.diagnostics[0].error.length,100008)
})
test('structural corruption and unknown outcomes are never repaired into a score',()=>{
 const parse=vm.runInNewContext(source+';__taskResult')
 for(const text of [JSON.stringify(raw).replace(',"compiled"','"compiled"'),JSON.stringify({...raw,outcome_state:'unknown'}),JSON.stringify(raw).slice(0,-1)]){
  assert.throws(()=>parse({test_result_path:p,test_result_json:text},p,17),e=>e.code==='TASK_RESULT_HOLD')
 }
})
