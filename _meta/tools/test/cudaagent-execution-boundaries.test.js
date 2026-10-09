'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm')
const root=path.resolve(__dirname,'../../..')
const source=fs.readFileSync(path.join(root,'CUDAAgent/cuda-agent-kernel-optimization.js'),'utf8')
const run=require('../lib/run-workflow.js')
const base={problem_path:'/fixture/task.md',exp_dir:'/fixture/exp',language:'cute-dsl',seed_candidates:2,max_turns:1,task_result_command:'trusted --candidate {kernel_path} --result {result_path}',task_workload_count:2}
function measured(p,lat=.02){return {test_result_path:p,test_result_json:{contract_version:'kersor-task-result-v1',test_result_path:p,compiled:true,correct:true,full_workload_set:true,measurement_valid:true,n_pass:2,n_total:2,candidate_latency_aggregate_ms:lat,speedup_vs_reference:.06/lat,candidate_path:p+'.artifact/candidate.py',source_binding:{verified:true,source_sha256:'a'.repeat(64)}}}}
function agents(){return {'generate-initial-kernel':{initial_candidates:[1,0].map(i=>({variant_path:`/fixture/exp/generated/initial_${i}.py`}))},'profile-baseline':{bottlenecks:[],optimization_strategy:'fixture'},'impl-0':{variant_path:'/fixture/exp/task_attempt_0.py'},'task-verify-0':measured('/fixture/exp/task_attempt_0.json',.025),'initial-0':measured('/fixture/exp/generated/initial_0.json',.020),'initial-1':measured('/fixture/exp/generated/initial_1.json',.030)}}
for(const host of [false,true])test(`CUDAAgent source generation precedes exactly one evaluation per seed (Host=${host})`,async()=>{
 const a=agents(),evaluations=[],ev=host?Object.fromEntries(['initial-0','initial-1','task-verify-0'].map(label=>[label,request=>{evaluations.push(request);return {stage:'complete',exit_code:0,stdout_json:a[label].test_result_json}}])):null
 const {result,calls}=await run(source,{...base,native_task_result_file_handoff:true},a,ev)
 const generation=calls.find(c=>c.label==='generate-initial-kernel')
 assert.deepEqual(calls.find(c=>c.label==='impl-0').schema.required,['variant_path'])
 assert.deepEqual(generation.schema.required,['initial_candidates'])
 assert.deepEqual(generation.schema.properties.initial_candidates.items.required,['variant_path'])
 assert.match(generation.prompt,/source files only/)
 assert.doesNotMatch(generation.prompt,/Await terminal and read each JSON|verbatim test_result_path\/test_result_json/)
 assert.equal(result.generated_kernel_path,'/fixture/exp/generated/initial_0.json.artifact/candidate.py')
 if(host){assert.deepEqual(evaluations.map(r=>r.label),['initial-0','initial-1','task-verify-0']);assert.equal(calls.filter(c=>c.label.startsWith('initial-')).length,0)}
 else{const evals=calls.filter(c=>c.label.startsWith('initial-'));assert.deepEqual(evals.map(c=>c.label),['initial-0','initial-1']);assert.ok(evals.every(c=>c.prompt.includes('test_result_json={}')));assert.ok(evals.every(c=>c.seq>generation.seq))}
})
test('CUDAAgent rejects an incomplete generated seed set before any evaluation',async()=>{
 const a=agents();a['generate-initial-kernel'].initial_candidates.pop();let executions=0
 const ev={'initial-0':()=>{executions++;return {stage:'complete',exit_code:0,stdout_json:measured('/fixture/exp/generated/initial_0.json').test_result_json}}}
 await assert.rejects(run(source,base,a,ev),/initial candidate.*(?:count|set|cover)|seed.*(?:count|set|cover)/i)
 assert.equal(executions,0)
})
for(const code of ['KERSOR_PERMISSION_DENIED','KERSOR_CONFIG_INVALID','KERSOR_AGENT_RESULT_CONTRACT_VIOLATION','KERSOR_CLAUDE_TRANSPORT_FAULT'])test('CUDAAgent preserves non-timeout implementation error '+code,async()=>{
 const failure=Object.assign(new Error('retained diagnostic'),{code,retryable:false})
 const a=agents();a['task-accepted-parent']=measured('/fixture/exp/accepted_parent.task.json');a['impl-0']={then(_resolve,reject){reject(failure)}}
 await assert.rejects(run(source,{...base,kernel_path:'/fixture/parent.py'},a,null),e=>e===failure)
})
for(const code of ['KERSOR_CODEX_TIMEOUT','KERSOR_CLAUDE_TIMEOUT','KERSOR_TURN_TIMEOUT'])test('CUDAAgent records a typed timeout without losing its accepted parent '+code,async()=>{
 const a=agents();a['task-accepted-parent']=measured('/fixture/exp/accepted_parent.task.json');a['impl-0']={then(_resolve,reject){reject(Object.assign(new Error('bounded timeout'),{code}))}}
 const {result}=await run(source,{...base,kernel_path:'/fixture/parent.py'},a,null)
 assert.equal(result.convergence_status,'timeout');assert.equal(result.generated_kernel_path,'/fixture/exp/accepted_parent.task.json.artifact/candidate.py')
})
test('canonical watchdog emits a typed timeout and retains rejected errors unchanged',async()=>{
 const text=fs.readFileSync(path.join(root,'_meta/scaffolding/turn-timeout.js'),'utf8')
 const wrap=vm.runInNewContext(text+';withTurnTimeout',{args:{turn_timeout_min:1},setTimeout:fn=>{fn();return 1},clearTimeout:()=>{}})
 await assert.rejects(wrap(new Promise(()=>{}),'fixture'),e=>e.code==='KERSOR_TURN_TIMEOUT')
 const failure=Object.assign(new Error('permission'),{code:'KERSOR_PERMISSION_DENIED'})
 const direct=vm.runInNewContext(text+';withTurnTimeout',{args:{}})
 await assert.rejects(direct(Promise.reject(failure),'fixture'),e=>e===failure)
})
test('task contract distinguishes diagnostics and uses hook file references consistently',()=>{
 const text=fs.readFileSync(path.join(root,'_meta/scaffolding/typed-args.js'),'utf8')
 const prompt=vm.runInNewContext(text+';__taskContractBlock()',{args:{...base,native_task_result_file_handoff:true}})
 assert.match(prompt,/diagnostic/i);assert.match(prompt,/scored/i);assert.match(prompt,/test_result_json=\{\}/)
 assert.doesNotMatch(prompt,/Return the resulting complete contract object/)
})

for(const code of ['KERSOR_PERMISSION_DENIED','KERSOR_AGENT_RESULT_CONTRACT_VIOLATION','KERSOR_CLAUDE_TRANSPORT_FAULT'])test('CUDAAgent preserves non-timeout legacy verification error '+code,async()=>{
 const failure=Object.assign(new Error('verification diagnostic'),{code,retryable:false})
 const a={'profile-baseline':{eager_time_ms:.04,compile_time_ms:.02,bottlenecks:[],optimization_strategy:'fixture'},'impl-0':{kernel_code:'source',binding_code:'binding',model_new_code:'model'},'verify-0':{then(_resolve,reject){reject(failure)}}}
 await assert.rejects(run(source,{kernel_path:'/fixture/model.py',exp_dir:'/fixture/exp',max_turns:1},a,null),e=>e===failure)
})
for(const host of [false,true])test(`fixed task execution preserves its original cause without retry (Host=${host})`,async()=>{
 const text=fs.readFileSync(path.join(root,'_meta/scaffolding/task-result.js'),'utf8'),failure=Object.assign(new Error('unknown transport outcome'),{code:'KERSOR_CLAUDE_TRANSPORT_FAULT'})
 let executions=0,prompt
 const fail=async input=>{executions++;prompt=input;throw failure}
 const execute=vm.runInNewContext(text+';__nativeTaskEvaluate',{...(host?{evaluate:fail}:{agent:fail,agentRetry:fn=>fn()})})
 await assert.rejects(execute({candidatePath:'/candidate.py',candidateSource:'DO NOT REBUILD THIS',resultPath:'/result.json',command:'trusted',workloadCount:2}),e=>e.code==='TASK_RESULT_HOLD'&&e.retryable===false&&e.cause===failure)
 assert.equal(executions,1)
 if(!host){assert.match(prompt,/existing candidate file/);assert.doesNotMatch(prompt,/DO NOT REBUILD THIS|write the COMPLETE returned source/)}
})

test('a missing source-only initialization reply is not retried as another seed batch',async()=>{
 let generations=0
 const a=agents();Object.defineProperty(a,'generate-initial-kernel',{get(){generations++;return null}})
 await assert.rejects(run(source,base,a,null),/returned null after 1 attempt/)
 assert.equal(generations,1)
})
