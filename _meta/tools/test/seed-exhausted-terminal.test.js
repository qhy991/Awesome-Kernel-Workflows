'use strict'
const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const run=require('../lib/run-workflow.js')
const root=path.resolve(__dirname,'../../..')
const args={problem_path:'/fixture/task.md',exp_dir:'/fixture/exp',language:'cute-dsl',
  task_result_command:'trusted {kernel_path} {result_path}',task_workload_count:17}
function raw(i,extra={}) {
  const p=`/fixture/exp/generated/initial_${i}.json`
  return {contract_version:'kersor-task-result-v1',test_result_path:p,candidate_path:p+'.artifact/candidate.py',
    source_binding:{verified:true,source_sha256:String(i+1).repeat(64)},
    compiled:i===1,correct:false,measurement_valid:false,full_workload_set:true,n_pass:0,n_total:17,
    outcome_state:'candidate_failure',failure_origin:'candidate',
    diagnostics:[{message:'COMPLETE_DIAGNOSTIC_'+i+'x'.repeat(10000)}],
    evidence:{compile_stderr:`/fixture/exp/full-stderr-${i}.txt`},...extra}
}
function fixture(extra={}) {
  const agents={'generate-initial-kernel':{initial_candidates:[0,1,2].map(i=>({variant_path:`/fixture/exp/generated/initial_${i}.py`}))}}
  for(let i=0;i<3;i++) agents[`initial-${i}`]={test_result_path:`/fixture/exp/generated/initial_${i}.json`,test_result_json:raw(i,i===1?extra:{})}
  return agents
}
for(const workflow of ['AccelOpt/accelopt-kernel-optimization.js','CUDAAgent/cuda-agent-kernel-optimization.js']) {
  const source=fs.readFileSync(path.join(root,workflow),'utf8')
  test(workflow+' returns typed seed exhaustion after every declared failure, without another request',async()=>{
    const {result,calls}=await run(source,args,fixture(),null)
    assert.equal(result.status,'no_valid_candidate')
    assert.equal(result.outcome_state,'no_valid_candidate')
    assert.equal(result.stop_reason,'seed_exhausted')
    assert.equal(result.termination_reason,'seed_exhausted')
    assert.equal(result.correct,false)
    assert.equal(result.compiled,true)
    assert.equal(result.canonical_metric,null)
    assert.equal(result.generated_kernel_path,'')
    assert.equal(result.best_candidate_binding,null)
    assert.equal(Object.hasOwn(result,'resources_released'),false)
    assert.deepEqual(calls.map(c=>c.label),['generate-initial-kernel','initial-0','initial-1','initial-2'])
    assert.deepEqual(JSON.parse(JSON.stringify(result.seed_budget)),{declared:3,evaluated:3,remaining:0})
    assert.equal(result.declared_seed_slots.length,3)
    for(let i=0;i<3;i++) {
      assert.equal(result.declared_seed_slots[i].index,i)
      assert.equal(result.declared_seed_slots[i].candidate_path,`/fixture/exp/generated/initial_${i}.py`)
      assert.equal(result.initial_candidates[i].test_result_path,`/fixture/exp/generated/initial_${i}.json`)
      assert.deepEqual(JSON.parse(JSON.stringify(result.initial_candidates[i].raw_task_result)),raw(i))
    }
  })
  test(workflow+' retains explicitly unknown compile status for known candidate failures',async()=>{
    const {result}=await run(source,args,fixture({compiled:null}),null)
    assert.equal(result.status,'no_valid_candidate')
    assert.equal(result.compiled,null)
    assert.equal(result.correct,false)
  })
  for(const [name,extra] of Object.entries({missing:{source_binding:null},unknown:{outcome_state:'unknown'},permission:{failure_origin:'permission'},partial:{full_workload_set:false},passed:{correct:true,measurement_valid:true,n_pass:17,candidate_latency_aggregate_ms:.02,speedup_vs_reference:1.5}})) {
    if(name==='passed') continue
    test(workflow+' '+name+' remains HOLD, never completed exhaustion',async()=>{
      const agents=fixture(extra),seen=[]
      for(const label of ['initial-0','initial-1','initial-2']) {
        const value=agents[label]
        Object.defineProperty(agents,label,{get(){seen.push(label);return value}})
      }
      await assert.rejects(run(source,args,agents,null),e=>e.code==='TASK_RESULT_HOLD'&&e.retryable===false)
      assert.deepEqual(seen,['initial-0','initial-1'])
    })
  }
}

test('seed terminal reducer rechecks the exact complete normalized and raw evidence set',()=>{
  const scaffold=fs.readFileSync(path.join(root,'_meta/scaffolding/task-result.js'),'utf8')
  const {parse,exhausted}=vm.runInNewContext(scaffold+';({parse:__taskResult,exhausted:__nativeTaskSeedExhausted})')
  const original=[0,1,2].map(i=>({...parse({test_result_json:raw(i)},raw(i).test_result_path,17),variant_path:`/fixture/exp/generated/initial_${i}.py`}))
  for(const mutation of ['missing','duplicate','source','raw','compile','measurement']) {
    const candidates=JSON.parse(JSON.stringify(original))
    if(mutation==='missing') candidates.pop()
    else if(mutation==='duplicate') candidates[1]=candidates[0]
    else if(mutation==='source') candidates[0].source_binding.verified=false
    else if(mutation==='raw') delete candidates[0].raw_task_result
    else if(mutation==='compile') delete candidates[0].raw_task_result.compiled
    else candidates[0].raw_task_result.measurement_valid=true
    assert.throws(()=>exhausted({workflow:'fixture',expDir:args.exp_dir,seedCount:3,workloadCount:17,candidates}),e=>e.code==='TASK_RESULT_HOLD',mutation)
  }
})
