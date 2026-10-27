'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path')
const root=path.resolve(__dirname,'../../..')
const run=(args,agents)=>require('../lib/run-workflow.js')(fs.readFileSync(path.join(root,'KernelFoundry/kernelfoundry-kernel-optimization.js'),'utf8'),args,agents,null)
const base={problem_path:'/fixture/task.md',exp_dir:'/fixture/exp',language:'cute-dsl',kernel_path:'/fixture/accepted-parent.py',
  task_result_command:'trusted --candidate {kernel_path} --result {result_path}',task_workload_count:17,meta_prompt_interval:99,test_command:'trusted',benchmark_command:'trusted'}
function measured(p,score,extra={}){return {test_result_path:p,test_result_json:JSON.stringify({contract_version:'kersor-task-result-v1',test_result_path:p,compiled:true,correct:true,full_workload_set:true,measurement_valid:true,n_pass:17,n_total:17,candidate_latency_aggregate_ms:.06/score,speedup_vs_reference:score,candidate_path:p+'.artifact/candidate.py',source_binding:{verified:true,source_sha256:'a'.repeat(64)},...extra})}}
const unknown={outcome_state:'unknown',failure_origin:'infrastructure_or_incomplete_evidence',correct:false,measurement_valid:false,n_pass:0}

// An accepted parent already at 10x the reference, then `generations` offspring.
function agents(generations,child){
  const a={'accepted-parent-task':measured('/fixture/exp/accepted_parent.task.json',10),setup:{operator_code:'spec',baseline_time_ms:.001},
    'load-checkpoint':{present:false},'accepted-parent-descriptor':{d_mem:1,d_algo:2,d_sync:3}}
  for(let i=0;i<generations;i++){a[`vary-${i}`]={kernel_code:'DISPLAY',variant_path:`/fixture/exp/gen_${i}.py`,d_mem:1,d_algo:2,d_sync:3}
    a[`task-eval-${i}`]=child(i)}
  return a
}
const evaluated=calls=>calls.filter(c=>/^task-eval-\d+$/.test(c.label)).length

test('default parent basis: an inherited 10x kernel does not meet a 2x target before searching',async()=>{
  const {result,calls}=await run({...base,generations:6},agents(6,i=>measured(`/fixture/exp/gen_${i}_result.json`,11)))
  assert.equal(result.target_basis,'parent')
  assert.equal(result.target_speedup,20)
  assert.equal(result.target_met,false)
  assert.equal(evaluated(calls),6)
})

test('reference basis keeps the absolute target and stops after MIN_GENERATIONS',async()=>{
  const {result,calls}=await run({...base,generations:6,target_basis:'reference'},agents(6,i=>measured(`/fixture/exp/gen_${i}_result.json`,11)))
  assert.equal(result.target_basis,'reference')
  assert.equal(result.target_speedup,2)
  assert.equal(result.target_met,true)
  assert.equal(result.termination_reason,'speedup_target_reached')
  assert.equal(evaluated(calls),2)
})

test('parent basis: reaching 2x the parent still stops on target',async()=>{
  const {result}=await run({...base,generations:6},agents(6,i=>measured(`/fixture/exp/gen_${i}_result.json`,21)))
  assert.equal(result.target_met,true)
  assert.equal(result.termination_reason,'speedup_target_reached')
})

test('an unknown offspring is discarded and the search continues to a later improvement',async()=>{
  const {result,calls}=await run({...base,generations:3},agents(3,i=>i===0
    ?measured('/fixture/exp/gen_0_result.json',1,unknown):measured(`/fixture/exp/gen_${i}_result.json`,12)))
  assert.equal(evaluated(calls),3)
  assert.equal(result.best_speedup,12)
})

test('three consecutive unknown offspring still HOLD the workflow',async()=>{
  await assert.rejects(run({...base,generations:3},agents(3,i=>measured(`/fixture/exp/gen_${i}_result.json`,1,unknown))),
    e=>e.code==='TASK_RESULT_HOLD'&&/3 consecutive unknown/.test(e.message))
})
