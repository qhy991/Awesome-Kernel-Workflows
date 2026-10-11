'use strict'
const {test}=require('node:test'), assert=require('node:assert/strict')
const fs=require('node:fs'),path=require('node:path')
const run=require('../lib/run-workflow.js')
const root=path.resolve(__dirname,'../../..')
const sources={ksearch:fs.readFileSync(path.join(root,'KSearch/ksearch-kernel-optimization.js'),'utf8'),
  foundry:fs.readFileSync(path.join(root,'KernelFoundry/kernelfoundry-kernel-optimization.js'),'utf8')}
const base={problem_path:'/fixture/task.md',exp_dir:'/fixture/exp',language:'cute-dsl',
  task_result_command:'trusted {kernel_path} {result_path}',task_workload_count:2,
  test_command:'trusted',benchmark_command:'trusted'}
function result(p,valid=true) {return {test_result_path:p,test_result_json:{
  contract_version:'kersor-task-result-v1',test_result_path:p,compiled:valid,correct:valid,
  full_workload_set:true,measurement_valid:valid,n_pass:valid?2:0,n_total:2,
  candidate_latency_aggregate_ms:valid?.02:null,speedup_vs_reference:valid?.75:null,
  outcome_state:valid?'passed':'candidate_failure',...(valid?{}:{failure_origin:'candidate'}),
  candidate_path:p+'.artifact/candidate.py',source_binding:{verified:true,source_sha256:'a'.repeat(64)},
  diagnostics:valid?[]:[{error:'candidate compiler rejection'}]}}}
function tree() {return {root:{node_id:'root',parent_id:null,status:'solved'},
  n1:{node_id:'n1',parent_id:'root',status:'open',action:{title:'tile',description:'tile action',difficulty_1_to_5:2,score_0_to_1:.8}}}}
function ksearchAgents() {
  const a={'read-spec':{spec_text:'spec',op_type:'gemm'},'load-checkpoint':{present:false},'init-tree':{decision_tree:tree()}}
  for(let i=0;i<2;i++) {
    a[`propose-${i}`]={updated_tree:tree()}
    a[`select-${i}`]={selected_node_id:'n1',action_title:'tile',parent_is_root:true,parent_metric:1}
    a[`gen-${i}-0`]={variant_path:`/fixture/exp/cycle_${i}_a0.py`}
    a[`debug-${i}-1`]={variant_path:`/fixture/exp/cycle_${i}_a1.py`}
    a[`task-eval-${i}-0`]=result(`/fixture/exp/task_cycle_${i}_a0.json`,false)
    a[`task-eval-${i}-1`]=result(`/fixture/exp/task_cycle_${i}_a1.json`,false)
  }
  return a
}
for(const code of ['KERSOR_PERMISSION_DENIED','KERSOR_BUDGET_EXHAUSTED','KERSOR_CLAUDE_TRANSPORT_FAULT','KERSOR_TURN_UNSETTLED'])
test('KSearch propagates '+code+' without backtracking or another cycle',async()=>{
  const a=ksearchAgents(), seen=[]
  const error=Object.assign(new Error('original diagnostic'),{code,retryable:false})
  a['debug-0-1']={then(_resolve,reject){reject(error)}}
  const tracked=new Proxy(a,{get(target,key){seen.push(key);return target[key]}})
  await assert.rejects(run(sources.ksearch,{...base,iterations:2,attempts_per_cycle:2,seed_candidates:1},tracked,null),e=>e===error)
  assert.ok(!seen.includes('backtrack-0'))
  assert.ok(!seen.includes('propose-1'))
})
for(const code of ['KERSOR_CODEX_TIMEOUT','KERSOR_CLAUDE_TIMEOUT'])
test('KSearch retains cycle completion for settled '+code,async()=>{
  const a=ksearchAgents()
  a['debug-0-1']={then(_resolve,reject){reject(Object.assign(new Error('settled turn timeout'),{code,retryable:false}))}}
  const {calls}=await run(sources.ksearch,{...base,iterations:1,attempts_per_cycle:2,seed_candidates:1},a,null)
  assert.ok(calls.some(c=>c.label==='backtrack-0'))
  assert.ok(calls.some(c=>c.label==='checkpoint-0'))
})
function foundryAgents(valid=false) {
  const a={setup:{operator_code:'spec',operator_type:'gemm'}}
  for(let i=0;i<2;i++) {
    a[`vary-${i}`]={kernel_code:'display',variant_path:`/fixture/exp/gen_${i}.py`,strategy_description:'tile',d_mem:1,d_algo:1,d_sync:1}
    a[`task-eval-${i}`]=result(`/fixture/exp/gen_${i}_result.json`,valid)
  }
  return a
}
test('KernelFoundry observes cooperative stop after a failed candidate',async()=>{
  const a=foundryAgents();a['checkpoint-0']={termination_requested:true,termination_reason:'test_stop'}
  const {calls,result:r}=await run(sources.foundry,{...base,generations:2},a,null)
  assert.equal(calls.filter(c=>c.label.startsWith('task-eval-')).length,1)
  assert.equal(calls.filter(c=>c.label.startsWith('checkpoint-')).length,1)
  assert.equal(r.archive_coverage,0)
  assert.equal(r.generations_completed,1)
})
test('KernelFoundry checkpoints every rejected generation without inserting it',async()=>{
  const {calls,result:r}=await run(sources.foundry,{...base,generations:2},foundryAgents(),null)
  assert.equal(calls.filter(c=>c.label.startsWith('checkpoint-')).length,2)
  assert.equal(r.archive_coverage,0)
  assert.equal(r.generations_completed,2)
  assert.ok(calls.find(c=>c.label==='vary-1').prompt.includes('/fixture/exp/gen_0_result.json'))
})
for(const coordinate of [1.5,-1,4,'1']) test('KernelFoundry rejects non-cell descriptor '+JSON.stringify(coordinate),async()=>{
  const a=foundryAgents(true);a['vary-0'].d_mem=coordinate
  await assert.rejects(run(sources.foundry,{...base,generations:2},a,null),/behavioral descriptor/)
})

test('KSearch native frontier selection ignores model-invented node, source and parent score',async()=>{
  const a=ksearchAgents()
  a['select-0']={selected_node_id:'invented',action_title:'unregistered action',parent_solution_path:'/foreign.py',parent_metric:999}
  const {calls}=await run(sources.ksearch,{...base,iterations:1,attempts_per_cycle:1,seed_candidates:1},a,null)
  const prompt=calls.find(c=>c.label==='gen-0-0').prompt
  assert.ok(prompt.includes('tile action'))
  assert.ok(!prompt.includes('unregistered action'))
  assert.ok(!prompt.includes('/foreign.py'))
})
test('KSearch native frontier does not relax its declared difficulty cap',async()=>{
  const a=ksearchAgents(),t=tree();t.n1.action.difficulty_1_to_5=5
  a['init-tree']={decision_tree:t};a['propose-0']={updated_tree:t}
  const {calls}=await run(sources.ksearch,{...base,max_difficulty:4,iterations:1,attempts_per_cycle:1,seed_candidates:1},a,null)
  assert.equal(calls.filter(c=>c.label.startsWith('gen-')).length,0)
})

test('KSearch deterministic selection derives scores and source only from the measured parent',()=>{
  const start=sources.ksearch.indexOf('function ksearchTaskSelection(')
  const end=sources.ksearch.indexOf('\nconst ksearchCheckpointReadProgram',start)
  const select=require('node:vm').runInNewContext(sources.ksearch.slice(start,end)+';ksearchTaskSelection',
    {__taskHold:message=>{throw Error(message)}})
  const t=tree();t.n1.status='solved';t.n1.solution_id='measured'
  const action={title:'child',description:'child action',difficulty_1_to_5:2,score_0_to_1:.9}
  t.c2={parent_id:'n1',status:'open',action};t.c1={parent_id:'n1',status:'open',action}
  const measured={id:'measured',node_id:'n1',eval:{is_valid:true,metric_value:2,host_candidate_path:'/measured.py'}}
  const s=select(t,[measured],null,1,4)
  assert.equal(s.selected_node_id,'c1')
  assert.equal(s.parent_metric,2)
  assert.equal(s.parent_solution_path,'/measured.py')
  assert.throws(()=>select(t,[{...measured,node_id:'foreign'}],null,1,4),/bound passing measurement/)
  assert.throws(()=>select(t,[{...measured,eval:{...measured.eval,host_candidate_path:''}}],null,1,4),/bound passing measurement/)
})
