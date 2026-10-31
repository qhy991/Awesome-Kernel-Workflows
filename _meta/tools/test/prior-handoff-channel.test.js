'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os')
const run=require('../lib/run-workflow.js')
const root=path.resolve(__dirname,'../../..')
const SSOT=fs.readFileSync(path.join(root,'_meta/scaffolding/typed-args.js'),'utf8')
const HANDOFFS=['/runs/r1/stages/0001/handoff','/runs/r1/stages/0002/handoff']

function block(args){
  const b=SSOT.indexOf('// --- BEGIN typed-args'),e=SSOT.indexOf('// --- END typed-args ---')
  return Function('args',`${SSOT.slice(b,e)}\nreturn __priorHandoffBlock()`)(args)
}
const names=prompt=>HANDOFFS.filter(h=>prompt.includes(`${h}/INDEX.md`))

test('the block names every handoff index in stage order and points to the author README',()=>{
  const text=block({prior_handoffs:[...HANDOFFS,7,'',null]})
  assert.ok(text.indexOf(`stage 1: ${HANDOFFS[0]}/INDEX.md`)<text.indexOf(`stage 2: ${HANDOFFS[1]}/INDEX.md`))
  assert.ok(text.includes(`${HANDOFFS[1]}/handoff.json`))
  assert.ok(!/stage 3/.test(text))
  assert.match(text,/author notes are unverified/)
  assert.match(text,/author's README\.md/)
  assert.ok(!text.includes(`${HANDOFFS[0]}/README.md`),'the index is INDEX.md; README.md belongs to the author')
  assert.match(text,/only inherited implementation/)
})

test('no channel, an empty list or a malformed value adds nothing',()=>{
  for(const args of [{},{prior_handoffs:[]},{prior_handoffs:'/runs/r1/stages/0001/handoff'}]) assert.equal(block(args),'')
})

const measured=(p,lat,extra={})=>({contract_version:'kersor-task-result-v1',test_result_path:p,compiled:true,correct:true,
  full_workload_set:true,measurement_valid:true,n_pass:17,n_total:17,candidate_latency_aggregate_ms:lat,speedup_vs_reference:.06/lat,
  candidate_path:p+'.artifact/candidate.py',source_binding:{verified:true,source_sha256:'a'.repeat(64)},...extra})

// A clean-start native task run: seed generation, then one plan/implement round.
async function cleanStart(file,extra){
  const exp=fs.mkdtempSync(path.join(os.tmpdir(),'akw-prior-handoff-'))
  try{
    fs.mkdirSync(path.join(exp,'generated'))
    for(let i=0;i<3;i++) fs.writeFileSync(path.join(exp,'generated',`initial_${i}.py`),`fixture source ${i}`)
    const agents={'generate-initial-kernel':{initial_candidates:[0,1,2].map(i=>({variant_path:`${exp}/generated/initial_${i}.py`}))},
      'profile-baseline':{bottlenecks:[],optimization_strategy:'fixture'},
      'read-baseline':{kernel_code:'display',op_type:'gemm',key_functions:['run']},
      'ncu-baseline':{ncu_available:false,profile_summary:'unavailable',bottleneck_diagnosis:'unknown'},
      'plan-0-0':{title:'p',plan:'fixture'},'impl-0-p-v0':{code:'display',variant_path:`${exp}/iter_0_plan_0_sample_0.py`}}
    for(let i=0;i<3;i++){const p=`${exp}/generated/initial_${i}.json`;agents[`initial-${i}`]={test_result_path:p,test_result_json:measured(p,[.03,.02,.04][i])}}
    for(let i=0;i<15;i++){
      agents[`impl-${i}`]={kernel_code:'display',variant_path:`${exp}/task_attempt_${i}.py`}
      agents[`task-verify-${i}`]={test_result_path:`${exp}/task_attempt_${i}.json`,test_result_json:measured(`${exp}/task_attempt_${i}.json`,.01)}
    }
    const variant=`${exp}/iter_0_plan_0_sample_0.task.json`
    agents['task-eval-iter_0_plan_0_sample_0']={test_result_path:variant,test_result_json:measured(variant,.01)}
    const args={problem_path:'/fixture/task.md',exp_dir:exp,language:'cute-dsl',task_result_command:'trusted --candidate {kernel_path} --result {result_path}',
      task_workload_count:17,seed_candidates:3,iterations:1,breadth:1,samples_per_plan:1,...extra}
    return (await run(fs.readFileSync(path.join(root,file),'utf8'),args,agents,null)).calls
  }finally{fs.rmSync(exp,{recursive:true,force:true})}
}

const sites={
  CUDAAgent:{file:'CUDAAgent/cuda-agent-kernel-optimization.js',labels:[/^generate-initial-kernel/,/^impl-\d+$/]},
  AccelOpt:{file:'AccelOpt/accelopt-kernel-optimization.js',labels:[/^generate-initial-kernel$/,/^plan-0-0$/,/^impl-0-p-v0$/]},
}
for(const [method,{file,labels}] of Object.entries(sites)){
  test(`${method}: generation, planning and implementation read the earlier stages`,async()=>{
    const calls=await cleanStart(file,{prior_handoffs:HANDOFFS})
    for(const label of labels){
      const prompts=calls.filter(c=>label.test(c.label))
      assert.ok(prompts.length,`${label} ran`)
      for(const c of prompts) assert.deepEqual(names(c.prompt),HANDOFFS,`${c.label} names both handoffs`)
    }
  })
  test(`${method}: without the channel no prompt mentions an earlier stage`,async()=>{
    const calls=await cleanStart(file,{})
    assert.ok(calls.length)
    for(const c of calls) assert.ok(!/Earlier stages of this chain/.test(c.prompt),c.label)
  })
}

test('KernelFoundry: every offspring generation reads the earlier stages',async()=>{
  const m=(p,score)=>({test_result_path:p,test_result_json:JSON.stringify(measured(p,.06/score))})
  const agents={'accepted-parent-task':m('/fixture/exp/accepted_parent.task.json',2),setup:{operator_code:'spec',baseline_time_ms:.001},
    'load-checkpoint':{present:false},'accepted-parent-descriptor':{d_mem:1,d_algo:2,d_sync:3}}
  for(let i=0;i<3;i++){agents[`vary-${i}`]={kernel_code:'DISPLAY',variant_path:`/fixture/exp/gen_${i}.py`,d_mem:1,d_algo:2,d_sync:3}
    agents[`task-eval-${i}`]=m(`/fixture/exp/gen_${i}_result.json`,2.5)}
  const args={problem_path:'/fixture/task.md',exp_dir:'/fixture/exp',language:'cute-dsl',kernel_path:'/fixture/accepted-parent.py',
    task_result_command:'trusted --candidate {kernel_path} --result {result_path}',task_workload_count:17,meta_prompt_interval:99,
    test_command:'trusted',benchmark_command:'trusted',generations:3,prior_handoffs:HANDOFFS}
  const {calls}=await run(fs.readFileSync(path.join(root,'KernelFoundry/kernelfoundry-kernel-optimization.js'),'utf8'),args,agents,null)
  const vary=calls.filter(c=>/^vary-\d+$/.test(c.label))
  assert.equal(vary.length,3)
  for(const c of vary) assert.deepEqual(names(c.prompt),HANDOFFS,c.label)
})
