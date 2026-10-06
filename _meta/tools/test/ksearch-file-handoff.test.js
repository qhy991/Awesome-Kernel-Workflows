'use strict'
const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto')
const {spawnSync}=require('node:child_process')
const run=require('../lib/run-workflow.js')
const root=path.resolve(__dirname,'../../..')
const source=fs.readFileSync(path.join(root,'KSearch/ksearch-kernel-optimization.js'),'utf8')
const args={problem_path:'/fixture/task.md',exp_dir:'/fixture/exp',language:'cute-dsl',iterations:4,seed_candidates:2,attempts_per_cycle:2,task_result_command:'trusted --candidate {kernel_path} --result {result_path}',task_workload_count:29}
function measured(cycle,attempt,score,rejected=false){
 const resultPath=`/fixture/exp/task_cycle_${cycle}_a${attempt}.json`
 return {test_result_path:resultPath,test_result_json:JSON.stringify({
  contract_version:'kersor-task-result-v1',test_result_path:resultPath,
  compiled:true,correct:!rejected,full_workload_set:true,measurement_valid:!rejected,
  n_pass:rejected?28:29,n_total:29,candidate_latency_aggregate_ms:rejected?null:.03,
  speedup_vs_reference:rejected?null:score,candidate_path:resultPath+'.artifact/candidate.py',
  source_binding:{verified:true,source_sha256:'a'.repeat(64)},
  ...(rejected?{outcome_state:'candidate_failure',failure_origin:'candidate'}:{})
 })}
}
function fixture(){
 const tree={root:{},n1:{parent_id:'root',status:'open'}}
 const agents={'read-spec':{spec_text:'GEMM task',op_type:'gemm',baseline_code:'',constraints:[],design_dimensions:[]},'init-tree':{decision_tree:tree},'load-checkpoint':{present:false},'final-report':'file-backed report'}
 for(let cycle=0;cycle<4;cycle++){
  agents[`propose-${cycle}`]={updated_tree:tree,open_frontier_count:1}
  agents[`select-${cycle}`]={selected_node_id:'n1',action_title:'tile',parent_solution_path:cycle?`/fixture/exp/task_cycle_${cycle-1}_a1.json.artifact/candidate.py`:'',parent_metric:1}
  agents[`refine-${cycle}`]={updated_tree:tree}
  agents[`checkpoint-${cycle}`]={termination_requested:false,checkpoint_path:'/fixture/exp/checkpoint.json'}
  for(let attempt=0;attempt<3;attempt++)agents[`task-eval-${cycle}-${attempt}`]=measured(cycle,attempt,1+cycle*.1+attempt*.02)
  for(let attempt=0;attempt<2;attempt++)agents[`gen-${cycle}-${attempt}`]={variant_path:`/fixture/exp/cycle_${cycle}_a${attempt}.py`}
 }
 return agents
}
function payload(call){return JSON.parse(call.prompt.split('3. Start from this exact checkpoint object:\n')[1].split('\n   If step 2')[0])}

test('four native cycles accept file-only producers and preserve measured best without source echoes',async()=>{
 const {result,calls}=await run(source,args,fixture(),null)
 assert.equal(result.cycles_completed,4);assert.equal(result.solutions_evaluated,8)
 assert.equal(result.generated_kernel_path,'/fixture/exp/task_cycle_3_a1.json.artifact/candidate.py')
 assert.equal(result.source_binding.source_sha256,'a'.repeat(64));assert.equal(result.best_solution_code,'')
 const saves=calls.filter(x=>x.label?.startsWith('checkpoint-'))
 assert.equal(saves.length,4)
 const final=payload(saves[3]);assert.equal(final.progress.completed,4);assert.equal(final.solutionDb.length,8)
 for(const record of [...final.solutionDb,final.bestSolution]){
  assert.equal('code' in record,false);assert.equal(record.path,record.eval.host_candidate_path)
  assert.equal(record.eval.source_binding.source_sha256,'a'.repeat(64));assert.equal(record.eval.n_total,29)
 }
 assert.deepEqual(calls.find(x=>x.label==='gen-0-0').schema.required,['variant_path'])
 assert.ok(calls.find(x=>x.label==='gen-1-0').prompt.includes('task_cycle_0_a1.json.artifact/candidate.py'))
 assert.ok(calls.find(x=>x.label==='final-report').prompt.includes(result.generated_kernel_path))
})

test('large compatibility code returned by eight producers never enters later task prompts or checkpoints',async()=>{
 const agents=fixture(),marker='INLINE_SOURCE_SHOULD_STAY_IN_FILE'
 for(let c=0;c<4;c++)for(let a=0;a<2;a++)agents[`gen-${c}-${a}`].code=marker+'x'.repeat(120000)
 const {calls,result}=await run(source,args,agents,null)
 assert.equal(result.solutions_evaluated,8)
 assert.ok(calls.every(x=>!x.prompt.includes(marker)))
 const final=calls.find(x=>x.label==='checkpoint-3')
 assert.ok(final.prompt.length<20000,'eight source bodies must not enlarge the checkpoint prompt')
 assert.equal(payload(final).solutionDb.length,8)
})

test('native debug and improve use frozen input paths and distinct output files without requiring code',async()=>{
 const agents=fixture();agents['gen-0-0']={variant_path:'/fixture/exp/cycle_0_a0.py'}
 agents['task-eval-0-0']=measured(0,0,0,true)
 agents['debug-0-1']={variant_path:'/fixture/exp/cycle_0_a1.py'}
 agents['improve-0-2']={variant_path:'/fixture/exp/cycle_0_a2.py'}
 const {calls,result}=await run(source,{...args,iterations:1,seed_candidates:1,attempts_per_cycle:3},agents,null)
 assert.equal(result.solutions_evaluated,3)
 for(const [label,input,output] of [['debug-0-1','task_cycle_0_a0.json.artifact/candidate.py','cycle_0_a1.py'],['improve-0-2','task_cycle_0_a1.json.artifact/candidate.py','cycle_0_a2.py']]){
  const call=calls.find(x=>x.label===label);assert.ok(call.prompt.includes(input));assert.ok(call.prompt.includes(output))
  assert.deepEqual(call.schema.required,['variant_path'])
 }
})

test('old inline checkpoint resumes from its next cycle and measured parent path without mutating input',async()=>{
 const agents=fixture(),evaluation=JSON.parse(measured(0,1,2).test_result_json)
 const oldRecord={id:'old-best',node_id:'n1',code:'OLD_INLINE_SOURCE'.repeat(20000),eval:{is_valid:true,metric_value:2,host_candidate_path:evaluation.candidate_path,source_binding:evaluation.source_binding}}
 const old={cycle:2,bestMetric:2,best_candidate_id:'old-best',decisionTree:{root:{},n1:{parent_id:'root'}},solutionDb:[oldRecord],bestSolution:oldRecord}
 agents['load-checkpoint']=old;agents['select-2'].parent_solution_path=evaluation.candidate_path
 const {result,calls}=await run(source,{...args,iterations:3},agents,null)
 assert.equal(calls.some(x=>x.label==='gen-0-0'),false);assert.equal(calls.some(x=>x.label==='gen-1-0'),false)
 assert.ok(calls.find(x=>x.label==='gen-2-0').prompt.includes(evaluation.candidate_path))
 assert.equal(result.best_metric,2);assert.equal(result.generated_kernel_path,evaluation.candidate_path)
 assert.ok(oldRecord.code.startsWith('OLD_INLINE_SOURCE'))
 assert.ok(calls.every(x=>!x.prompt.includes('OLD_INLINE_SOURCE')))
 assert.equal('code' in payload(calls.find(x=>x.label==='checkpoint-2')).bestSolution,false)
})

test('checkpoint reader executes on CPU, retains state and original files, rejects missing/changed source',async()=>{
 const {calls}=await run(source,{...args,iterations:0},fixture(),null)
 const program=calls.find(x=>x.label==='load-checkpoint').prompt.match(/```python\n([\s\S]*?)\n```/)[1]
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'ksearch-checkpoint-'))
 try{
  const candidate=path.join(dir,'candidate.py'),checkpoint=path.join(dir,'checkpoint.json'),bytes='complete source\n'
  fs.writeFileSync(candidate,bytes)
  const record={id:'best',code:'INLINE_SOURCE'.repeat(50000),eval:{host_candidate_path:candidate,source_binding:{source_sha256:crypto.createHash('sha256').update(bytes).digest('hex')},metric_value:1.2,evidence:{trace:'original-trace'}}}
  const old={cycle:3,decisionTree:{root:{children:['n1']}},bestMetric:1.2,solutionDb:[record],bestSolution:record}
  fs.writeFileSync(checkpoint,JSON.stringify(old));const original=fs.readFileSync(checkpoint)
  const execute=()=>{
   const p=spawnSync('python3',['-',checkpoint],{input:program,encoding:'utf8'});assert.equal(p.status,0,p.stderr)
   return JSON.parse(p.stdout)
  }
  const compact=execute();assert.equal(compact.cycle,3);assert.deepEqual(compact.decisionTree,old.decisionTree)
  assert.equal('code' in compact.bestSolution,false);assert.equal(compact.bestSolution.path,candidate)
  assert.equal(compact.bestSolution.eval.evidence.trace,'original-trace');assert.ok(JSON.stringify(compact).length<2000)
  assert.deepEqual(fs.readFileSync(checkpoint),original);assert.equal(fs.readFileSync(candidate,'utf8'),bytes)
  fs.writeFileSync(candidate,'changed');assert.match(execute().checkpoint_error,/binding changed/)
  fs.rmSync(candidate);assert.match(execute().checkpoint_error,/source is unavailable/)
  fs.rmSync(checkpoint);assert.deepEqual(execute(),{present:false})
 }finally{fs.rmSync(dir,{recursive:true,force:true})}
})
