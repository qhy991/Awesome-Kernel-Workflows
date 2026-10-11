'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const runWorkflow = require('../lib/run-workflow.js')
const root = path.resolve(__dirname, '../../..')
const source = fs.readFileSync(path.join(root, 'AccelOpt/accelopt-kernel-optimization.js'), 'utf8')
const args = {problem_path:'/fixture/task.md', exp_dir:'/fixture/ao', language:'cute-dsl',
  kernel_path:'/fixture/accepted.py', task_result_command:'trusted --candidate {kernel_path} --result {result_path}',
  task_workload_count:17, iterations:2, breadth:1, samples_per_plan:1,
  termination_file:'/fixture/STOP', deadline_epoch:2000000000}

function measurement(resultPath, latency, extra = {}) {
  return {test_result_path:resultPath, test_result_json:{contract_version:'kersor-task-result-v1',
    test_result_path:resultPath, candidate_path:resultPath + '.artifact/candidate.py',
    source_binding:{verified:true,source_sha256:'a'.repeat(64)}, compiled:true,correct:true,
    full_workload_set:true, measurement_valid:true, n_pass:17,n_total:17,
    candidate_latency_aggregate_ms:latency, speedup_vs_reference:.015/latency, ...extra}}
}
function fixture({fail = false, stop = false, profile = false} = {}) {
  const agents = {
    'task-accepted-parent':measurement('/fixture/ao/accepted_parent.task.json', .020),
    'read-baseline':{kernel_code:'BASELINE_DISPLAY',op_type:'gemm',key_functions:['run']},
    'ncu-baseline':{ncu_available:profile,report_path:profile?'/fixture/profile.ncu-rep':'',profile_summary:'initial source profile',bottleneck_diagnosis:'fixture'},
  }
  for (let i=0;i<2;i++) {
    agents[`plan-${i}-0`] = {title:'tile',plan:'fixture',ncu_evidence:'fixture'}
    agents[`impl-${i}-tile-v0`] = {code:'DISPLAY_ONLY',variant_path:`/fixture/ao/iter_${i}_plan_0_sample_0.py`}
    agents[`task-eval-iter_${i}_plan_0_sample_0`] = measurement(`/fixture/ao/iter_${i}_plan_0_sample_0.task.json`, .016,
      fail ? {compiled:false,correct:false,measurement_valid:false,n_pass:0,
        candidate_latency_aggregate_ms:null,speedup_vs_reference:0,outcome_state:'candidate_failure',failure_origin:'candidate',
        diagnostics:[{message:'COMPLETE_ORIGINAL_COMPILER_FAILURE'.repeat(1000)}], evidence:{compile_stderr:'/fixture/full-compiler.txt'}} :
      {source_binding:{verified:true,source_sha256:'b'.repeat(64)}})
    agents[`checkpoint-${i}`] = {termination_requested:stop&&i===0,termination_reason:stop?'supervisor_request':'',checkpoint_path:'/fixture/ao/checkpoint.json'}
  }
  return agents
}
function checkpoint(call) {
  return JSON.parse(call.prompt.match(/Start from this exact checkpoint object:\n([^\n]+)/)[1])
}

test('AccelOpt runtime safe point is generated from the canonical helper', () => {
  const canonical = fs.readFileSync(path.join(root,'_meta/scaffolding/runtime-safe-point.js'),'utf8')
  const helper = canonical.slice(canonical.indexOf('async function __workflowRuntimeSafePoint(ctx) {')).trim()
  const begin = '// --- BEGIN inlined runtime-safe-point scaffolding (from _meta/scaffolding/runtime-safe-point.js) ---'
  const end = '// --- END inlined runtime-safe-point scaffolding ---'
  assert.equal(source.split(begin+'\n')[1].split('\n'+end)[0],helper)
})

test('AccelOpt feeds typed candidate failures by complete source/result references to next planner and implementer, outside performance Learn', async () => {
  const {result,calls} = await runWorkflow(source,args,fixture({fail:true}),null)
  const first = '/fixture/ao/iter_0_plan_0_sample_0.task.json'
  for (const label of ['plan-1-0','impl-1-tile-v0']) {
    const prompt = calls.find(c=>c.label===label).prompt
    assert.ok(prompt.includes(first))
    assert.ok(prompt.includes(first+'.artifact/candidate.py'))
    assert.ok(prompt.includes('Read each relevant COMPLETE test_result_path'))
    assert.ok(prompt.includes('separate from performance learning'))
    assert.ok(!prompt.includes('COMPLETE_ORIGINAL_COMPILER_FAILURE'))
  }
  assert.equal(calls.filter(c=>c.label.startsWith('learn-')).length,0)
  assert.equal(result.failure_evidence.length,2)
  assert.equal(result.failure_evidence[0].source_binding.source_sha256,'a'.repeat(64))
  assert.equal(result.generated_kernel_path,'/fixture/ao/accepted_parent.task.json.artifact/candidate.py')
  assert.equal(result.iterations_completed,2)
  assert.equal(calls.filter(c=>c.label.startsWith('task-eval-')).length,2)
})

test('AccelOpt stops after complete iteration even when every candidate fails, retains source-bound incumbent, and reports actual budget use', async () => {
  const {result,calls} = await runWorkflow(source,args,fixture({fail:true,stop:true}),null)
  assert.equal(result.iterations_completed,1)
  assert.equal(result.iterations_requested,2)
  assert.equal(result.termination_reason,'supervisor_request')
  assert.equal(result.report,null)
  assert.equal(calls.filter(c=>c.label.startsWith('task-eval-')).length,1)
  assert.ok(!calls.some(c=>c.label==='plan-1-0'||c.label==='final-report'))
  const saved = checkpoint(calls.find(c=>c.label==='checkpoint-0'))
  assert.deepEqual(saved.progress,{unit:'iteration',completed:1,total:2})
  assert.equal(saved.failure_evidence[0].outcome_state,'candidate_failure')
  assert.equal(saved.best_candidate_binding.source_sha256,'a'.repeat(64))
  assert.equal(saved.best_candidate_binding.test_result_path,'/fixture/ao/accepted_parent.task.json')
  assert.deepEqual(saved.replay_binding,{mode:'native_same_activation_only',arguments:args})
  assert.equal(saved.metric.value,1)
  assert.ok(calls.find(c=>c.label==='checkpoint-0').prompt.includes('/fixture/STOP'))
})

test('AccelOpt explicitly labels initial profile historical after a new measured incumbent', async () => {
  const {calls,result} = await runWorkflow(source,args,fixture({profile:true}),null)
  const first = calls.find(c=>c.label==='plan-0-0').prompt
  const second = calls.find(c=>c.label==='plan-1-0').prompt
  assert.ok(first.includes('profiled source is still the incumbent'))
  assert.ok(second.includes('HISTORICAL baseline evidence'))
  assert.ok(second.includes('current incumbent has no newly verified profile'))
  assert.ok(second.includes('/fixture/ao/accepted_parent.task.json.artifact/candidate.py'))
  assert.ok(second.includes('/fixture/profile.ncu-rep'))
  assert.equal(result.overall_speedup,1.25)
})

test('AccelOpt seed failures also remain available after selecting a valid seed', async () => {
  const agents = fixture({fail:true})
  agents['generate-initial-kernel'] = {initial_candidates:[0,1,2].map(i=>({variant_path:`/fixture/ao/generated/initial_${i}.py`}))}
  for (let i=0;i<3;i++) agents[`initial-${i}`] = measurement(`/fixture/ao/generated/initial_${i}.json`,.02,
    i===0?{compiled:false,correct:false,measurement_valid:false,n_pass:0,outcome_state:'candidate_failure',failure_origin:'candidate'}:{})
  const {result,calls} = await runWorkflow(source,{...args,kernel_path:''},agents,null)
  assert.ok(calls.find(c=>c.label==='plan-0-0').prompt.includes('/fixture/ao/generated/initial_0.json'))
  assert.equal(result.failure_evidence[0].candidate_id,'initial_0')
})

test('AccelOpt legacy checkpoint never inlines beam source or invents absent source bindings',async()=>{
  const legacyArgs=JSON.parse(fs.readFileSync(path.join(root,'_meta/tools/fixtures/accelopt-cuda-args.json'),'utf8'))
  const agents=JSON.parse(fs.readFileSync(path.join(root,'_meta/tools/fixtures/accelopt-cuda-agent-returns.json'),'utf8'))
  agents['read-baseline'].kernel_code='LEGACY_CODE_MUST_STAY_IN_FILE'.repeat(1000)
  agents['checkpoint-0']={termination_requested:true,termination_reason:'supervisor_request',checkpoint_path:'/fixture/checkpoint.json'}
  const {result,calls}=await runWorkflow(source,{...legacyArgs,termination_file:'/fixture/STOP'},agents)
  const call=calls.find(c=>c.label==='checkpoint-0')
  assert.ok(!call.prompt.includes('LEGACY_CODE_MUST_STAY_IN_FILE'))
  assert.equal(checkpoint(call).candidate_beam[0].source_binding,null)
  assert.equal(checkpoint(call).candidate_beam[0].source_path,null)
  assert.equal(result.iterations_completed,1)
  assert.ok(!calls.some(c=>c.label==='final-report'))
})
