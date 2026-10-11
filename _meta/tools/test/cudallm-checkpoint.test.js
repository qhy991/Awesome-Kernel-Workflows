'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const runWorkflow = require('../lib/run-workflow.js')
const root = path.resolve(__dirname, '../../..')
const source = fs.readFileSync(path.join(root, 'CUDALLM/cudallm-fsr-kernel-generation.js'), 'utf8')
const args = {problem_path:'/fixture/task.md',exp_dir:'/fixture/fsr',language:'cute-dsl',
  task_result_command:'trusted --candidate {kernel_path} --result {result_path}',task_workload_count:17,
  iterations:2,samples_per_feature_set:2,termination_file:'/fixture/STOP',deadline_epoch:2000000000}
function fixture(stop, failure) {
  const agents = {'feature-catalog':{features:[{id:'tile',name:'tile'}],baseline_feature_ids:[]}}
  for (let i=0;i<2;i++) for (let j=0;j<2;j++) {
    const p=`/fixture/fsr/cudallm_iter_${i}_sample_${j}.json`
    agents[`select-features-${i}-${j}`]={selected_feature_ids:['tile'],rationale:'fixture'}
    agents[`generate-kernel-${i}-${j}`]={candidate_code:'DO_NOT_INLINE_SOURCE',implemented_feature_ids:['tile']}
    agents[`task-eval-${i}-${j}`]={test_result_path:p,test_result_json:{contract_version:'kersor-task-result-v1',
      test_result_path:p,candidate_path:p+'.artifact/candidate.py',source_binding:{verified:true,source_sha256:'a'.repeat(64)},
      compiled:!failure,correct:!failure,full_workload_set:true,measurement_valid:!failure,n_pass:failure?0:17,n_total:17,
      candidate_latency_aggregate_ms:failure?null:.02,speedup_vs_reference:failure?0:.75,
      ...(failure?{outcome_state:'candidate_failure',failure_origin:'candidate',diagnostics:[{message:'DO_NOT_INLINE_FAILURE'.repeat(1000)}]}:{})}}
    agents[`reinforce-${i}-${j}`]={updated_scores:[{id:'tile',name:'tile',attempts:i*2+j+1,compiled:failure?0:i*2+j+1,
      correct:failure?0:i*2+j+1,reward:failure?-.3:.3,best_speedup:failure?0:.75,failures:failure?[p]:[]}]}
    agents[`checkpoint-${i}-${j}`]={termination_requested:stop&&i===0&&j===0,termination_reason:stop?'deadline':'',checkpoint_path:'/fixture/fsr/checkpoint.json'}
  }
  return agents
}
function checkpoint(calls,label) {
  return JSON.parse(calls.find(c=>c.label===label).prompt.match(/Start from this exact checkpoint object:\n([^\n]+)/)[1])
}
test('FSR runtime safe point is generated from the canonical helper', () => {
  const canonical = fs.readFileSync(path.join(root,'_meta/scaffolding/runtime-safe-point.js'),'utf8')
  const helper = canonical.slice(canonical.indexOf('async function __workflowRuntimeSafePoint(ctx) {')).trim()
  const begin = '// --- BEGIN inlined runtime-safe-point scaffolding (from _meta/scaffolding/runtime-safe-point.js) ---'
  const end = '// --- END inlined runtime-safe-point scaffolding ---'
  assert.equal(source.split(begin+'\n')[1].split('\n'+end)[0],helper)
})
for (const failure of [false,true]) test(`FSR cooperatively stops after candidate feedback and reinforcement, failed=${failure}`,async()=>{
  const {result,calls}=await runWorkflow(source,args,fixture(true,failure),null)
  assert.equal(result.candidates_completed,1)
  assert.equal(result.iterations_completed,0)
  assert.equal(result.iterations,2)
  assert.equal(result.termination_reason,'deadline')
  assert.equal(result.report,null)
  assert.ok(!calls.some(c=>c.label==='select-features-0-1'||c.label==='final-report'))
  assert.ok(calls.findIndex(c=>c.label==='reinforce-0-0')<calls.findIndex(c=>c.label==='checkpoint-0-0'))
  const saved=checkpoint(calls,'checkpoint-0-0')
  assert.deepEqual(saved.progress,{unit:'candidate',completed:1,total:4})
  assert.deepEqual(saved.replay_binding,{mode:'native_same_activation_only',arguments:args})
  assert.equal(saved.feature_scores.tile.reward,failure?-.3:.3)
  assert.equal(saved.candidate_history[0].task_result_path,'/fixture/fsr/cudallm_iter_0_sample_0.json')
  assert.equal(saved.candidate_history[0].source_binding.source_sha256,'a'.repeat(64))
  assert.equal(saved.correct,!failure)
  assert.equal(saved.best_candidate===null,failure)
  assert.equal(saved.candidate_history[0].measurement.outcome_state,failure?'candidate_failure':'passed')
  assert.ok(!JSON.stringify(saved).includes('DO_NOT_INLINE'))
})
test('FSR unchanged declared search budget completes all candidates and checkpoint includes reduced feature history',async()=>{
  const {result,calls}=await runWorkflow(source,args,fixture(false,false),null)
  assert.equal(result.candidates_completed,4)
  assert.equal(result.iterations_completed,2)
  assert.equal(result.feature_scores.tile.attempts,4)
  assert.equal(calls.filter(c=>c.label.startsWith('task-eval-')).length,4)
  assert.equal(calls.filter(c=>c.label.startsWith('reinforce-')).length,4)
  assert.equal(calls.filter(c=>c.label==='final-report').length,1)
  const saved=checkpoint(calls,'checkpoint-1-1')
  assert.equal(saved.termination_reason,'budget_exhausted')
  assert.equal(saved.best_candidate.source_path,'/fixture/fsr/cudallm_iter_0_sample_0.json.artifact/candidate.py')
  assert.equal(saved.candidate_history.length,4)
})

test('FSR failed candidate checkpoint preserves the independently measured accepted parent',async()=>{
  const agents=fixture(true,true)
  const p='/fixture/fsr/accepted_parent.task.json'
  agents['accepted-parent-task']={test_result_path:p,test_result_json:{contract_version:'kersor-task-result-v1',
    test_result_path:p,candidate_path:p+'.artifact/candidate.py',source_binding:{verified:true,source_sha256:'b'.repeat(64)},
    compiled:true,correct:true,full_workload_set:true,measurement_valid:true,n_pass:17,n_total:17,
    candidate_latency_aggregate_ms:.02,speedup_vs_reference:.75}}
  const {result,calls}=await runWorkflow(source,{...args,kernel_path:'/fixture/accepted.py'},agents,null)
  const saved=checkpoint(calls,'checkpoint-0-0')
  assert.equal(result.generated_kernel_path,p+'.artifact/candidate.py')
  assert.equal(result.best_candidate_id,'accepted-parent')
  assert.equal(saved.best_candidate.task_result_path,p)
  assert.equal(saved.best_candidate.source_binding.source_sha256,'b'.repeat(64))
  assert.equal(saved.accepted_parent.source_path,p+'.artifact/candidate.py')
  assert.equal(saved.metric.value,.75)
  assert.equal(saved.candidate_history[0].measurement.correct,false)
})

test('FSR legacy checkpoint uses file records without duplicating source or inventing native bindings',async()=>{
  const legacyArgs=JSON.parse(fs.readFileSync(path.join(root,'_meta/tools/fixtures/cudallm-args.json'),'utf8'))
  const agents=JSON.parse(fs.readFileSync(path.join(root,'_meta/tools/fixtures/cudallm-agent-returns.json'),'utf8'))
  agents['generate-kernel-0-0'].candidate_code='LEGACY_CODE_MUST_STAY_IN_FILE'.repeat(1000)
  agents['checkpoint-0-0']={termination_requested:true,termination_reason:'supervisor_request',checkpoint_path:'/fixture/checkpoint.json'}
  const {calls}=await runWorkflow(source,{...legacyArgs,termination_file:'/fixture/STOP'},agents)
  const call=calls.find(c=>c.label==='checkpoint-0-0')
  assert.ok(!call.prompt.includes('LEGACY_CODE_MUST_STAY_IN_FILE'))
  const saved=checkpoint(calls,'checkpoint-0-0')
  assert.equal(saved.candidate_history[0].source_path,legacyArgs.exp_dir+'/cudallm_iter_0_sample_0.cu')
  assert.equal(saved.best_candidate.source_binding,null)
  assert.equal(saved.best_candidate.task_result_path,null)
  assert.ok(!calls.some(c=>c.label==='final-report'))
})
