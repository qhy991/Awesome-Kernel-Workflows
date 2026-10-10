'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const run = require('../lib/run-workflow.js')
const root = path.resolve(__dirname, '../../..')
const accel = fs.readFileSync(path.join(root, 'AccelOpt/accelopt-kernel-optimization.js'), 'utf8')
const foundry = fs.readFileSync(path.join(root, 'KernelFoundry/kernelfoundry-kernel-optimization.js'), 'utf8')
const args = {problem_path:'/fixture/task.md', exp_dir:'/fixture/exp', language:'cute-dsl',
  task_result_command:'trusted --candidate {kernel_path} --result {result_path}', task_workload_count:17,
  iterations:1, breadth:1, samples_per_plan:1, seed_candidates:3}
function measured(p, latency, extra = {}) {
  return {contract_version:'kersor-task-result-v1', test_result_path:p,
    compiled:true, correct:true, full_workload_set:true, measurement_valid:true,
    n_pass:17, n_total:17, candidate_latency_aggregate_ms:latency,
    speedup_vs_reference:.06 / latency, candidate_path:p + '.artifact/candidate.py',
    source_binding:{verified:true, source_sha256:'a'.repeat(64)}, ...extra}
}
function accelAgents(profile = {}) {
  return {'generate-initial-kernel':{initial_candidates:[2,0,1].map(i => ({variant_path:`/fixture/exp/generated/initial_${i}.py`}))},
    'read-baseline':{kernel_code:'display', op_type:'gemm', key_functions:['run']},
    'ncu-baseline':{ncu_available:false, profile_summary:'No admitted profiling command', bottleneck_diagnosis:'unknown', ...profile},
    'plan-0-0':{title:'p', plan:'try tiling', expected_impact:'hypothesis'},
    'impl-0-p-v0':{code:'display candidate', variant_path:'/fixture/exp/iter_0_plan_0_sample_0.py'}}
}
function accelEvaluations(extra = {}, seen = []) {
  const returns = {}
  for (let i=0; i<3; i++) returns[`initial-${i}`] = request => {
    seen.push(request.label)
    assert.ok(request.argv.join(' ').includes(`/fixture/exp/generated/initial_${i}.py`))
    return {stage:'complete', exit_code:0,
      stdout_json:measured(`/fixture/exp/generated/initial_${i}.json`, [.03,.02,.04][i], extra)}
  }
  returns['task-eval-iter_0_plan_0_sample_0'] = request => {
    seen.push(request.label)
    return {stage:'complete', exit_code:0, stdout_json:measured('/fixture/exp/iter_0_plan_0_sample_0.task.json', .025)}
  }
  return returns
}
test('AccelOpt native initialization produces sources then measures each fixed seed once in index order', async () => {
  const seen = []
  const {calls, result} = await run(accel, args, accelAgents(), accelEvaluations({}, seen))
  const generation = calls.find(c => c.label === 'generate-initial-kernel')
  assert.deepEqual(generation.schema.required, ['initial_candidates'])
  assert.deepEqual(generation.schema.properties.initial_candidates.items.required, ['variant_path'])
  assert.ok(!generation.prompt.includes('run the trusted command once, serially'))
  assert.ok(generation.prompt.includes('diagnostic'))
  assert.deepEqual(seen, ['initial-0','initial-1','initial-2','task-eval-iter_0_plan_0_sample_0'])
  assert.equal(result.baseline_latency_ms, .02)
  assert.equal(result.generated_kernel_path, '/fixture/exp/generated/initial_1.json.artifact/candidate.py')
  assert.equal(result.overall_speedup, 1)
  assert.deepEqual(JSON.parse(JSON.stringify(result.initial_candidates.map(c => c.test_result_path))), [0,1,2].map(i => `/fixture/exp/generated/initial_${i}.json`))
  assert.ok(result.initial_candidates.every(c => c.source_binding.verified && c.host_candidate_path === c.test_result_path + '.artifact/candidate.py'))
  assert.ok(calls.find(c => c.label === 'impl-0-p-v0').schema.required.includes('variant_path'))
})
test('AccelOpt rejects incomplete seed counts before submitting any measurement', async () => {
  for (const candidates of [[], [{variant_path:'/foreign.py'}]]) {
    const agents = accelAgents(), seen = []
    agents['generate-initial-kernel'] = {initial_candidates:candidates}
    await assert.rejects(run(accel, args, agents, accelEvaluations({}, seen)), /initial candidate|seed/i)
    assert.deepEqual(seen, [])
  }
})
test('AccelOpt does not replay native seed generation after a failed producer response', async () => {
  const agents = accelAgents(), seen = []
  let attempts = 0
  Object.defineProperty(agents, 'generate-initial-kernel', {get() { attempts++; return null }})
  await assert.rejects(run(accel, args, agents, accelEvaluations({}, seen)))
  assert.equal(attempts, 1)
  assert.deepEqual(seen, [])
})
test('AccelOpt unknown seed evidence stops before another candidate measurement', async () => {
  const seen = []
  await assert.rejects(run(accel, args, accelAgents(), accelEvaluations({outcome_state:'unknown'}, seen)),
    error => error.code === 'TASK_RESULT_HOLD' && error.retryable === false)
  assert.deepEqual(seen, ['initial-0'])
})
test('AccelOpt rejected seed retains its diagnostic reference while later seeds are measured', async () => {
  const seen = [], evaluations = accelEvaluations({}, seen)
  evaluations['initial-0'] = request => {
    seen.push(request.label)
    return {stage:'complete',exit_code:1,stdout_json:measured('/fixture/exp/generated/initial_0.json',.03,
      {compiled:false,correct:false,measurement_valid:false,n_pass:0,
       outcome_state:'candidate_failure',failure_origin:'candidate',diagnostics:[{error:'compile rejected'}]})}
  }
  const {result} = await run(accel,args,accelAgents(),evaluations)
  assert.deepEqual(seen,['initial-0','initial-1','initial-2','task-eval-iter_0_plan_0_sample_0'])
  assert.equal(result.initial_candidates[0].outcome_state,'candidate_failure')
  assert.equal(result.initial_candidates[0].diagnostics[0].error,'compile rejected')
  assert.equal(result.generated_kernel_path,'/fixture/exp/generated/initial_1.json.artifact/candidate.py')
})
test('AccelOpt unavailable profiling remains an explicit hypothesis and uses frozen GPU scope', async () => {
  const {calls} = await run(accel, args, accelAgents(), accelEvaluations())
  const profile = calls.find(c => c.label === 'ncu-baseline').prompt
  const plan = calls.find(c => c.label === 'plan-0-0')
  assert.ok(!profile.includes('allowedGpuIds=[6,7]'))
  assert.match(profile, /frozen.*site|site.*allowlist/i)
  assert.ok(!plan.prompt.includes('REAL MEASURED DATA'))
  assert.ok(!plan.prompt.includes('REAL Nsight Compute'))
  assert.match(plan.prompt, /unavailable|not verified/i)
  assert.match(plan.prompt, /hypothes/i)
  assert.ok(!plan.schema.required.includes('ncu_evidence'))
})
test('AccelOpt available profiling gives the planner the complete measured report path', async () => {
  const {calls} = await run(accel, args,
    accelAgents({ncu_available:true, report_path:'/fixture/profile/full.ncu-rep', profile_summary:'memory-bound'}), accelEvaluations())
  assert.ok(calls.find(c => c.label === 'plan-0-0').prompt.includes('/fixture/profile/full.ncu-rep'))
})
test('KernelFoundry native setup accepts no invented baseline and requires candidate file identity', async () => {
  const agents = {setup:{operator_code:'spec', operator_type:'gemm'},
    'vary-0':{kernel_code:'display', variant_path:'/fixture/exp/gen_0.py', strategy_description:'tile',d_mem:1,d_algo:1,d_sync:1}}
  const evaluations = {'task-eval-0':{stage:'complete',exit_code:0,stdout_json:measured('/fixture/exp/gen_0_result.json',.02)}}
  const {calls,result} = await run(foundry, {...args,generations:1,meta_prompt_interval:99}, agents, evaluations)
  const setup = calls.find(c => c.label === 'setup')
  assert.deepEqual(setup.schema.required, ['operator_code'])
  assert.ok(!setup.prompt.includes('(estimate)'))
  assert.ok(!setup.prompt.includes('Establish PyTorch baseline performance'))
  assert.equal(result.baseline_time_ms, null)
  assert.ok(calls.find(c => c.label === 'vary-0').schema.required.includes('variant_path'))
})
test('KernelFoundry legacy setup retains its baseline contract', async () => {
  const setupOnly = foundry.slice(0, foundry.indexOf('const operatorCode = setupResult.operator_code')) + '\nreturn setupResult'
  const {calls} = await run(setupOnly, {exp_dir:'/fixture/exp',op_description:'fixture',language:'cuda'},
    {setup:{operator_code:'spec',baseline_time_ms:.1}}, null)
  const setup = calls.find(c => c.label === 'setup')
  assert.deepEqual(setup.schema.required, ['operator_code','baseline_time_ms'])
  assert.ok(setup.prompt.includes('Establish PyTorch baseline performance: (estimate)'))
})
