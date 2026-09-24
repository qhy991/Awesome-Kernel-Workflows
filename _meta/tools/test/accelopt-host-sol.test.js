'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const runWorkflow = require('../lib/run-workflow.js')

const source = fs.readFileSync(path.resolve(__dirname, '../../..', 'AccelOpt/accelopt-kernel-optimization.js'), 'utf8')
const args = {
  kernel_path: '/tmp/accelopt-sol/seed.cu', problem_path: '/tmp/task',
  exp_dir: '/tmp/accelopt-sol', iterations: 1, breadth: 1, samples_per_plan: 1,
  integration_pattern: 'sol_execbench_solution',
  sol_cli: '/bin/sol', sol_task_dir: '/tmp/task', sol_bench_config: '/tmp/b.json',
  sol_seed_dir: '/tmp/seed', sol_substrate_dir: '/tmp/sub',
  sol_definition_path: '/tmp/task/definition.json',
}
const agents = {
  'read-baseline': {kernel_code: 'seed source', op_type: 'gemm', key_functions: [], current_approach: 'seed'},
  'ncu-baseline': {latency_ms: 1, bottleneck_diagnosis: 'compute', profile_summary: 'no counters'},
  'plan-0-0': {title: 'tile', ncu_evidence: 'none', plan: 'tile', expected_impact: 'faster'},
  'impl-0-tile-v0': {code: 'candidate source'},
}
const seed = {
  compiled: true, correct: true, full_workload_set: true,
  output_contract_valid: true, measurement_valid: true,
  candidate_latency_aggregate_ms: 0.02,
}
function measured(latency, binding = true) {
  return {
    compiled: true, correct: true, full_workload_set: true,
    output_contract_valid: true, measurement_valid: true,
    candidate_latency_aggregate_ms: latency, speedup: 1.2,
    candidate_path: '/tmp/accelopt-sol/accelopt_plan_0_sample_0.cu',
    candidate_sha256: 'a'.repeat(64),
    artifact_binding: binding ? {verified: true, binding_path: '/tmp/accelopt-sol/bindings/plan_0_sample_0.json'} : null,
  }
}
function evals(candidate) {
  return {'sol-seed-baseline': request => {
    assert.equal(request.baselineSolutionPath, '/tmp/seed/seed.solution.json')
    return seed
  }, 'sol-eval-plan_0_sample_0': request => {
    assert.equal(request.bindingWorkflow, 'accelopt-kernel-optimization')
    assert.equal(request.candidateId, 'plan_0_sample_0')
    return candidate
  }}
}

test('AccelOpt Sol returns the exact Host candidate only when it beats the supplied seed', async () => {
  const {result} = await runWorkflow(source, args, agents, evals(measured(0.015)))
  assert.equal(result.baseline_latency_ms, 0.02)
  assert.equal(result.generated_kernel_path, '/tmp/accelopt-sol/accelopt_plan_0_sample_0.cu')
  assert.equal(result.artifact_binding_required, true)
  assert.equal(result.best_kernel_code, 'candidate source')
  assert.ok(result.overall_speedup > 1.3)
})

test('AccelOpt Sol retains the supplied seed on a slower Host candidate', async () => {
  const {result} = await runWorkflow(source, args, agents, evals(measured(0.022)))
  assert.equal(result.generated_kernel_path, '')
  assert.equal(result.artifact_binding_required, false)
  assert.equal(result.best_kernel_code, '')
  assert.equal(result.overall_speedup, 1)
})

test('AccelOpt Sol refuses a correct measurement without source binding', async () => {
  await assert.rejects(() => runWorkflow(source, args, agents, evals(measured(0.015, false))),
    /artifact binding missing/)
})
