'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const runWorkflow = require('../lib/run-workflow.js')

const source = fs.readFileSync(path.resolve(__dirname, '../../..', 'Generalist/generalist-kernel-optimization.js'), 'utf8')
const args = {
  kernel_path: '/tmp/generalist-sol/seed.cu',
  language: 'cuda', integration_pattern: 'sol_execbench_solution',
  can_standalone: 'yes', exp_dir: '/tmp/generalist-sol',
  iterations: 1, breadth: 1, topk: 1, target_speedup: 1.01,
  sol_cli: '/bin/sol', sol_task_dir: '/tmp/task',
  sol_bench_config: '/tmp/bench.json', sol_seed_dir: '/tmp/seed',
  sol_substrate_dir: '/tmp/substrate', sol_definition_path: '/tmp/task/definition.json',
}
const agentReturns = {
  'profile-1': {compiled: true, correct: true, speedup: null, metrics: {}},
  'diagnose-1': {bottleneck_class: 'unknown'},
  'retrieve-1': {techniques: [], dead_ends: []},
  'gate-1': {allowed_methods: ['conservative_tiling']},
  'plan-1-1': {method: 'conservative_tiling', plan: 'modify the supplied seed'},
  'impl-1-1': {kernel_code: '__global__ void run_kernel() {}'},
  'learn-1-conservative_tiling': {updated: false},
  'refute-1': {refuted: false},
  'verify-insight-1': {confidence: 'unknown'},
  'final-report': {},
}
function measured(latency, candidatePath) {
  return {compiled: true, correct: true, full_workload_set: true,
    output_contract_valid: true, measurement_valid: true,
    candidate_latency_aggregate_ms: latency, candidate_path: candidatePath,
    speedup: 4, n_pass: 14, n_total: 14}
}
function run(candidateLatency) {
  return runWorkflow(source, args, agentReturns, {
    'sol-seed-baseline': request => {
      assert.equal(request.baselineSolutionPath, '/tmp/seed/seed.solution.json')
      return measured(0.02, '/tmp/generalist-sol/host_seed.cu')
    },
    'sol-eval-1-1': request => measured(candidateLatency, request.candidatePath),
  })
}
test('Generalist Sol promotes only Host-measured seed-relative improvement', async () => {
  const {calls, result} = await run(0.016)
  assert.ok(!calls.some(call => call.label === 'anticheat-1-1'))
  assert.equal(result.overall_speedup, 1.25)
  assert.equal(result.generated_kernel_path, '/tmp/generalist-sol/run-1/cand-1/kernel.cu')
  assert.equal(result.best_kernel_code, result.generated_kernel_path)
  assert.equal(result.attempts[0].valid, true)
  assert.equal(result.attempts[0].measured_speedup, 1.25)
  const implementation = calls.find(call => call.label === 'impl-1-1')
  assert.deepEqual(implementation.schema.required, ['kernel_code'])
  assert.doesNotMatch(implementation.prompt, /run IN THIS EXACT ORDER|Append exactly one line/)
})
test('Generalist Sol retains the seed when a Host-correct candidate regresses', async () => {
  const {result} = await run(0.025)
  assert.equal(result.overall_speedup, 1)
  assert.equal(result.best_kernel_code, args.kernel_path)
  assert.equal(result.generated_kernel_path, '')
})
test('Generalist Sol refuses a missing measured seed baseline', async () => {
  await assert.rejects(() => runWorkflow(source, args, agentReturns, {
    'sol-seed-baseline': {compiled: false, correct: false, failure_code: 'compile_error'},
  }), /Host could not establish a complete measured Sol seed baseline/)
})
