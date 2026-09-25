'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const runWorkflow = require('../lib/run-workflow.js')

const source = fs.readFileSync(path.resolve(__dirname, '../../..', 'CUDALLM/cudallm-fsr-kernel-generation.js'), 'utf8')
const args = {
  problem_definition: 'gemm', problem_path: '/tmp/task',
  reference_code_path: '/tmp/seed/kernel.cu', exp_dir: '/tmp/cudallm-sol',
  iterations: 1, samples_per_feature_set: 1, feature_budget: 1,
  integration_pattern: 'sol_execbench_solution',
  sol_cli: '/bin/sol', sol_task_dir: '/tmp/task', sol_bench_config: '/tmp/b.json',
  sol_seed_dir: '/tmp/seed', sol_substrate_dir: '/tmp/sub',
  sol_definition_path: '/tmp/task/definition.json',
}
const agents = {
  'setup-task': {problem_definition: 'gemm', reference_code: 'seed code',
    operation_type: 'gemm', constraints: []},
  'feature-catalog': {features: [{id: 'tile', name: 'tile'}], baseline_feature_ids: []},
  'generate-tests': {test_cases: [], tolerance_policy: 'strict'},
  'select-features-0-0': {selected_feature_ids: ['tile'], rationale: 'tile'},
  'generate-kernel-0-0': {candidate_code: 'candidate code', implemented_feature_ids: ['tile']},
  // The agent is advisory; its invented score must never decide promotion.
  'evaluate-0-0': {compiled: false, correct: false, speedup: 999, latency_ms: 0.0001},
  'reinforce-0-0': {updated_scores: []},
}
const seed = {compiled: true, correct: true, full_workload_set: true,
  output_contract_valid: true, measurement_valid: true,
  candidate_latency_aggregate_ms: 0.02}
function measured(latency, binding = true) {
  return {compiled: true, correct: true, full_workload_set: true,
    output_contract_valid: true, measurement_valid: true,
    candidate_latency_aggregate_ms: latency, speedup: 1.2,
    n_pass: 25, n_total: 25,
    candidate_path: '/tmp/cudallm-sol/cudallm_iter_0_sample_0.cu',
    candidate_sha256: 'a'.repeat(64),
    artifact_binding: binding ? {verified: true,
      binding_path: '/tmp/cudallm-sol/bindings/cudallm_0_0.json'} : null}
}
function evals(candidate) {
  return {'sol-seed-baseline': request => {
    assert.equal(request.baselineSolutionPath, '/tmp/seed/seed.solution.json')
    return seed
  }, 'sol-eval-0-0': request => {
    assert.equal(request.bindingWorkflow, 'cudallm-fsr-kernel-generation')
    assert.equal(request.candidateId, 'iter_0_sample_0')
    return candidate
  }}
}

test('CUDALLM-FSR Sol inherits and beats the measured seed with a bound source', async () => {
  const {result} = await runWorkflow(source, args, agents, evals(measured(0.015)))
  assert.equal(result.input_mode, 'optimize_existing')
  assert.equal(result.reference_code_path, args.reference_code_path)
  assert.equal(result.generated_kernel_path, '/tmp/cudallm-sol/cudallm_iter_0_sample_0.cu')
  assert.equal(result.artifact_binding_required, true)
  assert.equal(result.best_kernel_code, 'candidate code')
  assert.ok(result.best_speedup > 1.3)
})

test('CUDALLM-FSR Sol ignores an agent score and retains a faster seed', async () => {
  const {result} = await runWorkflow(source, args, agents, evals(measured(0.022)))
  assert.equal(result.generated_kernel_path, '')
  assert.equal(result.artifact_binding_required, false)
  assert.equal(result.best_kernel_code, '')
  assert.equal(result.best_speedup, 0)
})

test('CUDALLM-FSR Sol refuses a correct unbound Host candidate', async () => {
  await assert.rejects(() => runWorkflow(source, args, agents, evals(measured(0.015, false))),
    /artifact binding missing/)
})

test('CUDALLM-FSR CuTe SOL sends Python source with explicit Host language', async () => {
  const cuteArgs = {...args, language: 'cute-dsl', reference_code_path: '/tmp/seed/kernel.py'}
  const cuteAgents = {...agents,
    'generate-kernel-0-0': {candidate_code: 'from cutlass import cute\n@cute.jit\ndef kernel(): pass\ndef run(*args): pass',
      implemented_feature_ids: ['tile']}}
  const candidate = {...measured(0.015),
    candidate_path: '/tmp/cudallm-sol/cudallm_iter_0_sample_0.py'}
  const evaluations = {
    'sol-seed-baseline': request => {
      assert.equal(request.candidatePath, '/tmp/cudallm-sol/host_seed.py')
      assert.equal(request.candidateLanguage, 'cute-dsl')
      return seed
    },
    'sol-eval-0-0': request => {
      assert.equal(request.candidatePath, candidate.candidate_path)
      assert.equal(request.candidateLanguage, 'cute-dsl')
      assert.match(request.candidateSource, /from cutlass import cute/)
      return candidate
    },
  }
  const {result} = await runWorkflow(source, cuteArgs, cuteAgents, evaluations)
  assert.equal(result.generated_kernel_path, candidate.candidate_path)
  assert.equal(result.artifact_binding_required, true)
})
