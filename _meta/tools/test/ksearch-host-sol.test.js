'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const runWorkflow = require('../lib/run-workflow.js')

const source = fs.readFileSync(path.resolve(__dirname, '../../..', 'KSearch/ksearch-kernel-optimization.js'), 'utf8')
const args = {
  problem_path: '/tmp/task', op_description: 'gemm', language: 'cuda', target_gpu: 'B300',
  iterations: 1, attempts_per_cycle: 1, seed_candidates: 1, stagnation_window: 3,
  kernel_path: '/tmp/seed.cu', exp_dir: '/tmp/ksearch-sol',
  integration_pattern: 'sol_execbench_solution', sol_cli: '/bin/sol',
  sol_task_dir: '/tmp/task', sol_bench_config: '/tmp/b.json',
  sol_seed_dir: '/tmp/seed', sol_substrate_dir: '/tmp/sub',
  sol_definition_path: '/tmp/task/definition.json',
}
const agents = {
  'read-spec': {spec_text: 'gemm', op_type: 'gemm', baseline_code: 'seed',
    design_dimensions: ['tile'], key_challenges: []},
  'eval-baseline': {baseline_metric: 1, baseline_latency_ms: 0.02, eval_passed: true},
  'init-tree': {decision_tree: {root: {}, n1: {}}, node_count: 2, open_actions: 1},
  'propose-0': {updated_tree: {root: {}, n1: {}}, open_frontier_count: 1, nodes_added: 0},
  'select-0': {selected_node_id: 'n1', action_title: 'tile', action_description: 'tile',
    action_score: 0.7, action_difficulty: 2, parent_solution_code: 'seed',
    parent_metric: 1, parent_is_root: true, context_for_generation: {}},
  'gen-0-0': {code: 'candidate source', implementation_notes: 'tile', design_choices: []},
  'refine-0': {updated_tree: {root: {}, n1: {status: 'solved'}}, new_actions_added: 1,
    score_updates: []},
}
const seed = {compiled: true, correct: true, full_workload_set: true,
  output_contract_valid: true, measurement_valid: true,
  candidate_latency_aggregate_ms: 0.02}
function measured(latency, binding = true) {
  return {compiled: true, correct: true, full_workload_set: true,
    output_contract_valid: true, measurement_valid: true,
    candidate_latency_aggregate_ms: latency, speedup: 1.2,
    n_pass: 25, n_total: 25,
    candidate_path: '/tmp/ksearch-sol/ksearch_c0_a0.cu',
    candidate_sha256: 'a'.repeat(64),
    artifact_binding: binding ? {verified: true,
      binding_path: '/tmp/ksearch-sol/bindings/ksearch_c0_a0.json'} : null}
}
function evals(candidate) {
  return {'sol-seed-baseline': request => {
    assert.equal(request.baselineSolutionPath, '/tmp/seed/seed.solution.json')
    return seed
  }, 'sol-eval-0-0': request => {
    assert.equal(request.bindingWorkflow, 'ksearch-kernel-optimization')
    assert.equal(request.candidateId, 'cycle-0-a0')
    return candidate
  }}
}

test('KSearch Sol returns the bound Host candidate that beats its incoming seed', async () => {
  const {result} = await runWorkflow(source, args, agents, evals(measured(0.015)))
  assert.equal(result.generated_kernel_path, '/tmp/ksearch-sol/ksearch_c0_a0.cu')
  assert.equal(result.best_candidate_id, 'cycle-0-a0')
  assert.equal(result.artifact_binding_required, true)
  assert.equal(result.artifact_binding_path, '/tmp/ksearch-sol/bindings/ksearch_c0_a0.json')
  assert.ok(result.best_metric > 1.3)
})

test('KSearch Sol retains the seed when a correct candidate is slower', async () => {
  const {result} = await runWorkflow(source, args, agents, evals(measured(0.022)))
  assert.equal(result.generated_kernel_path, '')
  assert.equal(result.artifact_binding_required, false)
})

test('KSearch Sol refuses a correct candidate without Host source binding', async () => {
  await assert.rejects(() => runWorkflow(source, args, agents, evals(measured(0.015, false))),
    /artifact binding missing/)
})

test('KSearch CuTe DSL sends Python source to the Host and retains the CUDA incumbent as seed', async () => {
  const observed = evals(measured(0.015))
  const evaluateCandidate = observed['sol-eval-0-0']
  observed['sol-eval-0-0'] = request => {
    assert.match(request.candidatePath, /\.py$/)
    return evaluateCandidate(request)
  }
  const {result} = await runWorkflow(source, {...args, language: 'cute-dsl'}, agents, observed)
  assert.equal(result.artifact_binding_required, true)
  assert.equal(result.best_candidate_id, 'cycle-0-a0')
})
