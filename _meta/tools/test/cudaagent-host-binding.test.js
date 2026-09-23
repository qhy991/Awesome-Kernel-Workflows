'use strict'

const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const runWorkflow = require('../lib/run-workflow.js')

const source = fs.readFileSync(path.resolve(__dirname, '../../..', 'CUDAAgent/cuda-agent-kernel-optimization.js'), 'utf8')
const candidatePath = '/tmp/cudaagent-bound/kernels/sol_t0.cu'
const candidateSha = 'a'.repeat(64)
const bindingPath = '/tmp/cudaagent-bound/bindings/sol_t0.json'
const args = {
  language: 'cuda', integration_pattern: 'sol_execbench_solution',
  problem_definition: 'rms_norm(x, weight)', target_gpu: 'RTX4090',
  kernel_path: '/tmp/cudaagent-bound/seed.cu',
  exp_dir: '/tmp/cudaagent-bound', max_turns: 1, turn_timeout_min: 20,
  sol_cli: '/bin/sol', sol_task_dir: '/tmp/task', sol_bench_config: '/tmp/bench.json',
  sol_seed_dir: '/tmp/seed', sol_substrate_dir: '/tmp/substrate',
  sol_definition_path: '/tmp/task/definition.json',
}
const agentReturns = {
  'impl-0': {
    kernel_code: '__global__ void run_kernel() {}',
    binding_code: 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", []{}); }',
    model_new_code: '', implementation_notes: 'bounded candidate',
  },
}

function measured(binding) {
  return {
    compiled: true, correct: true, measurement_valid: true, full_workload_set: true,
    output_contract_valid: true, speedup: 1.2,
    candidate_path: candidatePath, candidate_sha256: candidateSha,
    artifact_binding: binding,
  }
}

test('CUDAAgent Sol returns the exact Host-bound source and matching binding', async () => {
  const {result} = await runWorkflow(source, args, agentReturns, {
    'sol-eval-0': measured({verified: true, candidate_sha256: candidateSha,
      binding_path: bindingPath}),
  })
  assert.equal(result.artifact_binding_required, true)
  assert.equal(result.artifact_binding_path, bindingPath)
  assert.equal(result.generated_kernel_path, candidatePath)
  assert.equal(result.best_kernel_path, candidatePath)
  assert.equal(result.best_candidate_id, 'attempt-0')
  assert.deepEqual(JSON.parse(JSON.stringify(result.canonical_metric)),
    {name: 'speedup', value: 1.2})
})

test('CUDAAgent Sol refuses to promote a measured candidate with no Host binding', async () => {
  const {result} = await runWorkflow(source, args, agentReturns, {
    'sol-eval-0': measured(null),
  })
  assert.equal(result.artifact_binding_required, true)
  assert.equal(result.artifact_binding_path, '')
  assert.equal(result.best_candidate_id, '')
  assert.equal(result.canonical_metric.value, 0)
})
