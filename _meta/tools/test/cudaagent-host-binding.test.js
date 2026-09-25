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

function seedBaseline(request) {
  assert.equal(request.baselineSolutionPath, '/tmp/seed/seed.solution.json')
  return {
    compiled: true, correct: true, measurement_valid: true, full_workload_set: true,
    output_contract_valid: true, candidate_latency_aggregate_ms: 0.02,
  }
}

function measured(binding, latency = 0.016) {
  return {
    compiled: true, correct: true, measurement_valid: true, full_workload_set: true,
    output_contract_valid: true, speedup: 1.2,
    candidate_latency_aggregate_ms: latency,
    candidate_path: candidatePath, candidate_sha256: candidateSha,
    artifact_binding: binding,
  }
}

test('CUDAAgent Sol returns the exact Host-bound source and matching binding', async () => {
  const {result} = await runWorkflow(source, args, agentReturns, {
    'sol-seed-baseline': seedBaseline,
    'sol-eval-0': request => {
      assert.equal(request.bindingOut, bindingPath)
      assert.equal(request.bindingWorkflow, 'cuda-agent-kernel-optimization')
      assert.equal(request.candidateId, 'attempt-0')
      return measured({verified: true, candidate_sha256: candidateSha,
        binding_path: bindingPath})
    },
  })
  assert.equal(result.artifact_binding_required, true)
  assert.equal(result.artifact_binding_path, bindingPath)
  assert.equal(result.generated_kernel_path, candidatePath)
  assert.equal(result.best_kernel_path, candidatePath)
  assert.equal(result.best_candidate_id, 'attempt-0')
  assert.deepEqual(JSON.parse(JSON.stringify(result.canonical_metric)),
    {name: 'speedup', value: 1.2})
  assert.equal(result.seed_relative_speedup, 1.25)
  assert.equal(result.target_met, true)
})

test('CUDAAgent Sol fails when a correct Host measurement lacks binding', async () => {
  await assert.rejects(() => runWorkflow(source, args, agentReturns, {
    'sol-seed-baseline': seedBaseline,
    'sol-eval-0': measured(null),
  }), /Host artifact binding missing/)
})

test('CUDAAgent Sol retains the supplied seed when the Host candidate regresses', async () => {
  const {result} = await runWorkflow(source, args, agentReturns, {
    'sol-seed-baseline': seedBaseline,
    'sol-eval-0': measured({verified: true, candidate_sha256: candidateSha,
      binding_path: bindingPath}, 0.022),
  })
  assert.equal(result.seed_relative_speedup, 1)
  assert.equal(result.target_met, false)
  assert.equal(result.generated_kernel_path, '')
  assert.equal(result.artifact_binding_required, false)
  assert.equal(result.best_kernel_path, args.kernel_path)
})

test('CUDAAgent Sol refuses an unmeasured supplied seed', async () => {
  await assert.rejects(() => runWorkflow(source, args, agentReturns, {
    'sol-seed-baseline': {compiled: false, correct: false},
  }), /Host could not establish a complete measured Sol seed baseline/)
})

test('CUDAAgent Sol explores without a numeric target', async () => {
  const {result} = await runWorkflow(source, {...args, target_speedup: 'none'}, agentReturns, {
    'sol-seed-baseline': seedBaseline,
    'sol-eval-0': measured({verified: true, candidate_sha256: candidateSha,
      binding_path: bindingPath}),
  })
  assert.equal(result.seed_relative_speedup, 1.25)
  assert.equal(result.target_met, false)
})

test('CUDAAgent CuTe SOL optimizes a Python CuTe seed through the Host evaluator', async () => {
  const cuteArgs = {...args, language: 'cute-dsl',
    kernel_path: '/tmp/cudaagent-bound/kernel.py'}
  const cutePath = '/tmp/cudaagent-bound/kernels/sol_t0.py'
  const cuteAgents = {'impl-0': {
    kernel_code: 'from cutlass import cute\n@cute.jit\ndef kernel(): pass\ndef run(*args): pass',
    binding_code: '', model_new_code: '', implementation_notes: 'CuTe candidate',
  }}
  const {result} = await runWorkflow(source, cuteArgs, cuteAgents, {
    'sol-seed-baseline': request => {
      assert.equal(request.candidatePath, '/tmp/cudaagent-bound/host_seed.py')
      assert.equal(request.candidateLanguage, 'cute-dsl')
      return seedBaseline(request)
    },
    'sol-eval-0': request => {
      assert.equal(request.candidatePath, cutePath)
      assert.equal(request.candidateLanguage, 'cute-dsl')
      assert.match(request.candidateSource, /from cutlass import cute/)
      return {...measured({verified: true, candidate_sha256: candidateSha,
        binding_path: bindingPath}), candidate_path: cutePath}
    },
  })
  assert.equal(result.generated_kernel_path, cutePath)
  assert.equal(result.seed_relative_speedup, 1.25)
  assert.equal(result.artifact_binding_required, true)
})
