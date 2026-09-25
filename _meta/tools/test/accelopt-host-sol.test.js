'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const runWorkflow = require('../lib/run-workflow.js')

const source = fs.readFileSync(path.resolve(__dirname, '../../..', 'AccelOpt/accelopt-kernel-optimization.js'), 'utf8')
const manifest = fs.readFileSync(path.resolve(__dirname, '../../..', 'AccelOpt/manifest.yaml'), 'utf8')
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

test('AccelOpt CuTe SOL uses Host latency and explicit Python CuTe candidate transport', async () => {
  assert.match(manifest, /fidelity_boundary: idea_preserving_cute_host_latency_adaptation/)
  assert.match(manifest, /requires_ncu: false/)
  assert.match(manifest, /  - sol_execbench_solution/)
  const cuteArgs = {...args, language: 'cute-dsl', kernel_path: '/tmp/accelopt-sol/kernel.py'}
  const cuteAgents = {...agents,
    'read-baseline': {kernel_code: 'from cutlass import cute\ndef run(*args): pass',
      op_type: 'gemm', key_functions: ['run'], current_approach: 'CuTe seed'},
    'plan-0-0': {title: 'tile', source_evidence: 'source structure only',
      plan: 'change CuTe tile', expected_impact: 'lower Host latency'},
    'impl-0-tile-v0': {code: 'from cutlass import cute\n@cute.jit\ndef kernel(): pass\ndef run(*args): pass'},
  }
  const candidate = {...measured(0.015),
    candidate_path: '/tmp/accelopt-sol/accelopt_plan_0_sample_0.py'}
  const evaluations = {
    'sol-seed-baseline': request => {
      assert.equal(request.candidatePath, '/tmp/accelopt-sol/host_seed.py')
      assert.equal(request.candidateLanguage, 'cute-dsl')
      return seed
    },
    'sol-eval-plan_0_sample_0': request => {
      assert.equal(request.candidatePath, candidate.candidate_path)
      assert.equal(request.candidateLanguage, 'cute-dsl')
      assert.match(request.candidateSource, /from cutlass import cute/)
      return candidate
    },
  }
  const {result, calls} = await runWorkflow(source, cuteArgs, cuteAgents, evaluations)
  assert.equal(result.generated_kernel_path, candidate.candidate_path)
  assert.equal(result.artifact_binding_required, true)
  assert.match(result.ncu_baseline_profile, /Host full-workload seed latency/)
  assert.doesNotMatch(result.ncu_baseline_profile, /SM Throughput|Top Stall Reason/)
  const evaluate = calls.find(call => call.label === 'eval-plan_0_sample_0')
  const learn = calls.find(call => call.label?.startsWith('learn-'))
  const report = calls.find(call => call.label === 'final-report')
  assert.match(evaluate.prompt, /CuTe DSL Python module/)
  assert.doesNotMatch(evaluate.prompt, /NCU Evidence|NCU Metric Comparison/)
  assert.match(learn.prompt, /No profiler counters were collected/)
  assert.doesNotMatch(learn.prompt, /NCU Evidence|NCU Metric Comparison/)
  assert.match(report.prompt, /CuTe DSL Host-latency adaptation/)
  assert.doesNotMatch(report.prompt, /NCU-driven optimization journey/)
  assert.ok(result.experience_patterns.every(pattern => !pattern.includes('NCU trigger:')))
  assert.equal(result.evidence_mode, 'host_full_workload_latency')
  assert.equal(result.profiler_counters_available, false)
})

test('AccelOpt CuTe SOL cannot promote a static estimate when Host returns no result', async () => {
  const cuteArgs = {...args, language: 'cute-dsl', kernel_path: '/tmp/accelopt-sol/kernel.py'}
  const agentsWithOptimisticEstimate = {...agents,
    'eval-plan_0_sample_0': {is_correct: true, is_compilable: true,
      estimated_speedup: 100, estimated_latency_ms: 0.0002},
  }
  const {result} = await runWorkflow(source, cuteArgs, agentsWithOptimisticEstimate, {
    'sol-seed-baseline': seed,
    'sol-eval-plan_0_sample_0': null,
  })
  assert.equal(result.generated_kernel_path, '')
  assert.equal(result.best_kernel_code, '')
  assert.equal(result.overall_speedup, 1)
  assert.equal(result.experience_patterns_count, 0)
  assert.equal(result.candidate_beam.map(item => item.plan_title).join(','), 'baseline')
})

test('AccelOpt CuTe SOL refuses a problem-only input instead of generating CUDA', async () => {
  const problemOnly = {...args, kernel_path: '', problem_definition: 'GEMM', language: 'cute-dsl'}
  await assert.rejects(() => runWorkflow(source, problemOnly, agents),
    /requires an inherited CuTe kernel_path/)
})

test('AccelOpt stops at an iteration boundary and keeps the bound Host candidate', async () => {
  const stoppingAgents = {...agents,
    'read-baseline': {kernel_code: 'from cutlass import cute\ndef run(*args): pass',
      op_type: 'gemm', key_functions: ['run'], current_approach: 'CuTe seed'},
    'plan-0-0': {title: 'tile', source_evidence: 'source structure',
      plan: 'change tile', expected_impact: 'lower Host latency'},
    'impl-0-tile-v0': {code: 'from cutlass import cute\n@cute.jit\ndef kernel(): pass\ndef run(*args): pass'},
    'checkpoint-1': {termination_requested: true, termination_reason: 'wall_clock_limit',
      checkpoint_path: '/tmp/accelopt-sol/checkpoint.json'},
  }
  const candidate = {...measured(0.015),
    candidate_path: '/tmp/accelopt-sol/accelopt_plan_0_sample_0.py'}
  const {result, calls} = await runWorkflow(source, {
    ...args, language: 'cute-dsl', kernel_path: '/tmp/accelopt-sol/kernel.py',
    deadline_epoch: 123, termination_file: '/tmp/accelopt-sol/STOP',
  }, stoppingAgents, evals(candidate))
  assert.equal(result.termination_reason, 'wall_clock_limit')
  assert.equal(result.iterations_completed, 1)
  assert.equal(result.generated_kernel_path, candidate.candidate_path)
  assert.ok(calls.some(call => call.label === 'checkpoint-1' &&
    call.prompt.includes('deadline epoch: 123')))
  assert.ok(!calls.some(call => call.label === 'final-report'))
})

test('AccelOpt rejects a wall deadline on the CUDA path it cannot honor', async () => {
  await assert.rejects(() => runWorkflow(source, {...args, deadline_epoch: 123}, agents),
    /cooperative wall controls are supported only for CuTe SOL/)
})
