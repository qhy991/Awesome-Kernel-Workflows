'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const runWorkflow = require(path.resolve(__dirname, '..', 'lib/run-workflow.js'))

const ROOT = path.resolve(__dirname, '..', '..', '..')
const SSOT = fs.readFileSync(path.join(ROOT, '_meta/scaffolding/runtime-safe-point.js'), 'utf8')
const CANON = SSOT.slice(SSOT.indexOf('async function __workflowRuntimeSafePoint(ctx) {')).trim()
const WORKFLOWS = [
  'KSearch/ksearch-kernel-optimization.js',
  'CUDAAgent/cuda-agent-kernel-optimization.js',
  'AdaExplore/adaexplore-kernel-optimization.js',
  'KernelAgent/kernelagent-triton-synthesis.js',
  'KernelFoundry/kernelfoundry-kernel-optimization.js',
]

for (const relative of WORKFLOWS) {
  test(`${relative}: uses the canonical cooperative runtime safe point`, () => {
    const source = fs.readFileSync(path.join(ROOT, relative), 'utf8')
    assert.match(source, /BEGIN inlined runtime-safe-point scaffolding/)
    assert.ok(source.includes(CANON), 'inlined helper must byte-match the SSOT')
    assert.match(source, /termination_file/)
    assert.match(source, /deadline_epoch/)
    assert.match(source, /await __workflowRuntimeSafePoint\(/)
  })
}

test('safe point performs atomic checkpointing and observes both supervisor controls', () => {
  assert.match(CANON, /temporary file in the same directory followed by os\.replace\/rename/)
  assert.match(CANON, /termination file:/)
  assert.match(CANON, /deadline epoch:/)
  assert.match(CANON, /date \+%s/)
  assert.match(CANON, /termination_requested/)
})

test('CUDAAgent cooperative stop skips the expensive final-report turn', async () => {
  const source = fs.readFileSync(
    path.join(ROOT, 'CUDAAgent/cuda-agent-kernel-optimization.js'),
    'utf8',
  )
  const { calls } = await runWorkflow(source, {
    kernel_path: '/tmp/input.cu',
    op_description: 'safe-point-test',
    max_turns: 1,
    target_speedup: 'none',
    exp_dir: '/tmp/cudaagent-safe-point-test',
  }, {
    'setup-workspace': {
      model_code: 'x',
      environment_info: 'x',
      gpu_info: 'x',
      available_tools: [],
    },
    'profile-baseline': {
      eager_time_ms: 1,
      compile_time_ms: 1,
      per_operator_breakdown: [],
      bottlenecks: [],
      optimization_strategy: 'x',
      fusion_plan: 'x',
    },
    'impl-0': {
      kernel_code: 'extern "C" __global__ void k(){}',
      binding_code: '',
      model_new_code: '',
      implementation_notes: 'x',
    },
    'verify-0': {
      compiled: true,
      correct: true,
      kernel_time_ms: 0.5,
      speedup_vs_eager: 2,
      speedup_vs_compile: 2,
      reward: 3,
    },
    'checkpoint-0': {
      termination_requested: true,
      termination_reason: 'test_stop',
      checkpoint_path: '/tmp/cudaagent-safe-point-test/checkpoint.json',
    },
  })
  const labels = calls.map(call => call.label)
  assert.ok(labels.includes('checkpoint-0'))
  assert.ok(!labels.includes('final-report'))
})
