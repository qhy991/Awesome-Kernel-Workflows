'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'AdaExplore/adaexplore-kernel-optimization.js')

const baseArgs = {
  problem_definition: 'class M(nn.Module):\n  def forward(self, x):\n    return torch.relu(x)\n',
  op_description: 'elementwise_relu',
  iterations: 1,
  small_step_limit: 2,
  exp_dir: '/tmp/adaexplore-guard',
  language: 'triton',
  benchmark_command: 'python eval.py --kernel {kernel_path}',
  memory_update: false,
}

async function run(extra, agentReturns) {
  return capturePrompts({ workflowPath: WORKFLOW, args: { ...baseArgs, ...extra }, agentReturns: agentReturns || {} })
}

test('legacy path: no backend/no backend_dir -> renders Triton vocabulary, no driver-load agent', async () => {
  const caps = await run({})
  assert.ok(caps.length > 0)
  assert.ok(caps.some(c => /Triton/.test(c.prompt)), 'legacy path should keep Triton vocabulary')
  assert.ok(!caps.some(c => c.label === 'load-driver'),
    'legacy path must NOT issue load-driver agent (USE_DRIVER off-by-default)')
})

test('§6.4: conflicting backend + language (post-normalize) -> throws naming both values', async () => {
  await assert.rejects(
    run({ backend: 'cuda', language: 'triton' }),
    (err) => {
      assert.match(err.message, /Conflicting args/)
      assert.match(err.message, /backend="cuda"/)
      assert.match(err.message, /language="triton"/)
      return true
    },
  )
})

test('§6.4: args.backend without args.backend_dir -> throws (no implicit-resolve)', async () => {
  await assert.rejects(
    run({ backend: 'cuda', language: undefined }),
    (err) => {
      assert.match(err.message, /requires args\.backend_dir/)
      return true
    },
  )
})

test('§6.4: backend + language equal post-normalize -> no throw (with backend_dir)', async () => {
  await assert.doesNotReject(run(
    { backend: 'triton', language: 'triton', backend_dir: '_substrate/backends/triton' },
    {
      'load-driver': { present: true, backend_id: 'triton', source_ext: '.py', lang_fence: 'python' },
      'setup': { operator_code: 'x', evaluator_command: 'echo', baseline_time_ms: 0, hardware_info: '', initial_skill_memory: [], reference_path: '/tmp/r.py' },
      'propose-1': { kernel_code: 'pass', strategy: 's' },
      'eval-1': { compiled: true, correct: true, speedup: 1.0 },
      'driver-build-1': { ok: true },
      'driver-run-1': { ok: true, latency_ms: 0.5 },
      'driver-profile-1': { ok: true },
      'driver-to-evidence-1': { ok: true, metrics: {} },
      'driver-diagnose-1': { bottleneck_class: 'unknown' },
      'driver-anti-cheat-1': { ok: true },
      'final-report': 'r',
    },
  ))
})

test('alias normalization: hip -> rocm in conflict message', async () => {
  await assert.rejects(
    run({ backend: 'hip', language: 'triton' }),
    (err) => {
      assert.match(err.message, /backend="hip"/)
      assert.match(err.message, /language="triton"/)
      return true
    },
  )
})

test('USE_DRIVER on with backend_dir: load-driver agent fires before setup', async () => {
  const driverReturn = {
    present: true,
    backend_id: 'triton',
    source_ext: '.py',
    lang_fence: 'python',
    impl_requirements: 'Provide a @triton.jit kernel plus a plain Python launcher.',
    methods: {},
    hw_vendor: 'nvidia',
  }
  const caps = await run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton' },
    {
      'load-driver': driverReturn,
      'setup': { operator_code: 'x', evaluator_command: 'echo', baseline_time_ms: 0, hardware_info: 'A100', initial_skill_memory: [], reference_path: '/tmp/r.py' },
      'propose-1': { kernel_code: 'pass', strategy: 's' },
      'eval-1': { compiled: true, correct: true, speedup: 1.0 },
      'driver-build-1': { ok: true },
      'driver-run-1': { ok: true, latency_ms: 0.5, compiled: true, correct: true },
      'driver-profile-1': { ok: true, native_path: '/tmp/p.native' },
      'driver-to-evidence-1': { ok: true, metrics: { latency_ms: 0.5 }, coverage: ['latency_ms'] },
      'driver-diagnose-1': { bottleneck_class: 'memory_bound' },
      'driver-anti-cheat-1': { ok: true, suspicious: false },
      'final-report': 'driver-path report',
    },
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver', 'load-driver must be the first agent call when USE_DRIVER')
  assert.ok(labels.includes('driver-build-1'))
  assert.ok(labels.includes('driver-run-1'))
  assert.ok(labels.includes('driver-profile-1'))
  assert.ok(labels.includes('driver-to-evidence-1'))
  assert.ok(labels.includes('driver-diagnose-1'))
  assert.ok(labels.includes('driver-anti-cheat-1'))
})

test('driver-injected substrings: triton driver lang_fence reaches setup prompt', async () => {
  const driverReturn = {
    present: true,
    backend_id: 'triton',
    source_ext: '.py',
    lang_fence: 'python',
    impl_requirements: 'Provide a @triton.jit kernel plus a plain Python launcher.',
    methods: {},
  }
  const caps = await run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton' },
    {
      'load-driver': driverReturn,
      'setup': { operator_code: 'x', evaluator_command: 'echo', baseline_time_ms: 0, hardware_info: 'A100', initial_skill_memory: [], reference_path: '/tmp/r.py' },
      'propose-1': { kernel_code: 'pass', strategy: 's' },
      'eval-1': { compiled: true, correct: true, speedup: 1.0 },
      'driver-build-1': { ok: true },
      'driver-run-1': { ok: true, latency_ms: 0.5 },
      'driver-profile-1': { ok: true },
      'driver-to-evidence-1': { ok: true, metrics: {} },
      'driver-diagnose-1': { bottleneck_class: 'unknown' },
      'driver-anti-cheat-1': { ok: true },
      'final-report': 'r',
    },
  )
  const setupCall = caps.find(c => c.label === 'setup')
  assert.ok(setupCall, 'setup agent call must be present')
  assert.ok(/python kernel/.test(setupCall.prompt),
    `setup prompt should contain driver lang_fence-derived "python kernel"; got: ${setupCall.prompt.slice(0, 200)}`)
  const proposeCall = caps.find(c => c.label === 'propose-1')
  assert.ok(/python implementation/.test(proposeCall.prompt) || /@triton\.jit/.test(proposeCall.prompt),
    'large-step prompt should reflect driver lang_fence or impl_requirements')
})

test('manifest backend_id mismatch with args.backend -> throws', async () => {
  const driverReturn = {
    present: true,
    backend_id: 'cuda',
    source_ext: '.cu',
    lang_fence: 'cuda',
  }
  await assert.rejects(
    run(
      { backend_dir: '_substrate/backends/triton', backend: 'triton' },
      { 'load-driver': driverReturn },
    ),
    (err) => {
      assert.match(err.message, /backend_id="cuda"/)
      assert.match(err.message, /conflicts/)
      return true
    },
  )
})
