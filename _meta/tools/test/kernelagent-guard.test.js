'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'KernelAgent/kernelagent-triton-synthesis.js')

const baseArgs = {
  problem_definition: 'Implement softmax over Tensor[B,N] in pure Triton',
  seed_candidates: 1,
  iterations: 1,
  verify: false,
  compose: false,
  exp_dir: '/tmp/kernelagent-guard',
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
    run({ backend: 'cuda' }),
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
      'load-driver': { present: true, backend_id: 'triton', source_ext: '.py', aux_ext: '.py', lang_fence: 'python' },
      'setup-problem': { problem_definition: 'p', input_tensors: [], operations: ['x'], output_spec: {shape:'',dtype:''}, complexity_signals: [] },
      'setup-test': { test_code: 'pass' },
      'route-analysis': { path: 'direct', reason: 'r', subgraph_count: 1, estimated_difficulty: 'easy' },
      'gen-main-seed0': { kernel_code: 'pass', approach: 'a', potential_issues: '' },
      'report-summary': { outcome: 'success', summary: 's' },
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

test('USE_DRIVER on with backend_dir: load-driver agent fires before setup-problem', async () => {
  const driverReturn = {
    present: true,
    backend_id: 'cuda',
    source_ext: '.cu',
    aux_ext: '.cpp',
    lang_fence: 'cuda',
    impl_requirements: 'Provide a __global__ kernel plus a host launcher.',
    methods: {},
    hw_vendor: 'nvidia',
  }
  const caps = await run(
    { backend_dir: '_substrate/backends/cuda', backend: 'cuda', verify: true, seed_candidates: 1 },
    {
      'load-driver': driverReturn,
      'setup-problem': { problem_definition: 'p', input_tensors: [], operations: ['x'], output_spec: {shape:'',dtype:''}, complexity_signals: [] },
      'setup-test': { test_code: 'pass' },
      'route-analysis': { path: 'direct', reason: 'r', subgraph_count: 1, estimated_difficulty: 'easy' },
      'gen-main-seed0': { kernel_code: '__global__ void k() {}', approach: 'a', potential_issues: '' },
      'verify-candidate_0': { passed: true, exit_code: 0, stdout: 'PASS', stderr: '', error_summary: '', verification_result: 'pass' },
      'driver-build-candidate_0': { ok: true },
      'driver-run-candidate_0': { ok: true, latency_ms: 0.5, compiled: true, correct: true },
      'driver-profile-candidate_0': { ok: true, native_path: '/tmp/p.native' },
      'driver-to-evidence-candidate_0': { ok: true, metrics: { latency_ms: 0.5 }, coverage: ['latency_ms'] },
      'driver-diagnose-candidate_0': { bottleneck_class: 'memory_bound' },
      'driver-anti-cheat-candidate_0': { ok: true, suspicious: false },
      'report-summary': { outcome: 'success', summary: 's' },
    },
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver', 'load-driver must be the first agent call when USE_DRIVER')
  assert.ok(labels.includes('driver-build-candidate_0'))
  assert.ok(labels.includes('driver-run-candidate_0'))
  assert.ok(labels.includes('driver-profile-candidate_0'))
  assert.ok(labels.includes('driver-to-evidence-candidate_0'))
  assert.ok(labels.includes('driver-diagnose-candidate_0'))
  assert.ok(labels.includes('driver-anti-cheat-candidate_0'))
})

test('driver-injected substrings: cuda driver lang_fence reaches setup-problem prompt', async () => {
  const driverReturn = {
    present: true,
    backend_id: 'cuda',
    source_ext: '.cu',
    aux_ext: '.cpp',
    lang_fence: 'cuda',
    impl_requirements: 'Provide a __global__ kernel + pybind11 host launcher.',
    methods: {},
  }
  const caps = await run(
    { backend_dir: '_substrate/backends/cuda', backend: 'cuda' },
    {
      'load-driver': driverReturn,
      'setup-problem': { problem_definition: 'p', input_tensors: [], operations: ['x'], output_spec: {shape:'',dtype:''}, complexity_signals: [] },
      'setup-test': { test_code: 'pass' },
      'route-analysis': { path: 'direct', reason: 'r', subgraph_count: 1, estimated_difficulty: 'easy' },
      'gen-main-seed0': { kernel_code: 'k', approach: 'a', potential_issues: '' },
      'report-summary': { outcome: 'success', summary: 's' },
    },
  )
  const setupCall = caps.find(c => c.label === 'setup-problem')
  assert.ok(setupCall, 'setup-problem agent call must be present')
  assert.ok(/cuda kernel synthesis expert/.test(setupCall.prompt),
    `setup-problem prompt should contain driver lang_fence "cuda kernel synthesis expert"; got: ${setupCall.prompt.slice(0, 200)}`)
})

test('manifest backend_id mismatch with args.backend -> throws', async () => {
  const driverReturn = {
    present: true,
    backend_id: 'triton',
    source_ext: '.py',
    lang_fence: 'python',
  }
  await assert.rejects(
    run(
      { backend_dir: '_substrate/backends/cuda', backend: 'cuda' },
      { 'load-driver': driverReturn },
    ),
    (err) => {
      assert.match(err.message, /backend_id="triton"/)
      assert.match(err.message, /conflicts/)
      return true
    },
  )
})
