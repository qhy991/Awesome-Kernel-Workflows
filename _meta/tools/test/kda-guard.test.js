'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'KDA/kda-kernel-workflow.js')

const baseArgs = {
  kernel_path: '/tmp/kda-guard/baseline.cu',
  task_name: 'kda-guard',
  objective: 'Optimize the target kernel',
  test_command: 'python /tmp/kda-guard/test.py {kernel_path} {result_path}',
  benchmark_command: 'python /tmp/kda-guard/bench.py {kernel_path} {result_path}',
  max_candidates: 1,
  exp_dir: '/tmp/kda-guard',
}

function ds(candidateId) {
  return {
    [`driver-build-${candidateId}`]: { ok: true },
    [`driver-run-${candidateId}`]: { ok: true, latency_ms: 1, compiled: true, correct: true },
    [`driver-profile-${candidateId}`]: { ok: true, native_path: '/tmp/p.native' },
    [`driver-to-evidence-${candidateId}`]: { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
    [`driver-diagnose-${candidateId}`]: { bottleneck_class: 'memory_bound' },
    [`driver-anti-cheat-${candidateId}`]: { ok: true, suspicious: false },
  }
}

const minimalReturns = {
  'load-driver': { present: true, backend_id: 'triton', source_ext: '.py', aux_ext: '.py', lang_fence: 'python' },
  'inspect-workspace': { kernel_code: 'k', key_functions: ['f'], current_approach: 'naive' },
  'write-draft': { draft_content: 'd', candidate_directions: [{ title: 't', approach: 'a' }] },
  'write-plan': { plan_content: 'p', candidates: [{ id: 'c1', title: 't', changes: 'c' }] },
  'impl-candidate-1': { code: 'k' },
  'validate-candidate-1': { is_correct: true, estimated_speedup: 1.5, validation_ran: true, addresses_goal: true },
  ...ds('candidate-1'),
  'final-report': { report_md: 'r' },
}

async function run(extra, agentReturns) {
  return capturePrompts({
    workflowPath: WORKFLOW,
    args: { ...baseArgs, ...extra },
    agentReturns: agentReturns || {},
  })
}

test('legacy path: no backend/no backend_dir -> renders CUDA vocabulary, no driver-load agent', async () => {
  const caps = await run({}, {
    'inspect-workspace': { kernel_code: 'k', key_functions: ['f'], current_approach: 'naive' },
    'write-draft': { draft_content: 'd', candidate_directions: [{ title: 't', approach: 'a' }] },
    'write-plan': { plan_content: 'p', candidates: [{ id: 'c1', title: 't', changes: 'c' }] },
    'impl-candidate-1': { code: 'k' },
    'validate-candidate-1': { is_correct: true, estimated_speedup: 1.5, validation_ran: true, addresses_goal: true },
    'final-report': { report_md: 'r' },
  })
  assert.ok(caps.length > 0)
  assert.ok(caps.some(c => /CUDA/.test(c.prompt)), 'legacy path should keep CUDA vocabulary')
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
    minimalReturns,
  ))
})

test('USE_DRIVER on with backend_dir: load-driver agent fires before inspect-workspace', async () => {
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
    { backend_dir: '_substrate/backends/cuda', backend: 'cuda' },
    { ...minimalReturns, 'load-driver': driverReturn },
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver', 'load-driver must be the first agent call when USE_DRIVER')
  assert.ok(labels.includes('driver-build-candidate-1'))
  assert.ok(labels.includes('driver-run-candidate-1'))
  assert.ok(labels.includes('driver-profile-candidate-1'))
  assert.ok(labels.includes('driver-to-evidence-candidate-1'))
  assert.ok(labels.includes('driver-diagnose-candidate-1'))
  assert.ok(labels.includes('driver-anti-cheat-candidate-1'))
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

test('alias normalization: hip -> rocm in conflict message', async () => {
  await assert.rejects(
    run({ backend: 'hip', language: 'cuda' }),
    (err) => {
      assert.match(err.message, /backend="hip"/)
      assert.match(err.message, /language="cuda"/)
      return true
    },
  )
})
