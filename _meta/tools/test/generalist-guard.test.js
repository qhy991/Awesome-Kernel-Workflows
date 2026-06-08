'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'Generalist/generalist-kernel-optimization.js')

const baseArgs = {
  kernel_path: '/tmp/generalist-guard/kernel.cu',
  op_description: 'matmul',
  language: 'cuda',
  target_gpu: 'A100',
  iterations: 1,
  breadth: 1,
  topk: 1,
  target_speedup: 1.5,
  benchmark_command: 'python /tmp/generalist-guard/bench.py {kernel_path} {result_path}',
  substrate_dir: '_substrate',
  substrate_command_prefix: 'python3',
  exp_dir: '/tmp/generalist-guard',
}

function ds(suffix) {
  return {
    [`driver-build-${suffix}`]: { ok: true },
    [`driver-run-${suffix}`]: { ok: true, latency_ms: 1, compiled: true, correct: true },
    [`driver-profile-${suffix}`]: { ok: true, native_path: '/tmp/p.native' },
    [`driver-to-evidence-${suffix}`]: { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
    [`driver-diagnose-${suffix}`]: { bottleneck_class: 'memory_bound' },
    [`driver-anti-cheat-${suffix}`]: { ok: true, suspicious: false },
  }
}

const minimalReturns = {
  'load-driver': { present: true, backend_id: 'cuda', source_ext: '.cu', aux_ext: '.cu', lang_fence: 'cuda', impl_requirements: 'Provide a __global__ CUDA kernel.', methods: {}, hw_vendor: 'nvidia' },
  'profile-1': { compiled: true, correct: true, speedup: 1.0, metrics: {} },
  'diagnose-1': { bottleneck_class: 'memory_bound' },
  'retrieve-1': { techniques: [], dead_ends: [] },
  'gate-1': { allowed_methods: ['tiling'] },
  'plan-1-1': { method: 'tiling', plan: 'tile' },
  'impl-1-1': { compiled: true, correct: true, speedup: 1.2, metrics: {} },
  'anticheat-1-1': { valid: true, reward: 0.2, recorded_speedup: 1.2 },
  'learn-1-tiling': { updated: true },
  'refute-1': { refuted: false },
  'verify-insight-1': { confidence: 'measured' },
  'final-report': {},
  ...ds('setup'),
  ...ds('1-1'),
}

async function run(extra, agentReturns) {
  return capturePrompts({
    workflowPath: WORKFLOW,
    args: { ...baseArgs, ...extra },
    agentReturns: agentReturns || {},
  })
}

test('legacy path: no backend_dir -> renders cuda vocabulary, no load-driver agent', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({}, legacyReturns)
  assert.ok(caps.length > 0)
  assert.ok(!caps.some(c => c.label === 'load-driver'),
    'legacy path must NOT issue load-driver agent (USE_DRIVER off-by-default)')
})

test('legacy path: language=cuda is accepted by suitability guard', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({ language: 'cuda' }, legacyReturns)
  assert.ok(caps.length > 0)
})

test('legacy path: language=triton is REJECTED by suitability guard', async () => {
  await assert.rejects(
    run({ language: 'triton' }, minimalReturns),
    (err) => {
      assert.match(err.message, /not suitable for language="triton"/)
      assert.match(err.message, /Supported languages\/backends: cuda/)
      return true
    },
  )
})

test('legacy path: language=sycl is REJECTED by suitability guard', async () => {
  await assert.rejects(
    run({ language: 'sycl' }, minimalReturns),
    (err) => {
      assert.match(err.message, /not suitable for language="sycl"/)
      return true
    },
  )
})

test('§6.4: args.backend + args.language conflict (post-normalize) -> throws naming both', async () => {
  await assert.rejects(
    run({ backend: 'triton', language: 'cuda', backend_dir: '_substrate/backends/triton' }),
    (err) => {
      assert.match(err.message, /Conflicting args/)
      assert.match(err.message, /backend="triton"/)
      assert.match(err.message, /language="cuda"/)
      return true
    },
  )
})

test('§6.4: args.backend + args.language equal post-normalize -> no throw', async () => {
  await assert.doesNotReject(run(
    { backend: 'cuda', language: 'cuda', backend_dir: '_substrate/backends/cuda' },
    minimalReturns,
  ))
})

test('§6.4: args.backend without args.backend_dir -> throws (no implicit-resolve)', async () => {
  const { language, ...noLang } = baseArgs
  await assert.rejects(
    capturePrompts({
      workflowPath: WORKFLOW,
      args: { ...noLang, backend: 'cuda' },
      agentReturns: {},
    }),
    (err) => {
      assert.match(err.message, /requires args\.backend_dir/)
      return true
    },
  )
})

test('manifest backend_id mismatch with args.backend -> throws naming both ids', async () => {
  const driverReturn = { present: true, backend_id: 'triton', source_ext: '.py', lang_fence: 'python' }
  const { language, ...noLang } = baseArgs
  await assert.rejects(
    capturePrompts({
      workflowPath: WORKFLOW,
      args: { ...noLang, backend_dir: '_substrate/backends/cuda', backend: 'cuda' },
      agentReturns: { 'load-driver': driverReturn },
    }),
    (err) => {
      assert.match(err.message, /backend_id="triton"/)
      assert.match(err.message, /conflicts/)
      return true
    },
  )
})

test('USE_DRIVER on with valid driver: load-driver fires first + per-eval envelope agents present', async () => {
  const caps = await run(
    { backend_dir: '_substrate/backends/cuda', backend: 'cuda', language: 'cuda' },
    minimalReturns,
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver',
    'load-driver must be the first agent call when USE_DRIVER')
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-setup`),
      `expected driver-${sub}-setup envelope agent under USE_DRIVER, got labels=${labels.join(',')}`)
    assert.ok(labels.includes(`driver-${sub}-1-1`),
      `expected driver-${sub}-1-1 envelope agent (iter 1) under USE_DRIVER, got labels=${labels.join(',')}`)
  }
})

test('USE_DRIVER on: missing load-driver result (present=false) -> throws', async () => {
  await assert.rejects(
    run(
      { backend_dir: '_substrate/backends/cuda', backend: 'cuda', language: 'cuda' },
      { ...minimalReturns, 'load-driver': { present: false } },
    ),
    (err) => {
      assert.match(err.message, /No backend driver present/)
      return true
    },
  )
})
