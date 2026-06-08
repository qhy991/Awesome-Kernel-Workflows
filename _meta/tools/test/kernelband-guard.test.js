'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'KernelBand/kernelband-kernel-optimization.js')

const baseArgs = {
  kernel_path: '/tmp/kband-guard/kernel.py',
  op_description: 'matmul',
  language: 'triton',
  target_gpu: 'A100',
  iterations: 1,
  num_clusters: 3,
  recluster_period: 10,
  saturation_threshold: 0.75,
  strategies: ['tiling', 'vectorization'],
  exp_dir: '/tmp/kband-guard',
  benchmark_command: 'python /tmp/kband-guard/bench.py {kernel_path} {result_path}',
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
  'load-driver': { present: true, backend_id: 'triton', source_ext: '.py', aux_ext: '.py', lang_fence: 'python', impl_requirements: 'Provide a @triton.jit kernel.', methods: {}, hw_vendor: 'nvidia' },
  'setup': {
    kernel_code: '@triton.jit\ndef k(): pass\n',
    baseline_latency_us: 1000,
    hardware_signature: { dram_throughput_pct: 50, l2_throughput_pct: 30, sm_throughput_pct: 40, dominant_bottleneck: 'memory' },
    behavioral_features: { normalized_time: 1.0, registers_per_thread: 32, shared_mem_bytes: 0, block_dimension: 256, occupancy: 0.5 },
    platform_info: 'A100',
  },
  'generate-t1-tiling': { optimized_kernel: 'k', changes_description: 'c', expected_improvement: 'e' },
  'eval-t1': { compiled: true, correct: true, latency_us: 800, speedup: 1.25 },
  ...ds('setup'),
  ...ds('t1'),
  'report': 'r',
}

async function run(extra, agentReturns) {
  return capturePrompts({
    workflowPath: WORKFLOW,
    args: { ...baseArgs, ...extra },
    agentReturns: agentReturns || {},
  })
}

test('legacy path: no backend_dir -> renders triton vocabulary, no load-driver agent', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({}, legacyReturns)
  assert.ok(caps.length > 0)
  assert.ok(caps.some(c => /```triton\b/.test(c.prompt)), 'legacy path with language=triton should render ```triton code fence')
  assert.ok(!caps.some(c => c.label === 'load-driver'),
    'legacy path must NOT issue load-driver agent (USE_DRIVER off-by-default)')
})

test('legacy path: language=cuda renders cuda vocabulary, no load-driver', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({ language: 'cuda' }, legacyReturns)
  assert.ok(caps.length > 0)
  assert.ok(caps.some(c => /```cuda\b/.test(c.prompt)), 'legacy path with language=cuda should render ```cuda code fence')
  assert.ok(!caps.some(c => c.label === 'load-driver'))
})

test('legacy path: language=sycl is REJECTED by suitability guard', async () => {
  await assert.rejects(
    run({ language: 'sycl' }, minimalReturns),
    (err) => {
      assert.match(err.message, /not suitable for language="sycl"/)
      assert.match(err.message, /Supported languages\/backends: triton, cuda/)
      return true
    },
  )
})

test('§6.4: args.backend + args.language conflict (post-normalize) -> throws naming both', async () => {
  await assert.rejects(
    run({ backend: 'cuda', language: 'triton', backend_dir: '_substrate/backends/cuda' }),
    (err) => {
      assert.match(err.message, /Conflicting args/)
      assert.match(err.message, /backend="cuda"/)
      assert.match(err.message, /language="triton"/)
      return true
    },
  )
})

test('§6.4: args.backend + args.language equal post-normalize -> no throw', async () => {
  await assert.doesNotReject(run(
    { backend: 'triton', language: 'triton', backend_dir: '_substrate/backends/triton' },
    minimalReturns,
  ))
})

test('§6.4: args.backend without args.backend_dir -> throws (no implicit-resolve)', async () => {
  const { language, ...noLang } = baseArgs
  await assert.rejects(
    capturePrompts({
      workflowPath: WORKFLOW,
      args: { ...noLang, backend: 'triton' },
      agentReturns: {},
    }),
    (err) => {
      assert.match(err.message, /requires args\.backend_dir/)
      return true
    },
  )
})

test('manifest backend_id mismatch with args.backend -> throws naming both ids', async () => {
  const driverReturn = { present: true, backend_id: 'cuda', source_ext: '.cu', lang_fence: 'cuda' }
  const { language, ...noLang } = baseArgs
  await assert.rejects(
    capturePrompts({
      workflowPath: WORKFLOW,
      args: { ...noLang, backend_dir: '_substrate/backends/triton', backend: 'triton' },
      agentReturns: { 'load-driver': driverReturn },
    }),
    (err) => {
      assert.match(err.message, /backend_id="cuda"/)
      assert.match(err.message, /conflicts/)
      return true
    },
  )
})

test('USE_DRIVER on with valid triton driver: load-driver fires first + per-iteration envelope agents present', async () => {
  const caps = await run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton', language: 'triton' },
    minimalReturns,
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver',
    'load-driver must be the first agent call when USE_DRIVER')
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-setup`),
      `expected driver-${sub}-setup envelope agent under USE_DRIVER, got labels=${labels.join(',')}`)
    assert.ok(labels.includes(`driver-${sub}-t1`),
      `expected driver-${sub}-t1 envelope agent (iter 1) under USE_DRIVER, got labels=${labels.join(',')}`)
  }
})

test('USE_DRIVER on: missing load-driver result (present=false) -> throws', async () => {
  await assert.rejects(
    run(
      { backend_dir: '_substrate/backends/triton', backend: 'triton', language: 'triton' },
      { ...minimalReturns, 'load-driver': { present: false } },
    ),
    (err) => {
      assert.match(err.message, /No backend driver present/)
      return true
    },
  )
})

// --- phi-gate (SATURATION_THRESHOLD) tests ---

test('phi-gate fallback: when driver does NOT supply saturation_threshold, LEGACY value 0.75 is used', async () => {
  const driverNoThreshold = {
    present: true,
    backend_id: 'triton',
    source_ext: '.py',
    aux_ext: '.py',
    lang_fence: 'python',
    impl_requirements: 'Provide a @triton.jit kernel.',
    methods: {},
    hw_vendor: 'nvidia',
    // no saturation_threshold field
  }
  const caps = await run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton', language: 'triton' },
    { ...minimalReturns, 'load-driver': driverNoThreshold },
  )
  // The generate prompt should reference 75% (0.75 * 100)
  const genPrompt = caps.find(c => c.label === 'generate-t1-tiling')
  assert.ok(genPrompt, 'generate-t1-tiling prompt must exist')
  assert.ok(genPrompt.prompt.includes('75%'),
    'phi-gate with LEGACY fallback should produce 75% threshold in generate prompt')
})

test('phi-gate driver-resolved: when driver supplies saturation_threshold=0.80, that value is used', async () => {
  const driverWithThreshold = {
    present: true,
    backend_id: 'triton',
    source_ext: '.py',
    aux_ext: '.py',
    lang_fence: 'python',
    impl_requirements: 'Provide a @triton.jit kernel.',
    methods: {},
    hw_vendor: 'nvidia',
    saturation_threshold: 0.80,
  }
  const caps = await run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton', language: 'triton' },
    { ...minimalReturns, 'load-driver': driverWithThreshold },
  )
  const genPrompt = caps.find(c => c.label === 'generate-t1-tiling')
  assert.ok(genPrompt, 'generate-t1-tiling prompt must exist')
  assert.ok(genPrompt.prompt.includes('80%'),
    'phi-gate with driver-resolved threshold should produce 80% in generate prompt')
  assert.ok(!genPrompt.prompt.includes('75%'),
    'phi-gate with driver-resolved threshold should NOT fall back to 75%')
})

test('phi-gate legacy path: hardcoded 0.75 is always used when USE_DRIVER is off', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({ saturation_threshold: 0.75 }, legacyReturns)
  const genPrompt = caps.find(c => c.label === 'generate-t1-tiling')
  assert.ok(genPrompt, 'generate-t1-tiling prompt must exist')
  assert.ok(genPrompt.prompt.includes('75%'),
    'legacy phi-gate should produce 75% threshold in generate prompt')
})
