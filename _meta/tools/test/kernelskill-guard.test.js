'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'KernelSkill/kernelskill-kernel-optimization.js')

const baseArgs = {
  problem_path: '/tmp/ks-guard/relu.py',
  problem_definition: 'class Model(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)',
  op_description: 'ReLU',
  target_gpu: 'A100-80GB',
  iterations: 1,
  seed_candidates: 1,
  exp_dir: '/tmp/ks-guard',
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
  'load-driver': { present: true, backend_id: 'cuda', source_ext: '.cu', aux_ext: '.h', lang_fence: 'cuda', impl_requirements: 'Provide a CUDA __global__ kernel.', methods: {}, hw_vendor: 'nvidia' },
  'read-reference': { reference_code: 'class Model(nn.Module): pass', op_type: 'relu', forward_summary: 'relu', input_shapes: '[N]', dtype: 'float32', num_ops: 1, has_matmul: false },
  'eager-baseline': { baseline_latency_ms: 0.5, baseline_available: true, harness_notes: 'n/a' },
  'seed-0': { code: '__global__ void k(){}', strategy: 's', kernels_materialized: 1 },
  'seed-eval-0': { is_compilable: true, is_correct: true, latency_ms: 0.4, speedup: 1.25, issues: [] },
  'review-0': {
    is_compilable: true, is_correct: true, latency_ms: 0.4, speedup: 1.25, error_excerpt: '',
    ncu_metrics: { dram_throughput_pct: 80 }, profile_summary: 'mem-bound',
  },
  'features-0': { has_reuse: false, streaming_no_reuse: true, kernel_structure_id: 0 },
  'gate-0': { tier: 'Tier-H', bottleneck_id: 'memory_bandwidth_bound', matched_case_id: 'M1', allowed_methods: ['vectorized_load_store'], key_metrics: 'k', derived_values: 'd' },
  'plan-0': { method_name: 'vectorized_load_store', rationale: 'r', plan: 'p' },
  'optimize-0': { code: '__global__ void k2(){}', implementation_notes: 'n' },
  ...ds('0'),
  'final-report': 'r',
}

async function run(extra, agentReturns) {
  return capturePrompts({
    workflowPath: WORKFLOW,
    args: { ...baseArgs, ...extra },
    agentReturns: agentReturns || {},
  })
}

test('legacy path: no backend_dir, language=cuda -> renders cuda vocabulary, no load-driver agent', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({ language: 'cuda' }, legacyReturns)
  assert.ok(caps.length > 0)
  assert.ok(caps.some(c => /```cuda\b/.test(c.prompt)), 'legacy path with language=cuda should render ```cuda code fence')
  assert.ok(!caps.some(c => c.label === 'load-driver'),
    'legacy path must NOT issue load-driver agent (USE_DRIVER off-by-default)')
})

test('legacy path: language=triton is REJECTED by suitability guard (KernelSkill is CUDA-only)', async () => {
  await assert.rejects(
    run({ language: 'triton' }, minimalReturns),
    (err) => {
      assert.match(err.message, /not suitable for language="triton"/)
      assert.match(err.message, /Supported languages\/backends: cuda/)
      return true
    },
  )
})

test('legacy path: language=sycl is REJECTED by suitability guard (KernelSkill is CUDA-only)', async () => {
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
    { backend: 'cuda', language: 'cuda', backend_dir: '_substrate/backends/cuda' },
    minimalReturns,
  ))
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

test('manifest backend_id mismatch with args.backend -> throws naming both ids', async () => {
  const driverReturn = { present: true, backend_id: 'triton', source_ext: '.py', lang_fence: 'python' }
  const noLang = { ...baseArgs }
  delete noLang.language
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

test('USE_DRIVER on with valid cuda driver: load-driver fires first + per-round envelope agents present', async () => {
  const caps = await run(
    { backend_dir: '_substrate/backends/cuda', backend: 'cuda', language: 'cuda' },
    minimalReturns,
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver',
    'load-driver must be the first agent call when USE_DRIVER')
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-0`),
      `expected driver-${sub}-0 envelope agent (round 0) under USE_DRIVER, got labels=${labels.join(',')}`)
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
