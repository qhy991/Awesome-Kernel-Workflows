'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'AKO4X/ako4x-kernel-optimizer.js')

const baseArgs = {
  kernel_path: '/tmp/ako4x-guard/baseline.py',
  op_description: 'rmsnorm fused kernel',
  benchmark_command: 'python /tmp/ako4x-guard/bench.py {kernel_path} {result_path}',
  smoke_test_command: 'python /tmp/ako4x-guard/smoke.py {kernel_path}',
  exp_dir: '/tmp/ako4x-guard',
  iterations: 1,
  iters_per_round: 1,
  breadth: 1,
  samples_per_plan: 1,
  target_gpu: 'b200',
  mode: 2,
  use_ako4x_skills: false,
}

function ds(idx) {
  return {
    [`driver-build-${idx}`]: { ok: true },
    [`driver-run-${idx}`]: { ok: true, latency_ms: 1, compiled: true, correct: true },
    [`driver-profile-${idx}`]: { ok: true, native_path: '/tmp/p.native' },
    [`driver-to-evidence-${idx}`]: { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
    [`driver-diagnose-${idx}`]: { bottleneck_class: 'memory_bound' },
    [`driver-anti-cheat-${idx}`]: { ok: true, suspicious: false },
  }
}

const minimalReturns = {
  'load-driver': { present: true, backend_id: 'triton', source_ext: '.py', aux_ext: '.py', lang_fence: 'python', impl_requirements: 'Provide a @triton.jit kernel.', methods: {}, hw_vendor: 'nvidia' },
  'read-baseline': { kernel_code: 'k', language: 'triton', op_type: 'rmsnorm', key_functions: ['k'], current_approach: 'a' },
  'create-workspace': { ok: true },
  'benchmark-baseline': { score: 0.085, latency_ms: 0.085, raw_output: '' },
  'hypothesis-0-0': { title: 'vec', bottleneck: 'mem', ncu_evidence: 'na', hypothesis: 'h', expected_impact: '1.2x', risk: 'r' },
  'impl-0-vec-v0': { code: 'k', implementation_notes: 'n' },
  'smoke-r1-iter1-vec': { passed: true, error_message: '', compile_error: false, correctness_error: false },
  'bench-r1-iter1-vec': { score: 0.070, latency_ms: 0.070, speedup: 1.21, passed_workloads: '2/2', ab_compare_delta: '', variance_info: '', raw_output: '' },
  ...ds(0),
  'iterations-1': { ok: true },
  'silent-skip-check-1': { is_legitimate: true, suspicion_level: 'low', concerns: [], recommendation: 'archive' },
  'lib-check-1': { is_own_kernel: true, banned_libs_found: [], concerns: '' },
  'archive-iter-1-vec': { ok: true },
  'update-state-1': { ok: true },
  'final-report': { report_md: 'r' },
}

async function run(extra, agentReturns) {
  return capturePrompts({
    workflowPath: WORKFLOW,
    args: { ...baseArgs, ...extra },
    agentReturns: agentReturns || {},
  })
}

test('legacy path: no backend/no backend_dir -> renders triton vocabulary, no load-driver agent', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({ language: 'triton' }, legacyReturns)
  assert.ok(caps.length > 0)
  assert.ok(!caps.some(c => c.label === 'load-driver'),
    'legacy path must NOT issue load-driver agent (USE_DRIVER off-by-default)')
})

test('§6.4: args.backend matches manifest backend_id -> ok', async () => {
  await assert.doesNotReject(run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton', language: 'triton' },
    minimalReturns,
  ))
})

test('§6.4: conflicting backend + language -> throws naming both', async () => {
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

test('§6.4: args.backend without backend_dir -> throws (no implicit-resolve)', async () => {
  await assert.rejects(
    run({ backend: 'cuda' }),
    (err) => {
      assert.match(err.message, /requires args\.backend_dir/)
      return true
    },
  )
})

test('intersectional guard #1: ncu_binary + non-cuda driver -> throws naming both', async () => {
  await assert.rejects(
    run(
      {
        backend_dir: '_substrate/backends/triton',
        backend: 'triton',
        language: 'triton',
        ncu_binary: '/usr/local/cuda/bin/ncu',
      },
    ),
    (err) => {
      assert.match(err.message, /ncu_binary="\/usr\/local\/cuda\/bin\/ncu"/)
      assert.match(err.message, /backend_dir=_substrate\/backends\/triton/)
      assert.match(err.message, /CUDA driver/)
      return true
    },
  )
})

test('intersectional guard #1: ncu_binary + cuda driver -> ok', async () => {
  const cudaReturns = {
    ...minimalReturns,
    'load-driver': { present: true, backend_id: 'cuda', source_ext: '.cu', aux_ext: '.cpp', lang_fence: 'cuda', impl_requirements: '', methods: {}, hw_vendor: 'nvidia' },
  }
  await assert.doesNotReject(run(
    {
      backend_dir: '_substrate/backends/cuda',
      backend: 'cuda',
      language: 'cuda',
      ncu_binary: '/usr/local/cuda/bin/ncu',
    },
    cudaReturns,
  ))
})

test('intersectional guard #2: mode=3 + USE_DRIVER -> throws naming both', async () => {
  await assert.rejects(
    run(
      {
        backend_dir: '_substrate/backends/triton',
        backend: 'triton',
        language: 'triton',
        mode: 3,
      },
    ),
    (err) => {
      assert.match(err.message, /mode=3/)
      assert.match(err.message, /harness co-evolution/)
      assert.match(err.message, /backend_dir=_substrate\/backends\/triton/)
      assert.match(err.message, /_substrate driver tree/)
      return true
    },
  )
})

test('intersectional guard #2: mode=3 + USE_DRIVER + cuda backend -> also throws (vendor-independent)', async () => {
  await assert.rejects(
    run(
      {
        backend_dir: '_substrate/backends/cuda',
        backend: 'cuda',
        language: 'cuda',
        mode: 3,
      },
    ),
    (err) => {
      assert.match(err.message, /mode=3/)
      assert.match(err.message, /_substrate driver tree/)
      return true
    },
  )
})

test('intersectional guard #2: mode=2 + USE_DRIVER -> ok (default mode)', async () => {
  await assert.doesNotReject(run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton', language: 'triton', mode: 2 },
    minimalReturns,
  ))
})

test('USE_DRIVER on with valid driver: load-driver fires first + per-attempt envelope agents present', async () => {
  const caps = await run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton', language: 'triton' },
    minimalReturns,
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver',
    'load-driver must be the first agent call when USE_DRIVER')
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-0`),
      `expected driver-${sub}-0 envelope agent under USE_DRIVER, got labels=${labels.join(',')}`)
  }
})

test('manifest backend_id mismatch with args.backend -> throws', async () => {
  const driverReturn = { present: true, backend_id: 'triton', source_ext: '.py', lang_fence: 'python' }
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
