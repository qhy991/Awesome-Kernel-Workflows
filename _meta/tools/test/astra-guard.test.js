'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'Astra/astra-kernel-optimization.js')

const baseArgs = {
  kernel_path: '/tmp/astra-guard/baseline.cu',
  compare_kind: 'rmsnorm',
  baseline_module: 'sgl_kernel',
  baseline_func: 'fused_add_rmsnorm',
  generated_export_func: 'sgl_fused_add_rmsnorm',
  test_command: 'python /tmp/astra-guard/test.py {kernel_path} {result_path}',
  benchmark_command: 'python /tmp/astra-guard/bench.py {kernel_path} {result_path}',
  iterations: 1,
  target_gpu: 'H100',
  exp_dir: '/tmp/astra-guard',
  integration_mode: 'standalone',
}

function ds(iter) {
  return {
    [`driver-build-${iter}`]: { ok: true },
    [`driver-run-${iter}`]: { ok: true, latency_ms: 1, compiled: true, correct: true },
    [`driver-profile-${iter}`]: { ok: true, native_path: '/tmp/p.native' },
    [`driver-to-evidence-${iter}`]: { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
    [`driver-diagnose-${iter}`]: { bottleneck_class: 'memory_bound' },
    [`driver-anti-cheat-${iter}`]: { ok: true, suspicious: false },
  }
}

const minimalReturns = {
  'load-driver': { present: true, backend_id: 'triton', source_ext: '.py', aux_ext: '.py', lang_fence: 'python', impl_requirements: 'Provide a @triton.jit kernel.', methods: {}, hw_vendor: 'nvidia' },
  'setup-astra': { initial_kernel_code: 'k', kernel_summary: 's', entry_points: ['e'], integration_contract: 'c' },
  'prepare-tests': { test_cases: [{ shape: '[1]' }], correctness_tolerance: 't', harness_plan: 'h' },
  'profile-baseline': { measured: true, baseline_runtime_ms: 0.1, per_shape_runtime: [], bottlenecks: ['mem'] },
  'plan-0': { optimization_goal: 'g', target_regions: ['r'], proposed_changes: ['c'], correctness_risks: [] },
  'code-0': { candidate_code: 'k', changed_regions: [], implementation_notes: '' },
  'evaluate-0': { compiled: true, correct: true, speedup: 1.2, runtime_ms: 0.08, baseline_runtime_ms: 0.1 },
  ...ds(0),
  'record-0': { lessons: ['l'] },
  'post-process': { reintegration_notes: 'n', manual_review_items: [], limitations: [], rollback_criteria: [] },
  'final-report': { report_md: 'r' },
}

async function run(extra, agentReturns) {
  return capturePrompts({
    workflowPath: WORKFLOW,
    args: { ...baseArgs, ...extra },
    agentReturns: agentReturns || {},
  })
}

test('legacy path: no backend/no backend_dir -> renders CUDA vocabulary, no load-driver agent', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({}, legacyReturns)
  assert.ok(caps.length > 0)
  assert.ok(caps.some(c => /CUDA/.test(c.prompt)), 'legacy path should keep CUDA vocabulary')
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

test('intersectional guard: integration_mode=sglang + non-cuda driver -> throws naming both', async () => {
  await assert.rejects(
    run(
      {
        backend_dir: '_substrate/backends/triton',
        backend: 'triton',
        language: 'triton',
        integration_mode: 'sglang',
      },
    ),
    (err) => {
      assert.match(err.message, /integration_mode="sglang"/)
      assert.match(err.message, /backend_dir=_substrate\/backends\/triton/)
      assert.match(err.message, /CUDA driver/)
      return true
    },
  )
})

test('USE_DRIVER on with valid driver: load-driver fires first + per-iteration envelope agents present', async () => {
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
