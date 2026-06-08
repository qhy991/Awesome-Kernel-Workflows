'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'KernelFoundryDx/kernelfoundrydx-kernel-optimization.js')

const baseArgs = {
  problem_path: '/tmp/kfdx-guard/spec.py',
  problem_definition: 'Implement softmax over Tensor[B,N] in pure Triton',
  islands: 1,
  iterations: 1,
  seed_candidates: 1,
  population_size: 1,
  benchmark_command: '',
  test_command: '',
  exp_dir: '/tmp/kfdx-guard',
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
    run({ backend: 'triton' }),
    (err) => {
      assert.match(err.message, /requires args\.backend_dir/)
      return true
    },
  )
})

test('triton-only gate: args.backend="cuda" with backend_dir -> throws naming the triton-only restriction', async () => {
  await assert.rejects(
    run({ backend: 'cuda', backend_dir: '_substrate/backends/cuda' }),
    (err) => {
      assert.match(err.message, /supports only backend="triton"/)
      assert.match(err.message, /got "cuda"/)
      return true
    },
  )
})

test('triton-only gate: args.language="cuda" alone -> throws naming the triton-only restriction (legacy path)', async () => {
  // args.language='cuda' without backend_dir: resolveBackendAxis runs first
  // (module body), throws before the legacy suitability check.
  await assert.rejects(
    run({ language: 'cuda' }),
    (err) => {
      assert.match(err.message, /supports only backend="triton"/)
      return true
    },
  )
})

test('alias normalization: hip -> rocm rejected by triton-only gate', async () => {
  await assert.rejects(
    run({ backend: 'hip', backend_dir: '_substrate/backends/rocm' }),
    (err) => {
      assert.match(err.message, /supports only backend="triton"/)
      assert.match(err.message, /got "rocm"/)
      return true
    },
  )
})

test('§6.4: backend + language equal post-normalize -> no throw (with backend_dir)', async () => {
  await assert.doesNotReject(run(
    { backend: 'triton', language: 'triton', backend_dir: '_substrate/backends/triton' },
    {
      'load-driver': { present: true, backend_id: 'triton', source_ext: '.py', aux_ext: '.py', lang_fence: 'python' },
      'read-task': { task_text: 't', op_type: 'x', op_chain: ['x'], input_shapes: '', fusion_opportunities: [], numerical_notes: '' },
      'baseline-and-seed': { baseline_latency_ms: 1.0, baseline_available: true, initial_hints: [], notes: '' },
      'seed-0': { code: 'pass', approach: 'a' },
      'validate-seed-0': { cheating_likelihood: 0.1, is_genuine_triton: true, missing_ops: [], static_risk_notes: '' },
      'mutate-0-isl0': { code: 'pass', applied_hints: [], change_summary: '' },
      'eval-0-isl0': { compiles: true, runs: true, is_correct: true, latency_ms: 1, speedup: 1, launch_config: '', runtime_characterization: '', error_summary: '' },
      'driver-build-0-isl0': { ok: true },
      'driver-run-0-isl0': { ok: true, latency_ms: 1, compiled: true, correct: true },
      'driver-profile-0-isl0': { ok: true, native_path: '/tmp/p.native' },
      'driver-to-evidence-0-isl0': { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
      'driver-diagnose-0-isl0': { bottleneck_class: 'memory_bound' },
      'driver-anti-cheat-0-isl0': { ok: true, suspicious: false },
      'diagnose-0-isl0': { diagnosis_type: 'performance', limiter: 'memory_bound', rationale: 'r', hint: { trigger: 't', bottleneck_class: 'memory_bound', suggestion: 's' }, cheating_likelihood: 0.1 },
      'final-report': { outcome: 'success', summary: 's' },
    },
  ))
})

test('manifest backend_id mismatch with triton-only -> throws (cuda driver rejected)', async () => {
  const driverReturn = {
    present: true,
    backend_id: 'cuda',
    source_ext: '.cu',
    lang_fence: 'cuda',
  }
  await assert.rejects(
    run(
      { backend_dir: '_substrate/backends/cuda' },
      { 'load-driver': driverReturn },
    ),
    (err) => {
      assert.match(err.message, /backend_id="cuda"/)
      assert.match(err.message, /triton-only/)
      return true
    },
  )
})

test('USE_DRIVER on with triton driver: load-driver agent fires before read-task', async () => {
  const driverReturn = {
    present: true,
    backend_id: 'triton',
    source_ext: '.py',
    aux_ext: '.py',
    lang_fence: 'python',
    impl_requirements: 'Provide a @triton.jit kernel plus a host wrapper.',
    methods: {},
    hw_vendor: 'nvidia',
  }
  const caps = await run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton' },
    {
      'load-driver': driverReturn,
      'read-task': { task_text: 't', op_type: 'x', op_chain: ['x'], input_shapes: '', fusion_opportunities: [], numerical_notes: '' },
      'baseline-and-seed': { baseline_latency_ms: 1.0, baseline_available: true, initial_hints: [], notes: '' },
      'seed-0': { code: 'pass', approach: 'a' },
      'validate-seed-0': { cheating_likelihood: 0.1, is_genuine_triton: true, missing_ops: [], static_risk_notes: '' },
      'mutate-0-isl0': { code: 'pass', applied_hints: [], change_summary: '' },
      'eval-0-isl0': { compiles: true, runs: true, is_correct: true, latency_ms: 1, speedup: 1, launch_config: '', runtime_characterization: '', error_summary: '' },
      'driver-build-0-isl0': { ok: true },
      'driver-run-0-isl0': { ok: true, latency_ms: 1, compiled: true, correct: true },
      'driver-profile-0-isl0': { ok: true, native_path: '/tmp/p.native' },
      'driver-to-evidence-0-isl0': { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
      'driver-diagnose-0-isl0': { bottleneck_class: 'memory_bound' },
      'driver-anti-cheat-0-isl0': { ok: true, suspicious: false },
      'diagnose-0-isl0': { diagnosis_type: 'performance', limiter: 'memory_bound', rationale: 'r', hint: { trigger: 't', bottleneck_class: 'memory_bound', suggestion: 's' }, cheating_likelihood: 0.1 },
      'final-report': { outcome: 'success', summary: 's' },
    },
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver', 'load-driver must be the first agent call when USE_DRIVER')
  assert.ok(labels.includes('driver-build-0-isl0'))
  assert.ok(labels.includes('driver-run-0-isl0'))
  assert.ok(labels.includes('driver-profile-0-isl0'))
  assert.ok(labels.includes('driver-to-evidence-0-isl0'))
  assert.ok(labels.includes('driver-diagnose-0-isl0'))
  assert.ok(labels.includes('driver-anti-cheat-0-isl0'))
})
