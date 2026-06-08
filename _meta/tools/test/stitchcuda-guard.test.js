'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'StitchCUDA/stitchcuda-kernel-optimization.js')

const baseArgs = {
  language: 'cuda',
  problem_type: 'cuda-kernel-generation',
  workspace: '/tmp/stitchcuda-guard',
}

function ds(attempt) {
  return {
    [`driver-build-${attempt}`]: { ok: true },
    [`driver-run-${attempt}`]: { ok: true, latency_ms: 1, compiled: true, correct: true },
    [`driver-profile-${attempt}`]: { ok: true, native_path: '/tmp/p.native' },
    [`driver-to-evidence-${attempt}`]: { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
    [`driver-diagnose-${attempt}`]: { bottleneck_class: 'memory_bound' },
    [`driver-anti-cheat-${attempt}`]: { ok: true, suspicious: false },
  }
}

const setupReturn = {
  cuda_version: '12.3',
  target_architecture: 'sm_90',
  pytorch_available: true,
  kernel_spec: { operation: 'gemm', shapes: '[M,N,K]', dtypes: ['float16'], baseline_gflops: 100.0 },
  kernelbench_config: { metrics: ['correctness'] },
  replan_heuristics: { compile_failure_threshold: 1, correctness_failure_threshold: 1, stagnation_iterations: 3 },
  max_attempts: 1,
}

const minimalReturns = {
  'load-driver': { present: true, backend_id: 'triton', source_ext: '.py', aux_ext: '.py', lang_fence: 'python', impl_requirements: 'Provide a @triton.jit kernel.', methods: {}, hw_vendor: 'nvidia' },
  'Setup StitchCUDA': setupReturn,
  'Plan attempt 1': { attempt: 1, plan_summary: 's', optimization_approach: 'balanced', key_strategies: [], implementation_steps: [], threading_config: {}, memory_strategy: 'm', expected_bottleneck: 'memory' },
  'Code attempt 1': { attempt: 1, kernel_code: 'k', host_code: 'h', kernel_name: 'kn', implementation_notes: 'n' },
  ...ds(0),
  'Verify attempt 1': { attempt: 1, compilation_success: true, compilation_errors: [], resource_usage: {}, correctness_passed: true, correctness_errors: [], max_error: 0.0, performance_gflops: 100.0, execution_time_ms: 1.0, speedup_vs_baseline: 1.0, kernelbench_score: 0.9, verification_passed: true, failure_reason: null },
  'Generate report': { summary: 's', total_attempts: 1, successful_attempts: 1, best_gflops: 100.0, speedup: 1.0, best_attempt: 1, report_path: '/tmp/r.md' },
}

async function run(extra, agentReturns) {
  return capturePrompts({
    workflowPath: WORKFLOW,
    args: { ...baseArgs, ...extra },
    agentReturns: agentReturns || {},
  })
}

test('legacy path: no backend_dir -> renders CUDA vocabulary, no load-driver agent', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  // remove driver-* (legacy path won't request them)
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

test('intersectional guard: kernelbench_config.benchmark_suite + non-cuda driver -> throws naming both', async () => {
  await assert.rejects(
    run(
      {
        backend_dir: '_substrate/backends/triton',
        backend: 'triton',
        language: 'triton',
        kernelbench_config: { benchmark_suite: 'level1' },
      },
    ),
    (err) => {
      assert.match(err.message, /kernelbench_config\.benchmark_suite="level1"/)
      assert.match(err.message, /backend_dir=_substrate\/backends\/triton/)
      assert.match(err.message, /CUDA driver/)
      return true
    },
  )
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
