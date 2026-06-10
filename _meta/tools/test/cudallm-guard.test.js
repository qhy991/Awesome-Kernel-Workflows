'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'CUDALLM/cudallm-fsr-kernel-generation.js')

const baseArgs = {
  problem_definition: 'Implement softmax over Tensor[B,N] in pure CUDA',
  benchmark_command: 'python /tmp/cudallm-guard/bench.py {kernel_path} {result_path}',
  iterations: 1,
  feature_budget: 2,
  samples_per_feature_set: 1,
  exp_dir: '/tmp/cudallm-guard',
}

const minimalReturns = {
  'load-driver': { present: true, backend_id: 'triton', source_ext: '.py', aux_ext: '.py', lang_fence: 'python' },
  'setup-task': { problem_definition: 'p', operation_type: 'softmax', constraints: [] },
  'feature-catalog': { features: [], baseline_feature_ids: [] },
  'generate-tests': { test_cases: [], tolerance_policy: 'r' },
  'select-features-0-0': { selected_feature_ids: ['x'], rationale: 'r' },
  'generate-kernel-0-0': { candidate_code: 'k', implemented_feature_ids: ['x'] },
  'evaluate-0-0': { compiled: true, correct: true, speedup: 1.0 },
  'reinforce-0-0': { updated_scores: [] },
  'driver-build-0-0': { ok: true },
  'driver-run-0-0': { ok: true, latency_ms: 1, compiled: true, correct: true },
  'driver-profile-0-0': {
    ok: true,
    profiler: 'proton',
    native_profile: '/tmp/p.native',
    format: 'proton-hatchet',
  },
  'driver-to-evidence-0-0': { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
  'driver-diagnose-0-0': { bottleneck_class: 'memory_bound' },
  'driver-anti-cheat-0-0': { ok: true, suspicious: false },
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
    'setup-task': { problem_definition: 'p', operation_type: 'softmax', constraints: [] },
    'feature-catalog': { features: [], baseline_feature_ids: [] },
    'generate-tests': { test_cases: [], tolerance_policy: 'r' },
    'select-features-0-0': { selected_feature_ids: ['x'], rationale: 'r' },
    'generate-kernel-0-0': { candidate_code: 'k', implemented_feature_ids: ['x'] },
    'evaluate-0-0': { compiled: true, correct: true, speedup: 1.0 },
    'reinforce-0-0': { updated_scores: [] },
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

test('USE_DRIVER on with backend_dir: load-driver agent fires before setup-task', async () => {
  const driverReturn = {
    present: true,
    backend_id: 'cuda',
    source_ext: '.cu',
    aux_ext: '.cpp',
    lang_fence: 'cuda',
    impl_requirements: 'Provide a __global__ kernel plus a host launcher.',
    methods: {},
    hw_vendor: 'nvidia',
    feature_catalog: '# Required feature families\n- shared memory tiling\n- warp shuffle',
  }
  const caps = await run(
    { backend_dir: '_substrate/backends/cuda', backend: 'cuda' },
    { ...minimalReturns, 'load-driver': driverReturn },
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver', 'load-driver must be the first agent call when USE_DRIVER')
  assert.ok(labels.includes('driver-build-0-0'))
  assert.ok(labels.includes('driver-run-0-0'))
  assert.ok(labels.includes('driver-profile-0-0'))
  assert.ok(labels.includes('driver-to-evidence-0-0'))
  assert.ok(labels.includes('driver-diagnose-0-0'))
  assert.ok(labels.includes('driver-anti-cheat-0-0'))
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

test('cuda driver: to_evidence uses nsys-sqlite when profile pointer says nsys', async () => {
  const driverReturn = {
    present: true,
    backend_id: 'cuda',
    source_ext: '.cu',
    lang_fence: 'cuda',
    impl_requirements: 'Provide a __global__ kernel + pybind11 host launcher.',
    methods: {},
    hw_vendor: 'nvidia',
    profiler_name: 'ncu',
    profiler_format: 'ncu-csv',
  }
  const caps = await run(
    {
      backend_dir: '_substrate/backends/cuda',
      backend: 'cuda',
      reference_code_path: '/tmp/cudallm-guard/bench.py',
    },
    {
      ...minimalReturns,
      'load-driver': driverReturn,
      'driver-profile-0-0': {
        ok: true,
        profiler: 'nsys',
        native_profile: '/tmp/cudallm-guard/cudallm_iter_0_sample_0.prof.sqlite',
        format: 'nsys-sqlite',
      },
    },
  )
  const profileCall = caps.find(c => c.label === 'driver-profile-0-0')
  assert.ok(profileCall, 'driver-profile-0-0 must be present')
  assert.match(profileCall.prompt, /--out \/tmp\/cudallm-guard\/cudallm_iter_0_sample_0\.prof\.sqlite/,
    'profile.sh --out must use .sqlite suffix for nsys fallback compatibility')
  assert.match(profileCall.prompt, /--source \/tmp\/cudallm-guard\/bench\.py/,
    'profile.sh should pass reference_code_path as --source for nsys launcher')
  const evidenceCall = caps.find(c => c.label === 'driver-to-evidence-0-0')
  assert.ok(evidenceCall, 'driver-to-evidence-0-0 must be present')
  assert.match(evidenceCall.prompt, /--format nsys-sqlite/,
    `to_evidence must use format from profile pointer: ${evidenceCall.prompt.slice(0, 300)}`)
})

test('cuda driver: to_evidence degrades gracefully when profile pointer ok=false', async () => {
  const driverReturn = {
    present: true,
    backend_id: 'cuda',
    source_ext: '.cu',
    lang_fence: 'cuda',
    impl_requirements: 'Provide a __global__ kernel + pybind11 host launcher.',
    methods: {},
  }
  const caps = await run(
    { backend_dir: '_substrate/backends/cuda', backend: 'cuda' },
    {
      ...minimalReturns,
      'load-driver': driverReturn,
      'driver-profile-0-0': {
        ok: false,
        profiler: 'nsys',
        native_profile: null,
        error: 'neither ncu nor nsys profiler available',
      },
    },
  )
  const evidenceCall = caps.find(c => c.label === 'driver-to-evidence-0-0')
  assert.ok(evidenceCall, 'driver-to-evidence-0-0 must be present')
  assert.match(evidenceCall.prompt, /Profiler unavailable/,
    'to_evidence step should skip parsing when profile.sh failed')
})

test('driver-injected substrings: driver lang_fence reaches setup-task prompt', async () => {
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
    { ...minimalReturns, 'load-driver': driverReturn },
  )
  const setupCall = caps.find(c => c.label === 'setup-task')
  assert.ok(setupCall, 'setup-task agent call must be present')
  assert.match(setupCall.prompt, /cuda kernel generation expert/,
    `setup-task prompt should contain driver lang_fence "cuda kernel generation expert"; got: ${setupCall.prompt.slice(0, 200)}`)
})
