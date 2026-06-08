'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'KernelFoundry/kernelfoundry-kernel-optimization.js')

const baseArgs = {
  problem_definition: 'class Model(nn.Module):\n    def forward(self, x):\n        return torch.softmax(x, dim=-1)',
  op_description: 'softmax',
  target_gpu: 'NVIDIA H100',
  generations: 1,
  meta_prompt_interval: 99,
  exp_dir: '/tmp/kf-guard',
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
  'setup': { operator_code: 'op', operator_type: 'softmax', input_shapes: '[B,N]', baseline_time_ms: 0.5, hardware_info: 'H100', feasible_cells: [] },
  'vary-0': { kernel_code: '__global__ void k(){}', strategy_description: 's', memory_pattern: 'm', algorithm_type: 'a', parallelism_level: 'p', is_templated: false, template_params: [] },
  'eval-0': { compiled: true, correct: true, speedup: 1.2, kernel_time_ms: 0.4, d_mem: 1, d_algo: 1, d_sync: 1, error_message: '', performance_notes: 'p' },
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

test('legacy path: language=sycl is accepted (multi-language, no driver required)', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({ language: 'sycl' }, legacyReturns)
  assert.ok(caps.some(c => /```sycl\b/.test(c.prompt)),
    'legacy path with language=sycl must keep SYCL fence — KernelFoundry is multi-language')
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
  // Use only args.backend (no language) so the §6.4 backend-vs-language guard
  // does not preempt the manifest-vs-axis mismatch branch.
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

test('USE_DRIVER on with valid cuda driver: load-driver fires first + per-generation envelope agents present', async () => {
  const caps = await run(
    { backend_dir: '_substrate/backends/cuda', backend: 'cuda', language: 'cuda' },
    minimalReturns,
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver',
    'load-driver must be the first agent call when USE_DRIVER')
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-0`),
      `expected driver-${sub}-0 envelope agent (generation 0) under USE_DRIVER, got labels=${labels.join(',')}`)
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

test('SYCL deferred-driver: args.backend=sycl with empty backend_dir -> §6.4 implicit-resolve guard throws', async () => {
  // SYCL has no _substrate/backends/sycl/ as of P5d. The §6.4 guard catches
  // any args.backend without args.backend_dir before driver dispatch.
  await assert.rejects(
    run({ backend: 'sycl' }),
    (err) => {
      assert.match(err.message, /requires args\.backend_dir/)
      return true
    },
  )
})
