'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'ReGraphT/regrapht-kernel-optimization.js')

const baseArgs = {
  kernel_path: '/tmp/regrapht-guard/source.cu',
  op_description: 'stencil',
  language: 'cuda',
  target_gpu: 'H100',
  iterations: 1,
  rollouts_per_select: 4,
  max_path_length: 4,
  benchmark_command: 'python /tmp/regrapht-guard/bench.py {kernel_path} {result_path}',
  exp_dir: '/tmp/regrapht-guard',
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
  'load-driver': { present: true, backend_id: 'cuda', source_ext: '.cu', aux_ext: '.cu', lang_fence: 'cuda', impl_requirements: 'Provide a __global__ kernel.', methods: {}, hw_vendor: 'nvidia' },
  'setup-task': { source_code: '__global__ void k() {}', operation_type: 'stencil', input_contract: 'x', output_contract: 'y', kernel_entry_points: ['k'], baseline_metric: 1.0, baseline_time_ms: 0.2, evaluator_contract: 'json', optimization_dimensions: ['tiling'] },
  'build-regraph': { graph: { nodes: [{ id: 'v_init', method: 'init', visits: 0, reward: 0 }], edges: [] }, method_labels: ['tile'], trace_count: 0 },
  ...ds('root'),
  'select-0': { path_node_ids: ['v_init'], method_sequence: ['tile'], selected_examples: [], selection_rationale: 'r' },
  'generate-0': { candidate_code: '__global__ void k_v1() {}', applied_methods: ['tile'], skipped_methods: [] },
  'evaluate-0': { compiled: true, correct: true, speedup: 1.1, kernel_time_ms: 0.18, baseline_time_ms: 0.2, error_message: '', error_type: '', evidence_summary: 's' },
  ...ds('0'),
  'update-graph-0': { updated_graph: { nodes: [{ id: 'v_init', method: 'init', visits: 1, reward: 1.1 }], edges: [] }, reward: 1.1, added_nodes: [], added_edges: [] },
  'final-report': { outcome: 'success', summary: 's', best_speedup: 1.1, baseline_metric: 1.0, speedup: 1.1, iterations: 1 },
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
  assert.ok(caps.some(c => /```cuda\b/.test(c.prompt)), 'legacy path with language=cuda should render ```cuda code fence')
  assert.ok(!caps.some(c => c.label === 'load-driver'),
    'legacy path must NOT issue load-driver agent (USE_DRIVER off-by-default)')
})

test('§6.4: args.backend matches manifest backend_id -> ok', async () => {
  await assert.doesNotReject(run(
    { backend_dir: '_substrate/backends/cuda', backend: 'cuda', language: 'cuda' },
    minimalReturns,
  ))
})

test('§6.4: conflicting backend + language -> throws naming both', async () => {
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

test('§6.4: args.backend without backend_dir -> throws (no implicit-resolve)', async () => {
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

test('USE_DRIVER on with valid driver: load-driver fires first + per-eval envelope agents present', async () => {
  const caps = await run(
    { backend_dir: '_substrate/backends/cuda', backend: 'cuda', language: 'cuda' },
    minimalReturns,
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver',
    'load-driver must be the first agent call when USE_DRIVER')
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-root`),
      `expected driver-${sub}-root envelope agent (root) under USE_DRIVER, got labels=${labels.join(',')}`)
    assert.ok(labels.includes(`driver-${sub}-0`),
      `expected driver-${sub}-0 envelope agent (attempt 0) under USE_DRIVER, got labels=${labels.join(',')}`)
  }
})

test('manifest backend_id mismatch with args.backend -> throws', async () => {
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
