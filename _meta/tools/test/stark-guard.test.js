'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'STARK/stark-kernel-optimization.js')

const baseArgs = {
  kernel_path: '/tmp/stark-guard/reference.cu',
  test_harness_path: '/tmp/stark-guard/test.py',
  iterations: 1,
  epsilon: 0.35,
  n_root: 5,
  n_child_max: 3,
  leaderboard_size: 5,
  compile_command: 'nvcc {kernel_path} -o {result_path}',
  benchmark_command: 'python /tmp/stark-guard/bench.py {kernel_path} {result_path}',
  exp_dir: '/tmp/stark-guard',
  target_gpu: 'H100',
  seed_candidates: 3,
  rng_seed: 42,
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
  'setup-read-reference': { kernel_code: 'k', op_type: 'rms', algorithm_description: 'd', launch_config: 'lc', optimization_assessment: 'oa', potential_directions: ['p'] },
  'setup-eval-root': { compile_ok: true, correct: true, runtime_ms: 0.1, logs: 'ok' },
  ...ds('root'),
  'plan-root-a1': { plan: 'p', anchored_scaffold: 'sc', anchors: [{ name: 'a', begin_line: 1, end_line: 2, description: 'd' }], rationale: 'r' },
  'code-root-a1': { kernel_code: 'k1', implementation_notes: 'n', anchor_resolutions: [] },
  'eval-a1': { compile_ok: true, correct: true, runtime_ms: 0.09, speedup_vs_baseline: 1.1, logs: 'ok', error_category: 'none' },
  ...ds(1),
  'report-final': { outcome: 'success', summary: 's', total_attempts: 1, correct_kernels: 2, failed_kernels: 0 },
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
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({}, legacyReturns)
  assert.ok(caps.length > 0)
  assert.ok(caps.some(c => /```cuda\b/.test(c.prompt)), 'legacy path should keep ```cuda code fence')
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

test('USE_DRIVER on with valid driver: load-driver fires first + per-attempt envelope agents present', async () => {
  const caps = await run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton', language: 'triton' },
    minimalReturns,
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver',
    'load-driver must be the first agent call when USE_DRIVER')
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-root`),
      `expected driver-${sub}-root envelope agent (root eval) under USE_DRIVER, got labels=${labels.join(',')}`)
    assert.ok(labels.includes(`driver-${sub}-1`),
      `expected driver-${sub}-1 envelope agent (attempt 1) under USE_DRIVER, got labels=${labels.join(',')}`)
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

// AWK #50: the Workflow runtime FORBIDS Math.random() (non-determinism breaks
// resume; the KerSor catalog forbidden-API scan marks any workflow containing
// it known_broken). STARK's selectNode() RNG must always be seeded.
test('§#50: workflow body contains no Math.random() — always-seeded RNG', () => {
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  assert.ok(!/Math\s*\.\s*random\s*\(/.test(src),
    'STARK must not call Math.random() — the KerSor catalog flags it known_broken (AWK #50). ' +
    'selectNode() rng() must be deterministically seeded.')
})
