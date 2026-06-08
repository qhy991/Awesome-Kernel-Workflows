'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'KernelFoundryDx/kernelfoundrydx-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'kernelfoundrydx-args-triton.json'), 'utf8'))

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

const RETURNS = {
  'load-driver': {
    present: true,
    backend_id: 'triton',
    source_ext: '.py',
    aux_ext: '.py',
    lang_fence: 'python',
    impl_requirements: 'Provide a @triton.jit kernel plus a host wrapper that allocates outputs and launches the kernel.',
    methods: {},
    hw_vendor: 'nvidia',
  },
  'read-task': {
    task_text: 'class Model(torch.nn.Module): pass',
    op_type: 'matmul_fusion',
    op_chain: ['matmul', 'add', 'swish'],
    input_shapes: 'x[B,N,K] w[K,M] b[M]',
    fusion_opportunities: ['matmul+add'],
    numerical_notes: '',
  },
  'baseline-and-seed': {
    baseline_latency_ms: 1.25,
    baseline_available: true,
    initial_hints: [],
    notes: '',
  },
  'seed-0': { code: '@triton.jit\ndef k0(): pass\n', approach: 'a', retrieved_examples: [] },
  'seed-1': { code: '@triton.jit\ndef k1(): pass\n', approach: 'b', retrieved_examples: [] },
  'validate-seed-0': { cheating_likelihood: 0.1, is_genuine_triton: true, missing_ops: [], static_risk_notes: '' },
  'validate-seed-1': { cheating_likelihood: 0.2, is_genuine_triton: true, missing_ops: [], static_risk_notes: '' },
  'mutate-0-isl0': { code: '@triton.jit\ndef k0_v1(): pass\n', applied_hints: [], change_summary: 'fuse' },
  'mutate-0-isl1': { code: '@triton.jit\ndef k1_v1(): pass\n', applied_hints: [], change_summary: 'mask' },
  'eval-0-isl0': { compiles: true, runs: true, is_correct: true, latency_ms: 0.95, speedup: 1.31, launch_config: '', runtime_characterization: '', error_summary: '' },
  'eval-0-isl1': { compiles: true, runs: true, is_correct: true, latency_ms: 1.10, speedup: 1.13, launch_config: '', runtime_characterization: '', error_summary: '' },
  ...ds('0-isl0'),
  ...ds('0-isl1'),
  'diagnose-0-isl0': { diagnosis_type: 'performance', limiter: 'memory_bound', rationale: 'r', hint: { trigger: 't', bottleneck_class: 'memory_bound', suggestion: 's' }, cheating_likelihood: 0.05 },
  'diagnose-0-isl1': { diagnosis_type: 'performance', limiter: 'latency_bound', rationale: 'r', hint: { trigger: 't', bottleneck_class: 'latency_bound', suggestion: 's' }, cheating_likelihood: 0.05 },
  'final-report': { outcome: 'success', summary: 'driver dry-run report' },
}

const CUDA_TOKENS = [
  /\bnvcc\b/,
  /__global__/,
  /__syncthreads/,
  /PYBIND11_MODULE/,
  /cuda_runtime\.h/,
  /NCU Profile Results/,
  /```cuda\b/,
]

test('kernelfoundrydx triton dry-run: USE_DRIVER on, no CUDA tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of CUDA_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `CUDA token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 300)}`)
    }
  }
})

test('kernelfoundrydx triton dry-run: rendered driver-run command starts with _substrate/backends/triton/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('kernelfoundrydx triton dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede read-task when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('kernelfoundrydx triton dry-run: per-(iter,island) envelope labels emitted for both islands', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const labels = caps.map(c => c.label)
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    for (const isl of [0, 1]) {
      assert.ok(labels.includes(`driver-${sub}-0-isl${isl}`),
        `expected driver-${sub}-0-isl${isl} in kfoundrydx triton dry-run, got labels=${labels.join(',')}`)
    }
  }
})

test('kernelfoundrydx triton dry-run: setup-style prompts use driver lang_fence ("python") not "Triton"', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const baseline = caps.find(c => c.label === 'baseline-and-seed')
  assert.ok(baseline, 'baseline-and-seed call must be present')
  // langToken("Triton") returns DRIVER_LANG_FENCE under USE_DRIVER. Triton driver
  // declares lang_fence=python (see _substrate/backends/triton/idioms.json), so the
  // first-line "Triton kernel optimization run" becomes "python kernel optimization run".
  assert.match(baseline.prompt, /python kernel optimization run/,
    `baseline-and-seed prompt should use driver lang_fence: ${baseline.prompt.slice(0, 200)}`)
})
