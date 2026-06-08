'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'Astra/astra-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'astra-args-triton.json'), 'utf8'))

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
  'setup-astra': {
    initial_kernel_code: '@triton.jit\ndef k(): pass',
    kernel_summary: 'triton kernel',
    entry_points: ['k'],
    integration_contract: 'standalone',
  },
  'prepare-tests': {
    test_cases: [{ shape: '[1, 4096]' }],
    correctness_tolerance: 'rtol=1e-3',
    harness_plan: 'run test_command',
  },
  'profile-baseline': {
    measured: true,
    baseline_runtime_ms: 0.1,
    per_shape_runtime: [],
    bottlenecks: ['memory bandwidth'],
  },
  'plan-0': { optimization_goal: 'tile', target_regions: ['load'], proposed_changes: ['BLOCK=128'], correctness_risks: [] },
  'code-0': { candidate_code: '@triton.jit\ndef k(): pass', changed_regions: [], implementation_notes: '' },
  'evaluate-0': { compiled: true, correct: true, speedup: 1.2, runtime_ms: 0.08, baseline_runtime_ms: 0.1 },
  ...ds(0),
  'record-0': { lessons: ['tile size matters'] },
  'plan-1': { optimization_goal: 'reduce', target_regions: ['reduction'], proposed_changes: ['tl.sum'], correctness_risks: [] },
  'code-1': { candidate_code: '@triton.jit\ndef k(): pass', changed_regions: [], implementation_notes: '' },
  'evaluate-1': { compiled: true, correct: true, speedup: 1.3, runtime_ms: 0.07, baseline_runtime_ms: 0.1 },
  ...ds(1),
  'record-1': { lessons: ['reduction works'] },
  'post-process': { reintegration_notes: 'standalone', manual_review_items: [], limitations: [], rollback_criteria: [] },
  'final-report': { report_md: 'r' },
}

const CUDA_TOKENS = [
  /\bnvcc\b/,
  /\bncu\b/,
  /__global__/,
  /__syncthreads/,
  /PYBIND11_MODULE/,
  /cuda_runtime\.h/,
  /NCU Profile Results/,
  /```cuda\b/,
]

test('astra triton dry-run: USE_DRIVER on, no CUDA tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of CUDA_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `CUDA token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 300)}`)
    }
  }
})

test('astra triton dry-run: rendered driver-run command starts with _substrate/backends/triton/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('astra triton dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede setup-astra when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('astra triton dry-run: setup-astra prompt swaps PyBind/CUDA vocabulary for driver fence', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const setup = caps.find(c => c.label === 'setup-astra')
  assert.ok(setup, 'setup-astra call must be present')
  assert.doesNotMatch(setup.prompt, /PyBind\/CUDA/,
    'setup-astra must not retain PyBind/CUDA vocabulary under driver path')
})

test('astra triton dry-run: evaluate per-iteration kernel_path uses driver source_ext (.py)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const evalCalls = caps.filter(c => /^evaluate-/.test(c.label))
  assert.ok(evalCalls.length > 0, 'expected at least one evaluate-* call')
  for (const e of evalCalls) {
    assert.match(e.prompt, /astra_iter_\d+\.py/,
      `evaluate prompt should reference .py kernel path under triton driver: ${e.prompt.slice(0, 300)}`)
    assert.doesNotMatch(e.prompt, /astra_iter_\d+\.cu/,
      'evaluate prompt must not retain .cu kernel path under triton driver')
  }
})
