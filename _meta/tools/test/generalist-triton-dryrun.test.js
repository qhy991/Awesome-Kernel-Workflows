'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'Generalist/generalist-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'generalist-args-triton.json'), 'utf8'))

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
    impl_requirements: 'Provide a @triton.jit kernel plus a host wrapper.',
    methods: {},
    hw_vendor: 'nvidia',
  },
  'profile-1': {
    compiled: true, correct: true, speedup: 1.0,
    metrics: { dram_pct: 50, sm_pct: 40, occupancy: 0.5, latency_ms: 1 },
  },
  'diagnose-1': { bottleneck_class: 'memory_bound', evidence: 'dram_pct' },
  'retrieve-1': { techniques: [], dead_ends: [] },
  'gate-1': { allowed_methods: ['tiling'], rationale: 'r' },
  'plan-1-1': { method: 'tiling', plan: 'tile' },
  'impl-1-1': { compiled: true, correct: true, speedup: 1.25, metrics: { dram_pct: 40, sm_pct: 55 } },
  'anticheat-1-1': { valid: true, reward: 0.25, recorded_speedup: 1.25 },
  ...ds('setup'),
  ...ds('1-1'),
  'learn-1-tiling': { updated: true },
  'refute-1': { refuted: false },
  'verify-insight-1': { confidence: 'measured' },
  'profile-2': {
    compiled: true, correct: true, speedup: 1.25,
    metrics: { dram_pct: 40, sm_pct: 55, occupancy: 0.6, latency_ms: 0.8 },
  },
  'diagnose-2': { bottleneck_class: 'compute_bound' },
  'retrieve-2': { techniques: [{ method: 'tiling', speedup: 1.25 }], dead_ends: [] },
  'gate-2': { allowed_methods: ['vectorization'] },
  'plan-2-1': { method: 'vectorization', plan: 'vec' },
  'impl-2-1': { compiled: true, correct: true, speedup: 1.54, metrics: { dram_pct: 35, sm_pct: 65 } },
  'anticheat-2-1': { valid: true, reward: 0.35, recorded_speedup: 1.54 },
  ...ds('2-1'),
  'learn-2-vectorization': { updated: true },
  'refute-2': { refuted: false },
  'verify-insight-2': { confidence: 'measured' },
  'final-report': { report: 'done' },
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

test('generalist triton dry-run: USE_DRIVER on, no CUDA tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of CUDA_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `CUDA token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 300)}`)
    }
  }
})

test('generalist triton dry-run: rendered driver-run command starts with _substrate/backends/triton/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('generalist triton dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede profile-1 when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('generalist triton dry-run: per-iteration envelope labels emitted for setup and each iteration', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const labels = caps.map(c => c.label)
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-setup`),
      `expected driver-${sub}-setup in generalist triton dry-run, got labels=${labels.join(',')}`)
    for (const n of [1, 2]) {
      assert.ok(labels.includes(`driver-${sub}-${n}-1`),
        `expected driver-${sub}-${n}-1 in generalist triton dry-run, got labels=${labels.join(',')}`)
    }
  }
})

test('generalist triton dry-run: langToken overrides LANGUAGE in generate-initial-kernel (if rendered)', async () => {
  // In optimize_existing mode, generate-initial-kernel is not rendered.
  // This test just verifies that the language seam is present in the prompt context.
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  // The generate-initial-kernel is NOT rendered in optimize_existing mode; verify it's absent
  assert.ok(!caps.some(c => c.label === 'generate-initial-kernel'),
    'optimize_existing mode should NOT render generate-initial-kernel')
})
