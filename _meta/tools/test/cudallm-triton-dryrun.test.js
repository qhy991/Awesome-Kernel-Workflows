'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'CUDALLM/cudallm-fsr-kernel-generation.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'cudallm-args-triton.json'), 'utf8'))

function ds(label) {
  return {
    [`driver-build-${label}`]: { ok: true },
    [`driver-run-${label}`]: { ok: true, latency_ms: 1, compiled: true, correct: true },
    [`driver-profile-${label}`]: { ok: true, native_path: '/tmp/p.native' },
    [`driver-to-evidence-${label}`]: { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
    [`driver-diagnose-${label}`]: { bottleneck_class: 'memory_bound' },
    [`driver-anti-cheat-${label}`]: { ok: true, suspicious: false },
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
    feature_catalog: '# Required feature families (Triton)\n- block tiling\n- masked tl.load/tl.store\n- autotune configs',
  },
  'setup-task': { problem_definition: 'softmax', operation_type: 'softmax', constraints: [] },
  'feature-catalog': { features: [], baseline_feature_ids: [] },
  'generate-tests': { test_cases: [], tolerance_policy: 'r' },
  'select-features-0-0': { selected_feature_ids: ['t'], rationale: 'r' },
  'generate-kernel-0-0': { candidate_code: '@triton.jit\ndef k(): pass', implemented_feature_ids: ['t'] },
  'evaluate-0-0': { compiled: true, correct: true, speedup: 1.0 },
  'reinforce-0-0': { updated_scores: [] },
  ...ds('0-0'),
  'select-features-0-1': { selected_feature_ids: ['t'], rationale: 'r' },
  'generate-kernel-0-1': { candidate_code: '@triton.jit\ndef k(): pass', implemented_feature_ids: ['t'] },
  'evaluate-0-1': { compiled: true, correct: true, speedup: 1.0 },
  'reinforce-0-1': { updated_scores: [] },
  ...ds('0-1'),
  'select-features-1-0': { selected_feature_ids: ['t'], rationale: 'r' },
  'generate-kernel-1-0': { candidate_code: '@triton.jit\ndef k(): pass', implemented_feature_ids: ['t'] },
  'evaluate-1-0': { compiled: true, correct: true, speedup: 1.0 },
  'reinforce-1-0': { updated_scores: [] },
  ...ds('1-0'),
  'select-features-1-1': { selected_feature_ids: ['t'], rationale: 'r' },
  'generate-kernel-1-1': { candidate_code: '@triton.jit\ndef k(): pass', implemented_feature_ids: ['t'] },
  'evaluate-1-1': { compiled: true, correct: true, speedup: 1.0 },
  'reinforce-1-1': { updated_scores: [] },
  ...ds('1-1'),
  'final-report': { report_md: 'r' },
}

const CUDA_TOKENS = [
  /\bnvcc\b/,
  /\bncu\b/,
  /__global__/,
  /PYBIND11_MODULE/,
  /cuda_runtime\.h/,
  /NCU Profile Results/,
  /```cuda\b/,
]

test('triton dry-run: USE_DRIVER on, no CUDA tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of CUDA_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `CUDA token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 200)}`)
    }
  }
})

test('triton dry-run: rendered driver-run command starts with bash _substrate/backends/triton/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('triton dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede setup-task when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('triton dry-run: setup-task prompt uses driver lang_fence ("python kernel"), not "CUDA kernel"', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const setupCall = caps.find(c => c.label === 'setup-task')
  assert.ok(setupCall, 'setup-task call must be present')
  assert.match(setupCall.prompt, /python kernel generation expert/,
    `setup-task prompt must use driver lang_fence: ${setupCall.prompt.slice(0, 300)}`)
  assert.doesNotMatch(setupCall.prompt, /CUDA kernel generation expert/,
    'setup-task prompt must not retain CUDA vocabulary')
})

test('triton dry-run: evaluate prompt references workspace path with driver source_ext (.py), not .cu', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const evalCall = caps.find(c => c.label === 'evaluate-0-0')
  assert.ok(evalCall, 'evaluate-0-0 call must be present')
  assert.match(evalCall.prompt, /cudallm_iter_0_sample_0\.py/,
    `evaluate prompt should reference .py kernel filename under triton driver: ${evalCall.prompt.slice(0, 400)}`)
  assert.doesNotMatch(evalCall.prompt, /cudallm_iter_0_sample_0\.cu/,
    'evaluate prompt must not retain .cu kernel filename under triton driver')
})

test('triton dry-run: feature-catalog prompt uses driver-supplied feature_catalog (or fallback), not legacy CUDA list', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const catalogCall = caps.find(c => c.label === 'feature-catalog')
  assert.ok(catalogCall, 'feature-catalog call must be present')
  assert.doesNotMatch(catalogCall.prompt, /shared memory staging/,
    'feature-catalog must not leak the LEGACY_FEATURE_CATALOG block on the driver path')
  assert.doesNotMatch(catalogCall.prompt, /warp-level primitives/,
    'feature-catalog must not leak the LEGACY_FEATURE_CATALOG block on the driver path')
  assert.doesNotMatch(catalogCall.prompt, /CUDA intrinsics/,
    'feature-catalog must not leak CUDA intrinsics line on the driver path')
})
