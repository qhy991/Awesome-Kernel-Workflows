'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'KernelAgent/kernelagent-triton-synthesis.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'kernelagent-args-cuda.json'), 'utf8'))

const RETURNS = {
  'load-driver': {
    present: true,
    backend_id: 'cuda',
    source_ext: '.cu',
    aux_ext: '.cpp',
    lang_fence: 'cuda',
    impl_requirements: 'Provide a __global__ CUDA kernel plus a host launcher that invokes it via a pybind11 binding and returns the output tensor.',
    methods: {},
    hw_vendor: 'nvidia',
  },
  'setup-problem': { problem_definition: 'softmax', input_tensors: [], operations: ['softmax'], output_spec: {shape:'',dtype:''}, complexity_signals: [] },
  'setup-test': { test_code: 'import torch\nprint("PASS")\n' },
  'route-analysis': { path: 'direct', reason: 'simple', subgraph_count: 1, estimated_difficulty: 'easy' },
  'gen-main-seed0': { kernel_code: '__global__ void softmax_kernel() {}', approach: 'baseline', potential_issues: '' },
  'gen-main-seed1': { kernel_code: '__global__ void softmax_kernel() { /*shared*/ }', approach: 'shared-mem', potential_issues: '' },
  'verify-candidate_0': { passed: false, exit_code: 1, stdout: '', stderr: 'compile err', error_summary: 'compile', verification_result: 'fail' },
  'verify-candidate_1': { passed: false, exit_code: 1, stdout: '', stderr: 'compile err', error_summary: 'compile', verification_result: 'fail' },
  'driver-build-candidate_0': { ok: false },
  'driver-run-candidate_0': { ok: false, latency_ms: 0, compiled: false, correct: false },
  'driver-profile-candidate_0': { ok: false },
  'driver-to-evidence-candidate_0': { ok: false, metrics: {}, coverage: [] },
  'driver-diagnose-candidate_0': { bottleneck_class: 'unknown' },
  'driver-anti-cheat-candidate_0': { ok: true, suspicious: false },
  'driver-build-candidate_1': { ok: false },
  'driver-run-candidate_1': { ok: false, latency_ms: 0, compiled: false, correct: false },
  'driver-profile-candidate_1': { ok: false },
  'driver-to-evidence-candidate_1': { ok: false, metrics: {}, coverage: [] },
  'driver-diagnose-candidate_1': { bottleneck_class: 'unknown' },
  'driver-anti-cheat-candidate_1': { ok: true, suspicious: false },
  'refine-candidate_0-r1': { kernel_code: '__global__ void k() {}', changes_made: ['fix'], confidence: 'high', fix_explanation: 'fixed' },
  'refine-candidate_1-r1': { kernel_code: '__global__ void k() {}', changes_made: ['fix'], confidence: 'high', fix_explanation: 'fixed' },
  'reverify-candidate_0-r1': { passed: true, exit_code: 0, stdout: 'PASS', stderr: '', error_summary: '', verification_result: 'pass' },
  'reverify-candidate_1-r1': { passed: false, exit_code: 1, stdout: '', stderr: 'still', error_summary: 'still failing', verification_result: 'fail' },
  'report-summary': { outcome: 'success', summary: 'driver-path cuda dry-run report' },
}

const TRITON_TOKENS = [
  /@triton\.jit/,
  /triton\.language/,
  /\bimport triton\b/,
  /tl\.program_id/,
  /tl\.constexpr/,
]

test('cuda dry-run: USE_DRIVER on, no Triton tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of TRITON_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `Triton token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 200)}`)
    }
  }
})

test('cuda dry-run: rendered driver-run command starts with bash _substrate/backends/cuda/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/cuda\/run\.sh/,
      `driver-run prompt should reference cuda driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('cuda dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede setup-problem when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/cuda/,
    'load-driver prompt must reference the cuda backend_dir')
})

test('cuda dry-run: setup-problem prompt uses driver lang_fence ("cuda kernel"), not "Triton kernel"', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const setupCall = caps.find(c => c.label === 'setup-problem')
  assert.ok(setupCall, 'setup-problem call must be present')
  assert.match(setupCall.prompt, /cuda kernel synthesis expert/,
    `setup-problem prompt must use driver lang_fence: ${setupCall.prompt.slice(0, 300)}`)
  assert.doesNotMatch(setupCall.prompt, /Triton kernel synthesis expert/,
    'setup-problem prompt must not retain Triton vocabulary')
})

test('cuda dry-run: verify prompt references driver-supplied kernel/test filenames (.cu/.cpp)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const verifyCall = caps.find(c => c.label === 'verify-candidate_0')
  assert.ok(verifyCall, 'verify-candidate_0 call must be present')
  assert.match(verifyCall.prompt, /kernel\.cu/,
    `verify prompt should reference kernel.cu under cuda driver: ${verifyCall.prompt.slice(0, 400)}`)
  assert.match(verifyCall.prompt, /test_kernel\.cpp/,
    `verify prompt should reference test_kernel.cpp under cuda driver`)
})
