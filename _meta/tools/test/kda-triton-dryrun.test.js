'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'KDA/kda-kernel-workflow.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'kda-args-triton.json'), 'utf8'))

function ds(candidateId) {
  return {
    [`driver-build-${candidateId}`]: { ok: true },
    [`driver-run-${candidateId}`]: { ok: true, latency_ms: 1, compiled: true, correct: true },
    [`driver-profile-${candidateId}`]: { ok: true, native_path: '/tmp/p.native' },
    [`driver-to-evidence-${candidateId}`]: { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
    [`driver-diagnose-${candidateId}`]: { bottleneck_class: 'memory_bound' },
    [`driver-anti-cheat-${candidateId}`]: { ok: true, suspicious: false },
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
  'inspect-workspace': {
    kernel_code: '@triton.jit\ndef k(): pass',
    key_functions: ['k'],
    current_approach: 'naive triton kernel',
  },
  'write-draft': {
    draft_content: 'Draft.',
    candidate_directions: [
      { title: 'Block tiling', approach: 'tune BLOCK', expected_value: '1.3x', risk: 'low' },
      { title: 'Vectorized loads', approach: 'tl.load with mask', expected_value: '1.2x', risk: 'low' },
    ],
  },
  'write-plan': {
    plan_content: 'Plan.',
    candidates: [
      { id: 'c1', title: 'Block tiling', changes: 'set BLOCK=128', expected_effect: '1.3x' },
      { id: 'c2', title: 'Vectorized loads', changes: 'tl.load masked', expected_effect: '1.2x' },
    ],
  },
  'impl-candidate-1': { code: '@triton.jit\ndef k(): pass' },
  'validate-candidate-1': { is_correct: false, correctness_issues: ['mismatch'], estimated_speedup: 0, validation_ran: true, addresses_goal: false },
  ...ds('candidate-1'),
  'impl-candidate-2': { code: '@triton.jit\ndef k(): pass' },
  'validate-candidate-2': { is_correct: true, estimated_speedup: 1.4, validation_ran: true, addresses_goal: true },
  ...ds('candidate-2'),
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

test('kda triton dry-run: USE_DRIVER on, no CUDA tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of CUDA_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `CUDA token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 300)}`)
    }
  }
})

test('kda triton dry-run: rendered driver-run command starts with _substrate/backends/triton/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('kda triton dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede inspect-workspace when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('kda triton dry-run: inspect-workspace prompt drops cuda-kernel-development binding under driver path', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const inspect = caps.find(c => c.label === 'inspect-workspace')
  assert.ok(inspect, 'inspect-workspace call must be present')
  assert.doesNotMatch(inspect.prompt, /cuda-kernel-development/,
    'inspect-workspace must not retain cuda-kernel-development skill binding under driver path')
})

test('kda triton dry-run: validate prompt drops ncu-report-skill + CUDA static-analysis hints', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const validateCalls = caps.filter(c => /^validate-candidate-/.test(c.label))
  assert.ok(validateCalls.length > 0, 'expected at least one validate-candidate-* call')
  for (const v of validateCalls) {
    assert.doesNotMatch(v.prompt, /ncu-report-skill/,
      `validate prompt must not retain ncu-report-skill under driver path: ${v.prompt.slice(0, 200)}`)
    assert.doesNotMatch(v.prompt, /warp shuffle logic/,
      'validate static-analysis hints must not retain warp-shuffle vocabulary')
  }
})
