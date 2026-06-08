'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'STARK/stark-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'stark-args-triton.json'), 'utf8'))

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
  'setup-read-reference': {
    kernel_code: '@triton.jit\ndef k(): pass',
    op_type: 'rmsnorm',
    algorithm_description: 'row-wise rms norm',
    launch_config: 'grid=(N/BLOCK,)',
    optimization_assessment: 'baseline',
    potential_directions: ['tile size'],
  },
  'setup-eval-root': { compile_ok: true, correct: true, runtime_ms: 0.1, logs: 'ok' },
  ...ds('root'),
  'plan-root-a1': { plan: 'tile', anchored_scaffold: '@triton.jit\ndef k(): pass', anchors: [{ name: 'a', begin_line: 1, end_line: 2, description: 'd' }], rationale: 'r' },
  'code-root-a1': { kernel_code: '@triton.jit\ndef k_v1(): pass', implementation_notes: 'n', anchor_resolutions: [] },
  'eval-a1': { compile_ok: true, correct: true, runtime_ms: 0.09, speedup_vs_baseline: 1.1, logs: 'ok', error_category: 'none' },
  ...ds(1),
  'plan-node_1-a2': { plan: 'reduce', anchored_scaffold: '@triton.jit\ndef k_v1(): pass', anchors: [{ name: 'b', begin_line: 1, end_line: 2, description: 'd' }], rationale: 'r' },
  'code-node_1-a2': { kernel_code: '@triton.jit\ndef k_v2(): pass', implementation_notes: 'n', anchor_resolutions: [] },
  'eval-a2': { compile_ok: true, correct: true, runtime_ms: 0.08, speedup_vs_baseline: 1.2, logs: 'ok', error_category: 'none' },
  ...ds(2),
  'report-final': { outcome: 'success', summary: 's', total_attempts: 2, correct_kernels: 3, failed_kernels: 0 },
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

test('stark triton dry-run: USE_DRIVER on, no CUDA tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of CUDA_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `CUDA token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 300)}`)
    }
  }
})

test('stark triton dry-run: rendered driver-run command starts with _substrate/backends/triton/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('stark triton dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede setup-read-reference when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('stark triton dry-run: plan/code/debug context builders use ```python fence (driver lang_fence)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  // At least one plan-* and one code-* must include the python fence somewhere
  // (the context builders embed kernel code under ```<fence>).
  const planCalls = caps.filter(c => /^plan-/.test(c.label))
  const codeCalls = caps.filter(c => /^code-/.test(c.label))
  assert.ok(planCalls.length > 0, 'expected plan-* prompts')
  assert.ok(codeCalls.length > 0, 'expected code-* prompts')
  for (const c of [...planCalls, ...codeCalls]) {
    assert.match(c.prompt, /```python\b/,
      `${c.label} should embed kernel under \`\`\`python fence under triton driver: ${c.prompt.slice(0, 200)}`)
  }
})

test('stark triton dry-run: per-attempt envelope labels emitted for both root and each iteration', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const labels = caps.map(c => c.label)
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-root`),
      `expected driver-${sub}-root in triton dry-run, got labels=${labels.join(',')}`)
    for (const n of [1, 2]) {
      assert.ok(labels.includes(`driver-${sub}-${n}`),
        `expected driver-${sub}-${n} in triton dry-run, got labels=${labels.join(',')}`)
    }
  }
})
