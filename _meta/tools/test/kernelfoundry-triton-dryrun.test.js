'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'KernelFoundry/kernelfoundry-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'kernelfoundry-args-triton.json'), 'utf8'))

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
  'setup': {
    operator_code: 'def softmax(x): ...',
    operator_type: 'softmax',
    input_shapes: '[B,N]',
    baseline_time_ms: 0.5,
    hardware_info: 'NVIDIA H100',
    feasible_cells: ['1,1,1'],
  },
  'vary-0': {
    kernel_code: '@triton.jit\ndef k(): pass\n',
    strategy_description: 'tile + vectorized loads',
    memory_pattern: 'coalesced',
    algorithm_type: 'fused',
    parallelism_level: 'subgroup',
    is_templated: false,
    template_params: [],
  },
  'eval-0': {
    compiled: true, correct: true, speedup: 1.4, kernel_time_ms: 0.35,
    d_mem: 1, d_algo: 1, d_sync: 1,
    error_message: '', performance_notes: 'mem-bound',
  },
  ...ds('0'),
  'vary-1': {
    kernel_code: '@triton.jit\ndef k2(): pass\n',
    strategy_description: 'online softmax',
    memory_pattern: 'shared',
    algorithm_type: 'reformulated',
    parallelism_level: 'subgroup',
    is_templated: true,
    template_params: ['BLOCK_SIZE'],
  },
  'eval-1': {
    compiled: true, correct: true, speedup: 1.8, kernel_time_ms: 0.27,
    d_mem: 2, d_algo: 2, d_sync: 2,
    error_message: '', performance_notes: 'reduction tail amortized',
  },
  ...ds('1'),
  'final-report': 'KernelFoundry triton dry-run report.',
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

test('kernelfoundry triton dry-run: USE_DRIVER on, no CUDA tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of CUDA_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `CUDA token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 300)}`)
    }
  }
})

test('kernelfoundry triton dry-run: rendered driver-run command starts with _substrate/backends/triton/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('kernelfoundry triton dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede setup when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('kernelfoundry triton dry-run: vary/eval prompts use ```python fence (driver lang_fence) and never ```triton or ```cuda', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const varyCalls = caps.filter(c => /^vary-/.test(c.label))
  const evalCalls = caps.filter(c => /^eval-/.test(c.label))
  assert.ok(varyCalls.length > 0, 'expected vary-* prompts')
  assert.ok(evalCalls.length > 0, 'expected eval-* prompts')
  for (const c of [...varyCalls, ...evalCalls]) {
    assert.ok(!/```triton\b/.test(c.prompt),
      `${c.label} must not embed kernel under \`\`\`triton fence under driver (driver supplies fence=python): ${c.prompt.slice(0, 200)}`)
    assert.ok(!/```cuda\b/.test(c.prompt),
      `${c.label} must not embed kernel under \`\`\`cuda fence under triton driver: ${c.prompt.slice(0, 200)}`)
  }
})

test('kernelfoundry triton dry-run: per-generation envelope labels emitted for each generation', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const labels = caps.map(c => c.label)
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    for (const n of [0, 1]) {
      assert.ok(labels.includes(`driver-${sub}-${n}`),
        `expected driver-${sub}-${n} in kernelfoundry triton dry-run, got labels=${labels.join(',')}`)
    }
  }
})

test('kernelfoundry triton dry-run: DEFERRED — Layer-A envelope is rendered (dry-run only); real GPU verification deferred per docs/superpowers/specs', () => {
  // This marker test records that the triton driver-dispatch path is exercised
  // at the prompt-rendering layer only — no GPU runs in CI. Promoted to
  // measured evidence when the GPU-deferred verification pipeline lands per
  // _meta/tools/test/DEFERRED-GPU-VERIFICATION.md. See KSearch/KernelFoundryDx
  // triton-dryrun tests for the sibling pattern.
  assert.ok(true, 'placeholder for the deferred-GPU evidence row')
})
