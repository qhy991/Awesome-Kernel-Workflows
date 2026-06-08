'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'KernelBand/kernelband-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'kernelband-args-triton.json'), 'utf8'))

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
    kernel_code: '@triton.jit\ndef matmul_kernel(): pass\n',
    baseline_latency_us: 1000,
    hardware_signature: { dram_throughput_pct: 50, l2_throughput_pct: 30, sm_throughput_pct: 40, dominant_bottleneck: 'memory' },
    behavioral_features: { normalized_time: 1.0, registers_per_thread: 32, shared_mem_bytes: 0, block_dimension: 256, occupancy: 0.5 },
    platform_info: 'A100',
  },
  ...ds('setup'),
  'generate-t1-tiling': { optimized_kernel: '@triton.jit\ndef k_v1(): pass\n', changes_description: 'tiling', expected_improvement: '10%' },
  'eval-t1': { compiled: true, correct: true, latency_us: 800, speedup: 1.25 },
  ...ds('t1'),
  'generate-t2-vectorization': { optimized_kernel: '@triton.jit\ndef k_v2(): pass\n', changes_description: 'vectorization', expected_improvement: '5%' },
  'eval-t2': { compiled: true, correct: true, latency_us: 700, speedup: 1.43 },
  ...ds('t2'),
  'report': 'KernelBand optimization report.',
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

test('kernelband triton dry-run: USE_DRIVER on, no CUDA tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of CUDA_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `CUDA token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 300)}`)
    }
  }
})

test('kernelband triton dry-run: rendered driver-run command starts with _substrate/backends/triton/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('kernelband triton dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede setup when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('kernelband triton dry-run: gen/eval prompts use ```python fence (driver lang_fence)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const genCalls = caps.filter(c => /^generate-/.test(c.label))
  const evalCalls = caps.filter(c => /^eval-/.test(c.label))
  assert.ok(genCalls.length > 0, 'expected generate-* prompts')
  assert.ok(evalCalls.length > 0, 'expected eval-* prompts')
  for (const c of [...genCalls, ...evalCalls]) {
    assert.match(c.prompt, /```python\b/,
      `${c.label} should embed kernel under \`\`\`python fence under triton driver: ${c.prompt.slice(0, 200)}`)
  }
})

test('kernelband triton dry-run: per-iteration envelope labels emitted for setup and each iteration', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const labels = caps.map(c => c.label)
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-setup`),
      `expected driver-${sub}-setup in kernelband triton dry-run, got labels=${labels.join(',')}`)
    for (const n of [1, 2]) {
      assert.ok(labels.includes(`driver-${sub}-t${n}`),
        `expected driver-${sub}-t${n} in kernelband triton dry-run, got labels=${labels.join(',')}`)
    }
  }
})

test('kernelband triton dry-run: phi-gate uses LEGACY_SATURATION_THRESHOLD (0.75) when driver omits it', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const genPrompt = caps.find(c => c.label === 'generate-t1-tiling')
  assert.ok(genPrompt, 'generate-t1-tiling must exist')
  assert.ok(genPrompt.prompt.includes('75%'),
    'under triton driver without saturation_threshold, phi-gate should fall back to LEGACY 75%')
})
