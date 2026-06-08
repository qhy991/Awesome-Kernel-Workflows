'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'AdaExplore/adaexplore-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'adaexplore-args-triton.json'), 'utf8'))

// agent-returns map for the driver path: 1 iteration ample to exercise the
// large-step (forced at root) branch + the full Layer-A envelope.
const RETURNS = {
  'load-driver': {
    present: true,
    backend_id: 'triton',
    source_ext: '.py',
    lang_fence: 'python',
    impl_requirements: 'Provide a @triton.jit kernel plus a plain Python launcher that computes the grid and calls kernel[grid](...). No PYBIND11; the launcher returns the output tensor directly.',
    methods: {
      vectorized_load_store: {
        idiom: 'block-shaped tl.load / tl.store over contiguous offsets',
        prompt_guidance: 'Widen tl.load/tl.store blocks to power-of-two BLOCK sizes for vectorized memory access.',
      },
    },
    hw_vendor: 'nvidia',
  },
  'setup': {
    operator_code: 'import torch\nclass Model(torch.nn.Module):\n  def forward(self, x):\n    return torch.relu(x)\n',
    evaluator_command: '',
    baseline_time_ms: 0.5,
    hardware_info: 'NVIDIA A100',
    initial_skill_memory: [],
    reference_path: '/tmp/adaexplore-fixture/reference.py',
  },
  'propose-1': {
    kernel_code: 'import triton\nimport triton.language as tl\n@triton.jit\ndef relu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):\n  pass\n',
    strategy: '1D tiling',
    novelty_vs_pool: 'first',
  },
  'eval-1': { compiled: true, correct: true, speedup: 1.2, kernel_time_ms: 0.4, baseline_time_ms: 0.48, error_message: '', error_type: '' },
  'driver-build-1': { ok: true, artifact: '/tmp/artifact' },
  'driver-run-1': { ok: true, latency_ms: 0.4, compiled: true, correct: true, log: '' },
  'driver-profile-1': { ok: true, native_path: '/tmp/p.native' },
  'driver-to-evidence-1': { ok: true, metrics: { latency_ms: 0.4, dram_pct: 50, sm_pct: 30, occupancy: 0.7 }, coverage: ['latency_ms','dram_pct','sm_pct','occupancy'], source_backend: 'triton' },
  'driver-diagnose-1': { bottleneck_class: 'memory_bound', evidence: 'high dram' },
  'driver-anti-cheat-1': { ok: true, suspicious: false, reasons: [] },
  'revise-2': { suggestions: ['Increase BLOCK to 2048.'] },
  'tune-2': { kernel_code: '# tuned\n', changes_applied: ['Bumped BLOCK'] },
  'eval-2': { compiled: true, correct: true, speedup: 1.35 },
  'driver-build-2': { ok: true },
  'driver-run-2': { ok: true, latency_ms: 0.35 },
  'driver-profile-2': { ok: true },
  'driver-to-evidence-2': { ok: true, metrics: { latency_ms: 0.35 } },
  'driver-diagnose-2': { bottleneck_class: 'memory_bound' },
  'driver-anti-cheat-2': { ok: true },
  'final-report': 'driver-path triton dry-run report',
}

const CUDA_TOKENS = [
  /\bnvcc\b/,
  /\bncu\b/,
  /\b__global__\b/,
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
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-N agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('triton dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede setup when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('triton dry-run: setup prompt uses driver lang_fence ("python kernel"), not "Triton kernel"', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const setupCall = caps.find(c => c.label === 'setup')
  assert.ok(setupCall, 'setup call must be present')
  assert.match(setupCall.prompt, /python kernel optimization run/,
    `setup prompt must use driver lang_fence: ${setupCall.prompt.slice(0, 300)}`)
})

test('triton dry-run: large-step proposer prompt uses driver impl_requirements (@triton.jit kernel + Python launcher)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const proposeCall = caps.find(c => c.label === 'propose-1')
  assert.ok(proposeCall)
  assert.match(proposeCall.prompt, /@triton\.jit kernel/,
    'propose-1 prompt should pull driver impl_requirements')
})
