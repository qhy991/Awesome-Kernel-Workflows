'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'AKO4X/ako4x-kernel-optimizer.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'ako4x-args-triton.json'), 'utf8'))

function ds(idx) {
  return {
    [`driver-build-${idx}`]: { ok: true },
    [`driver-run-${idx}`]: { ok: true, latency_ms: 1, compiled: true, correct: true },
    [`driver-profile-${idx}`]: { ok: true, native_path: '/tmp/p.native' },
    [`driver-to-evidence-${idx}`]: { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
    [`driver-diagnose-${idx}`]: { bottleneck_class: 'memory_bound' },
    [`driver-anti-cheat-${idx}`]: { ok: true, suspicious: false },
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
  'read-baseline': {
    kernel_code: '@triton.jit\ndef rmsnorm_kernel(x, y, N): pass\n',
    language: 'triton',
    op_type: 'rmsnorm',
    key_functions: ['rmsnorm_kernel'],
    current_approach: 'baseline',
  },
  'create-workspace': { ok: true },
  'benchmark-baseline': { score: 0.085, latency_ms: 0.085, raw_output: '' },
  'hypothesis-0-0': {
    title: 'vec-loads',
    bottleneck: 'mem',
    ncu_evidence: 'na',
    hypothesis: 'h',
    expected_impact: '1.2x',
    risk: 'r',
  },
  'impl-0-vec-loads-v0': { code: '@triton.jit\ndef k(): pass', implementation_notes: 'n' },
  'smoke-r1-iter1-vec-loads': { passed: true, error_message: '', compile_error: false, correctness_error: false },
  'bench-r1-iter1-vec-loads': { score: 0.070, latency_ms: 0.070, speedup: 1.21, passed_workloads: '2/2', ab_compare_delta: '', variance_info: '', raw_output: '' },
  ...ds(0),
  'iterations-1': { ok: true },
  'silent-skip-check-1': { is_legitimate: true, suspicion_level: 'low', concerns: [], recommendation: 'archive' },
  'lib-check-1': { is_own_kernel: true, banned_libs_found: [], concerns: '' },
  'archive-iter-1-vec-loads': { ok: true },
  'update-state-1': { ok: true },
  'final-report': { report_md: 'r' },
}

// DEFERRED-GPU-VERIFICATION: this dry-run does not invoke the triton
// driver's build/run/profile scripts. It only asserts the rendered prompts
// reference the correct envelope (substrate path tokens, source_ext from
// the driver manifest, no CUDA-only token leakage). Actual driver execution
// is gated behind a GPU host and tracked in
// _meta/tools/test/DEFERRED-GPU-VERIFICATION.md.

const CUDA_TOKENS = [
  /(?:^|\s)nvcc(?=\s|$)/,
  /(?:^|\s)ncu(?=\s|$)/,
  /PYBIND11_MODULE/,
  /cuda_runtime\.h/,
  /```cuda\b/,
]
// Notes:
//   - __global__/__syncthreads excluded: AKO4X's generic read-baseline
//     prompt lists them as illustrative examples ("especially __global__
//     or @triton.jit functions") — language-agnostic, not CUDA-only.
//   - nvcc/ncu use word-isolated patterns to avoid matching the hard-coded
//     workspace path EXP_DIR/ncu-profiles/ which is an artifact directory
//     created on the legacy NCU-harness path but never written under the
//     driver path (no ncu_binary => no profiler invocation).

test('ako4x triton dry-run: USE_DRIVER on, no CUDA tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of CUDA_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `CUDA token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 300)}`)
    }
  }
})

test('ako4x triton dry-run: load-driver is the first agent call and references triton backend_dir', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede read-baseline when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('ako4x triton dry-run: rendered driver-run command starts with _substrate/backends/triton/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('ako4x triton dry-run: per-iter kernel path uses driver source_ext (.py)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const envCalls = caps.filter(c => /^driver-(build|run|profile|anti-cheat)-\d+$/.test(c.label))
  assert.ok(envCalls.length > 0, 'expected envelope agent calls')
  for (const e of envCalls) {
    assert.match(e.prompt, /\/kernel\.py\b/,
      `envelope prompt should reference .py kernel path under triton driver: ${e.prompt.slice(0, 300)}`)
    assert.doesNotMatch(e.prompt, /\/kernel\.cu\b/,
      'envelope prompt must not retain .cu kernel path under triton driver')
  }
})

test('ako4x triton dry-run: full per-attempt Layer-A envelope labels emitted (build/run/profile/to-evidence/diagnose/anti-cheat)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const labels = caps.map(c => c.label)
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-0`),
      `expected driver-${sub}-0 envelope agent under USE_DRIVER, got labels=${labels.join(',')}`)
  }
  // Envelope must be in Iterate phase (per-attempt, after bench)
  for (const c of caps) {
    if (/^driver-/.test(c.label) && c.label !== 'load-driver') {
      assert.equal(c.phase, 'Iterate',
        `envelope agent ${c.label} should be in Iterate phase, got ${c.phase}`)
    }
  }
})

test('ako4x triton dry-run: to-evidence + diagnose + anti-cheat reference substrate helpers', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const evidence = caps.find(c => c.label === 'driver-to-evidence-0')
  assert.ok(evidence, 'to-evidence call must be present')
  assert.match(evidence.prompt, /_substrate\/backends\/triton\/to_evidence\.py/,
    `to-evidence prompt should reference triton driver to_evidence.py: ${evidence.prompt.slice(0, 200)}`)
  const diagnose = caps.find(c => c.label === 'driver-diagnose-0')
  assert.match(diagnose.prompt, /_substrate\/diagnose\.py/,
    'diagnose must reference substrate-collapse diagnose.py')
  const antiCheat = caps.find(c => c.label === 'driver-anti-cheat-0')
  assert.match(antiCheat.prompt, /_substrate\/anti_cheat\.py/,
    'anti-cheat must reference substrate anti_cheat.py')
})
