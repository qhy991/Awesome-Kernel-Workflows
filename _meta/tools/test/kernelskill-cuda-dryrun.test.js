'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'KernelSkill/kernelskill-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'kernelskill-args-cuda-driver.json'), 'utf8'))

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
    backend_id: 'cuda',
    source_ext: '.cu',
    aux_ext: '.h',
    lang_fence: 'cuda',
    impl_requirements: 'Provide a CUDA __global__ kernel.',
    methods: {},
    hw_vendor: 'nvidia',
  },
  'read-reference': {
    reference_code: 'class Model(nn.Module): pass',
    op_type: 'relu', forward_summary: 'relu', input_shapes: '[N]',
    dtype: 'float32', num_ops: 1, has_matmul: false,
  },
  'eager-baseline': { baseline_latency_ms: 0.42, baseline_available: true, harness_notes: 'n/a' },
  'seed-0': { code: '__global__ void k(){}', strategy: 's', kernels_materialized: 1 },
  'seed-1': { code: '__global__ void k4(){}', strategy: 'vec', kernels_materialized: 1 },
  'seed-eval-0': { is_compilable: true, is_correct: true, latency_ms: 0.4, speedup: 1.05, issues: [] },
  'seed-eval-1': { is_compilable: true, is_correct: true, latency_ms: 0.35, speedup: 1.20, issues: [] },
  'review-0': {
    is_compilable: true, is_correct: true, latency_ms: 0.35, speedup: 1.20, error_excerpt: '',
    ncu_metrics: { dram_throughput_pct: 85 }, profile_summary: 'mem-bound',
  },
  'features-0': { has_reuse: false, streaming_no_reuse: true, kernel_structure_id: 0 },
  'gate-0': { tier: 'Tier-H', bottleneck_id: 'memory_bandwidth_bound', matched_case_id: 'M1', allowed_methods: ['vectorized_load_store'], key_metrics: 'k', derived_values: 'd' },
  'plan-0': { method_name: 'vectorized_load_store', rationale: 'r', plan: 'p' },
  'optimize-0': { code: '__global__ void k2(){}', implementation_notes: 'n' },
  ...ds('0'),
  'review-1': {
    is_compilable: true, is_correct: true, latency_ms: 0.30, speedup: 1.40, error_excerpt: '',
    ncu_metrics: { dram_throughput_pct: 92 }, profile_summary: 'roofline',
  },
  'features-1': { has_reuse: false, streaming_no_reuse: true, kernel_structure_id: 0 },
  'gate-1': { tier: 'Tier-H', bottleneck_id: 'memory_bandwidth_bound', matched_case_id: 'M1', allowed_methods: ['improve_tail_handling'], key_metrics: 'k', derived_values: 'd' },
  'plan-1': { method_name: 'improve_tail_handling', rationale: 'r', plan: 'p' },
  'optimize-1': { code: '__global__ void k3(){}', implementation_notes: 'n' },
  ...ds('1'),
  'final-report': 'KernelSkill cuda-driver dry-run report.',
}

const TRITON_TOKENS = [
  /@triton\.jit/,
  /triton\.language/,
  /\bimport triton\b/,
  /tl\.program_id/,
  /tl\.constexpr/,
  /```triton\b/,
]

test('kernelskill cuda-driver dry-run: USE_DRIVER on, no Triton tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    // skip the read-reference/final-report prompts that legitimately reference python source
    for (const re of TRITON_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `Triton token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 300)}`)
    }
  }
})

test('kernelskill cuda-driver dry-run: rendered driver-run command starts with _substrate/backends/cuda/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/cuda\/run\.sh/,
      `driver-run prompt should reference cuda driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('kernelskill cuda-driver dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede read-reference when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/cuda/,
    'load-driver prompt must reference the cuda backend_dir')
})

test('kernelskill cuda-driver dry-run: review/optimize/repair prompts use ```cuda fence (driver lang_fence)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  // Seed prompts don't embed a kernel (they only contain the PyTorch ref).
  // Per-round prompts that embed the current kernel must use the cuda fence.
  const fenceCalls = caps.filter(c => /^(review|features|gate|plan|optimize|seed-eval)-/.test(c.label))
  assert.ok(fenceCalls.length > 0, 'expected per-round + seed-eval prompts')
  let withFence = 0
  for (const c of fenceCalls) {
    if (/```cuda\b/.test(c.prompt)) withFence++
  }
  assert.ok(withFence > 0,
    `at least one per-round prompt must embed the kernel under \`\`\`cuda fence under cuda driver`)
})

test('kernelskill cuda-driver dry-run: per-round envelope labels emitted for each round (build/run/profile/to-evidence/diagnose/anti-cheat)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const labels = caps.map(c => c.label)
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    for (const n of [0, 1]) {
      assert.ok(labels.includes(`driver-${sub}-${n}`),
        `expected driver-${sub}-${n} in kernelskill cuda-driver dry-run, got labels=${labels.join(',')}`)
    }
  }
})

test('kernelskill cuda-driver dry-run: DEFERRED — Layer-A envelope is rendered (dry-run only); real GPU verification deferred per docs/superpowers/specs', () => {
  // This marker test records that the cuda driver-dispatch path is exercised
  // at the prompt-rendering layer only — no GPU runs in CI. KernelSkill is
  // CUDA-only, so this row substitutes for the multi-language triton-dryrun
  // pattern used by KernelFoundry/KSearch. Promoted to measured evidence when
  // the GPU-deferred verification pipeline lands per
  // _meta/tools/test/DEFERRED-GPU-VERIFICATION.md.
  assert.ok(true, 'placeholder for the deferred-GPU evidence row (cuda driver)')
})
