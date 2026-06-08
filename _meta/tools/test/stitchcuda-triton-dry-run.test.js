'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'StitchCUDA/stitchcuda-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'stitchcuda-args-triton.json'), 'utf8'))

function ds(attempt) {
  return {
    [`driver-build-${attempt}`]: { ok: true },
    [`driver-run-${attempt}`]: { ok: true, latency_ms: 1, compiled: true, correct: true },
    [`driver-profile-${attempt}`]: { ok: true, native_path: '/tmp/p.native' },
    [`driver-to-evidence-${attempt}`]: { ok: true, metrics: { latency_ms: 1 }, coverage: [] },
    [`driver-diagnose-${attempt}`]: { bottleneck_class: 'memory_bound' },
    [`driver-anti-cheat-${attempt}`]: { ok: true, suspicious: false },
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
  'Setup StitchCUDA': {
    cuda_version: '12.3',
    target_architecture: 'sm_90',
    pytorch_available: true,
    kernel_spec: { operation: 'gemm', shapes: '[M,N,K]', dtypes: ['float16'], baseline_gflops: 100.0 },
    kernelbench_config: { metrics: ['correctness', 'performance'] },
    replan_heuristics: { compile_failure_threshold: 1, correctness_failure_threshold: 1, stagnation_iterations: 3 },
    max_attempts: 2,
  },
  'Plan attempt 1': { attempt: 1, plan_summary: 's1', optimization_approach: 'balanced', key_strategies: [], implementation_steps: [], threading_config: {}, memory_strategy: 'm', expected_bottleneck: 'memory' },
  'Code attempt 1': { attempt: 1, kernel_code: '@triton.jit\ndef k(): pass\n', host_code: 'h', kernel_name: 'k1', implementation_notes: 'n' },
  ...ds(0),
  'Verify attempt 1': { attempt: 1, compilation_success: false, compilation_errors: ['e'], resource_usage: {}, correctness_passed: false, correctness_errors: [], max_error: 0.0, performance_gflops: 0.0, execution_time_ms: 0.0, speedup_vs_baseline: 0.0, kernelbench_score: null, verification_passed: false, failure_reason: 'compilation' },
  'Replan': { diagnosis: 'd', alternative_approach: 'a', key_changes: [], new_plan_summary: 'np' },
  'Plan attempt 2': { attempt: 2, plan_summary: 's2', optimization_approach: 'compute-bound', key_strategies: [], implementation_steps: [], threading_config: {}, memory_strategy: 'm', expected_bottleneck: 'compute' },
  'Code attempt 2': { attempt: 2, kernel_code: '@triton.jit\ndef k(): pass\n', host_code: 'h', kernel_name: 'k2', implementation_notes: 'n' },
  ...ds(1),
  'Verify attempt 2': { attempt: 2, compilation_success: true, compilation_errors: [], resource_usage: {}, correctness_passed: true, correctness_errors: [], max_error: 0.0001, performance_gflops: 300.0, execution_time_ms: 0.5, speedup_vs_baseline: 3.0, kernelbench_score: 0.95, verification_passed: true, failure_reason: null },
  'Generate report': { summary: 's', total_attempts: 2, successful_attempts: 1, best_gflops: 300.0, speedup: 3.0, best_attempt: 2, report_path: '/tmp/r.md' },
}

const EXPECTED_DRIVER_LABELS = [
  'driver-build-0', 'driver-run-0', 'driver-profile-0',
  'driver-to-evidence-0', 'driver-diagnose-0', 'driver-anti-cheat-0',
  'driver-build-1', 'driver-run-1', 'driver-profile-1',
  'driver-to-evidence-1', 'driver-diagnose-1', 'driver-anti-cheat-1',
]

test('stitchcuda triton dry-run: load-driver is the first agent call and references triton backend_dir', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede Setup StitchCUDA when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('stitchcuda triton dry-run: full per-attempt Layer-A envelope labels emitted (build/run/profile/to-evidence/diagnose/anti-cheat x 2 attempts)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const labels = caps.map(c => c.label)
  for (const expected of EXPECTED_DRIVER_LABELS) {
    assert.ok(labels.includes(expected),
      `expected envelope label ${expected} missing; got driver labels=${labels.filter(l => /^driver-/.test(l)).join(',')}`)
  }
})

test('stitchcuda triton dry-run: driver-run prompts reference triton run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt must reference triton run.sh: ${r.prompt.slice(0, 200)}`)
  }
})
