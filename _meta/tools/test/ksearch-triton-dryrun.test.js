'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'KSearch/ksearch-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'ksearch-args-triton.json'), 'utf8'))

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
  'read-spec': {
    spec_text: 'RMSNorm spec',
    op_type: 'rmsnorm',
    input_shapes: 'x [B,H] fp16',
    output_shape: '[B,H] fp16',
    constraints: ['fp16'],
    baseline_code: '@triton.jit\ndef k(): pass\n',
    key_challenges: ['reduction'],
    design_dimensions: ['tile'],
  },
  'eval-baseline': { baseline_metric: 1.0, baseline_latency_ms: 0.12, eval_passed: true, performance_profile: 'mem-bound', bottleneck_analysis: 'reduction tail' },
  ...ds('root'),
  'init-tree': { decision_tree: { root: {}, n1: {} }, node_count: 2, open_actions: 1 },
  'propose-0': { updated_tree: { root: {}, n1: {} }, open_frontier_count: 1, nodes_added: 0 },
  'select-0': { selected_node_id: 'n1', action_title: 'tile_64', action_description: 'tile 64', action_score: 0.7, action_difficulty: 2, parent_solution_code: '@triton.jit\ndef k(): pass\n', parent_metric: 1.0, parent_is_root: true, context_for_generation: {}, reasoning: 'r' },
  'gen-0-0': { code: '@triton.jit\ndef k_v1(): pass\n', implementation_notes: 'n', design_choices: [] },
  'eval-0-0': { is_valid: true, metric_value: 1.18, latency_ms: 0.10, speedup_vs_baseline: 1.18, pass_rate: 'all', error_log: '', performance_analysis: 'p', remaining_bottleneck: 'rb' },
  ...ds('0-0'),
  'refine-0': { updated_tree: { root: {}, n1: { status: 'solved' } }, new_actions_added: 2, score_updates: [], reflection: 'r' },
  'propose-1': { updated_tree: { root: {}, n1: { status: 'solved' } }, open_frontier_count: 2, nodes_added: 0 },
  'select-1': { selected_node_id: 'n1c1', action_title: 'warp_shuffle', action_description: 'warp shuffle reduction', action_score: 0.65, action_difficulty: 3, parent_solution_code: '@triton.jit\ndef k_v1(): pass\n', parent_metric: 1.18, parent_is_root: false, context_for_generation: {}, reasoning: 'r' },
  'gen-1-0': { code: '@triton.jit\ndef k_v2(): pass\n', implementation_notes: 'n', design_choices: [] },
  'eval-1-0': { is_valid: true, metric_value: 1.32, latency_ms: 0.09, speedup_vs_baseline: 1.32, pass_rate: 'all', error_log: '', performance_analysis: 'p', remaining_bottleneck: 'rb' },
  ...ds('1-0'),
  'refine-1': { updated_tree: { root: {}, n1: { status: 'solved' } }, new_actions_added: 2, score_updates: [], reflection: 'r' },
  'final-report': { outcome: 'success', summary: 's', best_metric: 1.32, baseline_metric: 1.0, speedup: 1.32, cycles_completed: 2, solutions_evaluated: 2, best_node_id: 'n1c1' },
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

test('ksearch triton dry-run: USE_DRIVER on, no CUDA tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of CUDA_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `CUDA token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 300)}`)
    }
  }
})

test('ksearch triton dry-run: rendered driver-run command starts with _substrate/backends/triton/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('ksearch triton dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede read-spec when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('ksearch triton dry-run: gen/debug/improve/eval prompts use ```python fence (driver lang_fence)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const genCalls = caps.filter(c => /^gen-/.test(c.label))
  const evalCalls = caps.filter(c => /^eval-\d/.test(c.label))
  assert.ok(genCalls.length > 0, 'expected gen-* prompts')
  assert.ok(evalCalls.length > 0, 'expected eval-* prompts')
  for (const c of [...genCalls, ...evalCalls]) {
    assert.match(c.prompt, /```python\b/,
      `${c.label} should embed kernel under \`\`\`python fence under triton driver: ${c.prompt.slice(0, 200)}`)
  }
})

test('ksearch triton dry-run: per-iteration envelope labels emitted for root and each cycle', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const labels = caps.map(c => c.label)
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-root`),
      `expected driver-${sub}-root in ksearch triton dry-run, got labels=${labels.join(',')}`)
    for (const n of [0, 1]) {
      assert.ok(labels.includes(`driver-${sub}-${n}-0`),
        `expected driver-${sub}-${n}-0 in ksearch triton dry-run, got labels=${labels.join(',')}`)
    }
  }
})
