'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const WORKFLOW = path.join(ROOT, 'ReGraphT/regrapht-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX, 'regrapht-args-triton.json'), 'utf8'))

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
    impl_requirements: 'Provide a @triton.jit kernel plus a host wrapper.',
    methods: {},
    hw_vendor: 'nvidia',
  },
  'setup-task': {
    source_code: '@triton.jit\ndef k(): pass\n',
    operation_type: 'stencil',
    input_contract: 'x',
    output_contract: 'y',
    kernel_entry_points: ['k'],
    baseline_metric: 1.0,
    baseline_time_ms: 0.20,
    evaluator_contract: 'json',
    optimization_dimensions: ['tile'],
  },
  'build-regraph': {
    graph: { nodes: [{ id: 'v_init', method: 'init', visits: 0, reward: 0 }], edges: [] },
    method_labels: ['tile'],
    trace_count: 0,
  },
  ...ds('root'),
  'select-0': { path_node_ids: ['v_init'], method_sequence: ['tile'], selected_examples: [], selection_rationale: 'r' },
  'generate-0': { candidate_code: '@triton.jit\ndef k_v1(): pass\n', applied_methods: ['tile'], skipped_methods: [] },
  'evaluate-0': { compiled: true, correct: true, speedup: 1.18, kernel_time_ms: 0.17, baseline_time_ms: 0.20, error_message: '', error_type: '', evidence_summary: 's' },
  ...ds('0'),
  'update-graph-0': { updated_graph: { nodes: [{ id: 'v_init', method: 'init', visits: 1, reward: 1.18 }], edges: [] }, reward: 1.18, added_nodes: [], added_edges: [] },
  'select-1': { path_node_ids: ['v_init'], method_sequence: ['shared'], selected_examples: [], selection_rationale: 'r' },
  'generate-1': { candidate_code: '@triton.jit\ndef k_v2(): pass\n', applied_methods: ['shared'], skipped_methods: [] },
  'evaluate-1': { compiled: true, correct: true, speedup: 1.32, kernel_time_ms: 0.15, baseline_time_ms: 0.20, error_message: '', error_type: '', evidence_summary: 's' },
  ...ds('1'),
  'update-graph-1': { updated_graph: { nodes: [{ id: 'v_init', method: 'init', visits: 2, reward: 2.5 }], edges: [] }, reward: 1.32, added_nodes: [], added_edges: [] },
  'final-report': { outcome: 'success', summary: 's', best_speedup: 1.32, baseline_metric: 1.0, speedup: 1.32, iterations: 2 },
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

test('regrapht triton dry-run: USE_DRIVER on, no CUDA tokens leak into any rendered prompt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(caps.length > 0, 'expected captured prompts')
  for (const c of caps) {
    for (const re of CUDA_TOKENS) {
      assert.ok(!re.test(c.prompt),
        `CUDA token ${re} leaked into prompt label=${c.label}: ${c.prompt.slice(0, 300)}`)
    }
  }
})

test('regrapht triton dry-run: rendered driver-run command starts with _substrate/backends/triton/run.sh', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const runCalls = caps.filter(c => /^driver-run-/.test(c.label))
  assert.ok(runCalls.length > 0, 'expected at least one driver-run-* agent call')
  for (const r of runCalls) {
    assert.match(r.prompt, /_substrate\/backends\/triton\/run\.sh/,
      `driver-run prompt should reference triton driver run.sh: ${r.prompt.slice(0, 200)}`)
  }
})

test('regrapht triton dry-run: load-driver is the first agent call', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(caps[0].label, 'load-driver',
    'load-driver must precede setup-task when USE_DRIVER is on')
  assert.match(caps[0].prompt, /_substrate\/backends\/triton/,
    'load-driver prompt must reference the triton backend_dir')
})

test('regrapht triton dry-run: generate/evaluate prompts use ```python fence (driver lang_fence)', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const genCalls = caps.filter(c => /^generate-/.test(c.label))
  const evalCalls = caps.filter(c => /^evaluate-\d/.test(c.label))
  assert.ok(genCalls.length > 0, 'expected generate-* prompts')
  assert.ok(evalCalls.length > 0, 'expected evaluate-* prompts')
  for (const c of [...genCalls, ...evalCalls]) {
    assert.match(c.prompt, /```python\b/,
      `${c.label} should embed kernel under \`\`\`python fence under triton driver: ${c.prompt.slice(0, 200)}`)
  }
})

test('regrapht triton dry-run: per-iteration envelope labels emitted for root and each attempt', async () => {
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const labels = caps.map(c => c.label)
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-root`),
      `expected driver-${sub}-root in regrapht triton dry-run, got labels=${labels.join(',')}`)
    for (const n of [0, 1]) {
      assert.ok(labels.includes(`driver-${sub}-${n}`),
        `expected driver-${sub}-${n} in regrapht triton dry-run, got labels=${labels.join(',')}`)
    }
  }
})
