'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))
const runWorkflow = require(path.resolve(__dirname, '..', 'lib', 'run-workflow.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'KSearch/ksearch-kernel-optimization.js')

const baseArgs = {
  problem_path: '/tmp/ksearch-guard/spec.yaml',
  op_description: 'rmsnorm',
  language: 'triton',
  target_gpu: 'H100',
  iterations: 1,
  attempts_per_cycle: 1,
  stagnation_window: 3,
  max_difficulty: 4,
  benchmark_command: 'python /tmp/ksearch-guard/bench.py {kernel_path} {result_path}',
  kernel_path: '/tmp/ksearch-guard/baseline.py',
  rtol: 0.01,
  atol: 0.01,
  exp_dir: '/tmp/ksearch-guard',
}

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

const minimalReturns = {
  'load-driver': { present: true, backend_id: 'triton', source_ext: '.py', aux_ext: '.py', lang_fence: 'python', impl_requirements: 'Provide a @triton.jit kernel.', methods: {}, hw_vendor: 'nvidia' },
  'read-spec': { spec_text: 's', op_type: 'rms', input_shapes: 'x', output_shape: 'y', constraints: [], baseline_code: 'k', key_challenges: [], design_dimensions: ['d'] },
  'eval-baseline': { baseline_metric: 1.0, baseline_latency_ms: 0.1, eval_passed: true, performance_profile: 'p', bottleneck_analysis: 'b' },
  ...ds('root'),
  'init-tree': { decision_tree: { root: {}, n1: {} }, node_count: 2, open_actions: 1 },
  'propose-0': { updated_tree: { root: {}, n1: {} }, open_frontier_count: 1, nodes_added: 0 },
  'select-0': { selected_node_id: 'n1', action_title: 't', action_description: 'd', action_score: 0.7, action_difficulty: 2, parent_solution_code: 'k', parent_metric: 1.0, parent_is_root: true, context_for_generation: {}, reasoning: 'r' },
  'gen-0-0': { code: 'k1', implementation_notes: 'n', design_choices: [] },
  'eval-0-0': { is_valid: true, metric_value: 1.1, latency_ms: 0.09, speedup_vs_baseline: 1.1, pass_rate: 'all', error_log: '', performance_analysis: 'p', remaining_bottleneck: 'rb' },
  ...ds('0-0'),
  'refine-0': { updated_tree: { root: {}, n1: { status: 'solved' } }, new_actions_added: 2, score_updates: [], reflection: 'r' },
  'final-report': { outcome: 'success', summary: 's', best_metric: 1.1, baseline_metric: 1.0, speedup: 1.1, cycles_completed: 1, solutions_evaluated: 1, best_node_id: 'n1' },
}

async function run(extra, agentReturns) {
  return capturePrompts({
    workflowPath: WORKFLOW,
    args: { ...baseArgs, ...extra },
    agentReturns: agentReturns || {},
  })
}

test('legacy path: no backend_dir -> renders triton vocabulary, no load-driver agent', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({}, legacyReturns)
  assert.ok(caps.length > 0)
  assert.ok(caps.some(c => /```triton\b/.test(c.prompt)), 'legacy path with language=triton should render ```triton code fence')
  assert.ok(!caps.some(c => c.label === 'load-driver'),
    'legacy path must NOT issue load-driver agent (USE_DRIVER off-by-default)')
})

test('legacy CUDA path gives generated candidates a CUDA source extension', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const caps = await run({ language: 'cuda' }, legacyReturns)
  const generated = caps.find(c => c.label === 'gen-0-0')
  assert.ok(generated, 'legacy CUDA path should generate a candidate')
  assert.match(generated.prompt, /cycle_0_a0\.cu\b/)
  assert.doesNotMatch(generated.prompt, /cycle_0_a0\.py\b/)
})

test('explicit CuTe DSL reaches generation as Python source with DSL guidance', async () => {
  const legacyReturns = { ...minimalReturns }
  delete legacyReturns['load-driver']
  for (const k of Object.keys(legacyReturns)) {
    if (/^driver-/.test(k)) delete legacyReturns[k]
  }
  const calls = await run({ language: 'cute-dsl' }, legacyReturns)
  const generated = calls.find(c => c.label === 'gen-0-0')
  assert.ok(generated, 'CuTe DSL must reach the candidate generation branch')
  assert.match(generated.prompt, /cycle_0_a0\.py\b/)
  assert.match(generated.prompt, /Requested DSL: CuTe DSL/)
  assert.match(generated.prompt, /cutlass\.cute/)
  assert.match(generated.prompt, /do not delegate GEMM to torch\.matmul/)
  assert.ok(!calls.some(c => c.label === 'load-driver'))
})

test('CuTe DSL rejects the CUDA C++ driver and binds a Host-measured Python candidate', async () => {
  await assert.rejects(
    run({ language: 'cute-dsl', backend_dir: '_substrate/backends/cuda' }),
    /CuTe DSL source is Python/,
  )
  const source = fs.readFileSync(WORKFLOW, 'utf8')
  const candidatePath = '/tmp/ksearch-guard/ksearch_c0_a0.py'
  const candidateSha = 'a'.repeat(64)
  const {result} = await runWorkflow(source, {
    ...baseArgs, language: 'cute-dsl', integration_pattern: 'sol_execbench_solution',
    sol_cli: '/tmp/sol', sol_task_dir: '/tmp/task', sol_bench_config: '/tmp/config',
    sol_seed_dir: '/tmp/seed', sol_substrate_dir: '/tmp/substrate',
  }, minimalReturns, {
    'sol-cute-seed-baseline': request => {
      assert.equal(request.candidateLanguage, 'cute-dsl')
      assert.equal(request.baselineSolutionPath, '/tmp/seed/seed.solution.json')
      return {compiled: true, correct: true, full_workload_set: true,
        output_contract_valid: true, measurement_valid: true,
        candidate_latency_aggregate_ms: 0.02,
        result_path: '/tmp/ksearch-guard/host_seed.bench.jsonl.result.json'}
    },
    'sol-eval-0-0': request => {
      assert.equal(request.candidateLanguage, 'cute-dsl')
      assert.equal(request.candidatePath, candidatePath)
      assert.equal(request.bindingOut, '/tmp/ksearch-guard/bindings/ksearch_c0_a0.json')
      assert.equal(request.baselineEvaluationPath, '/tmp/ksearch-guard/host_seed.bench.jsonl.result.json')
      assert.equal(request.parentSolutionPath, '/tmp/seed/seed.solution.json')
      return {
        compiled: true, correct: true, full_workload_set: true,
        output_contract_valid: true, measurement_valid: true,
        n_pass: 1, n_total: 1, speedup: 1.2,
        candidate_latency_aggregate_ms: 0.016,
        candidate_path: candidatePath, candidate_sha256: candidateSha,
        artifact_binding: {verified: true, candidate_id: 'cycle_0_attempt_0',
          candidate_sha256: candidateSha,
          binding_path: '/tmp/ksearch-guard/bindings/ksearch_c0_a0.json'},
      }
    },
  })
  assert.equal(result.generated_kernel_path, candidatePath)
  assert.equal(result.artifact_binding_required, true)
  assert.equal(result.best_metric, 1.25)
})

test('§6.4: args.backend matches manifest backend_id -> ok', async () => {
  await assert.doesNotReject(run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton', language: 'triton' },
    minimalReturns,
  ))
})

test('§6.4: conflicting backend + language -> throws naming both', async () => {
  await assert.rejects(
    run({ backend: 'cuda', language: 'triton', backend_dir: '_substrate/backends/cuda' }),
    (err) => {
      assert.match(err.message, /Conflicting args/)
      assert.match(err.message, /backend="cuda"/)
      assert.match(err.message, /language="triton"/)
      return true
    },
  )
})

test('§6.4: args.backend without backend_dir -> throws (no implicit-resolve)', async () => {
  const { language, ...noLang } = baseArgs
  await assert.rejects(
    capturePrompts({
      workflowPath: WORKFLOW,
      args: { ...noLang, backend: 'cuda' },
      agentReturns: {},
    }),
    (err) => {
      assert.match(err.message, /requires args\.backend_dir/)
      return true
    },
  )
})

test('USE_DRIVER on with valid driver: load-driver fires first + per-eval envelope agents present', async () => {
  const caps = await run(
    { backend_dir: '_substrate/backends/triton', backend: 'triton', language: 'triton' },
    minimalReturns,
  )
  const labels = caps.map(c => c.label)
  assert.equal(labels[0], 'load-driver',
    'load-driver must be the first agent call when USE_DRIVER')
  for (const sub of ['build', 'run', 'profile', 'to-evidence', 'diagnose', 'anti-cheat']) {
    assert.ok(labels.includes(`driver-${sub}-root`),
      `expected driver-${sub}-root envelope agent (baseline eval) under USE_DRIVER, got labels=${labels.join(',')}`)
    assert.ok(labels.includes(`driver-${sub}-0-0`),
      `expected driver-${sub}-0-0 envelope agent (cycle 0 attempt 0) under USE_DRIVER, got labels=${labels.join(',')}`)
  }
})

test('manifest backend_id mismatch with args.backend -> throws', async () => {
  const driverReturn = { present: true, backend_id: 'triton', source_ext: '.py', lang_fence: 'python' }
  // Strip language to avoid the §6.4 backend≠language guard firing first;
  // we want to exercise the manifest-vs-axis mismatch branch.
  const { language, ...noLang } = baseArgs
  await assert.rejects(
    capturePrompts({
      workflowPath: WORKFLOW,
      args: { ...noLang, backend_dir: '_substrate/backends/cuda', backend: 'cuda' },
      agentReturns: { 'load-driver': driverReturn },
    }),
    (err) => {
      assert.match(err.message, /backend_id="triton"/)
      assert.match(err.message, /conflicts/)
      return true
    },
  )
})

test('empty propose result preserves initialized decision tree', async () => {
  const returns = {
    ...minimalReturns,
    'propose-0': { updated_tree: {}, open_frontier_count: 0, nodes_added: 0 },
  }
  const caps = await run({}, returns)
  const select = caps.find(c => c.label === 'select-0')
  assert.ok(select, 'selection should still run after an empty propose response')
  assert.match(select.prompt, /"n1"/, 'initialized frontier must survive an empty updated_tree')
})

test('empty init-tree result seeds an executable deterministic frontier', async () => {
  const returns = {
    ...minimalReturns,
    'init-tree': { decision_tree: {}, node_count: 29, open_actions: 20 },
    'propose-0': { updated_tree: {}, open_frontier_count: 0, nodes_added: 0 },
  }
  const caps = await run({}, returns)
  const select = caps.find(c => c.label === 'select-0')
  assert.ok(select, 'selection should run against the fallback frontier')
  assert.match(select.prompt, /fallback-regime-dispatch/)
  assert.match(select.prompt, /fallback-small-m/)
  assert.match(select.prompt, /"status": "open"/)
})

test('null or sentinel selection stops before candidate generation', async () => {
  for (const selectedNodeId of [null, 'null', 'search_exhausted']) {
    const returns = {
      ...minimalReturns,
      'select-0': {
        selected_node_id: selectedNodeId,
        action_title: 'search_exhausted',
      },
    }
    const caps = await run({}, returns)
    assert.ok(!caps.some(c => c.label === 'gen-0-0'),
      `selection ${String(selectedNodeId)} must not generate a candidate`)
  }
})
