'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))
const WORKFLOW = path.join(ROOT, 'AccelOpt/accelopt-kernel-optimization.js')
const ARGS = JSON.parse(fs.readFileSync(path.join(ROOT, '_meta/tools/fixtures/accelopt-triton-args.json'), 'utf8'))

const triton = JSON.parse(fs.readFileSync(path.join(ROOT, '_substrate/backends/triton/idioms.json'), 'utf8'))
const tritonManifest = JSON.parse(fs.readFileSync(path.join(ROOT, '_substrate/backends/triton/manifest.json'), 'utf8'))

const agentReturns = {
  'load-driver': {
    present: true, capability_ok: true, missing: [],
    backend_id: 'triton',
    source_ext: tritonManifest.source_ext,
    lang_fence: triton.lang_fence,
    hw_vendor: tritonManifest.hw_vendor,
    profiler_name: tritonManifest.profiler.name,
    profiler_format: tritonManifest.profiler.format,
    capability_metrics: tritonManifest.capabilities.metrics,
    supported_classes: tritonManifest.capabilities.bottleneck_classes,
    problem_types: tritonManifest.capabilities.problem_types,
    requires_tools: tritonManifest.requires_tools,
    impl_requirements: triton.impl_requirements,
    read_metric_guide: triton.read_metric_guide,
    idioms: triton.methods,
    unsupported_methods: triton.unsupported_methods || [],
  },
  'read-baseline': { kernel_code: '@triton.jit\ndef k():\n    pass', op_type: 'gemm', key_functions: ['k'], current_approach: 'x' },
  'ncu-baseline': { ok: true, metrics: { latency_ms: 1.0, dram_pct: 50, sm_pct: 50, occupancy: 0.5 }, coverage: ['latency_ms', 'dram_pct', 'sm_pct', 'occupancy'], profiler_available: true, latency_ms: 1.0, sm_throughput_pct: 50, dram_throughput_pct: 50, achieved_occupancy_pct: 50, bottleneck_diagnosis: 'memory', profile_summary: 's' },
  'diagnose-baseline': { bottleneck_class: 'memory_bound', evidence: ['dram 50% high'] },
  'plan-0-0': { title: 't', focus_area: 'memory', profile_evidence: 'dram high', plan: 'p', expected_impact: '2x' },
  'impl-0-t-v0': { code: '@triton.jit\ndef k_opt():\n    pass', implementation_notes: 'n' },
  'eval-plan_0_sample_0': { is_correct: true, is_compilable: true, estimated_latency_ms: 0.94, estimated_speedup: 1.06, profile_comparison: 'c', bottleneck_addressed: true, new_bottleneck: 'none', performance_analysis: 'pa' },
  'learn-t': { title: 'rule', profile_trigger: 'dram high', rule: 'tile', original_snippet: 'a', optimized_snippet: 'b', why: 'reuse' },
  'assemble-evidence': { valid: true, normalized: {} },
}

async function tritonPrompts() {
  return capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns })
}

test('triton dry-run: a Setup load-driver step is present and cats the .json driver files', async () => {
  const caps = await tritonPrompts()
  const loader = caps.find(c => c.label === 'load-driver')
  assert.ok(loader, 'expected a load-driver agent call in Setup')
  assert.equal(loader.phase, 'Setup')
  assert.match(loader.prompt, /backends\/triton\/manifest\.json/, 'load-driver must cat the triton manifest.json (not .yaml)')
  assert.match(loader.prompt, /idioms\.json/, 'load-driver must cat idioms.json')
})

test('triton dry-run: Evaluate and Learn prompts DO render (agentReturns unlocked the loop)', async () => {
  const caps = await tritonPrompts()
  const labels = caps.map(c => c.label)
  assert.ok(labels.includes('eval-plan_0_sample_0'), 'Evaluate prompt must render')
  assert.ok(labels.includes('learn-t'), 'Learn prompt must render')
})

test('triton dry-run: NO CUDA-only tokens leak into ANY prompt (incl. Evaluate/Learn/Iterate)', async () => {
  const caps = await tritonPrompts()
  const all = caps.map(c => c.prompt).join('\n----\n')
  for (const banned of ['__global__', 'PYBIND11_MODULE', 'NCU Profile Results', 'cuda_fp16.h', 'cuda_runtime.h']) {
    assert.ok(!all.includes(banned), `triton prompts must not contain "${banned}"`)
  }
  assert.ok(!/\.cu\b/.test(all), 'triton prompts must not reference .cu sources')
  assert.ok(!/```cuda/.test(all), 'triton prompts must not open a ```cuda fence')
})

test('triton dry-run: uses the python lang fence and triton ABI requirements', async () => {
  const caps = await tritonPrompts()
  const all = caps.map(c => c.prompt).join('\n----\n')
  assert.match(all, /```python/, 'triton path must use the python code fence')
  assert.match(all, /@triton\.jit/, 'executor prompt must carry triton impl requirements')
})

test('triton dry-run: load-driver is GATED — absent when no backend_dir is passed', async () => {
  const { backend_dir, ...noDir } = ARGS
  const caps = await capturePrompts({ workflowPath: WORKFLOW, args: noDir, agentReturns })
  assert.ok(!caps.some(c => c.label === 'load-driver'),
    'load-driver must NOT render without backend_dir (USE_DRIVER gate)')
})

test('triton dry-run: this checks PROMPT WIRING, not execution (documented)', () => {
  assert.ok(true)
})
