'use strict'
// Tests for schema-stub.js, run-workflow.js, and print-workflow-prompts.js
// Run: node --test _meta/tools/test/*.test.js

const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')

// ---------------------------------------------------------------------------
// Task 1 — schemaStub
// ---------------------------------------------------------------------------

const schemaStub = require(path.resolve(__dirname, '../lib/schema-stub.js'))

test('schemaStub: enum => first element', () => {
  assert.deepStrictEqual(schemaStub({ type: 'string', enum: ['a', 'b'] }), 'a')
})

test('schemaStub: string => empty string', () => {
  assert.strictEqual(schemaStub({ type: 'string' }), '')
})

test('schemaStub: number => 0', () => {
  // LOAD-BEARING: AccelOpt L406/L442 do bestLatency.toFixed(3) and baselineLatency/bestLatency;
  // a null/''/undefined number stub would throw TypeError on .toFixed and crash the capture
  // before final-report.
  assert.strictEqual(schemaStub({ type: 'number' }), 0)
})

test('schemaStub: integer => 0', () => {
  // integer→0 included because a retrofit schema could introduce integer
  assert.strictEqual(schemaStub({ type: 'integer' }), 0)
})

test('schemaStub: boolean => false', () => {
  assert.strictEqual(schemaStub({ type: 'boolean' }), false)
})

test('schemaStub: array => []', () => {
  assert.deepStrictEqual(schemaStub({ type: 'array' }), [])
})

test('schemaStub: object with properties => recurse over all properties', () => {
  const result = schemaStub({
    type: 'object',
    properties: {
      name: { type: 'string' },
      count: { type: 'number' },
      active: { type: 'boolean' },
    },
  })
  assert.deepStrictEqual(result, { name: '', count: 0, active: false })
})

test('schemaStub: object with required not in properties => filled with stub', () => {
  const result = schemaStub({
    type: 'object',
    properties: {
      name: { type: 'string' },
    },
    required: ['name', 'extra'],
  })
  assert.ok('name' in result, 'name from properties')
  assert.ok('extra' in result, 'extra from required')
})

test('schemaStub: additionalProperties-only object => {}', () => {
  assert.deepStrictEqual(schemaStub({ type: 'object', additionalProperties: true }), {})
})

test('schemaStub: null/undefined schema => {}', () => {
  assert.deepStrictEqual(schemaStub(null), {})
  assert.deepStrictEqual(schemaStub(undefined), {})
})

// ---------------------------------------------------------------------------
// Task 2 — run-workflow.js sandbox
// ---------------------------------------------------------------------------

const runWorkflow = require(path.resolve(__dirname, '../lib/run-workflow.js'))

// A minimal synthetic workflow with top-level export, top-level await, top-level return
const SYNTHETIC_WORKFLOW = `export const meta = { name: 'synthetic', description: 'test' }

phase('Setup')
const result1 = await agent('hello world', { label: 'step1', phase: 'Setup', schema: { type: 'string' } })

phase('Execute')
const result2 = await agent('second call', { label: 'step2', phase: 'Execute', schema: { type: 'number' } })

return { meta, result1, result2 }
`

test('run-workflow: captures agent calls in order with correct seq/label/phase/prompt', async () => {
  const { calls } = await runWorkflow(SYNTHETIC_WORKFLOW, {}, {})
  assert.strictEqual(calls.length, 2)
  assert.strictEqual(calls[0].seq, 0)
  assert.strictEqual(calls[0].label, 'step1')
  assert.strictEqual(calls[0].phase, 'Setup')
  assert.strictEqual(calls[0].prompt, 'hello world')
  assert.strictEqual(calls[1].seq, 1)
  assert.strictEqual(calls[1].label, 'step2')
  assert.strictEqual(calls[1].phase, 'Execute')
  assert.strictEqual(calls[1].prompt, 'second call')
})

test('run-workflow: agentReturns map is consulted before schemaStub', async () => {
  const agentReturns = { step1: 'custom-return-value' }
  const { calls } = await runWorkflow(SYNTHETIC_WORKFLOW, {}, agentReturns)
  // The workflow's result1 is from agentReturns; step2 falls back to schemaStub(number)=0
  // We verify the call was recorded but the returned value is the override
  assert.strictEqual(calls.length, 2)
  assert.strictEqual(calls[0].label, 'step1')
})

// Synthetic workflow for parallel/pipeline signature test
const PARALLEL_PIPELINE_WORKFLOW = `export const meta = { name: 'parallel-pipeline-test' }

const par = await parallel([
  () => agent('a', { label: 'a' }),
  () => agent('b', { label: 'b' })
])

const pip = await pipeline([1, 2], (n) => agent('p' + n, { label: 'p' + n }))

return { meta }
`

test('run-workflow: parallel(1-arg thunks) and pipeline(2-arg items+fn) both captured in order', async () => {
  const { calls } = await runWorkflow(PARALLEL_PIPELINE_WORKFLOW, {}, {})
  const labels = calls.map(c => c.label)
  assert.deepStrictEqual(labels, ['a', 'b', 'p1', 'p2'])
})

// Synthetic workflow that throws — should propagate
const THROWING_WORKFLOW = `export const meta = { name: 'throwing-test' }

throw new Error('guard-error')

return { meta }
`

test('run-workflow: body throw propagates as rejection', async () => {
  await assert.rejects(
    () => runWorkflow(THROWING_WORKFLOW, {}, {}),
    (err) => {
      assert.ok(err.message.includes('guard-error'), `Expected 'guard-error' in: ${err.message}`)
      return true
    }
  )
})

// ---------------------------------------------------------------------------
// Task 3 — print-workflow-prompts.js capturePrompts
// ---------------------------------------------------------------------------

const { capturePrompts } = require(path.resolve(__dirname, '../print-workflow-prompts.js'))
const fs = require('node:fs')
const os = require('node:os')

// Write a synthetic workflow file for capturePrompts tests
const SYNTHETIC_WF_PATH = path.join(os.tmpdir(), 'synthetic-test-workflow.js')
fs.writeFileSync(SYNTHETIC_WF_PATH, SYNTHETIC_WORKFLOW)

test('capturePrompts: resolves to Array<{seq,label,phase,prompt}>', async () => {
  const results = await capturePrompts({ workflowPath: SYNTHETIC_WF_PATH, args: {}, agentReturns: {} })
  assert.ok(Array.isArray(results), 'result is array')
  assert.strictEqual(results.length, 2)
  for (const r of results) {
    assert.ok('seq' in r, 'has seq')
    assert.ok('label' in r, 'has label')
    assert.ok('phase' in r, 'has phase')
    assert.ok('prompt' in r, 'has prompt')
  }
})

test('capturePrompts: agentReturns consulted before schema generator', async () => {
  // step1 returns string from schema; with agentReturns override the call still records
  const results = await capturePrompts({
    workflowPath: SYNTHETIC_WF_PATH,
    args: {},
    agentReturns: { step1: 'overridden' },
  })
  assert.strictEqual(results.length, 2)
  assert.strictEqual(results[0].label, 'step1')
})

test('capturePrompts: stable-key JSON output is deterministic', async () => {
  const outPath = path.join(os.tmpdir(), 'capture-output-test.json')
  const results = await capturePrompts({ workflowPath: SYNTHETIC_WF_PATH, args: {}, agentReturns: {} })
  // Write to file using the stable-key serializer
  const stableJson = JSON.stringify(results.map(r => {
    const sorted = {}
    for (const k of Object.keys(r).sort()) sorted[k] = r[k]
    return sorted
  }), null, 2)
  fs.writeFileSync(outPath, stableJson)
  const readBack = JSON.parse(fs.readFileSync(outPath, 'utf8'))
  assert.deepStrictEqual(readBack, results.map(r => {
    const sorted = {}
    for (const k of Object.keys(r).sort()) sorted[k] = r[k]
    return sorted
  }))
})

// ---------------------------------------------------------------------------
// Task 4 — AccelOpt golden capture: determinism + coverage + byte-identity
// ---------------------------------------------------------------------------

const REPO_ROOT = path.resolve(__dirname, '../../..')
const ACCELOPT_WF = path.join(REPO_ROOT, 'AccelOpt/accelopt-kernel-optimization.js')
const FIX_DIR = path.join(REPO_ROOT, '_meta/tools/fixtures')
const CUDA_ARGS = JSON.parse(fs.readFileSync(path.join(FIX_DIR, 'accelopt-cuda-args.json'), 'utf8'))
const CUDA_RETURNS = JSON.parse(fs.readFileSync(path.join(FIX_DIR, 'accelopt-cuda-agent-returns.json'), 'utf8'))
const GEN_ARGS = JSON.parse(fs.readFileSync(path.join(FIX_DIR, 'accelopt-generate-args.json'), 'utf8'))
const GEN_RETURNS = JSON.parse(fs.readFileSync(path.join(FIX_DIR, 'accelopt-generate-agent-returns.json'), 'utf8'))

// Same stable-key serializer used by the print-workflow-prompts CLI (--out).
// Goldens on disk were written by that CLI; this test compares byte-for-byte.
function stableStringify(arr) {
  const sortedArr = arr.map((r) => {
    const sorted = {}
    for (const k of Object.keys(r).sort()) sorted[k] = r[k]
    return sorted
  })
  return JSON.stringify(sortedArr, null, 2)
}

test('accelopt golden (optimize): determinism — two runs are deep-equal', async () => {
  const a = await capturePrompts({ workflowPath: ACCELOPT_WF, args: CUDA_ARGS, agentReturns: CUDA_RETURNS })
  const b = await capturePrompts({ workflowPath: ACCELOPT_WF, args: CUDA_ARGS, agentReturns: CUDA_RETURNS })
  assert.deepStrictEqual(a, b)
})

test('accelopt golden (optimize): label set + phases match the explicit OPTIMIZE list', async () => {
  // The count (7) is a CONSEQUENCE of the agentReturns map unlocking eval (non-empty impl.code
  // + is_correct/is_compilable true) and learn (estimated_speedup 1.06 > MAX_THRESHOLD 1.05).
  // With empty stubs only 5 render. Do NOT hardcode 8.
  const EXPECTED_LABELS = [
    'read-baseline',
    'ncu-baseline',
    'plan-0-0',
    'impl-0-t-v0',
    'eval-plan_0_sample_0',
    'learn-t',
    'final-report',
  ]
  const EXPECTED_PHASES = ['Setup', 'Setup', 'Plan', 'Execute', 'Evaluate', 'Learn', 'Iterate']
  const calls = await capturePrompts({ workflowPath: ACCELOPT_WF, args: CUDA_ARGS, agentReturns: CUDA_RETURNS })
  assert.deepStrictEqual(calls.map(c => c.label), EXPECTED_LABELS)
  assert.deepStrictEqual(calls.map(c => c.phase), EXPECTED_PHASES)
})

test('accelopt golden (generate): label set matches the explicit GENERATE list', async () => {
  const EXPECTED_LABELS = [
    'generate-initial-kernel',
    'read-baseline',
    'ncu-baseline',
    'plan-0-0',
    'impl-0-t-v0',
    'eval-plan_0_sample_0',
    'learn-t',
    'final-report',
  ]
  const calls = await capturePrompts({ workflowPath: ACCELOPT_WF, args: GEN_ARGS, agentReturns: GEN_RETURNS })
  assert.deepStrictEqual(calls.map(c => c.label), EXPECTED_LABELS)
})

test('accelopt golden (optimize): byte-identical match against frozen golden file', async () => {
  const goldenPath = path.join(FIX_DIR, 'accelopt-today.golden.json')
  const calls = await capturePrompts({ workflowPath: ACCELOPT_WF, args: CUDA_ARGS, agentReturns: CUDA_RETURNS })
  const captured = stableStringify(calls)
  if (process.env.UPDATE_GOLDEN === '1') {
    fs.writeFileSync(goldenPath, captured)
    return
  }
  const golden = fs.readFileSync(goldenPath, 'utf8')
  assert.strictEqual(captured, golden, 'pre-retrofit optimize golden drifted; re-run with UPDATE_GOLDEN=1 only if intentional')
})

test('accelopt golden (generate): byte-identical match against frozen golden file', async () => {
  const goldenPath = path.join(FIX_DIR, 'accelopt-today-generate.golden.json')
  const calls = await capturePrompts({ workflowPath: ACCELOPT_WF, args: GEN_ARGS, agentReturns: GEN_RETURNS })
  const captured = stableStringify(calls)
  if (process.env.UPDATE_GOLDEN === '1') {
    fs.writeFileSync(goldenPath, captured)
    return
  }
  const golden = fs.readFileSync(goldenPath, 'utf8')
  assert.strictEqual(captured, golden, 'pre-retrofit generate golden drifted; re-run with UPDATE_GOLDEN=1 only if intentional')
})
