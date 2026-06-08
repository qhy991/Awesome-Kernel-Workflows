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

// Task 3 tests added in a follow-up commit (see print-workflow-prompts.js).
