'use strict'
// Tests for the inlined agent-retry scaffolding (_meta/scaffolding/agent-retry.js).
//
// The helper is a copy-paste snippet with no `module.exports` (it is inlined
// verbatim into each workflow because the Workflow sandbox cannot `import`), so
// we evaluate its three function declarations through a `new Function` wrapper
// and pull agentRetry/expect/guard out. The wrapper runs in the host realm, so
// the async functions it defines return host Promises that `await`/assert.rejects
// can observe normally.
//
// Run: node --test _meta/tools/test/*.test.js

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const SRC = fs.readFileSync(
  path.resolve(__dirname, '../../scaffolding/agent-retry.js'), 'utf8')

function loadHelper() {
  // eslint-disable-next-line no-new-func
  const wrapper = new Function(
    'exports',
    SRC + '\nexports.agentRetry = agentRetry; exports.expect = expect; exports.guard = guard;',
  )
  const exports = {}
  wrapper(exports)
  return exports
}

test('agentRetry: returns the first non-null result without extra calls', async () => {
  const { agentRetry } = loadHelper()
  let calls = 0
  const fn = async () => { calls++; return { ok: calls } }
  const r = await agentRetry(fn, { retries: 5, label: 'probe' })
  assert.strictEqual(r.ok, 1)
  assert.strictEqual(calls, 1)
})

test('agentRetry: retries on null, then succeeds', async () => {
  const { agentRetry } = loadHelper()
  let calls = 0
  const fn = async () => { calls++; return calls < 3 ? null : { ok: true } }
  const r = await agentRetry(fn, { retries: 5, label: 'probe' })
  assert.deepStrictEqual(r, { ok: true })
  assert.strictEqual(calls, 3)
})

test('agentRetry: a thrown-then-succeeded sequence returns the result', async () => {
  const { agentRetry } = loadHelper()
  let calls = 0
  const fn = async () => {
    calls++
    if (calls < 2) throw new Error('transient')
    return { ok: calls }
  }
  const r = await agentRetry(fn, { retries: 5 })
  assert.deepStrictEqual(r, { ok: 2 })
  assert.strictEqual(calls, 2)
})

test('agentRetry: THROWS an attributable error after exhausting retries (fail-safe default)', async () => {
  // This is the core #20 fix: a null-returning agent must NOT silently return
  // null to be dereferenced later (causing a cryptic TypeError). By default it
  // throws a labeled error so the round aborts cleanly with a recorded reason.
  const { agentRetry } = loadHelper()
  const fn = async () => null
  await assert.rejects(
    agentRetry(fn, { retries: 2, label: 'diagnose' }),
    (err) => {
      assert.ok(err instanceof Error, 'should be an Error')
      assert.match(err.message, /agentRetry/, 'message names agentRetry')
      assert.match(err.message, /diagnose/, 'message includes the label')
      assert.match(err.message, /null/, 'message mentions null')
      return true
    },
  )
})

test('agentRetry: default retries is 5 (=> 6 attempts) then throws', async () => {
  const { agentRetry } = loadHelper()
  let calls = 0
  const fn = async () => { calls++; return null }
  await assert.rejects(() => agentRetry(fn), /agentRetry/)
  assert.strictEqual(calls, 6)
})

test('agentRetry: returns null after retries when { allowNull: true }', async () => {
  // Opt-out preserves intentional graceful-degradation sites (e.g. optional
  // probing agents whose null result is handled with `(x && x.field) || fallback`).
  const { agentRetry } = loadHelper()
  const fn = async () => null
  const r = await agentRetry(fn, { retries: 2, allowNull: true, label: 'gate' })
  assert.strictEqual(r, null)
})

test('agentRetry: rethrows the last error when fn throws every attempt', async () => {
  const { agentRetry } = loadHelper()
  const fn = async () => { throw new Error('boom') }
  await assert.rejects(
    agentRetry(fn, { retries: 2, label: 'x' }),
    (err) => { assert.strictEqual(err.message, 'boom'); return true },
  )
})

test('expect: throws attributable error on null / missing required field', () => {
  const { expect } = loadHelper()
  assert.throws(() => expect(null, 'code', 'impl'), /agentRetry: required field "code"/)
  assert.throws(() => expect({}, 'code', 'impl'), /agentRetry: required field "code"/)
  assert.strictEqual(expect({ code: 'x' }, 'code', 'impl'), 'x')
})

test('guard: returns fallback on null / missing optional field', () => {
  const { guard } = loadHelper()
  assert.deepStrictEqual(guard(null, 'techniques', []), [])
  assert.deepStrictEqual(guard({}, 'techniques', []), [])
  assert.deepStrictEqual(guard({ techniques: [1] }, 'techniques', []), [1])
})
