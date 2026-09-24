'use strict'

const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')

const root = path.resolve(__dirname, '..', '..', '..')
const source = fs.readFileSync(path.join(root, '_meta/scaffolding/agent-retry.js'), 'utf8')
const start = source.indexOf('async function agentRetry(')
const end = source.indexOf('/**\n * Null-guard', start)
const agentRetry = vm.runInNewContext(`${source.slice(start, end)}\nagentRetry`)

test('provider safeguard refusal is never resubmitted by agentRetry', async () => {
  for (const [code, message] of [
    ['KERSOR_PROVIDER_SAFEGUARD_REFUSAL', 'provider rejected the request'],
    ['KERSOR_CLAUDE_EXEC_FAILED', "API Error: safeguards flagged this message"],
  ]) {
    let calls = 0
    const refusal = Object.assign(new Error(message), {code})
    await assert.rejects(agentRetry(async () => {
      calls++
      throw refusal
    }, {retries: 5, allowNull: true}), error => error === refusal)
    assert.equal(calls, 1)
  }
})

test('transient transport failure may still retry', async () => {
  let calls = 0
  const result = await agentRetry(async () => {
    calls++
    if (calls === 1) throw Object.assign(new Error('socket hang up'), {code: 'KERSOR_CLAUDE_EXEC_FAILED'})
    return {ok: true}
  }, {retries: 5})
  assert.equal(calls, 2)
  assert.equal(result.ok, true)
})
