'use strict'

const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const {listAllWorkflows} = require('../../../scripts/add-agent-retry-scaffolding.js')

const root = path.resolve(__dirname, '..', '..', '..')
const source = fs.readFileSync(path.join(root, '_meta/scaffolding/agent-retry.js'), 'utf8')
const start = source.indexOf('async function agentRetry(')
const end = source.indexOf('/**\n * Null-guard', start)
const agentRetry = vm.runInNewContext(`${source.slice(start, end)}\nagentRetry`)

test('provider safeguard refusal is never resubmitted by agentRetry', async () => {
  for (const [code, message] of [
    ['KERSOR_PROVIDER_SAFEGUARD_REFUSAL', 'provider rejected the request'],
    ['KERSOR_CLAUDE_EXEC_FAILED', "API Error: safeguards flagged this message"],
    ['KERSOR_CLAUDE_EXEC_FAILED', "API Error: claude-opus-5-5 can't help with this. Start a new session to continue."],
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

test('every workflow and authoring template carries the provider-refusal guard', () => {
  const templates = ['_templates', '_meta/templates'].flatMap(dir =>
    fs.readdirSync(path.join(root, dir))
      .filter(name => name.endsWith('.js'))
      .map(name => path.join(root, dir, name)))
  const files = [...listAllWorkflows(), ...templates]
  assert.ok(files.length >= 43)
  for (const file of files) {
    const body = fs.readFileSync(file, 'utf8')
    assert.match(body,
      /safeguards\? flagged .*provider safeguard refusal.*Start a new session to continue/,
      path.relative(root, file))
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

test('persistent transport fault receives at most one repeat', async () => {
  for (const code of ['KERSOR_CLAUDE_TRANSPORT_FAULT',
                       'KERSOR_CLAUDE_EXEC_FAILED', 'KERSOR_CLAUDE_TIMEOUT']) {
    let calls = 0
    const fault = Object.assign(new Error('API Error: 524'), {code})
    await assert.rejects(agentRetry(async () => {
      calls++
      throw fault
    }, {retries: 5}), error => error === fault)
    assert.equal(calls, 2, code)
  }
})

test('configuration and authorization failures are not repeated', async () => {
  for (const code of ['KERSOR_CLAUDE_CONFIG_INVALID',
                       'KERSOR_AUTHORIZATION_FAILED', 'KERSOR_PERMISSION_DENIED']) {
    let calls = 0
    const terminal = Object.assign(new Error('terminal provider setup error'), {code})
    await assert.rejects(agentRetry(async () => {
      calls++
      throw terminal
    }, {retries: 5}), error => error === terminal)
    assert.equal(calls, 1, code)
  }
})

test('observed-model mismatch is never resubmitted by agentRetry', async () => {
  let calls = 0
  const mismatch = Object.assign(new Error('observed model differs from pinned model'),
    {code: 'KERSOR_CLAUDE_MODEL_IDENTITY_MISMATCH'})
  await assert.rejects(agentRetry(async () => {
    calls++
    throw mismatch
  }, {retries: 5, allowNull: true}), error => error === mismatch)
  assert.equal(calls, 1)
})
