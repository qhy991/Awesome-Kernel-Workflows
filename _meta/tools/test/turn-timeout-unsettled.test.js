'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const retry = fs.readFileSync(path.resolve(__dirname, '../../scaffolding/agent-retry.js'), 'utf8')
const helper = fs.readFileSync(path.resolve(__dirname, '../../scaffolding/turn-timeout.js'), 'utf8')

function runtime() {
  let fire, cleared = 0
  const api = vm.runInNewContext(retry + '\n' + helper + ';({withTurnTimeout,isTurnTimeout})', {
    args: {turn_timeout_min: 1},
    setTimeout(callback) { fire = callback; return 7 },
    clearTimeout(id) { assert.equal(id, 7); cleared++ },
  })
  return {api, expire: () => fire(), cleared: () => cleared}
}

test('deadline cannot enter the ordinary timeout path before its active invocation settles', async () => {
  const r = runtime()
  let finish, sideEffects = 0, returned = false
  const child = new Promise(resolve => { finish = resolve }).then(() => { sideEffects++; return 'late result' })
  const guarded = r.api.withTurnTimeout(() => child, 'implement-0', {retries:0})
  guarded.then(() => {returned=true}, () => {returned=true})
  r.expire()
  await Promise.resolve(); await Promise.resolve()
  assert.equal(returned, false)
  assert.equal(sideEffects, 0)
  finish()
  await assert.rejects(guarded, error => {
    assert.equal(error.code, 'KERSOR_TURN_DRAINED')
    assert.equal(error.retryable, false)
    assert.equal(error.invocation_settled, true)
    assert.equal(error.resource_release_verified, false)
    assert.equal(error.phase_label, 'implement-0')
    assert.equal(r.api.isTurnTimeout(error), true)
    return true
  })
  assert.equal(sideEffects, 1)
  assert.equal(r.cleared(), 1)
})

test('settled broker timeout is distinct from a legacy unverified watchdog', () => {
  const {api} = runtime()
  for (const code of ['KERSOR_TURN_TIMEOUT', 'KERSOR_TURN_UNSETTLED', 'PERMISSION_DENIED'])
    assert.equal(api.isTurnTimeout({code}), false)
  assert.equal(api.isTurnTimeout({code:'KERSOR_TURN_DRAINED'}),false)
  assert.equal(api.isTurnTimeout({code:'KERSOR_TURN_DRAINED',invocation_settled:true}),true)
  for (const code of ['KERSOR_CODEX_TIMEOUT', 'KERSOR_CLAUDE_TIMEOUT'])
    assert.equal(api.isTurnTimeout({code}), true)
})

test('settled child preserves its value or original error before the watchdog', async () => {
  const a = runtime()
  assert.equal(await a.api.withTurnTimeout(() => Promise.resolve('done'), 'normal'), 'done')
  assert.equal(a.cleared(), 1)
  const b = runtime(), failure = Object.assign(new Error('refused'), {code: 'PERMISSION_DENIED'})
  await assert.rejects(b.api.withTurnTimeout(() => Promise.reject(failure), 'refused'), error => error === failure)
  assert.equal(b.cleared(), 1)
})
