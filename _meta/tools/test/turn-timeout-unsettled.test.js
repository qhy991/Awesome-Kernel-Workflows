'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const helper = fs.readFileSync(path.resolve(__dirname, '../../scaffolding/turn-timeout.js'), 'utf8')

function runtime() {
  let fire, cleared = 0
  const api = vm.runInNewContext(helper + ';({withTurnTimeout,isTurnTimeout})', {
    args: {turn_timeout_min: 1},
    setTimeout(callback) { fire = callback; return 7 },
    clearTimeout(id) { assert.equal(id, 7); cleared++ },
  })
  return {api, expire: () => fire(), cleared: () => cleared}
}

test('watchdog expiry is an unsettled HOLD even if the child later writes a result', async () => {
  const r = runtime()
  let finish, sideEffects = 0
  const child = new Promise(resolve => { finish = resolve }).then(() => { sideEffects++; return 'late result' })
  const guarded = r.api.withTurnTimeout(child, 'implement-0')
  const rejected = assert.rejects(guarded, error => {
    assert.equal(error.code, 'KERSOR_TURN_UNSETTLED')
    assert.equal(error.retryable, false)
    assert.equal(error.outcome_state, 'unknown')
    assert.equal(error.resources_released, false)
    assert.equal(error.child_terminal_verified, false)
    assert.equal(error.phase_label, 'implement-0')
    assert.equal(r.api.isTurnTimeout(error), false, 'must not continue another cycle or return an incumbent')
    return true
  })
  r.expire()
  await rejected
  finish()
  await child
  assert.equal(sideEffects, 1, 'Promise.race never cancelled the child')
  assert.equal(r.cleared(), 1)
})

test('settled broker timeout is distinct from a legacy unverified watchdog', () => {
  const {api} = runtime()
  for (const code of ['KERSOR_TURN_TIMEOUT', 'KERSOR_TURN_UNSETTLED', 'PERMISSION_DENIED'])
    assert.equal(api.isTurnTimeout({code}), false)
  for (const code of ['KERSOR_CODEX_TIMEOUT', 'KERSOR_CLAUDE_TIMEOUT'])
    assert.equal(api.isTurnTimeout({code}), true)
})

test('settled child preserves its value or original error before the watchdog', async () => {
  const a = runtime()
  assert.equal(await a.api.withTurnTimeout(Promise.resolve('done'), 'normal'), 'done')
  assert.equal(a.cleared(), 1)
  const b = runtime(), failure = Object.assign(new Error('refused'), {code: 'PERMISSION_DENIED'})
  await assert.rejects(b.api.withTurnTimeout(Promise.reject(failure), 'refused'), error => error === failure)
  assert.equal(b.cleared(), 1)
})
