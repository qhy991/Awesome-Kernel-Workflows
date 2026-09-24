'use strict'

const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')

const root = path.resolve(__dirname, '..', '..', '..')
const source = fs.readFileSync(path.join(root, '_meta/scaffolding/turn-timeout.js'), 'utf8')
const block = source.slice(source.lastIndexOf('const TURN_TIMEOUT_MS ='))

test('timeout waits for its live activation to settle before returning', async () => {
  const timers = new Map()
  let nextId = 0
  const context = {
    args: {turn_timeout_min: 1 / 60},
    setTimeout(fn) { const id = ++nextId; timers.set(id, fn); return id },
    clearTimeout(id) { timers.delete(id) },
  }
  const wrap = vm.runInNewContext(`${block}\nwithTurnTimeout`, context)
  let settle
  const activation = new Promise(resolve => { settle = resolve })
  let observed = false
  const result = wrap(activation, 'Generate').catch(error => {
    observed = true
    return error
  })
  assert.equal(timers.size, 1)
  const fireTimer = [...timers.values()][0]
  fireTimer()
  await Promise.resolve()
  await Promise.resolve()
  assert.equal(observed, false, 'guard must not return while activation is live')
  settle({ok: true})
  const error = await result
  assert.match(error.message, /turn-timeout: Generate/)
  assert.equal(timers.size, 0)
})
