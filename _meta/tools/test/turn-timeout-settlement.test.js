'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const root = path.resolve(__dirname, '../../..')
const retry = fs.readFileSync(path.join(root, '_meta/scaffolding/agent-retry.js'), 'utf8')
const timeout = fs.readFileSync(path.join(root, '_meta/scaffolding/turn-timeout.js'), 'utf8')

function scope(source = timeout, timers = true) {
  let expire, cleared = 0
  const context = {args:{turn_timeout_min:1}, ...(timers ? {
    setTimeout:callback => {expire = callback; return 1},
    clearTimeout:() => {cleared += 1},
  } : {})}
  const api = vm.runInNewContext(retry + '\n' + source + '\n;({withTurnTimeout,isTurnTimeout})', context)
  return {...api, expire:() => expire(), cleared:() => cleared}
}
function deferred() {
  let resolve, reject
  const promise = new Promise((a,b) => {resolve=a;reject=b})
  return {promise,resolve,reject}
}

for (const outcome of ['value','null','error']) {
  test(`deadline waits for active ${outcome} outcome and prevents another attempt`, async () => {
    const timer = scope(), child = deferred()
    let calls = 0, finished = false, lateWrite = false
    const pending = timer.withTurnTimeout(() => {calls += 1; return child.promise}, 'impl', {retries:5})
    pending.then(() => {finished=true}, () => {finished=true})
    timer.expire()
    await Promise.resolve(); await Promise.resolve()
    assert.equal(finished, false, 'expiry must not finish the parent while its child is pending')
    lateWrite = true
    const original = new Error('retryable transport failure')
    if (outcome==='error') child.reject(original)
    else child.resolve(outcome==='null' ? null : {ok:true})
    await assert.rejects(pending, error => outcome==='error' ? error===original
      : error.code==='KERSOR_TURN_TIMEOUT' && error.retryable===false)
    assert.equal(calls, 1)
    assert.equal(lateWrite, true)
    assert.equal(timer.cleared(), 1)
  })
}

test('deadline cannot hide a late permission or policy refusal', async () => {
  const timer = scope(), child = deferred()
  const original = Object.assign(new Error('permission refused'), {code:'KERSOR_PERMISSION_DENIED',retryable:false})
  const pending = timer.withTurnTimeout(() => child.promise, 'impl', {retries:5})
  timer.expire(); child.reject(original)
  await assert.rejects(pending, error => error === original)
  assert.equal(timer.isTurnTimeout(original), false)
})

test('timely retries and null policy are preserved', async () => {
  const timer=scope()
  let calls=0
  const value=await timer.withTurnTimeout(() => ++calls===1 ? null : {ok:true}, 'impl', {retries:1})
  assert.equal(value.ok,true);assert.equal(calls,2);assert.equal(timer.cleared(),1)
  assert.equal(await scope().withTurnTimeout(() => null, 'optional', {retries:0,allowNull:true}),null)
})

test('a zero-retry late null is a drained timeout', async () => {
  const timer=scope(), child=deferred()
  const pending=timer.withTurnTimeout(() => child.promise,'impl',{retries:0})
  timer.expire();child.resolve(null)
  await assert.rejects(pending,error => timer.isTurnTimeout(error))
})

test('without sandbox timers the helper retains bounded retry behavior', async () => {
  let calls=0
  const value=await scope(timeout,false).withTurnTimeout(() => ++calls===1 ? null : 'done','impl',{retries:1})
  assert.equal(value,'done');assert.equal(calls,2)
})

test('a pre-started promise is rejected as an invalid helper contract', async () => {
  await assert.rejects(scope().withTurnTimeout(Promise.resolve('old'),'impl'),
    error => error.code==='KERSOR_REQUEST_INVALID' && error.retryable===false)
})

test('the retained baseline demonstrates parent completion before child settlement', async () => {
  const old = fs.readFileSync(path.join(__dirname,'fixtures/turn-timeout-before-drain.js'),'utf8')
  const timer=scope(old), child=deferred()
  let childFinished=false
  const pending=timer.withTurnTimeout(child.promise,'impl')
  timer.expire()
  await assert.rejects(pending,error=>error.code==='KERSOR_TURN_TIMEOUT')
  assert.equal(childFinished,false)
  childFinished=true;child.resolve('late result')
})

for (const retries of [0,1,5]) {
  test(`late rejection preserves its identity with ${retries} retries remaining`,async()=>{
    const timer=scope(), child=deferred(), original=new Error('transport failed')
    let calls=0
    const pending=timer.withTurnTimeout(()=>{calls++;return child.promise},'impl',{retries})
    timer.expire();child.reject(original)
    await assert.rejects(pending,e=>e===original)
    assert.equal(calls,1)
  })
}
