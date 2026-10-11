'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm')
const {migrateCalls}=require('../../../scripts/patch-turn-timeout.js')
const {checkSrc}=require('../../../scripts/check-agent-retry-guards.js')

test('migration preserves the prompt, label and exact retry options',()=>{
  const source='return await withTurnTimeout(agentRetry(() => agent(`x ${f(1, 2)}`, {label:"a,b"}), {retries:0,allowNull:true},), `turn ${i + 1}`)'
  const migrated=migrateCalls(source)
  assert.equal(migrated.count,1)
  assert.equal(migrated.src,'return await withTurnTimeout(() => agent(`x ${f(1, 2)}`, {label:"a,b"}), `turn ${i + 1}`, {retries:0,allowNull:true})')
  new vm.Script('(async function(){'+migrated.src+'})')
  assert.equal(migrateCalls(migrated.src).src,migrated.src)
})

test('migration leaves comments and quoted examples untouched',()=>{
  const source='// withTurnTimeout(agentRetry(() => agent("x"), {retries:5}),"L")\nconst x="withTurnTimeout(agentRetry("'
  assert.equal(migrateCalls(source).src,source)
})

test('migration refuses an unrecognized active shape before rewriting',()=>{
  assert.throws(()=>migrateCalls('withTurnTimeout(agentRetry(existingPromise, {}), "L")'),/requires/)
})

test('the retry guard recognizes the timeout owner but still refuses bare agents',()=>{
  const prefix='function agentRetry() {}\nfunction withTurnTimeout() {}\n'
  assert.deepEqual(checkSrc(prefix+'const x=await withTurnTimeout(() => agent("x", {}),"impl",{retries:0})','fixture.js'),[])
  assert.ok(checkSrc(prefix+'const x=await agent("x", {})','fixture.js').length>0)
})

test('nullable timeout results retain the same dereference guard',()=>{
  const call='const x=await withTurnTimeout(() => agent("x",{}),"L",{retries:0,allowNull:true});'
  const prefix='function agentRetry(){}; function withTurnTimeout(){};'
  assert.ok(checkSrc(prefix+call+'x.field','fixture.js').some(v=>v.code==='allowNull-unguarded'))
  assert.deepEqual(checkSrc(prefix+call+'if(x) x.field','fixture.js'),[])
})
