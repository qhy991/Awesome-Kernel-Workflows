'use strict'
// Enforcement test for issue #20 null-safety.
//
// (1) Asserts the repo's workflows + templates pass the agent-retry guard linter
//     (0 violations) — catches regressions where someone adds a bare agent() call
//     or an { allowNull: true } without a null-guard.
// (2) Self-tests the linter on synthetic snippets so the checks are known-good.
//
// Run: node --test _meta/tools/test/*.test.js

const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const fs = require('node:fs')
const { checkSrc, run } = require(path.resolve(__dirname, '../../../scripts/check-agent-retry-guards.js'))
const { listAllWorkflows } = require(path.resolve(__dirname, '../../../scripts/add-agent-retry-scaffolding.js'))

const SCAFFOLD = `
async function agentRetry(fn, opts) {
  const retries = (opts && opts.retries != null) ? opts.retries : 5
  let lastError = null
  for (let attempt = 0; attempt <= retries; attempt++) {
    try { const result = await fn(); if (result != null) return result } catch (e) { lastError = e }
  }
  if (lastError) throw lastError
  if (opts && opts.allowNull === true) return null
  throw new Error('agentRetry: returned null')
}
function expect(o, f, c) { if (o == null || o[f] == null) throw new Error('missing'); return o[f] }
function guard(o, f, fb) { if (o == null || o[f] == null) return fb; return o[f] }
`

test('lint: all repo workflows + templates are clean (0 violations)', () => {
  const workflows = listAllWorkflows()
  const templateDirs = ['_meta/templates', '_templates']
  const templates = []
  for (const d of templateDirs) {
    const base = path.resolve(__dirname, '../../..', d)
    if (fs.existsSync(base)) for (const f of fs.readdirSync(base)) if (f.endsWith('.js')) templates.push(path.join(base, f))
  }
  const files = [...workflows, ...templates]
  const vs = run(files)
  assert.equal(vs.length, 0,
    `expected 0 violations but found:\n${vs.map((v) => `  ${path.relative(process.cwd(), v.file)}:${v.line} ${v.code}: ${v.msg}`).join('\n')}`)
})

test('lint: detects an un-wrapped agent() call in a scaffolded workflow', () => {
  const src = SCAFFOLD + `
const diag = await agent('inspect', { label: 'inspect', phase: 'Inspect' })
const bclass = diag.bottleneck_class
`
  const vs = checkSrc(src, 'synthetic.js')
  assert.ok(vs.some((v) => v.code === 'unwrapped-agent'), `expected unwrapped-agent violation, got: ${JSON.stringify(vs)}`)
})

test('lint: does not flag a properly wrapped agent() call', () => {
  const src = SCAFFOLD + `
const diag = await agentRetry(() => agent('inspect', { label: 'inspect', phase: 'Inspect' }), { retries: 5 })
const bclass = guard(diag, 'bottleneck_class', 'unknown')
`
  const vs = checkSrc(src, 'synthetic.js')
  assert.equal(vs.length, 0, `expected clean, got: ${JSON.stringify(vs)}`)
})

test('lint: flags { allowNull: true } result that is never null-guarded', () => {
  const src = SCAFFOLD + `
const gate = await agentRetry(() => agent('gate', { label: 'gate', phase: 'Retrieve' }), { retries: 5, allowNull: true })
const allowed = gate.allowed_methods
`
  const vs = checkSrc(src, 'synthetic.js')
  assert.ok(vs.some((v) => v.code === 'allowNull-unguarded'), `expected allowNull-unguarded violation, got: ${JSON.stringify(vs)}`)
})

test('lint: accepts { allowNull: true } result that IS null-guarded', () => {
  const src = SCAFFOLD + `
const gate = await agentRetry(() => agent('gate', { label: 'gate', phase: 'Retrieve' }), { retries: 5, allowNull: true })
const allowed = (gate && gate.allowed_methods) || []
`
  const vs = checkSrc(src, 'synthetic.js')
  assert.equal(vs.length, 0, `expected clean, got: ${JSON.stringify(vs)}`)
})
