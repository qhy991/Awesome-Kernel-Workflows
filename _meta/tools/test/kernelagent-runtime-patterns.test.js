'use strict'
// Enforcement test for issue #22.
//
// The Claude Code Workflow runtime FORBIDS `Date.now()`, `Math.random()`, and
// argless `new Date()` — they throw and break resume/replay semantics. Issue #22
// surfaced `Date.now()` at KernelAgent/kernelagent-triton-synthesis.js:514, which
// crashed the workflow mid-dispatch (after the Setup-phase agents had already
// burned ~90-127K tokens) and could not resume past line 514.
//
// Why this is a source-level grep and NOT an execution test: the repo's prompt-
// capture harness (_meta/tools/lib/run-workflow.js) runs workflows under Node's
// `vm`, which exposes `Date` automatically — so it does NOT reproduce the real
// Workflow runtime's throw. Only a source-level assertion catches this class of
// bug. Mirrors the issue's acceptance grep.
//
// Run: node --test _meta/tools/test/kernelagent-runtime-patterns.test.js

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'KernelAgent/kernelagent-triton-synthesis.js')

// Forbidden by the Workflow runtime: nondeterministic built-ins that throw and
// break resume. `new Date()` is matched argless (with-args is also disallowed by
// the runtime, but argless is the form the docs call out and the one that ever
// appears in practice).
const FORBIDDEN = /Date\.now\(\)|Math\.random\(\)|new Date\(\s*\)/

test('KernelAgent workflow source contains no Workflow-forbidden runtime patterns', () => {
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  const hits = []
  src.split('\n').forEach((line, i) => {
    const m = line.match(FORBIDDEN)
    if (m) hits.push(`${i + 1}: ${line.trim()}`)
  })
  assert.equal(hits.length, 0,
    `KernelAgent uses Workflow-forbidden nondeterministic built-ins (they throw and break resume):\n${hits.join('\n')}`)
})

test('sessionDir is derived deterministically (args-driven), not from a runtime clock', () => {
  // Regression guard for issue #22: sessionDir must not depend on Date.now().
  // It is a reported label only (no filesystem writes), so any deterministic
  // value is acceptable — but it MUST be args-derived, not clock-derived.
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  const line = src.split('\n').find((l) => /sessionDir\s*=/.test(l) && /session_/.test(l))
  assert.ok(line, 'expected a `sessionDir = ...session_...` assignment')
  assert.ok(!/Date\.now\(\)|new Date\(\s*\)/.test(line),
    `sessionDir must not use a runtime clock: ${line.trim()}`)
  assert.ok(/args\./.test(line),
    `sessionDir should be args-derived for resume determinism: ${line.trim()}`)
})
