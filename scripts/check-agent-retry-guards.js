#!/usr/bin/env node
'use strict'
// check-agent-retry-guards.js — enforcement linter for issue #20 null-safety.
//
// Verifies two invariants in every agent()-based workflow:
//   A. No un-wrapped agent() call: if a workflow inlines the agentRetry scaffolding,
//      every agent(...) MUST be wrapped as agentRetry(() => agent(...)). A bare
//      `await agent(...)` whose structured result is dereferenced crashes on a null
//      return (the original #20 bug).
//   B. Every { allowNull: true } result is null-guarded: an agentRetry opted out of
//      the fail-safe throw returns null on failure, so its result variable must be
//      referenced in a null-tolerant way (X && X.field, X?.field, if (X) / if (!X),
//      guard()/expect()). Otherwise a null return crashes at the dereference.
//
// Reuses the analysis primitives from add-agent-retry-scaffolding.js (paren/brace
// aware, comment/string safe). Exits non-zero on any violation. Intended for CI and
// `node --test` (see _meta/tools/test/agent-retry-guard-lint.test.js).

const fs = require('fs')
const path = require('path')
const {
  buildCodeMask,
  findAgentSites,
  buildPairMaps,
  findAgentRetryAssignSites,
  usedNullTolerant,
  listAllWorkflows,
} = require(path.resolve(__dirname, 'add-agent-retry-scaffolding.js'))

const REPO = path.resolve(__dirname, '..')

function offsetToLine(src, off) {
  let line = 1
  for (let i = 0; i < off && i < src.length; i++) if (src[i] === '\n') line++
  return line
}

function checkSrc(src, file) {
  const mask = buildCodeMask(src)
  const violations = []
  const hasScaffold = /\bfunction agentRetry\s*\(/.test(src)

  // Check A: no un-wrapped agent() calls in scaffolded workflows.
  if (hasScaffold) {
    for (const s of findAgentSites(src, mask)) {
      violations.push({
        file,
        line: offsetToLine(src, s.callStart),
        code: 'unwrapped-agent',
        msg: 'agent() call not wrapped in agentRetry(() => …) — a null return will crash the dereference (issue #20)',
      })
    }
  }

  // Check B: every { allowNull: true } result must be null-guarded.
  const maps = buildPairMaps(src, mask)
  for (const s of findAgentRetryAssignSites(src, mask, maps)) {
    const optsSpan = src.slice(s.optsOpenBrace, s.optsCloseBrace + 1)
    if (!/\ballowNull\s*:\s*true\b/.test(optsSpan)) continue
    if (!usedNullTolerant(src, mask, s.closeParen, s.varName)) {
      violations.push({
        file,
        line: offsetToLine(src, s.callStart),
        code: 'allowNull-unguarded',
        msg: `agentRetry result "${s.varName}" uses { allowNull: true } but is never null-guarded (X && X / X?. / if (X) / guard()) — a null return will crash the dereference`,
      })
    }
  }
  return violations
}

function checkFile(file) {
  return checkSrc(fs.readFileSync(file, 'utf8'), file)
}

function run(files) {
  const all = []
  for (const f of files) all.push(...checkFile(f))
  return all
}

function main() {
  const argv = process.argv.slice(2)
  const files = argv.includes('--all') ? listAllWorkflows() : argv.filter((a) => !a.startsWith('--'))
  if (files.length === 0) {
    console.error('usage: check-agent-retry-guards.js [--all | file1.js ...]')
    process.exit(2)
  }
  const vs = run(files)
  for (const v of vs) console.log(`${path.relative(REPO, v.file)}:${v.line}: ${v.code}: ${v.msg}`)
  console.log(`${vs.length === 0 ? 'OK' : 'FAIL'} — ${vs.length} violation(s) across ${files.length} file(s).`)
  process.exit(vs.length === 0 ? 0 : 1)
}

module.exports = { checkSrc, checkFile, run, offsetToLine }
if (require.main === module) main()
