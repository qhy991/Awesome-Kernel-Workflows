'use strict'
// Verify ⑤ (turn-timeout propagation): the withTurnTimeout watchdog capability
// (from _meta/scaffolding/turn-timeout.js) is inlined into every workflow with
// agent-retry, so every iterative workflow HAS the capability available.
//
// The function is DECLARED but not necessarily INVOKED — activation (wrapping
// doer agent() calls with `withTurnTimeout(agentRetry(...), label)`) is a
// per-workflow follow-up. The 3 that already call it (ARGUS, CUDAAgent, KSearch)
// have it from #30; the other 30 got the declaration via patch-turn-timeout.js.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const SSOT = fs.readFileSync(path.join(ROOT, '_meta/scaffolding/turn-timeout.js'), 'utf8')
const BEGIN = '// --- BEGIN inlined turn-timeout scaffolding (from _meta/scaffolding/turn-timeout.js) ---'
const END = '// --- END inlined turn-timeout scaffolding ---'

function workflowFiles() {
  return fs.readdirSync(ROOT).filter((d) => {
    const dir = path.join(ROOT, d)
    return fs.statSync(dir).isDirectory() && fs.existsSync(path.join(dir, 'manifest.yaml'))
  }).map((d) => fs.readdirSync(path.join(ROOT, d)).filter((f) => f.endsWith('.js')).map((f) => path.join(ROOT, d, f))).flat()
}

// Canonical block (from the SSOT's actual const to the last } — the real code,
// not the USAGE-comment example). Matches the codemod's readBlock extraction.
function canonBlock() {
  const s = SSOT.lastIndexOf('const TURN_TIMEOUT_MS =')
  const e = SSOT.lastIndexOf('}')
  return SSOT.slice(s, e + 1)
}
const CANON = canonBlock()

test('SSOT: turn-timeout.js defines TURN_TIMEOUT_MS + withTurnTimeout', () => {
  assert.match(SSOT, /const TURN_TIMEOUT_MS =/, 'SSOT must define TURN_TIMEOUT_MS')
  assert.match(SSOT, /function withTurnTimeout\(promise, label\)/, 'SSOT must define withTurnTimeout')
})

test('every workflow with agent-retry has withTurnTimeout (capability propagated)', () => {
  let withFn = 0, withRetry = 0
  for (const f of workflowFiles()) {
    const src = fs.readFileSync(f, 'utf8')
    if (src.includes('END inlined agent-retry scaffolding')) {
      withRetry++
      if (src.includes('function withTurnTimeout(')) withFn++
    }
  }
  assert.equal(withFn, withRetry, `every workflow with agent-retry should have withTurnTimeout; ${withRetry} have agent-retry, ${withFn} have withTurnTimeout`)
})

test('every inlined turn-timeout block has the canonical const + function (no missing pieces)', () => {
  // Not a strict byte-match (the 3 from #30 include a comment header before the
  // const; the codemod-inserted 30 start at the const). Instead assert the
  // meaningful pieces are present in every sentinel-wrapped block.
  for (const f of workflowFiles()) {
    const src = fs.readFileSync(f, 'utf8')
    const rel = path.relative(ROOT, f)
    const bi = src.indexOf(BEGIN)
    if (bi === -1) continue
    const ei = src.indexOf(END, bi)
    assert.notEqual(ei, -1, `${rel}: BEGIN without END (malformed)`)
    const block = src.slice(bi, ei)
    assert.match(block, /const TURN_TIMEOUT_MS =/, `${rel}: turn-timeout block must include TURN_TIMEOUT_MS`)
    assert.match(block, /function withTurnTimeout\(/, `${rel}: turn-timeout block must include withTurnTimeout`)
  }
})
