'use strict'
// Verify ④ (typed-args SSOT): the KerSor ②③ channel block is propagated to every
// workflow + byte-matches the SSOT (drift detection).
//
// The block (EXPERIENCE_EXCERPTS + __experienceBlock / ATTEMPT_EVIDENCE+PLAN +
// FAILED_STRATEGY_IDS + __attemptBlock) declares the KerSor dispatch channels so
// every solver can consume them as TYPED data. Was in 5 workflows; the
// patch-typed-args.js codemod wrapped those + propagated to the rest (declaring
// the channels is harmless — consts degrade to null/[] when absent; surfacing
// them in prompts is a per-workflow follow-up).
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const SSOT = fs.readFileSync(path.join(ROOT, '_meta/scaffolding/typed-args.js'), 'utf8')
const BEGIN = '// --- BEGIN inlined typed-args (from _meta/scaffolding/typed-args.js) ---'
const END = '// --- END inlined typed-args ---'

function workflowFiles() {
  return fs.readdirSync(ROOT).filter((d) => {
    const dir = path.join(ROOT, d)
    return fs.statSync(dir).isDirectory() && fs.existsSync(path.join(dir, 'manifest.yaml'))
  }).map((d) => fs.readdirSync(path.join(ROOT, d)).filter((f) => f.endsWith('.js')).map((f) => path.join(ROOT, d, f))).flat()
}

// Canonical block CONTENT (between the SSOT's old BEGIN/END markers, exclusive) —
// what every workflow inlines between the inlined sentinels. Matches the codemod's
// readCanonContent extraction.
function canonContent() {
  const OLD_BEGIN = '// --- BEGIN typed-args (channel ② experience_excerpts) ---'
  const OLD_END = '// --- END typed-args ---'
  const b = SSOT.indexOf(OLD_BEGIN)
  const e = SSOT.indexOf(OLD_END, b)
  return SSOT.slice(b + OLD_BEGIN.length + 1, e - 1)  // between BEGIN+\n and \n+END
}
const CANON = canonContent()

function failedStrategyIds(args) {
  return Function('args', `${CANON}\nreturn FAILED_STRATEGY_IDS`)(args)
}

test('SSOT: typed-args.js defines the canonical ②③ channel block', () => {
  assert.match(SSOT, /EXPERIENCE_EXCERPTS/, 'SSOT must declare EXPERIENCE_EXCERPTS (channel ②)')
  assert.match(SSOT, /ATTEMPT_EVIDENCE/, 'SSOT must declare ATTEMPT_EVIDENCE (channel ③)')
  assert.match(SSOT, /FAILED_STRATEGY_IDS/, 'SSOT must declare FAILED_STRATEGY_IDS (the hard constraint)')
  assert.match(SSOT, /function __experienceBlock/, 'SSOT must define __experienceBlock')
  assert.match(SSOT, /function __attemptBlock/, 'SSOT must define __attemptBlock')
})

test('cumulative failed-strategy ids override the legacy per-round fallback', () => {
  const attempt_evidence = {
    transfer_items: [{ kind: 'failed_strategy', id: 'latest-round-only' }],
  }
  assert.deepEqual(
    failedStrategyIds({ failed_strategy_ids: ['cumulative'], attempt_evidence }),
    ['cumulative'],
  )
  assert.deepEqual(failedStrategyIds({ attempt_evidence }), ['latest-round-only'])
})

test('every workflow has the inlined typed-args block (②③ contract is universal)', () => {
  let count = 0
  for (const f of workflowFiles()) {
    const src = fs.readFileSync(f, 'utf8')
    if (src.includes(BEGIN)) count++
  }
  assert.ok(count >= 30, `expected >=30 workflows with the typed-args block; got ${count}`)
})

test('every inlined typed-args block byte-matches the SSOT (no drift)', () => {
  for (const f of workflowFiles()) {
    const src = fs.readFileSync(f, 'utf8')
    const rel = path.relative(ROOT, f)
    const bi = src.indexOf(BEGIN)
    if (bi === -1) continue
    const ei = src.indexOf(END, bi)
    assert.notEqual(ei, -1, `${rel}: BEGIN without END (malformed)`)
    // The content between the inlined sentinels must byte-match the SSOT canonical.
    const block = src.slice(bi, ei + END.length)
    assert.ok(block.includes(CANON), `${rel}: typed-args block must byte-match _meta/scaffolding/typed-args.js (run \`node scripts/patch-typed-args.js --refresh\` to re-sync)`)
  }
})
