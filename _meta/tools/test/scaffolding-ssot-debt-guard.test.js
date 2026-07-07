'use strict'
// Verify ③ (SSOT-debt): arg-guard.js + embedded-eval.js give the already-inlined
// snippets a single-source-of-truth home (matching agent-retry.js /
// backend-axis.js). The codemods (patch-arg-guard.js, patch-embedded-eval.js)
// already inline these; the SSOT files let a fix propagate via `--refresh` + a
// guard test catch drift.
//
// Drift detection: every workflow that has the function must byte-match the SSOT
// canonical (a hand-edit to one copy fails this).
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const ARG_GUARD_SSOT = fs.readFileSync(path.join(ROOT, '_meta/scaffolding/arg-guard.js'), 'utf8')
const EMBEDDED_SSOT = fs.readFileSync(path.join(ROOT, '_meta/scaffolding/embedded-eval.js'), 'utf8')

function workflowFiles() {
  return fs.readdirSync(ROOT).filter((d) => {
    const dir = path.join(ROOT, d)
    return fs.statSync(dir).isDirectory() && fs.existsSync(path.join(dir, 'manifest.yaml'))
  }).map((d) => fs.readdirSync(path.join(ROOT, d)).filter((f) => f.endsWith('.js')).map((f) => path.join(ROOT, d, f))).flat()
}

// Extract the canonical __unwrapArgs function + the args= reassignment from the SSOT.
function canonUnwrap() {
  const s = ARG_GUARD_SSOT.indexOf('function __unwrapArgs(rawArgs) {')
  const e = ARG_GUARD_SSOT.indexOf('args = __unwrapArgs(typeof args === \'undefined\' ? undefined : args)')
  return ARG_GUARD_SSOT.slice(s, e + 'args = __unwrapArgs(typeof args === \'undefined\' ? undefined : args)'.length)
}
// Extract the canonical __embeddedEvalPlan function from the SSOT.
function canonEmbedded() {
  const s = EMBEDDED_SSOT.indexOf('function __embeddedEvalPlan(ctx) {')
  const e = EMBEDDED_SSOT.lastIndexOf('}', EMBEDDED_SSOT.length)
  return EMBEDDED_SSOT.slice(s, e + 1)
}

test('SSOT: arg-guard.js + embedded-eval.js define the canonical functions', () => {
  assert.match(ARG_GUARD_SSOT, /function __unwrapArgs\(rawArgs\) \{/, 'arg-guard.js must define __unwrapArgs')
  assert.match(ARG_GUARD_SSOT, /args = __unwrapArgs\(/, 'arg-guard.js must include the args = __unwrapArgs(...) reassignment')
  assert.match(EMBEDDED_SSOT, /function __embeddedEvalPlan\(ctx\) \{/, 'embedded-eval.js must define __embeddedEvalPlan')
  assert.match(EMBEDDED_SSOT, /EMBEDDING_CONTRACT = \[/, 'embedded-eval.js must define EMBEDDING_CONTRACT')
})

test('every workflow with __unwrapArgs byte-matches the arg-guard SSOT (or is a known drift)', () => {
  const canon = canonUnwrap()
  // All workflows with __unwrapArgs now byte-match the SSOT (the 5 previously
  // drifted — AscendC/AutoMegaKernel/InPlacePatch/LlamacppEmbeddedSearch/WarpSpeed —
  // were re-synced to the canonical, including AscendC which gained the key=value
  // regex). An empty set here means any NEW drift fails CI.
  const KNOWN_DRIFT = new Set([])
  let count = 0
  const unexpected = []
  for (const f of workflowFiles()) {
    const src = fs.readFileSync(f, 'utf8')
    const rel = path.relative(ROOT, f)
    const wf = path.basename(path.dirname(f))
    if (src.includes('function __unwrapArgs(rawArgs) {')) {
      count++
      if (!src.includes(canon) && !KNOWN_DRIFT.has(wf)) unexpected.push(rel)
    }
  }
  assert.deepEqual(unexpected, [], `unexpected __unwrapArgs drift (not in known-drift snapshot): ${unexpected.join(', ')}. Run \`node scripts/patch-arg-guard.js --refresh\` to re-sync.`)
  assert.ok(count >= 30, `expected >=30 workflows with __unwrapArgs; got ${count}`)
})

test('every workflow with __embeddedEvalPlan byte-matches the embedded-eval SSOT (no drift)', () => {
  const canon = canonEmbedded()
  let count = 0
  for (const f of workflowFiles()) {
    const src = fs.readFileSync(f, 'utf8')
    const rel = path.relative(ROOT, f)
    if (src.includes('function __embeddedEvalPlan(ctx) {')) {
      count++
      assert.ok(src.includes(canon), `${rel}: __embeddedEvalPlan must byte-match _meta/scaffolding/embedded-eval.js (run \`node scripts/patch-embedded-eval.js --refresh\` to re-sync)`)
    }
  }
  assert.ok(count >= 6, `expected >=6 workflows with __embeddedEvalPlan; got ${count}`)
})
