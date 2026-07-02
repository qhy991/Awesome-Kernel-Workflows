'use strict'
// Verify ARGUS carries a per-turn wall-clock watchdog (parity with CUDAAgent
// #12/#14; propagated by #30). Without it, a single hung Validator/Lowering
// doer turn — which runs build+test+bench behind the agent — stalls the whole
// linear pipeline indefinitely. Static-source assertion is sufficient: the
// guard is a deterministic Promise.race(setTimeout) wrapper around the doer
// agentRetry calls, and the timeout reject propagates as an attributable
// `turn-timeout:` error that aborts the round cleanly (#20-style).
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'ARGUS/argus-kernel-optimization.js')
const SOURCE = fs.readFileSync(WORKFLOW, 'utf8')

test('argus: withTurnTimeout scaffolding inlined from _meta/scaffolding/turn-timeout.js', () => {
  assert.match(SOURCE, /--- BEGIN inlined turn-timeout scaffolding \(from _meta\/scaffolding\/turn-timeout\.js\) ---/,
    'ARGUS must inline the turn-timeout scaffolding (single source of truth)')
  assert.match(SOURCE, /function withTurnTimeout\(promise, label\)/,
    'withTurnTimeout must be defined')
})

test('argus: TURN_TIMEOUT_MS sourced from args.turn_timeout_min with 12-minute default', () => {
  assert.match(SOURCE, /const TURN_TIMEOUT_MS = \(args\.turn_timeout_min \|\| 12\) \* 60 \* 1000/,
    'TURN_TIMEOUT_MS must read args.turn_timeout_min and default to 12 min (parity with CUDAAgent)')
})

test('argus: withTurnTimeout uses Promise.race + setTimeout (no forbidden APIs)', () => {
  assert.match(SOURCE, /return Promise\.race\(\[promise, guard\]\)/,
    'withTurnTimeout must race the doer promise against the setTimeout guard')
  assert.match(SOURCE, /typeof setTimeout !== 'function'/,
    'withTurnTimeout must degrade to passthrough when the runtime exposes no timers')
  // The forbidden-API scan (KerSor catalog) already rejects Date.now/Math.random
  // workflow-wide; assert the watchdog itself does not introduce any.
  const guard = SOURCE.match(/function withTurnTimeout[\s\S]*?^}/m)
  assert.ok(guard, 'withTurnTimeout body must be locatable')
  assert.doesNotMatch(guard[0], /Date\.now\(\)|Math\.random\(\)|new Date\(\)/,
    'withTurnTimeout must not use forbidden runtime APIs')
})

test('argus: eval-bearing doer turns are wrapped with withTurnTimeout', () => {
  // The stall risk in ARGUS is the Validator/Lowering/baseline-eval turns that
  // run build+test+bench behind the agent — the same failure mode CUDAAgent's
  // Implement/Verify watchdog bounds. Assert those turns are wrapped.
  assert.match(SOURCE, /withTurnTimeout\(agentRetry\(\(\) => agent\(`You are a GPU kernel validator\. Run the baseline kernel/,
    'baseline-eval turn must be wrapped with withTurnTimeout')
  assert.match(SOURCE, /withTurnTimeout\(agentRetry\(\(\) => agent\(`You are the ARGUS Validator Agent/,
    'Validator turn must be wrapped with withTurnTimeout')
  assert.match(SOURCE, /'baseline-eval'\)/,
    "baseline-eval wrap must pass the 'baseline-eval' label")
  assert.match(SOURCE, /'validate'\)/,
    "Validator wrap must pass the 'validate' label")
})
