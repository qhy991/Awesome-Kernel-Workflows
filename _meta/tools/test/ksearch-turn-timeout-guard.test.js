'use strict'
// Verify KSearch carries a per-turn wall-clock watchdog on the Generate doer
// turn (parity with CUDAAgent #12/#14; propagated by #30). KSearch already
// bounds the *eval* step with EVAL_TIMEOUT_SEC (shell `timeout Ns`), but a
// hung non-eval agent() turn (Generate stalled in_progress) had no wall-clock
// cap and could stall the search indefinitely. Static-source assertion is
// sufficient: the guard is a deterministic Promise.race(setTimeout) wrapper
// around the Generate agentRetry call, and a timeout breaks the attempt loop
// (treated like stagnation) so the search continues with the next cycle.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'KSearch/ksearch-kernel-optimization.js')
const SOURCE = fs.readFileSync(WORKFLOW, 'utf8')

test('ksearch: withTurnTimeout scaffolding inlined from _meta/scaffolding/turn-timeout.js', () => {
  assert.match(SOURCE, /--- BEGIN inlined turn-timeout scaffolding \(from _meta\/scaffolding\/turn-timeout\.js\) ---/,
    'KSearch must inline the turn-timeout scaffolding (single source of truth)')
  assert.match(SOURCE, /function withTurnTimeout\(promise, label\)/,
    'withTurnTimeout must be defined')
})

test('ksearch: TURN_TIMEOUT_MS sourced from args.turn_timeout_min with 12-minute default', () => {
  assert.match(SOURCE, /const TURN_TIMEOUT_MS = \(args\.turn_timeout_min \|\| 12\) \* 60 \* 1000/,
    'TURN_TIMEOUT_MS must read args.turn_timeout_min and default to 12 min (parity with CUDAAgent)')
})

test('ksearch: withTurnTimeout uses Promise.race + setTimeout (no forbidden APIs)', () => {
  assert.match(SOURCE, /return Promise\.race\(\[promise, guard\]\)/,
    'withTurnTimeout must race the doer promise against the setTimeout guard')
  assert.match(SOURCE, /typeof setTimeout !== 'function'/,
    'withTurnTimeout must degrade to passthrough when the runtime exposes no timers')
  const guard = SOURCE.match(/function withTurnTimeout[\s\S]*?^}/m)
  assert.ok(guard, 'withTurnTimeout body must be locatable')
  assert.doesNotMatch(guard[0], /Date\.now\(\)|Math\.random\(\)|new Date\(\)/,
    'withTurnTimeout must not use forbidden runtime APIs')
})

test('ksearch: all three Generate doer-turn variants are wrapped with withTurnTimeout', () => {
  // The Generate if/else chain has three prompt variants (gen / debug / improve);
  // each must be wrapped so a hung turn cannot stall the attempt loop.
  assert.match(SOURCE, /withTurnTimeout\(agentRetry\(\(\) => agent\(`You are an expert \$\{langToken\(LANGUAGE\)\} kernel developer\. Generate a high-performance kernel/,
    'first-attempt (gen) Generate turn must be wrapped')
  assert.match(SOURCE, /withTurnTimeout\(agentRetry\(\(\) => agent\(`You are an expert \$\{langToken\(LANGUAGE\)\} kernel developer\. The previous attempt has bugs/,
    'debug Generate turn must be wrapped')
  assert.match(SOURCE, /withTurnTimeout\(agentRetry\(\(\) => agent\(`You are an expert \$\{langToken\(LANGUAGE\)\} kernel developer\. You have a working solution/,
    'improve Generate turn must be wrapped')
})

test('ksearch: Generate chain translates a turn timeout into an early cycle exit', () => {
  // A timeout must not crash the run; it must break the attempt loop (treated
  // like stagnation) so the search continues with the next cycle.
  assert.match(SOURCE, /} catch \(e\) \{\s*\n\s*log\(`  Cycle \$\{cycle \+ 1\} attempt \$\{attempt \+ 1\}: turn watchdog tripped — ending cycle/,
    'a turn-timeout reject must be caught and logged')
  assert.match(SOURCE, /turn watchdog tripped — ending cycle[\s\S]*?break/,
    'a turn timeout must break out of the attempt loop')
})
