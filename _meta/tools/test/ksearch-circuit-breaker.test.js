'use strict'
// Verify KSearch carries a run-level stagnation circuit breaker (#31a).
// KSearch already had per-cycle stagnation (STAGNATION_WINDOW) and per-call
// agent-failure handling ({allowNull:true}), but a search that plateaued at the
// GLOBAL best would still burn the remaining MAX_CYCLES budget. The breaker
// mirrors CUDAAgent's STAGNATION_LIMIT: consecutive cycles with no global-best
// improvement -> stop early. (A cycle where the doer kept failing also lands
// here: it produces no new global best, so bestMetric is unchanged at cycle end,
// which is exactly what the breaker detects.) Static-source assertion suffices.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'KSearch/ksearch-kernel-optimization.js')
const SOURCE = fs.readFileSync(WORKFLOW, 'utf8')

test('ksearch: RUN_STAGNATION_LIMIT config sourced from args with default 3', () => {
  assert.match(SOURCE, /const RUN_STAGNATION_LIMIT = args\.run_stagnation_limit \|\| 3/,
    'RUN_STAGNATION_LIMIT must read args.run_stagnation_limit and default to 3 (parity with CUDAAgent STAGNATION_LIMIT)')
})

test('ksearch: run-stagnation counter + cycle-start snapshot declared', () => {
  assert.match(SOURCE, /let runStagnation = 0/,
    'a runStagnation counter must be declared before the cycle loop')
  assert.match(SOURCE, /const bestAtCycleStart = bestMetric/,
    'bestMetric must be snapshotted at the start of each cycle to detect no-improvement')
})

test('ksearch: breaker increments on no global-best improvement and resets on improvement', () => {
  // The increment/reset must key on whether bestMetric advanced this cycle.
  assert.match(SOURCE, /if \(bestMetric === bestAtCycleStart\) \{\s*\n\s*runStagnation\+\+\s*\n\s*\} else \{\s*\n\s*runStagnation = 0\s*\n\s*\}/,
    'breaker must increment runStagnation when bestMetric is unchanged across a cycle and reset it when it improved')
})

test('ksearch: breaker stops the search early at the limit', () => {
  assert.match(SOURCE, /if \(runStagnation >= RUN_STAGNATION_LIMIT\) \{[\s\S]*?break/,
    'reaching RUN_STAGNATION_LIMIT must break out of the cycle loop (stop early)')
  assert.match(SOURCE, /Run-level stagnation: no global-best improvement for/,
    'the early stop must log an attributable reason')
})

test('ksearch: breaker uses no forbidden runtime APIs', () => {
  const m = SOURCE.match(/#31a: Run-level circuit breaker[\s\S]*?cycleCount\+\+/)
  assert.ok(m, 'breaker block must be locatable')
  assert.doesNotMatch(m[0], /Date\.now\(\)|Math\.random\(\)|new Date\(\)/,
    'the circuit breaker must not use forbidden runtime APIs')
})
