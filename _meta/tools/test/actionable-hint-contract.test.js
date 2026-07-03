'use strict'
// Verify #48: the substrate requires every evidence-schema insight to carry an
// actionable_hint (a concrete next-step, not just an observation).
//
// Layers: SOLVER-SDK (evidence_schema.py — TEMPLATE + _validate_item require it),
// perf_to_evidence.py (the profiling normalizer emits it on every insight), and
// the insight-emitting workflows (Generalist, AccelOpt). claim = what we saw;
// actionable_hint = what to do about it — downstream rounds get a diagnosis WITH
// an action.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { execSync } = require('node:child_process')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const SCHEMA = fs.readFileSync(path.join(ROOT, '_substrate/evidence_schema.py'), 'utf8')
const PERF = fs.readFileSync(path.join(ROOT, '_substrate/profiling/perf_to_evidence.py'), 'utf8')
const GENERALIST = fs.readFileSync(path.join(ROOT, 'Generalist/generalist-kernel-optimization.js'), 'utf8')
const ACCELOPT = fs.readFileSync(path.join(ROOT, 'AccelOpt/accelopt-kernel-optimization.js'), 'utf8')

test('SOLVER-SDK: evidence_schema TEMPLATE + _validate_item require actionable_hint', () => {
  assert.match(SCHEMA, /"actionable_hint":/, 'TEMPLATE insight must include actionable_hint')
  assert.match(SCHEMA, /if not it\.get\("actionable_hint"\):\s*\n\s*errs\.append\(f"\{where\}\.actionable_hint missing\/empty"\)/,
    '_validate_item must flag a missing/empty actionable_hint')
})

// Run evidence_schema.py validate; return parsed JSON. evidence_schema exits 1
// on invalid input (still printing the JSON to stdout), so capture via try/catch.
function validate(record) {
  const cmd = `python3 ${ROOT}/_substrate/evidence_schema.py validate -`
  let stdout
  try {
    stdout = execSync(cmd, { input: JSON.stringify(record), stdio: ['pipe', 'pipe', 'ignore'] }).toString()
  } catch (e) {
    stdout = (e.stdout && e.stdout.toString()) || ''
  }
  return JSON.parse(stdout)
}

test('SOLVER-SDK: evidence_schema rejects an insight WITHOUT actionable_hint (functional)', () => {
  const bad = { attempt_id: 'a1', compiled: true, correct: true, speedup: 1.0,
    metrics: {}, convergence_status: 'unknown',
    insights: [{ kind: 'bottleneck', directive: 'explore', evidence: 'ncu', confidence: 'measured', claim: 'x' }] }
  const j = validate(bad)
  assert.equal(j.valid, false, 'an insight without actionable_hint must be invalid')
  assert.ok(j.errors.some((e) => e.includes('actionable_hint missing/empty')), `errors must name actionable_hint; got ${JSON.stringify(j.errors)}`)
})

test('SOLVER-SDK: evidence_schema accepts an insight WITH actionable_hint (functional)', () => {
  const good = { attempt_id: 'a1', compiled: true, correct: true, speedup: 1.0,
    metrics: {}, convergence_status: 'unknown',
    insights: [{ kind: 'bottleneck', directive: 'explore', evidence: 'ncu', confidence: 'measured', claim: 'x', actionable_hint: 'tile to 128' }] }
  const j = validate(good)
  assert.equal(j.valid, true, 'an insight with actionable_hint must be valid')
})

test('perf_to_evidence.py: every emitted insight carries actionable_hint', () => {
  // bottleneck insight (derived from _hint_for) + validated_win + failed_strategy
  assert.match(PERF, /"claim": f"\{row\['op'\]}\(\{row\['params'\]}\): \{claim\}",\s*\n\s*"actionable_hint": _hint_for\(bclass\)/,
    'bottleneck insight must carry actionable_hint: _hint_for(bclass)')
  assert.match(PERF, /validated_win[\s\S]*?"claim": f"strategy '\{strategy_id\}' gave[\s\S]*?"actionable_hint":/,
    'validated_win insight must carry an actionable_hint')
  assert.match(PERF, /kind": "failed_strategy", "directive": "avoid"[\s\S]*?"actionable_hint":/,
    'failed_strategy insight must carry an actionable_hint')
  // _hint_for covers the three bottleneck classes.
  assert.match(PERF, /"memory_bound":|"compute_bound":|"latency_bound":/,
    '_hint_for must map the three bottleneck classes to concrete hints')
})

test('workflow emitters: Generalist + AccelOpt insights carry actionable_hint', () => {
  assert.match(GENERALIST, /claim: `bottleneck_class=\$\{bclass} dominates[\s\S]*?actionable_hint:/,
    'Generalist roundInsightRaw must carry actionable_hint keyed off bclass')
  assert.match(ACCELOPT, /claim: e,\s*\n\s*actionable_hint:/,
    'AccelOpt insightItems must carry actionable_hint')
})
