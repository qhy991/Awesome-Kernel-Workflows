'use strict'
// Verify KSearch surfaces an eval wall-clock budget. Without it, fattn R1
// ran 90+ minutes of correctness rechecks while never emitting a benchmark
// latency. Static-source assertion is sufficient: the budget is rendered
// into deterministic prompts and the embedded paths wrap build/test/bench
// with coreutils `timeout`.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'KSearch/ksearch-kernel-optimization.js')
const MANIFEST = path.join(ROOT, 'KSearch/manifest.yaml')

const SOURCE = fs.readFileSync(WORKFLOW, 'utf8')

test('ksearch: EVAL_TIMEOUT_SEC sourced from args with 600s default', () => {
  assert.match(SOURCE, /const\s+EVAL_TIMEOUT_SEC\s*=/,
    'EVAL_TIMEOUT_SEC must be declared at the top-level config block')
  assert.match(SOURCE, /args\.eval_timeout_sec/,
    'EVAL_TIMEOUT_SEC must read args.eval_timeout_sec')
  assert.match(SOURCE, /:\s*600/,
    'EVAL_TIMEOUT_SEC must default to 600s when args.eval_timeout_sec is unset')
})

test('ksearch: TIMEOUT_WRAP renders the coreutils timeout shell prefix', () => {
  assert.match(SOURCE, /const\s+TIMEOUT_WRAP\s*=\s*`timeout\s+\$\{EVAL_TIMEOUT_SEC\}s`/,
    'TIMEOUT_WRAP must render `timeout Ns` for shell wrapping')
})

test('ksearch: embedded_inplace prompt announces the wall-clock budget', () => {
  const m = SOURCE.match(/EMBEDDED-INPLACE EVAL[\s\S]*?ALWAYS restore/)
  assert.ok(m, 'embedded_inplace block must exist')
  const block = m[0]
  assert.match(block, /WALL-CLOCK BUDGET:\s*\$\{EVAL_TIMEOUT_SEC\}s/,
    'embedded_inplace prompt must surface the wall-clock budget to the agent')
  assert.match(block, /reason:"timeout"/,
    'embedded_inplace prompt must require {reason:"timeout"} on budget overrun')
})

test('ksearch: embedded_inplace Build/Test/Benchmark commands wrapped in TIMEOUT_WRAP', () => {
  const m = SOURCE.match(/EMBEDDED-INPLACE EVAL[\s\S]*?ALWAYS restore/)
  assert.ok(m)
  const block = m[0]
  for (const cmdRe of [/Build:\s*\$\{TIMEOUT_WRAP\}\s*\$\{BUILD_CMD\}/,
                       /Test:\s*\$\{TIMEOUT_WRAP\}\s*\$\{PROJECT_TEST_CMD\}/,
                       /Benchmark:\s*\$\{TIMEOUT_WRAP\}\s*\$\{PROJECT_BENCH_CMD\}/]) {
    assert.match(block, cmdRe,
      `embedded_inplace command must be wrapped with TIMEOUT_WRAP: ${cmdRe}`)
  }
})

test('ksearch: embedded_dispatch __embeddedEvalPlan receives TIMEOUT_WRAP-wrapped commands', () => {
  const idx = SOURCE.indexOf('__embeddedEvalPlan(')
  assert.ok(idx > 0)
  const slice = SOURCE.slice(idx, idx + 800)
  assert.match(slice, /buildCmd:\s*`\$\{TIMEOUT_WRAP\}\s*\$\{BUILD_CMD\}`/,
    'embedded_dispatch buildCmd must be wrapped with TIMEOUT_WRAP')
  assert.match(slice, /testCmd:\s*`\$\{TIMEOUT_WRAP\}\s*\$\{PROJECT_TEST_CMD\}`/,
    'embedded_dispatch testCmd must be wrapped with TIMEOUT_WRAP')
  assert.match(slice, /benchmarkCmd:\s*`\$\{TIMEOUT_WRAP\}\s*\$\{PROJECT_BENCH_CMD\}`/,
    'embedded_dispatch benchmarkCmd must be wrapped with TIMEOUT_WRAP')
})

test('ksearch: main eval prompt surfaces the budget to the LLM', () => {
  // The non-embedded eval path lets the LLM choose how to execute, so the
  // budget must be conveyed in prose rather than via a shell wrapper. There
  // are two `You are a kernel evaluation expert` prompts in the source — the
  // first is the baseline eval (no candidate, no budget needed); the second
  // is the per-attempt eval that the fattn R1 incident bypassed.
  const first = SOURCE.indexOf('You are a kernel evaluation expert')
  const second = SOURCE.indexOf('You are a kernel evaluation expert', first + 1)
  assert.ok(second > first, 'per-attempt eval prompt must exist after the baseline eval')
  const slice = SOURCE.slice(second, second + 1200)
  assert.match(slice, /WALL-CLOCK BUDGET:\s*\$\{EVAL_TIMEOUT_SEC\}s/,
    'per-attempt eval prompt must announce the budget')
  assert.match(slice, /timeout_in_correctness/,
    'per-attempt eval prompt must define the timeout abort signal')
})

test('ksearch manifest documents eval_timeout_sec', () => {
  const text = fs.readFileSync(MANIFEST, 'utf8')
  assert.match(text, /^\s*-\s*name:\s*eval_timeout_sec\b/m,
    'manifest must declare args.eval_timeout_sec so callers can override the budget')
})
