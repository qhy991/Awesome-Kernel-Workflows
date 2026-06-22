'use strict'
// Verify KSearch separates test_command from benchmark_command in both
// embedded eval branches. Before this split, both Test and Benchmark steps
// shared PROJECT_BENCH_CMD, so the NMSE/correctness gate would silently run
// the perf path instead of a correctness path. Static-source assertion is
// sufficient: the embedded prompts are deterministic strings that bake in
// the PROJECT_TEST_CMD / PROJECT_BENCH_CMD identifiers we care about.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const yaml = (() => { try { return require('js-yaml') } catch { return null } })()

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'KSearch/ksearch-kernel-optimization.js')
const MANIFEST = path.join(ROOT, 'KSearch/manifest.yaml')

const SOURCE = fs.readFileSync(WORKFLOW, 'utf8')

test('ksearch: PROJECT_TEST_CMD is declared and falls back to PROJECT_BENCH_CMD', () => {
  assert.match(SOURCE, /const\s+TEST_CMD\s*=\s*args\.test_command/,
    'TEST_CMD must be sourced from args.test_command')
  assert.match(SOURCE, /const\s+PROJECT_TEST_CMD\s*=\s*args\.test_command\s*\|\|\s*TEST_CMD\s*\|\|\s*PROJECT_BENCH_CMD/,
    'PROJECT_TEST_CMD must fall back to PROJECT_BENCH_CMD when no test_command is provided (back-compat)')
})

test('ksearch: embedded_inplace branch Test step uses PROJECT_TEST_CMD, Benchmark uses PROJECT_BENCH_CMD', () => {
  // The embedded_inplace prompt is a single template literal containing
  // `Test: ${PROJECT_TEST_CMD}` and `Benchmark: ${PROJECT_BENCH_CMD}`.
  const inplaceBlock = SOURCE.match(/EMBEDDED-INPLACE EVAL[\s\S]*?ALWAYS restore/)
  assert.ok(inplaceBlock, 'embedded_inplace prompt block must exist')
  const block = inplaceBlock[0]
  assert.match(block, /Test:\s*\$\{PROJECT_TEST_CMD\}/,
    'Test step in embedded_inplace prompt must use PROJECT_TEST_CMD, not PROJECT_BENCH_CMD')
  assert.match(block, /Benchmark:\s*\$\{PROJECT_BENCH_CMD\}/,
    'Benchmark step must use PROJECT_BENCH_CMD')
  // Pin the contract: Test and Benchmark must not be the same identifier.
  assert.ok(
    !/Test:\s*\$\{PROJECT_BENCH_CMD\}/.test(block),
    'Test step must not use PROJECT_BENCH_CMD — that was the bug',
  )
})

test('ksearch: embedded_dispatch __embeddedEvalPlan call separates testCmd and benchmarkCmd', () => {
  // The source contains exactly one __embeddedEvalPlan( call site in KSearch.
  // Find the line and the following few lines to assert testCmd / benchmarkCmd.
  const idx = SOURCE.indexOf('__embeddedEvalPlan(')
  assert.ok(idx > 0, '__embeddedEvalPlan call site must exist')
  const slice = SOURCE.slice(idx, idx + 600)
  assert.match(slice, /testCmd:\s*PROJECT_TEST_CMD/,
    'embedded_dispatch testCmd must be PROJECT_TEST_CMD')
  assert.match(slice, /benchmarkCmd:\s*PROJECT_BENCH_CMD/,
    'embedded_dispatch benchmarkCmd must be PROJECT_BENCH_CMD')
  // Regression pin: testCmd and benchmarkCmd must not both be PROJECT_BENCH_CMD.
  assert.ok(
    !/testCmd:\s*PROJECT_BENCH_CMD/.test(slice),
    'testCmd must not collapse onto PROJECT_BENCH_CMD — that was the bug',
  )
})

test('ksearch manifest documents test_command arg', () => {
  const text = fs.readFileSync(MANIFEST, 'utf8')
  if (yaml) {
    const doc = yaml.load(text)
    const names = (doc.args || []).map(a => a.name)
    assert.ok(names.includes('test_command'),
      `manifest must declare args.test_command; got ${names.join(',')}`)
    assert.ok(names.includes('benchmark_command'),
      'manifest must still declare args.benchmark_command')
  } else {
    assert.match(text, /^\s*-\s*name:\s*test_command\b/m,
      'manifest must declare a test_command arg')
  }
})
