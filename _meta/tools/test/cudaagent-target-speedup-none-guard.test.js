'use strict'
// Verify #41: CUDAAgent handles target_speedup="none" (explore mode) without
// breaking the numeric TARGET_SPEEDUP sites.
//
// Bug: `const TARGET_SPEEDUP = args.target_speedup || 1.05` kept the truthy
// string "none" (explore mode passes "none"), so `compileTime / "none"` = NaN
// and `speedup >= "none"` coerced to NaN — the target check silently broke.
// Fix: parse to a positive number, else null = no target (explore); guard every
// numeric site with `TARGET_SPEEDUP !== null`. Static-source assertion suffices.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'CUDAAgent/cuda-agent-kernel-optimization.js')
const SOURCE = fs.readFileSync(WORKFLOW, 'utf8')

test('cudaagent: TARGET_SPEEDUP parses to a positive number or null (no truthy-string bug)', () => {
  // Must NOT use the bare `|| 1.05` form that keeps "none" as a truthy string.
  assert.doesNotMatch(SOURCE, /const TARGET_SPEEDUP = args\.target_speedup \|\| 1\.05/,
    'TARGET_SPEEDUP must NOT use `args.target_speedup || 1.05` (keeps "none" as a truthy string)')
  assert.match(SOURCE, /const TARGET_SPEEDUP = \(\(\) => \{[\s\S]*?parseFloat\([\s\S]*?Number\.isFinite\(v\) && v > 0 \? v : null[\s\S]*?\}\)\(\)/,
    'TARGET_SPEEDUP must parseFloat and return null for non-numeric/none (explore mode)')
})

test('cudaagent: every numeric TARGET_SPEEDUP site is guarded against null', () => {
  // The three sites that dereference TARGET_SPEEDUP as a number (division at the
  // Target line, the >= comparison, the report line) must each guard null.
  const guards = SOURCE.match(/TARGET_SPEEDUP !== null/g) || []
  assert.ok(guards.length >= 3,
    `expected >=3 "TARGET_SPEEDUP !== null" guards (Target line, target-met check, report line); got ${guards.length}`)
})

test('cudaagent: target-met check skips when TARGET_SPEEDUP is null (explore = no target)', () => {
  assert.match(SOURCE, /if \(TARGET_SPEEDUP !== null && verifyResult\.correct && \(verifyResult\.speedup_vs_compile \|\| 0\) >= TARGET_SPEEDUP\)/,
    'target-met check must be gated on TARGET_SPEEDUP !== null so explore mode (null) never sets targetMet')
})
