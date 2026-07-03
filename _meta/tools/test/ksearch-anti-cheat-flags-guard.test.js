'use strict'
// Verify #42: KSearch calls anti_cheat.py with the script's actual CLI flags
// (--source/--metrics), not the stale --kernel/--result.
//
// anti_cheat.py's argparse (in _substrate/anti_cheat.py) accepts --source,
// --source-text, --vendor-patterns-file, and --metrics (REQUIRED). The docstring
// shows `anti_cheat.py --source kernel.cu --metrics metrics.json`. KSearch (and
// 4 other workflows) lagged the #25 substrate sync and passed --kernel/--result,
// which argparse rejects (--metrics required) — the anti_cheat check silently
// failed. This test pins KSearch's two anti_cheat calls to --source/--metrics.
//
// Note: KSearch's `integration_strategist.py --kernel` call is CORRECT and
// intentionally NOT flagged — integration_strategist.py's own argparse accepts
// --kernel (it is a strategist script with its own CLI, not a #25 run.sh call).
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'KSearch/ksearch-kernel-optimization.js')
const SOURCE = fs.readFileSync(WORKFLOW, 'utf8')

test('ksearch: anti_cheat.py calls use --source/--metrics (the script CLI), not --kernel/--result', () => {
  const calls = SOURCE.match(/anti_cheat\.py [^\n]+/g) || []
  assert.ok(calls.length >= 2, `expected >=2 anti_cheat.py calls; got ${calls.length}`)
  for (const c of calls) {
    assert.match(c, /--source \$\{kPath\}/, 'anti_cheat.py call must pass --source ${kPath}')
    assert.match(c, /--metrics /, 'anti_cheat.py call must pass --metrics <result.json>')
    assert.doesNotMatch(c, /--kernel /, 'anti_cheat.py call must NOT pass --kernel (script CLI is --source)')
    assert.doesNotMatch(c, /--result /, 'anti_cheat.py call must NOT pass --result (script CLI is --metrics)')
  }
})

test('ksearch: integration_strategist.py --kernel call is preserved (correct — its own CLI)', () => {
  // integration_strategist.py accepts --kernel (its argparse); this is NOT a #25
  // run.sh call. Guard against a future "fix all --kernel" sweep breaking it.
  assert.match(SOURCE, /integration_strategist\.py resolve[\s\S]{0,80}--kernel "\$\{INTEG_KERNEL_PATH\}"/,
    'integration_strategist.py resolve call must keep --kernel (the script accepts it)')
})
