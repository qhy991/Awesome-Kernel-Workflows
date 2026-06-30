'use strict'
// Enforcement test for issue #25 — shared substrate-flag schema.
//
// Each substrate's run.sh CLI contract is declared in `_substrate/backends/<B>/flags.yaml`
// (single source of truth). The while/case parser in run.sh is GENERATED from it
// by `_substrate/backends/_gen_flag_parser.py` (between AUTO-GENERATED sentinels).
//
// This test (PR 2a — schema + generator, zero behavior change):
//   (a) every substrate run.sh parser is in sync with its flags.yaml (generator --check);
//   (b) REPORT-ONLY: workflow .js call-sites that pass substrate flags not in the
//       common schema (--kernel/--result/--test) — these are the PR 2b fix targets.
//       Non-blocking here; PR 2b makes it enforcing after the call-sites are fixed.
//
// Run: node --test _meta/tools/test/substrate-flags-contract.test.js

const { test } = require('node:test')
const assert = require('node:assert/strict')
const { execFileSync } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const GEN = path.join(ROOT, '_substrate/backends/_gen_flag_parser.py')
const BACKENDS = ['cuda', 'metal', 'metax', 'triton', 'ascend', 'rocm']

test('(a) every substrate run.sh parser is in sync with its flags.yaml', () => {
  const fails = []
  for (const b of BACKENDS) {
    try {
      const out = execFileSync('python3', [GEN, '--check', b], { cwd: ROOT, encoding: 'utf8' })
      if (!/in sync/.test(out)) fails.push(`${b}: unexpected --check output: ${out.trim()}`)
    } catch (e) {
      fails.push(`${b}: --check failed: ${(e.stdout || e.message).toString().trim()}`)
    }
  }
  assert.equal(fails.length, 0,
    `substrate parsers out of sync with flags.yaml (regenerate via _gen_flag_parser.py --write <backend>):\n${fails.join('\n')}`)
})

test('(b) ENFORCING: run.sh substrate calls (driverSh("run.sh",...) with --artifact) use schema-declared flag names only', () => {
  // PR 2b: the EVALUATE driver-run pattern `driverSh('run.sh', '--artifact ... --kernel ... --result ...')`
  // was fixed to use the schema-declared names (--artifact/--problem/--out). No run.sh call that passes
  // --artifact may also pass the legacy --kernel/--result/--test (not in cuda/metal/metax/triton/rocm
  // flags.yaml; rejected with "unknown arg" exit 3).
  //
  // Out of scope (left as a separate, documented case): KernelAgent's VERIFY-only run.sh calls
  // `driverSh('run.sh', '--kernel X --test Y')` (no --artifact) — a different source-based verify
  // contract that does not match run.sh's run-artifact contract and needs its own resolution.
  // profile.sh calls (different script), anti_cheat.py / integ_probe `resolve --kernel` (different
  // tools with their own CLIs) are also not run.sh and not flagged here.
  const skip = new Set(['_tools', '_meta', '_substrate', '_templates', 'scripts', 'node_modules', '.git'])
  const legacy = /--(?:kernel|result|test)\b/
  const fails = []
  for (const entry of fs.readdirSync(ROOT, { withFileTypes: true })) {
    if (!entry.isDirectory() || skip.has(entry.name)) continue
    const dir = path.join(ROOT, entry.name)
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith('.js')) continue
      const full = path.join(dir, f)
      const lines = fs.readFileSync(full, 'utf8').split('\n')
      for (let i = 0; i < lines.length; i++) {
        const l = lines[i]
        if (/driverSh\(['"]run\.sh['"]/.test(l) && /--artifact/.test(l) && legacy.test(l)) {
          fails.push(`${path.relative(ROOT, full)}:${i + 1}: ${l.trim().slice(0, 140)}`)
        }
      }
    }
  }
  assert.equal(fails.length, 0,
    `run.sh calls (with --artifact) still pass legacy flags (--kernel/--result/--test) rejected by the substrate parser:\n${fails.join('\n')}`)
})
