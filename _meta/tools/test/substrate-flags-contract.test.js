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

test('(b) REPORT-ONLY: workflow call-sites passing non-common substrate flags (--kernel/--result/--test)', () => {
  // These flags are NOT in the common cuda/metal/metax/triton/rocm schema (ascend
  // accepts --kernel/--result as aliases; no substrate accepts bare --test). They
  // are the PR 2b fix targets. Reported, not enforced, in PR 2a.
  const skip = new Set(['_tools', '_meta', '_substrate', '_templates', 'scripts', 'node_modules', '.git'])
  const re = /--(?:kernel|result|test)\b/g
  const hits = []
  for (const entry of fs.readdirSync(ROOT, { withFileTypes: true })) {
    if (!entry.isDirectory() || skip.has(entry.name)) continue
    const dir = path.join(ROOT, entry.name)
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith('.js')) continue
      const full = path.join(dir, f)
      const src = fs.readFileSync(full, 'utf8')
      const lines = src.split('\n')
      for (let i = 0; i < lines.length; i++) {
        if (re.test(lines[i])) hits.push(`${path.relative(ROOT, full)}:${i + 1}: ${lines[i].trim().slice(0, 120)}`)
        re.lastIndex = 0
      }
    }
  }
  // Non-blocking: log only.
  if (hits.length) {
    console.log(`[substrate-flags-contract (b) REPORT-ONLY] ${hits.length} call-site(s) pass non-common substrate flags (PR 2b targets):\n${hits.join('\n')}`)
  } else {
    console.log('[substrate-flags-contract (b) REPORT-ONLY] no non-common substrate flag call-sites found')
  }
  assert.ok(true)
})
