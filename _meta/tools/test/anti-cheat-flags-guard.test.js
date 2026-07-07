'use strict'
// Verify ⑦ (substrate-invocation SSOT) + the #42 anti_cheat flag-fix codemod.
//
// anti_cheat.py's argparse REQUIRES --source / --metrics; 17 workflows hand-wrote
// --kernel / --result (argparse rejected; the check silently failed for every
// candidate). #42 fixed KSearch only; scripts/patch-anti-cheat-flags.js fixed the
// other 17. This test asserts the bug stays fixed everywhere + the SSOT builder
// (substrateAntiCheat in _meta/scaffolding/substrate-invocation.js) carries the
// canonical flag schema for NEW calls.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const SSOT = fs.readFileSync(path.join(ROOT, '_meta/scaffolding/substrate-invocation.js'), 'utf8')

function workflowFiles() {
  return fs.readdirSync(ROOT).filter((d) => {
    const dir = path.join(ROOT, d)
    return fs.statSync(dir).isDirectory() && fs.existsSync(path.join(dir, 'manifest.yaml'))
  }).map((d) => fs.readdirSync(path.join(ROOT, d)).filter((f) => f.endsWith('.js')).map((f) => path.join(ROOT, d, f))).flat()
}

test('SSOT: substrate-invocation.js defines substrateAntiCheat with --source/--metrics', () => {
  assert.match(SSOT, /function substrateAntiCheat\(/, 'substrateAntiCheat builder must exist')
  assert.match(SSOT, /anti_cheat\.py --source \$\{source\} --metrics \$\{metrics\}/,
    'substrateAntiCheat must use --source/--metrics (the script CLI), not --kernel/--result')
})

test('no workflow calls anti_cheat.py with the stale --kernel / --result flags (#42 stays fixed)', () => {
  const offenders = []
  for (const f of workflowFiles()) {
    const src = fs.readFileSync(f, 'utf8')
    const rel = path.relative(ROOT, f)
    // Strip comments so a comment that *mentions* --kernel isn't mistaken for usage.
    const code = src.replace(/^\s*\/\/.*$/gm, '')
    if (/\banti_cheat\.py --kernel\b/.test(code)) offenders.push(`${rel}: --kernel`)
    if (/\banti_cheat\.py\b[\s\S]*?--result\b/.test(code)) {
      // --result is only a bug on an anti_cheat line; check line-scoped
      const badLine = src.split('\n').find((l) => l.includes('anti_cheat.py') && /--result /.test(l))
      if (badLine && !badLine.includes('--metrics')) offenders.push(`${rel}: --result on anti_cheat line`)
    }
  }
  assert.deepEqual(offenders, [], `stale anti_cheat flags found: ${offenders.join('; ')}. Run \`node scripts/patch-anti-cheat-flags.js --all\` to fix.`)
})

test('every workflow anti_cheat.py call uses --source (no stale --kernel/--result)', () => {
  // Asserts the #42 flag-DRIFT fix (--kernel -> --source). A separate concern:
  // anti_cheat.py also REQUIRES --metrics (a result.json file); StitchCUDA's call
  // is incomplete (passes --source only, no --metrics) — a pre-existing bug, not
  // flag drift, tracked separately.
  let calls = 0
  for (const f of workflowFiles()) {
    const src = fs.readFileSync(f, 'utf8')
    const rel = path.relative(ROOT, f)
    for (const line of src.split('\n')) {
      if (line.includes('anti_cheat.py --source')) {
        calls++
        assert.match(line, /--source /, `${rel}: anti_cheat call must use --source`)
        assert.doesNotMatch(line, /--kernel /, `${rel}: anti_cheat call must NOT use --kernel (#42 drift)`)
        assert.doesNotMatch(line, /--result /, `${rel}: anti_cheat call must NOT use --result (#42 drift)`)
      }
    }
  }
  assert.ok(calls >= 17, `expected >=17 anti_cheat --source calls across workflows; got ${calls}`)
})

test('known-incomplete: StitchCUDA anti_cheat call is missing --metrics (separate bug, not #42 drift)', () => {
  // anti_cheat.py REQUIRES --metrics (a result.json file). StitchCUDA passes only
  // --source (no --metrics) — argparse would reject it. This is a pre-existing
  // incomplete-call bug, NOT the #42 flag drift fixed by the codemod. Snapshot'd
  // here so it's tracked; resolve by passing the eval result.json as --metrics.
  const src = fs.readFileSync(path.join(ROOT, 'StitchCUDA/stitchcuda-kernel-optimization.js'), 'utf8')
  const hasIncomplete = src.split('\n').some((l) => l.includes('anti_cheat.py --source') && !l.includes('--metrics'))
  assert.ok(hasIncomplete, 'StitchCUDA anti_cheat call should still be missing --metrics (snapshot); if fixed, update this test')
})
