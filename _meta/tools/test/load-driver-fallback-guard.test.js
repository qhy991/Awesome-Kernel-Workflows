'use strict'
// Verify load-driver degrades to the legacy path on a transient agent failure
// instead of aborting the whole round (#31b).
//
// Background: load-driver is `{ retries: 5, allowNull: true }`, so a transient
// failure (sustained 429 / agent skipped) returns null after retries. Previously
// `if (!DRIVER || DRIVER.present === false) throw` treated a transient null the
// same as `present===false` (a real config error) and aborted the round. The
// workflow sandbox cannot cat+parse the JSON files directly (all Bash runs
// through agent()), so a literal fs.readFileSync fallback is impossible. The
// fix: on a transient null, warn + continue without idioms (legacy path);
// `present===false` still throws. Static-source assertion suffices.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')

const WORKFLOWS = [
  ['KDA', path.join(ROOT, 'KDA/kda-kernel-workflow.js')],
  ['KernelFoundry', path.join(ROOT, 'KernelFoundry/kernelfoundry-kernel-optimization.js')],
  ['ReGraphT', path.join(ROOT, 'ReGraphT/regrapht-kernel-optimization.js')],
]

for (const [name, wfPath] of WORKFLOWS) {
  const SOURCE = fs.readFileSync(wfPath, 'utf8')

  test(`${name}: load-driver distinguishes transient null from present===false`, () => {
    // The previous `if (!DRIVER || DRIVER.present === false)` collapsed both
    // cases into one throw. The fix must split them: null -> degrade, false -> throw.
    assert.match(SOURCE, /if \(!DRIVER\) \{\s*\n\s*log\(`WARNING: load-driver agent returned null after retries — continuing without backend idioms \(legacy path\)/,
      'a transient null load-driver result must warn and continue (legacy path), not throw')
    assert.match(SOURCE, /\} else if \(DRIVER\.present === false\) \{\s*\n\s*throw new Error\(`No backend driver present at \$\{BACKEND_DIR\}/,
      'present===false must STILL throw (real config error, not a transient failure)')
  })

  test(`${name}: field-derference block guarded behind the non-null branch`, () => {
    // DRIVER.backend_id / .lang_fence etc. must only be dereferenced when DRIVER
    // is present (else-if/else branch), never on the null path.
    const m = SOURCE.match(/if \(!DRIVER\) \{[\s\S]*?\} else if \(DRIVER\.present === false\) \{[\s\S]*?\} else \{([\s\S]*?)\n  \}/)
    assert.ok(m, 'load-driver must be a 3-way if/else-if/else: null | present===false | present')
    const elseBlock = m[1]
    assert.match(elseBlock, /DRIVER\.backend_id/, 'backend_id deref must live in the present (else) branch')
    assert.match(elseBlock, /DRIVER\.lang_fence/, 'lang_fence deref must live in the present (else) branch')
    assert.doesNotMatch(m[0].replace(elseBlock, ''), /DRIVER\.(backend_id|lang_fence|impl_requirements|source_ext)/,
      'no DRIVER field may be dereferenced on the null or present===false paths')
  })

  test(`${name}: load-driver fallback uses no forbidden runtime APIs`, () => {
    const m = SOURCE.match(/#31b: load-driver fallback[\s\S]*?\n  \}/)
    assert.ok(m, 'load-driver fallback block must be locatable')
    assert.doesNotMatch(m[0], /Date\.now\(\)|Math\.random\(\)|new Date\(\)/,
      'the load-driver fallback must not use forbidden runtime APIs')
  })
}
