'use strict'
// Verify #32: AKW workflows declare fair_baseline_id and echo it as baseline_id
// in their returned output (which becomes run-N/output.json).
//
// Contract (verified against KerSor #39, current main):
//   - INPUT:  KerSor resolve-args.sh:191-197 reads contract.env::baseline_id
//             (sourced from test-method.md 'Baseline:' label via
//             consolidate-spec.sh:113,154) and injects it as args.fair_baseline_id
//             ONLY when the workflow declares fair_baseline_id in routing.all_args
//             (resolve-args.sh:303).
//   - OUTPUT: the workflow's returned object is run-N/output.json. It must carry
//             `baseline_id` (the field check-acceptance-gate.sh Check 2c reads
//             via jq -r '.baseline_id // ""' at line ~249). When the workflow
//             echoes args.fair_baseline_id there, Check 2c compares it to
//             contract.env::baseline_id and vetoes on drift; when absent/null
//             the gate skips (back-compat). The field names differ deliberately:
//             fair_baseline_id (input, spec-frozen expectation) vs baseline_id
//             (output, workflow's actual measured axis).
// Static-source assertion suffices.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')

const WORKFLOWS = [
  ['CUDAAgent', path.join(ROOT, 'CUDAAgent/cuda-agent-kernel-optimization.js'), path.join(ROOT, 'CUDAAgent/manifest.yaml')],
  ['KDA', path.join(ROOT, 'KDA/kda-kernel-workflow.js'), path.join(ROOT, 'KDA/manifest.yaml')],
  ['AKO4X', path.join(ROOT, 'AKO4X/ako4x-kernel-optimizer.js'), path.join(ROOT, 'AKO4X/manifest.yaml')],
  ['KernelFoundry', path.join(ROOT, 'KernelFoundry/kernelfoundry-kernel-optimization.js'), path.join(ROOT, 'KernelFoundry/manifest.yaml')],
]

for (const [name, wfPath, mfPath] of WORKFLOWS) {
  const SOURCE = fs.readFileSync(wfPath, 'utf8')
  const MANIFEST = fs.readFileSync(mfPath, 'utf8')

  test(`${name}: declares fair_baseline_id in routing.all_args`, () => {
    // resolve-args.sh:303 only pushes the field when it appears in all_args.
    assert.match(MANIFEST, /all_args:\s*\n\s*-\s+fair_baseline_id\b/,
      'manifest routing.all_args must list fair_baseline_id so KerSor #39(a) pushes it into dispatch-args')
  })

  test(`${name}: return object echoes baseline_id from args.fair_baseline_id`, () => {
    // The returned object becomes output.json; Check 2c reads .baseline_id.
    // Must echo args.fair_baseline_id (the frozen label), NOT args.baseline
    // (a filename — would mismatch the test-method label and trip a false veto).
    assert.match(SOURCE, /return \{\s*\n\s*baseline_id: args\.fair_baseline_id \|\| null/,
      'return object must carry baseline_id: args.fair_baseline_id || null as the first field')
  })

  test(`${name}: does NOT fabricate baseline_id from args.baseline (filename)`, () => {
    // args.baseline is a filename (e.g. reference.py), not the test-method label;
    // using it as baseline_id would drift from contract.env::baseline_id -> veto.
    assert.doesNotMatch(SOURCE, /baseline_id:\s*args\.baseline\b/,
      'baseline_id must NOT be sourced from args.baseline (filename); only args.fair_baseline_id is the frozen label')
  })

  test(`${name}: baseline_id echo uses no forbidden runtime APIs`, () => {
    const m = SOURCE.match(/return \{\s*\n\s*baseline_id:[^\n]*\n/)
    assert.ok(m, 'baseline_id return line must be locatable')
    assert.doesNotMatch(m[0], /Date\.now\(\)|Math\.random\(\)|new Date\(\)/,
      'the baseline_id echo must not use forbidden runtime APIs')
  })
}

test('contract sanity: input field is fair_baseline_id, output field is baseline_id (distinct, deliberate)', () => {
  // Guard against a future refactor collapsing the two names. The asymmetry is
  // the contract: fair_ prefix = spec-frozen expectation (input); no prefix =
  // workflow's actual measured axis (output).
  for (const [, wfPath] of WORKFLOWS) {
    const SOURCE = fs.readFileSync(wfPath, 'utf8')
    assert.match(SOURCE, /args\.fair_baseline_id/, 'input side reads args.fair_baseline_id')
    // output side key is baseline_id (no fair_ prefix)
    assert.match(SOURCE, /baseline_id: args\.fair_baseline_id/, 'output side key is baseline_id, value sourced from args.fair_baseline_id')
  }
})
