'use strict'
// RE-BASELINE: cudallm-golden.json was captured from the PRE-RETROFIT
// CUDALLM body at the commit recorded in
// _meta/tools/fixtures/CUDALLM-GOLDEN-BASELINE-SHA.txt. Do NOT regenerate
// from the current tree once Stage-B retrofit lands — that would make this
// gate tautological. To intentionally change a legacy-path prompt:
//   (1) git worktree the SHA in CUDALLM-GOLDEN-BASELINE-SHA.txt,
//   (2) apply the SAME logical edit there,
//   (3) re-run the capture with --agent-returns into the golden,
//   (4) commit the new golden + new SHA alone, explaining the intent.
// Stage-A commit SHA: 1614c7a2c32934524cc5067f1f86d697bd0718a0
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

function stableStringify(arr) {
  const sortedArr = arr.map((r) => {
    const sorted = {}
    for (const k of Object.keys(r).sort()) sorted[k] = r[k]
    return sorted
  })
  return JSON.stringify(sortedArr, null, 2)
}

const WORKFLOW = path.join(ROOT, 'CUDALLM/cudallm-fsr-kernel-generation.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')

function load(f) { return JSON.parse(fs.readFileSync(path.join(FIX, f), 'utf8')) }

const ARGS = load('cudallm-args.json')
const RETURNS = load('cudallm-agent-returns.json')
const GOLDEN = load('cudallm-golden.json')

// Expected label sequence — pre-amble (setup, catalog, tests) + nested
// iter x sample loop (iterations=2, samples_per_feature_set=2 → 4 samples;
// each sample renders select-features → generate-kernel → evaluate → reinforce)
// + final-report. Total 20 keys.
const EXPECTED_LABELS = [
  'setup-task',
  'feature-catalog',
  'generate-tests',
  'select-features-0-0',
  'generate-kernel-0-0',
  'evaluate-0-0',
  'reinforce-0-0',
  'select-features-0-1',
  'generate-kernel-0-1',
  'evaluate-0-1',
  'reinforce-0-1',
  'select-features-1-0',
  'generate-kernel-1-0',
  'evaluate-1-0',
  'reinforce-1-0',
  'select-features-1-1',
  'generate-kernel-1-1',
  'evaluate-1-1',
  'reinforce-1-1',
  'final-report',
]

const EXPECTED_PHASES = [
  'Setup',
  'FeatureCatalog',
  'GenerateTests',
  'SelectFeatures', 'GenerateKernel', 'Evaluate', 'Reinforce',
  'SelectFeatures', 'GenerateKernel', 'Evaluate', 'Reinforce',
  'SelectFeatures', 'GenerateKernel', 'Evaluate', 'Reinforce',
  'SelectFeatures', 'GenerateKernel', 'Evaluate', 'Reinforce',
  'Report',
]

test('cudallm legacy path: rendered label set matches the enumerated FSR iter x sample sequence', async () => {
  const captured = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(captured.map(c => c.label), EXPECTED_LABELS,
    'label set drifted — agentReturns map must unlock setup-task, feature-catalog, generate-tests, ' +
    '4x{select-features-i-s, generate-kernel-i-s, evaluate-i-s, reinforce-i-s} for (0,0)(0,1)(1,0)(1,1), ' +
    'and final-report.')
  assert.deepEqual(captured.map(c => c.phase), EXPECTED_PHASES,
    'phase tags drifted from FSR phase sequence.')
})

test('cudallm legacy path: prompt sequence is byte-identical to pre-retrofit golden', async () => {
  const captured = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.equal(captured.length, GOLDEN.length,
    `agent() call count changed: golden=${GOLDEN.length} captured=${captured.length}. If intended, re-baseline per file header.`)
  for (let i = 0; i < GOLDEN.length; i++) {
    const g = GOLDEN[i], c = captured[i]
    assert.equal(c.label, g.label, `seq ${i}: label drift ${g.label} -> ${c.label}`)
    assert.equal(c.phase, g.phase, `seq ${i} (${g.label}): phase drift ${g.phase} -> ${c.phase}`)
    if (c.prompt !== g.prompt) {
      let k = 0
      while (k < g.prompt.length && k < c.prompt.length && g.prompt[k] === c.prompt[k]) k++
      assert.fail(
        `seq ${i} (${g.label}/${g.phase}): prompt NOT byte-identical at offset ${k}.\n` +
        `  golden  : ${JSON.stringify(g.prompt.slice(Math.max(0, k - 30), k + 30))}\n` +
        `  captured: ${JSON.stringify(c.prompt.slice(Math.max(0, k - 30), k + 30))}\n` +
        `If this change is intended, re-baseline per file header.`)
    }
    assert.deepEqual(c.schema, g.schema, `seq ${i} (${g.label}): schema drift`)
  }
})

test('cudallm legacy path: capturePrompts is deterministic across repeated invocations', async () => {
  const a = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const b = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(a, b,
    'workflow body must be deterministic per P5b Spike 2 — no Math.random/Date.now/performance.now in seam-rendered prompts.')
})

test('cudallm legacy path: stable-stringified capture is byte-identical to frozen golden file (UPDATE_GOLDEN=1 to regenerate)', async () => {
  const goldenPath = path.join(FIX, 'cudallm-golden.json')
  const calls = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const captured = stableStringify(calls)
  if (process.env.UPDATE_GOLDEN === '1') {
    fs.writeFileSync(goldenPath, captured)
    return
  }
  const golden = fs.readFileSync(goldenPath, 'utf8')
  assert.strictEqual(captured, golden, 'pre-retrofit cudallm golden drifted; re-run with UPDATE_GOLDEN=1 only if intentional')
})

test('re-baseline SHA file is present and non-empty', () => {
  const sha = fs.readFileSync(path.join(FIX, 'CUDALLM-GOLDEN-BASELINE-SHA.txt'), 'utf8').trim()
  assert.match(sha, /^[0-9a-f]{40}$/, 'SHA file must contain a full 40-char git commit hash')
  const me = fs.readFileSync(__filename, 'utf8')
  assert.ok(me.includes(sha), 'this test file must cite the SHA from CUDALLM-GOLDEN-BASELINE-SHA.txt for human-readable provenance')
})
