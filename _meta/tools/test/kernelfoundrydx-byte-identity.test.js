'use strict'
// RE-BASELINE: kernelfoundrydx-golden.json was captured from the PRE-RETROFIT
// KernelFoundryDx body at the commit recorded in
// _meta/tools/fixtures/KERNELFOUNDRYDX-GOLDEN-BASELINE-SHA.txt. Do NOT regenerate
// from the current tree once Stage-B retrofit lands — that would make this
// gate tautological. To intentionally change a legacy-path prompt:
//   (1) git worktree the SHA in KERNELFOUNDRYDX-GOLDEN-BASELINE-SHA.txt,
//   (2) apply the SAME logical edit there,
//   (3) re-run the capture with --agent-returns into the golden,
//   (4) commit the new golden + new SHA alone, explaining the intent.
// Stage-A commit SHA: f540c2f6a2f08048a3b02e3fbcea0b9fd4e86b90
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

const WORKFLOW = path.join(ROOT, 'KernelFoundryDx/kernelfoundrydx-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')

function load(f) { return JSON.parse(fs.readFileSync(path.join(FIX, f), 'utf8')) }

const ARGS = load('kernelfoundrydx-args.json')
const RETURNS = load('kernelfoundrydx-agent-returns.json')
const GOLDEN = load('kernelfoundrydx-golden.json')

// Expected label sequence — Setup (read-task, baseline-and-seed), Init (2 seeds
// + 2 anti-cheat validators), one iteration of Evolve/Evaluate/Diagnose across
// 2 islands, then Report.
const EXPECTED_LABELS = [
  'read-task',
  'baseline-and-seed',
  'seed-0',
  'seed-1',
  'validate-seed-0',
  'validate-seed-1',
  'mutate-0-isl0',
  'mutate-0-isl1',
  'eval-0-isl0',
  'eval-0-isl1',
  'diagnose-0-isl0',
  'diagnose-0-isl1',
  'final-report',
]

const EXPECTED_PHASES = [
  'Setup',
  'Setup',
  'Init',
  'Init',
  'Init',
  'Init',
  'Evolve',
  'Evolve',
  'Evaluate',
  'Evaluate',
  'Diagnose',
  'Diagnose',
  'Report',
]

test('kernelfoundrydx legacy path: rendered label set matches the enumerated 13-key sequence', async () => {
  const captured = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(captured.map(c => c.label), EXPECTED_LABELS,
    'label set drifted — agentReturns map must unlock read-task, baseline-and-seed, ' +
    'seed-0/1, validate-seed-0/1, mutate-0-isl0/1, eval-0-isl0/1, diagnose-0-isl0/1, final-report.')
  assert.deepEqual(captured.map(c => c.phase), EXPECTED_PHASES,
    'phase tags drifted from Setup/Init/Evolve/Evaluate/Diagnose/Report sequence.')
})

test('kernelfoundrydx legacy path: prompt sequence is byte-identical to pre-retrofit golden', async () => {
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

test('kernelfoundrydx legacy path: capturePrompts is deterministic across repeated invocations', async () => {
  const a = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const b = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(a, b,
    'workflow body must be deterministic — no Math.random/Date.now/performance.now in seam-rendered prompts.')
})

test('kernelfoundrydx legacy path: stable-stringified capture is byte-identical to frozen golden file (UPDATE_GOLDEN=1 to regenerate)', async () => {
  const goldenPath = path.join(FIX, 'kernelfoundrydx-golden.json')
  const calls = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const captured = stableStringify(calls)
  if (process.env.UPDATE_GOLDEN === '1') {
    fs.writeFileSync(goldenPath, captured)
    return
  }
  const golden = fs.readFileSync(goldenPath, 'utf8')
  assert.strictEqual(captured, golden, 'pre-retrofit kernelfoundrydx golden drifted; re-run with UPDATE_GOLDEN=1 only if intentional')
})

test('kernelfoundrydx re-baseline SHA file is present and non-empty', () => {
  const sha = fs.readFileSync(path.join(FIX, 'KERNELFOUNDRYDX-GOLDEN-BASELINE-SHA.txt'), 'utf8').trim()
  assert.match(sha, /^[0-9a-f]{40}$/, 'SHA file must contain a full 40-char git commit hash')
  const me = fs.readFileSync(__filename, 'utf8')
  assert.ok(me.includes(sha), 'this test file must cite the SHA from KERNELFOUNDRYDX-GOLDEN-BASELINE-SHA.txt for human-readable provenance')
})
