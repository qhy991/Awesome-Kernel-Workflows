'use strict'
// RE-BASELINE: generalist-golden.json was captured from the PRE-Stage-B
// Generalist body at the commit recorded in
// _meta/tools/fixtures/GENERALIST-GOLDEN-BASELINE-SHA.txt. Do NOT regenerate
// from the current tree once Stage-B retrofit lands — that would make this
// gate tautological. To intentionally change a legacy-path prompt:
//   (1) git worktree the SHA in GENERALIST-GOLDEN-BASELINE-SHA.txt,
//   (2) apply the SAME logical edit there,
//   (3) re-run the capture with --agent-returns into the golden,
//   (4) commit the new golden + new SHA alone, explaining the intent.
// Stage-A commit SHA: 1ff85ba8ea4f93eca795c2f82acb8bb077cff434
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

const WORKFLOW = path.join(ROOT, 'Generalist/generalist-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')

function load(f) { return JSON.parse(fs.readFileSync(path.join(FIX, f), 'utf8')) }

const ARGS = load('generalist-args.json')
const RETURNS = load('generalist-agent-returns.json')
const GOLDEN = load('generalist-golden.json')

// Expected label sequence — optimize_existing mode, iterations=2,
// breadth=1; both iterations succeed -> 21 labels.
const EXPECTED_LABELS = [
  'profile-1',
  'diagnose-1',
  'retrieve-1',
  'gate-1',
  'plan-1-1',
  'impl-1-1',
  'anticheat-1-1',
  'learn-1-tiling',
  'refute-1',
  'verify-insight-1',
  'profile-2',
  'diagnose-2',
  'retrieve-2',
  'gate-2',
  'plan-2-1',
  'impl-2-1',
  'anticheat-2-1',
  'learn-2-vectorization',
  'refute-2',
  'verify-insight-2',
  'final-report',
]

const EXPECTED_PHASES = [
  'Profile',
  'Diagnose',
  'Retrieve',
  'Retrieve',
  'Plan',
  'Evaluate',
  'Evaluate',
  'Learn',
  'Learn',
  'Learn',
  'Profile',
  'Diagnose',
  'Retrieve',
  'Retrieve',
  'Plan',
  'Evaluate',
  'Evaluate',
  'Learn',
  'Learn',
  'Learn',
  'Report',
]

test('generalist legacy path: rendered label set matches the beam profile/diagnose/retrieve/plan/evaluate/learn sequence', async () => {
  const captured = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(captured.map(c => c.label), EXPECTED_LABELS,
    'label set drifted — agentReturns map must unlock 2x{profile-N, diagnose-N, retrieve-N, gate-N, ' +
    'plan-N-1, impl-N-1, anticheat-N-1, learn-N-*, refute-N, verify-insight-N}, and final-report.')
  assert.deepEqual(captured.map(c => c.phase), EXPECTED_PHASES,
    'phase tags drifted from Generalist Profile/Diagnose/Retrieve/Plan/Evaluate/Learn/Report sequence.')
})

test('generalist legacy path: prompt sequence is byte-identical to pre-retrofit golden', async () => {
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

test('generalist legacy path: capturePrompts is deterministic across repeated invocations', async () => {
  const a = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const b = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(a, b,
    'workflow body must be deterministic — Generalist has no RNG; agentReturns map drives iteration.')
})

test('generalist legacy path: stable-stringified capture is byte-identical to frozen golden file (UPDATE_GOLDEN=1 to regenerate)', async () => {
  const goldenPath = path.join(FIX, 'generalist-golden.json')
  const calls = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const captured = stableStringify(calls)
  if (process.env.UPDATE_GOLDEN === '1') {
    fs.writeFileSync(goldenPath, captured)
    return
  }
  const golden = fs.readFileSync(goldenPath, 'utf8')
  assert.strictEqual(captured, golden, 'pre-retrofit generalist golden drifted; re-run with UPDATE_GOLDEN=1 only if intentional')
})

test('re-baseline SHA file is present and non-empty', () => {
  const sha = fs.readFileSync(path.join(FIX, 'GENERALIST-GOLDEN-BASELINE-SHA.txt'), 'utf8').trim()
  assert.match(sha, /^[0-9a-f]{40}$/, 'SHA file must contain a full 40-char git commit hash')
  const me = fs.readFileSync(__filename, 'utf8')
  assert.ok(me.includes(sha), 'this test file must cite the SHA from GENERALIST-GOLDEN-BASELINE-SHA.txt for human-readable provenance')
})
