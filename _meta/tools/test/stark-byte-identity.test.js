'use strict'
// RE-BASELINE: stark-golden.json was captured from the POST-A0 / PRE-Stage-B
// STARK body at the commit recorded in
// _meta/tools/fixtures/STARK-GOLDEN-BASELINE-SHA.txt. Do NOT regenerate
// from the current tree once Stage-B retrofit lands — that would make this
// gate tautological. To intentionally change a legacy-path prompt:
//   (1) git worktree the SHA in STARK-GOLDEN-BASELINE-SHA.txt,
//   (2) apply the SAME logical edit there,
//   (3) re-run the capture with --agent-returns into the golden,
//   (4) commit the new golden + new SHA alone, explaining the intent.
// Byte-identity is CONDITIONAL on args.rng_seed=42 (pinned in the
// fixture). selectNode()'s RNG is ALWAYS seeded (rng_seed when pinned, else a
// fixed deterministic seed of 1 — never Math.random(); see AWK #50), so the
// capture is deterministic regardless.
// Stage-A commit SHA: 6c75cefc289dfc0acd20276502befe8137c88f3a
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

const WORKFLOW = path.join(ROOT, 'STARK/stark-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')

function load(f) { return JSON.parse(fs.readFileSync(path.join(FIX, f), 'utf8')) }

const ARGS = load('stark-args.json')
const RETURNS = load('stark-agent-returns.json')
const GOLDEN = load('stark-golden.json')

// Expected label sequence — optimize_existing mode (kernel_path provided),
// rng_seed=42, iterations=2; both attempts succeed (no debug fork) -> 9 keys.
const EXPECTED_LABELS = [
  'setup-read-reference',
  'setup-eval-root',
  'plan-root-a1',
  'code-root-a1',
  'eval-a1',
  'plan-node_1-a2',
  'code-node_1-a2',
  'eval-a2',
  'report-final',
]

const EXPECTED_PHASES = [
  'Setup',
  'Setup',
  'Plan',
  'Code',
  'Evaluate',
  'Plan',
  'Code',
  'Evaluate',
  'Report',
]

test('stark legacy path: rendered label set matches the enumerated STARK select/plan/code/eval sequence', async () => {
  const captured = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(captured.map(c => c.label), EXPECTED_LABELS,
    'label set drifted — agentReturns map must unlock setup-read-reference, setup-eval-root, ' +
    '2x{plan-<id>-aN, code-<id>-aN, eval-aN}, and report-final under rng_seed=42.')
  assert.deepEqual(captured.map(c => c.phase), EXPECTED_PHASES,
    'phase tags drifted from STARK Setup/Plan/Code/Evaluate/Report sequence.')
})

test('stark legacy path: prompt sequence is byte-identical to pre-retrofit golden', async () => {
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

test('stark legacy path: capturePrompts is deterministic across repeated invocations (rng_seed=42)', async () => {
  const a = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const b = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(a, b,
    'workflow body must be deterministic per P5b Spike 2 — selectNode() RNG is pinned via rng_seed.')
})

test('stark legacy path: stable-stringified capture is byte-identical to frozen golden file (UPDATE_GOLDEN=1 to regenerate)', async () => {
  const goldenPath = path.join(FIX, 'stark-golden.json')
  const calls = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const captured = stableStringify(calls)
  if (process.env.UPDATE_GOLDEN === '1') {
    fs.writeFileSync(goldenPath, captured)
    return
  }
  const golden = fs.readFileSync(goldenPath, 'utf8')
  assert.strictEqual(captured, golden, 'pre-retrofit stark golden drifted; re-run with UPDATE_GOLDEN=1 only if intentional')
})

test('re-baseline SHA file is present and non-empty', () => {
  const sha = fs.readFileSync(path.join(FIX, 'STARK-GOLDEN-BASELINE-SHA.txt'), 'utf8').trim()
  assert.match(sha, /^[0-9a-f]{40}$/, 'SHA file must contain a full 40-char git commit hash')
  const me = fs.readFileSync(__filename, 'utf8')
  assert.ok(me.includes(sha), 'this test file must cite the SHA from STARK-GOLDEN-BASELINE-SHA.txt for human-readable provenance')
})
