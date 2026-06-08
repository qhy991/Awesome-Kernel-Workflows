'use strict'
// RE-BASELINE: ksearch-golden.json was captured from the PRE-Stage-B
// KSearch body at the commit recorded in
// _meta/tools/fixtures/KSEARCH-GOLDEN-BASELINE-SHA.txt. Do NOT regenerate
// from the current tree once Stage-B retrofit lands — that would make this
// gate tautological. To intentionally change a legacy-path prompt:
//   (1) git worktree the SHA in KSEARCH-GOLDEN-BASELINE-SHA.txt,
//   (2) apply the SAME logical edit there,
//   (3) re-run the capture with --agent-returns into the golden,
//   (4) commit the new golden + new SHA alone, explaining the intent.
// Stage-A commit SHA: c0434e2ebaf4350ff8c4438621c0c26db983651d
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

const WORKFLOW = path.join(ROOT, 'KSearch/ksearch-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')

function load(f) { return JSON.parse(fs.readFileSync(path.join(FIX, f), 'utf8')) }

const ARGS = load('ksearch-args.json')
const RETURNS = load('ksearch-agent-returns.json')
const GOLDEN = load('ksearch-golden.json')

// Expected label sequence — optimize_existing mode, iterations=2,
// attempts_per_cycle=1; both cycles succeed (no debug fork) -> 14 keys.
const EXPECTED_LABELS = [
  'read-spec',
  'eval-baseline',
  'init-tree',
  'propose-0',
  'select-0',
  'gen-0-0',
  'eval-0-0',
  'refine-0',
  'propose-1',
  'select-1',
  'gen-1-0',
  'eval-1-0',
  'refine-1',
  'final-report',
]

const EXPECTED_PHASES = [
  'Setup',
  'Setup',
  'Initialize',
  'Select',
  'Select',
  'Generate',
  'Evaluate',
  'Refine',
  'Select',
  'Select',
  'Generate',
  'Evaluate',
  'Refine',
  'Report',
]

test('ksearch legacy path: rendered label set matches the K-Search propose/select/generate/eval/refine sequence', async () => {
  const captured = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(captured.map(c => c.label), EXPECTED_LABELS,
    'label set drifted — agentReturns map must unlock read-spec, eval-baseline, init-tree, ' +
    '2x{propose-N, select-N, gen-N-0, eval-N-0, refine-N}, and final-report.')
  assert.deepEqual(captured.map(c => c.phase), EXPECTED_PHASES,
    'phase tags drifted from KSearch Setup/Initialize/Select/Generate/Evaluate/Refine/Report sequence.')
})

test('ksearch legacy path: prompt sequence is byte-identical to pre-retrofit golden', async () => {
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

test('ksearch legacy path: capturePrompts is deterministic across repeated invocations', async () => {
  const a = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const b = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(a, b,
    'workflow body must be deterministic — KSearch has no RNG; agentReturns map drives selection.')
})

test('ksearch legacy path: stable-stringified capture is byte-identical to frozen golden file (UPDATE_GOLDEN=1 to regenerate)', async () => {
  const goldenPath = path.join(FIX, 'ksearch-golden.json')
  const calls = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const captured = stableStringify(calls)
  if (process.env.UPDATE_GOLDEN === '1') {
    fs.writeFileSync(goldenPath, captured)
    return
  }
  const golden = fs.readFileSync(goldenPath, 'utf8')
  assert.strictEqual(captured, golden, 'pre-retrofit ksearch golden drifted; re-run with UPDATE_GOLDEN=1 only if intentional')
})

test('re-baseline SHA file is present and non-empty', () => {
  const sha = fs.readFileSync(path.join(FIX, 'KSEARCH-GOLDEN-BASELINE-SHA.txt'), 'utf8').trim()
  assert.match(sha, /^[0-9a-f]{40}$/, 'SHA file must contain a full 40-char git commit hash')
  const me = fs.readFileSync(__filename, 'utf8')
  assert.ok(me.includes(sha), 'this test file must cite the SHA from KSEARCH-GOLDEN-BASELINE-SHA.txt for human-readable provenance')
})
