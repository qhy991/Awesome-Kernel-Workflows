'use strict'
// RE-BASELINE: kernelfoundry-golden.json was captured from the PRE-Stage-B
// KernelFoundry body at the commit recorded in
// _meta/tools/fixtures/KERNELFOUNDRY-GOLDEN-BASELINE-SHA.txt. Do NOT regenerate
// from the current tree once Stage-B retrofit lands — that would make this
// gate tautological. To intentionally change a legacy-path prompt:
//   (1) git worktree the SHA in KERNELFOUNDRY-GOLDEN-BASELINE-SHA.txt,
//   (2) apply the SAME logical edit there,
//   (3) re-run the capture with --agent-returns into the golden,
//   (4) commit the new golden + new SHA alone, explaining the intent.
// Stage-A commit SHA: d283e8f8a0f1db48486d7d590cb19a030c4d3322
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

const WORKFLOW = path.join(ROOT, 'KernelFoundry/kernelfoundry-kernel-optimization.js')
const FIX = path.join(ROOT, '_meta/tools/fixtures')

function load(f) { return JSON.parse(fs.readFileSync(path.join(FIX, f), 'utf8')) }

const ARGS = load('kernelfoundry-args.json')
const RETURNS = load('kernelfoundry-agent-returns.json')
const GOLDEN = load('kernelfoundry-golden.json')

// Expected label sequence — generations=2, meta_prompt_interval=99 (skipped) ->
// setup, {vary-N, eval-N} x 2, final-report = 6 calls.
const EXPECTED_LABELS = [
  'setup',
  'vary-0',
  'eval-0',
  'vary-1',
  'eval-1',
  'final-report',
]

const EXPECTED_PHASES = [
  'Setup',
  'Vary',
  'Evaluate',
  'Vary',
  'Evaluate',
  'Evaluate',
]

test('kernelfoundry legacy path: rendered label set matches the MAP-Elites Setup/Vary/Evaluate sequence', async () => {
  const captured = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(captured.map(c => c.label), EXPECTED_LABELS,
    'label set drifted — agentReturns map must unlock setup, 2x{vary-N, eval-N}, and final-report.')
  assert.deepEqual(captured.map(c => c.phase), EXPECTED_PHASES,
    'phase tags drifted from KernelFoundry Setup/Vary/Evaluate sequence.')
})

test('kernelfoundry legacy path: prompt sequence is byte-identical to pre-retrofit golden', async () => {
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

test('kernelfoundry legacy path: capturePrompts is deterministic across repeated invocations', async () => {
  const a = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const b = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.deepEqual(a, b,
    'workflow body must be deterministic — KernelFoundry uses a fixed PRNG over (generation*13+7), agentReturns map drives the rest.')
})

test('kernelfoundry legacy path: stable-stringified capture is byte-identical to frozen golden file (UPDATE_GOLDEN=1 to regenerate)', async () => {
  const goldenPath = path.join(FIX, 'kernelfoundry-golden.json')
  const calls = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const captured = stableStringify(calls)
  if (process.env.UPDATE_GOLDEN === '1') {
    fs.writeFileSync(goldenPath, captured)
    return
  }
  const golden = fs.readFileSync(goldenPath, 'utf8')
  assert.strictEqual(captured, golden, 'pre-retrofit kernelfoundry golden drifted; re-run with UPDATE_GOLDEN=1 only if intentional')
})

test('re-baseline SHA file is present and non-empty', () => {
  const sha = fs.readFileSync(path.join(FIX, 'KERNELFOUNDRY-GOLDEN-BASELINE-SHA.txt'), 'utf8').trim()
  assert.match(sha, /^[0-9a-f]{40}$/, 'SHA file must contain a full 40-char git commit hash')
  const me = fs.readFileSync(__filename, 'utf8')
  assert.ok(me.includes(sha), 'this test file must cite the SHA from KERNELFOUNDRY-GOLDEN-BASELINE-SHA.txt for human-readable provenance')
})
