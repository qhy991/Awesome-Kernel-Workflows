'use strict'
// RE-BASELINE: the goldens (accelopt-today.golden.json, accelopt-today-generate.golden.json)
// were captured from PRE-RETROFIT AccelOpt at the commit recorded in
// _meta/tools/fixtures/GOLDEN-BASELINE-SHA.txt. Do NOT regenerate them from the current
// (retrofitted) tree — that would make this gate tautological. To intentionally change a
// cuda prompt: (1) git worktree the SHA in GOLDEN-BASELINE-SHA.txt, (2) apply the SAME
// logical edit there, (3) re-run the capture with --agent-returns into the golden, (4) commit
// the new golden alone, explaining the intent.
// Stage-A Task 4 commit SHA: 2fa47418e613889725563f79296479a747af56a7
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))
const WORKFLOW = path.join(ROOT, 'AccelOpt/accelopt-kernel-optimization.js')

function load(f) { return JSON.parse(fs.readFileSync(path.join(ROOT, '_meta/tools/fixtures', f), 'utf8')) }

const CASES = [
  {
    name: 'optimize',
    args: load('accelopt-cuda-args.json'),
    agentReturns: load('accelopt-cuda-agent-returns.json'),
    golden: load('accelopt-today.golden.json'),
    expectedLabels: ['read-baseline', 'ncu-baseline', 'plan-0-0', 'impl-0-t-v0',
                     'eval-plan_0_sample_0', 'learn-t', 'final-report'],
  },
  {
    name: 'generate',
    args: load('accelopt-generate-args.json'),
    agentReturns: load('accelopt-generate-agent-returns.json'),
    golden: load('accelopt-today-generate.golden.json'),
    expectedLabels: ['generate-initial-kernel', 'read-baseline', 'ncu-baseline', 'plan-0-0',
                     'impl-0-t-v0', 'eval-plan_0_sample_0', 'learn-t', 'final-report'],
  },
]

for (const C of CASES) {
  test(`cuda legacy path (${C.name}): rendered label set is exactly the unlocked seams`, async () => {
    const captured = await capturePrompts({ workflowPath: WORKFLOW, args: C.args, agentReturns: C.agentReturns })
    assert.deepEqual(captured.map(c => c.label), C.expectedLabels,
      `label set drifted — the agentReturns map must unlock eval/learn(+seed). ` +
      `NOTE: count is NOT a magic 8; it is the enumerated set for ${C.name} mode.`)
  })

  test(`cuda legacy path (${C.name}): prompt sequence is byte-identical to today's AccelOpt`, async () => {
    const captured = await capturePrompts({ workflowPath: WORKFLOW, args: C.args, agentReturns: C.agentReturns })
    assert.equal(captured.length, C.golden.length,
      `agent() call count changed: today=${C.golden.length} retrofit=${captured.length}. If intended, re-baseline per Step 5.`)
    for (let i = 0; i < C.golden.length; i++) {
      const g = C.golden[i], c = captured[i]
      assert.equal(c.label, g.label, `seq ${i}: label drift ${g.label} -> ${c.label}`)
      assert.equal(c.phase, g.phase, `seq ${i} (${g.label}): phase drift ${g.phase} -> ${c.phase}`)
      if (c.prompt !== g.prompt) {
        let k = 0
        while (k < g.prompt.length && k < c.prompt.length && g.prompt[k] === c.prompt[k]) k++
        assert.fail(
          `seq ${i} (${g.label}/${g.phase}): prompt NOT byte-identical at offset ${k}.\n` +
          `  golden : ${JSON.stringify(g.prompt.slice(Math.max(0, k - 30), k + 30))}\n` +
          `  cuda   : ${JSON.stringify(c.prompt.slice(Math.max(0, k - 30), k + 30))}\n` +
          `If this change is intended, re-baseline per Step 5.`)
      }
    }
  })
}

test('cuda legacy path: ncu_* schema aliases still present in baseline schema', async () => {
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  for (const k of ['sm_throughput_pct', 'dram_throughput_pct', 'sectors_per_request', 'ncu_available']) {
    assert.match(src, new RegExp(`\\b${k}\\b`), `legacy ncu baseline-schema key "${k}" missing from retrofit source`)
  }
  for (const k of ['ncu_evidence', 'ncu_comparison', 'ncu_trigger']) {
    assert.match(src, new RegExp(`\\b${k}\\b`), `legacy schema alias "${k}" must be retained (optional) on the cuda path`)
  }
})

test('cuda legacy path: ncu_baseline_profile return key retained', async () => {
  const src = fs.readFileSync(WORKFLOW, 'utf8')
  assert.match(src, /\bncu_baseline_profile\b/, 'return key ncu_baseline_profile must be preserved for back-compat')
  assert.match(src, /\bbaseline_profile\b/, 'neutral alias baseline_profile must be added alongside ncu_baseline_profile')
})

test('re-baseline SHA placeholder is filled (no unresolved <RECORD ...> marker)', () => {
  const me = fs.readFileSync(__filename, 'utf8')
  const MARK = '<' + 'RECORD Stage-A Task 4 commit SHA here'
  assert.ok(!me.includes(MARK),
    'fill the <RECORD ...> placeholder with the SHA from GOLDEN-BASELINE-SHA.txt before committing')
  const sha = fs.readFileSync(path.join(ROOT, '_meta/tools/fixtures/GOLDEN-BASELINE-SHA.txt'), 'utf8').trim()
  assert.ok(me.includes(sha), 'the test comment must cite the GOLDEN-BASELINE-SHA.txt commit SHA')
})
