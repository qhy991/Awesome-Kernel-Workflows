'use strict'
// Verify #43: KSearch cycle checkpoint (write + resume).
//
// State (decisionTree, solutionDb, bestSolution, bestMetric, cycleCount) was
// in-memory only — a crashed search lost cycles 1..N. Fix: a mechanical agent
// atomically writes ${EXP_DIR}/checkpoint.json at each cycle safe point (the
// resumable state plus the cross-workflow recovery contract), and a mechanical
// agent reads it at startup to resume at the recorded cycle.
//
// #43(a) exponential backoff was SKIPPED — the runtime exposes no sleep() and
// setTimeout-as-blocking-sleep would hang under a non-real-time source; that
// belongs at the runtime layer (spun off as a separate issue). This test only
// covers #43(b) checkpoint. Static-source assertion suffices.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'KSearch/ksearch-kernel-optimization.js')
const SOURCE = fs.readFileSync(WORKFLOW, 'utf8')

test('ksearch: startup loads checkpoint.json and resumes at the recorded cycle', () => {
  assert.match(SOURCE, /label: 'load-checkpoint'/,
    'a load-checkpoint mechanical agent call must exist at startup')
  assert.match(SOURCE, /Read \$\{EXP_DIR\}\/checkpoint\.json[\s\S]*?present.: false/,
    'load-checkpoint must read ${EXP_DIR}/checkpoint.json and return {present:false} if absent')
  assert.match(SOURCE, /_checkpoint\.decisionTree \|\| decisionTree/,
    'resume must restore decisionTree from the checkpoint')
  assert.match(SOURCE, /_checkpoint\.bestMetric != null\) bestMetric = _checkpoint\.bestMetric/,
    'resume must restore bestMetric from the checkpoint')
  assert.match(SOURCE, /_startCycle = Math\.min\(Number\(_checkpoint\.cycle\)/,
    'resume must compute _startCycle from the recorded cycle (clamped to MAX_CYCLES)')
})

test('ksearch: cycle loop starts at _startCycle (resumes, not from 0)', () => {
  assert.match(SOURCE, /for \(let cycle = _startCycle; cycle < MAX_CYCLES; cycle\+\+\)/,
    'cycle loop must start at _startCycle so a resumed search skips completed cycles')
})

test('ksearch: each cycle end writes checkpoint.json with the 5 state vars', () => {
  assert.match(SOURCE, /label: `checkpoint-\$\{cycle\}`/,
    'a checkpoint-${cycle} mechanical agent call must write at each cycle end')
  assert.match(SOURCE, /checkpointPath: CHECKPOINT_PATH/,
    'checkpoint write must target the canonical checkpoint path')
  // The 5 state vars must be in the checkpoint payload.
  assert.match(SOURCE, /cycle: cycleCount,[\s\S]*?decisionTree,[\s\S]*?bestMetric,[\s\S]*?bestSolution,[\s\S]*?solutionDb/,
    'checkpoint payload must carry cycle + decisionTree + bestMetric + bestSolution + solutionDb')
  assert.match(SOURCE, /await __workflowRuntimeSafePoint\(\{/,
    'checkpoint must be committed through the shared atomic safe point')
})

test('ksearch: checkpoint runtime_metadata is a static marker (NO Date.now())', () => {
  assert.match(SOURCE, /checkpoint_written_at: 'cycle-' \+ cycleCount \+ '-end'/,
    'runtime_metadata.checkpoint_written_at must be a static loop-counter-derived string (cycle-N-end) for postmortem')
  // The whole checkpoint/resume block must not use forbidden runtime APIs.
  // Strip line comments so a comment that *mentions* a forbidden API (e.g.
  // "NO Date.now()") isn't mistaken for actual usage.
  const stripComments = (s) => s.replace(/^\s*\/\/.*$/gm, '')
  const block = stripComments((SOURCE.match(/#43: resume from checkpoint[\s\S]*?for \(let cycle = _startCycle/) || [''])[0])
  assert.ok(block, 'resume block must be locatable')
  assert.doesNotMatch(block, /Date\.now\(\)|Math\.random\(\)|new Date\(\)/,
    'resume block must not use forbidden runtime APIs')
  const writeBlock = stripComments((SOURCE.match(/const checkpointPayload = \{[\s\S]*?await __workflowRuntimeSafePoint\(\{[\s\S]*?\n\s*\}\)/) || [''])[0])
  assert.ok(writeBlock, 'checkpoint-write block must be locatable')
  assert.doesNotMatch(writeBlock, /Date\.now\(\)|Math\.random\(\)|new Date\(\)/,
    'checkpoint-write block must not use forbidden runtime APIs')
})
