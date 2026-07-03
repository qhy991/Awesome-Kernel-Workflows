'use strict'
// Verify the #37 fix: CUDAAgent's Verify phase directs the agent to route ALL
// verification artifacts it authors (test harnesses, kernel copies, `.verify_*`
// dirs, `verify_*`/`test_*` sentinels) under ${EXP_DIR}/verify/attempt-N/ — not
// the project root / CWD.
//
// The #37 root cause (per experiments/kersor-20260629-102046 issue-draft fix #9):
// CUDAAgent's verify phase ran the verify/test loop in the project root CWD, so
// the agent created `.verify_*/` dirs + stray files (verify_task, verify_candidate3,
// test_harness.py, kernel.py, test_kernel.py) there — 8-20 strays per round. This
// is a RUNTIME stray (agent-autonomous file creation), not a static prompt bare
// reference; the fix is a prompt directive pinning the verify workspace to
// ${EXP_DIR}/verify/. Static-source assertion confirms the directive is present.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'CUDAAgent/cuda-agent-kernel-optimization.js')
const SOURCE = fs.readFileSync(WORKFLOW, 'utf8')

test('cudaagent: Verify phase pins the verify workspace to ${EXP_DIR}/verify/attempt-N/', () => {
  // The directive must name the EXP_DIR-prefixed verify workspace and tell the
  // agent to mkdir it, so authored artifacts don't land in the project root.
  assert.match(SOURCE, /Step 0: Isolated verify workspace/,
    'Verify prompt must have a Step 0 verify-workspace directive')
  assert.match(SOURCE, /\$\{EXP_DIR\}\/verify\/attempt-\$\{currentAttempt\}\//,
    'Verify prompt must name the ${EXP_DIR}/verify/attempt-${currentAttempt}/ workspace')
  assert.match(SOURCE, /mkdir -p/,
    'Verify prompt must instruct mkdir -p for the verify workspace')
})

test('cudaagent: Verify phase forbids verify artifacts in project root / CWD', () => {
  assert.match(SOURCE, /Do NOT write any verify artifact to the project root or the current\s+working directory/,
    'Verify prompt must explicitly forbid verify artifacts in project root / CWD')
  assert.match(SOURCE, /leak into the user's tree as strays/,
    'Verify prompt must explain the stray-files rationale')
})

test('cudaagent: Step 0 verify-workspace directive precedes Step 1 Compile', () => {
  const i0 = SOURCE.indexOf('Step 0: Isolated verify workspace')
  const i1 = SOURCE.indexOf('Step 1: Compile')
  assert.ok(i0 > -1 && i1 > -1 && i0 < i1, 'Step 0 (verify workspace) must come before Step 1 (Compile) in the Verify prompt')
})
