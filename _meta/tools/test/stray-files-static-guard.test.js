'use strict'
// STATIC stray-files guard for AKO4X (#37-adjacent hygiene, NOT the #37 fix).
//
// Scope: this catches only STATIC bare-path write references in workflow prompts
// (e.g. `Write to ITERATIONS.md` with no ${EXP_DIR} prefix). It does NOT catch
// RUNTIME strays — files the agent creates autonomously while running verify/test
// loops (the real #37 cause: CUDAAgent's verify phase writes `.verify_*/` to the
// project root). The runtime #37 fix lives in the CUDAAgent Verify prompt
// (directive to route verify artifacts under ${EXP_DIR}/verify/).
//
// What this guards: every primary file-write instruction (Write/Create/Update/
// Append to <path>) in AKO4X must target ${EXP_DIR}/... A bare filename like
// `ITERATIONS.md` lands in the CWD (project root) — a stray. Static-source
// assertion suffices for this class of bug.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'AKO4X/ako4x-kernel-optimizer.js')
const SOURCE = fs.readFileSync(WORKFLOW, 'utf8')

// Strip line comments so comment-only references (e.g. `// Log to ITERATIONS.md`)
// don't count as write instructions.
const CODE = SOURCE.replace(/^\s*\/\/.*$/gm, '')

test('ako4x: every primary Write/Create/Update target is EXP_DIR-prefixed', () => {
  // Match primary write instructions: `Write to <path>`, `Create <path>`,
  // `Update <path>`, `Append ... to <path>` where <path> ends in a file ext.
  // The path must contain `${EXP_DIR}` (or `exp_dir`); a bare filename is a
  // stray-write bug (lands in CWD = project root).
  const re = /\b(?:Write|Create|Update|Append)\b[^.\n]*?\bto\s+([^`<\n\s]+\.(?:md|json|jsonl|txt))\b/g
  const offenders = []
  let m
  while ((m = re.exec(CODE)) !== null) {
    const target = m[1]
    if (!/EXP_DIR|exp_dir/.test(target)) offenders.push(target)
  }
  assert.deepEqual(offenders, [], `bare (non-EXP_DIR) write targets found: ${offenders.join(', ')}`)
})

test('ako4x: proposals evidence-pointer references the real iteration-log path, not a bare ITERATIONS.md', () => {
  // The proposals.md template's evidence pointer used to say `<ITERATIONS.md line>`,
  // but no bare ITERATIONS.md is written (the log lives at
  // ${EXP_DIR}/round-logs/round-N-iterations.md). A bare reference can prompt the
  // agent to create a stray ITERATIONS.md in CWD to satisfy the pointer.
  assert.match(CODE, /evidence pointer.*\$\{EXP_DIR\}\/round-logs\/round-\$\{round \+ 1\}-iterations\.md/,
    'evidence pointer must reference the real ${EXP_DIR}/round-logs/round-N-iterations.md path')
  assert.doesNotMatch(CODE, /evidence pointer[^<]*<ITERATIONS\.md/,
    'evidence pointer must NOT reference a bare ITERATIONS.md (no such file is written; would cause a stray)')
})
