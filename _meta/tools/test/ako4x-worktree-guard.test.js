'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')

test('AKO4X initializes the round evaluation counter before hypothesis dispatch', () => {
  const source = fs.readFileSync(path.join(ROOT, 'AKO4X', 'ako4x-kernel-optimizer.js'), 'utf8')
  assert.match(source, /let evaluatedThisRound = 0/)
  assert.match(source, /const evaluatedThisHypothesis = impls\.filter\(Boolean\)\.length/)
  assert.match(source, /evaluatedThisRound \+= evaluatedThisHypothesis/)
  assert.doesNotMatch(source, /const evaluatedThisRound = impls\.filter\(Boolean\)\.length/)
})

test('AKO4X switches away from worktree isolation when runtime git capability is absent', () => {
  const source = fs.readFileSync(path.join(ROOT, 'AKO4X', 'ako4x-kernel-optimizer.js'), 'utf8')
  assert.match(source, /const WORKTREE_ISOLATION = args\.in_git_repo === true \|\| args\.in_git_repo === 'true'/)
  assert.match(source, /isolation: WORKTREE_ISOLATION \? 'worktree' : 'fresh-process'/)
})
