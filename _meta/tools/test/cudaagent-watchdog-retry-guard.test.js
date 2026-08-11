'use strict'

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const SOURCE = fs.readFileSync(
  path.join(ROOT, 'CUDAAgent', 'cuda-agent-kernel-optimization.js'),
  'utf8',
)

test('cudaagent: watchdog-bounded Implement does not start a background retry', () => {
  assert.match(
    SOURCE,
    /\}\), \{ retries: 0 \}\), `Implement turn \$\{currentAttempt \+ 1\}`\)/,
  )
})

test('cudaagent: watchdog-bounded Verify does not start a background retry', () => {
  assert.match(
    SOURCE,
    /\}\), \{ retries: 0 \}\), `Verify turn \$\{currentAttempt \+ 1\}`\)/,
  )
})

test('cudaagent: Implement forbids unbounded external-repository exploration', () => {
  assert.match(SOURCE, /STRICT BOUNDED GENERATION/)
  assert.match(SOURCE, /Do NOT recursively search \/home/)
  assert.match(SOURCE, /Use at most two short file-inspection shell commands/)
})

test('cudaagent: timed doers use explicit runtime model tiers', () => {
  assert.match(SOURCE, /model: MODEL\.judgment,\n\s+label: `impl-/)
  assert.match(SOURCE, /model: MODEL\.profile,\n\s+label: `verify-/)
})

test('cudaagent: Sol verification receives complete kernel and binding sources', () => {
  assert.match(SOURCE, /FULL kernel_code and FULL binding_code/)
  assert.match(SOURCE, /\$\{IS_SOL \? implResult\.kernel_code : implResult\.kernel_code\.substring\(0, 4000\)\}/)
  assert.match(SOURCE, /\$\{IS_SOL \? implResult\.binding_code : implResult\.binding_code\.substring\(0, 2000\)\}/)
})
