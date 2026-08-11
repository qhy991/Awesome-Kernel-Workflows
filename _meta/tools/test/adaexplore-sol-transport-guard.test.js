'use strict'

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'AdaExplore/adaexplore-kernel-optimization.js')

function source() {
  return fs.readFileSync(WORKFLOW, 'utf8')
}

test('AdaExplore preserves the full selected source for Sol tuner edits', () => {
  const text = source()
  assert.match(text, /IS_SOL \? selectedNode\.code : selectedNode\.code\.substring\(0, 5000\)/)
  assert.match(text, /IS_SOL \? selectedNode\.code : selectedNode\.code\.substring\(0, 6000\)/)
  assert.match(text, /never omit an unchanged suffix from the selected kernel/)
})

test('AdaExplore requires blocking RUN completion before Sol PARSE', () => {
  const text = source()
  assert.match(text, /three strictly sequential blocking operations/)
  assert.match(text, /Do not start PARSE until RUN has reached a terminal exit/)
  assert.match(text, /Never launch RUN and PARSE concurrently/)
})

test('AdaExplore binds Sol materialization to the exact MCTS producer', () => {
  const text = source()
  assert.match(text, /const producerCallPrefix = `Expand\//)
  assert.match(text, /use\s+only result\.output\.kernel_code from that exact call_id prefix/)
  assert.match(text, /Never reuse a\s+proposal or candidate from another MCTS step/)
  assert.match(text, /verify the file\s+bytes equal the selected journal string before PACK/)
})

test('AdaExplore Sol evaluation does not retry beyond its watchdog', () => {
  const text = source()
  const solBlock = text.slice(text.indexOf('const evalResult = IS_SOL'), text.indexOf(': await agentRetry', text.indexOf('const evalResult = IS_SOL')))
  assert.match(solBlock, /\}\), \{ retries: 0 \}\)/)
})

test('AdaExplore checkpoints copy the immutable verified candidate bytes', () => {
  const text = source()
  assert.match(text, /candidate_sha256: \{ type: 'string' \}/)
  assert.match(text, /required: \['compiled', 'correct', 'speedup', 'candidate_sha256'\]/)
  assert.match(text, /candidateSha256: IS_SOL \? evalResult\.candidate_sha256 : null/)
  assert.match(text, /kernelPath: authoritativeKernelPath/)
  assert.match(text, /bestKernelSourcePath: IS_SOL \? globalBest\.kernelPath : null/)
  assert.match(text, /bestKernelExpectedSha256: IS_SOL \? globalBest\.candidateSha256 : null/)
  assert.match(text, /bestKernelCode: IS_SOL \? null : globalBest\.code/)
})
