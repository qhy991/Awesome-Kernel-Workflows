'use strict'

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'KernelFoundry/kernelfoundry-kernel-optimization.js')
const source = () => fs.readFileSync(WORKFLOW, 'utf8')

test('KernelFoundry preserves full Sol parent source during variation', () => {
  const text = source()
  assert.match(text, /IS_SOL \? selectedParent\.code : selectedParent\.code\.substring\(0, 4000\)/)
  assert.match(text, /authoritative producer call_id prefix is Vary\/vary-/)
  assert.match(text, /never reuse another generation/)
  assert.match(text, /Verify the written bytes equal that selected JSON string/)
})

test('KernelFoundry blocks on Sol RUN before PARSE', () => {
  const text = source()
  assert.match(text, /three strictly sequential blocking operations/)
  assert.match(text, /Do not start PARSE until RUN has reached a terminal exit/)
  assert.match(text, /Never launch RUN and PARSE concurrently/)
})

test('KernelFoundry Sol evaluation is watchdog-bounded without retries', () => {
  const text = source()
  const block = text.slice(text.indexOf('const evalResult = IS_SOL'), text.indexOf(': await agentRetry', text.indexOf('const evalResult = IS_SOL')))
  assert.match(block, /\}\), \{ retries: 0 \}\)/)
})

test('KernelFoundry retries only transient zero-reference measurements once', () => {
  const text = source()
  assert.match(text, /\.retry\.bench\.jsonl/)
  assert.match(text, /RETRY_RUN_ONCE/)
  assert.match(text, /PASSED.*non-positive[\s\S]*reference_latency_ms/)
  assert.match(text, /exactly one measurement-/)
  assert.match(text, /Do not repack, edit, or regenerate the candidate/)
  assert.match(text, /never retry for compile,[\s\S]*correctness,[\s\S]*timeout,[\s\S]*missing-row/)
})
