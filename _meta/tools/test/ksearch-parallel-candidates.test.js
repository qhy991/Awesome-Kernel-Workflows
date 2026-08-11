'use strict'

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, 'KSearch/ksearch-kernel-optimization.js')
const MANIFEST = path.join(ROOT, 'KSearch/manifest.yaml')
const FIXTURES = path.join(ROOT, '_meta/tools/fixtures')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const source = fs.readFileSync(WORKFLOW, 'utf8')
const manifest = fs.readFileSync(MANIFEST, 'utf8')
const baseArgs = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'ksearch-args.json'), 'utf8'))
const agentReturns = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'ksearch-agent-returns.json'), 'utf8'))

test('ksearch: parallel seed width defaults to four and is capped by the attempt budget', () => {
  assert.match(source, /const _requestedSeedCandidates = Number\(args\.seed_candidates\)/)
  assert.match(source, /const SEED_CANDIDATES = Math\.max\(1, Math\.min\([\s\S]*?ATTEMPTS_PER_CYCLE,[\s\S]*?: 4,[\s\S]*?\)\)/)
})

test('ksearch: independent seed generation uses runtime parallel thunks', () => {
  assert.match(source, /await parallel\(seedIndexes\.map\(\(seedAttempt\) => \(\) => generateSeedCandidate\(seedAttempt\)\)\)/)
})

test('ksearch: fan-out is generated before serial evaluation reduction', async () => {
  const calls = await capturePrompts({
    workflowPath: WORKFLOW,
    args: {
      ...baseArgs,
      iterations: 1,
      attempts_per_cycle: 4,
      seed_candidates: 4,
    },
    agentReturns,
  })
  const labels = calls.map((call) => call.label)
  const seeds = ['gen-0-0', 'gen-0-1', 'gen-0-2', 'gen-0-3']
  for (const label of seeds) assert.ok(labels.includes(label), `missing ${label}`)
  const firstEval = labels.findIndex((label) => label === 'eval-0-0')
  assert.ok(firstEval > labels.indexOf('gen-0-3'), `generation did not fan out before eval: ${labels.join(',')}`)
})

test('ksearch manifest exposes bounded candidate fan-out but keeps evaluation serial', () => {
  assert.match(manifest, /name: seed_candidates[\s\S]*?default: '4'/)
  assert.match(manifest, /label: gen-\{cycle\}-\{attempt\}[\s\S]*?parallelism: parallel_fan_out/)
  assert.match(manifest, /description: Evaluate the generated candidate[\s\S]*?parallelism: single/)
})
