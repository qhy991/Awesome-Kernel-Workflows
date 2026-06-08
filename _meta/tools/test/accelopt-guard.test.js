'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const WORKFLOW = path.resolve(__dirname, '..', '..', '..', 'AccelOpt/accelopt-kernel-optimization.js')

const baseArgs = {
  kernel_path: '/tmp/fixture/kernel.cu',
  problem_path: '/tmp/fixture/problem.json',
  iterations: 1, breadth: 1, samples_per_plan: 1,
  substrate_command_prefix: 'python3',
}

async function run(extra) {
  return capturePrompts({ workflowPath: WORKFLOW, args: { ...baseArgs, ...extra } })
}

test('no backend/language -> defaults to cuda (CUDA appears in some prompt)', async () => {
  const caps = await run({})
  assert.ok(caps.length > 0, 'expected at least one rendered prompt')
  assert.ok(caps.some(c => /CUDA/.test(c.prompt)), 'default cuda path should render CUDA vocabulary')
})

test('language:"cuda" alias resolves like backend:"cuda"', async () => {
  const caps = await run({ language: 'cuda' })
  assert.ok(caps.some(c => /CUDA/.test(c.prompt)))
})

test('backend:"triton" is method-supported (does NOT throw at the guard)', async () => {
  await assert.doesNotReject(run({ backend: 'triton' }), /does not support backend/)
})

test('backend:"metal" -> throws naming the supported set', async () => {
  await assert.rejects(run({ backend: 'metal' }), (err) => {
    assert.match(err.message, /backend="metal"/)
    assert.match(err.message, /cuda/)
    assert.match(err.message, /triton/)
    return true
  })
})

test('conflicting backend:"triton" + language:"cuda" -> throws', async () => {
  await assert.rejects(run({ backend: 'triton', language: 'cuda' }), /Conflicting args/)
})

test('hip alias normalizes (backend:"hip" -> rocm, rejected, names rocm not hip)', async () => {
  await assert.rejects(run({ backend: 'hip' }),
    (err) => { assert.match(err.message, /backend="rocm"/); return true })
})
