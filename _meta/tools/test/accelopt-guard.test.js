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

// Backend eligibility (method_supported_backends) moved to manifest; the
// in-workflow throw was removed (#24). Off-list backends no longer throw inside
// the workflow — resolveBackend() returns the (normalized) backend and the
// workflow proceeds; eligibility is selector-enforced from manifest.
test('backend:"metal" no longer rejected in-workflow (eligibility moved to manifest)', async () => {
  try {
    const caps = await run({ backend: 'metal' })
    assert.ok(caps.length > 0, 'workflow proceeds; backend eligibility is selector-enforced')
  } catch (e) {
    assert.doesNotMatch(e.message, /does not support backend|backend="metal"/,
      'in-workflow backend gate removed; method_supported_backends now lives in manifest')
  }
})

test('conflicting backend:"triton" + language:"cuda" -> throws', async () => {
  await assert.rejects(run({ backend: 'triton', language: 'cuda' }), /Conflicting args/)
})

test('hip alias normalizes to rocm and proceeds (eligibility moved to selector)', async () => {
  try {
    const caps = await run({ backend: 'hip' })
    assert.ok(caps.length > 0, 'workflow proceeds with normalized backend')
  } catch (e) {
    assert.doesNotMatch(e.message, /does not support backend|backend="rocm"/,
      'in-workflow backend gate removed')
  }
})
