'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const runWorkflow = require('../lib/run-workflow.js')
const source = fs.readFileSync(path.resolve(__dirname, '../../..', 'KernelBand/kernelband-kernel-optimization.js'), 'utf8')

async function run({wrongPath = false, readFailed = false, inline = false} = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kernelband-handoff-'))
  const candidate = path.join(root, 'kernelband_iter_1.cu')
  const code = '// candidate larger than the observed failed JSON payload\n'
    + '/*' + 'x'.repeat(140000) + '*/\nvoid run() {}\n'
  fs.writeFileSync(candidate, code)
  const args = {
    kernel_path: path.join(root, 'seed.cu'), language: 'cuda', exp_dir: root,
    integration_pattern: 'sol_execbench_solution', can_standalone: 'yes', iterations: 1,
    strategies: ['tiling'], sol_cli: '/bin/sol', sol_task_dir: '/tmp/task',
    sol_bench_config: '/tmp/bench.json', sol_seed_dir: '/tmp/seed',
    sol_substrate_dir: '/tmp/substrate', sol_definition_path: '/tmp/task/definition.json',
  }
  const measured = latency => ({compiled: true, correct: true, measurement_valid: true,
    latency_ms: latency, speedup: 1.25, n_pass: 33, n_total: 33, candidate_path: candidate})
  let readCount = 0, measurementCount = 0
  try {
    const result = await runWorkflow(source, args, {
      setup: {}, 'eval-t1': {}, report: {},
      'generate-t1-tiling': inline ? {optimized_kernel: code}
        : {candidate_path: wrongPath ? path.join(root, 'other.cu') : candidate},
    }, {
      'sol-eval-baseline': measured(0.02),
      'read-candidate-t1': request => {
        readCount++
        assert.deepEqual(Array.from(request.argv), ['cat', candidate])
        assert.equal(request.filesystem_policy, 'read-only')
        return readFailed ? {passed: false, stdout: ''} : {passed: true, stdout: fs.readFileSync(candidate, 'utf8')}
      },
      'sol-eval-t1': request => {
        measurementCount++
        assert.equal(request.candidateSource, code)
        assert.equal(request.candidatePath, candidate)
        return measured(0.016)
      },
    })
    return {...result, candidate, readCount, measurementCount}
  } finally {
    fs.rmSync(root, {recursive: true, force: true})
  }
}

test('KernelBand hands a large source file to Host measurement and returns the measured path', async () => {
  const {calls, result, candidate, readCount, measurementCount} = await run()
  assert.equal(readCount, 1)
  assert.equal(measurementCount, 1)
  assert.equal(result.generated_kernel_path, candidate)
  assert.equal(result.best_latency_us, 16)
  assert.equal(result.candidate_pool_size, 2)
  const generator = calls.find(c => c.label === 'generate-t1-tiling')
  assert.deepEqual(generator.schema.required, ['candidate_path'])
  assert.match(generator.prompt, /do not put it in StructuredOutput/)
})
test('KernelBand rejects a different source path before measurement', async () => {
  await assert.rejects(run({wrongPath: true}), /must match the assigned iteration source path/)
})
test('KernelBand rejects an unreadable source before measurement', async () => {
  await assert.rejects(run({readFailed: true}), /could not read the complete candidate/)
})
test('KernelBand retains inline source compatibility at the input boundary', async () => {
  const {readCount, measurementCount} = await run({inline: true})
  assert.equal(readCount, 0)
  assert.equal(measurementCount, 1)
})
