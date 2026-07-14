'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const PATCHER = fs.readFileSync(path.join(ROOT, 'scripts', 'patch-sol-execbench-eval.js'), 'utf8')

function read(rel) {
  return fs.readFileSync(path.join(ROOT, rel), 'utf8')
}

test('sol-execbench opt-in pool includes KDA and KernelBlaster', () => {
  assert.match(PATCHER, /\['KDA', 'kda-kernel-workflow\.js'\]/)
  assert.match(PATCHER, /\['KernelBlaster', 'kernelblaster-kernel-optimization\.js'\]/)

  for (const [dir, file] of [
    ['CUDAAgent', 'cuda-agent-kernel-optimization.js'],
    ['ARGUS', 'argus-kernel-optimization.js'],
    ['Generalist', 'generalist-kernel-optimization.js'],
    ['KDA', 'kda-kernel-workflow.js'],
    ['KernelBlaster', 'kernelblaster-kernel-optimization.js'],
  ]) {
    const source = read(path.join(dir, file))
    assert.match(source, /const SOL_SOLUTION_CONTRACT/)
    assert.match(source, /function __solExecbenchEvalPlan/)
    assert.match(source, /sol_execbench_solution/)
  }
})

test('new sol workflows declare the runtime evaluation arguments', () => {
  for (const manifest of ['KDA/manifest.yaml', 'KernelBlaster/manifest.yaml']) {
    const source = read(manifest)
    for (const arg of ['sol_cli', 'sol_task_dir', 'sol_bench_config', 'sol_substrate_dir']) {
      assert.match(source, new RegExp(`- ${arg}\\b`), `${manifest} is missing ${arg}`)
    }
    assert.match(source, /sol_execbench_solution/)
  }
})
