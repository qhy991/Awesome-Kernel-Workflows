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
    assert.match(source, /sol_execbench_cli: !!SOL_CLI/)
    assert.match(source, /--preferred-method sol_execbench_solution/)
    assert.match(source, /envPrefix: SOL_ENV_PREFIX/)
    assert.match(source, /definitionPath: SOL_DEFINITION_PATH/)
    assert.match(source, /INTEGRATION_PATTERN === 'sol_execbench_solution' \? 'sol_execbench_solution' : 'standalone'|INTEGRATION_PATTERN === 'sol_execbench_solution'\s*\?\s*'sol_execbench_solution'/)
    assert.match(source, /sol_execbench_solution requires non-empty/)
  }
})

test('new sol workflows declare the runtime evaluation arguments', () => {
  for (const manifest of ['CUDAAgent/manifest.yaml', 'KDA/manifest.yaml', 'KernelBlaster/manifest.yaml']) {
    const source = read(manifest)
    for (const arg of ['sol_cli', 'sol_task_dir', 'sol_bench_config', 'sol_env_prefix',
      'sol_definition_path', 'sol_substrate_dir']) {
      assert.match(source, new RegExp(`- ${arg}\\b`), `${manifest} is missing ${arg}`)
    }
    assert.match(source, /sol_execbench_solution/)
  }
})
