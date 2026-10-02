'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const stub = require('../lib/schema-stub.js')

const root = path.resolve(__dirname, '../../..')
const workflows = fs.readdirSync(root, {withFileTypes: true})
  .filter(dir => dir.isDirectory() && !dir.name.startsWith('_'))
  .flatMap(dir => fs.readdirSync(path.join(root, dir.name))
    .filter(file => file.endsWith('.js'))
    .map(file => path.join(dir.name, file)))
  .filter(file => fs.readFileSync(path.join(root, file), 'utf8').includes('let INTEGRATION_DECISION'))

// Execute the real entrypoint up to the integration decision. Stopping at that
// boundary isolates routing from unrelated later optimization/budget behavior.
// Separate Host feedback tests cover the actual candidate evaluation branches.
async function decide(file, method) {
  const source = fs.readFileSync(path.join(root, file), 'utf8').replace(/^export\s+/, '')
  const calls = []
  let selected = null
  const stopped = Object.assign(new Error('decision observed'), {retryable: false})
  const args = {
    kernel_path: '/task/kernel.cu', problem_path: '/task/problem.py',
    reference_path: '/task/problem.py', reference_file: '/task/problem.py',
    reference_code_path: '/task/problem.py', model_path: '/task/problem.py',
    task_path: '/task/problem.py', task_spec_path: '/task/spec.json',
    problem_definition: 'Optimize a GEMM kernel', op_description: 'GEMM',
    exp_dir: '/experiment', project_root: '/task', register_script: '/task/register.py',
    build_command: 'build', compile_command: 'compile', test_command: 'test',
    benchmark_command: 'benchmark', eval_command: 'benchmark',
    profile_command: 'profile', binary_path: '/task/program', target_gpu: 'B300',
    sol_cli: '/tools/sol', sol_task_dir: '/task', sol_bench_config: '/task/config.json',
    sol_seed_dir: '/experiment', sol_substrate_dir: '/tools/integration',
    substrate_command_prefix: 'python3', iterations: 1, seed_candidates: 1,
    ...(method ? {integration_pattern: method} : {}),
  }
  const budget = Object.assign(() => {}, {remaining: () => 1e9, total: 1e9})
  const context = {
    args, console, Map, Set, JSON, Math, Promise, budget,
    phase() {},
    log(message) {
      const match = String(message).match(/^integration method = (\S+)/i)
      if (match) { selected = match[1]; throw stopped }
    },
    parallel: tasks => Promise.all(tasks.map(task => task())),
    pipeline: (items, fn) => Promise.all(items.map(fn)),
    async agent(prompt, options = {}) {
      calls.push(options.label)
      if (options.label === 'integration-classify') return {can_standalone: 'yes'}
      if (options.label === 'integration-strategist') return {method: 'standalone', build_fidelity: 'isolated'}
      if (file.startsWith('WarpSpeed/')) {
        if (options.label === 'init:materialize') return {
          grounded: true, harness_ready: true,
          preflight: {git_ok: true, wsdb_ok: true, reviewer_ok: true},
          config: {
            commands: {wsdb: 'wsdb', correctness: 'test', screen: 'screen', confirm: 'confirm', calibrate: 'calibrate', ncu_profile: 'profile'},
            paths: {project_dir: '/task', state_dir: '/experiment'},
          },
        }
        if (options.label === 'calibrate') return {grounded: true, skipped: true}
      }
      const response = stub(options.schema)
      if (file.startsWith('CutlassGEMM/') && response && typeof response === 'object') {
        if ('compilation_success' in response) response.compilation_success = true
      }
      return response
    },
  }
  try {
    await new vm.Script(`(async function(){${source}\n})()`, {filename: file}).runInNewContext(context)
  } catch (error) {
    if (error !== stopped) throw error
  }
  assert.ok(selected, `${file}: did not reach its integration decision`)
  return {selected, calls}
}

for (const file of workflows) {
  test(`${file}: explicit integration is used without another classification activation`, async () => {
    for (const method of ['standalone', 'embedded_inplace', 'embedded_dispatch', 'sol_execbench_solution']) {
      const {selected, calls} = await decide(file, method)
      assert.equal(selected, method)
      assert.deepEqual(calls.filter(label => /^integration-(classify|strategist)$/.test(label)), [])
    }
  })
}

test('undecided KernelSkill integration still uses discovery', async () => {
  const {selected, calls} = await decide('KernelSkill/kernelskill-kernel-optimization.js')
  assert.equal(selected, 'standalone')
  assert.equal(calls.filter(label => label === 'integration-strategist').length, 1)
})
