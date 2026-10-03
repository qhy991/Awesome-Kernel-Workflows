'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const os = require('node:os')
const vm = require('node:vm')
const {execFileSync} = require('node:child_process')
const source = fs.readFileSync(path.resolve(__dirname, '../../../KernelBand/kernelband-kernel-optimization.js'), 'utf8')

async function run({regression = false, missingBaseline = false, invalidCandidate = false, wrongPath = false, missingTurn = false, taskOnly = false, nodeHost = false} = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kernelband-native-task-'))
  const seed = path.join(root, 'seed.py'), candidate = path.join(root, 'kernelband_iter_1.py')
  const verifier = path.join(root, 'verify.py')
  fs.writeFileSync(seed, 'def run():\n    return sum(range(20000))\n')
  fs.writeFileSync(path.join(root, 'task.md'), 'Run verify.py SOURCE RESULT. It validates the output and measures CPU latency. This is a CPU execution canary, not a GPU performance experiment.\n')
  fs.writeFileSync(verifier, `import runpy,sys,time,json\nfrom pathlib import Path\nf=runpy.run_path(sys.argv[1])['run']\nassert f() == 199990000\nt=time.perf_counter_ns()\nfor _ in range(20): f()\nr={'compiled':True,'correct':True,'latency_ms':(time.perf_counter_ns()-t)/20/1e6,'n_pass':1,'n_total':1}\nPath(sys.argv[2]).write_text(json.dumps(r))\nprint(json.dumps(r))\n`)
  const calls = [], phases = [], tests = []
  const features = {normalized_time: 1, registers_per_thread: 32, shared_mem_bytes: 0, block_dimension: 256, occupancy: 0.5}
  function measure(file, suffix) {
    const out = path.join(root, suffix + '.json')
    const raw = execFileSync('python3', [verifier, file, out], {encoding: 'utf8'}).trim()
    tests.push({file, out, raw})
    return {test_result_path: out, test_result_json: raw, behavioral_features: features}
  }
  const sandbox = {
    ...(nodeHost ? {evaluate: () => {throw new Error("task tests must not require Host evaluate")}} : {}),
    args: {kernel_path: seed, language: 'cuda', iterations: 1, num_clusters: 1,
      integration_pattern: 'standalone', strategies: ['tiling'], exp_dir: root,
      ...(taskOnly ? {problem_path: path.join(root, 'task.md')} : {benchmark_command: `python3 ${verifier} {kernel_path} {result_path}`})},
    phase: x => phases.push(x), log() {}, budget() {}, JSON, Math, Promise,
    agent: async (prompt, options) => {
      calls.push({prompt, ...options})
      if (options.label === 'setup') {
        if (missingBaseline) return {}
        fs.copyFileSync(seed, path.join(root, 'kernelband_seed.py'))
        return measure(seed, 'baseline')
      }
      if (options.label === 'generate-t1-tiling') {
        fs.writeFileSync(candidate, '# ' + 'x'.repeat(140000) + '\ndef run():\n    return 20000 * 19999 // 2\n')
        assert.match(prompt, /Read the complete source from .*kernelband_seed\.py/)
        // A task with a fixed test source installs the candidate at the original path.
        // The pool's seed must still refer to the unchanged snapshot.
        fs.copyFileSync(candidate, seed)
        assert.notEqual(fs.readFileSync(seed, 'utf8'), fs.readFileSync(path.join(root, 'kernelband_seed.py'), 'utf8'))
        return {candidate_path: wrongPath ? seed : candidate}
      }
      if (options.label === 'eval-t1') {
        if (missingTurn) return null
        const result = measure(candidate, 'candidate')
        if (regression) {
          const raw = JSON.parse(result.test_result_json)
          raw.latency_ms = JSON.parse(tests[0].raw).latency_ms * 2
          result.test_result_json = JSON.stringify(raw)
        }
        if (invalidCandidate) result.test_result_json = '{}'
        // Model-reported values must not override the actual test output.
        return {...result, compiled: true, correct: true, latency_us: 1e9, speedup: 1e9}
      }
      if (options.label === 'report') return 'CPU canary complete'
      throw new Error('unexpected agent: ' + options.label)
    },
  }
  try {
    const result = await vm.runInNewContext('(async function(){\n' + source.replace(/^export\s+/, '') + '\n})()', sandbox)
    return {result, calls, phases, tests, candidate, sourceBytes: fs.existsSync(candidate) ? fs.statSync(candidate).size : 0}
  } finally { fs.rmSync(root, {recursive: true, force: true}) }
}

test('native KernelBand without evaluate executes task tests and hands off a complete large file', async () => {
  const {result, calls, phases, tests, candidate, sourceBytes} = await run()
  assert.equal(tests.length, 2)
  assert.ok(sourceBytes > 140000)
  assert.equal(tests[1].file, candidate)
  assert.equal(result.generated_kernel_path, candidate)
  const measured = JSON.parse(tests[1].raw).latency_ms * 1000
  assert.equal(result.best_latency_us, measured)
  assert.equal(result.candidate_pool_size, 2)
  assert.ok(result.cumulative_reward > 0)
  assert.ok(phases.includes('Select') && phases.includes('Update'))
  const generator = calls.find(x => x.label === 'generate-t1-tiling')
  assert.deepEqual(Array.from(generator.schema.required), ['candidate_path'])
  assert.match(generator.prompt, /Read the complete source from/)
  const evaluator = calls.find(x => x.label === 'eval-t1')
  assert.match(evaluator.prompt, /task.md/)
  assert.match(evaluator.prompt, /Read the complete candidate from/)
  assert.equal(result.iteration_log[0].test_result_path, tests[1].out)
})
test('native KernelBand stops on missing baseline evidence instead of inventing a baseline', async () => {
  const {result, tests} = await run({missingBaseline: true})
  assert.equal(result.reason, 'baseline_unverified')
  assert.equal(tests.length, 0)
})
test('native KernelBand does not update the bandit from invalid or missing test evidence', async () => {
  for (const settings of [{invalidCandidate: true}, {missingTurn: true}]) {
    const {result} = await run(settings)
    assert.equal(result.cumulative_reward, 0)
    assert.equal(result.candidate_pool_size, 1)
    assert.equal(result.iteration_log[0].failure_code, 'measurement_invalid')
  }
})
test('native KernelBand rejects the wrong candidate file before testing', async () => {
  await assert.rejects(run({wrongPath: true}), /must match the assigned iteration source path/)
})

test('task.md alone selects task tests in both native and Node runtimes', async () => {
  for (const nodeHost of [false, true]) {
    const {result, tests} = await run({taskOnly: true, nodeHost})
    assert.equal(tests.length, 2)
    assert.equal(result.candidate_pool_size, 2)
    assert.equal(result.best_latency_us, JSON.parse(tests[1].raw).latency_ms * 1000)
  }
})

// The altered timing is a deterministic negative control, not a performance claim.
test('task testing retains its measured initial implementation when a candidate regresses', async () => {
  const {result, tests} = await run({regression: true})
  assert.equal(result.best_latency_us, JSON.parse(tests[0].raw).latency_ms * 1000)
  assert.match(result.generated_kernel_path, /kernelband_seed\.py$/)
})
