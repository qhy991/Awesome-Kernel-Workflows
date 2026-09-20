'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const stub = require('../lib/schema-stub.js')
const root = path.resolve(__dirname, '../../..')

async function run(file, handlers, extras = {}) {
  const source = fs.readFileSync(path.join(root, file), 'utf8').replace(/^export\s+/, '')
  const args = { problem_path: '/tasks/004', kernel_path: '/exp/seed.cu', exp_dir: '/exp', sol_cli: '/bin/sol',
    sol_task_dir: '/tasks/004', sol_substrate_dir: '/substrate', sol_seed_dir: '/exp',
    sol_bench_config: '/config', integration_pattern: 'sol_execbench_solution',
    iterations: 1, seed_candidates: 2, compile_command: 'compile', test_command: 'test',
    benchmark_command: 'bench', disassemble_command: 'disasm', target_gpu: 'B300', ...extras }
  const sandbox = { args, JSON, Math, Promise, Map, Set, console,
    phase() {}, log() {}, budget() {},
    parallel: tasks => Promise.all(tasks.map(task => task())),
    pipeline: (items, fn) => Promise.all(items.map(fn)),
    evaluate: handlers.evaluate,
    agent: async (prompt, options = {}) => handlers.agent(prompt, options) ?? stub(options.schema),
  }
  return vm.runInNewContext(`(async function(){${source}\n})()`, sandbox)
}
function result(speedup, extra = {}) {
  return { compiled: true, correct: true, n_pass: 2, n_total: 2, speedup, latency_ms: 2 / speedup,
    reference_latency_ms: 2, environment: { hardware: 'NVIDIA B300 SXM6 AC', compute_capability: '10.3', sms: 148 },
    solution_path: '/exp/solution.json', disassembly: { ok: true, instruction_verified: true,
      sass_path: '/exp/bench.sass', top_mnemonics: [['HMMA', 3]], note: '3 measured instructions' }, ...extra }
}

test('KernelSkill rejects model-owned 1.0/999 scores, measures final edit, and leaves absent counters unknown', async () => {
  const labels = [], prompts = []
  const output = await run('KernelSkill/kernelskill-kernel-optimization.js', {
    evaluate: async r => {
      labels.push(r.label)
      if (r.baselineSolutionPath) return result(1)
      if (r.candidateSource === 'seed0') return result(.33)
      if (r.candidateSource === 'seed1') return result(.2)
      assert.equal(r.candidateSource, 'improved-final')
      return result(.4)
    },
    agent: (prompt, o) => {
      prompts.push({label: o.label, prompt})
      if (o.label === 'read-reference') return { reference_code: 'def run(x): return x', op_type: 'gemm', input_shapes: 'M,N,K' }
      if (/^seed-\d+$/.test(o.label)) return { code: o.label.replace('-', ''), strategy: 'test' }
      if (o.label.startsWith('review-')) return { is_compilable: true, is_correct: true, speedup: 999,
        latency_ms: .001, ncu_metrics: { dram_throughput_pct: 0 }, profile_summary: 'untrusted' }
      if (o.label.startsWith('features-')) return { has_reuse: true }
      if (o.label.startsWith('plan-')) return { method_name: 'thread_coarsening', plan: 'source hypothesis', rationale: 'test' }
      if (o.label.startsWith('optimize-')) return { code: 'improved-final' }
      if (o.label.startsWith('gate-') || o.label.startsWith('seed-eval-') || o.label === 'eager-baseline') throw new Error('unmeasured agent path used')
    },
  })
  assert.equal(output.best_speedup, .4)
  assert.equal(output.best_kernel_code, 'improved-final')
  assert.ok(labels.some(l => l.includes('after0')))
  assert.equal(output.optimize_memory[0].speedup_before, .33)
  assert.equal(output.optimize_memory[0].speedup_after, .4)
  assert.equal(output.optimize_memory[0].outcome, 'improved')
  assert.ok(prompts.some(p => p.prompt.includes('Metrics are unknown, not zero')))
})

test('GemmPTX measures baseline and instruction gate before the profile turn; model score cannot win', async () => {
  const order = [], prompts = []
  const output = await run('GemmPTX/gemmptx-gemm-optimization.js', {
    evaluate: async r => { order.push(r.label); assert.equal(r.disassemble, true); return result(r.baselineSolutionPath ? 1 : 1.2) },
    agent: (prompt, o) => {
      prompts.push({label:o.label,prompt});order.push(o.label)
      if (o.label === 'gemm-signature') return { is_gemm: true, op_family: 'gemm', bottleneck_prior: 'compute' }
      if (o.label === 'instruction-plan') return { candidates: [{ candidate_id: 'one', target_instruction: 'mma.sync', sass_regex: 'HMMA', ptx_regex: 'mma', hypothesis: 'test' }] }
      if (o.label.startsWith('implement-')) return { implemented: true, kernel_code: 'complete-binding-source' }
      if (o.label.startsWith('profile-')) return { measured: true, correct: true, speedup_vs_baseline: 999, latency_ms: .0001 }
      if (o.label === 'hardware-census' || o.label === 'baseline-evidence' || o.label.startsWith('disassemble-verify-')) throw new Error('agent-owned measurement path used')
    },
  })
  assert.equal(output.best_speedup, 1.2)
  assert.ok(order.indexOf('sol-eval-baseline') < order.indexOf('gemm-signature'))
  assert.ok(order.indexOf('sol-eval-one') < order.indexOf('profile-one'))
  assert.ok(prompts.some(p => p.prompt.includes('sm_103')))
})

test('GemmPTX fails closed when the Host cannot establish disassembly', async () => {
  const output = await run('GemmPTX/gemmptx-gemm-optimization.js', {
    evaluate: async () => result(1, {disassembly:null}),
    agent: () => { throw new Error('no agent should run before baseline evidence') },
  })
  assert.equal(output.error, 'missing_evidence_contract')
})

test('FACT measures first composition before later slots and retains its receipt on a throwing later response', async () => {
  const order = []
  const output = await run('FACT/fact-kernel-optimization.js', {
    evaluate: async r => { order.push(r.label); return result(1.3) },
    agent: (prompt, o) => {
      order.push(o.label)
      if (o.label === 'Setup FACT') return { target_architecture: 'sm_103', kernel_spec: { operation:'gemm', dtypes:['fp16'] }, exemplar_kernels:[], composition_budget:3 }
      if (o.label === 'Discover patterns') return { patterns_discovered:[{pattern_id:'tile',pattern_name:'tiling',description:'change tile'}] }
      if (o.label.startsWith('Realize pattern ')) return { patterns_realized:[{pattern_id:'tile',pattern_name:'tiling',code_template:'template'}], dependency_graph:'tile',realization_summary:'one' }
      if (o.label === 'Compose patterns 1') return {composed_kernels:[{kernel_id:'k1',applied_patterns:['tile'],kernel_code:'complete-source'}],candidates_generated:1,composition_summary:'one'}
      if (o.label.startsWith('Compose patterns ')) throw new Error('structured output missing')
    },
  })
  assert.ok(order.indexOf('sol-eval-fact_k1') < order.indexOf('Compose patterns 2'))
  assert.equal(output.kernels_composed, 1)
  assert.equal(order.filter(x => x === 'sol-eval-fact_k1').length, 1)
})

test('STARK Debug context carries complete source and entry contract', () => {
  const source = fs.readFileSync(path.join(root,'STARK/stark-kernel-optimization.js'),'utf8')
  const begin=source.indexOf('function buildDebugContext('), end=source.indexOf('\n}',begin)+2
  const code='prefix\n'+'x'.repeat(5000)+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }'
  const context=vm.runInNewContext(`${source.slice(begin,end)}; buildDebugContext('root','cuda')`,{
    referenceKernelCode:code, getNode:()=>({kernel_code:code,logs:'binding missing',parent_id:null,children:[]}),getSiblings:()=>[],fenceToken:()=> 'cuda',
  })
  assert.ok(context.includes('PYBIND11_MODULE'))
  assert.match(source.slice(source.indexOf('You are a kernel debugging expert.'), source.indexOf("label: `debug-")), /SOL_CANDIDATE_CONTRACT/)
})
