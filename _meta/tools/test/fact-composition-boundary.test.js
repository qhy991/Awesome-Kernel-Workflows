'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const stub = require('../lib/schema-stub.js')

const source = fs.readFileSync(path.resolve(__dirname, '../../../FACT/fact-kernel-optimization.js'), 'utf8')

async function runComposition({ count = 3, failSlot = null } = {}) {
  const calls = []
  const sandbox = {
    args: { kernel_path: '/seed/kernel.cu', exp_dir: '/experiment', integration_pattern: 'standalone' },
    JSON, Math, Promise, console,
    phase() {}, log() {}, budget() {},
    parallel: async tasks => Promise.all(tasks.map(task => task())),
    pipeline: async (items, fn) => Promise.all(items.map(fn)),
    agent: async (prompt, options) => {
      const label = options.label
      if (label === 'Setup FACT') return {
        cutlass_version: '3.x', target_architecture: 'sm_103',
        kernel_spec: { operation: 'gemm', shapes: 'M,N,K', dtypes: ['fp16'] },
        exemplar_kernels: ['/seed/kernel.cu'], composition_budget: count,
      }
      if (label === 'Discover patterns') return { patterns_discovered: [{ pattern_id: 'tile', pattern_name: 'tile', description: 'change tiling' }] }
      if (label === 'Realize patterns') return { patterns_realized: [{ pattern_id: 'tile', code_template: 'PATTERN_IMPLEMENTATION' }], dependency_graph: { tile: [] } }
      if (label.startsWith('Compose patterns ')) {
        calls.push({ prompt, schema: options.schema, label })
        const index = Number(label.split(' ').at(-1))
        if (index === failSlot) return null
        return {
          candidates_generated: 1,
          composed_kernels: [{ kernel_id: `k${index}`, applied_patterns: ['tile'], kernel_code: `complete-source-${index}` }],
          composition_summary: 'one complete candidate',
        }
      }
      return stub(options.schema)
    },
  }
  const result = await vm.runInNewContext(`(async function(){\n${source.replace(/^export\s+/, '')}\n})()`, sandbox)
  return { result, calls }
}

test('FACT keeps the composition count with one full-source response per candidate', async () => {
  const { result, calls } = await runComposition()
  assert.equal(result.kernels_composed, 3)
  assert.equal(calls.length, 3)
  for (const [index, call] of calls.entries()) {
    assert.match(call.prompt, /Generate exactly ONE complete kernel candidate/)
    assert.match(call.prompt, /forward/)
    assert.match(call.prompt, /PATTERN_IMPLEMENTATION/)
    assert.match(call.prompt, /Dependency graph: \{"tile":\[\]\}/)
    assert.equal(call.schema.properties.composed_kernels.maxItems, 1)
    assert.equal(call.schema.properties.composed_kernels.minItems, 1)
    assert.equal(call.schema.properties.candidates_generated.const, 1)
    assert.equal(call.schema.properties.composed_kernels.items.properties.kernel_id.const, `k${index + 1}`)
  }
  assert.match(calls[2].prompt, /"kernel_id":"k1"/)
  assert.match(calls[2].prompt, /"kernel_id":"k2"/)
  assert.doesNotMatch(calls[2].prompt, /complete-source-/)
})

test('a failed composition preserves earlier candidates and later slots still execute', async () => {
  const { result, calls } = await runComposition({ failSlot: 2 })
  assert.equal(result.kernels_composed, 2)
  assert.ok(calls.some(call => call.label === 'Compose patterns 3'))
  const last = calls.at(-1).prompt
  assert.match(last, /"kernel_id":"k1"/)
  assert.doesNotMatch(last, /"kernel_id":"k2"/)
})
