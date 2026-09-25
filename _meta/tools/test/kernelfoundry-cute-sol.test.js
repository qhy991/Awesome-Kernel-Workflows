'use strict'

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')

const root = path.resolve(__dirname, '..', '..', '..')
const source = fs.readFileSync(path.join(root, 'KernelFoundry/kernelfoundry-kernel-optimization.js'), 'utf8')
const manifest = fs.readFileSync(path.join(root, 'KernelFoundry/manifest.yaml'), 'utf8')

function extract(name) {
  const match = source.match(new RegExp(`function ${name}\\([^)]*\\) \\{[\\s\\S]*?\\n\\}`))
  assert.ok(match, `${name} is present`)
  return match[0]
}

test('KernelFoundry CuTe SOL emits Python paths and keeps CUDA paths intact', () => {
  const functions = `${extract('kernelPathForGeneration')}\n${extract('bestKernelPath')}`
  const base = { EXP_DIR: '/tmp/exp', USE_DRIVER: false, DRIVER_SOURCE_EXT: '' }
  const cute = vm.runInNewContext(`${functions}\n[kernelPathForGeneration(1),bestKernelPath()]`,
    { ...base, TARGET_LANG: 'cute-dsl', CUTE_SOL: true })
  const cuda = vm.runInNewContext(`${functions}\n[kernelPathForGeneration(1),bestKernelPath()]`,
    { ...base, TARGET_LANG: 'cuda', CUTE_SOL: false })
  assert.deepEqual(Array.from(cute), ['/tmp/exp/gen_1.py', '/tmp/exp/best_kernel.py'])
  assert.deepEqual(Array.from(cuda), ['/tmp/exp/gen_1.cu', '/tmp/exp/best_kernel.cu'])
})

test('KernelFoundry CuTe SOL sends explicit language to the Host and retains a run contract', () => {
  assert.match(source, /candidateLanguage: CUTE_SOL \? 'cute-dsl' : ''/)
  assert.match(source, /CuTe DSL contract: emit one complete Python module/)
  assert.match(source, /module-level run\(\.\.\.\) matching the reference signature/)
  assert.match(manifest, /languages:\s*\n\s*- cuda\s*\n\s*- triton\s*\n\s*- cute-dsl/)
})
