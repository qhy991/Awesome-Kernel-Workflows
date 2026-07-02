'use strict'

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')

function read(rel) {
  return fs.readFileSync(path.join(ROOT, rel), 'utf8')
}

test('AutoMegaKernel workflow is a strict adapter to the AMK harness', () => {
  const workflow = read('AutoMegaKernel/automegakernel-megakernel-optimization.js')
  const manifest = read('AutoMegaKernel/manifest.yaml')

  assert.match(workflow, /name:\s*'automegakernel-megakernel-optimization'/)
  assert.match(workflow, /\bconst\s+AMK_ROOT\s*=/)
  assert.match(workflow, /\bconst\s+MODEL_ID\s*=/)
  assert.match(workflow, /\bconst\s+TARGET_GPU\s*=/)
  assert.match(workflow, /ScheduleConfig/)
  assert.match(workflow, /kernel_knobs/)
  assert.match(workflow, /amk propose/)
  assert.match(workflow, /amk eval/)
  assert.match(workflow, /amk loop/)
  assert.match(workflow, /validate-before-launch/)
  assert.match(workflow, /not a standalone reimplementation/)

  assert.match(manifest, /portability:\s*method_intrinsic/)
  assert.match(manifest, /intrinsic_to:\s*AutoMegaKernel/)
  assert.match(manifest, /requires_harness:\s*true/)
  assert.match(manifest, /- amk-schedule-search/)
  assert.match(manifest, /- megakernel-synthesis/)
})

test('AutoMegaKernel documentation states the dependency and fidelity boundary', () => {
  const readme = read('AutoMegaKernel/README.md')
  const zh = read('AutoMegaKernel/README.zh-CN.md')

  assert.match(readme, /Requires an existing AutoMegaKernel checkout/)
  assert.match(readme, /strict adapter/)
  assert.match(readme, /not a general CUDA kernel optimizer/)
  assert.match(zh, /依赖已有的 AutoMegaKernel 仓库/)
  assert.match(zh, /严格 adapter/)
  assert.match(zh, /不是通用 CUDA kernel optimizer/)
})
