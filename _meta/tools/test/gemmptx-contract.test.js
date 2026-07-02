'use strict'

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')

function read(rel) {
  return fs.readFileSync(path.join(ROOT, rel), 'utf8')
}

test('GemmPTX workflow is PTX/SASS-evidence driven for GEMM', () => {
  const workflow = read('GemmPTX/gemmptx-gemm-optimization.js')
  const manifest = read('GemmPTX/manifest.yaml')

  assert.match(workflow, /name:\s*'gemmptx-gemm-optimization'/)
  assert.match(workflow, /Hardware Census/)
  assert.match(workflow, /Instruction Plan/)
  assert.match(workflow, /Disassemble Verify/)
  assert.match(workflow, /\bconst\s+HARDWARE_PROBE_COMMAND\s*=/)
  assert.match(workflow, /\bconst\s+DISASSEMBLE_COMMAND\s*=/)
  assert.match(workflow, /ptx_regex/)
  assert.match(workflow, /sass_regex/)
  assert.match(workflow, /wgmma\.mma_async/)
  assert.match(workflow, /mma\.sync/)
  assert.match(workflow, /tcgen05/)
  assert.match(workflow, /hypothesis_not_realized/)
  assert.match(workflow, /do not claim an instruction path without disassembly evidence/)
  assert.match(workflow, /optionalSkills:\s*\[/)
  assert.match(workflow, /gemmptx-instruction-evidence/)

  assert.match(manifest, /name:\s*gemmptx-gemm-optimization/)
  assert.match(manifest, /method_category:\s*iterative_self_improving/)
  assert.match(manifest, /requires_harness:\s*true/)
  assert.match(manifest, /- gemm-ptx-optimization/)
  assert.match(manifest, /requires_ncu:\s*false/)
})

test('GemmPTX documentation states scope, required evidence, and limits', () => {
  const readme = read('GemmPTX/README.md')
  const zh = read('GemmPTX/README.zh-CN.md')

  assert.match(readme, /PTX\/SASS evidence/)
  assert.match(readme, /existing CUDA\/CuTe\/CUTLASS GEMM kernel/)
  assert.match(readme, /not a generic compute-bound optimizer/)
  assert.match(readme, /disassemble_command/)
  assert.match(readme, /hypothesis_not_realized/)

  assert.match(zh, /PTX\/SASS 证据/)
  assert.match(zh, /已有 CUDA\/CuTe\/CUTLASS GEMM kernel/)
  assert.match(zh, /不是通用 compute-bound optimizer/)
  assert.match(zh, /disassemble_command/)
  assert.match(zh, /hypothesis_not_realized/)
})

test('GemmPTX ships a workflow-local instruction evidence skill', () => {
  const skill = read('GemmPTX/skills/gemmptx-instruction-evidence/SKILL.md')
  const readme = read('GemmPTX/README.md')

  assert.match(skill, /name:\s*gemmptx-instruction-evidence/)
  assert.match(skill, /description:\s*Use when/)
  assert.match(skill, /mma\.sync/)
  assert.match(skill, /wgmma\.mma_async/)
  assert.match(skill, /cp\.async\.bulk\.tensor/)
  assert.match(skill, /tcgen05/)
  assert.match(skill, /hypothesis_not_realized/)
  assert.match(skill, /PTX\/SASS/)
  assert.match(skill, /Do not optimize from profiler counters until/)
  assert.match(skill, /expect_tx/)
  assert.match(skill, /async proxy fence/)
  assert.match(readme, /gemmptx-instruction-evidence/)
})
