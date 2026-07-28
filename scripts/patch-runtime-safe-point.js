#!/usr/bin/env node
// Inline or refresh the canonical runtime-safe-point helper in selected workflows.

const fs = require('fs')
const path = require('path')

const REPO = path.resolve(__dirname, '..')
const SSOT = path.join(REPO, '_meta', 'scaffolding', 'runtime-safe-point.js')
const BEGIN = '// --- BEGIN inlined runtime-safe-point scaffolding (from _meta/scaffolding/runtime-safe-point.js) ---'
const END = '// --- END inlined runtime-safe-point scaffolding ---'
const AGENT_RETRY_END = '// --- END inlined agent-retry scaffolding ---'

function readBlock() {
  const raw = fs.readFileSync(SSOT, 'utf8')
  const start = raw.indexOf('async function __workflowRuntimeSafePoint(ctx) {')
  if (start === -1) throw new Error('runtime-safe-point SSOT: helper not found')
  return raw.slice(start).trim()
}

function transform(src, block) {
  const inlined = `${BEGIN}\n${block}\n${END}`
  const start = src.indexOf(BEGIN)
  if (start !== -1) {
    const finish = src.indexOf(END, start)
    if (finish === -1) throw new Error('runtime-safe-point block has no end sentinel')
    return src.slice(0, start) + inlined + src.slice(finish + END.length)
  }
  const anchor = src.indexOf(AGENT_RETRY_END)
  if (anchor === -1) throw new Error('workflow has no agent-retry insertion anchor')
  const at = anchor + AGENT_RETRY_END.length
  return src.slice(0, at) + '\n\n' + inlined + src.slice(at)
}

function main() {
  const files = process.argv.slice(2).map((entry) => path.resolve(entry))
  if (!files.length) {
    throw new Error('usage: patch-runtime-safe-point.js <workflow.js> [...]')
  }
  const block = readBlock()
  for (const file of files) {
    const original = fs.readFileSync(file, 'utf8')
    const updated = transform(original, block)
    if (updated !== original) {
      fs.writeFileSync(file, updated)
      console.log(path.relative(REPO, file))
    }
  }
}

main()
