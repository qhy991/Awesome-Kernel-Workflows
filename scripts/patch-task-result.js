'use strict'
const fs = require('node:fs'), path = require('node:path')
const root = path.resolve(__dirname, '..')
const begin = '// --- BEGIN inlined task-result scaffolding (from _meta/scaffolding/task-result.js) ---'
const end = '// --- END inlined task-result scaffolding ---'
const helper = fs.readFileSync(path.join(root, '_meta/scaffolding/task-result.js'), 'utf8').trimEnd()
const block = begin + '\n' + helper + '\n' + end + '\n'
for (const name of process.argv.slice(2)) {
  const file = path.resolve(root, name)
  let source = fs.readFileSync(file, 'utf8')
  const existing = /\/\/ --- BEGIN inlined task-result scaffolding[^\n]*\n[\s\S]*?\/\/ --- END inlined task-result scaffolding ---\n?/
  if (existing.test(source)) {
    // Refresh at the existing owner boundary, including native-only workflows
    // that no longer carry the legacy sol-execbench insertion anchor.
    source = source.replace(existing, block)
  } else {
    const at = source.indexOf('\n// --- BEGIN sol-execbench-eval')
    if (at < 0) throw new Error('missing task-result block and post-meta insertion point: ' + name)
    source = source.slice(0, at) + '\n' + block + source.slice(at)
  }
  fs.writeFileSync(file, source)
}
