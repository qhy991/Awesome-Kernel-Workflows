'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')

const TOP_LEVEL_BLOCKLIST = new Set([
  '_manifests',
  '_meta',
  '_substrate',
  '_templates',
  '_tools',
  'scripts',
  'docs',
  'badges',
  'node_modules',
  '.git',
  'paper',
  'ppt',
  'outputs',
])

function relPath(abs) {
  return path.relative(ROOT, abs).split(path.sep).join('/')
}

function discoverTopLevelWorkflowJs() {
  const files = []
  for (const entry of fs.readdirSync(ROOT, { withFileTypes: true })) {
    if (!entry.isDirectory() || TOP_LEVEL_BLOCKLIST.has(entry.name)) continue
    const dir = path.join(ROOT, entry.name)
    for (const child of fs.readdirSync(dir, { withFileTypes: true })) {
      if (child.isFile() && child.name.endsWith('.js')) {
        files.push(path.join(dir, child.name))
      }
    }
  }
  return files.sort()
}

function discoverTemplateJs() {
  const files = []
  const templateDir = path.join(ROOT, '_templates')
  if (fs.existsSync(templateDir)) {
    for (const entry of fs.readdirSync(templateDir, { withFileTypes: true })) {
      if (entry.isFile() && entry.name.endsWith('.js')) {
        files.push(path.join(templateDir, entry.name))
      }
    }
  }
  return files.sort()
}

function discoverRuntimeMetaInputs() {
  const files = discoverTopLevelWorkflowJs()
  files.push(...discoverTemplateJs())
  files.push(path.join(ROOT, 'scripts/patch-genome-report.js'))
  return files.sort()
}

function discoverWorkflowNameInputs() {
  return [...discoverTopLevelWorkflowJs(), ...discoverTemplateJs()].sort()
}

function runtimeMetaRefs(filePath) {
  const text = fs.readFileSync(filePath, 'utf8')
  const refs = []
  for (const match of text.matchAll(/\bmeta\.[A-Za-z_][A-Za-z0-9_]*/g)) {
    if (text.slice(Math.max(0, match.index - 'import.'.length), match.index) === 'import.') {
      continue
    }
    const line = text.slice(0, match.index).split('\n').length
    refs.push(`${relPath(filePath)}:${line}: ${match[0]}`)
  }
  return refs
}

function declaredWorkflowNames(filePath) {
  const text = fs.readFileSync(filePath, 'utf8')
  const metaMatch = text.match(/export\s+const\s+meta\s*=\s*{[\s\S]*?\bname\s*:\s*(['"])(.*?)\1/)
  const workflowNameMatch = text.match(/\bconst\s+WORKFLOW_NAME\s*=\s*(['"])(.*?)\1/)
  if (!workflowNameMatch) return null
  return {
    file: relPath(filePath),
    metaName: metaMatch ? metaMatch[2] : null,
    workflowName: workflowNameMatch[2],
  }
}

test('workflow runtime bodies and generation helpers do not reference meta at runtime', () => {
  const offenders = discoverRuntimeMetaInputs().flatMap(runtimeMetaRefs)
  assert.equal(
    offenders.length,
    0,
    `Runtime code must not reference the exported meta object; use a body-scope WORKFLOW_NAME constant instead:\n${offenders.join('\n')}`,
  )
})

test('WORKFLOW_NAME constants stay aligned with exported meta names', () => {
  const mismatches = discoverWorkflowNameInputs()
    .map(declaredWorkflowNames)
    .filter(Boolean)
    .filter(({ metaName, workflowName }) => metaName !== workflowName)
    .map(({ file, metaName, workflowName }) => `${file}: meta.name=${metaName} WORKFLOW_NAME=${workflowName}`)

  assert.equal(
    mismatches.length,
    0,
    `WORKFLOW_NAME must equal the exported workflow name literal:\n${mismatches.join('\n')}`,
  )
})
