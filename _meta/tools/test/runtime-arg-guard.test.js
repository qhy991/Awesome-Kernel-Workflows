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

function topLevelWorkflowFiles() {
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

function templateFiles(dirname) {
  const dir = path.join(ROOT, dirname)
  if (!fs.existsSync(dir)) return []
  return fs.readdirSync(dir, { withFileTypes: true })
    .filter(entry => entry.isFile() && entry.name.endsWith('.js'))
    .map(entry => path.join(dir, entry.name))
    .sort()
}

function guardedRuntimeInputs() {
  return [
    ...topLevelWorkflowFiles(),
    ...templateFiles('_templates'),
    ...templateFiles('_meta/templates'),
  ].sort()
}

test('workflow runtime inputs inline arg_guard before reading args fields', () => {
  const failures = []
  for (const file of guardedRuntimeInputs()) {
    const text = fs.readFileSync(file, 'utf8')
    const guardFn = text.indexOf('function __unwrapArgs')
    const unwrap = text.indexOf('args = __unwrapArgs')
    const suitabilityCall = text.search(/^\s*assertWorkflowSuitability\(\)/m)
    if (guardFn < 0 || unwrap < 0) {
      failures.push(`${relPath(file)}: missing inlined __unwrapArgs guard`)
      continue
    }
    if (suitabilityCall >= 0 && unwrap > suitabilityCall) {
      failures.push(`${relPath(file)}: suitability guard runs before args unwrap`)
    }
    if (/import\s+{\s*unwrapArgs\b/.test(text)) {
      failures.push(`${relPath(file)}: static arg_guard import is not safe for bare-script Workflow runtime`)
    }
  }
  assert.equal(
    failures.length,
    0,
    `Every workflow/template must inline arg_guard before args.* reads:\n${failures.join('\n')}`,
  )
})

test('patch-arg-guard emits the bare-script-safe inlined guard', () => {
  const script = fs.readFileSync(path.join(ROOT, 'scripts/patch-arg-guard.js'), 'utf8')
  assert.ok(script.includes('function __unwrapArgs'), 'patch script must generate the inlined guard')
  assert.ok(script.includes('args = __unwrapArgs'), 'patch script must generate the unwrap assignment')
  assert.ok(!/import\s+{\s*unwrapArgs\b/.test(script), 'patch script must not generate static imports')
})
