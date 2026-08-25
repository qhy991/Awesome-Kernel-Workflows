'use strict'

const { test } = require('node:test')
const assert = require('node:assert/strict')
const { execFileSync } = require('node:child_process')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const SCRIPT = path.join(ROOT, 'scripts', 'patch-typed-args.js')
const SSOT = path.join(ROOT, '_meta', 'scaffolding', 'typed-args.js')
const BEGIN = '// --- BEGIN inlined typed-args (from _meta/scaffolding/typed-args.js) ---'
const END = '// --- END inlined typed-args ---'
const OLD_BEGIN = '// --- BEGIN typed-args (channel ② experience_excerpts) ---'
const OLD_END = '// --- END typed-args ---'

function canonicalContent() {
  const source = fs.readFileSync(SSOT, 'utf8')
  const begin = source.indexOf(OLD_BEGIN)
  const end = source.indexOf(OLD_END, begin)
  return source.slice(begin + OLD_BEGIN.length + 1, end - 1)
}

test('--refresh replaces an existing typed-args projection from the SSOT', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'akw-typed-args-'))
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const fixture = path.join(dir, 'workflow.js')
  fs.writeFileSync(fixture, `before\n${BEGIN}\nstale projection\n${END}\nafter\n`)

  execFileSync(process.execPath, [SCRIPT, '--refresh', fixture], { cwd: ROOT })

  const refreshed = fs.readFileSync(fixture, 'utf8')
  assert.equal(refreshed, `before\n${BEGIN}\n${canonicalContent()}\n${END}\nafter\n`)
})
