'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const crypto = require('node:crypto')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const BASELINE = JSON.parse(fs.readFileSync(path.join(FIX, 'substrate-baseline-shas.json'), 'utf8'))
const SUBSTRATE = path.join(ROOT, '_substrate')

function walk(dir, results) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      if (entry.name === 'tests' || entry.name === '_fixtures') continue
      walk(full, results)
    } else if (entry.name.endsWith('.py') || entry.name.endsWith('.sh')) {
      results.push(full)
    }
  }
  return results
}

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex')
}

function relPath(abs) {
  return path.relative(ROOT, abs)
}

test('substrate diff-guard: baseline SHA manifest is non-empty and well-formed', () => {
  const keys = Object.keys(BASELINE)
  assert.ok(keys.length > 0, 'baseline manifest must list at least one file')
  for (const k of keys) {
    assert.match(k, /^_substrate\//, `key ${k} must be a _substrate/ relative path`)
    assert.match(BASELINE[k], /^[0-9a-f]{64}$/, `value for ${k} must be a 64-char hex SHA-256`)
  }
})

test('substrate diff-guard: every baselined file exists on disk', () => {
  for (const rel of Object.keys(BASELINE)) {
    const abs = path.join(ROOT, rel)
    assert.ok(fs.existsSync(abs), `baselined file ${rel} is missing from disk`)
  }
})

test('substrate diff-guard: every baselined file is byte-identical to its baseline SHA', () => {
  const drifted = []
  for (const rel of Object.keys(BASELINE)) {
    const abs = path.join(ROOT, rel)
    if (!fs.existsSync(abs)) continue
    const actual = sha256(abs)
    if (actual !== BASELINE[rel]) {
      drifted.push({ file: rel, expected: BASELINE[rel], actual })
    }
  }
  assert.equal(drifted.length, 0,
    `substrate files drifted from baseline:\n${drifted.map(d => `  ${d.file}: expected ${d.expected.slice(0, 12)}... got ${d.actual.slice(0, 12)}...`).join('\n')}`)
})

test('substrate diff-guard: no unexpected .py/.sh additions outside tests/_fixtures', () => {
  const onDisk = walk(SUBSTRATE, []).map(f => relPath(f)).sort()
  const inBaseline = Object.keys(BASELINE).sort()
  const additions = onDisk.filter(f => !inBaseline.includes(f))
  assert.equal(additions.length, 0,
    `unexpected substrate files not in baseline (add to baseline if intentional):\n  ${additions.join('\n  ')}`)
})

test('substrate diff-guard: no unexpected deletions (baseline file missing from disk)', () => {
  const onDisk = new Set(walk(SUBSTRATE, []).map(f => relPath(f)))
  const deletions = Object.keys(BASELINE).filter(f => !onDisk.has(f))
  assert.equal(deletions.length, 0,
    `baselined files missing from disk (remove from baseline if intentional):\n  ${deletions.join('\n  ')}`)
})
