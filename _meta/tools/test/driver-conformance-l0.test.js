'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const { execSync } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const BACKENDS_DIR = path.join(ROOT, '_substrate/backends')
const VALIDATOR = path.join(BACKENDS_DIR, 'validate_backend.py')

const REQUIRED_FILES = [
  'manifest.json',
  'idioms.json',
  'build.sh',
  'run.sh',
  'profile.sh',
  'to_evidence.py',
]

const REQUIRED_MANIFEST_FIELDS = [
  'schema_version',
  'backend_id',
  'display_name',
  'source_ext',
  'hw_vendor',
  'threshold_profile',
  'capabilities',
]

function discoverDrivers() {
  const entries = fs.readdirSync(BACKENDS_DIR, { withFileTypes: true })
  return entries
    .filter(e => e.isDirectory() && !e.name.startsWith('_') && !e.name.startsWith('.'))
    .map(e => e.name)
    .sort()
}

const DRIVERS = discoverDrivers()

test('driver-conformance-l0: at least one registered driver exists', () => {
  assert.ok(DRIVERS.length > 0,
    `no driver directories found under ${BACKENDS_DIR}`)
})

for (const driver of DRIVERS) {
  const driverDir = path.join(BACKENDS_DIR, driver)

  test(`driver-conformance-l0 [${driver}]: all 6 required files exist`, () => {
    const missing = REQUIRED_FILES.filter(f => !fs.existsSync(path.join(driverDir, f)))
    assert.equal(missing.length, 0,
      `${driver} missing required files: ${missing.join(', ')}`)
  })

  test(`driver-conformance-l0 [${driver}]: manifest.json is valid JSON with required fields`, () => {
    const raw = fs.readFileSync(path.join(driverDir, 'manifest.json'), 'utf8')
    let manifest
    assert.doesNotThrow(() => { manifest = JSON.parse(raw) },
      `${driver}/manifest.json is not valid JSON`)
    for (const field of REQUIRED_MANIFEST_FIELDS) {
      assert.ok(field in manifest,
        `${driver}/manifest.json missing required field: ${field}`)
    }
  })

  test(`driver-conformance-l0 [${driver}]: manifest.backend_id matches directory name`, () => {
    const manifest = JSON.parse(fs.readFileSync(path.join(driverDir, 'manifest.json'), 'utf8'))
    assert.equal(manifest.backend_id, driver,
      `manifest.backend_id="${manifest.backend_id}" does not match directory name "${driver}"`)
  })

  test(`driver-conformance-l0 [${driver}]: idioms.json is valid JSON with methods object`, () => {
    const raw = fs.readFileSync(path.join(driverDir, 'idioms.json'), 'utf8')
    let idioms
    assert.doesNotThrow(() => { idioms = JSON.parse(raw) },
      `${driver}/idioms.json is not valid JSON`)
    assert.ok(typeof idioms.methods === 'object' && idioms.methods !== null,
      `${driver}/idioms.json must have a "methods" object`)
  })

  test(`driver-conformance-l0 [${driver}]: shell scripts exist and have shebang lines`, () => {
    for (const sh of ['build.sh', 'run.sh', 'profile.sh']) {
      const content = fs.readFileSync(path.join(driverDir, sh), 'utf8')
      assert.match(content, /^#!\//, `${driver}/${sh} must start with a shebang line`)
    }
  })

  test(`driver-conformance-l0 [${driver}]: to_evidence.py exists and is non-empty`, () => {
    const stat = fs.statSync(path.join(driverDir, 'to_evidence.py'))
    assert.ok(stat.size > 0, `${driver}/to_evidence.py must be non-empty`)
  })

  test(`driver-conformance-l0 [${driver}]: python validator passes (validate_backend.py)`, () => {
    assert.ok(fs.existsSync(VALIDATOR),
      `validate_backend.py must exist at ${VALIDATOR}`)
    const result = execSync(
      `python3 "${VALIDATOR}" "${driverDir}"`,
      { encoding: 'utf8', cwd: ROOT }
    )
    const parsed = JSON.parse(result)
    assert.equal(parsed.ok, true,
      `validate_backend.py failed for ${driver}: ${JSON.stringify(parsed.errors)}`)
  })
}
