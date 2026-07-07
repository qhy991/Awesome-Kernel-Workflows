'use strict'
// Verify the backend-axis scaffolding extraction (① from the公用组件 analysis).
//
// The cluster (normalizeSuitabilityValue / resolveBackendAxis / RESOLVED_BACKEND /
// USE_DRIVER + driverPath / driverSh) was copy-pasted across 15+ workflows and
// had drifted. The SSOT is _meta/scaffolding/backend-axis.js; the codemod
// scripts/patch-backend-axis.js wraps byte-identical copies in BEGIN/END
// sentinels (so `--refresh` can re-sync). This test:
//   1. Asserts the SSOT exists + both canonical sub-clusters extract.
//   2. For every sentinel-wrapped workflow: the block between sentinels MUST
//      byte-match the SSOT canonical (drift detection — a hand-edit to one
//      workflow's wrapped block fails this).
//   3. Snapshots the "known drift" set — workflows that HAVE the cluster's start
//      token but aren't (yet) sentinel-wrapped (their copy drifted from SSOT and
//      needs manual review before adoption). If this set changes, the test fails
//      to prompt an update (resolution or a new drift to triage).
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const SSOT = fs.readFileSync(path.join(ROOT, '_meta/scaffolding/backend-axis.js'), 'utf8')
const BEGIN_A = '// --- BEGIN inlined backend-axis (resolve) scaffolding (from _meta/scaffolding/backend-axis.js) ---'
const END_A = '// --- END inlined backend-axis (resolve) scaffolding ---'
const BEGIN_B = '// --- BEGIN inlined backend-axis (driver) scaffolding (from _meta/scaffolding/backend-axis.js) ---'
const END_B = '// --- END inlined backend-axis (driver) scaffolding ---'

// Extract the canonical sub-clusters from the SSOT (same logic as the codemod).
function readCanonical() {
  const aStart = SSOT.indexOf('function normalizeSuitabilityValue')
  const aEndTok = 'const USE_DRIVER = !!args.backend_dir'
  const aEnd = SSOT.indexOf(aEndTok, aStart)
  const A = SSOT.slice(aStart, aEnd + aEndTok.length)
  const bStart = SSOT.indexOf('function driverPath')
  const bEnd = SSOT.lastIndexOf('}')
  const B = SSOT.slice(bStart, bEnd + 1)
  return { A, B }
}
const { A: CANON_A, B: CANON_B } = readCanonical()

// All top-level workflow .js files (manifest.yaml + a .js entrypoint).
function workflowFiles() {
  return fs.readdirSync(ROOT).filter((d) => {
    const dir = path.join(ROOT, d)
    return fs.statSync(dir).isDirectory() && fs.existsSync(path.join(dir, 'manifest.yaml'))
  }).map((d) => fs.readdirSync(path.join(ROOT, d)).filter((f) => f.endsWith('.js')).map((f) => path.join(ROOT, d, f))).flat()
}

function blockBetween(src, begin, end) {
  const bi = src.indexOf(begin)
  const ei = src.indexOf(end)
  if (bi === -1 || ei === -1) return null
  return src.slice(bi + begin.length + 1, ei - 1)  // between begin+\n and \n+end
}

test('SSOT: backend-axis.js defines both canonical sub-clusters', () => {
  assert.ok(CANON_A.includes('function normalizeSuitabilityValue') && CANON_A.includes('const USE_DRIVER = !!args.backend_dir'),
    'sub-cluster A (resolve) must span normalizeSuitabilityValue → USE_DRIVER')
  assert.ok(CANON_B.includes('function driverPath') && CANON_B.includes('function driverSh'),
    'sub-cluster B (driver) must span driverPath → driverSh')
})

test('every sentinel-wrapped workflow byte-matches the SSOT canonical (no silent drift)', () => {
  let wrappedA = 0, wrappedB = 0
  for (const f of workflowFiles()) {
    const src = fs.readFileSync(f, 'utf8')
    const rel = path.relative(ROOT, f)
    const a = blockBetween(src, BEGIN_A, END_A)
    if (a !== null) {
      wrappedA++
      assert.equal(a, CANON_A, `${rel}: resolve block between sentinels must byte-match the SSOT (run \`node scripts/patch-backend-axis.js --refresh\` to re-sync)`)
    }
    const b = blockBetween(src, BEGIN_B, END_B)
    if (b !== null) {
      wrappedB++
      assert.equal(b, CANON_B, `${rel}: driver block between sentinels must byte-match the SSOT (run \`node scripts/patch-backend-axis.js --refresh\` to re-sync)`)
    }
  }
  assert.ok(wrappedA >= 10, `expected >=10 workflows with the resolve block sentinel-wrapped; got ${wrappedA}`)
  assert.ok(wrappedB >= 10, `expected >=10 workflows with the driver block sentinel-wrapped; got ${wrappedB}`)
})

test('known-drift snapshot: workflows with the cluster but NOT yet SSOT-adopted', () => {
  // These have `function normalizeSuitabilityValue` (the start token) but the full
  // canonical block drifted, so the codemod skipped them. They need manual review
  // (normalize to SSOT, or confirm intentional divergence). If this set changes,
  // update the snapshot — a new drift is a regression to triage.
  //
  // AccelOpt: INTENTIONAL divergence — its `resolveBackend` (not `resolveBackendAxis`)
  // has a default-backend fallback (WORKFLOW_META.method_supported_backends /
  // default_backend) and omits the `backend_dir`-required guard. Standardizing to
  // the SSOT canonical would change routing (drop the default fallback, add the
  // backend_dir guard) — a behavior change, not a syntactic fix. Kept as-is until
  // the default-fallback is parameterized into the SSOT or phased out.
  // (AKO4X / AdaExplore / Astra were previously in this snapshot but their blocks
  // turned out to be byte-identical to the SSOT after the ①/③ merges — wrapped.)
  const KNOWN_DRIFT = new Set(['AccelOpt'])
  const actualDrift = new Set()
  for (const f of workflowFiles()) {
    const src = fs.readFileSync(f, 'utf8')
    const wf = path.basename(path.dirname(f))
    // has the resolve start token but no sentinel wrap -> drifted from SSOT
    if (src.includes('function normalizeSuitabilityValue') && !src.includes(BEGIN_A)) {
      actualDrift.add(wf)
    }
  }
  assert.deepEqual([...actualDrift].sort(), [...KNOWN_DRIFT].sort(),
    `known-drift set changed; expected ${[...KNOWN_DRIFT].sort().join(', ')}, got ${[...actualDrift].sort().join(', ')}. ` +
    `Either resolve the drift (normalize to SSOT + re-run the codemod) or add new drifts to the snapshot with a note.`)
})
