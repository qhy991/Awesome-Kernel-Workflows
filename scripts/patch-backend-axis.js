#!/usr/bin/env node
// patch-backend-axis.js — codemod that adopts the backend-axis helper cluster
// (normalizeSuitabilityValue / resolveBackendAxis / RESOLVED_BACKEND / USE_DRIVER
//  + driverPath / driverSh) into the SSOT regime defined in
// _meta/scaffolding/backend-axis.js.
//
// For each workflow .js on the argv (or `--all` for every top-level workflow),
// this WRAPS the existing byte-identical canonical blocks in BEGIN/END sentinels:
//
//   // --- BEGIN inlined backend-axis (resolve) scaffolding (from _meta/scaffolding/backend-axis.js) ---
//   <normalizeSuitabilityValue + resolveBackendAxis + RESOLVED_BACKEND + USE_DRIVER>
//   // --- END inlined backend-axis (resolve) scaffolding ---
//   ...
//   // --- BEGIN inlined backend-axis (driver) scaffolding (from _meta/scaffolding/backend-axis.js) ---
//   <driverPath + driverSh>
//   // --- END inlined backend-axis (driver) scaffolding ---
//
// Idempotent: re-running leaves already-sentinel'd files unchanged.
//
// DRIFT handling: if a workflow's block does NOT byte-match the SSOT canonical
// (e.g. 4 workflows have a drifted `driverSh`), the codemod SKIPS that block
// (never corrupts) and prints a `DRIFT` warning — the guard test
// (backend-axis-ssot-guard) surfaces these for manual review.
//
// `--refresh`: instead of wrapping, REPLACE the content between existing
// sentinels with the CURRENT canonical block from the SSOT (propagate a fix).
// Files lacking the sentinels are skipped with a warning.
//
// Validation is the caller's job: run `node --check` on every output file.

const fs = require('fs')
const path = require('path')

const REPO = path.resolve(__dirname, '..')
const SSOT = path.join(REPO, '_meta', 'scaffolding', 'backend-axis.js')

const BEGIN_A = '// --- BEGIN inlined backend-axis (resolve) scaffolding (from _meta/scaffolding/backend-axis.js) ---'
const END_A = '// --- END inlined backend-axis (resolve) scaffolding ---'
const BEGIN_B = '// --- BEGIN inlined backend-axis (driver) scaffolding (from _meta/scaffolding/backend-axis.js) ---'
const END_B = '// --- END inlined backend-axis (driver) scaffolding ---'

// Extract the two canonical sub-clusters verbatim from the SSOT.
function readCanonical() {
  const raw = fs.readFileSync(SSOT, 'utf8')
  const aStart = raw.indexOf('function normalizeSuitabilityValue')
  const aEndTok = 'const USE_DRIVER = !!args.backend_dir'
  const aEnd = raw.indexOf(aEndTok, aStart)
  if (aStart === -1 || aEnd === -1) throw new Error('backend-axis SSOT: sub-cluster A not found')
  const A = raw.slice(aStart, aEnd + aEndTok.length)
  const bStart = raw.indexOf('function driverPath')
  const bEnd = raw.lastIndexOf('}', bStart)  // last '}' at/after bStart is the driverSh close
  const bEndReal = raw.lastIndexOf('}')
  if (bStart === -1 || bEndReal === -1 || bEndReal < bStart) throw new Error('backend-axis SSOT: sub-cluster B not found')
  const B = raw.slice(bStart, bEndReal + 1)
  return { A, B }
}

// Wrap a byte-identical canonical block in sentinels (idempotent; skip drift).
// startToken distinguishes "drift" (has the fn but differs from SSOT) from
// "missing" (workflow doesn't use this cluster at all).
function wrap(src, block, begin, end, startToken) {
  if (src.includes(begin)) return { src, status: 'already-wrapped' }
  const idx = src.indexOf(block)
  if (idx !== -1) {
    return { src: src.slice(0, idx) + begin + '\n' + block + '\n' + end + src.slice(idx + block.length), status: 'wrapped' }
  }
  if (startToken && src.includes(startToken)) return { src, status: 'drift' }
  return { src, status: 'missing' }
}

// --refresh: replace content between existing sentinels with the canonical block.
function refresh(src, block, begin, end, startToken) {
  const bi = src.indexOf(begin)
  const ei = src.indexOf(end)
  if (bi === -1 || ei === -1) return { src, status: 'no-sentinels' }
  const before = src.slice(0, bi)
  const after = src.slice(ei + end.length)
  return { src: before + begin + '\n' + block + '\n' + end + after, status: 'refreshed' }
}

function workflowFiles() {
  const all = process.argv.includes('--all')
  const refresh = process.argv.includes('--refresh')
  let files = process.argv.slice(2).filter((a) => !a.startsWith('-'))
  if (all || !files.length) {
    files = fs.readdirSync(REPO).filter((d) => {
      const dir = path.join(REPO, d)
      return fs.statSync(dir).isDirectory() && fs.existsSync(path.join(dir, 'manifest.yaml'))
    }).map((d) => fs.readdirSync(path.join(REPO, d)).filter((f) => f.endsWith('.js')).map((f) => path.join(REPO, d, f))).flat()
  }
  return { files: files.filter((f) => fs.existsSync(f)), refresh }
}

function main() {
  const { files, refresh } = workflowFiles()
  const { A, B } = readCanonical()
  const apply = refresh ? refresh : wrap
  const counts = {}
  for (const f of files) {
    const orig = fs.readFileSync(f, 'utf8')
    const rel = path.relative(REPO, f)
    const rA = apply(orig, A, BEGIN_A, END_A, 'function normalizeSuitabilityValue')
    const rB = apply(rA.src, B, BEGIN_B, END_B, 'function driverPath')
    for (const st of [rA.status, rB.status]) counts[st] = (counts[st] || 0) + 1
    if (rA.status === 'drift') console.warn(`  DRIFT  ${rel}: resolve block present but not byte-identical to SSOT — skipped`)
    if (rB.status === 'drift') console.warn(`  DRIFT  ${rel}: driver block present but not byte-identical to SSOT — skipped (likely driverSh prefix/wording drift)`)
    if (rB.src !== orig) {
      fs.writeFileSync(f, rB.src)
      console.log(`  ${rel}: A=${rA.status} B=${rB.status}`)
    }
  }
  console.log(JSON.stringify(counts))
}

main()
