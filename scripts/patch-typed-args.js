#!/usr/bin/env node
// patch-typed-args.js — codemod that adopts the KerSor ②③ typed-args channel
// block (from _meta/scaffolding/typed-args.js) into every workflow.
//
// For each workflow .js (or `--all`):
//   1. If it already has the `BEGIN inlined typed-args` sentinel -> skip (idempotent).
//   2. If it has the older `BEGIN typed-args (channel ②` block -> replace the old
//      markers with the SSOT inlined sentinels (content must byte-match the SSOT;
//      drift is skipped + reported).
//   3. If it has no typed-args block -> INSERT the SSOT block right after the
//      `// --- END inlined arg_guard ---` marker (the canonical location).
//
// `--refresh`: replace the content between existing inlined sentinels with the
// current canonical block. Files without a complete sentinel pair are skipped.
//
// Propagating the block DECLARES the KerSor ②③ channels (EXPERIENCE_EXCERPTS,
// ATTEMPT_EVIDENCE/PLAN, FAILED_STRATEGY_IDS) + the __experienceBlock() /
// __attemptBlock() helpers in every workflow. The consts degrade safely to
// null/[] when the channel is absent, so declaring the block is HARMLESS even
// if a workflow's prompts don't yet call the helpers (surfacing them in prompts
// is a per-workflow follow-up). This is contract completeness, not a behavior
// change.
//
// Validation is the caller's job: run `node --check` on every output file.

const fs = require('fs')
const path = require('path')

const REPO = path.resolve(__dirname, '..')
const SSOT = path.join(REPO, '_meta', 'scaffolding', 'typed-args.js')

const BEGIN_INLINED = '// --- BEGIN inlined typed-args (from _meta/scaffolding/typed-args.js) ---'
const END_INLINED = '// --- END inlined typed-args ---'
const OLD_BEGIN = '// --- BEGIN typed-args (channel ② experience_excerpts) ---'
const OLD_END = '// --- END typed-args ---'
const ARG_GUARD_END = '// --- END inlined arg_guard ---'

// Extract the canonical block CONTENT (between the old BEGIN/END markers) from the SSOT.
function readCanonContent() {
  const raw = fs.readFileSync(SSOT, 'utf8')
  const b = raw.indexOf(OLD_BEGIN)
  const e = raw.indexOf(OLD_END, b)
  if (b === -1 || e === -1) throw new Error('typed-args SSOT: old markers not found')
  return raw.slice(b + OLD_BEGIN.length + 1, e - 1)  // between BEGIN+\n and \n+END
}

function inlinedBlock(content) {
  return BEGIN_INLINED + '\n' + content + '\n' + END_INLINED
}

function transform(src, content, rel, refresh) {
  if (refresh) {
    const b = src.indexOf(BEGIN_INLINED)
    if (b === -1) return { src, status: 'no-sentinels' }
    const e = src.indexOf(END_INLINED, b)
    if (e === -1) return { src, status: 'malformed-inlined' }
    return {
      src: src.slice(0, b) + inlinedBlock(content) + src.slice(e + END_INLINED.length),
      status: 'refreshed',
    }
  }
  if (src.includes(BEGIN_INLINED)) return { src, status: 'already-wrapped' }
  // Case 2: has the old markers -> replace with inlined (verify content matches).
  if (src.includes(OLD_BEGIN)) {
    const b = src.indexOf(OLD_BEGIN)
    const e = src.indexOf(OLD_END, b)
    if (e === -1) return { src, status: 'malformed-old' }
    const existing = src.slice(b + OLD_BEGIN.length + 1, e - 1)
    if (existing !== content) return { src, status: 'drift' }
    return { src: src.slice(0, b) + inlinedBlock(content) + src.slice(e + OLD_END.length), status: 'rewrapped' }
  }
  // Case 3: no block -> insert after END inlined arg_guard.
  const gi = src.indexOf(ARG_GUARD_END)
  if (gi === -1) return { src, status: 'no-arg-guard-end' }
  const insertAt = gi + ARG_GUARD_END.length
  return { src: src.slice(0, insertAt) + '\n\n' + inlinedBlock(content) + src.slice(insertAt), status: 'inserted' }
}

function workflowFiles() {
  return fs.readdirSync(REPO).filter((d) => {
    const dir = path.join(REPO, d)
    return fs.statSync(dir).isDirectory() && fs.existsSync(path.join(dir, 'manifest.yaml'))
  }).map((d) => fs.readdirSync(path.join(REPO, d)).filter((f) => f.endsWith('.js')).map((f) => path.join(REPO, d, f))).flat()
}

function main() {
  const content = readCanonContent()
  const refresh = process.argv.includes('--refresh')
  const files = process.argv.slice(2).filter((a) => !a.startsWith('-')).map((p) => path.resolve(p)).filter((f) => fs.existsSync(f))
  const targets = files.length ? files : workflowFiles()
  const counts = {}
  for (const f of targets) {
    const orig = fs.readFileSync(f, 'utf8')
    const rel = path.relative(REPO, f)
    const r = transform(orig, content, rel, refresh)
    counts[r.status] = (counts[r.status] || 0) + 1
    if (r.status === 'drift') console.warn(`  DRIFT  ${rel}: typed-args content not byte-identical to SSOT — skipped`)
    if (r.status === 'no-arg-guard-end') console.warn(`  WARN  ${rel}: no '${ARG_GUARD_END}' marker — skipped (insert location unknown)`)
    if (r.status === 'no-sentinels') console.warn(`  WARN  ${rel}: no inlined typed-args sentinels — skipped`)
    if (r.status === 'malformed-inlined') console.warn(`  WARN  ${rel}: BEGIN without END — skipped`)
    if (r.src !== orig) {
      fs.writeFileSync(f, r.src)
      console.log(`  ${rel}: ${r.status}`)
    }
  }
  console.log(JSON.stringify(counts))
}

main()
