#!/usr/bin/env node
// patch-turn-timeout.js — codemod that propagates the withTurnTimeout watchdog
// (from _meta/scaffolding/turn-timeout.js) to every workflow that has agent-retry
// but not yet turn-timeout.
//
// For each workflow .js (or `--all`):
//   1. If it already has `function withTurnTimeout` -> skip (idempotent).
//   2. Else if it has `// --- END inlined agent-retry scaffolding ---` -> INSERT
//      the turn-timeout block right after it.
//   3. Else -> skip (no agent-retry; turn-timeout is meaningless without it).
//
// Declaring the function is HARMLESS: it's defined but not invoked until a
// workflow wraps its doer agent() calls with `withTurnTimeout(agentRetry(...), label)`.
// That call-site wrapping is a per-workflow follow-up (the 3 that already have it
// — ARGUS, CUDAAgent, KSearch — also call it in their loops). This codemod makes
// the CAPABILITY available everywhere; activation is per-workflow.
//
// Validation is the caller's job: run `node --check` on every output file.

const fs = require('fs')
const path = require('path')

const REPO = path.resolve(__dirname, '..')
const SSOT = path.join(REPO, '_meta', 'scaffolding', 'turn-timeout.js')

const BEGIN = '// --- BEGIN inlined turn-timeout scaffolding (from _meta/scaffolding/turn-timeout.js) ---'
const END = '// --- END inlined turn-timeout scaffolding ---'
const AGENT_RETRY_END = '// --- END inlined agent-retry scaffolding ---'

function readBlock() {
  const raw = fs.readFileSync(SSOT, 'utf8')
  // Use LAST occurrence — the SSOT's USAGE comment also contains `const TURN_TIMEOUT_MS`
  // (as an example), and indexOf would start the slice there (inside the comment),
  // corrupting the block. The actual const is the last occurrence (after the comment).
  const s = raw.lastIndexOf('const TURN_TIMEOUT_MS =')
  const e = raw.lastIndexOf('}')
  if (s === -1 || e === -1 || e < s) throw new Error('turn-timeout SSOT: block not found')
  return raw.slice(s, e + 1)
}

function transform(src, block) {
  if (src.includes('function withTurnTimeout(')) return { src, status: 'already-has' }
  const gi = src.indexOf(AGENT_RETRY_END)
  if (gi === -1) return { src, status: 'no-agent-retry' }
  const insertAt = gi + AGENT_RETRY_END.length
  const inlined = BEGIN + '\n' + block + '\n' + END
  return { src: src.slice(0, insertAt) + '\n\n' + inlined + src.slice(insertAt), status: 'inserted' }
}

function workflowFiles() {
  return fs.readdirSync(REPO).filter((d) => {
    const dir = path.join(REPO, d)
    return fs.statSync(dir).isDirectory() && fs.existsSync(path.join(dir, 'manifest.yaml'))
  }).map((d) => fs.readdirSync(path.join(REPO, d)).filter((f) => f.endsWith('.js')).map((f) => path.join(REPO, d, f))).flat()
}

function main() {
  const block = readBlock()
  const files = process.argv.slice(2).filter((a) => !a.startsWith('-')).map((p) => path.resolve(p)).filter((f) => fs.existsSync(f))
  const targets = files.length ? files : workflowFiles()
  const counts = {}
  for (const f of targets) {
    const orig = fs.readFileSync(f, 'utf8')
    const rel = path.relative(REPO, f)
    const r = transform(orig, block)
    counts[r.status] = (counts[r.status] || 0) + 1
    if (r.src !== orig) {
      fs.writeFileSync(f, r.src)
      console.log(`  ${rel}: ${r.status}`)
    }
  }
  console.log(JSON.stringify(counts))
}

main()
