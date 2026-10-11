#!/usr/bin/env node
// patch-turn-timeout.js — codemod that propagates the withTurnTimeout watchdog
// (from _meta/scaffolding/turn-timeout.js) to every workflow that has agent-retry
// but not yet turn-timeout.
//
// For each workflow .js (or `--all`):
//   1. With --refresh, replace an existing generated block from the SSOT.
//      Otherwise an existing `function withTurnTimeout` is left unchanged.
//   2. Else if it has `// --- END inlined agent-retry scaffolding ---` -> INSERT
//      the turn-timeout block right after it.
//   3. Else -> skip (no agent-retry; turn-timeout is meaningless without it).
//
// Declaring the function is HARMLESS: it's defined but not invoked until a
// workflow wraps its doer agent() calls with `withTurnTimeout(() => agent(...), label, retryOptions)`.
// That call-site wrapping is a per-workflow follow-up (the 3 that already have it
// — ARGUS, CUDAAgent, KSearch — also call it in their loops). This codemod makes
// the CAPABILITY available everywhere; activation is per-workflow.
//
// Validation is the caller's job: use the Host async-wrapper parser and guards.

const fs = require('fs')
const path = require('path')
const {buildCodeMask, buildPairMaps} = require('./add-agent-retry-scaffolding.js')

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

function transform(src, block, refresh = false) {
  if (refresh && src.includes(BEGIN)) {
    const start = src.indexOf(BEGIN), end = src.indexOf(END, start)
    if (end < 0) throw new Error('turn-timeout: missing END sentinel')
    return {src: src.slice(0, start) + BEGIN + '\n' + block + '\n' + END + src.slice(end + END.length), status: 'refreshed'}
  }
  if (src.includes('function withTurnTimeout(')) return { src, status: 'already-has' }
  const gi = src.indexOf(AGENT_RETRY_END)
  if (gi === -1) return { src, status: 'no-agent-retry' }
  const insertAt = gi + AGENT_RETRY_END.length
  const inlined = BEGIN + '\n' + block + '\n' + END
  return { src: src.slice(0, insertAt) + '\n\n' + inlined + src.slice(insertAt), status: 'inserted' }
}

// Migrate only the known canonical call shape. The existing code mask and
// delimiter map preserve prompt templates, nested expressions and retry options.
function migrateCalls(src) {
  const mask = buildCodeMask(src)
  const pairs = buildPairMaps(src, mask).openToClose
  const calls = []
  const pattern = /\bwithTurnTimeout\s*\(\s*agentRetry\s*\(/g
  for (const match of src.matchAll(pattern)) {
    if (!mask[match.index]) continue
    const outer = src.indexOf('(', match.index)
    const inner = match.index + match[0].lastIndexOf('(')
    const innerEnd = pairs.get(inner), outerEnd = pairs.get(outer)
    if (innerEnd == null || outerEnd == null) throw new Error('unbalanced timed agent call')
    const thunk = src.slice(inner+1, innerEnd).match(/^\s*\(\s*\)\s*=>\s*agent\s*\(/)
    if (!thunk) throw new Error('timed call migration requires () => agent(...)')
    const agentOpen = inner + 1 + thunk[0].lastIndexOf('(')
    const agentEnd = pairs.get(agentOpen)
    if (agentEnd == null || agentEnd >= innerEnd) throw new Error('unbalanced timed agent thunk')
    const options = src.slice(agentEnd+1, innerEnd).trim()
    const label = src.slice(innerEnd+1, outerEnd).trim()
    if (!options.startsWith(',') || !label.startsWith(',')) throw new Error('timed call arguments differ')
    const clean = value => value.slice(1).trim().replace(/,\s*$/, '')
    calls.push({start:match.index, end:outerEnd+1,
      value:`withTurnTimeout(${src.slice(inner+1,agentEnd+1).trim()}, ${clean(label)}, ${clean(options)})`})
  }
  for (let i=calls.length-1; i>=0; i--) {
    if (i && calls[i-1].end > calls[i].start) throw new Error('nested timed-call migration requires review')
    src=src.slice(0,calls[i].start)+calls[i].value+src.slice(calls[i].end)
  }
  return {src, count:calls.length}
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
    const migrated = process.argv.includes('--migrate-calls') ? migrateCalls(orig) : {src:orig,count:0}
    const r = transform(migrated.src, block, process.argv.includes('--refresh'))
    counts[r.status] = (counts[r.status] || 0) + 1
    if (r.src !== orig) {
      fs.writeFileSync(f, r.src)
      console.log(`  ${rel}: ${r.status}; migrated ${migrated.count} call(s)`)
    }
  }
  console.log(JSON.stringify(counts))
}

module.exports = {migrateCalls}
if (require.main === module) main()
