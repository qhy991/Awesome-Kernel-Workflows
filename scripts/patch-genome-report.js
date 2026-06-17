#!/usr/bin/env node
// scripts/patch-genome-report.js
//
// Injects a lightweight self-report scribe after every standalone `phase('X')`
// call in every workflow .js, so a running workflow emits a live, tail-able
// `${args.exp_dir}/genome.jsonl` (one line per stage entry / loop iteration).
// See _meta/genome-trajectory-schema.md. Idempotent: re-running is a no-op.
//
// Run from repo root:
//   node scripts/patch-genome-report.js [--dry]
//
// What it injects, once after `export const meta = {...}`:
//   async function __genomeReport(phaseName, wfName) { ... await agent(...) ... }
// and rewrites each   phase('X')   ->   phase('X'); await __genomeReport('X', meta.name)
//
// The orchestration script is sandboxed (no fs, no Date) so the actual write is
// done by a subagent; the call is wrapped in try/catch so observability can
// NEVER break the workflow. Each candidate is validated with `node --check`
// (ESM) before being written; a file that would not parse is left untouched.

import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { execFile } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const SELF_DIR = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.dirname(SELF_DIR)
const DRY = process.argv.includes('--dry')

const TOP_LEVEL_BLOCKLIST = new Set([
  '_manifests', '_meta', '_substrate', '_templates', '_tools',
  'scripts', 'docs', 'badges', 'node_modules', '.git', 'paper', 'ppt', 'outputs',
])

const MARKER = '__genomeReport'
// Standalone phase('X') / phase("X") — not `.phase(` (method) or `phase:` (option).
const PHASE_RE = /(?<![.\w])phase\((['"])(.+?)\1\)/g

async function findWorkflowJs() {
  const entries = await fs.readdir(REPO_ROOT, { withFileTypes: true })
  const dirs = entries
    .filter(e => e.isDirectory() && !TOP_LEVEL_BLOCKLIST.has(e.name))
    .map(e => path.join(REPO_ROOT, e.name))
  const out = []
  for (const dir of dirs) {
    for (const f of await fs.readdir(dir)) {
      if (f.endsWith('.js')) out.push(path.join(dir, f))
    }
  }
  // Templates too, so generated workflows inherit the scribe.
  const tplDir = path.join(REPO_ROOT, '_templates')
  try {
    for (const f of await fs.readdir(tplDir)) {
      if (f.endsWith('.js')) out.push(path.join(tplDir, f))
    }
  } catch (_) { /* ok */ }
  return out
}

function findMetaEnd(src) {
  const start = src.indexOf('export const meta')
  if (start < 0) return -1
  const braceOpen = src.indexOf('{', start)
  if (braceOpen < 0) return -1
  let depth = 0
  for (let i = braceOpen; i < src.length; ++i) {
    const c = src[i]
    if (c === '{') depth++
    else if (c === '}') {
      if (--depth === 0) {
        let j = i + 1
        while (j < src.length && (src[j] === ';' || src[j] === ' ' || src[j] === '\t')) j++
        if (src[j] === '\n') j++
        return j
      }
    }
  }
  return -1
}

const HELPER = [
  '',
  '// --- BEGIN genome-report (auto-inserted by scripts/patch-genome-report.js) ---',
  '// Self-reported, work-plane (forgeable) stage trace for observability + the',
  '// recombiner. NOT a trust anchor — see _meta/genome-trajectory-schema.md.',
  'async function __genomeReport(phaseName, wfName) {',
  '  try {',
  "    const __dir = (typeof args !== 'undefined' && args && args.exp_dir) ? args.exp_dir : '.'",
  '    await agent(',
  "      'Append exactly one line to ' + __dir + '/genome.jsonl (create it if missing; use a shell append: printf %s\\\\n ... >> file). ' +",
  '      \'The line must be this JSON on ONE line: {"workflow":"\' + wfName + \'","phase":"\' + phaseName + \'","ts":"<UTC>","status":"entered"}. \' +',
  "      'Produce <UTC> by running: date -u +%Y-%m-%dT%H:%M:%SZ . Do nothing else; modify no other file. Echo the exact line you appended.',",
  "      { label: 'genome:' + phaseName, phase: phaseName }",
  '    )',
  '  } catch (__e) { /* observability must never break the workflow */ }',
  '}',
  '// --- END genome-report ---',
  '',
].join('\n')

// Workflow bodies use top-level `return` / `await` (legal in the Workflow
// runtime, which wraps the body in a function, but NOT in a real ESM module). To
// syntax-check them we mirror that: hoist top-level imports, turn `export const
// meta` into `const meta`, and wrap the rest in an async function — the same
// trick as register-workflow-variant.sh.
function wrapForCheck(src) {
  const imports = []
  const body = src.replace(/^[ \t]*import\s.*$/gm, (m) => { imports.push(m.trim()); return '' })
                  .replace(/^([ \t]*)export\s+const\s+meta/m, '$1const meta')
  return imports.join('\n') + '\nasync function __wf__() {\n' + body + '\n}\n'
}

async function nodeCheck(source) {
  const tmp = path.join(os.tmpdir(), `genome-check-${process.pid}-${Math.floor(performance.now())}.mjs`)
  await fs.writeFile(tmp, wrapForCheck(source), 'utf8')
  try {
    await new Promise((resolve, reject) => {
      execFile('node', ['--check', tmp], (err, _o, stderr) =>
        err ? reject(new Error(stderr || String(err))) : resolve())
    })
    return { ok: true }
  } catch (e) {
    return { ok: false, error: String(e.message || e).split('\n').slice(0, 3).join(' ') }
  } finally {
    await fs.rm(tmp, { force: true })
  }
}

async function patchOne(filePath) {
  const src = await fs.readFile(filePath, 'utf8')
  if (src.includes(MARKER)) return { filePath, status: 'already_patched' }
  if (!src.includes('export const meta')) return { filePath, status: 'skipped_no_meta' }

  PHASE_RE.lastIndex = 0
  const nPhases = (src.match(PHASE_RE) || []).length
  if (nPhases === 0) return { filePath, status: 'skipped_no_phase' }

  const insertAt = findMetaEnd(src)
  if (insertAt < 0) return { filePath, status: 'skipped_meta_parse_failed' }

  const withHelper = src.slice(0, insertAt) + HELPER + src.slice(insertAt)
  PHASE_RE.lastIndex = 0
  const next = withHelper.replace(
    PHASE_RE,
    (_m, q, name) => `phase(${q}${name}${q}); await __genomeReport(${q}${name}${q}, meta.name)`)

  // Gate on node --check ONLY when the ORIGINAL file is parseable. Templates
  // carry non-JS placeholders ({{PHASES_ARRAY}}) so they never parse; for those
  // we apply the (purely additive) injection without a gate.
  const orig = await nodeCheck(src)
  if (orig.ok) {
    const check = await nodeCheck(next)
    if (!check.ok) return { filePath, status: 'skipped_check_failed', detail: check.error }
  }
  const tag = orig.ok ? '' : '_template_unchecked'

  if (!DRY) await fs.writeFile(filePath, next, 'utf8')
  return { filePath, status: (DRY ? 'would_patch' : 'patched') + tag, phases: nPhases }
}

async function main() {
  const files = await findWorkflowJs()
  const tally = {}
  for (const f of files) {
    const r = await patchOne(f)
    tally[r.status] = (tally[r.status] || 0) + 1
    const rel = path.relative(REPO_ROOT, r.filePath)
    const extra = r.phases ? ` (${r.phases} phases)` : (r.detail ? ` — ${r.detail}` : '')
    console.log(`${r.status.padEnd(24)} ${rel}${extra}`)
  }
  console.log('\nSummary: ' + Object.entries(tally).map(([k, v]) => `${k}=${v}`).join(' ') +
    ` total=${files.length}${DRY ? '  [DRY RUN — no files written]' : ''}`)
}

main().catch(e => { console.error(e); process.exit(1) })
