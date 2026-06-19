#!/usr/bin/env node
// scripts/patch-model-tier.js
//
// Injects a `const MODEL = { mechanical, profile, judgment }` tier declaration
// after `export const meta = {...}` and rewrites known-mechanical/profile
// agent() calls (matched by `label:` prefix) to add `model: MODEL.<tier>` to
// their opts object. Idempotent: re-running is a no-op. Generalist already
// hand-rolls its MODEL declaration; this script detects that and skips the
// declaration but still annotates any unannotated agent() calls.
//
// Run from repo root:
//   node scripts/patch-model-tier.js [--dry]
//
// Label-prefix classification (audit-derived):
//   mechanical: load-driver, driver-build-*, driver-to-evidence-*,
//               driver-diagnose-*, driver-anti-cheat-*, diagnose-baseline,
//               memory_store, verify_insight, refute-insight
//   profile:    ncu-baseline, driver-run-*, driver-profile-*, profile
// Everything else is left alone (judgment tier by default).

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

const MARKER = '__modelTierApplied'

const MECHANICAL_PREFIXES = [
  'load-driver',
  'driver-build-',
  'driver-to-evidence-',
  'driver-diagnose-',
  'driver-anti-cheat-',
  'diagnose-baseline',
  'memory_store',
  'verify_insight',
  'refute-insight',
]
const PROFILE_PREFIXES = [
  'ncu-baseline',
  'driver-run-',
  'driver-profile-',
  'profile',
]

// label:  '<prefix>...'  or  "<prefix>..."  or  `<prefix>...`
const LABEL_RE = /label\s*:\s*[`'"]([a-zA-Z][-_a-zA-Z0-9]*)/g

function classifyLabel(prefix) {
  for (const p of MECHANICAL_PREFIXES) {
    if (prefix === p || prefix.startsWith(p)) return 'mechanical'
  }
  for (const p of PROFILE_PREFIXES) {
    if (prefix === p || prefix.startsWith(p)) return 'profile'
  }
  return null
}

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
  // Templates too, so generated workflows inherit the tier defaults.
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

// Walk backward from `pos` to find the matching opening brace `{` that is the
// nearest enclosing object literal. Counts nested braces.
function findEnclosingOptsBrace(src, pos) {
  let depth = 0
  for (let i = pos - 1; i >= 0; i--) {
    const c = src[i]
    if (c === '}') depth++
    else if (c === '{') {
      if (depth === 0) return i
      depth--
    }
  }
  return -1
}

const MODEL_DECL_HELPER = [
  '',
  '// --- BEGIN model-tier (auto-inserted by scripts/patch-model-tier.js) ---',
  '// Tier-based model routing: mechanical steps (run substrate scripts, parse',
  '// JSON) use cheaper models; profile steps (run eval/ncu) use mid-tier;',
  '// judgment steps (plan/implement/report) use the top tier. Tuneable via',
  '// args.model_{mechanical,profile,judgment}.',
  'const MODEL = {',
  "  mechanical: (typeof args !== 'undefined' && args && args.model_mechanical) || 'haiku',",
  "  profile: (typeof args !== 'undefined' && args && args.model_profile) || 'sonnet',",
  "  judgment: (typeof args !== 'undefined' && args && args.model_judgment) || 'opus',",
  '}',
  `// ${MARKER}`,
  '// --- END model-tier ---',
  '',
].join('\n')

// Marker-only stub for files that already have a hand-rolled MODEL declaration
// (e.g. Generalist). We still want to record that the codemod ran.
const MARKER_ONLY_HELPER = `\n// ${MARKER} (declaration pre-existing)\n`

function injectModelTier(src) {
  const patches = []
  LABEL_RE.lastIndex = 0
  let m
  while ((m = LABEL_RE.exec(src)) !== null) {
    const tier = classifyLabel(m[1])
    if (!tier) continue

    const braceOpen = findEnclosingOptsBrace(src, m.index)
    if (braceOpen < 0) continue

    // Skip if this opts object already has a model: key
    const before = src.slice(braceOpen, m.index)
    if (/\bmodel\s*:/.test(before)) continue

    patches.push({
      insertAt: braceOpen + 1,
      text: ` model: MODEL.${tier},`,
      tier,
    })
  }
  // Apply in reverse so earlier indices stay valid
  patches.sort((a, b) => b.insertAt - a.insertAt)
  let out = src
  for (const p of patches) {
    out = out.slice(0, p.insertAt) + p.text + out.slice(p.insertAt)
  }
  const tally = patches.reduce((acc, p) => ((acc[p.tier] = (acc[p.tier] || 0) + 1), acc), {})
  return { source: out, count: patches.length, tally }
}

function wrapForCheck(src) {
  const imports = []
  const body = src.replace(/^[ \t]*import\s.*$/gm, (m) => { imports.push(m.trim()); return '' })
                  .replace(/^([ \t]*)export\s+const\s+meta/m, '$1const meta')
  return imports.join('\n') + '\nasync function __wf__() {\n' + body + '\n}\n'
}

async function nodeCheck(source) {
  const tmp = path.join(os.tmpdir(), `model-tier-check-${process.pid}-${Math.floor(performance.now())}.mjs`)
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

  const insertAt = findMetaEnd(src)
  if (insertAt < 0) return { filePath, status: 'skipped_meta_parse_failed' }

  const hasModel = /\bconst\s+MODEL\s*=/.test(src)
  const helper = hasModel ? MARKER_ONLY_HELPER : MODEL_DECL_HELPER
  const withHelper = src.slice(0, insertAt) + helper + src.slice(insertAt)

  const { source: withTiers, count, tally } = injectModelTier(withHelper)

  if (count === 0 && hasModel) {
    // No tier annotations needed and MODEL already exists — write only the
    // marker so future runs short-circuit, but skip the syntax gate (purely
    // additive single-line comment can't break parse).
    if (!DRY) await fs.writeFile(filePath, withHelper, 'utf8')
    return { filePath, status: DRY ? 'would_mark_only' : 'marked_only', tally }
  }

  const orig = await nodeCheck(src)
  if (orig.ok) {
    const check = await nodeCheck(withTiers)
    if (!check.ok) return { filePath, status: 'skipped_check_failed', detail: check.error }
  }
  const tag = orig.ok ? '' : '_template_unchecked'

  if (!DRY) await fs.writeFile(filePath, withTiers, 'utf8')
  return {
    filePath,
    status: (DRY ? 'would_patch' : 'patched') + (hasModel ? '_marker_only_decl' : '_full_decl') + tag,
    tiered: count,
    tally,
  }
}

async function main() {
  const files = await findWorkflowJs()
  const tally = {}
  const tierTotals = { mechanical: 0, profile: 0 }
  for (const f of files) {
    const r = await patchOne(f)
    tally[r.status] = (tally[r.status] || 0) + 1
    const rel = path.relative(REPO_ROOT, r.filePath)
    const extra = r.tiered != null
      ? ` (+${r.tiered} agents: ${Object.entries(r.tally || {}).map(([k, v]) => `${k}=${v}`).join(' ')})`
      : (r.detail ? ` — ${r.detail}` : '')
    if (r.tally) {
      tierTotals.mechanical += r.tally.mechanical || 0
      tierTotals.profile += r.tally.profile || 0
    }
    console.log(`${r.status.padEnd(30)} ${rel}${extra}`)
  }
  console.log('\nSummary: ' + Object.entries(tally).map(([k, v]) => `${k}=${v}`).join(' ') +
    ` total=${files.length} mechanical_tagged=${tierTotals.mechanical} profile_tagged=${tierTotals.profile}` +
    `${DRY ? '  [DRY RUN — no files written]' : ''}`)
}

main().catch(e => { console.error(e); process.exit(1) })
