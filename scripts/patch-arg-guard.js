#!/usr/bin/env node
// scripts/patch-arg-guard.js
//
// Inserts an inlined arg_guard + unwrap call into every workflow .js in the
// catalog (one per workflow subdirectory + templates). Idempotent: re-running
// it is a no-op.
//
// Run from repo root:
//   node scripts/patch-arg-guard.js
//
// What it inserts, right after `export const meta = {...}`:
//
//   function __unwrapArgs(rawArgs) { ... }
//   args = __unwrapArgs(args)
//
// Why inline instead of import: the Workflow runtime parses workflow files as
// bare scripts, not ES modules, so static imports are rejected.

import { promises as fs } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const SELF_DIR = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.dirname(SELF_DIR)

// Workflow .js files live in <repo>/<workflow>/<workflow>-*.js. Templates live in
// _templates/ and _meta/templates/. We exclude scripts/, _tools/, _substrate/,
// _meta/* except _meta/templates, etc.
const TOP_LEVEL_BLOCKLIST = new Set([
  '_manifests', '_meta', '_substrate', '_templates', '_tools',
  'scripts', 'docs', 'badges', 'node_modules', '.git',
])

async function findWorkflowJs() {
  const entries = await fs.readdir(REPO_ROOT, { withFileTypes: true })
  const dirs = entries
    .filter(e => e.isDirectory() && !TOP_LEVEL_BLOCKLIST.has(e.name))
    .map(e => path.join(REPO_ROOT, e.name))

  const out = []
  for (const dir of dirs) {
    const files = await fs.readdir(dir)
    for (const f of files) {
      if (f.endsWith('.js')) out.push(path.join(dir, f))
    }
  }
  // Also patch templates so newly-generated workflows inherit the guard.
  for (const tplDir of [
    path.join(REPO_ROOT, '_templates'),
    path.join(REPO_ROOT, '_meta', 'templates'),
  ]) {
    try {
      const tpls = await fs.readdir(tplDir)
      for (const f of tpls) {
        if (f.endsWith('.js')) out.push(path.join(tplDir, f))
      }
    } catch (_) { /* ok */ }
  }
  return out
}

function buildPatch(filePath) {
  return [
    '',
    '// --- BEGIN inlined arg_guard (Workflow runtime parses scripts as bare scripts,',
    '//                              not ES modules; static imports are rejected) ---',
    'function __unwrapArgs(rawArgs) {',
    '  if (rawArgs == null) return {}',
    "  if (typeof rawArgs === 'object' && !Array.isArray(rawArgs)) return rawArgs",
    "  if (typeof rawArgs === 'string') {",
    '    const trimmed = rawArgs.trim()',
    "    if (trimmed === '') return {}",
    "    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {",
    '      try {',
    '        const parsed = JSON.parse(trimmed)',
    "        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed",
    "        throw new Error('arg_guard: parsed JSON value is not a plain object')",
    "      } catch (e) { throw new Error(`arg_guard: invalid JSON args: ${e.message}`) }",
    '    }',
    '    const out = {}',
    '    const re = /(\\w[\\w.-]*)=("(?:\\\\\\\\\\"|[^"])*"|\\\'(?:\\\\\\\\\\\'|[^\\\'])*\\\'|\\S+)/g',
    '    let m',
    '    while ((m = re.exec(trimmed)) !== null) {',
    '      let v = m[2]',
    '      if ((v.startsWith(\'"\') && v.endsWith(\'"\')) || (v.startsWith("\'") && v.endsWith("\'"))) {',
    '        v = v.slice(1, -1)',
    '      }',
    '      out[m[1]] = v',
    '    }',
    '    if (Object.keys(out).length === 0) {',
    '      throw new Error(`arg_guard: workflow args is a non-empty string but contains no key=value pairs and is not JSON. First 160 chars: ${trimmed.slice(0, 160)}`)',
    '    }',
    '    return out',
    '  }',
    "  throw new Error(`arg_guard: workflow args has unexpected type: ${typeof rawArgs}`)",
    '}',
    '// eslint-disable-next-line no-global-assign',
    'args = __unwrapArgs(typeof args === \'undefined\' ? undefined : args)',
    '// --- END inlined arg_guard ---',
    '',
  ].join('\n')
}

function findMetaEnd(src) {
  // Find `export const meta = {` and return the index immediately after its
  // matching closing brace + optional trailing `}` semicolon or newline.
  const start = src.indexOf('export const meta')
  if (start < 0) return -1
  const braceOpen = src.indexOf('{', start)
  if (braceOpen < 0) return -1
  let depth = 0
  for (let i = braceOpen; i < src.length; ++i) {
    const c = src[i]
    if (c === '{') depth++
    else if (c === '}') {
      depth--
      if (depth === 0) {
        // skip optional trailing chars on the same line
        let j = i + 1
        while (j < src.length && (src[j] === ';' || src[j] === ' ' || src[j] === '\t')) j++
        if (src[j] === '\n') j++
        return j
      }
    }
  }
  return -1
}

async function patchOne(filePath) {
  const src = await fs.readFile(filePath, 'utf8')
  if (src.includes('args = __unwrapArgs')) {
    return { filePath, status: 'already_patched' }
  }
  if (!src.includes('export const meta')) {
    return { filePath, status: 'skipped_no_meta' }
  }
  let insertAt = findMetaEnd(src)
  if (insertAt < 0) {
    return { filePath, status: 'skipped_meta_parse_failed' }
  }
  const workflowNameMatch = src.match(/\bconst\s+WORKFLOW_NAME\s*=\s*['"][^'"]+['"]\s*\n/)
  if (workflowNameMatch && workflowNameMatch.index !== undefined) {
    insertAt = workflowNameMatch.index + workflowNameMatch[0].length
  }
  const patch = buildPatch(filePath)
  const next = src.slice(0, insertAt) + patch + src.slice(insertAt)
  await fs.writeFile(filePath, next, 'utf8')
  return { filePath, status: 'patched' }
}

async function main() {
  const files = await findWorkflowJs()
  let patched = 0, already = 0, skipped = 0
  for (const f of files) {
    const r = await patchOne(f)
    const rel = path.relative(REPO_ROOT, r.filePath)
    console.log(`${r.status.padEnd(28)} ${rel}`)
    if (r.status === 'patched') patched++
    else if (r.status === 'already_patched') already++
    else skipped++
  }
  console.log(`\nSummary: patched=${patched} already=${already} skipped=${skipped} total=${files.length}`)
}

main().catch(e => { console.error(e); process.exit(1) })
