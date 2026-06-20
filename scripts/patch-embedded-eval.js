#!/usr/bin/env node
// scripts/patch-embedded-eval.js
//
// Inlines the embedded-dispatch substrate (EMBEDDING_CONTRACT + __embeddedEvalPlan)
// from _substrate/embedded/embedded_eval.js into each workflow that opts into
// embedded mode. Mirrors scripts/patch-arg-guard.js, with two differences:
//
//   1. Opt-in, not blanket: only workflows in EMBEDDED_WORKFLOWS (below) are
//      patched. This AKW-side opt-in list must match each workflow manifest's
//      `routing.integration_patterns`.
//   2. Update-in-place: the inlined block is delimited by markers, so re-running
//      after editing the substrate source REPLACES the block instead of skipping.
//      Idempotent: identical source => no change.
//
// Run from repo root:  node scripts/patch-embedded-eval.js
// Dry run:             node scripts/patch-embedded-eval.js --check
//
// The block is injected right after `export const meta = {...}` (meta must stay a
// pure literal for the Workflow tool's static extraction).

import { promises as fs } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const SELF_DIR = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.dirname(SELF_DIR)

// Workflows that support embedded-dispatch evaluation. Keep in sync with the
// `integration_patterns` entries in the matching workflow manifest `routing:` block.
const EMBEDDED_WORKFLOWS = [
  'GPUForecasters',
  'FACT',
  'CUDAAgent',
  'ARGUS',
  'Generalist',
]

const BEGIN = '// --- BEGIN embedded-eval substrate (auto-inlined by scripts/patch-embedded-eval.js) ---'
const END = '// --- END embedded-eval substrate ---'
const INLINE_BEGIN = '// >>> EMBEDDED_INLINE_BEGIN'
const INLINE_END = '// <<< EMBEDDED_INLINE_END >>>'

async function extractInlineRegion() {
  const srcPath = path.join(REPO_ROOT, '_substrate', 'embedded', 'embedded_eval.js')
  const src = await fs.readFile(srcPath, 'utf8')
  const a = src.indexOf(INLINE_BEGIN)
  const b = src.indexOf(INLINE_END)
  if (a < 0 || b < 0 || b < a) {
    throw new Error(`embedded_eval.js inline sentinels not found/ordered (${a}, ${b})`)
  }
  // Drop the sentinel lines themselves, keep the body between them.
  const afterBeginLine = src.indexOf('\n', a) + 1
  const body = src.slice(afterBeginLine, b).trimEnd()
  return body
}

function buildBlock(inlineBody) {
  return ['', BEGIN, inlineBody, END, ''].join('\n')
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
      depth--
      if (depth === 0) {
        let j = i + 1
        while (j < src.length && (src[j] === ';' || src[j] === ' ' || src[j] === '\t')) j++
        if (src[j] === '\n') j++
        return j
      }
    }
  }
  return -1
}

function spliceBlock(src, block) {
  const bi = src.indexOf(BEGIN)
  if (bi >= 0) {
    // Replace the existing block (BEGIN..END inclusive, plus trailing newline).
    let ei = src.indexOf(END, bi)
    if (ei < 0) throw new Error('found BEGIN marker without matching END')
    ei += END.length
    if (src[ei] === '\n') ei += 1
    // Trim the leading blank line we originally inserted before BEGIN.
    let bStart = bi
    if (bStart > 0 && src[bStart - 1] === '\n') bStart -= 1
    const next = src.slice(0, bStart) + block + src.slice(ei)
    return next === src ? null : next
  }
  const insertAt = findMetaEnd(src)
  if (insertAt < 0) return null
  return src.slice(0, insertAt) + block + src.slice(insertAt)
}

async function workflowJsFor(dirName) {
  const dir = path.join(REPO_ROOT, dirName)
  let files
  try { files = await fs.readdir(dir) } catch { return null }
  const js = files.find(f => f.endsWith('.js'))
  return js ? path.join(dir, js) : null
}

async function main() {
  const check = process.argv.includes('--check')
  const inlineBody = await extractInlineRegion()
  const block = buildBlock(inlineBody)

  let changed = 0, ok = 0, skipped = 0
  for (const w of EMBEDDED_WORKFLOWS) {
    const file = await workflowJsFor(w)
    if (!file) { console.log(`skipped_no_js            ${w}`); skipped++; continue }
    const src = await fs.readFile(file, 'utf8')
    if (!src.includes('export const meta')) { console.log(`skipped_no_meta          ${w}`); skipped++; continue }
    const next = spliceBlock(src, block)
    const rel = path.relative(REPO_ROOT, file)
    if (next == null) { console.log(`up_to_date               ${rel}`); ok++; continue }
    if (check) { console.log(`would_update             ${rel}`); changed++; continue }
    await fs.writeFile(file, next, 'utf8')
    console.log(`updated                  ${rel}`); changed++
  }
  console.log(`\nSummary: ${check ? 'would_update' : 'updated'}=${changed} up_to_date=${ok} skipped=${skipped} total=${EMBEDDED_WORKFLOWS.length}`)
  if (check && changed > 0) process.exit(2)
}

main().catch(e => { console.error(e); process.exit(1) })
