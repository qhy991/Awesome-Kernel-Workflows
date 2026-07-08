#!/usr/bin/env node
// scripts/patch-sol-execbench-eval.js
// Inlines the sol-execbench-solution substrate (SOL_SOLUTION_CONTRACT +
// __solExecbenchEvalPlan) from _substrate/embedded/sol_execbench_eval.js into each
// opted-in workflow. Mirrors scripts/patch-embedded-eval.js exactly.
//   Run:      node scripts/patch-sol-execbench-eval.js
//   Dry run:  node scripts/patch-sol-execbench-eval.js --check
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const SELF_DIR = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.dirname(SELF_DIR)

// Opt-in list. Keep in sync with each workflow manifest's routing.integration_patterns.
const SOL_WORKFLOWS = [
  ['CUDAAgent', 'cuda-agent-kernel-optimization.js'],
  ['ARGUS', 'argus-kernel-optimization.js'],
  ['Generalist', 'generalist-kernel-optimization.js'],
]

const BEGIN = '// --- BEGIN sol-execbench-eval substrate (auto-inlined by scripts/patch-sol-execbench-eval.js) ---'
const END = '// --- END sol-execbench-eval substrate ---'
const INLINE_BEGIN = '// >>> SOL_INLINE_BEGIN'
const INLINE_END = '// <<< SOL_INLINE_END >>>'

async function extractInlineRegion() {
  const srcPath = path.join(REPO_ROOT, '_substrate', 'embedded', 'sol_execbench_eval.js')
  const src = await fs.readFile(srcPath, 'utf8')
  const a = src.indexOf(INLINE_BEGIN)
  const b = src.indexOf(INLINE_END)
  if (a < 0 || b < 0 || b < a) throw new Error(`sol_execbench_eval.js inline sentinels not found/ordered (${a}, ${b})`)
  const afterBeginLine = src.indexOf('\n', a) + 1
  return src.slice(afterBeginLine, b).trimEnd()
}

function buildBlock(inlineBody) {
  return ['', BEGIN, inlineBody, END, ''].join('\n')
}

async function patchOne(dir, file, block, check) {
  const p = path.join(REPO_ROOT, dir, file)
  let src
  try { src = await fs.readFile(p, 'utf8') } catch { return { file, status: 'missing' } }
  const already = src.indexOf(BEGIN)
  let next
  if (already >= 0) {
    const endIdx = src.indexOf(END, already)
    if (endIdx < 0) throw new Error(`${file}: BEGIN without END`)
    const after = endIdx + END.length
    next = src.slice(0, already).trimEnd() + '\n' + block.slice(1) + src.slice(after).replace(/^\n/, '')
    if (next === src) return { file, status: 'unchanged' }
  } else {
    const metaMatch = src.match(/export const meta = \{[\s\S]*?\n\}/)
    if (!metaMatch) throw new Error(`${file}: no 'export const meta = {...}' to anchor after`)
    const insertAt = metaMatch.index + metaMatch[0].length
    next = src.slice(0, insertAt) + '\n' + block + src.slice(insertAt)
  }
  if (check) return { file, status: already >= 0 ? 'would-update' : 'would-insert' }
  await fs.writeFile(p, next)
  return { file, status: already >= 0 ? 'updated' : 'inserted' }
}

async function main() {
  const check = process.argv.includes('--check')
  const body = await extractInlineRegion()
  const block = buildBlock(body)
  for (const [dir, file] of SOL_WORKFLOWS) {
    const r = await patchOne(dir, file, block, check)
    console.log(`${r.status.padEnd(14)} ${dir}/${r.file}`)
  }
}
main().catch(e => { console.error(e.message); process.exit(1) })
