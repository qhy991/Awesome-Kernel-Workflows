#!/usr/bin/env node
// patch-anti-cheat-flags.js — codemod that fixes the #42 anti_cheat.py flag
// drift across all workflows: `--kernel` -> `--source`, `--result` -> `--metrics`
// (the script's argparse REQUIRES --source/--metrics; --kernel/--result was
// rejected and the anti_cheat check silently failed for every candidate).
//
// Scope: only lines containing `anti_cheat.py --kernel` are touched (the
// `--result` -> `--metrics` rename is line-scoped so it can't hit unrelated
// `--result` tokens). Idempotent: lines already using `--source` are skipped.
//
// The canonical flag schema lives in _meta/scaffolding/substrate-invocation.js
// (substrateAntiCheat); NEW anti_cheat calls should use that builder instead of
// hand-writing flags. This codemod fixes the EXISTING 17 hand-written calls.
//
// Validation is the caller's job: run `node --check` on every output file.

const fs = require('fs')
const path = require('path')

const REPO = path.resolve(__dirname, '..')

function workflowFiles() {
  return fs.readdirSync(REPO).filter((d) => {
    const dir = path.join(REPO, d)
    return fs.statSync(dir).isDirectory() && fs.existsSync(path.join(dir, 'manifest.yaml'))
  }).map((d) => fs.readdirSync(path.join(REPO, d)).filter((f) => f.endsWith('.js')).map((f) => path.join(REPO, d, f))).flat()
}

function fixFile(f) {
  const src = fs.readFileSync(f, 'utf8')
  let fixed = 0
  const out = src.split('\n').map((line) => {
    if (line.includes('anti_cheat.py --kernel')) {
      fixed++
      return line.replace(/anti_cheat\.py --kernel/, 'anti_cheat.py --source').replace(/--result /, '--metrics ')
    }
    return line
  })
  if (fixed) fs.writeFileSync(f, out.join('\n'))
  return fixed
}

function main() {
  const files = process.argv.slice(2).filter((a) => !a.startsWith('-')).map((p) => path.resolve(p)).filter((f) => fs.existsSync(f))
  const targets = files.length ? files : workflowFiles()
  let total = 0
  for (const f of targets) {
    const n = fixFile(f)
    if (n) console.log(`  ${path.relative(REPO, f)}: fixed ${n} anti_cheat call(s)`)
    total += n
  }
  console.log(`total: ${total} anti_cheat call(s) fixed`)
}

main()
