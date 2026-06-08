#!/usr/bin/env node
'use strict'
// print-workflow-prompts.js — CLI + capturePrompts({workflowPath, args, agentReturns})
//
// Reads a Claude Workflow .js, runs it under the vm sandbox in run-workflow.js with stub
// runtime globals, and prints/writes the captured agent() prompts as stable-key JSON.
//
// LOAD-BEARING: agentReturns (label→value map) is consulted FIRST inside the sandbox; the
// schema generator is the fallback. This is what unlocks AccelOpt's full loop deterministically.

const fs = require('node:fs')
const path = require('node:path')
const runWorkflow = require('./lib/run-workflow.js')

async function capturePrompts({ workflowPath, args, agentReturns }) {
  const source = fs.readFileSync(workflowPath, 'utf8')
  const { calls } = await runWorkflow(source, args || {}, agentReturns || {})
  return calls
}

function stableStringify(arr) {
  const sortedArr = arr.map((r) => {
    const sorted = {}
    for (const k of Object.keys(r).sort()) sorted[k] = r[k]
    return sorted
  })
  return JSON.stringify(sortedArr, null, 2)
}

function parseArgs(argv) {
  const out = {}
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--workflow') out.workflow = argv[++i]
    else if (a === '--args') out.args = argv[++i]
    else if (a === '--agent-returns') out.agentReturns = argv[++i]
    else if (a === '--out') out.out = argv[++i]
    else if (!out.workflow && !a.startsWith('--')) out.workflow = a
    else if (!out.args && !a.startsWith('--')) out.args = a
    else if (!out.agentReturns && !a.startsWith('--')) out.agentReturns = a
  }
  return out
}

async function main() {
  const argv = process.argv.slice(2)
  const parsed = parseArgs(argv)
  if (!parsed.workflow) {
    process.stderr.write(
      'usage: node print-workflow-prompts.js --workflow <wf.js> [--args <args.json>] [--agent-returns <returns.json>] [--out <out.json>]\n'
    )
    process.exit(2)
  }
  const workflowPath = path.resolve(parsed.workflow)
  let args = {}
  let agentReturns = {}
  if (parsed.args) {
    const raw = fs.readFileSync(parsed.args, 'utf8')
    args = JSON.parse(raw)
  }
  if (parsed.agentReturns) {
    const raw = fs.readFileSync(parsed.agentReturns, 'utf8')
    agentReturns = JSON.parse(raw)
  }
  const calls = await capturePrompts({ workflowPath, args, agentReturns })
  const json = stableStringify(calls)
  if (parsed.out) {
    fs.writeFileSync(parsed.out, json)
  } else {
    process.stdout.write(json + '\n')
  }
}

module.exports = { capturePrompts }

if (require.main === module) {
  main().catch((err) => {
    process.stderr.write((err && err.stack ? err.stack : String(err)) + '\n')
    process.exit(1)
  })
}
