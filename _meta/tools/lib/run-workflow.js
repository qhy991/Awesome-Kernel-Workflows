'use strict'
// run-workflow.js — vm sandbox + stub runtime (textual async-wrap)
// Loads a Claude Workflow .js as a STRING, strips the leading `export`, textually wraps the
// body in `(async function(){ ... })()`, runs it under node:vm with stubbed runtime globals
// (agent/phase/parallel/pipeline/log/budget) + an injected `args`, and captures every rendered
// agent() call.
//
// LOAD-BEARING DESIGN CHOICE — textual async-wrap:
//   vm rejects the file's genuine top-level `await` (e.g. AccelOpt L292) AND top-level `return`
//   (AccelOpt L856) unless the body is textually wrapped in an async function BEFORE compilation.
//   The wrapper is built as a string; no AST manipulation.

const vm = require('node:vm')
const schemaStub = require('./schema-stub.js')

/**
 * Run a workflow source string under the vm sandbox.
 *
 * @param {string} source       — raw workflow source (may start with `export `)
 * @param {object} args         — injected as the `args` global in the sandbox
 * @param {object} agentReturns — label→value map; consulted BEFORE schemaStub fallback
 * @returns {Promise<{meta: any, calls: Array<{seq,label,phase,prompt,schema}>}>}
 */
async function runWorkflow(source, args, agentReturns) {
  // Strip the lone leading `export ` token so `export const meta = ...` becomes
  // `const meta = ...` and the body is no longer an ES module.
  const strippedBody = source.replace(/^export\s+/, '')

  // Textual async-wrap: the body's own `return {...}` becomes the IIFE's resolved value,
  // and its top-level `await`s are now legal because they're inside an async function.
  const wrapped =
    '(async function(){\n' +
    strippedBody +
    '\n})()'

  // State shared by stub closures
  const calls = []
  let seq = 0
  let currentPhase = ''

  // Stub implementations
  function agentStub(prompt, opts) {
    const label = opts && opts.label !== undefined ? opts.label : undefined
    const phase = opts && opts.phase !== undefined ? opts.phase : currentPhase
    // Clone schema across the vm-realm boundary so deepStrictEqual on captured calls
    // does not trip on Object.prototype identity (schemas literal-allocated inside the
    // sandbox have a different Object than the host runner).
    const schema = opts && opts.schema !== undefined
      ? JSON.parse(JSON.stringify(opts.schema))
      : undefined
    calls.push({ seq: seq++, label, phase, prompt, schema })
    if (agentReturns && label !== undefined && Object.prototype.hasOwnProperty.call(agentReturns, label)) {
      return agentReturns[label]
    }
    return schemaStub(opts && opts.schema)
  }

  function phaseStub(title) {
    currentPhase = title
  }

  async function parallel(thunks) {
    const o = []
    for (const t of thunks) o.push(await t())
    return o
  }

  async function pipeline(items, fn) {
    const o = []
    for (const it of items) o.push(await fn(it))
    return o
  }

  function log() {}
  function budget() {}

  const sandbox = {
    args,
    agent: agentStub,
    phase: phaseStub,
    parallel,
    pipeline,
    log,
    budget,
    // Provide console so any debug logging in the workflow doesn't crash
    console,
    // Provide JSON so workflows can use JSON.parse / JSON.stringify
    JSON,
    // Provide Math for deterministic shuffle (sampleWithoutReplacement)
    Math,
    // Provide common globals that workflow code may reference
    undefined,
    // Provide Promise for any explicit Promise usage
    Promise,
  }
  vm.createContext(sandbox)

  // Run the wrapped source. The IIFE resolves to the body's `return {...}` value.
  // Do NOT catch/swallow rejections — let them propagate so callers can assert.rejects.
  const result = await vm.runInContext(wrapped, sandbox, { filename: 'workflow.js' })

  const meta = result && result.meta !== undefined ? result.meta : null
  return { meta, calls }
}

module.exports = runWorkflow
