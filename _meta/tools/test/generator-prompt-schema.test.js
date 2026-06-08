'use strict'
// Generator-prompt-schema regression test (P5a Task 5).
//
// Pins the agent `schema:` outputs of generate-workflow.js so future prompt edits
// (P5a Tasks 3+4 + any subsequent generator work) cannot accidentally drift the
// structured-output keys downstream consumers rely on.
//
// Per spec §9.2 "the generator is not a substitution engine": this test does NOT
// assert prompt body content — that drifts freely with backend-axis prompt-text
// edits. It pins the SCHEMA SHAPE (property keys + required) only.

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const { capturePrompts } = require(path.resolve(__dirname, '..', 'print-workflow-prompts.js'))

const ROOT = path.resolve(__dirname, '..', '..', '..')
const WORKFLOW = path.join(ROOT, '_meta', 'tools', 'generate-workflow.js')
const FIX_DIR = path.join(ROOT, '_meta', 'tools', 'fixtures')
const ARGS = JSON.parse(fs.readFileSync(path.join(FIX_DIR, 'generate-workflow-args.json'), 'utf8'))
const RETURNS = JSON.parse(fs.readFileSync(path.join(FIX_DIR, 'generate-workflow-agent-returns.json'), 'utf8'))

// PINNED SCHEMAS — the exact `properties` keyset + `required` keyset for each
// generator agent. Drift here is a P5a contract violation and MUST update this
// test + the master plan in the same commit (per Task 5 / spec §9.2).
const PINNED_SCHEMAS = {
  'model-args': {
    properties: ['required_args', 'optional_args'],
    required: ['required_args', 'optional_args'],
    nested: {
      required_args_items: {
        properties: ['name', 'type', 'description', 'example'],
        required: ['name', 'type', 'description', 'example'],
      },
      optional_args_items: {
        properties: ['name', 'type', 'default_value', 'description', 'example'],
        required: ['name', 'type', 'default_value', 'description'],
      },
    },
  },
  'assemble-manifest': {
    properties: ['manifest_yaml', 'method_name', 'workflow_name'],
    required: ['manifest_yaml', 'method_name', 'workflow_name'],
  },
  'generate-workflow': {
    properties: ['workflow_code', 'filename', 'directory'],
    required: ['workflow_code', 'filename'],
  },
}

function keys(obj) {
  return obj ? Object.keys(obj).sort() : []
}

test('generate-workflow.js: every captured call has a schema field', async () => {
  const calls = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  assert.ok(calls.length > 0, 'expected at least one captured call')
  for (const c of calls) {
    assert.ok('schema' in c, `call ${c.label}: missing 'schema' key (run-workflow.js must capture opts.schema)`)
  }
})

test('generate-workflow.js: pinned agent schemas have stable property keys + required (P5a regression)', async () => {
  const calls = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const byLabel = Object.fromEntries(calls.map(c => [c.label, c]))

  for (const [label, pinned] of Object.entries(PINNED_SCHEMAS)) {
    const call = byLabel[label]
    assert.ok(call, `expected agent labeled '${label}' to be captured`)
    const schema = call.schema
    assert.ok(schema && typeof schema === 'object', `${label}: schema must be an object`)
    assert.deepStrictEqual(keys(schema.properties), pinned.properties.slice().sort(),
      `${label}: schema.properties keys drifted from pinned set`)
    assert.deepStrictEqual((schema.required || []).slice().sort(), pinned.required.slice().sort(),
      `${label}: schema.required drifted from pinned set`)
  }
})

test('generate-workflow.js: model-args nested item schemas (required_args/optional_args) are stable', async () => {
  const calls = await capturePrompts({ workflowPath: WORKFLOW, args: ARGS, agentReturns: RETURNS })
  const ma = calls.find(c => c.label === 'model-args')
  assert.ok(ma, 'model-args agent call must be captured')
  const nested = PINNED_SCHEMAS['model-args'].nested
  const req = ma.schema.properties.required_args.items
  assert.deepStrictEqual(keys(req.properties), nested.required_args_items.properties.slice().sort(),
    'model-args.required_args.items.properties drifted')
  assert.deepStrictEqual((req.required || []).slice().sort(), nested.required_args_items.required.slice().sort(),
    'model-args.required_args.items.required drifted')
  const opt = ma.schema.properties.optional_args.items
  assert.deepStrictEqual(keys(opt.properties), nested.optional_args_items.properties.slice().sort(),
    'model-args.optional_args.items.properties drifted')
  assert.deepStrictEqual((opt.required || []).slice().sort(), nested.optional_args_items.required.slice().sort(),
    'model-args.optional_args.items.required drifted')
})
