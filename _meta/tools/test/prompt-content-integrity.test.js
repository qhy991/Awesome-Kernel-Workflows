'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ROOT = path.resolve(__dirname, '../../..')
const {capturePrompts} = require('../print-workflow-prompts.js')

test('KSearch forwards task requirements and complete parent source beyond old character cuts', async () => {
  const fixtures = path.join(ROOT, '_meta/tools/fixtures')
  const args = JSON.parse(fs.readFileSync(path.join(fixtures, 'ksearch-args.json')))
  const returns = JSON.parse(fs.readFileSync(path.join(fixtures, 'ksearch-agent-returns.json')))
  const requirement = 'IKET_REQUIRED_AFTER_CORRECT_KERNEL__TASK_TAIL'
  const sourceTail = 'COMPLETE_PARENT_SOURCE_TAIL'
  returns['read-spec'].spec_text = 'Task details '.repeat(1000) + requirement
  returns['select-0'].parent_solution_code = '# full source\n'.repeat(1200) + sourceTail
  const calls = await capturePrompts({workflowPath:path.join(ROOT, 'KSearch/ksearch-kernel-optimization.js'),
    args:{...args, problem_path:'/task/task.md', iterations:1, attempts_per_cycle:1, seed_candidates:1}, agentReturns:returns})
  const generate = calls.find(c => c.label === 'gen-0-0')
  assert.ok(generate, 'real workflow must reach Generate')
  assert.ok(generate.prompt.includes(requirement), 'mandatory task suffix was lost')
  assert.ok(generate.prompt.includes(sourceTail), 'parent source was cut')
  assert.ok(generate.prompt.includes('Read the complete original task at /task/task.md'))
})

test('typed prior evidence remains complete parseable JSON beyond 4000 characters', () => {
  const source = fs.readFileSync(path.join(ROOT, '_meta/scaffolding/typed-args.js'), 'utf8')
  const evidence = {details:'evidence '.repeat(1200), required_tail:'KEEP_ME'}
  const sandbox = {args:{attempt_evidence:evidence}}
  vm.runInNewContext(source + '\nresult = __attemptBlock()', sandbox)
  const payload = sandbox.result.match(/```json\n([\s\S]*?)\n```/)[1]
  assert.deepEqual(JSON.parse(payload), evidence)
})

test('fresh activation task context does not depend on a summarized specification', () => {
  const source = fs.readFileSync(path.join(ROOT, '_meta/scaffolding/typed-args.js'), 'utf8')
  const task = 'Task '.repeat(1500) + 'MANDATORY_PROFILER_AT_END'
  const sandbox = {args:{problem_path:'/inputs/task.md', problem_definition:task}}
  vm.runInNewContext(source + '\nresult = __taskContractBlock()', sandbox)
  assert.ok(sandbox.result.includes(task))
  assert.ok(sandbox.result.includes('/inputs/task.md'))
  assert.ok(sandbox.result.includes('required skills and profiling instructions'))
})
