'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path')
const text=fs.readFileSync(path.resolve(__dirname,'../../scaffolding/typed-args.js'),'utf8')
test('task measurement ownership reaches each activation with the caller command',()=>{
 const f=vm.runInNewContext(text+';__taskContractBlock',{args:{problem_path:'/task.md',task_result_command:'trusted task --candidate {kernel_path} --result {result_path}'}})
 const p=f();assert.match(p,/trusted task/);assert.match(p,/Never pre-create/);assert.match(p,/Never run selftest.py or verify.py directly/)
})
test('non-task workflows receive no invented measurement command',()=>{
 const f=vm.runInNewContext(text+';__taskContractBlock',{args:{problem_path:'/task.md'}})
 assert.ok(!f().includes('Task measurement ownership'))
})
