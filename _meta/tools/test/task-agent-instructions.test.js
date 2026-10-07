'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm')
const root=path.resolve(__dirname,'../../..'),text=fs.readFileSync(path.join(root,'_meta/scaffolding/typed-args.js'),'utf8')
test('task-mode contract reads canonical agent and selected skill policy without changing legacy prompts',()=>{
 const legacy=vm.runInNewContext(text+';__taskContractBlock()', {args:{problem_path:'/task.md'}})
 const kernel=vm.runInNewContext(text+';__taskContractBlock()', {args:{problem_path:'/task.md',task_result_command:'trusted {kernel_path} {result_path}'}})
 assert.doesNotMatch(legacy,/GPU Infra/)
 assert.match(kernel,/agent.md or AGENTS.md and all selected task skills/)
 assert.match(kernel,/per-workload objective and GPU Infra broker requirement/)
 for(const file of ['CUDAAgent/cuda-agent-kernel-optimization.js','AccelOpt/accelopt-kernel-optimization.js']){
  const script=fs.readFileSync(path.join(root,file),'utf8');assert.doesNotMatch(script,/Read all3 task skills/);assert.match(script,/Read all selected task skills/)
 }
})
