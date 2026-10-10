'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const os = require('node:os')
const run = require('../lib/run-workflow.js')
const root = path.resolve(__dirname, '../../..')
const methods = {
  CUDAAgent:'CUDAAgent/cuda-agent-kernel-optimization.js',
  AccelOpt:'AccelOpt/accelopt-kernel-optimization.js',
}
for (const [method,file] of Object.entries(methods)) {
  for (const runtime of ['native','host']) {
    for (const mode of ['date-typo','duplicate-echo','missing-file','unbound-result']) {
      test(`${method} ${runtime} initial slots: ${mode}`,async()=>{
        const exp = fs.mkdtempSync(path.join(os.tmpdir(),'akw-initial-slots-'))
        try {
          fs.mkdirSync(path.join(exp,'generated'))
          for (let i=0;i<3;i++) if (!(mode==='missing-file' && i===1))
            fs.writeFileSync(path.join(exp,'generated',`initial_${i}.py`),`fixture source ${i}`)
          const measured=(p,lat,extra={})=>({contract_version:'kersor-task-result-v1',test_result_path:p,
            compiled:true,correct:true,full_workload_set:true,measurement_valid:true,n_pass:17,n_total:17,
            candidate_latency_aggregate_ms:lat,speedup_vs_reference:.06/lat,candidate_path:p+'.artifact/candidate.py',
            source_binding:{verified:true,source_sha256:'a'.repeat(64)},...extra})
          const agents={'generate-initial-kernel':{initial_candidates:[0,1,2].map(i=>({variant_path:mode==='duplicate-echo'?'/foreign/initial_0.py':`/wrong/cohort-20260907/initial_${i}.py`}))},
            'profile-baseline':{bottlenecks:[],optimization_strategy:'fixture'},
            'read-baseline':{kernel_code:'display',op_type:'gemm',key_functions:['run']},
            'ncu-baseline':{ncu_available:false,profile_summary:'unavailable',bottleneck_diagnosis:'unknown'},
            'plan-0-0':{title:'p',plan:'fixture'},'impl-0-p-v0':{code:'display',variant_path:`${exp}/iter_0_plan_0_sample_0.py`}}
          const evaluations={},seen=[]
          for(let i=0;i<3;i++) {
            const p=`${exp}/generated/initial_${i}`
            const response=()=>{
              seen.push(i)
              if(!fs.existsSync(p+'.py'))throw new Error('declared candidate file absent before device admission')
              return measured(p+'.json',[.03,.02,.04][i],mode==='unbound-result'?{source_binding:null}:{})
            }
            Object.defineProperty(agents,`initial-${i}`,{get(){return {test_result_path:p+'.json',test_result_json:response()}}})
            evaluations[`initial-${i}`]=request=>{
              assert.ok(request.argv.join(' ').includes(p+'.py'))
              assert.ok(!request.argv.join(' ').includes('/wrong/'))
              return {stage:'complete',exit_code:0,stdout_json:response()}
            }
          }
          for(let i=0;i<15;i++) {
            agents[`impl-${i}`]={kernel_code:'display',variant_path:`${exp}/task_attempt_${i}.py`}
            agents[`task-verify-${i}`]={test_result_path:`${exp}/task_attempt_${i}.json`,test_result_json:measured(`${exp}/task_attempt_${i}.json`,.01)}
            evaluations[`task-verify-${i}`]={stage:'complete',exit_code:0,stdout_json:measured(`${exp}/task_attempt_${i}.json`,.01)}
          }
          const variant=`${exp}/iter_0_plan_0_sample_0.task.json`
          agents['task-eval-iter_0_plan_0_sample_0']={test_result_path:variant,test_result_json:measured(variant,.01)}
          evaluations['task-eval-iter_0_plan_0_sample_0']={stage:'complete',exit_code:0,stdout_json:measured(variant,.01)}
          const args={problem_path:'/fixture/task.md',exp_dir:exp,language:'cute-dsl',task_result_command:'trusted --candidate {kernel_path} --result {result_path}',task_workload_count:17,seed_candidates:3,iterations:1,breadth:1,samples_per_plan:1}
          const execute=()=>run(fs.readFileSync(path.join(root,file),'utf8'),args,agents,runtime==='native'?null:evaluations)
          if(mode==='missing-file' || mode==='unbound-result') {
            await assert.rejects(execute(),e=>e.code==='TASK_RESULT_HOLD')
            assert.deepEqual(seen,mode==='missing-file'?[0,1]:[0])
          } else {
            const {calls,result}=await execute()
            assert.deepEqual(seen,[0,1,2])
            assert.ok(result.generated_kernel_path)
            if(runtime==='native')for(const call of calls.filter(c=>/^initial-\d+$/.test(c.label))) {
              assert.ok(call.prompt.includes(`${exp}/generated/initial_${call.label.split('-')[1]}.py`))
              assert.ok(!call.prompt.includes('/wrong/'))
              assert.ok(!call.prompt.includes('/foreign/'))
            }
          }
        } finally {fs.rmSync(exp,{recursive:true,force:true})}
      })
    }
  }
}
