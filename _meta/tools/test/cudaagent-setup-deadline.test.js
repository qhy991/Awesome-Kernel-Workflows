'use strict'
const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const source = fs.readFileSync(path.resolve(__dirname,'../../../CUDAAgent/cuda-agent-kernel-optimization.js'),'utf8')
for (const outcome of ['value','error']) {
  test(`initial generation deadline waits for settlement and blocks downstream work: ${outcome}`, async () => {
    let expire, resolve, reject, finished=false
    const child=new Promise((a,b)=>{resolve=a;reject=b}), calls=[]
    const fault=Object.assign(new Error('provider refused'),{retryable:false})
    const sandbox={args:{problem_path:'/fixture/task.md',exp_dir:'/fixture/exp',language:'cute-dsl',
      task_result_command:'trusted --candidate {kernel_path} --result {result_path}',task_workload_count:19,seed_candidates:3,turn_timeout_min:1},
      phase(){},log(){},budget(){},parallel(){throw Error('unexpected parallel')},pipeline(){throw Error('unexpected pipeline')},
      setTimeout(fn){expire=fn;return 1},clearTimeout(){},
      agent(prompt,options){calls.push(options.label);assert.equal(options.label,'generate-initial-kernel');return child}}
    const pending=vm.runInNewContext('(async function(){'+source.replace(/^export\s+/,'')+'\n})()',sandbox)
    pending.then(()=>{finished=true},()=>{finished=true})
    await Promise.resolve()
    assert.equal(typeof expire,'function','Setup must own a turn deadline')
    expire();await Promise.resolve();assert.equal(finished,false)
    if(outcome==='error')reject(fault)
    else resolve({initial_candidates:[0,1,2].map(i=>({variant_path:`/fixture/exp/generated/initial_${i}.py`}))})
    await assert.rejects(pending,e=>outcome==='error'?e===fault:e.code==='KERSOR_TURN_DRAINED'&&e.invocation_settled===true)
    assert.deepEqual(calls,['generate-initial-kernel'])
  })
}
