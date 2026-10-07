'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),os=require('node:os')
const root=path.resolve(__dirname,'../../..'),{capturePrompts}=require('../print-workflow-prompts.js')
const names=[['stark','STARK/stark-kernel-optimization.js'],['adaexplore','AdaExplore/adaexplore-kernel-optimization.js'],['regrapht','ReGraphT/regrapht-kernel-optimization.js']]
for(const [name,file] of names)test(name+' actual method prompts keep complete source on disk and reference it downstream',async()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'method-context-'))
 try{
  const args=JSON.parse(fs.readFileSync(path.join(__dirname,'../fixtures',name+'-args.json')));args.exp_dir=dir
  const returns=JSON.parse(fs.readFileSync(path.join(__dirname,'../fixtures',name+'-agent-returns.json')))
  function inject(x){if(!x||typeof x!=='object')return;for(const k of Object.keys(x)){if(['kernel_code','candidate_code','code'].includes(k)&&typeof x[k]==='string')x[k]='SOURCE_SENTINEL_'+name+'\n'+'x'.repeat(50000);else inject(x[k])}}
  inject(returns)
  const calls=await capturePrompts({workflowPath:path.join(root,file),args,agentReturns:returns})
  const writers=calls.filter(c=>c.prompt.includes("'python3' '-c'")&&c.prompt.includes('KERSOR_WORKSPACE_SOURCE_JSON'))
  assert.ok(writers.length>0)
  for(const c of writers){const start=c.prompt.indexOf("'python3' '-c'");const end=c.prompt.indexOf('\nKERSOR_WORKSPACE_SOURCE_JSON',start)+'\nKERSOR_WORKSPACE_SOURCE_JSON'.length;const run=cp.spawnSync('/bin/sh',[],{input:c.prompt.slice(start,end),encoding:'utf8'});assert.equal(run.status,0,run.stderr);const receipt=JSON.parse(run.stdout);assert.ok(fs.readFileSync(receipt.source_path,'utf8').includes('SOURCE_SENTINEL_'))}
  const phases=name==='stark'?['Plan','Code','Debug','Report']:name==='adaexplore'?['Expand','Learn','Report']:['Select','UpdateGraph','Report']
  for(const c of calls.filter(c=>phases.includes(c.phase))){assert.ok(!c.prompt.includes('SOURCE_SENTINEL_'),c.label)}
  assert.ok(calls.some(c=>phases.includes(c.phase)&&c.prompt.includes(dir)))
 }finally{fs.rmSync(dir,{recursive:true,force:true})}
})
const run=require('../lib/run-workflow.js')
test('cuPilot revised winners are handed to following generations by their per-candidate path',async()=>{
 const args={kernel_path:'/fixture/original.cu',exp_dir:'/fixture/cupilot',epochs:1,generations_per_epoch:2};const returns={'integration-strategist':{method:'standalone'}}
 for(let g=0;g<2;g++){
  returns[`strategize-e0-g${g}`]={new_strategies:[{strategy:'tile'}]};returns[`translate-e0-g${g}-s0`]={kernel_code:'FULL_TRANSLATION'}
  returns[`revise-e0-g${g}-k0`]={kernel_code:'REVISED_SOURCE_SENTINEL'.repeat(20000),compiled:true,correct:true,speedup:2+g}
 }
 const out=await run(fs.readFileSync(path.join(root,'cuPilot/cupilot-kernel-optimization.js'),'utf8'),args,returns,null)
 const next=out.calls.find(c=>c.label==='translate-e0-g1-s0');assert.ok(next);assert.ok(!next.prompt.includes('REVISED_SOURCE_SENTINEL'));assert.ok(next.prompt.includes('/fixture/cupilot/revised/e0-g0-k0.cu'))
 assert.ok(out.calls.find(c=>c.label==='revise-e0-g0-k0').prompt.includes('Write the COMPLETE final revised kernel'))
})
test('XeForge carries only file references after complete source persistence',async()=>{
 const src=fs.readFileSync(path.join(root,'Xe-Forge/xe-forge-kernel-optimization.js'),'utf8');const returns={
  'Setup Xe-Forge':{xpu_available:true,target_backend:'triton',cover_cycles:2,kernel_spec:{operation:'gemm',shapes:'m,n',baseline_gflops:1},profiling_tools:[]},
  'Generate initial impl':{backend:'triton',kernel_code:'INITIAL_SOURCE',initial_strategy:'tile'},
  'Refine decision':{continue:true,reason:'test'},
 }
 for(let i=1;i<=2;i++){
  returns[`Analyze cycle ${i}`]={cycle:i,performance_gflops:2,bottleneck_type:'memory',optimization_opportunities:[]}
  returns[`Plan cycle ${i}`]={cycle:i,strategies:[{name:'tile'}],rationale:'test'}
  returns[`Optimize cycle ${i}`]={cycle:i,optimized_kernel_code:'XE_SOURCE_SENTINEL'.repeat(5000),changes_applied:[]}
  returns[`Verify cycle ${i}`]={cycle:i,correctness_passed:true,performance_gflops:3,performance_improvement:1,verification_passed:true}
 }
 const out=await run(src,{exp_dir:'/fixture/xe'},returns,null);const next=out.calls.find(c=>c.label==='Analyze cycle 2');assert.ok(next);assert.ok(!next.prompt.includes('XE_SOURCE_SENTINEL'));assert.ok(next.prompt.includes('cycle_0_optimized.kernel'))
})
test('KernelBlaster candidate materialization uses distinct paths before reuse',async()=>{
 const returns={'integration-strategist':{method:'standalone'},'read-baseline':{kernel_code:'ORIGINAL',op_type:'gemm'},'ncu-baseline':{elapsed_cycles:100,profile_summary:'test'}}
 for(const [key,tech] of [['vectorized_m','v'],['memory_coale','m']])returns['plan-0-0-'+key]={technique:tech,plan:'test',predicted_improvement:20}
 for(const t of ['v','m']){returns['impl-0-0-'+t]={code:'KERNEL_PAYLOAD_'+t,technique:t};returns['eval-0-0-'+t]={is_correct:true,is_compilable:true,elapsed_cycles:50,speedup:2,improvement_pct:50}}
 const out=await run(fs.readFileSync(path.join(root,'KernelBlaster/kernelblaster-kernel-optimization.js'),'utf8'),{kernel_path:'/fixture/base.cu',exp_dir:'/fixture/blaster',rl_iterations:1,rollout_steps:1},returns,null)
 const evals=out.calls.filter(c=>c.label.startsWith('eval-0-0-'));assert.equal(evals.length,2)
 assert.ok(evals[0].prompt.includes('context-candidate.cu'));assert.ok(evals[1].prompt.includes('context-candidate.cu'));assert.notEqual(evals[0].prompt,evals[1].prompt)
 assert.ok(evals.every(c=>c.prompt.includes('candidate identity collision')))
})
