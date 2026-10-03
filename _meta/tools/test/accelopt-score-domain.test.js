'use strict'
const {test}=require('node:test'); const assert=require('node:assert/strict'); const fs=require('node:fs'); const path=require('node:path');
const run=require('../lib/run-workflow.js'); const source=fs.readFileSync(path.resolve(__dirname,'../../../AccelOpt/accelopt-kernel-optimization.js'),'utf8');
const args={kernel_path:'/fixture/seed.cu',exp_dir:'/fixture/exp',backend:'cuda',iterations:1,breadth:1,samples_per_plan:1,integration_pattern:'sol_execbench_solution',sol_cli:'/fixture/sol',sol_task_dir:'/fixture/task',sol_substrate_dir:'/fixture/substrate',sol_seed_dir:'/fixture/seed',sol_definition_path:'/fixture/definition.json'};
const valid=(lat,ratio)=>({compiled:true,correct:true,measurement_valid:true,full_workload_set:true,output_contract_valid:true,speedup:1.5,speedup_vs_seed:ratio,latency_ms:lat,candidate_latency_aggregate_ms:lat,n_pass:17,n_total:17,artifact_binding:{verified:true,binding_path:'/fixture/binding.json',candidate_sha256:'fixture-candidate'}});
async function replay(ncu,candidate=valid(.018,.020/.018),iterations=1){
 const agents={'read-baseline':{kernel_code:'SEED',op_type:'gemm',key_functions:['run']},'ncu-baseline':{latency_ms:ncu,ncu_available:true,bottleneck_diagnosis:'fixture',profile_summary:'diagnostic fixture'}};
 const evals={'sol-seed-baseline':{...valid(.020,1),candidate_role:'baseline'}};
 for(let i=0;i<iterations;i++){agents[`plan-${i}-0`]={title:'p',plan:'fixture',ncu_evidence:'fixture',expected_impact:'fixture'}; agents[`impl-${i}-p-v0`]={code:i?'CANDIDATE2':'CANDIDATE'};agents[`eval-plan_0_sample_0`]={is_compilable:true,is_correct:true,estimated_speedup:99,estimated_latency_ms:.0001};}
 evals['sol-eval-plan_0_sample_0']=candidate;
 return (await run(source,{...args,iterations},agents,evals)).result;
}
test('NCU diagnostic latency cannot change selection or measured baseline',async()=>{const a=await replay(.012),b=await replay(.030);assert.equal(a.best_kernel_code,'CANDIDATE');assert.equal(b.best_kernel_code,a.best_kernel_code);assert.equal(a.baseline_latency_ms,.020);assert.equal(a.overall_speedup,.020/.018);});
test('reference gain is not seed gain',async()=>{const a=await replay(.030,valid(.024,.020/.024));assert.equal(a.best_kernel_code,'SEED');assert.equal(a.overall_speedup,1);});
for(const [name,result] of [['missing',null],['partial',{...valid(.018,9),full_workload_set:false,n_pass:16}],['invalid',{...valid(.018,9),measurement_valid:false}],['output contract',{...valid(.018,9),output_contract_valid:false}],['unbound',{...valid(.018,9),artifact_binding:{verified:false}}]])test(`${name} Host measurement cannot retain static estimates`,async()=>{const a=await replay(.030,result);assert.equal(a.best_kernel_code,'SEED');assert.equal(a.overall_speedup,1);});
test('two rounds do not compound gain',async()=>{const a=await replay(.030,valid(.018,.020/.018),2);assert.equal(a.overall_speedup,.020/.018);assert.ok(a.candidate_beam.every(c=>c.speedup<=.020/.018));});
test('selected candidate returns verified source binding',async()=>{const a=await replay(.030);assert.equal(a.best_candidate_binding.binding_path,'/fixture/binding.json');assert.equal(a.best_kernel_code,'CANDIDATE');});

test('two improving rounds remain relative to the one seed',async()=>{let n=0;const a=await replay(.030,request=>{assert.equal(request.parentSolutionPath,'/fixture/seed/seed.solution.json');assert.equal(request.baselineEvaluationPath,'/fixture/exp/host_seed.bench.jsonl.result.json');assert.equal(request.bindingWorkflow,'accelopt-kernel-optimization');n++;return valid(n===1?.018:.016,n===1?.020/.018:.020/.016)},2);assert.equal(n,2);assert.equal(a.best_kernel_code,'CANDIDATE2');assert.equal(a.overall_speedup,1.25);assert.equal(a.candidate_beam[0].speedup,1.25);});
