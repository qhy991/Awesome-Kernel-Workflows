'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path')
const s=fs.readFileSync(path.resolve(__dirname,'../../scaffolding/task-result.js'),'utf8')
const p='/fixture/result.json'
const passed={contract_version:'kersor-task-result-v1',test_result_path:p,compiled:true,correct:true,full_workload_set:true,measurement_valid:true,n_pass:17,n_total:17,candidate_latency_aggregate_ms:.02,speedup_vs_reference:1.5,candidate_path:p+'.artifact/candidate.py',source_binding:{verified:true,source_sha256:'a'.repeat(64)}}
const unknown={...passed,outcome_state:'unknown',failure_origin:'infrastructure_or_incomplete_evidence',correct:false,measurement_valid:false,n_pass:0}

// A context whose agent returns the given task results in order, one per evaluation.
function load(results,extraArgs={}){
  let i=0
  const ctx={args:{...extraArgs},agentRetry:async fn=>fn(),agent:()=>({test_result_path:p,test_result_json:results[i++]})}
  vm.runInNewContext(s+';this.candidate=__nativeTaskEvaluateCandidate;this.parent=__nativeTaskAcceptedParent;this.counts=__unknownCandidates',ctx)
  return ctx
}
const req={candidatePath:'/fixture/c.py',resultPath:p,command:'trusted {kernel_path} {result_path}',workloadCount:17,label:'c'}

test('an unknown candidate is discarded unscored instead of ending the search',async()=>{
  const w=load([unknown])
  const out=await w.candidate(req)
  assert.equal(out.discarded_unknown,true)
  assert.equal(out.outcome_state,'unknown')
  assert.equal(out.is_valid,false)
  assert.equal(out.speedup,0)
  assert.equal(out.metric_value,0)
  assert.equal(w.counts.total,1)
})

test('a passing candidate is returned unchanged and resets the unknown streak',async()=>{
  const w=load([unknown,passed,unknown,unknown])
  await w.candidate(req)
  const ok=await w.candidate(req)
  assert.equal(ok.is_valid,true)
  assert.equal(ok.speedup,1.5)
  assert.equal(w.counts.consecutive,0)
  await w.candidate(req);await w.candidate(req)
  assert.equal(w.counts.consecutive,2)
})

test('the third consecutive unknown candidate HOLDs the workflow',async()=>{
  const w=load([unknown,unknown,unknown])
  await w.candidate(req);await w.candidate(req)
  await assert.rejects(w.candidate(req),e=>e.code==='TASK_RESULT_HOLD'&&e.retryable===false&&/3 consecutive unknown/.test(e.message))
})

test('unknown_candidate_limit sets the HOLD threshold',async()=>{
  const w=load([unknown],{unknown_candidate_limit:1})
  await assert.rejects(w.candidate(req),e=>e.code==='TASK_RESULT_HOLD')
})

for(const [name,raw] of Object.entries({
  partialWithoutOutcome:{...passed,full_workload_set:false,n_pass:16},
  missingBinding:{...passed,source_binding:null},
  contractMismatch:{...unknown,contract_version:'other'},
  unknownModelObservation:{...unknown,model_observation:'unknown'},
  missingJson:null,
}))test('a '+name+' result still HOLDs a search candidate at once',async()=>{
  const w=load([raw])
  await assert.rejects(w.candidate(req),e=>e.code==='TASK_RESULT_HOLD'&&e.retryable===false)
  assert.equal(w.counts.total,0)
})

test('the accepted parent still HOLDs on its first unknown outcome',async()=>{
  const w=load([unknown])
  await assert.rejects(w.parent({...req,label:'parent'}),e=>e.code==='TASK_RESULT_HOLD')
  assert.equal(w.counts.total,0)
})

test('an error that is not a task HOLD propagates and does not count as unknown',async()=>{
  const ctx={args:{},agentRetry:async fn=>fn(),agent:()=>({})}
  vm.runInNewContext(s+';__nativeTaskEvaluate=async()=>{throw new TypeError("bug")};this.candidate=__nativeTaskEvaluateCandidate;this.counts=__unknownCandidates',ctx)
  await assert.rejects(ctx.candidate(req),e=>e.name==='TypeError'&&e.message==='bug')
  assert.equal(ctx.counts.total,0)
})
