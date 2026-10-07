'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),cp=require('node:child_process')
const canonical=fs.readFileSync(path.resolve(__dirname,'../../scaffolding/typed-args.js'),'utf8')
const api=vm.runInNewContext(canonical+';({store:__workspaceStore,source:__workspaceSource,result:__workspaceResult})',{args:{}})
function execute(root,file,code){const instructions=api.store(file,code,root);return cp.spawnSync('/bin/sh',[],{input:instructions.slice(instructions.indexOf('\n')+1),encoding:'utf8'})}
test('real source persistence preserves UTF8, quotes, multiline bytes and replays idempotently',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'workflow-source-'))
 try{const f=path.join(root,'a.py'),code="# 中文 ' $HOME `echo unsafe`\n"+'x'.repeat(200000)+'\n';assert.equal(execute(root,f,code).status,0);assert.equal(fs.readFileSync(f,'utf8'),code);assert.equal(execute(root,f,code).status,0);assert.ok(execute(root,f,'different').status!==0);assert.equal(fs.readFileSync(f,'utf8'),code)}finally{fs.rmSync(root,{recursive:true,force:true})}
})
test('candidate writes refuse outside paths and symlink aliases without changing targets',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'workflow-source-')),outer=fs.mkdtempSync(path.join(os.tmpdir(),'workflow-outside-'))
 try{const target=path.join(outer,'target.py');fs.writeFileSync(target,'old');assert.notEqual(execute(root,target,'new').status,0);fs.symlinkSync(target,path.join(root,'link.py'));assert.notEqual(execute(root,path.join(root,'link.py'),'new').status,0);assert.equal(fs.readFileSync(target,'utf8'),'old')}finally{fs.rmSync(root,{recursive:true,force:true});fs.rmSync(outer,{recursive:true,force:true})}
})
test('references do not inline bytes; unpersisted fallback remains complete',()=>{
 const raw='COMPLETE'.repeat(10000);assert.ok(!api.source('/candidate.py',raw).includes(raw));assert.equal(api.source('',raw),raw);const e={logs:raw};assert.equal(api.result('',e),e);assert.equal(api.result('/result.json',e).result_path,'/result.json');assert.equal(e.logs,raw)
})
test('KernelSkill cache reuse retains the original materialized source location',async()=>{
 const text=fs.readFileSync(path.resolve(__dirname,'../../../KernelSkill/kernelskill-kernel-optimization.js'),'utf8');const start=text.indexOf('async function measureKernel('),end=text.indexOf('function validHostResult(',start)
 let calls=0;const env={HOST_SOL:true,EXP_DIR:'/fixture',hostMeasurements:new Map(),__solExecbenchEvaluate:async()=>{calls++;return {compiled:true}}}
 for(const k of text.slice(start,end).match(/SOL_[A-Z_]+/g)||[])env[k]='configured'
 const measure=vm.runInNewContext(text.slice(start,end)+';measureKernel',env)
 const a=await measure('same source','seed0'),b=await measure('same source','r2');assert.equal(calls,1);assert.equal(a,b);assert.equal(b.workspace_source_path,'/fixture/kernelskill_seed0.cu')
 assert.match(text,/currentSourcePath = __hostMeasured\.workspace_source_path/)
})
