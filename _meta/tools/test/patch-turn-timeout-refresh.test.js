'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os'),cp=require('node:child_process'),vm=require('node:vm')
const root=path.resolve(__dirname,'../../..'),script=path.join(root,'scripts/patch-turn-timeout.js')
const begin='// --- BEGIN inlined turn-timeout scaffolding (from _meta/scaffolding/turn-timeout.js) ---'
const end='// --- END inlined turn-timeout scaffolding ---'
test('timeout refresh replaces old generated code, preserves method body, and is idempotent',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'refresh-turn-timeout-')),file=path.join(dir,'fixture.js')
 try{
  const prefix='// method prefix\n',suffix='\n// method suffix\n'
  fs.writeFileSync(file,prefix+begin+'\nconst TURN_TIMEOUT_MS=1\nfunction withTurnTimeout(p){return p}\n'+end+suffix)
  const refresh=()=>{const r=cp.spawnSync(process.execPath,[script,'--refresh',file],{encoding:'utf8'});assert.equal(r.status,0,r.stderr)}
  refresh();const once=fs.readFileSync(file,'utf8');refresh();assert.equal(fs.readFileSync(file,'utf8'),once)
  assert.ok(once.startsWith(prefix+begin));assert.ok(once.endsWith(end+suffix))
  const isTimeout=vm.runInNewContext(once+';isTurnTimeout',{args:{}})
  assert.equal(isTimeout({code:'KERSOR_TURN_TIMEOUT'}),true)
  assert.equal(isTimeout({code:'KERSOR_PERMISSION_DENIED'}),false)
 }finally{fs.rmSync(dir,{recursive:true,force:true})}
})
