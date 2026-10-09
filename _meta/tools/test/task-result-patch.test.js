'use strict'
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),cp=require('node:child_process')
const root=path.resolve(__dirname,'../../..')
test('refresh native-only task-result block in place and idempotently without legacy anchor',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'akw-native-helper-'))
 try {
  const file=path.join(dir,'native.js'),prefix='export const meta = {name:"fixture"}\n',suffix='\nconst untouched = 42\n'
  fs.writeFileSync(file,prefix+'// --- BEGIN inlined task-result scaffolding (old) ---\noldHelper()\n// --- END inlined task-result scaffolding ---\n'+suffix)
  const run=()=>cp.execFileSync(process.execPath,[path.join(root,'scripts/patch-task-result.js'),file])
  run();const first=fs.readFileSync(file,'utf8');assert.ok(first.startsWith(prefix));assert.ok(first.endsWith(suffix));assert.ok(!first.includes('oldHelper()'));assert.ok(first.includes('missingStructuredOutput'))
  run();assert.equal(fs.readFileSync(file,'utf8'),first)
 } finally {fs.rmSync(dir,{recursive:true,force:true})}
})
