const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const helper=fs.readFileSync(require('node:path').join(__dirname,'../../scaffolding/task-result.js'),'utf8');
const reference=vm.runInNewContext(helper+';__generatedTaskResultPath');
test('explicit candidate reference is independent of array position',()=>{
 assert.equal(reference({test_result_path:'/w/generated/initial_0b.json'},'/w/generated'),'/w/generated/initial_0b.json');
 assert.equal(reference({test_result_path:'/typo',test_result_json:{candidate_path:'/w/generated/initial_2.json.artifact/candidate.py'}},'/w/generated'),'/w/generated/initial_2.json');
});
for(const p of ['/other/a.json','/w/generated/../tests/a.json','/w/generated/nested/a.json','/w/generated/a.py'])test('reject unowned reference '+p,()=>assert.throws(()=>reference({test_result_path:p},'/w/generated'),e=>e.code==='TASK_RESULT_HOLD'));
