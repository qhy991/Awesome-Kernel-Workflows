const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '../../..')

function read(rel) {
  return fs.readFileSync(path.join(ROOT, rel), 'utf8')
}

test('Harness Engineering freezes external evidence ownership and promotes only correct measured improvements', () => {
  const source = read('HarnessEngineering/harness-engineering-kernel-optimization.js')
  const manifest = read('HarnessEngineering/manifest.yaml')

  for (const arg of ['harness_root', 'kernel_path', 'test_command', 'benchmark_command']) {
    assert.match(manifest, new RegExp(`- ${arg}\\n`))
  }
  assert.match(source, /if \(!evidenceIsPromotable\(baseline\)\)/)
  assert.match(source, /candidatePath\.startsWith\(EXP_DIR \+ '\/'\)/)
  assert.match(source, /evidenceIsPromotable\(evidence\) && candidateSpeedup > incumbentSpeedup/)
  assert.match(source, /The benchmark command may run only after compiled=true, correct=true/)
})

test('Atrex adapter delegates one official campaign and audits canonical promotion evidence', () => {
  const source = read('Atrex/atrex-kernel-optimization.js')
  const manifest = read('Atrex/manifest.yaml')

  assert.match(source, /orchestrator\/optimize\.py/)
  assert.match(source, /memory\/live\.json is observability only/)
  assert.match(source, /same-allocation ABBA evidence/)
  assert.match(source, /const officiallyPromoted = .*campaign_ok.*promoted.*correct/s)
  assert.equal((source.match(/phase: 'Launch Campaign'/g) || []).length, 1)
  assert.match(source, /\{ retries: 0, label: 'atrex-launch-campaign' \}/)
  assert.match(manifest, /fidelity_boundary: strict_high_fidelity/)
})

test('shared verification profiles reference the correct primary sources', () => {
  const doc = read('_substrate/verification/README.md')
  const harnessDoc = read('HarnessEngineering/README.md')

  for (const id of ['2608.12700', '2607.16241', '2607.27231']) {
    assert.match(doc, new RegExp(id.replace('.', '\\.')))
    assert.match(harnessDoc, new RegExp(id.replace('.', '\\.')))
  }
  assert.doesNotMatch(`${doc}\n${harnessDoc}`, /2607\.(?:19539|04957)/)
  assert.match(doc, /not built-in executable harnesses/)
})
