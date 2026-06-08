'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')
const FIX = path.join(ROOT, '_meta/tools/fixtures')
const { capturePrompts } = require(path.join(ROOT, '_meta/tools/print-workflow-prompts.js'))

const LOAD_DRIVER_TRITON = {
  present: true, backend_id: 'triton', source_ext: '.py', aux_ext: '.py',
  lang_fence: 'python', impl_requirements: 'Triton kernel.', methods: {}, hw_vendor: 'nvidia',
}

const LOAD_DRIVER_CUDA = {
  present: true, backend_id: 'cuda', source_ext: '.cu', aux_ext: '.cuh',
  lang_fence: 'cuda', impl_requirements: 'CUDA kernel.', methods: {}, hw_vendor: 'nvidia',
}

const DRIVER_ENVELOPE_PREFIXES = [
  'driver-build', 'driver-run', 'driver-profile',
  'driver-to-evidence', 'driver-diagnose', 'driver-anti-cheat',
]

function loadJSON(absOrRel) {
  const p = path.isAbsolute(absOrRel) ? absOrRel : path.join(ROOT, absOrRel)
  return JSON.parse(fs.readFileSync(p, 'utf8'))
}

function discoverMatrixEligibleWorkflows() {
  const dirs = fs.readdirSync(ROOT, { withFileTypes: true })
    .filter(d => d.isDirectory() && !d.name.startsWith('.') && !d.name.startsWith('_'))
    .map(d => d.name)

  const results = []
  for (const dir of dirs) {
    const manifestPath = path.join(ROOT, dir, 'manifest.yaml')
    if (!fs.existsSync(manifestPath)) continue
    const content = fs.readFileSync(manifestPath, 'utf8')
    const eligibleMatch = content.match(/matrix_eligible:\s*(\S+)/)
    if (!eligibleMatch) continue
    const eligible = eligibleMatch[1]
    if (eligible !== 'true' && eligible !== 'partial') continue

    const entryMatch = content.match(/entrypoint:\s*(\S+)/)
    let entrypoint
    if (entryMatch) {
      entrypoint = entryMatch[1]
    } else {
      const jsFiles = fs.readdirSync(path.join(ROOT, dir)).filter(f => f.endsWith('.js'))
      if (jsFiles.length !== 1) continue
      entrypoint = jsFiles[0]
    }

    const supportedMatch = content.match(/method_supported_backends:\s*(\S+)/)
    const methodSupported = supportedMatch ? supportedMatch[1] : 'any'

    results.push({
      dir,
      entrypoint,
      eligible,
      methodSupported,
      workflowPath: path.join(ROOT, dir, entrypoint),
    })
  }
  return results
}

const WORKFLOWS = discoverMatrixEligibleWorkflows()

function slugFor(dir) {
  return dir.toLowerCase()
}

function argsFixturePath(slug, backend) {
  if (backend === 'cuda') {
    const candidates = [
      path.join(FIX, `${slug}-args-cuda.json`),
      path.join(FIX, `${slug}-args-cuda-driver.json`),
    ]
    return candidates.find(p => fs.existsSync(p)) || null
  }
  const p = path.join(FIX, `${slug}-args-${backend}.json`)
  return fs.existsSync(p) ? p : null
}

function agentReturnsPath(slug) {
  const p = path.join(FIX, `${slug}-agent-returns.json`)
  return fs.existsSync(p) ? p : null
}

function buildReturns(slug, loadDriver) {
  const retPath = agentReturnsPath(slug)
  const base = retPath ? loadJSON(retPath) : {}
  return { 'load-driver': loadDriver, ...base }
}

test('matrix-smoke: discovered at least 10 matrix-eligible workflows', () => {
  assert.ok(WORKFLOWS.length >= 10,
    `expected at least 10 matrix-eligible workflows, found ${WORKFLOWS.length}: ${WORKFLOWS.map(w => w.dir).join(', ')}`)
})

const POSITIVE_CELLS = []

for (const wf of WORKFLOWS) {
  const slug = slugFor(wf.dir)

  const tritonArgs = argsFixturePath(slug, 'triton')
  if (tritonArgs && wf.methodSupported !== 'cuda') {
    POSITIVE_CELLS.push({
      workflow: wf.dir,
      backend: 'triton',
      wfPath: wf.workflowPath,
      argsPath: tritonArgs,
      loadDriver: LOAD_DRIVER_TRITON,
      slug,
    })
  }

  const cudaArgs = argsFixturePath(slug, 'cuda')
  if (cudaArgs && wf.methodSupported !== 'triton') {
    POSITIVE_CELLS.push({
      workflow: wf.dir,
      backend: 'cuda',
      wfPath: wf.workflowPath,
      argsPath: cudaArgs,
      loadDriver: LOAD_DRIVER_CUDA,
      slug,
    })
  }
}

for (const cell of POSITIVE_CELLS) {
  test(`matrix-smoke [${cell.workflow} x ${cell.backend}]: guard passes, load-driver dispatched first`, async () => {
    const args = loadJSON(cell.argsPath)
    const returns = buildReturns(cell.slug, cell.loadDriver)
    const caps = await capturePrompts({
      workflowPath: cell.wfPath,
      args,
      agentReturns: returns,
    })
    assert.ok(caps.length > 0, `expected at least one agent call for ${cell.workflow} x ${cell.backend}`)
    assert.equal(caps[0].label, 'load-driver',
      `load-driver must be the first agent call for ${cell.workflow} x ${cell.backend}`)
  })

  test(`matrix-smoke [${cell.workflow} x ${cell.backend}]: Layer-A driver envelope emitted`, async () => {
    const args = loadJSON(cell.argsPath)
    const returns = buildReturns(cell.slug, cell.loadDriver)
    const caps = await capturePrompts({
      workflowPath: cell.wfPath,
      args,
      agentReturns: returns,
    })
    const labels = caps.map(c => c.label)
    const driverLabels = labels.filter(l => l && DRIVER_ENVELOPE_PREFIXES.some(p => l.startsWith(p)))
    assert.ok(driverLabels.length >= 6,
      `expected at least 6 driver-* labels (one full envelope iteration) for ${cell.workflow} x ${cell.backend}, got ${driverLabels.length}: ${driverLabels.join(', ')}`)

    const kinds = new Set(driverLabels.map(l => {
      for (const p of DRIVER_ENVELOPE_PREFIXES) {
        if (l.startsWith(p)) return p
      }
      return l
    }))
    for (const prefix of DRIVER_ENVELOPE_PREFIXES) {
      assert.ok(kinds.has(prefix),
        `missing ${prefix}-* label for ${cell.workflow} x ${cell.backend}; got: ${driverLabels.join(', ')}`)
    }
  })
}

const NEGATIVE_CELLS = [
  {
    workflow: 'KernelFoundryDx',
    backend: 'cuda',
    expectedError: /supports only backend="triton"/,
    description: 'triton-only workflow rejects cuda backend',
  },
]

for (const neg of NEGATIVE_CELLS) {
  const wf = WORKFLOWS.find(w => w.dir === neg.workflow)
  if (!wf) continue

  test(`matrix-smoke negative [${neg.workflow} x ${neg.backend}]: ${neg.description}`, async () => {
    const slug = slugFor(neg.workflow)
    const tritonArgs = argsFixturePath(slug, 'triton')
    assert.ok(tritonArgs, `need triton args fixture to derive ${neg.backend} negative cell`)
    const args = loadJSON(tritonArgs)
    args.backend = neg.backend
    args.backend_dir = `_substrate/backends/${neg.backend}`
    args.language = neg.backend

    await assert.rejects(
      capturePrompts({
        workflowPath: wf.workflowPath,
        args,
        agentReturns: {},
      }),
      neg.expectedError,
      `expected guard to reject ${neg.workflow} x ${neg.backend}`,
    )
  })
}

test(`matrix-smoke: total positive cells tested is at least 13`, () => {
  assert.ok(POSITIVE_CELLS.length >= 13,
    `expected at least 13 positive cells, found ${POSITIVE_CELLS.length}`)
})
