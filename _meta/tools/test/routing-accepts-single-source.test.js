'use strict'
// Enforcement test for issue #24 — routing.accepts: as single source of truth.
//
// Phase A+C: every workflow's assertWorkflowSuitability (and WORKFLOW_SUITABILITY
// const) was removed; eligibility (problem_type) now lives in manifest
// routing.accepts and is enforced by the KerSor selector (KerSor #31). This test
// guards against reintroducing the dual source of truth:
//   (a) no top-level workflow .js defines assertWorkflowSuitability or
//       WORKFLOW_SUITABILITY (the old JS eligibility gate);
//   (b) every manifest declares routing.accepts.problem_type.
//
// Note on manifest-internal field consistency (issue #24 review J4): a hard
// subset check across routing.languages / routing.backends / backend.supported /
// backend.method_supported_backends is intentionally NOT enforced here — those
// fields encode distinct concepts (broad backend support vs method-specific
// support vs languages incl. non-backend entries like python_reference) and a
// naive check false-positives on AccelOpt/others. Trust assumption documented in
// the PR body; a dedicated consistency issue can formalize it later.
//
// Run: node --test _meta/tools/test/routing-accepts-single-source.test.js

const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..', '..', '..')

function topLevelWorkflows() {
  // Top-level workflow .js files: <DIR>/<file>.js where DIR is a top-level dir
  // (one path segment), excluding tooling/script dirs.
  const skip = new Set(['_tools', '_meta', '_substrate', '_templates', 'scripts', 'node_modules', '.git'])
  const out = []
  for (const entry of fs.readdirSync(ROOT, { withFileTypes: true })) {
    if (!entry.isDirectory() || skip.has(entry.name)) continue
    const dir = path.join(ROOT, entry.name)
    for (const f of fs.readdirSync(dir)) {
      if (f.endsWith('.js')) out.push(path.join(dir, f))
    }
  }
  return out
}

function manifests() {
  return topLevelWorkflows()
    .map((wf) => path.join(path.dirname(wf), 'manifest.yaml'))
    .filter((m) => fs.existsSync(m))
}

test('(a) no top-level workflow .js defines assertWorkflowSuitability or WORKFLOW_SUITABILITY', () => {
  const fails = []
  for (const wf of topLevelWorkflows()) {
    const src = fs.readFileSync(wf, 'utf8')
    if (/\bfunction\s+assertWorkflowSuitability\s*\(/.test(src)) {
      fails.push(`${path.relative(ROOT, wf)}: defines assertWorkflowSuitability (eligibility moved to manifest routing.accepts)`)
    }
    if (/^\s*const\s+WORKFLOW_SUITABILITY\s*=/.test(src)) {
      fails.push(`${path.relative(ROOT, wf)}: defines WORKFLOW_SUITABILITY const (moved to manifest routing.accepts)`)
    }
  }
  assert.equal(fails.length, 0, `issue #24 regressions:\n${fails.join('\n')}`)
})

test('(b) every manifest declares routing.accepts.problem_type', () => {
  const fails = []
  for (const m of manifests()) {
    const src = fs.readFileSync(m, 'utf8')
    // Inserted block (2-space accepts, 4-space problem_type) — see docs/manifest-schema.yaml.
    const hasAcceptsProblemType = src.includes('\n  accepts:\n    problem_type:')
    if (!hasAcceptsProblemType) {
      fails.push(`${path.relative(ROOT, m)}: missing routing.accepts.problem_type`)
    }
  }
  assert.equal(fails.length, 0, `manifests missing routing.accepts.problem_type:\n${fails.join('\n')}`)
})
