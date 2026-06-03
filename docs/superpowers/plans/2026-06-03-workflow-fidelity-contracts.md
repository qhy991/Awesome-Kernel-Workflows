# Workflow Fidelity Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the audit findings into workflow-owned evidence contracts for high-risk paper-derived workflows.

**Architecture:** Add a lightweight Node static checker for fidelity contracts, then update the relevant workflow files so critical paper mechanisms are represented by explicit args, evidence modes, missing-evidence behavior, and return fields. Keep changes scoped to contract language and workflow state; do not rewrite the workflows.

**Tech Stack:** Claude Code workflow JavaScript files, Node.js built-in `fs`/`path`/`assert`, shell validation commands.

---

### Task 1: Add Static Fidelity Contract Check

**Files:**
- Create: `scripts/check-fidelity-contracts.js`

- [ ] **Step 1: Write the failing checker**

Create a Node script that reads the target workflow files and asserts required contract tokens:

```javascript
#!/usr/bin/env node
const fs = require('fs')
const path = require('path')

const root = process.cwd()
const checks = [
  ['ARGUS/argus-kernel-optimization.js', ['invariant_check_command', 'invariant_result_path', 'invariant_evidence', 'missing_invariant_evidence']],
  ['KernelBand/kernelband-kernel-optimization.js', ['feature_vector_result_path', 'hardware_signature_result_path', 'evidence_mode']],
  ['KernelFoundry/kernelfoundry-kernel-optimization.js', ['descriptor_result_path', 'archive_update_result_path', 'evidence_mode']],
  ['cuPilot/cupilot-kernel-optimization.js', ['roofline_result_path', 'strategy_corpus_path', 'evidence_mode']],
  ['TritorX/tritorx-operator-generation.js', ['strict_harness', 'harness_evidence', 'TritorX-style FSM']],
  ['CUDAAgent/cuda-agent-kernel-optimization.js', ['adaptation_scope', 'inference_time_adaptation']],
  ['CUDALLM/cudallm-fsr-kernel-generation.js', ['adaptation_scope', 'workflow_adaptation']],
  ['ReGraphT/regrapht-kernel-optimization.js', ['adaptation_scope', 'training_free_inference']],
]

let failures = []
for (const [file, tokens] of checks) {
  const text = fs.readFileSync(path.join(root, file), 'utf8')
  for (const token of tokens) {
    if (!text.includes(token)) failures.push(`${file}: missing ${token}`)
  }
}

if (failures.length) {
  console.error(failures.join('\n'))
  process.exit(1)
}
console.log(`fidelity contracts ok (${checks.length} workflows)`)
```

- [ ] **Step 2: Run checker to verify RED**

Run: `node scripts/check-fidelity-contracts.js`

Expected: FAIL with missing tokens for current workflows.

### Task 2: Add Contract Fields to Workflows

**Files:**
- Modify: `ARGUS/argus-kernel-optimization.js`
- Modify: `KernelBand/kernelband-kernel-optimization.js`
- Modify: `KernelFoundry/kernelfoundry-kernel-optimization.js`
- Modify: `cuPilot/cupilot-kernel-optimization.js`
- Modify: `TritorX/tritorx-operator-generation.js`
- Modify: `CUDAAgent/cuda-agent-kernel-optimization.js`
- Modify: `CUDALLM/cudallm-fsr-kernel-generation.js`
- Modify: `ReGraphT/regrapht-kernel-optimization.js`

- [ ] **Step 1: Add minimal implementation**

Add args, comments, prompt requirements, and return fields for the exact tokens checked by `scripts/check-fidelity-contracts.js`.

- [ ] **Step 2: Run checker to verify GREEN**

Run: `node scripts/check-fidelity-contracts.js`

Expected: PASS with `fidelity contracts ok`.

### Task 3: Run Broad Validation

**Files:**
- No further file edits expected.

- [ ] **Step 1: Run syntax validation**

Run: `for f in ...; do node --input-type=module --check ...; done` using the async-wrapper pattern for top-level returns.

Expected: all checked files parse.

- [ ] **Step 2: Check git diff**

Run: `git diff --stat && git diff -- scripts/check-fidelity-contracts.js ARGUS/argus-kernel-optimization.js KernelBand/kernelband-kernel-optimization.js KernelFoundry/kernelfoundry-kernel-optimization.js cuPilot/cupilot-kernel-optimization.js TritorX/tritorx-operator-generation.js CUDAAgent/cuda-agent-kernel-optimization.js CUDALLM/cudallm-fsr-kernel-generation.js ReGraphT/regrapht-kernel-optimization.js`

Expected: only scoped contract changes.
