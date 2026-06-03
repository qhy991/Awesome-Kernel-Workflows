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

const skillChecks = [
  ['AKO4X/ako4x-kernel-optimizer.js', ['optionalSkills', 'skill_binding_mode', 'prompt_reference_only']],
  ['KDA/kda-kernel-workflow.js', ['optionalSkills', 'skill_binding_mode', 'recommended_external_skills']],
  ['AdaExplore/adaexplore-kernel-optimization.js', ['skillMemoryContract', 'skill_memory_path', 'method_memory_file']],
]

let failures = []

for (const [file, tokens] of [...checks, ...skillChecks]) {
  const filePath = path.join(root, file)
  let text = ''
  try {
    text = fs.readFileSync(filePath, 'utf8')
  } catch (error) {
    failures.push(`${file}: cannot read (${error.message})`)
    continue
  }

  for (const token of tokens) {
    if (!text.includes(token)) {
      failures.push(`${file}: missing ${token}`)
    }
  }
}

if (failures.length > 0) {
  console.error(failures.join('\n'))
  process.exit(1)
}

console.log(`fidelity contracts ok (${checks.length} workflows, ${skillChecks.length} skill contracts)`)
