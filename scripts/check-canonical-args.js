#!/usr/bin/env node
const fs = require('fs')
const path = require('path')

const root = process.cwd()

const forbiddenArgs = new Map([
  ['gpu_target', 'target_gpu'],
  ['gpu_type', 'target_gpu'],
  ['gpu_arch', 'target_gpu'],
  ['hardware_target', 'target_gpu'],
  ['target_hardware', 'target_gpu'],
  ['kernel_language', 'language'],
  ['target_language', 'language'],
  ['bench_command', 'benchmark_command'],
  ['eval_command', 'benchmark_command'],
  ['evaluation_command', 'benchmark_command'],
  ['verify_command', 'test_command'],
  ['validation_command', 'test_command'],
  ['correctness_command', 'test_command'],
  ['max_iterations', 'iterations'],
  ['max_rounds', 'iterations'],
  ['max_cycles', 'iterations'],
  ['rounds', 'iterations'],
  ['budget', 'iterations'],
  ['steps', 'iterations'],
  ['initial_kernel_path', 'kernel_path'],
  ['baseline_code_path', 'kernel_path'],
  ['source_code_path', 'kernel_path'],
  ['reference_kernel_path', 'kernel_path'],
  ['model_path', 'kernel_path'],
  ['problem_description', 'problem_definition'],
  ['task_spec', 'problem_definition'],
  ['operator_spec', 'problem_definition'],
  ['task_spec_path', 'problem_path'],
  ['task_path', 'problem_path'],
  ['kernel_spec_path', 'problem_path'],
  ['seed_count', 'seed_candidates'],
  ['max_seeds', 'seed_candidates'],
  ['samples_per_hypothesis', 'samples_per_plan'],
])

const allowedArgRefs = new Map([
  ['KernelBlaster/kernelblaster-kernel-optimization.js', new Set(['rl_iterations'])],
  ['ReGraphT/regrapht-kernel-optimization.js', new Set(['rollouts_per_select'])],
  ['TritorX/tritorx-operator-generation.js', new Set(['max_llm_calls_per_attempt'])],
])

const ignoredPathPrefixes = [
  '.git',
  'node_modules',
  'docs/superpowers/specs',
  'docs/superpowers/plans',
  '_substrate',
  'paper',
]

const workflowFilePattern = /\.js$/
const docFilePattern = /\.(md|ya?ml)$/
const hardcodedCommandChecks = [
  {
    pattern: /\bpython3?[ \t]+[\w./$"{'<-]/g,
    description: 'hardcoded Python command; use a user-provided *_command parameter',
  },
  {
    pattern: /\bbash[ \t]+[\w./$"{'<-]/g,
    description: 'hardcoded bash command; use a user-provided *_command parameter',
  },
  {
    pattern: /\bnvcc\s+[-\w./$"{'<]/g,
    description: 'hardcoded nvcc command; use user-provided compile_command',
  },
  {
    pattern: /\bncu\s+--/g,
    description: 'hardcoded ncu command; use user-provided ncu_command or ncu_binary contract',
  },
  {
    pattern: /\bnsys\s+--/g,
    description: 'hardcoded nsys command; use a user-provided profiler contract',
  },
]
const commandParameterPattern = new RegExp(
  String.raw`(?:compile_command|test_command|benchmark_command|baseline_command|profile_command|ncu_command|harness_build_cmd|build_cmd|run_cmd|lint_command|invariant_check_command|example|ncu_binary|nsys_binary)\s*:\s*["']?(?:python3?\b|bash\b|nvcc\b|ncu(?:\s+--|\b)|nsys(?:\s+--|\b))`,
  'g',
)

function relPath(fullPath) {
  return path.relative(root, fullPath).split(path.sep).join('/')
}

function shouldIgnore(rel) {
  return ignoredPathPrefixes.some(prefix => rel === prefix || rel.startsWith(`${prefix}/`))
}

function walk(dir, files = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name)
    const rel = relPath(fullPath)
    if (shouldIgnore(rel)) continue

    if (entry.isDirectory()) {
      walk(fullPath, files)
    } else if (workflowFilePattern.test(entry.name) || docFilePattern.test(entry.name)) {
      files.push(rel)
    }
  }
  return files
}

function isAllowedArg(file, arg) {
  const allowed = allowedArgRefs.get(file)
  return allowed ? allowed.has(arg) : false
}

function shouldCheckHardcodedCommands(file) {
  return workflowFilePattern.test(file)
    && !file.startsWith('scripts/')
    && !file.startsWith('_tools/')
    && !file.startsWith('_meta/tools/')
}

function isTopLevelWorkflow(file) {
  return workflowFilePattern.test(file)
    && !file.startsWith('scripts/')
    && !file.startsWith('_tools/')
    && !file.startsWith('_meta/')
    && !file.startsWith('_templates/')
}

function lineNumberAt(text, index) {
  return text.slice(0, index).split('\n').length
}

function docMentionsParameter(text, name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const patterns = [
    new RegExp(`\\\`${escaped}\\\``),
    new RegExp(`(^|[\\s,{])${escaped}\\s*:`, 'm'),
    new RegExp(`(^|\\|)\\s*${escaped}\\s*(\\||$)`, 'm'),
    new RegExp(`name:\\s*["']${escaped}["']`),
    new RegExp(`name:\\s*${escaped}(\\s|$)`),
  ]
  return patterns.some(pattern => pattern.test(text))
}

const failures = []

for (const file of walk(root)) {
  const text = fs.readFileSync(path.join(root, file), 'utf8')

  if (workflowFilePattern.test(file)) {
    for (const match of text.matchAll(/args\.([A-Za-z_][A-Za-z0-9_]*)/g)) {
      const arg = match[1]
      if (forbiddenArgs.has(arg) && !isAllowedArg(file, arg)) {
        failures.push(`${file}: args.${arg} should be args.${forbiddenArgs.get(arg)}`)
      }
    }
  }

  if (isTopLevelWorkflow(file)) {
    const requiredSuitabilitySnippets = [
      'const WORKFLOW_SUITABILITY =',
      'supported_languages:',
      'supported_problem_types:',
      'problem_types:',
      'reason:',
      'function assertWorkflowSuitability()',
      'assertWorkflowSuitability()',
    ]
    for (const snippet of requiredSuitabilitySnippets) {
      if (!text.includes(snippet)) {
        failures.push(`${file}: missing workflow suitability contract snippet: ${snippet}`)
      }
    }
  }

  if (shouldCheckHardcodedCommands(file)) {
    for (const { pattern, description } of hardcodedCommandChecks) {
      pattern.lastIndex = 0
      for (const match of text.matchAll(pattern)) {
        failures.push(`${file}:${lineNumberAt(text, match.index)}: ${description} (${match[0].trim()})`)
      }
    }
  }

  if (workflowFilePattern.test(file) || docFilePattern.test(file)) {
    commandParameterPattern.lastIndex = 0
    for (const match of text.matchAll(commandParameterPattern)) {
      failures.push(`${file}:${lineNumberAt(text, match.index)}: command examples must use user-provided placeholders (${match[0].trim()})`)
    }
  }

  if (docFilePattern.test(file)) {
    for (const [oldName, newName] of forbiddenArgs) {
      if (docMentionsParameter(text, oldName)) {
        failures.push(`${file}: parameter ${oldName} should be ${newName}`)
      }
    }
  }
}

if (failures.length) {
  console.error(failures.join('\n'))
  process.exit(1)
}

console.log('canonical args ok')
