# Workflow Parameter Canonicalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate workflow arguments to one canonical API and let applicable optimizers start from either an existing `kernel_path` or a `problem_definition` / `problem_path` that first generates an initial kernel.

**Architecture:** Add a static canonical-argument checker, then migrate schema/generator/templates, then update workflow families in batches. The generate-then-optimize setup is copied into standalone workflows because workflow files do not share runtime imports.

**Tech Stack:** Claude Code workflow JavaScript, YAML manifests, Markdown docs, Node.js built-in `fs`/`path` checker scripts, shell validation commands.

---

### Task 1: Add Canonical Argument Static Checker

**Files:**
- Create: `scripts/check-canonical-args.js`

- [ ] **Step 1: Write the failing checker**

Create `scripts/check-canonical-args.js`:

```javascript
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
  ['max_iterations', 'iterations'],
  ['rounds', 'iterations'],
  ['budget', 'iterations'],
  ['steps', 'iterations'],
  ['initial_kernel_path', 'kernel_path'],
  ['baseline_code_path', 'kernel_path'],
  ['source_code_path', 'kernel_path'],
  ['reference_kernel_path', 'kernel_path'],
  ['problem_description', 'problem_definition'],
  ['task_spec', 'problem_definition'],
  ['operator_spec', 'problem_definition'],
  ['task_spec_path', 'problem_path'],
  ['task_path', 'problem_path'],
  ['seed_count', 'seed_candidates'],
  ['max_seeds', 'seed_candidates'],
])

const allowedArgRefs = new Map([
  ['CutlassGEMM/cutlass-gemm-optimization.js', new Set(['iterations'])],
  ['KernelBlaster/kernelblaster-kernel-optimization.js', new Set(['rl_iterations'])],
  ['TritorX/tritorx-operator-generation.js', new Set(['max_llm_calls_per_attempt'])],
  ['ReGraphT/regrapht-kernel-optimization.js', new Set(['rollouts_per_select'])],
])

const ignoredDirs = new Set(['.git', 'node_modules', 'docs/superpowers/specs', 'docs/superpowers/plans'])
const workflowFilePattern = /\.js$/
const docFilePattern = /\.(md|yaml)$/

function walk(dir, files = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name)
    const rel = path.relative(root, fullPath)
    if (entry.isDirectory()) {
      if ([...ignoredDirs].some(prefix => rel === prefix || rel.startsWith(`${prefix}${path.sep}`))) continue
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

  if (docFilePattern.test(file)) {
    for (const [oldName, newName] of forbiddenArgs) {
      const pattern = new RegExp(`(^|[^A-Za-z0-9_])${oldName}([^A-Za-z0-9_]|$)`, 'm')
      if (pattern.test(text)) {
        failures.push(`${file}: mention of ${oldName} should be ${newName}`)
      }
    }
  }
}

if (failures.length) {
  console.error(failures.join('\n'))
  process.exit(1)
}

console.log('canonical args ok')
```

- [ ] **Step 2: Run checker to verify RED**

Run: `node scripts/check-canonical-args.js`

Expected: FAIL with old names such as `gpu_target`, `bench_command`, `problem_description`, and `rounds`.

- [ ] **Step 3: Commit the checker after migration is green**

Run after all later tasks pass:

```bash
git add scripts/check-canonical-args.js
git commit -m "test: add canonical workflow argument check"
```

Expected: commit succeeds after the checker passes.

### Task 2: Update Manifest Schema, Templates, And Generator Prompts

**Files:**
- Modify: `_manifests/schema.yaml`
- Modify: `_meta/manifests/schema.yaml`
- Modify: `_tools/generate-workflow.js`
- Modify: `_meta/tools/generate-workflow.js`
- Modify: `_templates/iterative-loop.js`
- Modify: `_templates/search-based.js`
- Modify: `_templates/single-pass.js`
- Modify: `_templates/tree-exploration.js`
- Modify: `_meta/templates/iterative-loop.js`
- Modify: `_meta/templates/search-based.js`
- Modify: `_meta/templates/single-pass.js`
- Modify: `_meta/templates/tree-exploration.js`

- [ ] **Step 1: Add manifest `inputs` schema block**

Insert this section after `workflow:` in both schema files:

```yaml
# ---------------------------------------------------------------------------
# INPUT CONTRACT — How callers provide the starting point
# ---------------------------------------------------------------------------
inputs:
  supports_existing_kernel: true
  supports_problem_definition: true
  required_one_of:
    - kernel_path
    - problem_definition
    - problem_path
  generation:
    enabled: true
    seed_candidates_arg: seed_candidates
    output_arg: generated_kernel_path
    requires_correctness_evidence: true
```

- [ ] **Step 2: Update generator arg guidance**

In both generator workflow files, replace the common args guidance with this text:

```text
Use canonical common args:
- kernel_path: existing kernel/source file to optimize
- problem_definition: inline problem definition when no initial kernel exists
- problem_path: file containing the problem definition
- language: implementation language
- target_gpu: target accelerator string
- compile_command: compile/import command with {kernel_path}/{result_path}
- test_command: correctness command with {kernel_path}/{result_path}
- benchmark_command: performance command with {kernel_path}/{result_path}
- iterations: outer-loop budget unless a paper-specific budget name is required
- seed_candidates: number of initial generated kernels for generation mode
- exp_dir: artifact directory
Do not emit old aliases such as gpu_target, bench_command, eval_command, rounds, problem_description, task_spec, or task_path.
```

- [ ] **Step 3: Add template setup policy**

Add this comment block near the top of every template, before method-specific setup:

```javascript
// Canonical input policy:
// - If args.kernel_path is provided, optimize that existing kernel.
// - Else require args.problem_definition or args.problem_path, generate seed_candidates initial kernels,
//   verify them with test_command or benchmark_command, and optimize the best verified seed.
// - Return input_mode, generated_kernel_path, initial_candidates, and initial_generation_result.
```

- [ ] **Step 4: Verify generated files parse as text**

Run: `node --check scripts/check-fidelity-contracts.js`

Expected: PASS with no syntax errors.

### Task 3: Migrate Existing Problem-Definition Workflows

**Files:**
- Modify: `KernelAgent/kernelagent-triton-synthesis.js`
- Modify: `KernelAgent/README.md`
- Modify: `KernelAgent/manifest.yaml`
- Modify: `CUDALLM/cudallm-fsr-kernel-generation.js`
- Modify: `CUDALLM/README.md`
- Modify: `CUDALLM/README.zh-CN.md`
- Modify: `CUDALLM/manifest.yaml`
- Modify: `KernelFoundry/kernelfoundry-kernel-optimization.js`
- Modify: `KernelFoundry/README.md`
- Modify: `KernelFoundryDx/kernelfoundrydx-kernel-optimization.js`
- Modify: `KernelFoundryDx/README.md`
- Modify: `KernelFoundryDx/README.zh-CN.md`
- Modify: `KernelSkill/kernelskill-kernel-optimization.js`
- Modify: `KernelSkill/README.md`
- Modify: `KernelSkill/README.zh-CN.md`
- Modify: `KSearch/ksearch-kernel-optimization.js`
- Modify: `KSearch/README.md`
- Modify: `KSearch/README.zh-CN.md`
- Modify: `AdaExplore/adaexplore-kernel-optimization.js`
- Modify: `AdaExplore/README.md`

- [ ] **Step 1: Rename inline problem args**

Apply these code-level replacements where the current field is the authoritative problem input:

```text
args.problem_description -> args.problem_definition
args.task_spec -> args.problem_definition
args.operator_spec -> args.problem_definition
args.task_spec_path -> args.problem_path
args.task_path -> args.problem_path
args.reference_path -> args.problem_path only when it is a PyTorch task/problem file
```

- [ ] **Step 2: Rename generation budgets**

Apply these replacements:

```text
args.max_rounds -> args.iterations
args.rounds -> args.iterations when it is the generic refinement loop
args.max_seeds -> args.seed_candidates
args.seed_count -> args.seed_candidates
```

Keep method-specific budgets such as `rollouts_per_select`, `max_llm_calls_per_attempt`, and population controls.

- [ ] **Step 3: Add return fields for input contract**

Ensure each workflow return object includes these fields:

```javascript
input_mode: 'generate_then_optimize',
problem_definition: PROBLEM_DEFINITION,
problem_path: PROBLEM_PATH,
generated_kernel_path: bestKernelPath || '',
initial_candidates: candidates || [],
initial_generation_result: {
  verified: verifiedKernels.length > 0,
  selected_candidate_id: verifiedKernels[0]?.id || '',
},
```

Adapt variable names to the workflow's existing candidate arrays.

- [ ] **Step 4: Update docs and manifests**

Each README usage block and arg table should use `problem_definition`, `problem_path`, `language`, `target_gpu`, `benchmark_command`, `test_command`, `iterations`, `seed_candidates`, and `exp_dir`.

- [ ] **Step 5: Run partial scan**

Run:

```bash
rg -n "problem_description|task_spec|operator_spec|task_spec_path|task_path|max_rounds|max_seeds|seed_count|bench_command|eval_command" KernelAgent CUDALLM KernelFoundry KernelFoundryDx KernelSkill KSearch AdaExplore
```

Expected: No matches except explanatory migration text in committed design/plan docs.

### Task 4: Add Generate-Then-Optimize Setup To Existing-Kernel Optimizers

**Files:**
- Modify: `AKO4X/ako4x-kernel-optimizer.js`
- Modify: `AccelOpt/accelopt-kernel-optimization.js`
- Modify: `Generalist/generalist-kernel-optimization.js`
- Modify: `KernelBand/kernelband-kernel-optimization.js`
- Modify: `KernelBlaster/kernelblaster-kernel-optimization.js`
- Modify: `ARGUS/argus-kernel-optimization.js`
- Modify: `KDA/kda-kernel-workflow.js`
- Modify: `CUDAAgent/cuda-agent-kernel-optimization.js`
- Modify: `Astra/astra-kernel-optimization.js`
- Modify: `ReGraphT/regrapht-kernel-optimization.js`

- [ ] **Step 1: Add canonical input constants**

At the top of each applicable optimizer, replace required `kernel_path` only setup with this pattern:

```javascript
const KERNEL_PATH_ARG = args.kernel_path || ''
const PROBLEM_DEFINITION = args.problem_definition || ''
const PROBLEM_PATH = args.problem_path || ''
const LANGUAGE = args.language || 'auto'
const TARGET_GPU = args.target_gpu || 'A100'
const SEED_CANDIDATES = args.seed_candidates || 3
const INPUT_MODE = KERNEL_PATH_ARG ? 'optimize_existing' : 'generate_then_optimize'

if (!KERNEL_PATH_ARG && !PROBLEM_DEFINITION && !PROBLEM_PATH) {
  throw new Error('Provide one of kernel_path, problem_definition, or problem_path')
}

let selectedInitialKernelPath = KERNEL_PATH_ARG
let initialCandidates = []
let initialGenerationResult = null
```

Use the workflow's existing default for `TARGET_GPU` when it is method-specific.

- [ ] **Step 2: Insert generation phase before method-specific setup reads the kernel**

Before the first prompt that reads `KERNEL_PATH`, insert:

```javascript
if (INPUT_MODE === 'generate_then_optimize') {
  phase('Setup')

  const generation = await agent(`Generate initial ${LANGUAGE} GPU kernel candidates from the problem definition.

# Problem definition
${PROBLEM_DEFINITION || `(Read problem file: ${PROBLEM_PATH})`}

# Target GPU
${TARGET_GPU}

# Requirements
1. Produce ${SEED_CANDIDATES} diverse candidate kernels.
2. Each candidate must be complete source code, not a sketch.
3. State the expected kernel entry point and file extension.
4. Do not use library calls that replace the kernel's core computation.`, {
    label: 'generate-initial-kernels',
    phase: 'Setup',
    schema: {
      type: 'object',
      properties: {
        candidates: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              language: { type: 'string' },
              code: { type: 'string' },
              rationale: { type: 'string' },
              expected_kernel_entry: { type: 'string' },
              file_extension: { type: 'string' },
            },
            required: ['id', 'code'],
          },
        },
      },
      required: ['candidates'],
    },
  })

  initialCandidates = generation.candidates || []

  const evaluatedSeeds = await parallel(
    initialCandidates.map((candidate, idx) => () =>
      agent(`Materialize and evaluate this generated seed candidate.

# Candidate ID
${candidate.id || `seed-${idx}`}

# Output path
${EXP_DIR}/generated/${candidate.id || `seed-${idx}`}.${candidate.file_extension || 'cu'}

# Commands
- compile_command: ${COMPILE_CMD || '(not provided)'}
- test_command: ${TEST_CMD || '(not provided)'}
- benchmark_command: ${BENCHMARK_CMD || '(not provided)'}

# Contract
Write the candidate to the output path. If a command is provided, substitute {kernel_path}
with that output path and {result_path} with ${EXP_DIR}/generated/${candidate.id || `seed-${idx}`}.json.
Return whether the candidate compiled, passed correctness, and its measured score/latency if available.

# Candidate code
\`\`\`
${candidate.code}
\`\`\``, {
        label: `eval-generated-seed-${idx}`,
        phase: 'Setup',
        schema: {
          type: 'object',
          properties: {
            kernel_path: { type: 'string' },
            compiled: { type: 'boolean' },
            correct: { type: 'boolean' },
            latency_ms: { type: 'number' },
            speedup: { type: 'number' },
            score: { type: 'number' },
            raw_output: { type: 'string' },
          },
          required: ['kernel_path', 'compiled', 'correct'],
        },
      })
    )
  )

  const verifiedSeeds = evaluatedSeeds.filter(seed => seed && seed.compiled && seed.correct)
  if (verifiedSeeds.length === 0 && (TEST_CMD || BENCHMARK_CMD)) {
    return {
      success: false,
      input_mode: INPUT_MODE,
      problem_definition: PROBLEM_DEFINITION,
      problem_path: PROBLEM_PATH,
      initial_candidates: initialCandidates,
      initial_generation_result: { verified: false, evaluated: evaluatedSeeds },
      failure_reason: 'No generated seed compiled and passed correctness',
    }
  }

  const rankedSeeds = verifiedSeeds.length ? verifiedSeeds : evaluatedSeeds.filter(Boolean)
  rankedSeeds.sort((a, b) => {
    const aMetric = a.speedup || (a.latency_ms ? -a.latency_ms : a.score || 0)
    const bMetric = b.speedup || (b.latency_ms ? -b.latency_ms : b.score || 0)
    return bMetric - aMetric
  })
  selectedInitialKernelPath = rankedSeeds[0]?.kernel_path || ''
  initialGenerationResult = {
    verified: verifiedSeeds.length > 0,
    selected_kernel_path: selectedInitialKernelPath,
    evaluated: evaluatedSeeds,
  }
}

const KERNEL_PATH = selectedInitialKernelPath
```

- [ ] **Step 3: Rename old common args inside each workflow**

Apply these replacements in the files listed for this task:

```text
args.kernel_language -> args.language
args.gpu_target -> args.target_gpu
args.gpu_type -> args.target_gpu
args.hardware_target -> args.target_gpu
args.bench_command -> args.benchmark_command
args.eval_command -> args.benchmark_command
args.evaluation_command -> args.benchmark_command
args.verify_command -> args.test_command
args.validation_command -> args.test_command
args.correctness_command -> args.test_command
args.max_iterations -> args.iterations
args.rounds -> args.iterations when it is the generic outer loop
args.budget -> args.iterations when it is the generic outer loop
args.steps -> args.iterations when it is the generic outer loop
args.samples_per_hypothesis -> args.samples_per_plan
```

Keep `ncu_command`, `ncu_binary`, `invariant_check_command`, and evidence result paths unchanged.

- [ ] **Step 4: Add return fields**

Each optimizer return object should include:

```javascript
input_mode: INPUT_MODE,
problem_definition: PROBLEM_DEFINITION,
problem_path: PROBLEM_PATH,
generated_kernel_path: INPUT_MODE === 'generate_then_optimize' ? KERNEL_PATH : '',
initial_candidates: initialCandidates,
initial_generation_result: initialGenerationResult,
```

- [ ] **Step 5: Run partial scan**

Run:

```bash
rg -n "args\\.(kernel_language|gpu_target|gpu_type|hardware_target|bench_command|eval_command|evaluation_command|verify_command|validation_command|correctness_command|max_iterations|rounds|budget|steps|samples_per_hypothesis)" AKO4X AccelOpt Generalist KernelBand KernelBlaster ARGUS KDA CUDAAgent Astra ReGraphT
```

Expected: No matches except method-specific allowed names after inspection.

### Task 5: Migrate Docs, Manifests, And Catalog Guidance

**Files:**
- Modify: `Agent.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `_manifests/*.yaml`
- Modify: `_meta/manifests/*.yaml`
- Modify: workflow-specific `README.md` and `README.zh-CN.md` files touched by Tasks 3 and 4.

- [ ] **Step 1: Add canonical API guidance to `Agent.md`**

Add this short section after "证据契约":

```markdown
## 参数命名规范

新增或修改 workflow 时，共通入口参数使用统一名称：`kernel_path`、`problem_definition`、`problem_path`、`language`、`target_gpu`、`compile_command`、`test_command`、`benchmark_command`、`iterations`、`seed_candidates`、`exp_dir`。当只提供 `problem_definition` 或 `problem_path` 时，支持生成的 workflow 必须先生成并验证初始 kernel，再进入优化循环。证据契约参数如 `ncu_command`、`invariant_result_path`、`descriptor_result_path`、`archive_update_result_path` 不为统一命名而改写。
```

- [ ] **Step 2: Update README usage guidance**

Add a top-level README section with this example:

```javascript
Workflow({name: '<workflow-name>', args: {
  problem_definition: 'Implement y = gelu(x) for a contiguous fp32 tensor',
  language: 'cuda',
  target_gpu: 'H100',
  test_command: 'python test.py --kernel {kernel_path} --json {result_path}',
  benchmark_command: 'python bench.py --kernel {kernel_path} --json {result_path}',
  iterations: 5,
  seed_candidates: 3,
  exp_dir: '/tmp/kernel_workflow_exp',
}})
```

Also include the existing-kernel variant:

```javascript
Workflow({name: '<workflow-name>', args: {
  kernel_path: '/path/to/kernel.cu',
  target_gpu: 'H100',
  benchmark_command: 'python bench.py --kernel {kernel_path} --json {result_path}',
  iterations: 5,
  exp_dir: '/tmp/kernel_workflow_exp',
}})
```

- [ ] **Step 3: Update manifests**

Every manifest for a workflow that supports generation mode should include the `inputs:` block from Task 2 and use canonical arg names in `args.required` / `args.optional`.

- [ ] **Step 4: Run doc scan**

Run:

```bash
rg -n "gpu_target|gpu_type|gpu_arch|hardware_target|target_hardware|kernel_language|target_language|bench_command|eval_command|evaluation_command|verify_command|validation_command|max_iterations|problem_description|task_spec|operator_spec|task_spec_path|task_path|seed_count|max_seeds" -g '*.md' -g '*.yaml'
```

Expected: No matches except migration design/plan docs and deliberate method-specific terms documented as exceptions.

### Task 6: Final Validation And Commit

**Files:**
- No new files beyond modified source/docs.

- [ ] **Step 1: Run canonical checker**

Run: `node scripts/check-canonical-args.js`

Expected: PASS with `canonical args ok`.

- [ ] **Step 2: Run fidelity checker**

Run: `node scripts/check-fidelity-contracts.js`

Expected: PASS with `fidelity contracts ok (...)`.

- [ ] **Step 3: Run workflow syntax scan**

Run:

```bash
node - <<'NODE'
const fs = require('fs')
const cp = require('child_process')
const files = cp.execSync('rg --files -g "*.js"', {encoding: 'utf8'}).trim().split(/\n/).filter(Boolean)
let failures = []
for (const file of files) {
  const src = fs.readFileSync(file, 'utf8')
  const wrapped = `(async () => {\n${src}\n})`
  try {
    new Function(wrapped)
  } catch (error) {
    failures.push(`${file}: ${error.message}`)
  }
}
if (failures.length) {
  console.error(failures.join('\n'))
  process.exit(1)
}
console.log(`workflow syntax ok (${files.length} files)`)
NODE
```

Expected: PASS with `workflow syntax ok`.

- [ ] **Step 4: Review diff**

Run: `git diff --stat && git diff --check`

Expected: scoped source/docs changes and no whitespace errors.

- [ ] **Step 5: Commit migration**

Run:

```bash
git add Agent.md README.md README.zh-CN.md scripts/check-canonical-args.js _manifests _meta _templates _tools AKO4X AccelOpt Generalist KernelBand KernelBlaster ARGUS KDA CUDAAgent Astra ReGraphT KernelAgent CUDALLM KernelFoundry KernelFoundryDx KernelSkill KSearch AdaExplore
git commit -m "refactor: canonicalize workflow arguments"
```

Expected: commit succeeds.

## Self-Review

- Spec coverage: The plan covers canonical naming, problem-definition entry, generation-mode artifacts, schema/templates, docs/manifests, and verification.
- Placeholder scan: The plan uses concrete commands and code snippets. Ambiguous old names such as `reference_path` are explicitly scoped by meaning.
- Type consistency: The canonical names match the spec: `kernel_path`, `problem_definition`, `problem_path`, `language`, `target_gpu`, `test_command`, `benchmark_command`, `iterations`, `seed_candidates`, and `exp_dir`.
