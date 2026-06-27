// =============================================================================
// TEMPLATE: Iterative Self-Improving Kernel Optimization
// =============================================================================
//
// This is a PARAMETERIZED TEMPLATE, not a runnable workflow.
// The generate-workflow meta-workflow fills {{TOKENS}} and activates [BLOCK]s
// based on the manifest to produce a complete .js workflow.
//
// Topology: iterative (Plan → Execute → Evaluate → Learn → Repeat)
//
// Token reference:
//   {{META_NAME}}              — workflow.name from manifest
//   {{META_DESCRIPTION}}       — workflow.description
//   {{META_WHEN_TO_USE}}       — workflow.when_to_use
//   {{PHASES_ARRAY}}           — JSON array of {title, detail} from phases[]
//   {{HEADER_COMMENT}}         — Auto-generated args documentation
//   {{SOURCE_CITATION}}        — source.paper_title + source.paper_url
//   {{REQUIRED_ARGS}}          — const declarations for required args
//   {{OPTIONAL_ARGS}}          — const declarations with || defaults for optional args
//   {{STATE_VARIABLES}}        — let declarations for topology.iterative.state_variables
//   {{MAX_ITER_VAR}}           — variable name for max iterations (e.g. ITERATIONS)
//   {{MAX_ITER_ARG}}           — arg name (e.g. "iterations")
//   {{BREADTH_VAR}}            — variable name for plan breadth
//   {{SAMPLES_VAR}}            — variable name for samples per plan
//   {{PLAN_ANGLES_ARRAY}}      — JS array literal of plan angle strings
//   {{FEEDBACK_TOOL}}          — profiling tool name (ncu, perf, etc.)
//   {{SETUP_AGENTS}}           — complete agent() calls for Setup phase
//   {{PLAN_PROMPT_BASE}}       — base prompt text for planners
//   {{PLAN_SCHEMA}}            — JSON schema for plan agent output
//   {{EXECUTE_PROMPT}}         — prompt text for implementation agents
//   {{EXECUTE_SCHEMA}}         — JSON schema for implementation output
//   {{EVALUATE_PROMPT}}        — prompt text for evaluation agents
//   {{EVALUATE_SCHEMA}}        — JSON schema for evaluation output
//   {{LEARN_PROMPT}}           — prompt text for learn/summarize agents
//   {{LEARN_SCHEMA}}           — JSON schema for learn output
//   {{REPORT_PROMPT}}          — prompt text for final report agent
//   {{RETURN_OBJECT}}          — fields of the return {} statement
//   {{USE_DRIVER}}             — `Boolean(args.backend_dir)` gate (v1.1)
//   {{DRIVER_DISPATCH_BLOCK}}  — §6.1 driverSh / driverPy path helpers (v1.1)
//
// Block reference (conditional sections):
//   [BLOCK:experience_memory]  — Learn phase with experienceMemory accumulation
//   [BLOCK:ncu_profiling]      — NCU-specific setup and metric extraction
//   [BLOCK:custom_profiling]   — Generic profiling tool integration
//   [BLOCK:early_stopping]     — Convergence-based early termination
//
// =============================================================================

export const meta = {
  name: '{{META_NAME}}',
  description: '{{META_DESCRIPTION}}',
  whenToUse: '{{META_WHEN_TO_USE}}',
  phases: {{PHASES_ARRAY}},
}

// --- BEGIN inlined arg_guard (Workflow runtime parses scripts as bare scripts,
//                              not ES modules; static imports are rejected) ---
function __unwrapArgs(rawArgs) {
  if (rawArgs == null) return {}
  if (typeof rawArgs === 'object' && !Array.isArray(rawArgs)) return rawArgs
  if (typeof rawArgs === 'string') {
    const trimmed = rawArgs.trim()
    if (trimmed === '') return {}
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        const parsed = JSON.parse(trimmed)
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed
        throw new Error('arg_guard: parsed JSON value is not a plain object')
      } catch (e) { throw new Error(`arg_guard: invalid JSON args: ${e.message}`) }
    }
    const out = {}
    const re = /(\w[\w.-]*)=("(?:\\\\\"|[^"])*"|\'(?:\\\\\'|[^\'])*\'|\S+)/g
    let m
    while ((m = re.exec(trimmed)) !== null) {
      let v = m[2]
      if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
        v = v.slice(1, -1)
      }
      out[m[1]] = v
    }
    if (Object.keys(out).length === 0) {
      throw new Error(`arg_guard: workflow args is a non-empty string but contains no key=value pairs and is not JSON. First 160 chars: ${trimmed.slice(0, 160)}`)
    }
    return out
  }
  throw new Error(`arg_guard: workflow args has unexpected type: ${typeof rawArgs}`)
}
// eslint-disable-next-line no-global-assign
args = __unwrapArgs(typeof args === 'undefined' ? undefined : args)
// --- END inlined arg_guard ---

// --- BEGIN inlined agent-retry scaffolding (from _meta/scaffolding/agent-retry.js) ---
async function agentRetry(fn, opts) {
  const retries = (opts && opts.retries != null) ? opts.retries : 5
  let lastError = null
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const result = await fn()
      if (result != null) return result
      // null = agent skipped mid-run OR terminal subagent failure (e.g. transient 429) — retry.
    } catch (e) {
      lastError = e
    }
  }
  if (lastError) throw lastError
  // All attempts returned null (agent skipped mid-run OR a terminal subagent
  // failure such as a sustained 429). FAIL-SAFE DEFAULT: throw an attributable
  // error instead of returning null. A null return would later hit an unguarded
  // deref (`diag.bottleneck_class`, `impl.code`, ...) and crash the run with a
  // cryptic TypeError — issue #20. Throwing here makes the round abort cleanly
  // with a recorded reason, and inside `parallel()` a throwing thunk simply
  // resolves to a null slot that `.filter(Boolean)` drops (graceful). Callers
  // that INTENTIONALLY degrade on a missing result opt out with `{ allowNull: true }`.
  if (opts && opts.allowNull === true) return null
  throw new Error(
    `agentRetry: "${(opts && opts.label) || 'agent'}" returned null after ${retries + 1} attempt(s) ` +
    `(agent skipped or terminal API failure after retries).`,
  )
}

/**
 * Null-guard a REQUIRED structured field. Throws a clear, attributable error
 * (instead of a cryptic TypeError) when an agent returned null/malformed output,
 * so the run fails loudly at the dereference rather than producing garbage.
 */
function expect(obj, field, ctx) {
  if (obj == null || obj[field] == null) {
    throw new Error(
      `agentRetry: required field "${field}" is missing${ctx ? ' from ' + ctx : ''} ` +
      `(agent returned null or a malformed result after retries).`,
    )
  }
  return obj[field]
}

/**
 * Null-guard an OPTIONAL structured field with a fallback (no throw).
 * Use for deref points that have a sensible default (e.g. `[]`, `''`, `0`).
 */
function guard(obj, field, fallback) {
  if (obj == null || obj[field] == null) return fallback
  return obj[field]
}
// --- END inlined agent-retry scaffolding ---

// =============================================================================
// {{META_NAME}}
// =============================================================================
//
// Source: {{SOURCE_CITATION}}
//
// {{HEADER_COMMENT}}
//
// =============================================================================

// --- Required Args ---
{{REQUIRED_ARGS}}

// --- Optional Args ---
{{OPTIONAL_ARGS}}

// Canonical input policy:
// - If args.kernel_path is provided, optimize that existing kernel.
// - Else require args.problem_definition or args.problem_path, generate seed_candidates initial kernels,
//   verify them with test_command or benchmark_command, and optimize the best verified seed.
// - Do not hardcode evaluator/compiler/profiler commands; consume user-provided command args.
// - Return input_mode, generated_kernel_path, initial_candidates, and initial_generation_result.
// - Backend (v1.1): the workflow body never names a vendor profiler (`nvcc`/`ncu`),
//   a vendor metric (`sm_throughput_pct`/`dram_throughput_pct`), or a vendor idiom
//   (`__global__`/`@triton.jit`/`PYBIND11_MODULE`). All such tokens come from the
//   driver's `idioms.json` via the load-driver Setup agent (see spec §6.1/§6.2).
// - Backend (v1.1): args.backend_dir gates the driver path; when empty the body
//   falls back to the legacy inline-prompt path (USE_DRIVER = Boolean(BACKEND_DIR)).

// --- State ---
{{STATE_VARIABLES}}

// =============================================================================
// Phase: Setup
// =============================================================================
phase('Setup')

{{SETUP_AGENTS}}

// =============================================================================
// Iterative Self-Improvement Loop
// =============================================================================

for (let iter = 0; iter < {{MAX_ITER_VAR}}; iter++) {
  log(`\n=== Iteration ${iter + 1}/${{{MAX_ITER_VAR}}} ===`)

  //[BLOCK:early_stopping]
  // Early stopping check
  // if (convergence_condition) { log('Converged.'); break }
  //[/BLOCK:early_stopping]

  // ===========================================================================
  // Phase: Plan — Generate optimization plans
  // ===========================================================================
  phase('Plan')

  //[BLOCK:experience_memory]
  const experienceSection = experienceMemory.length > 0
    ? `\n\n# Learned Optimization Patterns (from previous iterations)\n${experienceMemory.map((e, i) => `${i + 1}. ${e}`).join('\n')}`
    : ''
  //[/BLOCK:experience_memory]

  const planAngles = {{PLAN_ANGLES_ARRAY}}

  const plans = await parallel(
    Array.from({length: {{BREADTH_VAR}}}, (_, i) => () =>
      agentRetry(() => agent(`{{PLAN_PROMPT_BASE}}

# YOUR FOCUS AREA: ${planAngles[i % planAngles.length]}`, {
        label: `plan-${iter}-${i}`,
        phase: 'Plan',
        schema: {{PLAN_SCHEMA}},
      }), { retries: 5 })
    )
  )

  const validPlans = plans.filter(Boolean)
  log(`Plans: ${validPlans.length} generated`)

  // ===========================================================================
  // Phase: Execute — Implement each plan
  // ===========================================================================
  phase('Execute')

  const implementations = await pipeline(
    validPlans,
    (plan) => parallel(
      Array.from({length: {{SAMPLES_VAR}}}, (_, sampleIdx) => () =>
        agentRetry(() => agent(`{{EXECUTE_PROMPT}}

# Plan: "${plan.title}"
Details: ${plan.plan}`, {
          label: `impl-${iter}-${plan.title.substring(0, 15)}-v${sampleIdx}`,
          phase: 'Execute',
          schema: {{EXECUTE_SCHEMA}},
        }), { retries: 5 })
      )
    )
  )

  // Flatten implementations into variants array
  const allVariants = []
  for (let planIdx = 0; planIdx < validPlans.length; planIdx++) {
    const planImpls = implementations[planIdx]
    if (!planImpls) continue
    for (let sIdx = 0; sIdx < planImpls.length; sIdx++) {
      const impl = planImpls[sIdx]
      if (impl && impl.code) {
        allVariants.push({
          plan: validPlans[planIdx],
          code: impl.code,
          id: `plan_${planIdx}_sample_${sIdx}`,
        })
      }
    }
  }

  log(`Generated ${allVariants.length} kernel variants`)

  // ===========================================================================
  // Phase: Evaluate — Profile and measure each variant
  // ===========================================================================
  phase('Evaluate')

  const evaluations = await parallel(
    allVariants.map((variant, varIdx) => () =>
      agentRetry(() => agent(`{{EVALUATE_PROMPT}}

# Variant: ${variant.id} — Plan: "${variant.plan.title}"
# Code:
\`\`\`
${variant.code.substring(0, 4000)}
\`\`\``, {
        label: `eval-${variant.id}`,
        phase: 'Evaluate',
        schema: {{EVALUATE_SCHEMA}},
      }), { retries: 5 })
    )
  )

  // Process results
  const results = []
  for (let i = 0; i < allVariants.length; i++) {
    const evalResult = evaluations[i]
    if (!evalResult) continue
    results.push({
      variant: allVariants[i],
      evaluation: evalResult,
      speedup: evalResult.estimated_speedup || 1.0,
    })
  }

  results.sort((a, b) => b.speedup - a.speedup)

  const improved = results.filter(r => r.speedup > 1.0 && r.evaluation.is_correct && r.evaluation.is_compilable)
  const degraded = results.filter(r => r.speedup < 1.0 && r.evaluation.is_correct && r.evaluation.is_compilable)

  log(`Results: ${improved.length} improved, ${degraded.length} degraded`)

  // Update best if improved
  if (improved.length > 0) {
    const best = improved[0]
    // {{UPDATE_BEST_LOGIC}}
    log(`NEW BEST: "${best.variant.plan.title}" — ${best.speedup.toFixed(2)}x`)
  }

  //[BLOCK:experience_memory]
  // ===========================================================================
  // Phase: Learn — Extract insights from slow-fast pairs
  // ===========================================================================
  phase('Learn')

  const pairsToSummarize = []

  for (const r of improved.slice(0, 3)) {
    pairsToSummarize.push({
      slow: bestKernelCode, // previous best (before this iteration's update)
      fast: r.variant.code,
      speedup: r.speedup,
      plan_title: r.variant.plan.title,
      type: 'positive',
    })
  }

  for (const r of degraded.slice(0, 2)) {
    pairsToSummarize.push({
      slow: r.variant.code,
      fast: bestKernelCode,
      speedup: 1.0 / r.speedup,
      plan_title: r.variant.plan.title + ' [ANTI-PATTERN]',
      type: 'negative',
    })
  }

  if (pairsToSummarize.length > 0) {
    const summaries = await parallel(
      pairsToSummarize.map((pair) => () =>
        agentRetry(() => agent(`{{LEARN_PROMPT}}

# Slow Kernel:
\`\`\`
${pair.slow.substring(0, 2500)}
\`\`\`

# Fast Kernel:
\`\`\`
${pair.fast.substring(0, 2500)}
\`\`\`

# Speedup: ${pair.speedup.toFixed(2)}x
# Type: ${pair.type === 'positive' ? 'POSITIVE (do this)' : 'NEGATIVE (avoid this)'}`, {
          label: `learn-${pair.plan_title.substring(0, 20)}`,
          phase: 'Learn',
          schema: {{LEARN_SCHEMA}},
        }), { retries: 5 })
      )
    )

    for (const s of summaries.filter(Boolean)) {
      experienceMemory.push(s.summary)
    }
    log(`Learned ${summaries.filter(Boolean).length} patterns. Bank: ${experienceMemory.length}`)
  }
  //[/BLOCK:experience_memory]

  phase('Iterate')
  log(`Iteration ${iter + 1} done.`)
}

// =============================================================================
// Final Report
// =============================================================================
const finalReport = await agentRetry(() => agent(`{{REPORT_PROMPT}}`, {
  label: 'final-report',
  phase: 'Iterate',
}), { retries: 5 })

return {
  {{RETURN_OBJECT}}
}
