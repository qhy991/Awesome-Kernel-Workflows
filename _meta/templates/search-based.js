// =============================================================================
// TEMPLATE: Search-Based Kernel Optimization (Autotuning / Evolutionary)
// =============================================================================
//
// This is a PARAMETERIZED TEMPLATE for search-based optimization methods.
// Topology: search (Sample → Evaluate → Prune → Refine → Repeat)
//
// Suitable for: autotuning, evolutionary search, Bayesian optimization, grid search
//
// Token reference:
//   {{META_NAME}}              — workflow.name from manifest
//   {{META_DESCRIPTION}}       — workflow.description
//   {{META_WHEN_TO_USE}}       — workflow.when_to_use
//   {{PHASES_ARRAY}}           — JSON array of {title, detail}
//   {{HEADER_COMMENT}}         — Auto-generated args documentation
//   {{SOURCE_CITATION}}        — source.paper_title + source.paper_url
//   {{REQUIRED_ARGS}}          — const declarations for required args
//   {{OPTIONAL_ARGS}}          — const declarations with defaults
//   {{STATE_VARIABLES}}        — let declarations for state
//   {{BUDGET_VAR}}             — variable name for search budget
//   {{POPULATION_SIZE_VAR}}    — variable name for population size
//   {{SEARCH_SPACE_DESC}}      — description of the search space
//   {{SAMPLER_TYPE}}           — random | bayesian | evolutionary | grid
//   {{SETUP_AGENTS}}           — agent calls for Setup phase
//   {{SAMPLE_PROMPT}}          — prompt for sampling agent
//   {{SAMPLE_SCHEMA}}          — schema for sample output
//   {{EVALUATE_PROMPT}}        — prompt for evaluation agent
//   {{EVALUATE_SCHEMA}}        — schema for evaluation output
//   {{PRUNE_PROMPT}}           — prompt for pruning/selection agent
//   {{PRUNE_SCHEMA}}           — schema for prune output
//   {{REFINE_PROMPT}}          — prompt for search space refinement agent
//   {{REFINE_SCHEMA}}          — schema for refine output
//   {{REPORT_PROMPT}}          — prompt for final report
//   {{RETURN_OBJECT}}          — return statement fields
//
// Block reference:
//   [BLOCK:bayesian_surrogate]  — Surrogate model fitting
//   [BLOCK:evolutionary_ops]    — Crossover and mutation operators
//   [BLOCK:early_stopping]      — Budget or convergence stopping
//   [BLOCK:search_space_shrink] — Progressive space reduction
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
let searchHistory = []
let bestConfig = null
let bestMetric = null

// =============================================================================
// Phase: Setup — Analyze target and define search space
// =============================================================================
phase('Setup')

{{SETUP_AGENTS}}

// =============================================================================
// Phase: Define Search Space
// =============================================================================
phase('Define')

const searchSpace = await agentRetry(() => agent(`{{SEARCH_SPACE_DESC}}`, {
  label: 'define-search-space',
  phase: 'Define',
  schema: {
    type: 'object',
    properties: {
      dimensions: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, type: { type: 'string' }, range: { type: 'string' } } } },
      constraints: { type: 'array', items: { type: 'string' } },
      total_space_size: { type: 'string' },
    },
    required: ['dimensions'],
  },
}), { retries: 5 })

log(`Search space: ${searchSpace.dimensions?.length || 0} dimensions`)

// =============================================================================
// Search Loop
// =============================================================================

for (let round = 0; round < {{BUDGET_VAR}}; round++) {
  log(`\n=== Search Round ${round + 1}/${{{BUDGET_VAR}}} | Best: ${bestMetric || 'N/A'} ===`)

  //[BLOCK:early_stopping]
  // Early stopping: if no improvement in last N rounds
  //[/BLOCK:early_stopping]

  // ===========================================================================
  // Phase: Sample — Generate candidate configurations
  // ===========================================================================
  phase('Sample')

  const candidates = await agentRetry(() => agent(`{{SAMPLE_PROMPT}}

# Search Space:
${JSON.stringify(searchSpace.dimensions || [])}

# Search History (last 10):
${JSON.stringify(searchHistory.slice(-10))}

# Current Best: ${JSON.stringify(bestConfig)}
# Best Metric: ${bestMetric}
# Round: ${round + 1}/${{{BUDGET_VAR}}}
# Sampler: {{SAMPLER_TYPE}}

Generate ${{{POPULATION_SIZE_VAR}}} candidate configurations.`, {
    label: `sample-${round}`,
    phase: 'Sample',
    schema: {{SAMPLE_SCHEMA}},
  }), { retries: 5 })

  const configs = candidates.configurations || []
  log(`Sampled ${configs.length} candidates`)

  // ===========================================================================
  // Phase: Evaluate — Measure each candidate
  // ===========================================================================
  phase('Evaluate')

  const evaluations = await parallel(
    configs.map((config, idx) => () =>
      agentRetry(() => agent(`{{EVALUATE_PROMPT}}

# Configuration ${idx + 1}/${configs.length}:
${JSON.stringify(config)}`, {
        label: `eval-${round}-${idx}`,
        phase: 'Evaluate',
        schema: {{EVALUATE_SCHEMA}},
      }), { retries: 5 })
    )
  )

  // Record results
  for (let i = 0; i < configs.length; i++) {
    const evalResult = evaluations[i]
    if (!evalResult) continue
    searchHistory.push({
      config: configs[i],
      metric: evalResult.metric_value,
      round,
      is_valid: evalResult.is_valid,
    })

    // Update best
    const isBetter = evalResult.is_valid && (
      bestMetric === null ||
      ({{HIGHER_IS_BETTER}} ? evalResult.metric_value > bestMetric : evalResult.metric_value < bestMetric)
    )
    if (isBetter) {
      bestConfig = configs[i]
      bestMetric = evalResult.metric_value
      log(`NEW BEST: metric=${bestMetric}`)
    }
  }

  // ===========================================================================
  // Phase: Prune — Select survivors and update search strategy
  // ===========================================================================
  phase('Prune')

  //[BLOCK:search_space_shrink]
  const refinement = await agentRetry(() => agent(`{{REFINE_PROMPT}}

# Search History Summary:
- Total evaluated: ${searchHistory.length}
- Best metric: ${bestMetric}
- Best config: ${JSON.stringify(bestConfig)}
- Recent results: ${JSON.stringify(searchHistory.slice(-{{POPULATION_SIZE_VAR}}))}

Analyze patterns in good vs bad configurations. Suggest how to narrow the search space.`, {
    label: `refine-${round}`,
    phase: 'Prune',
    schema: {{REFINE_SCHEMA}},
  }), { retries: 5 })
  //[/BLOCK:search_space_shrink]

  log(`Round ${round + 1} done. History: ${searchHistory.length} configs evaluated.`)
}

// =============================================================================
// Final Report
// =============================================================================
phase('Report')

const finalReport = await agentRetry(() => agent(`{{REPORT_PROMPT}}

# Search Results:
- Rounds: ${{{BUDGET_VAR}}}
- Total configs evaluated: ${searchHistory.length}
- Best metric: ${bestMetric}
- Best config: ${JSON.stringify(bestConfig)}

# Top 5 configurations:
${JSON.stringify(searchHistory.filter(h => h.is_valid).sort((a, b) => {{HIGHER_IS_BETTER}} ? b.metric - a.metric : a.metric - b.metric).slice(0, 5))}`, {
  label: 'final-report',
  phase: 'Report',
}), { retries: 5 })

return {
  {{RETURN_OBJECT}}
}
