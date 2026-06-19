// DEPRECATED MIRROR — see _meta/templates/search-based.js for the authoritative v1.1+ template.
// This file is preserved for back-compat against tools still hard-coding the _templates/ path.
// Do not edit. Spec §2.3.
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

// --- BEGIN model-tier (auto-inserted by scripts/patch-model-tier.js) ---
// Tier-based model routing: mechanical steps (run substrate scripts, parse
// JSON) use cheaper models; profile steps (run eval/ncu) use mid-tier;
// judgment steps (plan/implement/report) use the top tier. Tuneable via
// args.model_{mechanical,profile,judgment}.
const MODEL = {
  mechanical: (typeof args !== 'undefined' && args && args.model_mechanical) || 'haiku',
  profile: (typeof args !== 'undefined' && args && args.model_profile) || 'sonnet',
  judgment: (typeof args !== 'undefined' && args && args.model_judgment) || 'opus',
}
// __modelTierApplied
// --- END model-tier ---

const WORKFLOW_NAME = '{{META_NAME}}'


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
// --- BEGIN genome-report (auto-inserted by scripts/patch-genome-report.js) ---
// Self-reported, work-plane (forgeable) stage trace for observability + the
// recombiner. NOT a trust anchor — see _meta/genome-trajectory-schema.md.
async function __genomeReport(phaseName, wfName) {
  try {
    const __dir = (typeof args !== 'undefined' && args && args.exp_dir) ? args.exp_dir : '.'
    await agent(
      'Append exactly one line to ' + __dir + '/genome.jsonl (create it if missing; use a shell append: printf %s\\n ... >> file). ' +
      'The line must be this JSON on ONE line: {"workflow":"' + wfName + '","phase":"' + phaseName + '","ts":"<UTC>","status":"entered"}. ' +
      'Produce <UTC> by running: date -u +%Y-%m-%dT%H:%M:%SZ . Do nothing else; modify no other file. Echo the exact line you appended.',
      { label: 'genome:' + phaseName, phase: phaseName }
    )
  } catch (__e) { /* observability must never break the workflow */ }
}
// --- END genome-report ---

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

// --- State ---
{{STATE_VARIABLES}}
let searchHistory = []
let bestConfig = null
let bestMetric = null

// =============================================================================
// Phase: Setup — Analyze target and define search space
// =============================================================================
phase('Setup'); await __genomeReport('Setup', WORKFLOW_NAME)

{{SETUP_AGENTS}}

// =============================================================================
// Phase: Define Search Space
// =============================================================================
phase('Define'); await __genomeReport('Define', WORKFLOW_NAME)

const searchSpace = await agent(`{{SEARCH_SPACE_DESC}}`, {
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
})

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
  phase('Sample'); await __genomeReport('Sample', WORKFLOW_NAME)

  const candidates = await agent(`{{SAMPLE_PROMPT}}

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
  })

  const configs = candidates.configurations || []
  log(`Sampled ${configs.length} candidates`)

  // ===========================================================================
  // Phase: Evaluate — Measure each candidate
  // ===========================================================================
  phase('Evaluate'); await __genomeReport('Evaluate', WORKFLOW_NAME)

  const evaluations = await parallel(
    configs.map((config, idx) => () =>
      agent(`{{EVALUATE_PROMPT}}

# Configuration ${idx + 1}/${configs.length}:
${JSON.stringify(config)}`, {
        label: `eval-${round}-${idx}`,
        phase: 'Evaluate',
        schema: {{EVALUATE_SCHEMA}},
      })
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
  phase('Prune'); await __genomeReport('Prune', WORKFLOW_NAME)

  //[BLOCK:search_space_shrink]
  const refinement = await agent(`{{REFINE_PROMPT}}

# Search History Summary:
- Total evaluated: ${searchHistory.length}
- Best metric: ${bestMetric}
- Best config: ${JSON.stringify(bestConfig)}
- Recent results: ${JSON.stringify(searchHistory.slice(-{{POPULATION_SIZE_VAR}}))}

Analyze patterns in good vs bad configurations. Suggest how to narrow the search space.`, {
    label: `refine-${round}`,
    phase: 'Prune',
    schema: {{REFINE_SCHEMA}},
  })
  //[/BLOCK:search_space_shrink]

  log(`Round ${round + 1} done. History: ${searchHistory.length} configs evaluated.`)
}

// =============================================================================
// Final Report
// =============================================================================
phase('Report'); await __genomeReport('Report', WORKFLOW_NAME)

const finalReport = await agent(`{{REPORT_PROMPT}}

# Search Results:
- Rounds: ${{{BUDGET_VAR}}}
- Total configs evaluated: ${searchHistory.length}
- Best metric: ${bestMetric}
- Best config: ${JSON.stringify(bestConfig)}

# Top 5 configurations:
${JSON.stringify(searchHistory.filter(h => h.is_valid).sort((a, b) => {{HIGHER_IS_BETTER}} ? b.metric - a.metric : a.metric - b.metric).slice(0, 5))}`, {
  label: 'final-report',
  phase: 'Report',
})

return {
  {{RETURN_OBJECT}}
}
