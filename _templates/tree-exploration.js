// DEPRECATED MIRROR — see _meta/templates/tree-exploration.js for the authoritative v1.1+ template.
// This file is preserved for back-compat against tools still hard-coding the _templates/ path.
// Do not edit. Spec §2.3.
// =============================================================================
// TEMPLATE: Tree Exploration Kernel Optimization (World Model / ToT / Beam)
// =============================================================================
//
// This is a PARAMETERIZED TEMPLATE for tree-structured search methods.
// Topology: tree (Init Tree → Select Node → Generate → Evaluate → Refine/Backtrack → Repeat)
//
// Suitable for: co-evolving world models, tree-of-thought, beam search,
//   Monte-Carlo tree search, branch-and-bound kernel optimization
//
// Key difference from iterative-loop:
//   - State is a TREE (not a flat list), with nodes representing design decisions
//   - Selection is utility-based (pick best frontier node), not breadth-parallel
//   - Backtracking: failed nodes are downgraded, search redirects elsewhere
//   - Refinement: successful solutions update ancestor node scores
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
//   {{MAX_CYCLES_VAR}}         — variable for max search cycles
//   {{MAX_DEPTH_VAR}}          — variable for max tree depth
//   {{STAGNATION_WINDOW_VAR}}  — variable for stagnation detection window
//   {{ATTEMPTS_PER_CYCLE_VAR}} — variable for attempts per action cycle
//   {{INIT_TREE_PROMPT}}       — prompt for tree initialization agent
//   {{INIT_TREE_SCHEMA}}       — schema for init output (tree structure)
//   {{SELECT_PROMPT}}          — prompt for action selection agent
//   {{SELECT_SCHEMA}}          — schema for selection output
//   {{GENERATE_PROMPT}}        — prompt for code generation agent
//   {{GENERATE_SCHEMA}}        — schema for generation output
//   {{IMPROVE_PROMPT}}         — prompt for debug/improve agent
//   {{IMPROVE_SCHEMA}}         — schema for improve output
//   {{EVALUATE_PROMPT}}        — prompt for evaluation agent
//   {{EVALUATE_SCHEMA}}        — schema for evaluation output
//   {{REFINE_PROMPT}}          — prompt for tree refinement agent (success path)
//   {{REFINE_SCHEMA}}          — schema for refinement output
//   {{BACKTRACK_PROMPT}}       — prompt for backtrack/downgrade agent (failure path)
//   {{BACKTRACK_SCHEMA}}       — schema for backtrack output
//   {{REPORT_PROMPT}}          — prompt for final report
//   {{RETURN_OBJECT}}          — return statement fields
//
// Block reference:
//   [BLOCK:world_model]        — Full world model JSON management
//   [BLOCK:solution_db]        — Solution lineage tracking
//   [BLOCK:stagnation_detect]  — Stagnation detection and early cycle exit
//   [BLOCK:multi_attempt]      — Multiple generate/improve attempts per cycle
//   [BLOCK:difficulty_filter]  — Action difficulty-based filtering
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
// --- BEGIN genome-report (auto-inserted by scripts/patch-genome-report.js) ---
// Self-reported, work-plane (forgeable) stage trace for observability + the
// recombiner. NOT a trust anchor — see _meta/genome-trajectory-schema.md.
async function __genomeReport(phaseName, wfName) {
  try {
    const __dir = (typeof args !== 'undefined' && args && args.exp_dir) ? args.exp_dir : '.'
    await agentRetry(() => agent(
      'Append exactly one line to ' + __dir + '/genome.jsonl (create it if missing; use a shell append: printf %s\\n ... >> file). ' +
      'The line must be this JSON on ONE line: {"workflow":"' + wfName + '","phase":"' + phaseName + '","ts":"<UTC>","status":"entered"}. ' +
      'Produce <UTC> by running: date -u +%Y-%m-%dT%H:%M:%SZ . Do nothing else; modify no other file. Echo the exact line you appended.',
      { label: 'genome:' + phaseName, phase: phaseName }
    ), { retries: 5 })
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
let decisionTree = null        // The tree structure (nodes with scores, solutions, actions)
let solutionDb = []            // Array of {id, code, eval, parent_node_id}
let bestSolution = null        // Best solution found so far
let bestMetric = null          // Best metric value achieved
let cycleCount = 0             // Number of completed search cycles

// =============================================================================
// Phase: Setup — Read target artifact, establish baseline
// =============================================================================
phase('Setup'); await __genomeReport('Setup', WORKFLOW_NAME)

{{SETUP_AGENTS}}

// =============================================================================
// Phase: Initialize — Build the search tree / world model
// =============================================================================
phase('Initialize'); await __genomeReport('Initialize', WORKFLOW_NAME)

const initResult = await agentRetry(() => agent(`{{INIT_TREE_PROMPT}}`, {
  label: 'init-tree',
  phase: 'Initialize',
  schema: {{INIT_TREE_SCHEMA}},
}), { retries: 5 })

decisionTree = initResult.decision_tree || initResult
log(`Tree initialized: ${initResult.node_count || 'unknown'} nodes, ${initResult.open_actions || 'unknown'} open actions`)

// =============================================================================
// Search Cycles — Select → Generate → Evaluate → Refine/Backtrack
// =============================================================================

for (let cycle = 0; cycle < {{MAX_CYCLES_VAR}}; cycle++) {
  log(`\n=== Cycle ${cycle + 1}/${{{MAX_CYCLES_VAR}}} | Best: ${bestMetric || 'N/A'} | Nodes explored: ${solutionDb.length} ===`)

  // ===========================================================================
  // Phase: Select — Choose the best frontier action node
  // ===========================================================================
  phase('Select'); await __genomeReport('Select', WORKFLOW_NAME)

  const selection = await agentRetry(() => agent(`{{SELECT_PROMPT}}

# Current Decision Tree:
${JSON.stringify(decisionTree, null, 2).substring(0, 6000)}

# Best metric so far: ${bestMetric || 'none'}
# Solutions found: ${solutionDb.length}
# Cycle: ${cycle + 1}/${{{MAX_CYCLES_VAR}}}`, {
    label: `select-${cycle}`,
    phase: 'Select',
    schema: {{SELECT_SCHEMA}},
  }), { retries: 5 })

  if (!selection || !selection.selected_node_id) {
    log('No viable action node found — search exhausted.')
    break
  }

  log(`Selected: node=${selection.selected_node_id} — "${selection.action_title || 'unknown'}" (score=${selection.action_score || '?'})`)

  const activeNodeId = selection.selected_node_id
  const actionDescription = selection.action_description || ''
  const parentCode = selection.parent_solution_code || ''

  // ===========================================================================
  // Phase: Generate — Create kernel implementation for the selected action
  // ===========================================================================
  phase('Generate'); await __genomeReport('Generate', WORKFLOW_NAME)

  //[BLOCK:multi_attempt]
  let cycleBestCode = null
  let cycleBestEval = null
  let noImproveStreak = 0

  for (let attempt = 0; attempt < {{ATTEMPTS_PER_CYCLE_VAR}}; attempt++) {
    const isFirstAttempt = attempt === 0

    const genResult = isFirstAttempt
      ? await agentRetry(() => agent(`{{GENERATE_PROMPT}}

# Action to implement: "${selection.action_title || ''}"
${actionDescription}

# Base code (from parent node):
\`\`\`
${parentCode.substring(0, 5000)}
\`\`\`

# Decision tree context:
${JSON.stringify(selection.context_for_generation || {}).substring(0, 2000)}`, {
          label: `gen-${cycle}-${attempt}`,
          phase: 'Generate',
          schema: {{GENERATE_SCHEMA}},
        }), { retries: 5 })
      : await agentRetry(() => agent(`{{IMPROVE_PROMPT}}

# Action: "${selection.action_title || ''}"
${actionDescription}

# Current code (attempt ${attempt}):
\`\`\`
${(cycleBestCode || parentCode).substring(0, 5000)}
\`\`\`

# Previous evaluation:
${JSON.stringify(cycleBestEval || {}).substring(0, 2000)}

# Improvement target: beat metric ${bestMetric || 'baseline'}`, {
          label: `improve-${cycle}-${attempt}`,
          phase: 'Generate',
          schema: {{IMPROVE_SCHEMA}},
        }), { retries: 5 })

    if (!genResult || !genResult.code) continue

    // =========================================================================
    // Phase: Evaluate — Measure the generated variant
    // =========================================================================
    phase('Evaluate'); await __genomeReport('Evaluate', WORKFLOW_NAME)

    const evalResult = await agentRetry(() => agent(`{{EVALUATE_PROMPT}}

# Code to evaluate:
\`\`\`
${genResult.code.substring(0, 5000)}
\`\`\`

# Action: "${selection.action_title || ''}"
# Current best metric: ${bestMetric || 'N/A'}
# Parent metric: ${selection.parent_metric || 'N/A'}`, {
      label: `eval-${cycle}-${attempt}`,
      phase: 'Evaluate',
      schema: {{EVALUATE_SCHEMA}},
    }), { retries: 5 })

    if (!evalResult) continue

    // Track this solution
    const solution = {
      id: `cycle_${cycle}_attempt_${attempt}`,
      code: genResult.code,
      eval: evalResult,
      node_id: activeNodeId,
    }
    solutionDb.push(solution)

    // Update cycle best
    const isCycleBetter = evalResult.is_valid && (
      !cycleBestEval ||
      ({{HIGHER_IS_BETTER}} ? evalResult.metric_value > (cycleBestEval.metric_value || 0) : evalResult.metric_value < (cycleBestEval.metric_value || Infinity))
    )
    if (isCycleBetter) {
      cycleBestCode = genResult.code
      cycleBestEval = evalResult
      noImproveStreak = 0
    } else {
      noImproveStreak++
    }

    //[BLOCK:stagnation_detect]
    if (noImproveStreak >= {{STAGNATION_WINDOW_VAR}}) {
      log(`Stagnation detected after ${attempt + 1} attempts — ending cycle early`)
      break
    }
    //[/BLOCK:stagnation_detect]
  }
  //[/BLOCK:multi_attempt]

  // ===========================================================================
  // Phase: Refine or Backtrack — Update the tree based on cycle outcome
  // ===========================================================================
  phase('Refine'); await __genomeReport('Refine', WORKFLOW_NAME)

  const cycleSucceeded = cycleBestEval && cycleBestEval.is_valid

  if (cycleSucceeded) {
    // Update global best
    const isGlobalBetter = bestMetric === null ||
      ({{HIGHER_IS_BETTER}} ? cycleBestEval.metric_value > bestMetric : cycleBestEval.metric_value < bestMetric)
    if (isGlobalBetter) {
      bestMetric = cycleBestEval.metric_value
      bestSolution = { code: cycleBestCode, eval: cycleBestEval, node_id: activeNodeId }
      log(`NEW GLOBAL BEST: metric=${bestMetric}`)
    }

    // Refine the tree — attach solution, update scores, add continuation nodes
    const refineResult = await agentRetry(() => agent(`{{REFINE_PROMPT}}

# Cycle outcome: SUCCESS
# Node: ${activeNodeId}
# Achieved metric: ${cycleBestEval.metric_value}
# Global best: ${bestMetric}
# Code summary: ${cycleBestEval.performance_analysis || ''}

# Current tree:
${JSON.stringify(decisionTree, null, 2).substring(0, 5000)}

Tasks:
1. Attach the solution to node ${activeNodeId}
2. Update scores of ancestor nodes based on this result
3. Add 2-3 continuation child nodes (next optimization steps from this point)
4. Return the updated tree`, {
      label: `refine-${cycle}`,
      phase: 'Refine',
      schema: {{REFINE_SCHEMA}},
    }), { retries: 5 })

    if (refineResult && refineResult.updated_tree) {
      decisionTree = refineResult.updated_tree
    }
  } else {
    // Backtrack — downgrade this action node, propose alternatives
    const backtrackResult = await agentRetry(() => agent(`{{BACKTRACK_PROMPT}}

# Cycle outcome: FAILED (no valid solution produced)
# Node: ${activeNodeId}
# Action attempted: "${selection.action_title || ''}"
# Best attempt eval: ${JSON.stringify(cycleBestEval || {}).substring(0, 1000)}

# Current tree:
${JSON.stringify(decisionTree, null, 2).substring(0, 5000)}

Tasks:
1. Downgrade node ${activeNodeId} — reduce its score and increase difficulty rating
2. Add a note explaining why this action failed
3. Optionally propose 1-2 alternative action nodes that avoid this failure mode
4. Return the updated tree`, {
      label: `backtrack-${cycle}`,
      phase: 'Refine',
      schema: {{BACKTRACK_SCHEMA}},
    }), { retries: 5 })

    if (backtrackResult && backtrackResult.updated_tree) {
      decisionTree = backtrackResult.updated_tree
    }

    log(`Backtracked: node ${activeNodeId} downgraded`)
  }

  cycleCount++
  log(`Cycle ${cycle + 1} ${cycleSucceeded ? 'succeeded' : 'failed'}. Tree updated.`)
}

// =============================================================================
// Final Report
// =============================================================================
phase('Report'); await __genomeReport('Report', WORKFLOW_NAME)

const finalReport = await agentRetry(() => agent(`{{REPORT_PROMPT}}

# Search Results:
- Cycles completed: ${cycleCount}
- Solutions evaluated: ${solutionDb.length}
- Best metric: ${bestMetric}
- Best solution node: ${bestSolution?.node_id || 'none'}

# Final Decision Tree:
${JSON.stringify(decisionTree, null, 2).substring(0, 4000)}

# Solution Lineage (top 5):
${JSON.stringify(solutionDb.filter(s => s.eval?.is_valid).sort((a, b) => {{HIGHER_IS_BETTER}} ? (b.eval.metric_value - a.eval.metric_value) : (a.eval.metric_value - b.eval.metric_value)).slice(0, 5).map(s => ({id: s.id, node: s.node_id, metric: s.eval.metric_value})))}`, {
  label: 'final-report',
  phase: 'Report',
}), { retries: 5 })

return {
  {{RETURN_OBJECT}}
}
