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
phase('Setup')

{{SETUP_AGENTS}}

// =============================================================================
// Phase: Initialize — Build the search tree / world model
// =============================================================================
phase('Initialize')

const initResult = await agent(`{{INIT_TREE_PROMPT}}`, {
  label: 'init-tree',
  phase: 'Initialize',
  schema: {{INIT_TREE_SCHEMA}},
})

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
  phase('Select')

  const selection = await agent(`{{SELECT_PROMPT}}

# Current Decision Tree:
${JSON.stringify(decisionTree, null, 2).substring(0, 6000)}

# Best metric so far: ${bestMetric || 'none'}
# Solutions found: ${solutionDb.length}
# Cycle: ${cycle + 1}/${{{MAX_CYCLES_VAR}}}`, {
    label: `select-${cycle}`,
    phase: 'Select',
    schema: {{SELECT_SCHEMA}},
  })

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
  phase('Generate')

  //[BLOCK:multi_attempt]
  let cycleBestCode = null
  let cycleBestEval = null
  let noImproveStreak = 0

  for (let attempt = 0; attempt < {{ATTEMPTS_PER_CYCLE_VAR}}; attempt++) {
    const isFirstAttempt = attempt === 0

    const genResult = isFirstAttempt
      ? await agent(`{{GENERATE_PROMPT}}

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
        })
      : await agent(`{{IMPROVE_PROMPT}}

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
        })

    if (!genResult || !genResult.code) continue

    // =========================================================================
    // Phase: Evaluate — Measure the generated variant
    // =========================================================================
    phase('Evaluate')

    const evalResult = await agent(`{{EVALUATE_PROMPT}}

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
    })

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
  phase('Refine')

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
    const refineResult = await agent(`{{REFINE_PROMPT}}

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
    })

    if (refineResult && refineResult.updated_tree) {
      decisionTree = refineResult.updated_tree
    }
  } else {
    // Backtrack — downgrade this action node, propose alternatives
    const backtrackResult = await agent(`{{BACKTRACK_PROMPT}}

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
    })

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
phase('Report')

const finalReport = await agent(`{{REPORT_PROMPT}}

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
})

return {
  {{RETURN_OBJECT}}
}
