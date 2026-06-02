export const meta = {
  name: 'validate-workflow',
  description: 'Static and semantic validation of a Claude Code kernel optimization workflow',
  whenToUse: 'After generating a new workflow or before adding a workflow to the catalog. Validates meta completeness, phase consistency, agent schema correctness, args documentation, return envelope, and parallelism integrity.',
  phases: [
    { title: 'Parse', detail: 'Load workflow content and extract structural elements' },
    { title: 'Check', detail: 'Run all static checks in parallel' },
    { title: 'Semantic', detail: 'LLM-assisted semantic coherence verification' },
    { title: 'Report', detail: 'Aggregate violations and produce structured report' },
  ],
}

// =============================================================================
// Workflow Validator — Static + Semantic Checks
// =============================================================================
//
// Usage:
//   Workflow({name: 'validate-workflow', args: {
//     workflow_path: '/path/to/workflow.js',  // OR
//     workflow_content: '...',                // inline JS string
//     strict: false,                         // treat warnings as errors
//   }})
//
// Returns:
//   { passed, violation_count, error_count, warning_count, violations[] }
//
// =============================================================================

const WORKFLOW_PATH = args.workflow_path || ''
const WORKFLOW_CONTENT = args.workflow_content || ''
const STRICT = args.strict || false

// =============================================================================
// Phase 1: Parse — Load and extract structural elements
// =============================================================================
phase('Parse')

const parseResult = await agent(`You are a JavaScript static analysis tool. Parse this Claude Code Workflow file and extract its structural elements.

# Task
Read the following workflow source code and extract all structural elements needed for validation.

# Source Code:
\`\`\`javascript
${WORKFLOW_CONTENT || `(Read file at: ${WORKFLOW_PATH})`}
\`\`\`

${WORKFLOW_PATH && !WORKFLOW_CONTENT ? `Read the file at ${WORKFLOW_PATH} first, then analyze it.` : ''}

# Extract:
1. **meta object**: Does it exist? Extract name, description, whenToUse, phases array (each title + detail)
2. **phase() calls**: List every phase('...') call with its string argument
3. **agent() calls**: For each agent() call, extract:
   - The label (from options.label)
   - The phase (from options.phase)
   - The schema (from options.schema) — extract type, properties keys, required array
   - Whether it's wrapped in parallel() or pipeline()
4. **args references**: Every occurrence of args.FIELD_NAME in the code
5. **Header comment**: Is there a multi-line comment block after meta and before the first const?
6. **return statement**: The final return {} — extract its field names
7. **parallel() calls**: For each, is the argument an array of arrow functions?
8. **pipeline() calls**: For each, what are the arguments?
9. **State variables**: All let/var declarations at module scope

Return all findings as structured data.`, {
  label: 'parse-structure',
  phase: 'Parse',
  schema: {
    type: 'object',
    properties: {
      has_meta: { type: 'boolean' },
      meta_name: { type: 'string' },
      meta_description: { type: 'string' },
      meta_when_to_use: { type: 'string' },
      meta_phases: { type: 'array', items: { type: 'object', properties: { title: { type: 'string' }, detail: { type: 'string' } } } },
      phase_calls: { type: 'array', items: { type: 'string' } },
      agent_calls: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            label: { type: 'string' },
            phase: { type: 'string' },
            has_schema: { type: 'boolean' },
            schema_type: { type: 'string' },
            schema_properties: { type: 'array', items: { type: 'string' } },
            schema_required: { type: 'array', items: { type: 'string' } },
            wrapped_in: { type: 'string' },
          },
        },
      },
      args_references: { type: 'array', items: { type: 'string' } },
      has_header_comment: { type: 'boolean' },
      return_fields: { type: 'array', items: { type: 'string' } },
      has_return_statement: { type: 'boolean' },
      parallel_calls_valid: { type: 'boolean' },
      parallel_issues: { type: 'array', items: { type: 'string' } },
      state_variables: { type: 'array', items: { type: 'string' } },
      full_source: { type: 'string' },
    },
    required: ['has_meta', 'phase_calls', 'agent_calls', 'args_references', 'has_return_statement'],
  },
})

log(`Parsed: meta=${parseResult.has_meta}, phases=${parseResult.phase_calls?.length || 0}, agents=${parseResult.agent_calls?.length || 0}`)

// =============================================================================
// Phase 2: Check — Run all static checks in parallel
// =============================================================================
phase('Check')

const checks = await parallel([
  // --- check-meta ---
  () => agent(`You are a static validator. Check the meta object of this workflow for completeness.

# Parsed Structure:
- has_meta: ${parseResult.has_meta}
- meta.name: "${parseResult.meta_name || ''}"
- meta.description: "${parseResult.meta_description || ''}"
- meta.whenToUse: "${parseResult.meta_when_to_use || ''}"
- meta.phases: ${JSON.stringify(parseResult.meta_phases || [])}

# Rules:
1. meta must be exported (has_meta must be true) — ERROR if not
2. meta.name must be non-empty kebab-case string — ERROR if not
3. meta.description must be non-empty string — ERROR if not
4. meta.whenToUse must be non-empty string — ERROR if not (WARNING for missing)
5. meta.phases must be non-empty array — ERROR if not
6. Each phase entry must have both 'title' (non-empty string) and 'detail' (non-empty string) — ERROR for missing

Return violations found.`, {
    label: 'check-meta',
    phase: 'Check',
    schema: {
      type: 'object',
      properties: {
        violations: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              severity: { type: 'string' },
              message: { type: 'string' },
              location: { type: 'string' },
            },
            required: ['severity', 'message'],
          },
        },
      },
      required: ['violations'],
    },
  }),

  // --- check-phase-consistency ---
  () => agent(`You are a static validator. Check phase() call consistency against meta.phases.

# meta.phases titles: ${JSON.stringify((parseResult.meta_phases || []).map(p => p.title))}
# phase() call arguments: ${JSON.stringify(parseResult.phase_calls || [])}

# Rules:
1. Every phase() call argument must match a title in meta.phases — ERROR for orphan phase() calls
2. Every meta.phases title that has no matching phase() call — WARNING (may be structural-only)
3. phase() calls should not contain duplicates unless inside a loop — WARNING for unexpected duplicates

Return violations found.`, {
    label: 'check-phase-consistency',
    phase: 'Check',
    schema: {
      type: 'object',
      properties: {
        violations: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              severity: { type: 'string' },
              message: { type: 'string' },
              location: { type: 'string' },
            },
            required: ['severity', 'message'],
          },
        },
      },
      required: ['violations'],
    },
  }),

  // --- check-agent-schemas ---
  () => agent(`You are a static validator. Check all agent() calls for proper options.

# Agent calls found:
${JSON.stringify(parseResult.agent_calls || [], null, 2)}

# meta.phases titles: ${JSON.stringify((parseResult.meta_phases || []).map(p => p.title))}

# Rules:
1. Every agent() call must have a non-empty 'label' string — ERROR if missing
2. Every agent() call must have a 'phase' string — ERROR if missing
3. Every agent() 'phase' value must match one of the meta.phases titles — ERROR if not
4. If agent has schema, schema.type must be 'object' — ERROR if not
5. Every field in schema.required must exist in schema.properties — ERROR if orphan required field
6. Agent without schema is acceptable (returns free text) — no violation
7. The final agent (report) may lack schema — no violation

Return violations found.`, {
    label: 'check-agent-schemas',
    phase: 'Check',
    schema: {
      type: 'object',
      properties: {
        violations: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              severity: { type: 'string' },
              message: { type: 'string' },
              location: { type: 'string' },
            },
            required: ['severity', 'message'],
          },
        },
      },
      required: ['violations'],
    },
  }),

  // --- check-args-and-return ---
  () => agent(`You are a static validator. Check args documentation and return envelope.

# Args references in code: ${JSON.stringify(parseResult.args_references || [])}
# Has header comment block: ${parseResult.has_header_comment}
# Has return statement: ${parseResult.has_return_statement}
# Return fields: ${JSON.stringify(parseResult.return_fields || [])}

# Rules:
1. Header comment block should exist after meta — WARNING if missing
2. Each args.FIELD_NAME referenced in code should be mentioned in header comment — WARNING for undocumented args
3. File must end with a return {} statement (not inside a for/while loop body only) — ERROR if absent
4. Return object must be non-empty — ERROR if return {}
5. parallel() must receive array of arrow functions (() => ...) — ERROR if direct agent() calls passed

# Parallel call validity: ${parseResult.parallel_calls_valid}
# Parallel issues: ${JSON.stringify(parseResult.parallel_issues || [])}

Return violations found.`, {
    label: 'check-args-return',
    phase: 'Check',
    schema: {
      type: 'object',
      properties: {
        violations: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              severity: { type: 'string' },
              message: { type: 'string' },
              location: { type: 'string' },
            },
            required: ['severity', 'message'],
          },
        },
      },
      required: ['violations'],
    },
  }),
])

const staticViolations = checks
  .filter(Boolean)
  .flatMap(c => c.violations || [])

log(`Static checks: ${staticViolations.length} violations (${staticViolations.filter(v => v.severity === 'error').length} errors, ${staticViolations.filter(v => v.severity === 'warning').length} warnings)`)

// =============================================================================
// Phase 3: Semantic — LLM-assisted coherence check
// =============================================================================
phase('Semantic')

const semanticResult = await agent(`You are an expert code reviewer specializing in Claude Code Workflows. Perform a semantic coherence check on this workflow.

# Full Workflow Structure:
- State variables: ${JSON.stringify(parseResult.state_variables || [])}
- Phase sequence: ${JSON.stringify(parseResult.phase_calls || [])}
- Agent calls: ${(parseResult.agent_calls || []).map(a => a.label).join(', ')}
- Return fields: ${JSON.stringify(parseResult.return_fields || [])}

# Agent Details:
${JSON.stringify(parseResult.agent_calls || [], null, 2)}

# Check for these semantic issues:

1. **Variable reference coherence**: Do later phases reference state variables that are set in earlier phases? Are there variables used but never declared?

2. **Schema-to-usage coherence**: If an agent's schema declares field "X" in required, is "X" actually used in subsequent code? Are there schema fields that are extracted but never referenced?

3. **Data flow completeness**: Does the data flow from Setup → Plan → Execute → Evaluate → Learn make sense? Does each phase have the inputs it needs from previous phases?

4. **Return field coverage**: Are all return_fields actually assigned values by the time return is reached?

5. **Parallelism logic**: Are items passed to parallel() actually independent? Is pipeline() used where items have sequential dependencies?

For each issue found, classify as WARNING (suspicious but might be intentional) or ERROR (definitely broken).

Return violations found. Be conservative — only flag things that are clearly wrong, not just unusual.`, {
  label: 'check-semantic',
  phase: 'Semantic',
  schema: {
    type: 'object',
    properties: {
      violations: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            severity: { type: 'string' },
            message: { type: 'string' },
            location: { type: 'string' },
          },
          required: ['severity', 'message'],
        },
      },
      coherence_summary: { type: 'string' },
    },
    required: ['violations'],
  },
})

const allViolations = [...staticViolations, ...(semanticResult.violations || [])]

// =============================================================================
// Phase 4: Report — Aggregate and return
// =============================================================================
phase('Report')

const errors = allViolations.filter(v => v.severity === 'error')
const warnings = allViolations.filter(v => v.severity === 'warning')
const passed = STRICT ? allViolations.length === 0 : errors.length === 0

log(`Validation ${passed ? 'PASSED' : 'FAILED'}: ${errors.length} errors, ${warnings.length} warnings`)

return {
  passed,
  violation_count: allViolations.length,
  error_count: errors.length,
  warning_count: warnings.length,
  violations: allViolations,
  coherence_summary: semanticResult.coherence_summary || '',
}
