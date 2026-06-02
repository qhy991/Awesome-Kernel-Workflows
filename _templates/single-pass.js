// =============================================================================
// TEMPLATE: Single-Pass Pipeline Kernel Optimization
// =============================================================================
//
// This is a PARAMETERIZED TEMPLATE for single-pass transformation methods.
// Topology: pipeline (Analyze → Transform → Verify)
//
// Suitable for: compiler-style passes, rule-based optimization, static transforms
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
//   {{PASS_ORDER}}             — ordered array of pass names
//   {{ANALYZE_PROMPT}}         — prompt for analysis agent
//   {{ANALYZE_SCHEMA}}         — schema for analysis output
//   {{TRANSFORM_PROMPT}}       — prompt for each transform pass agent
//   {{TRANSFORM_SCHEMA}}       — schema for transform output
//   {{VERIFY_PROMPT}}          — prompt for verification agent
//   {{VERIFY_SCHEMA}}          — schema for verification output
//   {{REPORT_PROMPT}}          — prompt for final report
//   {{RETURN_OBJECT}}          — return statement fields
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

// =============================================================================
// Phase: Analyze — Identify transformation opportunities
// =============================================================================
phase('Analyze')

const analysis = await agent(`{{ANALYZE_PROMPT}}`, {
  label: 'analyze-kernel',
  phase: 'Analyze',
  schema: {{ANALYZE_SCHEMA}},
})

log(`Analysis: ${analysis.opportunities?.length || 0} transformation opportunities identified`)

// Identify which passes are applicable based on analysis
const passOrder = {{PASS_ORDER}}
const applicablePasses = passOrder.filter(pass => {
  const opportunity = (analysis.opportunities || []).find(o => o.pass === pass)
  return opportunity && opportunity.applicable
})

log(`Applicable passes: ${applicablePasses.join(' → ')}`)

// =============================================================================
// Phase: Transform — Apply optimization passes sequentially
// =============================================================================
phase('Transform')

let currentCode = analysis.source_code || ''
const transformResults = []

const transforms = await pipeline(
  applicablePasses,
  (pass, _, passIdx) => agent(`{{TRANSFORM_PROMPT}}

# Pass: "${pass}" (${passIdx + 1}/${applicablePasses.length})

# Current code (output of previous pass):
\`\`\`
${currentCode.substring(0, 5000)}
\`\`\`

# Analysis findings relevant to this pass:
${JSON.stringify((analysis.opportunities || []).find(o => o.pass === pass) || {})}

Apply this transformation pass. Output the complete transformed code.`, {
    label: `transform-${pass}`,
    phase: 'Transform',
    schema: {{TRANSFORM_SCHEMA}},
  })
)

// Accumulate transforms sequentially
for (const result of transforms.filter(Boolean)) {
  if (result.transformed_code) {
    currentCode = result.transformed_code
    transformResults.push(result)
  }
}

log(`Transforms applied: ${transformResults.length}/${applicablePasses.length}`)

// =============================================================================
// Phase: Verify — Check correctness of the final transformed code
// =============================================================================
phase('Verify')

const verification = await agent(`{{VERIFY_PROMPT}}

# Original code:
\`\`\`
${analysis.source_code?.substring(0, 3000) || ''}
\`\`\`

# Final transformed code:
\`\`\`
${currentCode.substring(0, 3000)}
\`\`\`

# Passes applied: ${transformResults.map(t => t.pass_name || 'unknown').join(' → ')}

Verify functional equivalence and check for introduced bugs.`, {
  label: 'verify-transforms',
  phase: 'Verify',
  schema: {{VERIFY_SCHEMA}},
})

log(`Verification: ${verification.is_correct ? 'PASSED' : 'FAILED'} — ${verification.issues?.length || 0} issues`)

// =============================================================================
// Phase: Report — Summary of transformations
// =============================================================================
phase('Report')

const finalReport = await agent(`{{REPORT_PROMPT}}

# Pipeline Results:
- Passes applied: ${transformResults.length}
- Verification: ${verification.is_correct ? 'PASSED' : 'FAILED'}
- Issues: ${JSON.stringify(verification.issues || [])}
- Transform chain: ${applicablePasses.join(' → ')}`, {
  label: 'final-report',
  phase: 'Report',
})

return {
  {{RETURN_OBJECT}}
}
