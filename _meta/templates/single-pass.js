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

// =============================================================================
// Phase: Analyze — Identify transformation opportunities
// =============================================================================
phase('Analyze')

const analysis = await agentRetry(() => agent(`{{ANALYZE_PROMPT}}`, {
  label: 'analyze-kernel',
  phase: 'Analyze',
  schema: {{ANALYZE_SCHEMA}},
}), { retries: 5 })

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
  (pass, _, passIdx) => agentRetry(() => agent(`{{TRANSFORM_PROMPT}}

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
  }), { retries: 5 })
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

const verification = await agentRetry(() => agent(`{{VERIFY_PROMPT}}

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
}), { retries: 5 })

log(`Verification: ${verification.is_correct ? 'PASSED' : 'FAILED'} — ${verification.issues?.length || 0} issues`)

// =============================================================================
// Phase: Report — Summary of transformations
// =============================================================================
phase('Report')

const finalReport = await agentRetry(() => agent(`{{REPORT_PROMPT}}

# Pipeline Results:
- Passes applied: ${transformResults.length}
- Verification: ${verification.is_correct ? 'PASSED' : 'FAILED'}
- Issues: ${JSON.stringify(verification.issues || [])}
- Transform chain: ${applicablePasses.join(' → ')}`, {
  label: 'final-report',
  phase: 'Report',
}), { retries: 5 })

return {
  {{RETURN_OBJECT}}
}
