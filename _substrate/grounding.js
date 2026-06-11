// _substrate/grounding.js
//
// Grounding contract: every subagent that returns a structured numeric result
// must have executed a real command that produced it. If the prescribed tool
// or script does not exist (e.g. GPUForecasters wanting `surrogate_train.py`
// when no such file is in the repo), the subagent MUST return
// `{ grounded: false, missing: "<what>" }` and stop, rather than fabricating
// values to satisfy the schema.
//
// Usage in a workflow .js:
//
//   import { GROUNDING_INSTRUCTION, withGroundingFields } from '../_substrate/grounding.js'
//   const RESULT_SCHEMA = withGroundingFields({
//     type: 'object',
//     properties: { speedup: { type: 'number' } },
//     required: ['speedup'],
//   })
//   const r = await agent(prompt + '\n\n' + GROUNDING_INSTRUCTION, { schema: RESULT_SCHEMA })
//   if (r.grounded === false) { ... handle not-grounded path; do not consume r.speedup ... }
//
// result-analyzer reads `grounded === false` as `verdict: not_grounded` rather
// than `failed`, so KerSor's selector can learn "this workflow cannot be
// grounded in this environment" and stop picking it.

export const GROUNDING_INSTRUCTION = `
GROUNDING CONTRACT (mandatory):

1. Before reporting any numeric or categorical result, you MUST have executed
   a real Bash command (compile / test / benchmark / profile) that produced
   the value. Citing a value from imagination, prior knowledge, or analogy
   is a contract violation.

2. If the workflow tells you to run a tool or script that does NOT exist on
   disk or on PATH (e.g. \`python3 surrogate_train.py\` when no such file is
   present), do NOT try to substitute, simulate, or rationalize. Return:
       { "grounded": false, "missing": "<exact tool or script that was absent>" }
   and stop. The schema accepts this shape.

3. If a Bash command fails (non-zero exit), do NOT invent the value it would
   have produced. Return:
       { "grounded": false, "error": "<stderr tail or short summary>" }
   and stop.

4. Numeric fields in the schema (latency_ms, speedup, occupancy, etc.) are
   REJECTED downstream when grounded === false. Do not work around the
   schema by filling them with placeholders, zeros, or "typical" values.

5. The string "fabricated", "simulated", "estimated", or "placeholder" must
   never appear in a value you report as measured.
`.trim()

// Add the grounding fields to a JSON Schema for an agent's structured output.
// Mutates the schema in place AND returns it for chained use.
export function withGroundingFields(schema) {
  schema.properties = schema.properties || {}
  if (!schema.properties.grounded) {
    schema.properties.grounded = {
      type: 'boolean',
      description: 'true iff every reported numeric value came from a real executed command (see GROUNDING_INSTRUCTION).',
    }
  }
  if (!schema.properties.missing) {
    schema.properties.missing = {
      type: 'string',
      description: 'When grounded=false: the absent tool or script that prevented grounding.',
    }
  }
  if (!schema.properties.error) {
    schema.properties.error = {
      type: 'string',
      description: 'When grounded=false: short stderr / failure summary.',
    }
  }
  return schema
}

// Helper for workflow scripts: collapse an agent result into one of three states.
//   { state: 'grounded', value }      -> use value
//   { state: 'not_grounded', missing, error }
//   { state: 'failed', error }        -> agent itself returned null / threw
export function classifyResult(agentResult) {
  if (agentResult == null) return { state: 'failed', error: 'agent returned null' }
  if (agentResult.grounded === false) {
    return {
      state: 'not_grounded',
      missing: agentResult.missing || null,
      error: agentResult.error || null,
    }
  }
  return { state: 'grounded', value: agentResult }
}
