// agent-retry.js — CANONICAL default-scaffolding snippet for `agent()`-based workflows.
//
// This is NOT a runnable workflow. It is the single source of truth for the
// agentRetry / null-guard scaffolding that every `agent()`-based workflow inlines
// (workflow .js files run in the Claude Code Workflow sandbox and cannot `import`,
// so the helper is copied verbatim into each file — the same convention already
// used for the inlined arg_guard).
//
// WHY: a transient API 429 (or any agent skip / terminal subagent failure) makes
// `await agent(...)` return null. Workflows that immediately dereference the
// structured result (`inspection.kernel_code`, `impl.code`, ...) then crash the
// whole run. On-host evidence (910b-exp 20260622-161357 §5) showed that wrapping
// every `agent()` call in a bounded retry + null-guarding the dereference points
// was the single highest-leverage robustness fix: round 3 then ran 14 agents over
// 2.4h with zero crashes. See issues #16 / #17.
//
// CONSTRAINT: the Workflow runtime forbids `Date.now()` / `Math.random()` /
// argless `new Date()` (they throw and would break resume) and exposes no
// `sleep()`. So this wrapper does immediate bounded retries — which the on-host
// evidence showed was sufficient for the sporadic 429s encountered. If the
// runtime later exposes a sleep, insert a short delay between attempts.
//
// USAGE (inline at the top of a workflow, after the arg_guard, before first use):
//
//   // --- BEGIN inlined agent-retry scaffolding (from _meta/scaffolding/agent-retry.js) ---
//   <paste the three functions below>
//   // --- END inlined agent-retry scaffolding ---
//
//   // Then wrap every deref-risk agent call and guard the dereference:
//   const inspection = await agentRetry(
//     () => agent(`...`, { label: 'inspect', phase: 'Inspect', schema: INSPECT_SCHEMA }),
//     { retries: 5, label: 'inspect' },
//   )
//   const kernelCode = expect(inspection, 'kernel_code', 'inspect')

/**
 * Run an agent-producing thunk with bounded retries on null/throw.
 * `fn` must be `() => agent(prompt, opts)` (the call itself, deferred).
 * Returns the first non-null result, or null/throws after `retries+1` attempts.
 */
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
  return null
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
