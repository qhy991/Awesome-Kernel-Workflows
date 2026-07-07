// arg-guard.js — CANONICAL default-scaffolding snippet for the bare-script arg
// unwrap every `agent()`-based workflow runs at the top of its file.
//
// This is NOT a runnable workflow. It is the single source of truth for the
// `__unwrapArgs` helper + the `args = __unwrapArgs(...)` reassignment that every
// workflow inlines (workflow .js files run in the Claude Code Workflow sandbox as
// bare scripts, not ES modules, so `args` can arrive as a JSON string, a
// `key=value` string, or a plain object — this normalizes all three to an object
// BEFORE the workflow reads any `args.*`).
//
// WHY: the helper was copy-pasted across 32 workflows with no SSOT file (only
// the codemod `scripts/patch-arg-guard.js` held the canonical text). This file
// gives it a home — matching the agent-retry.js / turn-timeout.js /
// backend-axis.js convention — so a fix to the unwrap is one edit here + a
// `patch-arg-guard.js --refresh` to propagate.
//
// CONSTRAINT: the Workflow runtime forbids `Date.now()` / `Math.random()` /
// argless `new Date()` (they throw and would break resume). This helper uses none.
//
// USAGE (inline at the top of a workflow, before the first `args.*` read):
//
//   // --- BEGIN inlined arg_guard (from _meta/scaffolding/arg-guard.js) ---
//   <paste __unwrapArgs + the args = __unwrapArgs(...) line>
//   // --- END inlined arg_guard ---

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
