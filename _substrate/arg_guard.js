// _substrate/arg_guard.js
//
// Defensive unwrapper for the `args` global injected into workflow scripts by
// Claude Code's Workflow tool. Some invocation paths deliver `args` as a JSON
// string instead of an object (notably when KerSor's optimize skill dispatches
// via `scriptPath` rather than `name`). Without this guard, every
// `args.<field>` access silently returns `undefined`, causing workflows to
// either throw cryptic "Provide one of ..." errors or, worse, to take the
// "no input given" fallback branch and fabricate results.
//
// Usage at the top of every workflow script, right after `export const meta`:
//
//   import { unwrapArgs } from '../_substrate/arg_guard.js'
//   args = unwrapArgs(args)
//
// After this line, `args` is guaranteed to be a plain object (possibly empty).
// Required-arg validation continues to live in the individual workflow.

export function unwrapArgs(rawArgs) {
  if (rawArgs == null) return {}

  if (typeof rawArgs === 'object' && !Array.isArray(rawArgs)) {
    return rawArgs
  }

  if (typeof rawArgs === 'string') {
    const trimmed = rawArgs.trim()
    if (trimmed === '') return {}

    // Plain JSON object/array case.
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        const parsed = JSON.parse(trimmed)
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          return parsed
        }
        throw new Error('parsed value is not a plain object')
      } catch (e) {
        throw new Error(
          `arg_guard: workflow args looked like JSON but did not parse to an object. ` +
          `First 160 chars: ${trimmed.slice(0, 160)}. Underlying error: ${e.message}`
        )
      }
    }

    // KerSor "key=value key=value" string form. Parse with whitespace as the
    // separator, splitting on the first '=' so that values containing '=' or
    // spaces inside quotes are preserved.
    const out = {}
    const re = /(\w[\w.-]*)=("(?:\\"|[^"])*"|'(?:\\'|[^'])*'|\S+)/g
    let m
    while ((m = re.exec(trimmed)) !== null) {
      let v = m[2]
      if ((v.startsWith('"') && v.endsWith('"')) ||
          (v.startsWith("'") && v.endsWith("'"))) {
        v = v.slice(1, -1)
      }
      out[m[1]] = v
    }
    if (Object.keys(out).length === 0) {
      throw new Error(
        `arg_guard: workflow args is a non-empty string but contains no key=value ` +
        `pairs and is not JSON. First 160 chars: ${trimmed.slice(0, 160)}`
      )
    }
    return out
  }

  throw new Error(`arg_guard: workflow args has unexpected type: ${typeof rawArgs}`)
}
