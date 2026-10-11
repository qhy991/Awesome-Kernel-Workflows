// Canonical per-turn deadline and child-settlement boundary.
// Run scripts/patch-turn-timeout.js --all --refresh after changing this helper.
//
// A deadline stops further attempts. It cannot cancel the native agent. The
// active invocation must settle before a workflow may handle the timeout or
// start its next turn. A stuck invocation remains under the existing outer
// controller deadline; this helper does not extend that budget or prove release.
//
// Pass the agent thunk, label and existing retry options. The helper owns the
// retry scope so no retry can start after expiry. Do not pass a started promise.
// Example: withTurnTimeout(() => agent(prompt, options), 'Implement', {retries: 0}).
// Only isTurnTimeout(error) may become convergence_status='timeout'. Original
// permission, policy, schema and transport failures retain their error identity.
// No clock reads, random values, imports or runtime cancellation API are used.

const TURN_TIMEOUT_MS = (args.turn_timeout_min || 12) * 60 * 1000  // stop admitting attempts at this deadline

async function withTurnTimeout(fn, label, retryOptions) {
  if (typeof fn !== 'function') {
    const error = new TypeError('withTurnTimeout requires an agent thunk, not a started promise')
    error.code = 'KERSOR_REQUEST_INVALID'
    error.retryable = false
    throw error
  }
  if (typeof setTimeout !== 'function' || !(TURN_TIMEOUT_MS > 0)) return agentRetry(fn, retryOptions)
  let expired = null
  const timer = setTimeout(() => {
    expired = new Error(`turn-timeout: ${label} exceeded ${Math.round(TURN_TIMEOUT_MS / 1000)}s`)
    expired.code = 'KERSOR_TURN_TIMEOUT'
    expired.retryable = false
  }, TURN_TIMEOUT_MS)
  try {
    return await agentRetry(async () => {
      if (expired) throw expired
      const result = await fn()
      if (expired) throw expired
      return result
    }, {...retryOptions, mayRetryError: () => !expired})
  } finally {
    if (typeof clearTimeout === 'function') clearTimeout(timer)
  }
}

// Broker timeouts and this watchdog are the only timeout classifications.
function isTurnTimeout(error) {
  return ['KERSOR_TURN_TIMEOUT', 'KERSOR_CODEX_TIMEOUT', 'KERSOR_CLAUDE_TIMEOUT'].includes(error?.code)
}
