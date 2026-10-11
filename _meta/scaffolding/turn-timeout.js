// turn-timeout.js — CANONICAL default-scaffolding snippet for `agent()`-based workflows.
//
// This is NOT a runnable workflow. It is the single source of truth for the
// per-turn wall-clock watchdog that every `agent()`-based workflow inlines
// (workflow .js files run in the Claude Code Workflow sandbox and cannot `import`,
// so the helper is copied verbatim into each file — the same convention already
// used for the inlined arg_guard and agent-retry scaffolding).
//
// WHY: a workflow loop bounded ONLY by MAX_TURNS / MAX_CYCLES can stall forever
// if a single doer `agent()` turn hangs (a subagent stuck in_progress, a model
// not responding, a shell step wedged behind a step the runtime cannot observe).
// On-host evidence (CUDAAgent issue #12 / PR #14) showed a single hung
// Implement/Verify turn stalling the whole run for 40+ minutes before an
// orchestrator manually killed it. TURN_TIMEOUT_MS bounds each doer turn; on
// expiry the wrapped promise rejects with `turn-timeout: <label> exceeded Ns`
// and the caller fails closed with an unresolved-child HOLD. Promise.race does
// not cancel the child: returning an incumbent or starting another cycle could
// overlap work or seal a source that the abandoned child is still modifying.
//
// PARITY: CUDAAgent has carried this guard since #12/#14. Issues #30 propagates
// the same guard to ARGUS and KSearch, which lacked it (KSearch already had
// EVAL_TIMEOUT_SEC — a shell-level `timeout Ns` on eval commands — but no
// agent-turn wall-clock cap, so a hung non-eval agent() call still stalled).
//
// CONSTRAINT: the Workflow runtime forbids `Date.now()` / `Math.random()` /
// argless `new Date()` (they throw and would break resume). This guard uses
// `setTimeout` / `clearTimeout` / `Promise.race` only — no forbidden APIs. It
// degrades to a passthrough when the runtime exposes no timers
// (`typeof setTimeout !== 'function'`) or when TURN_TIMEOUT_MS <= 0, so a host
// without timers still runs (just without the wall-clock bound).
//
// Run scripts/patch-turn-timeout.js --all --refresh after changing this helper.
// Wrap each doer activation with withTurnTimeout(agentRetry(..., {retries:0}), label).
// Only isTurnTimeout(error) may be translated into convergence_status='timeout';
// all other errors must retain their original identity.

const TURN_TIMEOUT_MS = (args.turn_timeout_min || 12) * 60 * 1000  // per-turn wall-clock cap

/**
 * Wrap a doer-turn promise with a wall-clock cap. On expiry the returned
 * promise rejects with KERSOR_TURN_UNSETTLED. This is not a child terminal or
 * resource-release receipt. The site must settle exact owned processes and
 * broker jobs before any reviewed recovery. Degrades to a
 * passthrough when the runtime has no timers or TURN_TIMEOUT_MS <= 0.
 */
function withTurnTimeout(promise, label) {
  if (typeof setTimeout !== 'function' || !(TURN_TIMEOUT_MS > 0)) return promise
  let timer
  const guard = new Promise((_, reject) => {
    timer = setTimeout(
      () => {
        const error = new Error(`KERSOR_TURN_UNSETTLED: ${label} exceeded ${Math.round(TURN_TIMEOUT_MS / 1000)}s; child terminal and resource release are unverified`)
        error.code = 'KERSOR_TURN_UNSETTLED'
        error.retryable = false
        error.outcome_state = 'unknown'
        error.child_terminal_verified = false
        error.resources_released = false
        error.phase_label = label
        reject(error)
      },
      TURN_TIMEOUT_MS)
  })
  return Promise.race([promise, guard]).finally(() => {
    if (typeof clearTimeout === 'function') clearTimeout(timer)
  })
}

// Broker process timeouts settle their child before rejecting. A local watchdog
// (including the old KERSOR_TURN_TIMEOUT code) cannot establish that fact.
// Permission, schema, transport, and task-evidence failures keep their identity.
function isTurnTimeout(error) {
  return ['KERSOR_CODEX_TIMEOUT', 'KERSOR_CLAUDE_TIMEOUT'].includes(error?.code)
}
