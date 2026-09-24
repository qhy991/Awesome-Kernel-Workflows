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
// The Host owns the hard timeout of each agent activation. This workflow guard
// records a turn timeout, but must drain the activation before returning:
// Promise.race alone abandoned live retries on B300 and caused
// KERSOR_UNAWAITED_AGENT_CALL in 005/006 Stage5.
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
// USAGE (inline at the top of a workflow, right after the agent-retry scaffolding,
// before the first doer turn):
//
//   // --- BEGIN inlined turn-timeout scaffolding (from _meta/scaffolding/turn-timeout.js) ---
//   const TURN_TIMEOUT_MS = (args.turn_timeout_min || 12) * 60 * 1000  // per-turn wall-clock cap
//   <canonical const + function below; do not hand-author the inline copy>
//   // --- END inlined turn-timeout scaffolding ---
//
//   // Then wrap a doer turn and translate a timeout into an early, recorded exit:
//   try {
//     const implResult = await agentRetry(
//       () => withTurnTimeout(agent(`...`, { label: 'impl', phase: 'Implement', schema: IMPL_SCHEMA }),
//                             `Implement turn ${currentAttempt + 1}`),
//       { retries: 5 })
//   } catch (e) {
//     log(`  Implement watchdog tripped — stopping (${e.message})`)
//     convergenceStatus = 'timeout'
//     break
//   }
//
//   // In a linear pipeline (no loop), let the timeout reject propagate — the
//   `turn-timeout:` error is attributable and aborts the round cleanly (#20-style).

const TURN_TIMEOUT_MS = (args.turn_timeout_min || 12) * 60 * 1000  // per-turn wall-clock cap

/**
 * Wrap a doer-turn promise with a wall-clock cap. On expiry the returned
 * promise rejects with `turn-timeout: <label> exceeded Ns`. Degrades to a
 * passthrough when the runtime has no timers or TURN_TIMEOUT_MS <= 0.
 */
function withTurnTimeout(promise, label) {
  if (typeof setTimeout !== 'function' || !(TURN_TIMEOUT_MS > 0)) return promise
  let timer
  let expired = false
  const guard = new Promise((_, reject) => {
    timer = setTimeout(
      () => {
        expired = true
        reject(new Error(`turn-timeout: ${label} exceeded ${Math.round(TURN_TIMEOUT_MS / 1000)}s`))
      },
      TURN_TIMEOUT_MS)
  })
  return Promise.race([promise, guard]).catch(async error => {
    if (expired) {
      // A race does not cancel agent(). Drain this Host-bounded activation so
      // no running call survives the workflow's return.
      try { await promise } catch (_) { /* preserve the guard error */ }
    }
    throw error
  }).finally(() => {
    if (typeof clearTimeout === 'function') clearTimeout(timer)
  })
}
