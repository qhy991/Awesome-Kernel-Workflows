# Turn deadlines and child settlement

The turn watchdog owns admission of new attempts. It does not cancel a native
child, terminate a broker job or prove resource release.

Use the canonical helper with a deferred call:

```js
const result = await withTurnTimeout(
  () => agent(prompt, options),
  'Implement',
  {retries: 0},
)
```

The helper starts the existing timer and runs the thunk through `agentRetry`.
At the deadline it records expiry. It continues waiting for the active invocation
so the parent cannot proceed while that invocation may still write. A late value
or null becomes the nonretryable `KERSOR_TURN_DRAINED` error with `invocation_settled: true` after the
invocation settles. A rejected invocation retains its original error, regardless
of the remaining retry count. The deadline scope supplies `mayRetryError` to the
retry owner so that a late rejection cannot start another attempt. Permission,
policy, budget and other nonretryable classification remains in `agentRetry`.

Timely calls retain their previous retries and null policy. Without timers the
helper still uses bounded `agentRetry`. A never-settling native child remains
subject to the existing outer controller deadline. This is drain-before-return,
not cancellation. Source-bound native terminal, process and broker observations
are still required before resource release or candidate settlement.

`withTurnTimeout(agentRetry(...), label)` is retired in this successor because
its promise has already started and owns its own retry loop. Migrate call sites
and generated helper blocks together:

```sh
node scripts/add-agent-retry-scaffolding.js --all --refresh
node scripts/patch-turn-timeout.js --all --refresh --migrate-calls
```

Refreshing or inserting the helper also migrates its call sites in the same
operation. The codemod preserves the original agent thunk, label and retry options. It
refuses unknown call shapes and is idempotent. The null-safety guard recognizes
timeout-owned retries and still requires guarding `allowNull: true` results.

CPU tests retain the ea08c83 race as a negative control and cover late values,
nulls and errors, nonretryable refusals, retry exhaustion, no timers and call-site
migration. Parse the workflows with the actual Host wrapper and run method
fidelity guards. Native site qualification remains separate; do not rewrite
frozen workflows or claim existing pending children were cancelled.

Integration with main 9d23aa9 preserves its finite candidate-failure records,
checkpoints and recovery-stop guards. Legacy `KERSOR_TURN_TIMEOUT` and
`KERSOR_TURN_UNSETTLED` remain unverified and cannot enter the ordinary timeout
path. The new drained code is distinct and never asserts GPU/resource release.
