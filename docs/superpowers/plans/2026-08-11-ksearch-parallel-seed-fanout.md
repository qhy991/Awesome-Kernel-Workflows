# K-Search bounded parallel seed fan-out

## Outcome and non-goals

K-Search must use the workflow runtime's available concurrency for independent
candidate exploration.  One selected world-model action fans out several seed
implementations concurrently, then reduces their measured results
deterministically before the existing debug/improve and tree-refinement path.

This change does not parallelize GPU compilation, correctness checks, benchmark
runs, embedded project mutation, world-model updates, or dependent debug turns.
It also does not claim that the upstream K-Search Python generator runs multiple
LLM rounds concurrently; this is a safe concurrency adaptation for the AKW
runtime.

## Constraints, invariants, and budget

- `attempts_per_cycle` remains the sole owner of the total candidate budget.
- `seed_candidates` is the canonical optional width capped to
  `1..attempts_per_cycle`; its default is `4`.
- Every parallel seed writes a unique `cycle_<cycle>_a<attempt>` candidate path.
- The decision tree and parent solution are immutable snapshots during fan-out.
- Candidate evaluation remains serial, so one GPU measurement and one embedded
  project mutation own the machine at a time.
- `parallel()` preserves input ordering; reduction therefore keeps stable
  attempt IDs independent of completion order.
- A failed seed occupies its existing null slot and cannot shift another
  candidate's identity or evidence.
- Stagnation cannot stop evaluation halfway through an already-generated seed
  batch.

## Minimal primitives and owners

| Fact | Canonical owner |
|---|---|
| Total per-cycle candidate budget | `attempts_per_cycle` |
| Concurrent seed width | `seed_candidates`, capped by the total budget |
| Candidate identity and source path | `(cycle, attempt)` |
| Correctness and performance | The existing serial evaluator result |
| Tree mutation and best-candidate promotion | The existing post-evaluation reducer |

## Dataflow and state transitions

```text
select one executable tree node
  -> snapshot node + parent + world model
  -> parallel(seed generation[0..width-1])
  -> ordered candidate slots
  -> serial evaluate + deterministic best reduction
  -> optional dependent debug/improve attempts within remaining budget
  -> one refine/backtrack tree transition
```

## Failure, rollback, and compatibility

Individual generation failures remain null branches and other branches continue.
If every seed fails, the existing backtrack path owns the outcome.  Setting
`seed_candidates=1` reproduces the prior prompt sequence and execution
semantics, so frozen golden tests and constrained deployments retain a bounded
compatibility lane.  Reverting this change restores the prior serial loop
without migrating stored session data.

## Acceptance evidence

- A prompt-capture test with `attempts_per_cycle=4` proves all four `gen-*`
  calls are issued as one fan-out before the first `eval-*` call.
- A structural guard proves the fan-out uses `parallel(thunks)` and the manifest
  declares the optional width and parallel generation phase.
- Structural guards prove width is capped by `attempts_per_cycle`; the existing
  frozen serial fixtures continue to exercise the width-`1` compatibility lane.
- K-Search guards and KerSor's full suites pass.  The complete AKW suite retains
  exactly the same pre-existing failure set as the clean baseline.
