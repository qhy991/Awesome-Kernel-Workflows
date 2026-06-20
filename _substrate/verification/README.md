# Shared verification-strategist

The **verification-depth entry point** for GENERATOR workflows (TritorX,
KernelAgent, …) that have no performance signal. Same general technique as the
[profiling-strategist](../profiling/README.md), applied to a different routed
quantity: instead of "which profiler", it routes **how deeply to verify a
generated candidate**.

## Why

Generators don't optimize a perf number; their loop is generate → compile → test.
But "test" has a depth spectrum, and the right depth depends on phase: cheap
compile+lint for early pruning, full reference comparison for final confirmation.
Inlining that dispatch per-workflow duplicates logic and over-pays during search.
This component centralizes it.

## The ladder (deepest first)

| method | evidence | confidence | requires | meaning |
|---|---|---|---|---|
| `reference_test` | `correctness` | `measured` | reference + harness | run vs reference, numerical compare |
| `smoke_test` | `runtime` | `measured` | harness | compile + run, shape/finiteness only |
| `compile_lint` | `compile` | `inferred` | compiler | compile + lint, no execution |
| `static` | `llm_inferred` | `hypothesized` | — | source read only |

`evidence` values reuse the existing `evidence_schema` EVIDENCE enum — **no new enum**.

## Needs (the agent's fuzzy input)

- `prune` — early screening. Intentionally **skips execution rungs** (no
  `reference_test`/`smoke_test`) so pruning stays cheap → `compile_lint` or `static`.
- `confirm` — final acceptance. Walks the full ladder to the deepest method the host supports.
- `default` — full ladder (like confirm).

## Asymmetric authority (same as the profiling twin)

| Decision | Owner |
|---|---|
| classify need (prune/confirm), probe host | **agent** (fuzzy) |
| method **selection** | `resolve()` **deterministic** — same (need, host) → same route |
| **confidence** stamp | registry row of the chosen method — **never the agent** |
| actual correctness verdict | the workflow's test harness (unchanged) |

## Call contract

```
python3 _substrate/verification/verification_strategist.py resolve \
    --need <prune|confirm|default> \
    --host-probe '{"compiler":true,"harness_runnable":true,"reference_available":false}' \
    --cache <exp_dir>/verify_cache.json --trajectory <exp_dir>/genome.jsonl
# -> {method, evidence_source, confidence, requires, abstained_from, rationale, cache_key}
```

The workflow runs the returned method; `confidence` is already stamped, so any
evidence the harness emits stays provenance-tagged.

## Files

- `verification_strategist.py` — selector + stamper (deterministic core)
- `verification_registry.json` — ladder + need-ladders + stamping
- `tests/test_strategist.sh` — 16 assertions (no GPU/harness needed)

## Scope boundary

Applies to **generators** (no perf signal). Workflows whose evaluation target is a
non-GPU simulator domain are out of scope for both strategists.
