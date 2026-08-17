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

## Research-backed evaluation profiles

The strategist above answers **how deeply to run an available project verifier**.
The following profile names answer a different question: **which evidence
obligations should the project-owned verifier cover**. They are vocabulary and
audit expectations, not built-in executable harnesses. A workflow must still
receive a real command/artifact owner and must fail closed when the requested
evidence is absent.

| profile | evidence obligations | primary source |
|---|---|---|
| `contract-grade` | Explicit operator contract; adversarial properties such as non-finite behavior, determinism, shape variation, and accumulation semantics; each gate reported separately, including tolerance-free gates where applicable | [A Contract-Grade Verifier for LLM-Generated GPU Kernels](https://arxiv.org/abs/2608.12700) |
| `kernelbench-verified` | Realistic TF32-enabled PyTorch baseline; all four hidden input distributions must pass; speed and peak-memory efficiency reported together; input-blind protection where the task requires it | [KernelBench-Verified](https://arxiv.org/abs/2607.16241), [official repository](https://github.com/facebookresearch/kernel_bench_verified) |
| `kernelgenbench` | Operator/source identity, target chip, generation track, correctness result, performance result, and run provenance retained so results can be compared across supported hardware | [KernelGenBench](https://arxiv.org/abs/2607.27231), [official repository](https://github.com/flagos-ai/KernelGenBench) |
| `custom` | A project-owned, explicitly frozen set of gates and aggregation rules | project contract |

These profiles are complementary rather than a single linear ladder. For
example, a campaign can use `reference_test` as its strategist method while the
underlying command implements the `kernelbench-verified` obligations. The method
and profile must therefore be recorded as separate fields in evidence.

## Scope boundary

Applies to **generators** (no perf signal). Workflows whose evaluation target is a
non-GPU simulator domain are out of scope for both strategists.
