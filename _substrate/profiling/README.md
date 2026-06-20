# Shared profiling-strategist

The **one profiling entry point** every AKW workflow with a profiling step should
call, instead of inlining its own `ncu`-or-latency dispatch. Picks a profiling
**method** per `(backend × task × host)`, then the substrate **stamps how much to
trust it**. Measurement-side twin of KerSor's `framework-integrator`.

## Why

Today each workflow hardcodes its profiler dispatch (`profiler_name ? ncu : latency`).
That is duplicated, NVIDIA-biased, and breaks on hosts without `ncu`. This component
centralizes the decision while preserving the project's asymmetric-authority rule.

## The three properties

- **Knows common backend methods** — `profiler_registry.json` defines the generic
  ladder (`native → perf_heuristic → static`) + stamping. The per-backend *native*
  profiler is read from each backend's `manifest.json` (the SSOT: `profiler.name` +
  `capabilities.metrics`), so this never drifts from the backends.
- **Generality** — any backend with a `manifest.json` resolves with zero strategist
  code (verified for cuda/rocm/ascend/metal/triton + a synthetic `sycl`).
- **Autonomy** — a backend with **no** manifest routes to `derive_adapter`: the agent
  derives a method at runtime (framework-integrator style) and it is cached back.

## Asymmetric authority (preserved)

| Decision | Owner | Why |
|---|---|---|
| classify task (op_class/size), probe host, derive unknown-backend adapter | **agent** (fuzzy) | needs judgment |
| method **selection** over known backends | `resolve()` **deterministic** | reproducible: same inputs → same route |
| **confidence** stamp (`measured`/`inferred`/`hypothesized`) | registry row of the chosen method — **never the agent** | provenance / parity-gate |
| metrics → bottleneck class | `diagnose.py` | unchanged |

So: **agent picks *which* analysis; substrate decides *how much to trust* it.**

## Call contract (every workflow uses this)

```
# decide once per session, cache per (backend, task-class, host)
python3 _substrate/profiling/profiling_strategist.py resolve \
    --backend-manifest <backend>/manifest.json \
    --task <attention|gemm|elementwise|reduction|default> --size <tiny|small|large> \
    --cache <exp_dir>/prof_cache.json --trajectory <exp_dir>/genome.jsonl
# -> {method, evidence_source, confidence, normalizer, profile_invoke,
#     requested_metrics, coverage_expected, abstained_from, rationale, cache_key}
```

The workflow then runs the returned method deterministically:
- `native_profiler` → run `profile_invoke` (`profile.sh`) → `to_evidence.py`
- `perf_heuristic`  → run `run.sh` / `test-backend-ops perf` → `perf_to_evidence.py`
- `static`          → source-read stub
- `derive_adapter`  → agent derives + caches, then re-resolve

All paths emit the canonical metrics dict `diagnose.py` consumes; `confidence` is
already stamped, so downstream evidence stays provenance-tagged.

## Files

- `profiling_strategist.py` — selector + stamper (deterministic core + autonomy hook)
- `profiler_registry.json` — generic ladder + stamping + probe-tool aliases
- `perf_to_evidence.py` — `perf_heuristic` normalizer (test-harness perf → evidence)
- `tests/test_strategist.sh` — 14 assertions on real manifests + a llama.cpp perf fixture

## Provenance (evidence enum)

`evidence_schema.py`'s EVIDENCE enum includes `native_profiler`. The strategist
stamps literal Nsight Compute as `evidence="ncu"` and every other native hw
profiler (rocprof/msprof/vtune/metal-capture) as `evidence="native_profiler"`,
so provenance is honest rather than借用 the ncu tag. `profiler_name` carries the
true tool in both cases.
