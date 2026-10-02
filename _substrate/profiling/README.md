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

For KerSor Host runs on CUDA, set `KERSOR_GPUQ` to the site's broker executable.
Generalist and KernelBand ask the agent for an argv plan containing the
literal `{artifact}` candidate placeholder and optional device-supported
metrics. The Host `ncu-v1` evaluator obtains an exclusive broker lease, runs
`cuda/profile.sh`, preserves the native output and broker receipt, then invokes
`to_evidence.py`. A site with admin-only counters can supply a verified wrapper
through `profile_binary` (Generalist) or `ncu_binary` (KernelBand); the wrapper
must retain the broker's `CUDA_VISIBLE_DEVICES`. Without a broker or runnable
launcher the native route returns missing evidence rather than model numbers.

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

## Inner-agent diagnostics

A Bash-enabled runtime may expose `KERSOR_NCU_COMMAND` and inject the focused
`ncu-profiling` skill. An inner profiler can call that CLI itself, choose its
launcher/counters/kernel selector, and inspect real counters before returning
a plan. Generalist and KernelBand explicitly permit this in their CUDA driver
profiler prompts. Read-only activations still return plans for Host evaluation.
The CLI uses the same exclusive GPU broker and collector as `ncu-v1`; retain
its native CSV and receipt. These workspace diagnostics do not replace the
Host's accepted score or the no-profiler performance evaluator.
