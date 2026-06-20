# Shared integration-strategist

The **build/integration-mode entry point** — the THIRD axis. Same general technique
as the [profiling-strategist](../profiling/README.md) and
[verification-strategist](../verification/README.md), applied to the
COMPILATION/INTEGRATION axis: routes **how a candidate kernel is built+tested**.

## Why

A kernel embedded in a project it cannot compile without (llama.cpp `fattn.cu`,
`mmq.cuh`) fails the default single-TU `build.sh`. Only 6 workflows hardcode an
`integration_pattern` for this; the other 23 default to `standalone` and break on
embedded kernels. This component lets ANY workflow ask "how do I build+test this
kernel?" and get the right mode — making all 29 able to handle llama.cpp, not just 6.

## The ladder

| method | build_fidelity | reversible | requires | eval mechanism |
|---|---|---|---|---|
| `standalone` | `isolated` | yes | compiler | driver `build.sh`+`run.sh` (single-TU) |
| `embedded_inplace` | `project_native` | yes | project_build + reversibility_net | project-native build/test/bench + backup/restore |
| `embedded_dispatch` | `production` | yes | project_build + register_script + reversibility_net | `embedded/embedded_eval.js` (register→build→test→bench→unregister) |
| `registry_dispatch` | `production` | yes | runtime_registry | python register/unregister (namespace-roundtrip) |

`build_fidelity` is the analog of profiling `confidence`: `isolated < project_native < production`.

## Routing (deterministic; agent only classifies the fuzzy input)

- **Agent's fuzzy input**: `can_compile_standalone` (yes/no/uncertain) — read the
  kernel's `#include`s / template deps / dispatch-symbol refs.
- `yes` + compiler → `standalone`.
- `uncertain` + compiler → `standalone` (optimistic; a build failure cleanly reveals
  the kernel is actually embedded, then re-resolve with `no`).
- `no` (confirmed embedded) → walk `embedded_dispatch > embedded_inplace >
  registry_dispatch` to the highest-fidelity reversible mode the host supports.
  `standalone` is intentionally NOT in this ladder — a doomed single-TU compile of a
  confirmed-embedded kernel is dishonest.
- Nothing embedded host-supported → `derive_adapter` (autonomy: agent derives a
  project adapter, framework-integrator style; cached back).

`build_fidelity` + `reversible` are stamped by the method's registry row — never by
the agent. The actual build/test/revert still belongs to the workflow + adapter
(`embedded_eval.js` / `ADAPTER_CONTRACT.md`); this component only routes the mode.

## Call contract

```
python3 _substrate/integration/integration_strategist.py resolve \
    --kernel <path> --project-root <path> --can-standalone <yes|no|uncertain> \
    --host-probe '{"compiler":true,"project_build":true,"register_script":true,"runtime_registry":false,"reversibility_net":true}' \
    --cache <exp_dir>/integ_cache.json --trajectory <exp_dir>/genome.jsonl
# -> {method, build_fidelity, reversible, eval_mechanism, abstained_from, rationale, cache_key}
```

The workflow then runs the returned method: `standalone` → existing `build.sh`/`run.sh`;
`embedded_dispatch` → `embedded_eval.js` sequence; `embedded_inplace` → project-native
build + backup/restore; `registry_dispatch` → python register/unregister.

## The three axes (now all universal)

| axis | component | routed quantity | stamp |
|---|---|---|---|
| profiling | profiling-strategist | analysis method (ncu/perf/static) | confidence |
| verification | verification-strategist | verification depth (reference/smoke/compile/static) | confidence |
| **integration** | **integration-strategist** | **build mode (standalone/embedded/registry)** | **build_fidelity + reversible** |

## Files

- `integration_strategist.py` — selector + stamper (deterministic core + autonomy hook)
- `integration_registry.json` — ladder + preference + stamping
- `tests/test_strategist.sh` — 18 assertions (no GPU/project needed)

## Scope

The component is built and verified. **Wiring it into the 23 standalone-default
workflows is a separate, larger step** (each must gate its build/eval step on the
method, switching to `embedded_eval.js` when non-standalone) — more invasive than the
profiling INJECT (a guidance sentence) because it changes the build flow, not just a
prompt. The 6 already-embedded workflows don't need it (they hardcode their mode).
