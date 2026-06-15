# Embedded-dispatch adapter contract

This is the **project-agnostic interface** that lets any AKW optimization workflow
evaluate a kernel that is *embedded* in a larger project (a `.cuh` / source file
that must be wired into the project's dispatch table and built by the project's
own build system), instead of compiled as a standalone translation unit.

A workflow stays oblivious to *how* a given project wires a candidate in. It only
speaks this contract. The per-project knowledge lives entirely in an **adapter**:
a single executable that implements the three verbs below. The reference adapter
is `scripts/llamacpp_register_variant.py` (llama.cpp ggml-cuda / `fattn.cu`).

## The three verbs

```
<adapter> register   --variant <name> --source <file> --project-root <path> [project params...]
<adapter> unregister --variant <name> --project-root <path>
<adapter> list       --project-root <path>
```

- `register`   — copy `--source` into the project, wire it into the dispatch
                 table under an env-gated entry named `<name>`, and (if the build
                 system needs a re-glob) refresh the build dir. Prints the env var
                 the binary reads to select this variant.
- `unregister` — remove every insertion for `<name>`, **byte-exact**. Idempotent:
                 unregistering an absent variant is a no-op success.
- `list`       — print the currently-registered variant names (one per line).

`--source` and `--project-root` are the two **generic** flags every adapter MUST
accept. The reference adapter also accepts their original names (`--cuh-src`,
`--ggml-root`) as aliases for backward compatibility.

## Project params

Anything a specific project needs beyond source+root (e.g. llama.cpp's `--dkq` /
`--dv` head dims, or `--cmake-build-dir`) is the adapter's own business. The
workflow passes these through verbatim from its `args` as an opaque
`register_extra_args` / `unregister_extra_args` string. The shared substrate never
parses them.

## Guarantees every adapter MUST honor

1. **Env-gated activation** — registering a variant does NOT change default
   dispatch. The variant runs only when its env gate (e.g. `KERSOR_VARIANT=<name>`)
   is set at runtime. Registered-but-unselected variants are safe to leave in
   place. This is what lets a fan-out workflow keep N variants registered while
   benchmarking them one at a time.
2. **Byte-exact reversibility** — all insertions are wrapped in marker comments
   and verifiable (md5 / diff) so `unregister` restores the touched files to their
   pristine bytes. The project must always return to pristine when a run ends.
3. **Idempotency** — re-`register` of the same name is a no-op-or-replace;
   `unregister` of an absent name succeeds quietly.

## Adding a new project

Write one executable that implements the three verbs above and the three
guarantees. Point a workflow at it via the `register_script` arg. No workflow code
and no substrate code changes. That is the whole extensibility story.

See `embedded_eval.js` for the shared evaluation sequence workflows run against any
conforming adapter.

## Integration modes

The adapter contract covers three integration modes a framework may require.
The KerSor `framework-integrator` agent derives the correct mode at runtime by
reading the framework source; the modes themselves are not per-framework code.

| Mode | Registration mechanism | Adapter form | Reversibility safety net |
|---|---|---|---|
| `embedded_dispatch` | Source-patch + rebuild (e.g. llama.cpp `fattn.cu` dispatch switch + `file(GLOB)` CMake) | Shell script implementing the three verbs above | Byte-exact file round-trip (`reversible_edit.py roundtrip`) |
| `embedded_inplace` | In-place patch, no registration mechanism | Same as `embedded_dispatch` | Same file round-trip |
| `registry_dispatch` | Python kernel-class registered into a runtime dispatch table (e.g. vLLM `ScaledMMLinearKernel.can_implement` + `register_linear_kernel()` — no C++ source touched) | Python snippet (register + unregister are import-side-effect, not file edits) | Namespace round-trip (`reversible_edit.py namespace-roundtrip`): register, assert presence, unregister, assert absence |

## Migration from hand-written adapters

The reference adapters `scripts/llamacpp_register_variant.py` and
`scripts/sgl_register_variant.py` are **deprecated**. A general LLM agent (the
KerSor `framework-integrator`) armed with the reversible-edit safety net and the
integration-mode methodology can derive each framework's adapter at runtime by
reading the framework's dispatch, build system, and harness source — and the
result is cached in KerSor's experience bank for reuse. The hand-written
adapters remain as **reference implementations** (showing the expected
complexity and the env-gate + ODR-suffix patterns), but are no longer the
intended integration path for new frameworks.

Per-framework edge cases the integrator playbook must cover:

- **SGLang**: kernels bind at static-init via `TORCH_LIBRARY` (no runtime
  dispatch table). An env-gate must wrap the `m.impl` line at library load.
- **llama.cpp**: `file(GLOB)` without `CONFIGURE_DEPENDS` requires explicit
  `cmake <build_dir>` after adding/removing `.cu` files. Variant symbols must
  be suffixed with `_<variant_name>` to avoid ODR violations when both
  original and variant headers are included in the same translation unit.
