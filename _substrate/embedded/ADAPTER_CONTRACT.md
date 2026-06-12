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
