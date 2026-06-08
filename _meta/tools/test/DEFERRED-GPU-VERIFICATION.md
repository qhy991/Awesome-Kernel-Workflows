# DEFERRED GPU VERIFICATION

Driver-gated tests in this directory exercise the rendered prompt /
command surface against fixture agent returns. They DO NOT execute the
substrate `build.sh` / `run.sh` / `profile.sh` against a live GPU.

Per `docs/superpowers/plans/2026-06-08-p5c-mid-complexity-batch.md`
§4.3, full end-to-end GPU verification (CUDA + Triton) is deferred to
**P5e** when CI lights up. Each row below records a workflow's dry-run
gate and the live execution it does NOT yet cover.

| Workflow | Dry-run test file | Driver path covered | Live GPU run deferred to | Notes |
|----------|-------------------|---------------------|--------------------------|-------|
| CUDALLM  | `cudallm-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | P5e | Asserts no CUDA-token leak, `run.sh` path, load-driver-first, `.py` source_ext in eval prompt, no LEGACY_FEATURE_CATALOG leak. Does not run @triton.jit kernels. |
| KDA      | `kda-triton-dryrun.test.js`     | triton (`_substrate/backends/triton`) | P5e | Asserts no CUDA-token leak, `run.sh` path, load-driver-first, no cuda-kernel-development binding in Inspect, no ncu-report-skill / warp-shuffle in Validate. Does not run @triton.jit kernels; per-candidate Layer-A envelope (build/run/profile/to_evidence/diagnose/anti_cheat) is dry-run only. |
| StitchCUDA | `stitchcuda-triton-dry-run.test.js` | triton (`_substrate/backends/triton`) | P5e | Asserts load-driver-first under triton backend_dir, full per-attempt Layer-A envelope label set emitted (driver-build/run/profile/to-evidence/diagnose/anti-cheat across both attempts of the fixture replan branch), and driver-run prompts reference triton `run.sh`. Does not run @triton.jit kernels; KernelBench harness wiring (intersectional-guarded for non-CUDA drivers) untested at runtime. |
| Astra    | `astra-triton-dryrun.test.js`   | triton (`_substrate/backends/triton`) | P5e | Asserts no CUDA-token leak, `run.sh` path, load-driver-first, no PyBind/CUDA vocab in setup-astra, `.py` source_ext in per-iteration evaluate kernel_path. Per-iteration Layer-A envelope (build/run/profile/to_evidence/diagnose/anti_cheat) is dry-run only. integration_mode='sglang' is intersectional-guarded for non-CUDA drivers and therefore not exercised on the triton path. |
| STARK    | `stark-triton-dryrun.test.js`   | triton (`_substrate/backends/triton`) | P5e | Asserts no CUDA-token leak (under rng_seed=42), load-driver-first under triton backend_dir, `run.sh` path in every driver-run prompt, ```python fence in plan-*/code-* dynamic-context-builder kernel snippets, and the full Layer-A envelope (build/run/profile/to-evidence/diagnose/anti-cheat) for both root eval (`-root` suffix) and each per-attempt eval (`-1`, `-2`). Does not run @triton.jit kernels; selectNode() ε-greedy tree exploration is fixture-pinned via rng_seed. |
