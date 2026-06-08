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
