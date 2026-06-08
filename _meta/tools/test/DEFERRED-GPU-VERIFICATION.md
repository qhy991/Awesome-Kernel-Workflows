# DEFERRED GPU VERIFICATION

## Summary

This directory contains 16 dry-run tests covering all retrofitted workflows in the
Backend Driver Axis initiative (P5a--P5e). These tests exercise the rendered prompt
and command surface against fixture agent returns on macOS, where no GPU is present.
They do NOT execute the substrate `build.sh` / `run.sh` / `profile.sh` against a
live GPU.

**Deferred GPU verification posture:** all 16 workflows listed below have been
validated structurally -- guard logic, driver dispatch, prompt-token seams (no CUDA
vocabulary leaking under non-cuda backends), Layer-A envelope emission (build / run /
profile / to-evidence / diagnose / anti-cheat labels), and driver-path correctness.
What remains deferred is end-to-end execution on real NVIDIA hardware: compiling
kernels with `nvcc`/Triton JIT, running them for correctness, profiling with `ncu`,
and verifying that `to_evidence.py` produces correct canonical metrics from real
profiler output. This is gated on a GPU CI runner (spec SS8.3, SS9.3) and is planned
for the hardware tier, not for the structural/dry-run tier that runs on every PR.

Additionally, three CI tiers complement these per-workflow dry-runs:

- **substrate-diff-guard** (`substrate-diff-guard.test.js`): asserts universal
  substrate scripts are byte-identical to the baseline SHA manifest.
- **driver-conformance-l0** (`driver-conformance-l0.test.js`): structural validation
  of `manifest.json` and `idioms.json` for each registered driver.
- **matrix-smoke** (`matrix-smoke.test.js`): cross-product of `matrix_eligible`
  workflows x matrix drivers, asserting guard-pass, correct driver dispatch, and
  Layer-A envelope structure.

Per `docs/superpowers/plans/2026-06-08-p5c-mid-complexity-batch.md` SS4.3, full
end-to-end GPU verification (CUDA + Triton) is deferred to the **GPU CI tier** when
hardware runners light up. Each row below records a workflow's dry-run gate and the
live execution it does NOT yet cover.

## Dry-run coverage table (16 workflows)

| Workflow | Dry-run test file | Driver path covered | Live GPU run deferred to | Notes |
|----------|-------------------|---------------------|--------------------------|-------|
| AccelOpt | `accelopt-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Vendor-locked pilot (P4). Asserts triton driver dispatch under `backend:'triton'`, `run.sh` path, load-driver-first, no CUDA-token leak in triton prompts. Does not run @triton.jit kernels; the beam-search evolution is fixture-pinned. |
| AdaExplore | `adaexplore-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Clean-tier MCTS solver (P5b, first batch). Asserts no CUDA-token leak, `run.sh` path, load-driver-first under triton backend_dir, full Layer-A envelope (build/run/profile/to-evidence/diagnose/anti-cheat). Does not run @triton.jit kernels; MCTS large-step/small-step selection is fixture-pinned. |
| KernelAgent | `kernelagent-cuda-dryrun.test.js` | cuda (`_substrate/backends/cuda`) | GPU CI | Clean-tier routing+parallel-seeds solver (P5b, first batch). Asserts cuda driver dispatch, `run.sh` path, load-driver-first under cuda backend_dir, full Layer-A envelope. Does not compile CUDA kernels; multi-seed parallel evaluation is fixture-pinned. |
| CUDALLM | `cudallm-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Asserts no CUDA-token leak, `run.sh` path, load-driver-first, `.py` source_ext in eval prompt, no LEGACY_FEATURE_CATALOG leak. Does not run @triton.jit kernels. |
| KDA | `kda-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Asserts no CUDA-token leak, `run.sh` path, load-driver-first, no cuda-kernel-development binding in Inspect, no ncu-report-skill / warp-shuffle in Validate. Does not run @triton.jit kernels; per-candidate Layer-A envelope (build/run/profile/to_evidence/diagnose/anti_cheat) is dry-run only. |
| StitchCUDA | `stitchcuda-triton-dry-run.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Asserts load-driver-first under triton backend_dir, full per-attempt Layer-A envelope label set emitted (driver-build/run/profile/to-evidence/diagnose/anti-cheat across both attempts of the fixture replan branch), and driver-run prompts reference triton `run.sh`. Does not run @triton.jit kernels; KernelBench harness wiring (intersectional-guarded for non-CUDA drivers) untested at runtime. |
| Astra | `astra-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Asserts no CUDA-token leak, `run.sh` path, load-driver-first, no PyBind/CUDA vocab in setup-astra, `.py` source_ext in per-iteration evaluate kernel_path. Per-iteration Layer-A envelope (build/run/profile/to_evidence/diagnose/anti_cheat) is dry-run only. integration_mode='sglang' is intersectional-guarded for non-CUDA drivers and therefore not exercised on the triton path. |
| STARK | `stark-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Asserts no CUDA-token leak (under rng_seed=42), load-driver-first under triton backend_dir, `run.sh` path in every driver-run prompt, ```python fence in plan-*/code-* dynamic-context-builder kernel snippets, and the full Layer-A envelope (build/run/profile/to-evidence/diagnose/anti-cheat) for both root eval (`-root` suffix) and each per-attempt eval (`-1`, `-2`). Does not run @triton.jit kernels; selectNode() epsilon-greedy tree exploration is fixture-pinned via rng_seed. |
| KSearch | `ksearch-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Asserts no CUDA-token leak, `run.sh` path, load-driver-first under triton backend_dir, ```python fence in every gen-*/eval-* prompt (driver lang_fence overrides args.language), and the full Layer-A envelope (build/run/profile/to-evidence/diagnose/anti-cheat) for both baseline-root (`-root` suffix) and each per-cycle-attempt eval (`-0-0`, `-1-0`). Does not run @triton.jit kernels; world-model decision-tree action selection is fixture-pinned via agentReturns map. |
| ReGraphT | `regrapht-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Asserts no CUDA-token leak, `run.sh` path, load-driver-first under triton backend_dir, ```python fence in every generate-*/evaluate-* prompt (driver lang_fence overrides args.language), and the full Layer-A envelope (build/run/profile/to-evidence/diagnose/anti-cheat) for both root (`-root` suffix, fired after build-regraph) and each per-attempt evaluate (`-0`, `-1`). Does not run @triton.jit kernels; Monte Carlo Graph Search path selection is fixture-pinned via agentReturns map. |
| KernelFoundry | `kernelfoundry-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Clean-tier evolutionary search solver (P5d). Asserts no CUDA-token leak, `run.sh` path, load-driver-first under triton backend_dir, driver lang_fence override, and the full per-(generation,island) Layer-A envelope (build/run/profile/to-evidence/diagnose/anti-cheat). Does not run @triton.jit kernels; multi-island evolution is fixture-pinned via agentReturns map. |
| KernelFoundryDx | `kernelfoundrydx-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Triton-only workflow (supported=[triton]; args.backend=cuda throws). Asserts no CUDA-token leak, `run.sh` path, load-driver-first under triton backend_dir, driver lang_fence override on the baseline-and-seed prompt headline, and the full per-(iter,island) Layer-A envelope (build/run/profile/to-evidence/diagnose/anti-cheat for both islands in iter 0). Does not run @triton.jit kernels; multi-island role-specialized evolution + hint-library reinforcement is fixture-pinned via agentReturns map. |
| KernelSkill | `kernelskill-cuda-dryrun.test.js` | cuda (`_substrate/backends/cuda`) | GPU CI | Clean-tier multi-skill solver (P5d). Asserts cuda driver dispatch, `run.sh` path, load-driver-first under cuda backend_dir, full Layer-A envelope (build/run/profile/to-evidence/diagnose/anti-cheat) for setup and per-iteration evals. Does not compile CUDA kernels; skill selection and routing is fixture-pinned via agentReturns map. |
| AKO4X | `ako4x-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | 6-backend manifest (triton/cuda/cute-dsl/tilelang/cpp/pytorch). Asserts no CUDA-token leak (nvcc/ncu word-isolated to ignore the hard-coded ncu-profiles workspace dir; __global__/__syncthreads excluded as language-agnostic examples in read-baseline), `run.sh` path, load-driver-first under triton backend_dir, `.py` source_ext in per-iter envelope kernel paths, full Layer-A envelope (build/run/profile/to-evidence/diagnose/anti-cheat) emitted in Iterate phase after bench, and substrate-collapse helpers (to_evidence.py from driver, diagnose.py + anti_cheat.py from `_substrate/`). Does not run @triton.jit kernels; two intersectional guards (ncu_binary + non-cuda; mode=3 + USE_DRIVER) are unit-tested separately in `ako4x-guard.test.js` and therefore not exercised on the triton dry-run path. |
| KernelBand | `kernelband-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Clean-tier 2-backend workflow (triton/cuda). Asserts no CUDA-token leak, `run.sh` path, load-driver-first under triton backend_dir, ```python fence in gen/eval prompts (driver lang_fence overrides args.language), full Layer-A envelope (build/run/profile/to-evidence/diagnose/anti-cheat) for setup baseline and each per-iteration eval (t1, t2), and phi-gate LEGACY_SATURATION_THRESHOLD fallback to 0.75 when driver omits saturation_threshold. Does not run @triton.jit kernels; bandit (cluster, strategy) selection is fixture-pinned via agentReturns map. |
| Generalist | `generalist-triton-dryrun.test.js` | triton (`_substrate/backends/triton`) | GPU CI | Substrate reference solver (clean tier, supported=[cuda]). Asserts no CUDA-token leak under triton driver, `run.sh` path, load-driver-first under triton backend_dir, full Layer-A envelope (build/run/profile/to-evidence/diagnose/anti-cheat) for both baseline setup (`-setup` suffix) and each per-iteration per-candidate eval (`-1-1`, `-2-1`). Does not run @triton.jit kernels; beam search (profile/diagnose/retrieve/gate/plan/impl/anticheat/learn/refute/verify-insight) is fixture-pinned via agentReturns map. Final workflow in the retrofit sequence (SS9.1 Phase 2). |
