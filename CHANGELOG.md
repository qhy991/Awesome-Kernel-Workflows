# Changelog

All notable changes to Awesome-Kernel-Workflows are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this
project adheres to [Semantic Versioning](https://semver.org/). See `AGENTS.md`
for the versioning policy.

## [Unreleased]

### Fixed

- **Faithful sol-execbench evaluation.** The solution packer now emits the
  supported `cuda_cpp` language enum. All five opted-in workflows preserve the
  benchmark environment and optional `--definition`, advertise the CLI in their
  host probe, and explicitly select `sol_execbench_solution` before standalone
  classification. The strategist fails closed when that preferred harness is
  unavailable.
- **#71 AKO4X zero-candidate rounds.** Round-level evaluation accounting is
  initialized before hypothesis dispatch and aggregates per-hypothesis results,
  so the zero-candidate path no longer references an undefined variable.
- **#104 worktree isolation.** AKO4X now selects `fresh-process` when KerSor
  reports that the runtime workspace is not a Git repository; Git-backed
  manifests declare the capability they require.

### Changed

- **#72 sol-execbench pool.** Added KDA and KernelBlaster to the opt-in pool,
  including solution-contract prompts, pack/run/parse evaluation, and declared
  runtime arguments; the pool now contains five workflows.
- **#73 integration vocabulary.** Replaced dead catalog values
  `external_harness` and `project_native` with supported vocabulary and removed
  the corresponding KerSor lint debt.

## [0.11.0] - 2026-07-09

### Added

- **Manifest `routing.emits[]` / `routing.consumes[]` for cross-DSL algorithmic
  priors.** New optional informational tags declare which workflows produce or
  honor the algorithmic-priors evidence class (partition strategy / bound
  classification / numeric floor) that survives a portable-DSL → backend-native
  escalation. `AKO4X/manifest.yaml` declares `routing.emits:
  [algorithmic_priors]`; `CUDAAgent/manifest.yaml` declares `routing.consumes:
  [algorithmic_priors]`. `docs/manifest-schema.yaml` documents both fields.
  These are informational only — consumed by KerSor's audit tooling, not by AKW
  dispatch. Files: `AKO4X/manifest.yaml`, `CUDAAgent/manifest.yaml`,
  `docs/manifest-schema.yaml`.
- **`CUDAAgent` Implement-phase prompt addendum for cross-DSL priors.** When a
  Triton / TileLang workflow's transfer object carries `validated_win`
  (partition strategy: `split_k` / `stream_k` / `persistent_kernel`),
  `bottleneck` (bound class: `compute_bound` / `memory_bound` / `latency_bound`),
  or `metric_contract` (numeric floor) items, the CUDAAgent Implement doer now
  honors them as the starting algorithmic shape of its first candidate — and is
  explicitly instructed to IGNORE tile shapes, warp counts, `num_stages`,
  `cluster_shape`, or any other fine schedule from the handoff, because those
  are Triton-compiler operating points that do not transfer to hand-tuned CUDA.
  Files: `CUDAAgent/cuda-agent-kernel-optimization.js`. Upstream design (in
  KerSor):
  `docs/superpowers/specs/2026-07-09-triton-first-cuda-escalation-priors-design.md`.

### Changed

- **`AGENTS.md`: added the "non-negotiable" workflow-code rules** to stop authoring
  drift at the source. New agents building/editing a workflow are now told, with
  the enforcing CI guard named for each: shared helpers are single-sourced in
  `_meta/scaffolding/` (never hand-edit the `BEGIN/END inlined` blocks — edit the
  SSOT + run the codemod); eligibility is manifest `routing.accepts`, not the
  retired `WORKFLOW_SUITABILITY`/`assertWorkflowSuitability`; and the runtime
  sandbox constraints (no `import`, no `Date.now`/`Math.random`, `agentRetry`-wrap,
  `--artifact/--problem/--out`, writes to `args.exp_dir`). Files: `AGENTS.md`.

### Fixed

- **`Agent.md`: corrected stale eligibility guidance.** The parameter-naming
  section instructed authors to emit `WORKFLOW_SUITABILITY` +
  `assertWorkflowSuitability()`, which was superseded by manifest
  `routing.accepts` + KerSor selector enforcement (issue #24) and is now forbidden
  by the generator. Rewritten to match. Files: `Agent.md`.

## [0.10.0] - 2026-07-08

### Added

- **sol-execbench as a first-class integration method.** Adds
  `sol_execbench_solution` to the shared `_substrate/integration` registry +
  strategist, gated on a new `sol_execbench_cli` host capability (hosts without
  sol-execbench resolve exactly as before — S9b/S9c cover this). A
  can-standalone=`no` kernel on a sol-equipped host now routes to
  `sol_execbench_solution` instead of throwing `derive_adapter`. New deterministic
  substrate scripts `_substrate/integration/pack_sol_candidate.py` (candidate
  kernel → sol-execbench `solution.json`; fails loudly on a bare kernel with no
  torch binding) and `parse_sol_bench.py` (bench.jsonl → per-workload
  `speedup_factor` geomean, `SPEEDUP=/STATUS=/WORKLOADS=` line). New
  `_substrate/embedded/sol_execbench_eval.js` substrate (pack → run → parse eval
  plan + proposal contract) inlined into CUDAAgent, ARGUS, and Generalist via
  `scripts/patch-sol-execbench-eval.js`; each gains an `IS_SOL` proposal + eval
  branch mutually exclusive with `IS_EMBEDDED`. Manifests advertise
  `sol_execbench_solution` in `routing.integration_patterns` (+ the 7 `sol_*` args
  in CUDAAgent's `all_args`). Unblocks KerSor routing of FlashInfer-Bench /
  sol-execbench tasks. Additive and backward-compatible: all behavior gated on
  `IS_SOL` / `sol_execbench_cli`; standalone/embedded paths byte-unchanged.

## [0.9.0] - 2026-07-07

### Added

- **公用组件提炼 — 7 SSOT components extracted + propagated.** A systematic
  audit of 32 workflows found ~100 instances of byte-identical code copy-pasted
  across files (with drift). Each was extracted to a `_meta/scaffolding/<name>.js`
  SSOT + a `scripts/patch-<name>.js` codemod + a guard test (drift detection):
  - **backend-axis** (#62): `normalizeSuitabilityValue` / `resolveBackendAxis` /
    `driverPath` / `driverSh` — 15 workflows wrapped (4 drifters resolved;
    AccelOpt kept as documented semantic divergence).
  - **substrate-invocation** (#63): `substrateAntiCheat` builder — SSOT for the
    anti_cheat.py CLI flag schema (the #42 root-cause fix).
  - **arg-guard** (#65): `__unwrapArgs` — 32/32 now byte-match the SSOT (5 older
    copies re-synced in #69, including AscendC which gained the key=value regex).
  - **embedded-eval** (#65): `EMBEDDING_CONTRACT` + `__embeddedEvalPlan` — 6/6.
  - **typed-args** (#66): KerSor ②③ channel block (`EXPERIENCE_EXCERPTS` /
    `ATTEMPT_EVIDENCE` / `FAILED_STRATEGY_IDS` + helpers) — propagated from 5 to
    all 33 workflows (contract completeness; consts degrade to null/[] when
    absent; surfacing in prompts is a per-workflow follow-up).
  - **turn-timeout** (#67): `withTurnTimeout` watchdog — propagated from 3 to all
    33 workflows (capability available; call-site activation is per-workflow).

### Fixed

- **#42 anti_cheat flag drift** (#63): 17 workflows passed `--kernel`/`--result`
  to anti_cheat.py (argparse requires `--source`/`--metrics`); the check silently
  failed for every candidate. Fixed everywhere via codemod; guard test prevents
  recurrence.
- **StitchCUDA anti_cheat missing --metrics** (#64): the call was `--source` only
  (argparse would reject); fixed to pass `--metrics ${buildOut}.run.json`.
- **backend-axis drift resolved** (#68): AKO4X/AdaExplore/Astra turned out to be
  byte-identical to the SSOT after the ①/③ merges — wrapped.
- **arg-guard drift resolved** (#69): 5 workflows had an older `__unwrapArgs`
  (AscendC lacked the key=value regex; 4 had a slightly older regex). Re-synced
  to the canonical; 0 drift remaining.

## [0.8.0] - 2026-07-07

### Added

- **Kernel-as-file authoring contract (#58/#59/#61).** AKO4X, KSearch, and
  KernelFoundryDx candidate-emission agents now write the kernel to an absolute
  `${exp_dir}/variants/<id>.<ext>` path and return `variant_path` (the file is
  the single source of truth inside the workflow; the `code` string is a
  display/compat payload that may truncate for >20KB kernels under the
  StructuredOutput cap). All internal consumption (smoke/bench/eval/archive)
  reads the full source from the path — `code.substring(0, 4000)` is orientation
  only. Losing candidates now persist to `exp_dir/variants/` before worktree
  teardown (#58). Returns carry `best_kernel_path` alongside `best_kernel_code`.
  (Orchestrator consumer-side: KerSor `materialize-best-kernel.sh` prefers the
  path, falls back to the code string, and flags `truncation_detected` when both
  are present and the file is larger — a live truncation detector turning the
  ~20KB cap from speculation into measured data.)

### Changed

- **AKO4X `Workflow()` guaranteed return (#56).** The multi-round loop is
  wrapped in try/catch; an abnormal exit (e.g. a bench `agentRetry` exhausting
  retries after a deterministic ptxas hang) surfaces as `dispatch_failed` +
  `failure_reason` in the return, so the orchestrator always gets a structured
  result — was: a thrown error leaving `output.json` blocked for 6+ hours.
  Bench + smoke `agentRetry` retries reduced 5→1 (deterministic hangs no longer
  retried 6× at harness-timeout cost).
- **AKO4X honest convergence (#57).** Zero-candidate rounds (every impl agent
  failed to spawn — e.g. worktree creation in a non-git dir) no longer tick
  `consecutive_no_improve`; the convergence guard requires ≥1 evaluated
  candidate across the session, else exits `dispatch_failed`/`round_empty`.
  Returns carry `round_empty` so the orchestrator routes the next round to a
  different workflow.
- **KernelFoundryDx directive→mutation binding (#60).** `args.mandatory_directive`
  pins one island per iteration (rotating) to the directive as a REQUIRED prompt
  section, with a cheap textual acceptance gate (the candidate's
  `change_summary` must reference the directive, else rejected before eval). The
  hint library is seeded from the directive. KerSor `resolve-args.sh` sources
  `mandatory_directive` from `contract.env.mandatory_directive`; the KFdx
  manifest declares it in `routing.all_args`.

## [0.7.0] - 2026-07-05

### Added

- **`_substrate/code_integrity.py` — code-integrity gate (#52).** Deterministic
  static gate rejecting truncated source (unbalanced braces / dangling opener —
  the 018 KSearch 6/6 truncation) and stub bodies (empty/placeholder — the
  L2-054 whole-file-Write stub) before a candidate's "speedup" enters memory.
  Wired into KSearch per-attempt verify; KSearch gen prompt gained a
  patch-first / no-truncation contract. 11 unit tests.
- **`_substrate/knowledge/sm100-blackwell.md` (#53, split).** Blackwell/sm_100
  reference (tcgen05 vs Hopper wgmma, TMEM budget, M-packing prerequisite). The
  NVIDIA-only `arch_lint.py` was reverted — arch-mismatch gating is done
  vendor-neutrally at the KerSor injection layer (KerSor #70).
- **`routing.seed_contract` on every workflow manifest (#55).** Each of 32
  manifests + the llamacpp-metal variant declares `greenfield` / `iterative` /
  `hybrid`, classified by reading each solver's `.js`. KerSor collects it to
  gate/warn when a provided seed would be silently discarded (KerSor #74).

### Changed

- **STARK RNG no longer falls back to `Math.random()` (#50).** Always seeded
  (`args.rng_seed`, else a fixed deterministic seed); the forbidden token is
  gone so KerSor's catalog scan no longer marks it `known_broken`.
- **KernelAgent verifier tolerance is spec-driven (#50 doer half; KerSor #73).**
  `args.rtol/atol` > dtype-aware default (fp16/bf16 → 1e-2, fp32 → 1e-3,
  fp64 → 1e-5) instead of a hardcoded 1e-3 (the 024 false-reject of 4/4 correct
  bf16 candidates); verify + reverify now capture a `LATENCY_MS` measured-latency
  signal instead of pass/fail only.

### Fixed

- `AutoMegaKernel/manifest.yaml` unquoted-colon scalar (`pruning_strategy`) that
  aborted KerSor catalog generation mid-parse.
- `kernelagent-triton-synthesis` reverify record typo (`candidate.appach` →
  `candidate.approach`).

## [Unreleased] — historical (pre-0.7.0, never version-sectioned)

### Added

- **`actionable_hint` mandatory across the evidence contract (#48).** An
  insight is no longer just a claim (what we observed) — it must carry an
  `actionable_hint` (what the next round should do about it).
  - `_substrate/evidence_schema.py`: `_validate_item` rejects an insight with
    an empty `actionable_hint`; the TEMPLATE documents the field.
  - `_substrate/profiling/perf_to_evidence.py`: new `_hint_for(bclass)` maps
    memory/compute/latency bottleneck classes to concrete next-steps; every
    emitted insight (bottleneck, validated_win, failed_strategy) + the
    channel-3 transfer_items carry it.
  - Generalist + AccelOpt emitters: `roundInsightRaw` derives the hint key
    from `bclass`; `insightItems` carry an "apply this learning" hint.
  - Out of scope: WarpSpeed lessons use a separate schema
    (type/mechanism/scope via its own lessons DB, not evidence_schema-validated).
  - New test: `_meta/tools/test/actionable-hint-contract.test.js` (5 assertions);
    S11 (evidence_schema validates perf_to_evidence output) stays 21/0.

- **`fair_baseline_id` → `baseline_id` output contract (#32).** CUDAAgent,
  KDA, AKO4X, KernelFoundry declare `fair_baseline_id` in `routing.all_args`
  (so KerSor #39(a) pushes the frozen `contract.env::baseline_id` into
  dispatch-args) and echo `baseline_id: args.fair_baseline_id || null` as the
  first field of the workflow return object (= `run-N/output.json`). KerSor
  #39(c) Check 2c already compares it to the frozen contract and vetoes
  wrong-baseline speedups; this is the missing AKW consumer. Input
  (`fair_baseline_id`, spec-frozen expectation) and output (`baseline_id`,
  workflow's measured axis) names differ deliberately. Other
  `resolveBackendAxis` workflows adopt via the same 2-line pattern when a
  session surfaces the need.

- **profiling-strategist `used_tools` contract field + `--deny-tools` (#36).**
  `perf_heuristic`'s contract is "no Nsight tools, use harness timing" — now
  explicit: the decision carries `used_tools` (`native_profiler` = primary +
  available `optional_tools` like nsys/cuobjdump; `perf_heuristic`/`static` =
  `[]`, even when Nsight tools are probed available). `cuobjdump` added to
  `PROBE_TOOLS` (was never probed). New `--deny-tools` arg makes the declarative
  "no Nsight this run" (previously only implicit via `ncu_binary=""`) a canonical
  contract: a denied primary tool forces `perf_heuristic` + `used_tools=[]`
  regardless of host probe. Tests S12–S14.

- **KSearch cycle checkpoint — write + resume (#43).** The 5 in-memory state
  vars (decisionTree, solutionDb, bestSolution, bestMetric, cycleCount) are now
  persisted to `${EXP_DIR}/checkpoint.json` at each cycle end (mechanical agent)
  and restored at startup, so a crashed search resumes at the next cycle instead
  of losing cycles 1..N. `runtime_metadata.checkpoint_written_at` is a static
  loop-counter-derived marker (no `Date.now()`) for postmortem. The API-exponential-
  backoff half of #43 was skipped — the runtime exposes no `sleep()` and
  `setTimeout`-as-blocking-sleep would hang under a non-real-time source; spun
  off as issue #47 (runtime-layer).

- **GemmPTX workflow `GemmPTX/`**. Adds a GEMM-specific CUDA/CuTe/CUTLASS
  optimizer that works from hardware census and PTX/SASS instruction evidence:
  candidates must compile, pass correctness, and prove the expected
  `mma.sync` / `wgmma.mma_async` / TMA / `tcgen05` path via disassembly before
  benchmark/profile evidence can promote them. This gives users a workflow for
  instruction-path GEMM tuning while explicitly avoiding generic compute-bound
  claims. The workflow now ships a local `gemmptx-instruction-evidence` skill
  with architecture/instruction mapping and PTX/SASS evidence gates; count/badge
  32 → 33.
  (`GemmPTX/`, `README.md`, `README.zh-CN.md`,
  `_meta/tools/test/gemmptx-contract.test.js`, `badges/workflows.json`)
- **AutoMegaKernel adapter workflow `AutoMegaKernel/`**. Adds the first strict
  external-harness adapter in AKW: `automegakernel-megakernel-optimization.js`,
  bilingual README, manifest, and a contract test. The workflow requires an
  existing AutoMegaKernel checkout and delegates authoritative ScheduleConfig /
  `kernel_knobs` search, validate-before-launch, correctness, latency/roofline
  evidence, and keep/revert to AMK (`amk propose/eval/loop/autoresearch`), so it
  is not a standalone CUDA optimizer or AMK reimplementation. Count/badge 31 →
  32.
  (`AutoMegaKernel/`, `README.md`, `README.zh-CN.md`,
  `_meta/tools/test/automegakernel-adapter-contract.test.js`,
  `badges/workflows.json`)
- **Canonical Ascend/AscendC workflow `AscendC/`** (#16, P0). A first-class,
  Ascend-native catalog entry (`ascendc-kernel-optimization.js` + README EN/zh +
  `manifest.yaml`) derived from the proven session-local variant evolved across
  910b-exp sessions. Targets AscendC on Ascend 910B via msprof and the substrate
  `ascend` backend (`ascendc_direct_launch`), so Ascend tasks no longer STALL at
  workflow selection and need not re-evolve a session-local variant every run.
  Count/badge 30 → 31.
- **`agentRetry` + null-guard default scaffolding across all `agent()`-based
  workflows** (#17). New canonical helper `_meta/scaffolding/agent-retry.js` +
  codemod `scripts/add-agent-retry-scaffolding.js` (string/template/regex-aware)
  wrap every `agent()` call (573 sites / 31 files) in a bounded retry and null-guard
  the dereference points, so a transient API 429 / agent-skip no longer crashes the
  run. KDA also gains a turn-boundary directive in its Implement prompt; the new
  AscendC workflow bakes in turn-boundary + per-file Bash write + NO HARNESS
  MANIPULATION.
- **`agentRetry` fail-safe default + null-safety enforcement** (#20, P1). The
  inlined `agentRetry` now THROWS an attributable error after exhausting retries
  on a null-returning agent, instead of returning null — so a transient 429 /
  terminal failure aborts the round cleanly with a recorded reason rather than
  crashing later at an unguarded dereference (`diag.bottleneck_class`,
  `impl.code`, …). Callers that intentionally degrade opt out with
  `{ allowNull: true }` (migrated at ~238 optional-probe sites via a new codemod
  `--allow-null` mode; results consumed inside `parallel()` need no opt-out —
  `parallel()` resolves a thrown thunk to a null slot). A new `--refresh` mode
  propagates canonical-helper edits to the 31 inlined copies idempotently.
  Enforcement: coded linter `scripts/check-agent-retry-guards.js` + `node:test`
  wrappers (`_meta/tools/test/agent-retry-*.test.js`) fail on a bare `agent()` or
  an unguarded `allowNull` dereference; `_tools/generate-workflow.js` Phase-4 /
  Validate now emit and check the scaffolding so generated workflows are born
  safe; `_meta/templates/` + `_templates/` skeletons retrofitted.
- **Ascend routing on the backend-agnostic universal workflows** (#16). `Generalist`
  and `KDA` (both `method_supported_backends: any`, `portability: clean`) now
  declare `ascendc`/`ascend` and route Ascend through the substrate ascend backend
  (faithful but simplified). `InPlacePatch` is intentionally NOT widened — it is
  `vendor_locked`/`intrinsic_to: nvcc/hipcc` with no Ascend (bisheng) path.
- **Manifest `routing:` blocks (KerSor metadata consolidation).** All 31 routable
  `.js` entrypoints now declare selector routing in `<Workflow>/manifest.yaml`
  (`variants[].routing` for multi-entrypoint dirs). Schema documented in
  `docs/manifest-schema.yaml`.
- **`scripts/validate-manifests.sh`** — every workflow `.js` must have manifest
  coverage; **`scripts/count-workflows.sh`** counts `.js` entrypoints (badge **31**).
- **GitHub CI** (`.github/workflows/ci.yml`) — count/badge sync + manifest validation.

### Changed

- **Workflow count unified to 31** — badge matches `generate-catalog.sh` scan
  (was 30 directory-based count; `LlamacppEmbeddedSearch` has two entrypoints).
- **Deprecated `_meta/manifests/*.yaml` and `_manifests/*.yaml` removed**; SoT is
  per-workflow `manifest.yaml`. Schema reference moved to `docs/manifest-schema.yaml`.

### Fixed

- **CUDAAgent verify phase routes artifacts to `${EXP_DIR}/verify/` (#37).**
  The verify/test loop ran in the project-root CWD, so the agent created
  `.verify_*/` dirs + stray files (verify_task, verify_candidate3,
  test_harness.py, kernel.py, test_kernel.py) there — 8–20 strays per round
  into the user's tree. The standalone Verify prompt now directs all authored
  artifacts to `${EXP_DIR}/verify/attempt-N/` (user-provided
  compile/test/profile commands still run where they expect). Runtime stray —
  a prompt directive, not fully enforceable; wsr-analyze should confirm the
  drop. (Earlier mis-attribution to AKO4X `ITERATIONS.md` corrected: that's a
  separate hygiene fix, #38.)
- **AKO4X proposals evidence-pointer references the real iteration-log path
  (#38, hygiene).** The template said `<ITERATIONS.md line>` but no such file
  is written (the log lives at `${EXP_DIR}/round-logs/round-N-iterations.md`);
  a bare reference could prompt the agent to create a stray `ITERATIONS.md`.
  Pointed at the real path. Added `stray-files-static-guard.test.js` (static
  bare-path guard — does NOT catch runtime strays, the #37 scope).

- **CUDAAgent `TARGET_SPEEDUP` handles "none" (explore mode) (#41).**
  `args.target_speedup || 1.05` kept the truthy string "none" when explore mode
  passed `target_speedup="none"`, breaking the numeric sites (NaN division /
  comparison). Now parses to a positive number, else null = no target (explore:
  run to MAX_TURNS / stagnation); the three numeric sites guard `null`. Missing
  keeps the 1.05 default for back-compat.
- **KSearch `anti_cheat.py` calls use `--source`/`--metrics` (#42).** The two
  calls passed `--kernel`/`--result`, but the script's argparse requires
  `--source`/`--metrics` — argparse rejected the call and the anti_cheat check
  silently failed. Lagged the #25 substrate sync. (`integration_strategist.py
  --kernel` is correct and preserved — that script accepts `--kernel`.) Same
  flag-mismatch exists in KDA/KernelFoundry/KernelFoundryDx/ReGraphT — propagate
  separately.

- **Wall-clock watchdog for ARGUS/KSearch + KSearch run-level circuit breaker
  (#30, #31a).** `withTurnTimeout` (per-turn `Promise.race(setTimeout)` cap,
  parity with CUDAAgent #12/#14) extracted to `_meta/scaffolding/turn-timeout.js`
  and inlined into ARGUS (eval-bearing turns) and KSearch (Generate chain +
  run-level `RUN_STAGNATION_LIMIT` breaker: stop early after N consecutive
  cycles with no global-best improvement). A hung non-eval `agent()` turn no
  longer stalls the run indefinitely.
- **load-driver degrades to legacy path on transient agent failure (#31b).**
  KDA, KernelFoundry, ReGraphT: a transient `load-driver` null (sustained 429
  after retries) now warns + continues without idioms instead of aborting the
  round; `present===false` still throws. Downstream DRIVER field derefs are
  guarded behind the non-null branch.

- **Collapsed residual profiling coupling in driver-backed workflows.** `Generalist`
  now profiles the baseline, current-best, and candidate attempts through a shared
  driver Layer-A envelope, so Triton/other driver paths no longer render legacy
  `ncu_command`/benchmark prompts. `AKO4X` now emits a uniform `driver-profile-*`
  envelope for perf-heuristic runs and keeps throughput normalization in the
  substrate profiling normalizer, uses neutral profile wording/artifact directories
  on driver-backed prompts, and the AKO4X Triton dry-run guard now checks
  case-insensitive `ncu` leaks while asserting the `perf_to_evidence.py` path.
  (`Generalist/generalist-kernel-optimization.js`,
  `AKO4X/ako4x-kernel-optimizer.js`,
  `_meta/tools/test/ako4x-triton-dryrun.test.js`)

- **Removed stale post-migration doc references.** Substrate docs now use the
  `/kersor:optimize` command name and `KerSor/docs/transfer-object.md`; the agent
  guide points manifest authors at `docs/manifest-schema.yaml` instead of the
  removed `_manifests/schema.yaml`.
  (`_substrate/ARCHITECTURE.md`, `_substrate/SOLVER-SDK.md`, `Agent.md`)

- **Workflow runtime meta reference crash.** Top-level workflows and legacy
  templates now use a body-scope `WORKFLOW_NAME` constant instead of reading the
  exported `meta` object at runtime, preventing Claude Code Workflow dispatches
  from failing with `ReferenceError: meta is not defined`. The genome-report
  codemod now emits the same safe constant for newly patched workflows, and a
  regression test guards against reintroducing runtime `meta.*` references.
  (all top-level workflow JS files, `_templates/*.js`,
  `scripts/patch-genome-report.js`,
  `_meta/tools/test/runtime-meta-reference.test.js`)
- **Workflow args string/object drift.** All top-level workflows and workflow
  templates now inline the bare-script-safe `arg_guard` unwrap before reading
  `args`, so Workflow dispatches that pass JSON strings or `key=value` strings no
  longer become empty-arg rounds. `patch-arg-guard.js` now emits the inlined guard
  instead of a static import, and a regression test keeps generated workflows on
  the same contract. (all top-level workflow JS files, `_templates/*.js`,
  `_meta/templates/*.js`, `scripts/patch-arg-guard.js`,
  `_meta/tools/test/runtime-arg-guard.test.js`)

### Added

- **WarpSpeed: align with AKW v0.2 genome + KerSor dispatch.** `exp_dir` for
  `genome.jsonl` and report mirror; KerSor arg aliases (`compile_command`,
  `kernel_path`, `ggml_root`); inline genome self-report on phases and
  Screen/Confirm/Profile; manifest topology/inputs/fidelity fields.
  (`WarpSpeed/warpspeed-kernel-search.js`, `WarpSpeed/manifest.yaml`)

## [Unreleased] - feat/proactive-knowledge-fetch

### Changed

- **cuda-agent: proactive knowledge fetch on retries (pilot).** The Implement
  doer now, when retrying after a failed attempt (history non-empty), is told to
  FIRST run the search command from the `## Knowledge Tools (on-demand)` block
  KerSor injects (e.g. `query.py` for kernel patterns, `chub search` for API/Triton
  docs), read 1-2 pages, then implement — instead of only consuming the
  round-start `## Retrieved Context`. Best-effort (absent/off when retrieval is
  off; never blocks). Turns the workflow from a passive consumer of injected
  context into an active caller of the knowledge tools. Other workflows retain
  the passive model until similarly upgraded.
  (`CUDAAgent/cuda-agent-kernel-optimization.js`)

## [Unreleased]

## [0.2.1] - 2026-06-17

### Added

- **Real genome example** in `_meta/genome-trajectory-schema.md` — an actual
  `run-N/genome.jsonl` (fused RMSNorm via cuda-agent) showing per-phase richness,
  per-iteration `candidate_id`, and a measured `speedup`, plus a robust-parsing
  note (skip non-JSON lines).

## [0.2.0] - 2026-06-17

### Added

- **Genome / trajectory self-report contract** — `_meta/genome-trajectory-schema.md`
  defines a lightweight, append-only `${exp_dir}/genome.jsonl` a running workflow
  emits so it is observable in real time (stage sequence + per-iteration outcomes),
  with an explicit trust boundary (work-plane / forgeable — for observability and
  the recombiner, never a loop-completion trust anchor).
- **Workflow tool storage & observability reference** —
  `_meta/workflow-tool-storage-and-observability.md` documents how the Claude Code
  `Workflow` tool stores state and what is observable from outside a running
  workflow (the factual basis for the self-report), marking each claim documented
  vs inferred.
- **`scripts/patch-genome-report.js`** — idempotent codemod that injects a generic
  per-`phase()` entry scribe; used to bootstrap observability and as the fallback
  for not-yet-upgraded / newly generated workflows.

### Changed

- **All 30 workflows now emit a rich, doer-written genome line per phase.** Each
  phase's primary doer agent appends one result-bearing line to
  `${exp_dir}/genome.jsonl` as its final action — written AFTER the work, so it
  carries real outcomes (`technique` / `speedup` / `candidate_id` /
  `status: done|error`) and emits once per loop iteration (per-iteration
  trajectory). This replaces the content-free entry scribe (which paid a full
  agent per phase to write only `"entered"`). Agentless phases and
  secondary/driver/passthrough helpers are not instrumented; the doer's task and
  return schema are unchanged; a failed append never breaks the workflow.
- **`_tools/generate-workflow.js`** notes that the genome scribe is added by the
  codemod post-generation (newly generated workflows inherit the entry scribe and
  can be upgraded to the rich doer-written form afterwards).

## [0.1.0] - 2026-06-17

- **Versioning & changelog introduced.** `AGENTS.md` now defines the SemVer +
  Keep-a-Changelog policy; the version lives in `VERSION`. The workflow library as
  it stood before this point (the catalogued workflows + substrate/templates/tools)
  is the `0.1.0` baseline; its prior evolution is recorded in the git history.
