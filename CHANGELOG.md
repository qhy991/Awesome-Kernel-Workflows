# Changelog

All notable changes to Awesome-Kernel-Workflows are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this
project adheres to [Semantic Versioning](https://semver.org/). See `AGENTS.md`
for the versioning policy.

## [Unreleased]

### Added

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
