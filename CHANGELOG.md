# Changelog

All notable changes to Awesome-Kernel-Workflows are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this
project adheres to [Semantic Versioning](https://semver.org/). See `AGENTS.md`
for the versioning policy.

## [Unreleased]

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
