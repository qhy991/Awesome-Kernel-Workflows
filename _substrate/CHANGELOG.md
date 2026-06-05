# Changelog — `_substrate/`

All notable changes to the KerSor Solver SDK substrate. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2026-06-05] Backend Driver Foundation (P1 + P2)

Branch: `dev/solver-substrate` · Plan:
[`docs/superpowers/plans/2026-06-05-backend-driver-foundation.md`](../docs/superpowers/plans/2026-06-05-backend-driver-foundation.md)
· Spec:
[`docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md`](../docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md)

Delivered via TDD (red → green → commit), subagent-driven execution. **36 tests,
all green** (`python3 -m unittest discover -s _substrate/tests -p 'test_*.py'`).

### Added

#### Test infrastructure (Task 0)

- `_substrate/tests/test_smoke.py` — discovery harness smoke test; one command runs
  the full suite.
- `_substrate/tests/test_diagnose.py` — golden characterization (NVIDIA byte-identity),
  null-rule, vendor-profile, occ-priority, and CLI returncode coverage.
- `_substrate/tests/test_anti_cheat.py` — golden characterization (CUDA no-file path
  byte-identity), `--vendor-patterns-file` integration (Metal MPS + C++ stub), and
  `load_vendor_patterns` unit tests.
- `_substrate/tests/test_validate_backend.py` — good + 4 bad fixture paths, non-dict
  JSON rejection, and exit-3 bad-args guard.
- `_substrate/tests/fixtures/{good,bad_*}/` — 10 JSON fixtures (1 conformant driver +
  4 single-defect variants) for L0 validation.

#### Backend driver axis (Tasks B1–B4)

- `_substrate/backends/validate_backend.py` — stdlib L0 structural validator for a
  driver directory (`manifest.json` + `idioms.json`). Imports live `method_gate.TABLE`
  for idiom method references. Exit codes: `0` ok · `1` L0 errors · `3` bad args.
- `_substrate/backends/REGISTRY.md` — human-facing driver index; seeded with `cuda`
  (`planned`).
- `_substrate/BACKEND-DRIVER-SDK.md` — six-file driver contract, manifest/idioms field
  tables, conformance ladder L0–L3, scoped Part-A substrate edits cross-reference, and
  **Deviations from the spec** (JSON not YAML; Python validator not Node/JSON-Schema;
  deferred `backend_id` canonical-form leg).

#### Design documentation

- `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md` — full backend-driver
  axis design spec (language × vendor translation layer).
- `docs/superpowers/plans/2026-06-05-backend-driver-foundation.md` — executable P1+P2
  implementation plan (1554 lines, 9 tasks, Definition of Done).

### Changed

#### `diagnose.py` (Tasks A1 + A2)

- **Vendor threshold profiles** — `PROFILES` map keyed by metrics `_vendor`
  (`nvidia` default; `apple` added). Existing NVIDIA inputs without `_vendor` stay
  **byte-identical** (golden tests lock this).
- **Null-rule for partial metrics** — when a two-sided discriminator (memory_bound /
  overhead_bound) needs both `dram_pct` and `sm_pct` but only one is measured, return
  `unknown` with `"insufficient measured metrics"` instead of inferring from a single
  operand.
- **Priority invariant preserved** — `latency_occupancy` (low `occupancy`) still wins
  over `compute_bound` when both signals are present (pinned by test).

#### `anti_cheat.py` (Tasks A3 + A4)

- **`--vendor-patterns-file`** — optional per-backend cheat-pattern file with
  `[fallback]` and `[skip]` sections (spec §5.3.3). Patterns merge with built-in CUDA
  lists; default no-file path stays **byte-identical**.
- **`load_vendor_patterns()`** — parses sectioned regex files; validates each regex with
  `re.compile` at load time (invalid patterns raise `ValueError`, not a late traceback in
  `static_flags`).

#### `validate_backend.py` hardening (code-review follow-ups)

- `_load_json` catches `OSError` / `PermissionError` — Bash callers always get JSON on
  stdout, never a raw traceback.
- Rejects non-dict top-level JSON (`null`, `[]`, `"str"`) with a clear L0 error.
- Exit-3 path (nonexistent driver dir) covered by test.

### Unchanged (explicit non-goals)

Per plan Definition of Done — these substrate scripts remain **byte-identical**:

- `method_gate.py`
- `evidence_schema.py`
- `memory_store.py`
- `verify_insight.py`

No workflow `.js` files were touched.

### Spec deviations (intentional; P3 must inherit)

| Topic | Spec says | Shipped |
|---|---|---|
| Machine-read driver files | `manifest.yaml` / `idioms.yaml` | `manifest.json` / `idioms.json` (stdlib `json`, no PyYAML/Node) |
| L0 validator | `_meta/tools/validate-backend.js` + JSON-Schema | `_substrate/backends/validate_backend.py` hand-rolled checks |
| `backend_id` L0 leg | three-way `== dir == normalizeSuitabilityValue(id)` | literal `backend_id == basename(dir)` only; canonical-form rules deferred to P3 |
| `bottleneck_classes` | 4 meaningful classes ∪ `{unknown}` | validator accepts `unknown` in the declared list (review fix #4) |

### Commit map (14 commits, `00b920b..HEAD`)

| Commit | Summary |
|---|---|
| `0bb3a8e` | docs: backend-driver axis design spec |
| `cec9010` | docs: backend-driver foundation P1+P2 plan |
| `13b6163` | Add `_substrate/tests` unittest discovery harness |
| `c0986b6` | Add diagnose golden + null-rule characterization tests (RED) |
| `b9914e1` | Add diagnose vendor profile + measured-operand null rule (GREEN) |
| `0681ac3` | Pin occ priority + apple vendor profile + CLI returncode in diagnose tests |
| `a81493e` | Add anti_cheat golden + vendor-pattern characterization tests (RED) |
| `28c9642` | Add anti_cheat `--vendor-patterns-file` (GREEN) |
| `9c7ac6b` | Validate vendor-pattern regexes at load time; add `load_vendor_patterns` unit tests |
| `31df6f5` | Add backend L0 validator fixtures and failing tests (RED) |
| `c3fde1c` | Add deterministic L0 backend driver validator (GREEN) |
| `23bba53` | Harden `validate_backend`: catch OSError, reject non-dict JSON, test exit-3 path |
| `33ddb05` | Add backend driver registry seeded with `cuda` |
| `a324efc` | Add backend driver SDK contract doc |

### Next (P3+, not in this release)

- First real driver directory: `_substrate/backends/cuda/` (six-file layout).
- Port `normalizeSuitabilityValue` canonical-form check into L0 validator.
- L1–L3 conformance smoke fixtures per `BACKEND-DRIVER-SDK.md`.
