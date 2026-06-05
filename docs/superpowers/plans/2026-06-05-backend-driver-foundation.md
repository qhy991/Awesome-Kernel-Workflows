# Backend Driver Foundation (P1 + P2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the backend-driver substrate edits (diagnose.py vendor profile + measured-operand null rule; anti_cheat.py per-vendor patterns) and the deterministic driver-contract scaffolding (validate_backend.py, REGISTRY, BACKEND-DRIVER-SDK.md) — all behind golden + TDD tests, touching no workflow.

**Architecture:** Python stdlib `unittest` harness under `_substrate/tests/`; two substrate scripts edited behind characterization (golden) tests that lock NVIDIA byte-identity before adding new behaviour; a stdlib JSON-based deterministic backend validator. Deviations from the design spec (validator is Python not JS; machine-read driver files are JSON not YAML; no separate JSON-Schema files) are documented in the Architecture note with rationale (dependency-freedom, consistency with the stdlib-Python substrate).

**Tech Stack:** Python 3 stdlib (unittest, json, argparse, re); Markdown docs. No third-party packages.

---

## File Structure

Every file created or modified by this plan, with its one-line responsibility:

| Path | Action | Responsibility |
|---|---|---|
| `_substrate/tests/test_smoke.py` | create | Trivial smoke test proving the `unittest discover` harness runs green (Task 0). |
| `_substrate/tests/test_diagnose.py` | create | Golden + null-rule characterization tests for `diagnose.classify()` (Tasks A1, A2). |
| `_substrate/diagnose.py` | modify | Add `PROFILES` vendor lookup + measured-operand null rule (Task A2; one documented hunk pair). |
| `_substrate/tests/test_anti_cheat.py` | create | Golden + vendor-pattern characterization tests for `anti_cheat` (Tasks A3, A4). |
| `_substrate/anti_cheat.py` | modify | Add `load_vendor_patterns` + `--vendor-patterns-file`; thread extras through `static_flags`/`evaluate`/`main` (Task A4; one documented hunk group). |
| `_substrate/backends/__init__.py` | create | Empty package marker so the validator dir is importable in future (Task B1). |
| `_substrate/tests/fixtures/good/manifest.json` | create | Valid driver manifest fixture (Task B1). |
| `_substrate/tests/fixtures/good/idioms.json` | create | Valid driver idioms fixture (Task B1). |
| `_substrate/tests/fixtures/bad_backend_id/manifest.json` | create | `backend_id != dir` defect fixture manifest (Task B1). |
| `_substrate/tests/fixtures/bad_backend_id/idioms.json` | create | Companion idioms for the backend_id defect (Task B1). |
| `_substrate/tests/fixtures/bad_idiom_method/manifest.json` | create | Valid manifest for the idiom-method defect fixture (Task B1). |
| `_substrate/tests/fixtures/bad_idiom_method/idioms.json` | create | `methods` key not in `method_gate.TABLE` defect fixture (Task B1). |
| `_substrate/tests/fixtures/bad_metric_key/manifest.json` | create | Non-canonical `metrics` key defect fixture (Task B1). |
| `_substrate/tests/fixtures/bad_metric_key/idioms.json` | create | Companion valid idioms (Task B1). |
| `_substrate/tests/fixtures/bad_bottleneck_class/manifest.json` | create | Bogus `bottleneck_classes` entry defect fixture (Task B1). |
| `_substrate/tests/fixtures/bad_bottleneck_class/idioms.json` | create | Companion valid idioms (Task B1). |
| `_substrate/tests/test_validate_backend.py` | create | CLI tests asserting validator exit codes + error substrings (Tasks B1, B2). |
| `_substrate/backends/validate_backend.py` | create | Deterministic stdlib L0 driver validator (Task B2). |
| `_substrate/backends/REGISTRY.md` | create | Human-facing backend-driver index seeded with `cuda — planned` (Task B3). |
| `_substrate/BACKEND-DRIVER-SDK.md` | create | The (language × vendor) driver-contract companion doc (Task B4). |

**Harness layout decision (binding for the whole plan).** `_substrate/tests/` is a **non-package** directory: it has **no** `_substrate/tests/__init__.py`. Discovery uses `-s _substrate/tests` and each test prepends `_substrate/` to `sys.path`, so a tests-package marker is neither needed nor wanted. (This resolves the only cross-part inconsistency: Part B's original "create `_substrate/tests/__init__.py`" step is dropped here; `_substrate/backends/__init__.py` is still created because the validator dir is intended to be importable.) All runs use the single canonical command from the repo root, no `cd`:

```
python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v
```

**Canonical run commands.**
- Whole suite (acceptance gate): `python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v`
- Single file (no test package exists, so `_substrate.tests.test_x` will NOT import): e.g. `python3 -m unittest discover -s _substrate/tests -p 'test_diagnose.py' -v`

**Parallelism.** P2 (Part A) is independent of P1 (Part B) per spec Appendix A and may proceed in parallel once Task 0 lands. Within each part the tasks are sequential (red → green → commit).

---

## Task 0 — Establish the test harness

Create the non-package `_substrate/tests/` directory with a trivial smoke test, prove the canonical discover command runs green, and commit. Keep it minimal. (No `__init__.py` in `_substrate/tests/`.)

### Files
- **Create**: `_substrate/tests/test_smoke.py`

### Steps

- [ ] **Step 1: Create the directory + a trivial smoke test.**
  First create the directory (a raw shell heredoc/redirect will NOT create a missing parent; the Write tool does so automatically, but do this explicitly to be safe): `mkdir -p _substrate/tests`. Then write `_substrate/tests/test_smoke.py` with exactly:
  ```python
  import unittest


  class TestHarnessSmoke(unittest.TestCase):
      """Proves `python3 -m unittest discover -s _substrate/tests -p 'test_*.py'`
      runs green. This is a non-package tests dir (no __init__.py): discovery uses
      -s _substrate/tests and each real test prepends _substrate/ to sys.path."""

      def test_discovery_runs_green(self):
          self.assertTrue(True)


  if __name__ == "__main__":
      unittest.main()
  ```

- [ ] **Step 2: Run the canonical discover command + see it PASS.**
  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v`
  Expected: 1 test, `OK` (e.g. `Ran 1 test in 0.00s` then `OK`). This confirms the harness directory exists and discovery works before any substrate edits.

- [ ] **Step 3: Commit the harness.**
  Run:
  ```
  git add _substrate/tests/test_smoke.py
  git commit -m "$(printf 'Add _substrate/tests unittest discovery harness\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
  ```
  Expected: one new file committed; commit subject `Add _substrate/tests unittest discovery harness`.

---

## Task A1 — `diagnose.py` characterization (golden) tests FIRST

Lock the current `(class, evidence)` of `diagnose.classify()` for every all-measured case and every already-working partial case, run against the UNCHANGED `diagnose.py` and see them PASS (baseline lock). Then add the three new-behaviour assertions and run to see the two genuinely-wrong cases FAIL on current code. (Note the asymmetry below: `{dram:null, sm:80}` already returns `compute_bound` on current code — it is a *regression guard*, asserted PASS now, not a red test.)

### Files
- **Create**: `_substrate/tests/test_diagnose.py`
- **Test**: `_substrate/tests/test_diagnose.py` (this file; run via discovery)
- **Modify**: none in this task (`_substrate/diagnose.py` is read-only here)

### Steps

- [ ] **Step 1: Create the golden-baseline test file.** Write `_substrate/tests/test_diagnose.py` with the complete content below. Every expected `(class, evidence)` pair was captured from the live unchanged `classify()` and is byte-exact. `TestDiagnoseGolden` asserts the current behaviour (must PASS on unchanged code). `TestDiagnoseNullRule` asserts the decided spec §4.6 behaviour (two cases must FAIL on unchanged code; one is a regression guard that already passes). Both unit (direct import) and one CLI/integration test (shell out) are included.

```python
import os, sys, json, subprocess, unittest
SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

import diagnose


class TestDiagnoseGolden(unittest.TestCase):
    """Characterization: lock the CURRENT (class, evidence) for every all-measured
    case and every partial case that already works today. These MUST stay
    byte-identical after the §5.3 vendor-profile + null-rule edit (NVIDIA invariant)."""

    def test_memory_bound_both_measured(self):
        self.assertEqual(
            diagnose.classify({"dram_pct": 80, "sm_pct": 30}),
            ("memory_bound", ["dram 80% high, sm 30% low"]),
        )

    def test_compute_bound_both_measured(self):
        self.assertEqual(
            diagnose.classify({"dram_pct": 30, "sm_pct": 85}),
            ("compute_bound", ["sm 85% high"]),
        )

    def test_overhead_bound_both_measured(self):
        self.assertEqual(
            diagnose.classify({"dram_pct": 20, "sm_pct": 20, "occupancy": 0.9}),
            ("overhead_bound", ["both utilizations low (dram 20%, sm 20%)"]),
        )

    def test_unknown_both_measured_no_dominant_signal(self):
        # measured-both, no branch fires -> the :.0f / occ-None evidence MUST be byte-identical
        self.assertEqual(
            diagnose.classify({"dram_pct": 80, "sm_pct": 60}),
            ("unknown", ["no dominant signal (dram 80%, sm 60%, occ None)"]),
        )

    def test_latency_occupancy_partial_occ_only(self):
        # already-working partial case: dram/sm null, occ low
        self.assertEqual(
            diagnose.classify({"occupancy": 0.2}),
            ("latency_occupancy", ["occupancy 0.20 < 0.40 (launch/occupancy limited)"]),
        )

    def test_unknown_all_null(self):
        self.assertEqual(
            diagnose.classify({}),
            ("unknown", ["no profiler metrics available"]),
        )


class TestDiagnoseNullRule(unittest.TestCase):
    """Decided spec §4.6 rule: a two-sided branch (memory/overhead) fires only when
    BOTH dram and sm are measured; single-signal branches (latency_occupancy on occ,
    compute_bound on sm) may fire alone. New cases."""

    def test_high_dram_unmeasured_sm_is_unknown(self):
        # KEYSTONE: current code coerces sm->0.0 and WRONGLY returns memory_bound.
        self.assertEqual(
            diagnose.classify({"dram_pct": 80, "sm_pct": None}),
            ("unknown", ["no dominant signal (insufficient measured metrics)"]),
        )

    def test_high_sm_unmeasured_dram_is_compute_bound(self):
        # Regression guard: current code already returns compute_bound (sm>=70 fires on
        # the sm-alone branch). Stays compute_bound after the edit.
        self.assertEqual(
            diagnose.classify({"dram_pct": None, "sm_pct": 80}),
            ("compute_bound", ["sm 80% high"]),
        )

    def test_partial_sm_low_occ_high_is_unknown(self):
        # current code coerces dram->0.0 and WRONGLY returns overhead_bound.
        self.assertEqual(
            diagnose.classify({"dram_pct": None, "sm_pct": 30, "occupancy": 0.6}),
            ("unknown", ["no dominant signal (insufficient measured metrics)"]),
        )


class TestDiagnoseCli(unittest.TestCase):
    """Integration: the --metrics - stdin path prints the canonical JSON envelope."""

    def _run(self, metrics):
        return subprocess.run(
            [sys.executable, os.path.join(SUB, "diagnose.py"), "--metrics", "-"],
            input=json.dumps(metrics), capture_output=True, text=True,
        )

    def test_cli_memory_bound_envelope(self):
        r = self._run({"dram_pct": 80, "sm_pct": 30})
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(out["bottleneck_class"], "memory_bound")
        self.assertEqual(out["evidence"], ["dram 80% high, sm 30% low"])

    def test_cli_high_dram_unmeasured_sm_is_unknown(self):
        r = self._run({"dram_pct": 80, "sm_pct": None})
        out = json.loads(r.stdout)
        self.assertEqual(out["bottleneck_class"], "unknown")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the golden baseline + see it PASS (lock the baseline).**
  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_diagnose.py' -v -k Golden`
  Expected: 6 tests, all OK. Final line: `OK` (e.g. `Ran 6 tests in 0.00s` then `OK`). The `TestDiagnoseGolden` class confirms the unchanged `classify()` matches every captured string byte-for-byte before any edit.

- [ ] **Step 3: Run the full new file + see the null-rule cases FAIL on current code (state the exact wrong class).**
  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_diagnose.py' -v`
  Expected: failures in `TestDiagnoseNullRule` and the matching CLI test. Exact failures on the UNCHANGED code:
    - `test_high_dram_unmeasured_sm_is_unknown` → FAIL: current code returns `('memory_bound', ['dram 80% high, sm 0% low'])` (because `s = sm or 0.0` makes sm 0, so `d>=70 and s<50` fires); assert wanted `('unknown', ['no dominant signal (insufficient measured metrics)'])`.
    - `test_partial_sm_low_occ_high_is_unknown` → FAIL: current code returns `('overhead_bound', ['both utilizations low (dram 0%, sm 30%)'])` (because `d = dram or 0.0` makes dram 0, so `d<40 and s<40` fires); assert wanted `('unknown', ['no dominant signal (insufficient measured metrics)'])`.
    - `test_cli_high_dram_unmeasured_sm_is_unknown` → FAIL: CLI prints `bottleneck_class: memory_bound`; assert wanted `unknown`.
    - `test_high_sm_unmeasured_dram_is_compute_bound` → already PASSES on current code (the sm-alone path fires `s>=70`); it is the regression guard, not a red test.
  Final line: `FAILED (failures=3)`.

- [ ] **Step 4: Commit the test file (RED state).**
  Run:
  ```
  git add _substrate/tests/test_diagnose.py
  git commit -m "$(printf 'Add diagnose golden + null-rule characterization tests\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
  ```
  Expected: one new file committed; commit subject `Add diagnose golden + null-rule characterization tests`.

---

## Task A2 — Implement the `diagnose.py` vendor-profile + measured-operand null rule

Replace the literal thresholds with a default-`nvidia` `PROFILES` lookup driven by `metrics["_vendor"]`, and replace the `or 0.0` coercion with the decided §4.6 measured-operand rule. Evidence strings for all measured-both / already-working cases stay byte-identical; null cases no longer crash on `:.0f` and emit a distinct non-crashing evidence.

### Files
- **Modify**: `_substrate/diagnose.py`
  - Insert `PROFILES` dict immediately after the `CLASSES = [...]` line (current line 15).
  - Replace the body of `classify(m)` (current lines 18–35).
- **Test**: `_substrate/tests/test_diagnose.py` (from Task A1; now must go fully green)

### Steps

- [ ] **Step 1: Add the `PROFILES` dict after `CLASSES`.**
  In `_substrate/diagnose.py`, the current line 15 is:
  ```python
  CLASSES = ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound", "unknown"]
  ```
  Replace that single line with these lines (CLASSES unchanged; `PROFILES` added below it):
  ```python
  CLASSES = ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound", "unknown"]

  # Per-vendor decision thresholds (spec §5.3.1). Default/absent _vendor -> "nvidia"
  # == today's literals, so every existing NVIDIA case stays byte-identical.
  PROFILES = {
      "nvidia": dict(occ_lat=0.40, dram_mem=70, sm_mem=50, sm_comp=70, both_low=40),
      "apple": dict(occ_lat=0.30, dram_mem=65, sm_mem=55, sm_comp=65, both_low=35),
  }
  ```

- [ ] **Step 2: Replace the body of `classify(m)`.**
  In `_substrate/diagnose.py`, the current `classify` (lines 18–35) is:
  ```python
  def classify(m):
      dram = m.get("dram_pct")
      sm = m.get("sm_pct")
      occ = m.get("occupancy")
      if dram is None and sm is None and occ is None:
          return "unknown", ["no profiler metrics available"]
      d = dram or 0.0
      s = sm or 0.0
      # occupancy-limited dominates when clearly low
      if occ is not None and occ < 0.4:
          return "latency_occupancy", [f"occupancy {occ:.2f} < 0.40 (launch/occupancy limited)"]
      if d >= 70 and s < 50:
          return "memory_bound", [f"dram {d:.0f}% high, sm {s:.0f}% low"]
      if s >= 70:
          return "compute_bound", [f"sm {s:.0f}% high"]
      if d < 40 and s < 40:
          return "overhead_bound", [f"both utilizations low (dram {d:.0f}%, sm {s:.0f}%)"]
      return "unknown", [f"no dominant signal (dram {d:.0f}%, sm {s:.0f}%, occ {occ})"]
  ```
  Replace it in full with:
  ```python
  def classify(m):
      dram = m.get("dram_pct")
      sm = m.get("sm_pct")
      occ = m.get("occupancy")
      prof = PROFILES.get(m.get("_vendor", "nvidia"), PROFILES["nvidia"])
      if dram is None and sm is None and occ is None:
          return "unknown", ["no profiler metrics available"]
      # latency_occupancy: single positive signal — occupancy measured and clearly low.
      if occ is not None and occ < prof["occ_lat"]:
          return "latency_occupancy", [f"occupancy {occ:.2f} < {prof['occ_lat']:.2f} (launch/occupancy limited)"]
      # compute_bound: single positive signal — sm measured and high.
      if sm is not None and sm >= prof["sm_comp"]:
          return "compute_bound", [f"sm {sm:.0f}% high"]
      # memory_bound / overhead_bound are two-sided: both dram AND sm must be measured.
      if dram is not None and sm is not None:
          if dram >= prof["dram_mem"] and sm < prof["sm_mem"]:
              return "memory_bound", [f"dram {dram:.0f}% high, sm {sm:.0f}% low"]
          if dram < prof["both_low"] and sm < prof["both_low"]:
              return "overhead_bound", [f"both utilizations low (dram {dram:.0f}%, sm {sm:.0f}%)"]
          return "unknown", [f"no dominant signal (dram {dram:.0f}%, sm {sm:.0f}%, occ {occ})"]
      # A required two-sided discriminator was unmeasured -> cannot conclude.
      return "unknown", ["no dominant signal (insufficient measured metrics)"]
  ```
  Why this preserves byte-identity (and the §4.6 rule):
    - `prof` defaults to the nvidia dict, whose values are today's literals, so on measured-both NVIDIA inputs every branch and every formatted string is unchanged. `prof['occ_lat']:.2f` formats `0.40` → `"0.40"`, preserving the occupancy evidence literal.
    - `compute_bound` is reordered ahead of the two-sided block; this is safe because when `sm` is measured, `memory_bound` requires `sm < sm_mem` (50) and `compute_bound` requires `sm >= sm_comp` (70) — disjoint — and the golden tests prove it.
    - The all-null `{}` case keeps the CURRENT string `"no profiler metrics available"` (NOT the spec sketch's `"no classifier metrics"`) so that golden case stays byte-identical.
    - For null cases (`{dram:80, sm:null}`, `{dram:null, sm:30, occ:0.6}`) the two-sided block is skipped (one operand unmeasured) and the new non-`:.0f` evidence `"no dominant signal (insufficient measured metrics)"` is emitted, so there is no `:.0f`-on-`None` crash.
    - `main()` is unchanged: it still calls `classify(m)` and prints `{"bottleneck_class": cls, "evidence": ev}` with `indent=2, ensure_ascii=False`.

- [ ] **Step 3: Run the full diagnose test file + see all PASS.**
  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_diagnose.py' -v`
  Expected: every test in `TestDiagnoseGolden`, `TestDiagnoseNullRule`, `TestDiagnoseCli` passes. Final line `OK` (e.g. `Ran 11 tests in 0.0Ns` then `OK`). Golden byte-identity preserved; the two previously-failing null cases (and the CLI null case) now return `unknown`.

- [ ] **Step 4: Run the whole suite to confirm no collateral.**
  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v`
  Expected: `OK` (no failures, no errors). At this point `test_smoke.py` and `test_diagnose.py` exist; this guards against discovery/import breakage.

- [ ] **Step 5: Commit the implementation (GREEN state).**
  Run:
  ```
  git add _substrate/diagnose.py
  git commit -m "$(printf 'Add diagnose vendor profile and measured-operand null rule\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
  ```
  Expected: `_substrate/diagnose.py` committed; subject `Add diagnose vendor profile and measured-operand null rule`.

---

## Task A3 — `anti_cheat.py` characterization tests FIRST

Lock the current `static_flags` / `evaluate` behaviour (golden, must PASS), then add a NEW failing test: a Metal-flavoured source (`MPSMatrixMultiplication` + a C++ `return;` stub) is NOT flagged today and there is no `--vendor-patterns-file` argument; assert it SHOULD be flagged/blocked when vendor patterns are supplied (must FAIL on current code).

### Files
- **Create**: `_substrate/tests/test_anti_cheat.py`
- **Test**: `_substrate/tests/test_anti_cheat.py` (this file; run via discovery)
- **Modify**: none in this task (`_substrate/anti_cheat.py` is read-only here)

### Steps

- [ ] **Step 1: Create the anti_cheat test file.** Write `_substrate/tests/test_anti_cheat.py` with the complete content below. `TestAntiCheatGolden` locks current `static_flags`/`evaluate` (PASS now). `TestVendorPatternsUnit` calls a not-yet-existing `static_flags(..., extra_fallback=, extra_skip=)` keyword form (raises `TypeError` today → red). `TestVendorPatternsCli` shells out with a not-yet-existing `--vendor-patterns-file` arg (argparse exits non-zero today → red). The Metal source asserts BOTH a `[fallback]` (`MPSMatrixMultiplication`) and a `[skip]` (`return;` C++ stub) catch — exercising both lists per spec §9.3.

```python
import os, sys, json, subprocess, tempfile, unittest
SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

import anti_cheat

# Metal-flavoured source: an MPS library fallback AND a C++ empty-return stub.
METAL_SRC = (
    "#import <MetalPerformanceShaders.h>\n"
    "MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] init];\n"
    "void stub() { return; }\n"
)

# A Metal vendor-patterns file body: [fallback] + [skip], one regex per line,
# optional "| label". Comments and blank lines ignored.
METAL_PATTERNS = (
    "# Metal vendor cheat patterns\n"
    "[fallback]\n"
    "MPSMatrixMultiplication | MPS matmul fallback\n"
    "MPSNDArray\n"
    "MPSNNGraph\n"
    "\n"
    "[skip]\n"
    "//\\s*TODO | C++ TODO placeholder\n"
    "return\\s*; | C++ empty return stub\n"
    "__builtin_unreachable\n"
)

GOOD_METRICS = {
    "compiled": True, "correct": True,
    "candidate_latency_ms": 0.5, "eager_latency_ms": 1.0,
    "compile_latency_ms": 0.8, "claimed_speedup": 2.0,
}


class TestAntiCheatGolden(unittest.TestCase):
    """Characterization: lock the CURRENT static_flags / evaluate behaviour. These
    MUST stay byte-identical after the --vendor-patterns-file edit (CUDA invariant)."""

    def test_flags_cublas(self):
        self.assertEqual(
            anti_cheat.static_flags("auto y = cublasSgemm(handle);"),
            [{"type": "library_fallback", "detail": "cuBLAS delegation"}],
        )

    def test_flags_torch_matmul(self):
        self.assertEqual(
            anti_cheat.static_flags("out = torch.matmul(a, b)"),
            [{"type": "library_fallback", "detail": "torch.matmul fallback"}],
        )

    def test_flags_empty_pass(self):
        self.assertEqual(
            anti_cheat.static_flags("def f():\n    pass\n"),
            [{"type": "skipped_compute", "detail": "empty pass body"}],
        )

    def test_flags_not_implemented(self):
        self.assertEqual(
            anti_cheat.static_flags("raise NotImplementedError"),
            [{"type": "skipped_compute", "detail": "NotImplementedError"}],
        )

    def test_clean_kernel_has_no_flags(self):
        self.assertEqual(
            anti_cheat.static_flags("__global__ void k(float* x){ x[threadIdx.x] *= 2.0f; }"),
            [],
        )

    def test_evaluate_clean_fast_kernel_valid(self):
        res = anti_cheat.evaluate("__global__ void k(){}", GOOD_METRICS)
        self.assertTrue(res["valid"])
        self.assertEqual(res["reward"], 3)
        self.assertEqual(res["recorded_speedup"], 2.0)
        self.assertEqual(res["blocking_flags"], [])

    def test_evaluate_cublas_fallback_invalid(self):
        res = anti_cheat.evaluate("cublasSgemm(h);", GOOD_METRICS)
        self.assertFalse(res["valid"])
        self.assertEqual(res["recorded_speedup"], 0.0)
        self.assertEqual(res["blocking_flags"], ["library_fallback"])

    def test_metal_source_not_flagged_by_default(self):
        # Today the CUDA/Python defaults never match MPS or a C++ return; stub.
        self.assertEqual(anti_cheat.static_flags(METAL_SRC), [])


class TestVendorPatternsUnit(unittest.TestCase):
    """NEW (red on current code): static_flags must accept appended vendor patterns
    so the Metal source is flagged on BOTH the fallback and skip lists."""

    def test_metal_flagged_with_vendor_patterns(self):
        flags = anti_cheat.static_flags(
            METAL_SRC,
            extra_fallback=[("MPSMatrixMultiplication", "MPS matmul fallback")],
            extra_skip=[(r"return\s*;", "C++ empty return stub")],
        )
        types = sorted(f["type"] for f in flags)
        self.assertIn("library_fallback", types)
        self.assertIn("skipped_compute", types)


class TestVendorPatternsCli(unittest.TestCase):
    """NEW (red on current code): --vendor-patterns-file does not exist yet, so
    argparse rejects it (exit 2). After the edit the Metal source is invalid."""

    def test_cli_metal_invalid_with_patterns_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".patterns", delete=False) as pf:
            pf.write(METAL_PATTERNS)
            patterns_path = pf.name
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(SUB, "anti_cheat.py"),
                 "--source-text", METAL_SRC,
                 "--vendor-patterns-file", patterns_path,
                 "--metrics", "-"],
                input=json.dumps(GOOD_METRICS), capture_output=True, text=True,
            )
        finally:
            os.unlink(patterns_path)
        # invalid -> exit 1; stdout JSON shows both blocking flag types.
        self.assertEqual(r.returncode, 1, r.stderr)
        out = json.loads(r.stdout)
        self.assertFalse(out["valid"])
        self.assertEqual(
            sorted(set(out["blocking_flags"])),
            ["library_fallback", "skipped_compute"],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the golden baseline + see it PASS.**
  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_anti_cheat.py' -v -k Golden`
  Expected: 8 tests in `TestAntiCheatGolden`, all OK. Final line `OK`. Confirms the unchanged `static_flags`/`evaluate` matches every captured value, and that the Metal source is unflagged today (`test_metal_source_not_flagged_by_default`).

- [ ] **Step 3: Run the full file + see the new vendor-pattern tests FAIL on current code (state the exact failure).**
  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_anti_cheat.py' -v`
  Expected failures on UNCHANGED code:
    - `TestVendorPatternsUnit.test_metal_flagged_with_vendor_patterns` → ERROR: `TypeError: static_flags() got an unexpected keyword argument 'extra_fallback'` (current `static_flags(src)` takes one positional arg only).
    - `TestVendorPatternsCli.test_cli_metal_invalid_with_patterns_file` → FAIL: the subprocess exits with code 2 (argparse: `unrecognized arguments: --vendor-patterns-file ...`), so `assertEqual(r.returncode, 1)` fails (and stdout is empty, not JSON).
  Final line: `FAILED (failures=1, errors=1)`.

- [ ] **Step 4: Commit the test file (RED state).**
  Run:
  ```
  git add _substrate/tests/test_anti_cheat.py
  git commit -m "$(printf 'Add anti_cheat golden + vendor-pattern characterization tests\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
  ```
  Expected: one new file committed; subject `Add anti_cheat golden + vendor-pattern characterization tests`.

---

## Task A4 — Implement `anti_cheat.py` `--vendor-patterns-file`

Add the optional argument plus a `load_vendor_patterns(path)` helper, thread the parsed patterns through `static_flags` and `evaluate` as optional-empty params (so default behaviour stays byte-identical), and wire the file into `main()`. Covers BOTH `[fallback]` and `[skip]` (spec §5.3.3 / §9.3).

### Files
- **Modify**: `_substrate/anti_cheat.py`
  - Add `load_vendor_patterns(path)` helper (insert after the `SKIP_PATTERNS = [...]` block + `HARDCODE_HINT` / `BLOCKING` lines, i.e. immediately before `def static_flags` at current line 39).
  - Replace `static_flags` (current lines 39–49) to accept `extra_fallback`/`extra_skip`.
  - Replace `evaluate` (current lines 71–84) to accept + thread `extra_fallback`/`extra_skip`.
  - Replace `main()` (current lines 87–96) to add the `--vendor-patterns-file` argparse arg and wire it.
- **Test**: `_substrate/tests/test_anti_cheat.py` (from Task A3; now must go fully green)

### Steps

- [ ] **Step 1: Add the `load_vendor_patterns` helper.**
  In `_substrate/anti_cheat.py`, immediately before the current `def static_flags(src):` (line 39), insert this function (it reads a `[fallback]`/`[skip]`-sectioned file, one regex per line, optional `| label`; blank lines and `#` comments ignored; a line with no `| label` uses the regex text itself as the label):
  ```python
  def load_vendor_patterns(path):
      """Parse a vendor cheat-pattern file into (fallback, skip) lists of (regex, label).

      Format (spec §5.3.3): two sections [fallback] and [skip]; one regex per line;
      optional "regex | label". Blank lines and lines starting with '#' are ignored.
      """
      fallback, skip = [], []
      section = None
      with open(path) as f:
          for raw in f:
              line = raw.strip()
              if not line or line.startswith("#"):
                  continue
              if line.startswith("[") and line.endswith("]"):
                  section = line[1:-1].strip().lower()
                  continue
              if "|" in line:
                  pat, label = line.split("|", 1)
                  pat, label = pat.strip(), label.strip()
              else:
                  pat, label = line, line
              if section == "fallback":
                  fallback.append((pat, label))
              elif section == "skip":
                  skip.append((pat, label))
      return fallback, skip
  ```

- [ ] **Step 2: Replace `static_flags` to thread appended patterns.**
  The current `static_flags` (lines 39–49) is:
  ```python
  def static_flags(src):
      flags = []
      for pat, msg in FALLBACK_PATTERNS:
          if re.search(pat, src):
              flags.append({"type": "library_fallback", "detail": msg})
      for pat, msg in SKIP_PATTERNS:
          if re.search(pat, src, re.M):
              flags.append({"type": "skipped_compute", "detail": msg})
      if len(HARDCODE_HINT.findall(src)) >= 2:
          flags.append({"type": "hardcoded_shape", "detail": "multiple hardcoded dimensions"})
      return flags
  ```
  Replace it in full with (default empty extras → byte-identical iteration order: CUDA defaults first, vendor patterns appended after):
  ```python
  def static_flags(src, extra_fallback=(), extra_skip=()):
      flags = []
      for pat, msg in list(FALLBACK_PATTERNS) + list(extra_fallback):
          if re.search(pat, src):
              flags.append({"type": "library_fallback", "detail": msg})
      for pat, msg in list(SKIP_PATTERNS) + list(extra_skip):
          if re.search(pat, src, re.M):
              flags.append({"type": "skipped_compute", "detail": msg})
      if len(HARDCODE_HINT.findall(src)) >= 2:
          flags.append({"type": "hardcoded_shape", "detail": "multiple hardcoded dimensions"})
      return flags
  ```

- [ ] **Step 3: Replace `evaluate` to accept + thread the extras.**
  The current `evaluate` (lines 71–84) is:
  ```python
  def evaluate(src, m):
      flags = static_flags(src or "")
      reward, reason = robust_reward(m)
      blocking = [f for f in flags if f["type"] in BLOCKING]
      valid = (reward >= 0) and (not blocking)
      recorded = float(m.get("claimed_speedup") or 0.0) if (valid and reward >= 2) else 0.0
      return {
          "valid": valid,
          "reward": reward,
          "recorded_speedup": recorded,
          "reward_reason": reason,
          "flags": flags,
          "blocking_flags": [f["type"] for f in blocking],
      }
  ```
  Replace it in full with:
  ```python
  def evaluate(src, m, extra_fallback=(), extra_skip=()):
      flags = static_flags(src or "", extra_fallback, extra_skip)
      reward, reason = robust_reward(m)
      blocking = [f for f in flags if f["type"] in BLOCKING]
      valid = (reward >= 0) and (not blocking)
      recorded = float(m.get("claimed_speedup") or 0.0) if (valid and reward >= 2) else 0.0
      return {
          "valid": valid,
          "reward": reward,
          "recorded_speedup": recorded,
          "reward_reason": reason,
          "flags": flags,
          "blocking_flags": [f["type"] for f in blocking],
      }
  ```

- [ ] **Step 4: Replace `main()` to add `--vendor-patterns-file` and wire it.**
  The current `main()` (lines 87–96) is:
  ```python
  def main():
      ap = argparse.ArgumentParser(description=__doc__)
      ap.add_argument("--source"); ap.add_argument("--source-text")
      ap.add_argument("--metrics", required=True)
      a = ap.parse_args()
      src = a.source_text if a.source_text is not None else (open(a.source).read() if a.source else "")
      m = json.loads(sys.stdin.read() if a.metrics == "-" else open(a.metrics).read())
      res = evaluate(src, m)
      print(json.dumps(res, indent=2, ensure_ascii=False))
      return 0 if res["valid"] else 1
  ```
  Replace it in full with:
  ```python
  def main():
      ap = argparse.ArgumentParser(description=__doc__)
      ap.add_argument("--source"); ap.add_argument("--source-text")
      ap.add_argument("--vendor-patterns-file")
      ap.add_argument("--metrics", required=True)
      a = ap.parse_args()
      src = a.source_text if a.source_text is not None else (open(a.source).read() if a.source else "")
      m = json.loads(sys.stdin.read() if a.metrics == "-" else open(a.metrics).read())
      extra_fallback, extra_skip = ([], [])
      if a.vendor_patterns_file:
          extra_fallback, extra_skip = load_vendor_patterns(a.vendor_patterns_file)
      res = evaluate(src, m, extra_fallback, extra_skip)
      print(json.dumps(res, indent=2, ensure_ascii=False))
      return 0 if res["valid"] else 1
  ```

- [ ] **Step 5: Run the full anti_cheat test file + see all PASS.**
  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_anti_cheat.py' -v`
  Expected: every test in `TestAntiCheatGolden`, `TestVendorPatternsUnit`, `TestVendorPatternsCli` passes. Final line `OK` (e.g. `Ran 10 tests in 0.0Ns` then `OK`). Golden byte-identity preserved (default extras empty); the Metal source is now flagged on BOTH lists and the CLI returns exit code 1 with `blocking_flags` containing `library_fallback` and `skipped_compute`.

- [ ] **Step 6: Run the WHOLE suite (all files) to confirm no collateral.**
  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v`
  Expected: `OK` — `test_smoke.py` + `test_diagnose.py` + `test_anti_cheat.py` all green; no failures, no errors. This is the P2 acceptance gate (the §9.3 substrate diff-guard analogue: NVIDIA byte-identity + new null/vendor behaviour both proven).

- [ ] **Step 7: Commit the implementation (GREEN state).**
  Run:
  ```
  git add _substrate/anti_cheat.py
  git commit -m "$(printf 'Add anti_cheat vendor-patterns-file for per-backend cheat detection\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
  ```
  Expected: `_substrate/anti_cheat.py` committed; subject `Add anti_cheat vendor-patterns-file for per-backend cheat detection`.

---

## Task B1 — Test fixtures + `validate_backend.py` L0 tests (write FIRST, must fail)

Create the fixture drivers and the failing CLI test for a validator that does not yet exist. The test shells out to `_substrate/backends/validate_backend.py <dir>` and asserts exit 0 + `{"ok": true}` for the good driver, and exit non-zero + a specific error substring for each bad driver. Running it now must fail because `validate_backend.py` does not exist yet.

### Files
- **Create** `_substrate/backends/__init__.py` *(empty; allows the validator dir to be importable in the future, harmless now)*
- **Create** `_substrate/tests/fixtures/good/manifest.json`
- **Create** `_substrate/tests/fixtures/good/idioms.json`
- **Create** `_substrate/tests/fixtures/bad_backend_id/manifest.json`
- **Create** `_substrate/tests/fixtures/bad_backend_id/idioms.json`
- **Create** `_substrate/tests/fixtures/bad_idiom_method/manifest.json`
- **Create** `_substrate/tests/fixtures/bad_idiom_method/idioms.json`
- **Create** `_substrate/tests/fixtures/bad_metric_key/manifest.json`
- **Create** `_substrate/tests/fixtures/bad_metric_key/idioms.json`
- **Create** `_substrate/tests/fixtures/bad_bottleneck_class/manifest.json`
- **Create** `_substrate/tests/fixtures/bad_bottleneck_class/idioms.json`
- **Test** `_substrate/tests/test_validate_backend.py`

> Note on packaging: per the harness-layout decision at the top of this plan, `_substrate/tests/` stays a **non-package** dir — do **not** create `_substrate/tests/__init__.py`. Only `_substrate/backends/__init__.py` is created here.
>
> Fixture naming rule (load-bearing for the L0 `backend_id == dir` check): the **directory name** is the canonical `backend_id`. The good driver lives in `good/` and declares `backend_id:"good"`; each bad driver's directory name is the failure label. Every bad fixture is a single-defect mutation of the good one so each test pins exactly one error.

- [ ] **Step 1: Create the package marker `_substrate/backends/__init__.py` (empty).**

  `_substrate/backends/__init__.py`:
  ```python
  ```

- [ ] **Step 2: Create the VALID driver `_substrate/tests/fixtures/good/manifest.json`.**

  A minimal cuda-like backend. `backend_id` equals the dir name `good`. `capabilities.metrics` keys are all in the canonical set `{latency_ms, dram_pct, sm_pct, occupancy}`. `bottleneck_classes` are all in `{memory_bound, compute_bound, latency_occupancy, overhead_bound}` (the 4 meaningful; per spec §4.9 `unknown` is also accepted but not required — this fixture omits it).

  ```json
  {
    "schema_version": 1,
    "backend_id": "good",
    "display_name": "Good Test Backend",
    "source_ext": ".cu",
    "aux_ext": [".cuh", ".h"],
    "artifact_ext": ".so",
    "hw_vendor": "nvidia",
    "threshold_profile": "nvidia",
    "compiler": { "name": "nvcc", "invoke": "build.sh" },
    "runner": { "invoke": "run.sh" },
    "profiler": {
      "name": "ncu",
      "invoke": "profile.sh",
      "format": "ncu-csv",
      "to_evidence": "to_evidence.py"
    },
    "capabilities": {
      "metrics": { "dram_pct": true, "sm_pct": true, "occupancy": true, "latency_ms": true },
      "bottleneck_classes": ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound"],
      "problem_types": ["kernel-optimization", "kernel-generation"],
      "precisions": ["fp32", "fp16"]
    },
    "requires_tools": ["nvcc", "python3"],
    "optional_tools": ["ncu"],
    "idioms": "idioms.json",
    "status": "stable"
  }
  ```

- [ ] **Step 3: Create the VALID driver `_substrate/tests/fixtures/good/idioms.json`.**

  Every key under `methods` and every entry in `unsupported_methods` is a real `method_gate.TABLE` method name (verified live: `tensor_core_mma`, `memory_coalescing`, `kernel_fusion` are valid; `library_fallback_hybrid` is valid and is the one routed to `unsupported_methods`).

  ```json
  {
    "schema_version": 1,
    "backend_id": "good",
    "lang_fence": "cuda",
    "impl_requirements": "PYBIND11_MODULE entrypoint with a forward() binding.",
    "unsupported_methods": ["library_fallback_hybrid"],
    "read_metric_guide": "High dram_pct with low sm_pct means memory-bound.",
    "methods": {
      "tensor_core_mma": {
        "idiom": "wmma / mma.sync",
        "prompt_guidance": "Use tensor-core MMA intrinsics for the inner GEMM."
      },
      "memory_coalescing": {
        "idiom": "coalesced 128-bit global loads",
        "prompt_guidance": "Reorder thread-to-element mapping so consecutive threads read consecutive addresses."
      },
      "kernel_fusion": {
        "idiom": "fuse epilogue into the main kernel",
        "prompt_guidance": "Avoid a second launch by fusing the elementwise epilogue."
      }
    }
  }
  ```

- [ ] **Step 4: Create INVALID driver `bad_backend_id/` — `backend_id` does not equal the dir name.**

  `_substrate/tests/fixtures/bad_backend_id/manifest.json` — identical to `good` except `backend_id` is `"not_the_dir_name"` (dir is `bad_backend_id`):

  ```json
  {
    "schema_version": 1,
    "backend_id": "not_the_dir_name",
    "display_name": "Bad Backend Id",
    "source_ext": ".cu",
    "aux_ext": [".cuh", ".h"],
    "artifact_ext": ".so",
    "hw_vendor": "nvidia",
    "threshold_profile": "nvidia",
    "compiler": { "name": "nvcc", "invoke": "build.sh" },
    "runner": { "invoke": "run.sh" },
    "profiler": {
      "name": "ncu",
      "invoke": "profile.sh",
      "format": "ncu-csv",
      "to_evidence": "to_evidence.py"
    },
    "capabilities": {
      "metrics": { "dram_pct": true, "sm_pct": true, "occupancy": true, "latency_ms": true },
      "bottleneck_classes": ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound"],
      "problem_types": ["kernel-optimization"],
      "precisions": ["fp32"]
    },
    "requires_tools": ["nvcc", "python3"],
    "optional_tools": ["ncu"],
    "idioms": "idioms.json",
    "status": "stable"
  }
  ```

  `_substrate/tests/fixtures/bad_backend_id/idioms.json` — valid idioms (so only the manifest defect trips), but its `backend_id` must also be `"not_the_dir_name"` so it does not introduce a *second* error:

  ```json
  {
    "schema_version": 1,
    "backend_id": "not_the_dir_name",
    "lang_fence": "cuda",
    "impl_requirements": "PYBIND11_MODULE entrypoint with a forward() binding.",
    "unsupported_methods": [],
    "methods": {
      "tensor_core_mma": {
        "idiom": "wmma / mma.sync",
        "prompt_guidance": "Use tensor-core MMA intrinsics for the inner GEMM."
      }
    }
  }
  ```

- [ ] **Step 5: Create INVALID driver `bad_idiom_method/` — `idioms.json` references a non-existent `method_gate` method.**

  `_substrate/tests/fixtures/bad_idiom_method/manifest.json` (valid; `backend_id` matches dir):

  ```json
  {
    "schema_version": 1,
    "backend_id": "bad_idiom_method",
    "display_name": "Bad Idiom Method",
    "source_ext": ".cu",
    "aux_ext": [".cuh", ".h"],
    "artifact_ext": ".so",
    "hw_vendor": "nvidia",
    "threshold_profile": "nvidia",
    "compiler": { "name": "nvcc", "invoke": "build.sh" },
    "runner": { "invoke": "run.sh" },
    "profiler": {
      "name": "ncu",
      "invoke": "profile.sh",
      "format": "ncu-csv",
      "to_evidence": "to_evidence.py"
    },
    "capabilities": {
      "metrics": { "dram_pct": true, "sm_pct": true, "occupancy": true, "latency_ms": true },
      "bottleneck_classes": ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound"],
      "problem_types": ["kernel-optimization"],
      "precisions": ["fp32"]
    },
    "requires_tools": ["nvcc", "python3"],
    "optional_tools": ["ncu"],
    "idioms": "idioms.json",
    "status": "stable"
  }
  ```

  `_substrate/tests/fixtures/bad_idiom_method/idioms.json` — `methods` contains a bogus key `not_a_real_method` (no such name in `method_gate.TABLE`):

  ```json
  {
    "schema_version": 1,
    "backend_id": "bad_idiom_method",
    "lang_fence": "cuda",
    "impl_requirements": "PYBIND11_MODULE entrypoint with a forward() binding.",
    "unsupported_methods": [],
    "methods": {
      "not_a_real_method": {
        "idiom": "made up",
        "prompt_guidance": "This method name does not exist in method_gate.TABLE."
      }
    }
  }
  ```

- [ ] **Step 6: Create INVALID driver `bad_metric_key/` — `capabilities.metrics` has a non-canonical key.**

  `_substrate/tests/fixtures/bad_metric_key/manifest.json` — `metrics` includes the bogus key `gpu_temp` (canonical set is only `{latency_ms, dram_pct, sm_pct, occupancy}`):

  ```json
  {
    "schema_version": 1,
    "backend_id": "bad_metric_key",
    "display_name": "Bad Metric Key",
    "source_ext": ".cu",
    "aux_ext": [".cuh", ".h"],
    "artifact_ext": ".so",
    "hw_vendor": "nvidia",
    "threshold_profile": "nvidia",
    "compiler": { "name": "nvcc", "invoke": "build.sh" },
    "runner": { "invoke": "run.sh" },
    "profiler": {
      "name": "ncu",
      "invoke": "profile.sh",
      "format": "ncu-csv",
      "to_evidence": "to_evidence.py"
    },
    "capabilities": {
      "metrics": { "dram_pct": true, "sm_pct": true, "occupancy": true, "latency_ms": true, "gpu_temp": true },
      "bottleneck_classes": ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound"],
      "problem_types": ["kernel-optimization"],
      "precisions": ["fp32"]
    },
    "requires_tools": ["nvcc", "python3"],
    "optional_tools": ["ncu"],
    "idioms": "idioms.json",
    "status": "stable"
  }
  ```

  `_substrate/tests/fixtures/bad_metric_key/idioms.json` (valid):

  ```json
  {
    "schema_version": 1,
    "backend_id": "bad_metric_key",
    "lang_fence": "cuda",
    "impl_requirements": "PYBIND11_MODULE entrypoint with a forward() binding.",
    "unsupported_methods": [],
    "methods": {
      "tensor_core_mma": {
        "idiom": "wmma / mma.sync",
        "prompt_guidance": "Use tensor-core MMA intrinsics for the inner GEMM."
      }
    }
  }
  ```

- [ ] **Step 7: Create INVALID driver `bad_bottleneck_class/` — `bottleneck_classes` has a bogus class.**

  `_substrate/tests/fixtures/bad_bottleneck_class/manifest.json` — `bottleneck_classes` includes `register_bound` (not one of the 4 meaningful classes; `unknown` would also be illegal to list but here the defect is a wholly bogus class):

  ```json
  {
    "schema_version": 1,
    "backend_id": "bad_bottleneck_class",
    "display_name": "Bad Bottleneck Class",
    "source_ext": ".cu",
    "aux_ext": [".cuh", ".h"],
    "artifact_ext": ".so",
    "hw_vendor": "nvidia",
    "threshold_profile": "nvidia",
    "compiler": { "name": "nvcc", "invoke": "build.sh" },
    "runner": { "invoke": "run.sh" },
    "profiler": {
      "name": "ncu",
      "invoke": "profile.sh",
      "format": "ncu-csv",
      "to_evidence": "to_evidence.py"
    },
    "capabilities": {
      "metrics": { "dram_pct": true, "sm_pct": true, "occupancy": true, "latency_ms": true },
      "bottleneck_classes": ["memory_bound", "compute_bound", "register_bound"],
      "problem_types": ["kernel-optimization"],
      "precisions": ["fp32"]
    },
    "requires_tools": ["nvcc", "python3"],
    "optional_tools": ["ncu"],
    "idioms": "idioms.json",
    "status": "stable"
  }
  ```

  `_substrate/tests/fixtures/bad_bottleneck_class/idioms.json` (valid):

  ```json
  {
    "schema_version": 1,
    "backend_id": "bad_bottleneck_class",
    "lang_fence": "cuda",
    "impl_requirements": "PYBIND11_MODULE entrypoint with a forward() binding.",
    "unsupported_methods": [],
    "methods": {
      "tensor_core_mma": {
        "idiom": "wmma / mma.sync",
        "prompt_guidance": "Use tensor-core MMA intrinsics for the inner GEMM."
      }
    }
  }
  ```

- [ ] **Step 8: Create the failing CLI test `_substrate/tests/test_validate_backend.py`.**

  Shells out to the validator (which does not exist yet → the test fails). Asserts exit 0 + `{"ok": true}` and empty `errors` for `good`, and exit non-zero + a specific error substring for each bad fixture. The substrings are the exact phrases the Task B2 validator will emit.

  ```python
  import os, sys, json, subprocess, unittest
  SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
  sys.path.insert(0, SUB)

  VALIDATOR = os.path.join(SUB, 'backends', 'validate_backend.py')
  FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


  def run_validator(driver_dir):
      """Shell out to validate_backend.py <dir>; return (returncode, parsed_stdout_json)."""
      proc = subprocess.run(
          [sys.executable, VALIDATOR, os.path.join(FIXTURES, driver_dir)],
          capture_output=True, text=True)
      try:
          payload = json.loads(proc.stdout)
      except json.JSONDecodeError:
          payload = {"_raw_stdout": proc.stdout, "_stderr": proc.stderr}
      return proc.returncode, payload


  class TestValidateBackendGood(unittest.TestCase):
      def test_good_driver_exits_zero_and_ok_true(self):
          code, payload = run_validator('good')
          self.assertEqual(code, 0, msg=f"expected exit 0, got {code}; payload={payload}")
          self.assertEqual(payload.get('ok'), True, msg=f"payload={payload}")
          self.assertEqual(payload.get('errors'), [], msg=f"payload={payload}")


  class TestValidateBackendBad(unittest.TestCase):
      def assert_bad(self, driver_dir, substring):
          code, payload = run_validator(driver_dir)
          self.assertNotEqual(code, 0, msg=f"expected non-zero exit; payload={payload}")
          self.assertEqual(payload.get('ok'), False, msg=f"payload={payload}")
          joined = " | ".join(payload.get('errors', []))
          self.assertIn(substring, joined,
                        msg=f"expected '{substring}' in errors; got: {joined}")

      def test_backend_id_mismatch(self):
          self.assert_bad('bad_backend_id', 'backend_id')

      def test_idiom_references_unknown_method(self):
          self.assert_bad('bad_idiom_method', 'not_a_real_method')

      def test_non_canonical_metric_key(self):
          self.assert_bad('bad_metric_key', 'gpu_temp')

      def test_bogus_bottleneck_class(self):
          self.assert_bad('bad_bottleneck_class', 'register_bound')


  if __name__ == '__main__':
      unittest.main()
  ```

- [ ] **Step 9: Run the test from the repo root and SEE IT FAIL (validator does not exist).**

  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_validate_backend.py' -v`

  Expected: every test fails (not errors-out at collection). Because `validate_backend.py` does not exist, `subprocess.run` produces empty stdout, `json.loads` raises, the fallback payload has no `ok`/`errors` keys, so `TestValidateBackendGood` fails its `assertEqual(code, 0)` (the Python interpreter exits non-zero: "can't open file ... validate_backend.py") and the bad tests fail their substring assertions. Exact tail:
  ```
  FAILED (failures=5)
  ```
  with `test_good_driver_exits_zero_and_ok_true` reporting `expected exit 0, got 2` and each bad test reporting `expected '...' in errors; got: ` (empty).

- [ ] **Step 10: Commit the fixtures + the failing test.**

  Run:
  ```
  git add _substrate/backends/__init__.py _substrate/tests/fixtures _substrate/tests/test_validate_backend.py
  git commit -m "$(cat <<'EOF'
  Add backend L0 validator fixtures and failing tests

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```
  Expected: one commit created on the current branch; `git status` clean for those paths.

---

## Task B2 — Implement `_substrate/backends/validate_backend.py` (make B1 green)

Stdlib-only, hand-rolled L0 structural validator. Reads `manifest.json` + `idioms.json` with the `json` module (no YAML, no JSON-Schema). It imports the live `method_gate` module to source the real `TABLE` method names, so the idiom-reference check cannot drift. Prints `{"ok": bool, "errors": [...]}` and exits `0` (ok) / `1` (errors) / `3` (bad args / unreadable).

### Files
- **Create** `_substrate/backends/validate_backend.py`
- **Test** `_substrate/tests/test_validate_backend.py` *(from Task B1; unchanged — now must pass)*

- [ ] **Step 1: Create `_substrate/backends/validate_backend.py` with the COMPLETE validator.**

  ```python
  #!/usr/bin/env python3
  """L0 structural validator for a backend driver directory.

  Spec: docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md §4.4/§4.7/§4.9 (L0).
  Stdlib only; hand-rolled checks (no YAML, no JSON-Schema, no Node). Machine-read driver
  files are JSON (manifest.json/idioms.json) — see BACKEND-DRIVER-SDK.md "Deviations".

  Usage:
    validate_backend.py <driver_dir>
  Prints: {"ok": bool, "errors": [str, ...]}
  Exit:   0 ok · 1 L0 errors · 3 bad args / unreadable input
  """
  import os, sys, json, argparse

  # Import the live method_gate so the idiom-reference check uses the REAL method names.
  _SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
  if _SUB not in sys.path:
      sys.path.insert(0, _SUB)
  import method_gate  # noqa: E402

  # The four canonical metric keys diagnose.py / to_evidence.py speak.
  CANONICAL_METRICS = {"latency_ms", "dram_pct", "sm_pct", "occupancy"}
  # The 4 meaningful bottleneck classes. Per spec §4.9 the allowed set a driver MAY list is
  # these 4 PLUS the implicit `unknown` (listing `unknown` is permitted, not required).
  MEANINGFUL_CLASSES = {"memory_bound", "compute_bound", "latency_occupancy", "overhead_bound"}
  ALLOWED_BCLASSES = MEANINGFUL_CLASSES | {"unknown"}
  # All real method_gate method names (sourced live so this can never drift from the table).
  KNOWN_METHODS = {m for methods in method_gate.TABLE.values() for m in methods}


  def _load_json(path, errors):
      """Read a JSON file; append a message to errors and return None on failure."""
      if not os.path.isfile(path):
          errors.append(f"missing file: {os.path.basename(path)}")
          return None
      try:
          with open(path, encoding="utf-8") as fh:
              return json.load(fh)
      except json.JSONDecodeError as exc:
          errors.append(f"unparseable JSON in {os.path.basename(path)}: {exc}")
          return None


  def validate(driver_dir):
      """Run all L0 checks on driver_dir. Return {'ok': bool, 'errors': [str, ...]}."""
      errors = []
      dir_name = os.path.basename(os.path.normpath(driver_dir))

      manifest = _load_json(os.path.join(driver_dir, "manifest.json"), errors)
      idioms = _load_json(os.path.join(driver_dir, "idioms.json"), errors)

      # --- manifest checks ---
      if isinstance(manifest, dict):
          # backend_id present and == dir name (the dispatch key)
          backend_id = manifest.get("backend_id")
          if not isinstance(backend_id, str) or not backend_id:
              errors.append("manifest.backend_id missing or not a string")
          elif backend_id != dir_name:
              errors.append(
                  f"manifest.backend_id '{backend_id}' != directory name '{dir_name}'")

          caps = manifest.get("capabilities")
          if not isinstance(caps, dict):
              errors.append("manifest.capabilities missing or not an object")
          else:
              # capabilities.metrics keys subset of canonical keys
              metrics = caps.get("metrics")
              if not isinstance(metrics, dict):
                  errors.append("manifest.capabilities.metrics missing or not an object")
              else:
                  for key in metrics:
                      if key not in CANONICAL_METRICS:
                          errors.append(
                              f"capabilities.metrics key '{key}' is not a canonical metric "
                              f"(allowed: {sorted(CANONICAL_METRICS)})")
              # bottleneck_classes subset of the 4 meaningful classes
              bclasses = caps.get("bottleneck_classes")
              if not isinstance(bclasses, list):
                  errors.append("manifest.capabilities.bottleneck_classes missing or not a list")
              else:
                  for bc in bclasses:
                      if bc not in ALLOWED_BCLASSES:
                          errors.append(
                              f"capabilities.bottleneck_classes entry '{bc}' is not an "
                              f"allowed class (allowed: {sorted(ALLOWED_BCLASSES)})")

      # --- idioms checks ---
      if isinstance(idioms, dict):
          # every methods key and every unsupported_methods entry is a real method_gate name
          methods = idioms.get("methods")
          if not isinstance(methods, dict):
              errors.append("idioms.methods missing or not an object")
          else:
              for name in methods:
                  if name not in KNOWN_METHODS:
                      errors.append(
                          f"idioms.methods references unknown method_gate method '{name}'")
          unsupported = idioms.get("unsupported_methods", [])
          if not isinstance(unsupported, list):
              errors.append("idioms.unsupported_methods is not a list")
          else:
              for name in unsupported:
                  if name not in KNOWN_METHODS:
                      errors.append(
                          f"idioms.unsupported_methods references unknown method_gate "
                          f"method '{name}'")

      return {"ok": len(errors) == 0, "errors": errors}


  def main():
      ap = argparse.ArgumentParser(description=__doc__)
      ap.add_argument("driver_dir", help="path to the backend driver directory")
      a = ap.parse_args()
      if not os.path.isdir(a.driver_dir):
          print(json.dumps({"ok": False, "errors": [f"not a directory: {a.driver_dir}"]},
                           indent=2, ensure_ascii=False))
          return 3
      result = validate(a.driver_dir)
      print(json.dumps(result, indent=2, ensure_ascii=False))
      return 0 if result["ok"] else 1


  if __name__ == "__main__":
      sys.exit(main())
  ```

- [ ] **Step 2: Run the B1 test and SEE IT PASS.**

  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_validate_backend.py' -v`

  Expected: all 5 tests pass:
  ```
  test_good_driver_exits_zero_and_ok_true (...) ... ok
  test_backend_id_mismatch (...) ... ok
  test_bogus_bottleneck_class (...) ... ok
  test_idiom_references_unknown_method (...) ... ok
  test_non_canonical_metric_key (...) ... ok

  Ran 5 tests in 0.0XXs

  OK
  ```

- [ ] **Step 3: Sanity-run the validator directly on the good and one bad fixture (CLI smoke).**

  Run: `python3 _substrate/backends/validate_backend.py _substrate/tests/fixtures/good; echo "exit=$?"`

  Expected:
  ```
  {
    "ok": true,
    "errors": []
  }
  exit=0
  ```

  Run: `python3 _substrate/backends/validate_backend.py _substrate/tests/fixtures/bad_metric_key; echo "exit=$?"`

  Expected (errors list contains the `gpu_temp` message; exit 1):
  ```
  {
    "ok": false,
    "errors": [
      "capabilities.metrics key 'gpu_temp' is not a canonical metric (allowed: ['dram_pct', 'latency_ms', 'occupancy', 'sm_pct'])"
    ]
  }
  exit=1
  ```

- [ ] **Step 4: Run the FULL test suite to confirm nothing else regressed.**

  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v`

  Expected: ends with `OK`. If Part A landed first, the diagnose + anti_cheat tests are also green; if Part B runs alone, `test_smoke.py` + `test_validate_backend.py` are green. No failures, no errors.

- [ ] **Step 5: Commit the validator.**

  Run:
  ```
  git add _substrate/backends/validate_backend.py
  git commit -m "$(cat <<'EOF'
  Add deterministic L0 backend driver validator

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```
  Expected: one commit created; `git status` clean for that path.

---

## Task B3 — Create `_substrate/backends/REGISTRY.md`

A human-facing index of backend drivers. Seeded with a single `cuda — planned` row. No automated test; a grep step verifies the required columns are present.

### Files
- **Create** `_substrate/backends/REGISTRY.md`

- [ ] **Step 1: Create `_substrate/backends/REGISTRY.md`.**

  ```markdown
  # Backend Driver Registry

  This is the human-facing index of backend drivers under `_substrate/backends/`. A backend
  driver is the `(source language) × (hardware/profiler vendor)` translation layer that adapts
  native backend tooling to the universal substrate vocabulary (see
  [`../BACKEND-DRIVER-SDK.md`](../BACKEND-DRIVER-SDK.md) for the full contract). Each row maps a
  canonical `backend id` (the `normalizeSuitabilityValue` form, which is also the driver
  directory name) to its directory, hardware vendor, lifecycle status, and owner.

  Add a row when you start a driver. Move it to `stable` only after it passes L0–L3 conformance
  (`validate_backend.py` for L0; see the SDK doc for the full ladder).

  | backend id | dir | hw_vendor | status | owner |
  |---|---|---|---|---|
  | cuda | `cuda/` | nvidia | planned | (unassigned) |

  **Status vocabulary:** `planned` (row reserved, no files yet) · `stub` · `experimental` ·
  `stable` (L0–L3 conformant). `status` here is the registry lifecycle and is distinct from the
  per-manifest `status` field, which only ranges over `stub | experimental | stable`.
  ```

- [ ] **Step 2: Grep the registry for the required columns and the seed row.**

  Run: `grep -E 'backend id.*dir.*hw_vendor.*status.*owner' _substrate/backends/REGISTRY.md && grep -E 'cuda.*nvidia.*planned' _substrate/backends/REGISTRY.md`

  Expected: both greps print their matching line (the header row and the seeded `cuda — planned` row); the command exits 0.

- [ ] **Step 3: Commit the registry.**

  Run:
  ```
  git add _substrate/backends/REGISTRY.md
  git commit -m "$(cat <<'EOF'
  Add backend driver registry seeded with cuda

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```
  Expected: one commit created; `git status` clean for that path.

---

## Task B4 — Create `_substrate/BACKEND-DRIVER-SDK.md`

The driver-contract companion to `SOLVER-SDK.md`. No automated test; a grep step verifies the file exists and contains the required section headers.

### Files
- **Create** `_substrate/BACKEND-DRIVER-SDK.md`

- [ ] **Step 1: Create `_substrate/BACKEND-DRIVER-SDK.md` with all required sections.**

  ```markdown
  # Backend Driver SDK — the (language × vendor) translation layer

  Companion to [`SOLVER-SDK.md`](./SOLVER-SDK.md) and [`ARCHITECTURE.md`](./ARCHITECTURE.md). A
  **backend driver** is a directory under `_substrate/backends/<backend_id>/` that adapts native
  backend tooling (compiler, runner, profiler) to the universal substrate vocabulary, so any
  clean optimization *method* can run on any *backend* by setting `args.backend` rather than
  re-typing shell strings and re-authoring prompts. The driver is data an agent reads via Bash;
  it is never `import`ed by a workflow `.js` body.

  Full design rationale: `docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md`.

  ## The six-file driver layout

  ```
  _substrate/backends/<backend_id>/
  ├── manifest.json      # static identity, toolchain, capabilities (machine-read; JSON)
  ├── build.sh           # compile/JIT source -> artifact (executable; shebang; NO python prefix)
  ├── run.sh             # artifact -> correctness + latencies in anti_cheat.py key shape (executable)
  ├── profile.sh         # artifact -> native profiler output, emits a pointer (executable)
  ├── to_evidence.py     # NEUTRAL INTERFACE: native profile -> canonical metrics dict (python prefix)
  └── idioms.json        # method translation table: abstract method_gate name -> backend idiom
  ```

  `manifest.json`/`idioms.json` are machine-read JSON (see "Deviations from the spec"). The three
  `.sh` files are executable and invoked WITHOUT a python interpreter; `to_evidence.py` IS invoked
  with the python prefix. `validate_backend.py` checks the JSON files at L0.

  ## `manifest.json` fields

  | field | req | meaning |
  |---|---|---|
  | `schema_version` | yes | manifest schema version (currently `1`) |
  | `backend_id` | yes | canonical id; MUST equal the directory name; the dispatch key |
  | `display_name` | yes | human label |
  | `source_ext` | yes | primary kernel source extension (e.g. `.cu`) |
  | `aux_ext` | no | auxiliary source extensions |
  | `artifact_ext` | yes | compiled artifact extension |
  | `hw_vendor` | yes | `nvidia \| amd \| intel \| apple \| cpu \| generic` |
  | `threshold_profile` | yes | `diagnose.py` profile key (default `nvidia`) |
  | `compiler` | yes | `{ name, invoke: "build.sh" }` |
  | `runner` | yes | `{ invoke: "run.sh" }` |
  | `profiler` | no | `{ name, invoke: "profile.sh", format, to_evidence }`; omit ⇒ no profiler ⇒ `unknown` |
  | `capabilities.metrics` | yes | which canonical keys are honestly populated (subset of the 4 canonical keys) |
  | `capabilities.bottleneck_classes` | yes | meaningful classes (subset of the 4; `unknown` implicit) |
  | `capabilities.problem_types` | yes | supported problem types |
  | `capabilities.precisions` | no | supported precisions |
  | `requires_tools` | yes | preflight tools that must resolve |
  | `optional_tools` | no | tools whose absence degrades gracefully |
  | `vendor_patterns_file` | no | per-vendor anti-cheat regex file (`[fallback]`+`[skip]`) |
  | `idioms` | yes | idioms filename (`idioms.json`) |
  | `status` | yes | `stable \| experimental \| stub` |

  ## `idioms.json` fields

  | field | req | meaning |
  |---|---|---|
  | `schema_version` | yes | idioms schema version (currently `1`) |
  | `backend_id` | yes | canonical id (matches manifest) |
  | `lang_fence` | yes | the ```` ``` ```` fence language for code blocks (`cuda`/`python`/`metal`/`cpp`) |
  | `impl_requirements` | yes | ABI/scaffolding the executor must emit (e.g. `PYBIND11_MODULE`) |
  | `unsupported_methods` | yes | `method_gate.TABLE` names this backend cannot honor (gated OUT) |
  | `read_metric_guide` | no | per-backend causal model for reading profiler data |
  | `methods.<name>.idiom` | yes | concrete construct for the abstract method `<name>` |
  | `methods.<name>.prompt_guidance` | yes | instruction injected into the executor prompt |
  | `methods.<name>.anti_idiom` | no | construct to avoid |
  | `methods.<name>.triggers_on_native` | no | `backend_native` keys justifying the method |
  | `methods.<name>.code_markers` | no | tokens the executor should emit (self-check) |

  Every key under `methods` and every entry in `unsupported_methods` MUST be a real
  `method_gate.TABLE` method name. `method_gate.py` is never edited.

  ## `to_evidence.py` — the neutral interface

  `to_evidence.py` is a pure function: native profiler output → the canonical `metrics` dict that
  `diagnose.py` consumes verbatim, plus a `_vendor` tag and a `coverage` list of honestly
  populated keys.

  ### Canonical-unit table (every backend's `to_evidence` MUST honor)

  | key | unit | example native source |
  |---|---|---|
  | `latency_ms` | milliseconds | duration (ns) ÷ 1e6 |
  | `dram_pct` | 0–100 | DRAM throughput pct-of-peak |
  | `sm_pct` | 0–100 | SM throughput pct-of-peak |
  | `occupancy` | **0–1** | warps-active pct-of-peak **÷ 100** |

  > The single most error-prone line: NCU reports occupancy as 0–100; `diagnose.py` compares
  > `occ < threshold` on a 0–1 scale. `to_evidence.py` MUST divide by 100.

  ### The decided null rule

  A backend that lacks a counter sets the key to **`null`** (JSON `null`) and omits it from
  `coverage` — never a fabricated `0.0`. A bottleneck branch may fire only when every metric it
  compares is measured:
  - `latency_occupancy` fires on `occupancy` alone;
  - `compute_bound` fires on `sm_pct` alone;
  - `memory_bound` and `overhead_bound` require **both** `dram_pct` and `sm_pct` measured;
  - any case missing a required measured discriminator → **`unknown`**, never a fabricated label.

  So `{dram:80, sm:null}` → `unknown`; `{dram:null, sm:80}` → `compute_bound`;
  `{dram:null, sm:30, occ:0.6}` → `unknown`. (This rule is enforced by the §5.3.2 `diagnose.py`
  edit, delivered in Part A / P2 — not in this part.)

  ## The three portability tiers

  - **clean** → `method_supported_backends: 'any'`; runs wherever a driver exists.
  - **vendor_locked** → e.g. `['cuda','triton']` plus a `requires_capability.metrics` floor.
    Locked to a *tool*, not a language.
  - **method_intrinsic** → a single-element hard whitelist (`['cpp']`/`['cutlass']`/`['rocm']`)
    plus pinned `requires_capability.problem_types`; the method *is* the backend;
    `matrix_eligible:false`.

  ## Conformance levels (L0–L3)

  - **L0 — Declared:** `manifest.json` validates; `backend_id == dir`; `idioms.json` references
    only real `method_gate.TABLE` names (including `unsupported_methods`);
    `capabilities.metrics ⊆` the 4 canonical keys; `bottleneck_classes ⊆` the 4 meaningful ∪
    `{unknown}`. **Checked deterministically by `backends/validate_backend.py`.**
  - **L1 — Buildable & Honest:** `build.sh`/`run.sh` executable; emit the contracted envelope +
    codes on a smoke fixture; `run.sh` output is accepted by `anti_cheat.py`; an incorrect
    fixture yields `valid:false`.
  - **L2 — Diagnostic:** `to_evidence.py` is pure (same native input ⇒ identical metrics; asserts
    `occupancy ∈ [0,1]`); its output into `diagnose.py` yields a declared class; the decided null
    rule holds.
  - **L3 — Compounding:** every gated method for every declared `bottleneck_class` resolves to an
    `idioms.json` entry or `unsupported_methods`; a full round-trip produces a valid Layer-A
    envelope; `requires_tools` preflight resolves.

  ## The scoped substrate edits (Part A / spec §5.3) — for cross-reference

  Part B (this contract scaffolding) changes **no** substrate script. The driver axis does force
  three scoped, default-`nvidia`, golden-tested substrate edits, delivered in **Part A / P2**:

  1. `diagnose.py` — a `_vendor` threshold-profile lookup (default `nvidia`).
  2. `diagnose.py` — null-vs-zero handling implementing the decided null rule above.
  3. `anti_cheat.py` — one optional `--vendor-patterns-file` argument feeding per-vendor
     `[fallback]` and `[skip]` regex sections.

  `method_gate.py`, `evidence_schema.py`, `memory_store.py`, `verify_insight.py` stay
  byte-identical.

  ## Deviations from the spec (recorded)

  - **Machine-read driver files are JSON, not YAML.** Spec §4.4/§4.7 write `manifest.yaml`/
    `idioms.yaml`; Part B uses `manifest.json`/`idioms.json` so the validator parses them with
    stdlib `json` (Python has no stdlib YAML parser; we add no `pyyaml`/Node dependency). Field
    names are unchanged.
  - **The L0 validator is Python in `_substrate/backends/`, not Node in `_meta/tools/`, and ships
    no JSON-Schema files.** Spec §4.1/§9.2 name `_meta/tools/validate-backend.js` + `_schema/`.
    Part B ships `_substrate/backends/validate_backend.py` with hand-rolled stdlib checks that
    `import method_gate` to read the live `TABLE`, keeping the validator + this doc as the
    contract. Any future generator (§9.2) work expecting `_schema/*.json` must consume this
    Python validator instead.
  - **L0 `backend_id` check is literal equality, not canonical-form.** Spec §4.9 L0 requires
    `backend_id == dir == normalizeSuitabilityValue(id)` (a three-way equality pinning the id to
    post-normalize canonical form). This plan's validator asserts only `backend_id == basename(dir)`;
    it does NOT port `normalizeSuitabilityValue`'s lowercase/`_`→`-`/alias rules into Python. P3
    driver authoring MUST name driver dirs in already-canonical form (lowercase, `-` not `_`); the
    canonical-form assertion is a deferred L0 leg to add when the JS normalize logic is ported.
  ```

- [ ] **Step 2: Verify the file exists and contains every required section header.**

  Run:
  ```
  test -f _substrate/BACKEND-DRIVER-SDK.md && \
  grep -E '^## The six-file driver layout$' _substrate/BACKEND-DRIVER-SDK.md && \
  grep -E "^## .manifest.json. fields$" _substrate/BACKEND-DRIVER-SDK.md && \
  grep -E "^## .idioms.json. fields$" _substrate/BACKEND-DRIVER-SDK.md && \
  grep -E '^## .to_evidence.py. — the neutral interface$' _substrate/BACKEND-DRIVER-SDK.md && \
  grep -E '^### Canonical-unit table' _substrate/BACKEND-DRIVER-SDK.md && \
  grep -E '^### The decided null rule$' _substrate/BACKEND-DRIVER-SDK.md && \
  grep -E '^## The three portability tiers$' _substrate/BACKEND-DRIVER-SDK.md && \
  grep -E '^## Conformance levels .L0–L3.$' _substrate/BACKEND-DRIVER-SDK.md && \
  grep -E '^## The scoped substrate edits' _substrate/BACKEND-DRIVER-SDK.md && \
  grep -E '^## Deviations from the spec' _substrate/BACKEND-DRIVER-SDK.md
  ```

  Expected: every `grep` prints its matching header line and the compound command exits 0 (all required sections present: six-file layout, manifest field table, idioms field table, to_evidence neutral interface, canonical-unit table, decided null rule, three portability tiers, L0–L3 levels, scoped-substrate-edits list, deviations).

- [ ] **Step 3: Commit the SDK doc.**

  Run:
  ```
  git add _substrate/BACKEND-DRIVER-SDK.md
  git commit -m "$(cat <<'EOF'
  Add backend driver SDK contract doc

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```
  Expected: one commit created; `git status` clean for that path.

---

## Final verification

- [ ] **Step 1: Run the full substrate test suite from the repo root.**

  Run: `python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v`

  Expected: ends with `OK`. With both parts landed the suite contains `test_smoke.py` (1), `test_diagnose.py` (11), `test_anti_cheat.py` (10), and `test_validate_backend.py` (5) — all green, no failures, no errors.

- [ ] **Step 2: Confirm exactly the intended files changed and no workflow or unrelated substrate script was touched.**

  Run: `git diff --name-only 00b920b..HEAD`

  Expected: only the paths in the File Structure table appear — `_substrate/tests/test_smoke.py`, `_substrate/tests/test_diagnose.py`, `_substrate/diagnose.py`, `_substrate/tests/test_anti_cheat.py`, `_substrate/anti_cheat.py`, `_substrate/backends/__init__.py`, `_substrate/backends/validate_backend.py`, `_substrate/backends/REGISTRY.md`, `_substrate/BACKEND-DRIVER-SDK.md`, `_substrate/tests/test_validate_backend.py`, and the ten `_substrate/tests/fixtures/**` JSON files. **No `.js` workflow file appears; the only substrate scripts modified are `diagnose.py` and `anti_cheat.py`** — `method_gate.py`, `evidence_schema.py`, `memory_store.py`, `verify_insight.py`, and every other substrate script stay byte-identical.

---

## Definition of Done

This checklist maps directly to spec Appendix A rows **P1** and **P2**.

**P2 — substrate edits (`diagnose.py`, `anti_cheat.py`):**
- [ ] `diagnose.py` carries a `_vendor` threshold **vendor profile** (`PROFILES`, default `nvidia` == today's literals). *(Task A2 Step 1)*
- [ ] `diagnose.py` implements the **full measured-operand null rule**, including the two-sided **`memory_bound`** *and* **`overhead_bound`** branches firing only when both `dram_pct` and `sm_pct` are measured; `compute_bound` (sm alone) and `latency_occupancy` (occ alone) may still fire; missing discriminator → `unknown` with `"no dominant signal (insufficient measured metrics)"` and no `:.0f`-on-`None` crash. *(Task A2 Step 2)*
- [ ] NVIDIA byte-identity preserved on every all-measured / already-working case (golden tests green). *(Task A1 golden + Task A2 Step 3)*
- [ ] `anti_cheat.py` `--vendor-patterns-file` is implemented and covers **both** the `[fallback]` and `[skip]` lists, threaded through `static_flags`/`evaluate`/`main` with byte-identical CUDA defaults. *(Task A4)*

**P1 — driver-contract scaffolding:**
- [ ] `validate_backend.py` performs the **L0 checks**: `backend_id == dir`, `capabilities.metrics ⊆` the 4 canonical keys, `bottleneck_classes ⊆` the 4 meaningful classes, and every `methods` key + `unsupported_methods` entry is a live `method_gate.TABLE` name. *(Tasks B1, B2)*
- [ ] `_substrate/backends/REGISTRY.md` exists with the required columns, seeded with the `cuda — planned` row. *(Task B3)*
- [ ] `_substrate/BACKEND-DRIVER-SDK.md` exists with every required section (six-file layout, manifest fields, idioms fields, to_evidence neutral interface, canonical-unit table, decided null rule, three portability tiers, L0–L3 conformance, scoped-substrate-edits cross-reference, recorded deviations). *(Task B4)*

**Cross-cutting gates:**
- [ ] All tests green via the single command `python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v` (final line `OK`). *(Final verification Step 1)*
- [ ] The substrate diff is **exactly** the two documented edits: the `diagnose.py` `PROFILES` + `classify()` hunk and the `anti_cheat.py` `load_vendor_patterns`/`static_flags`/`evaluate`/`main` hunk — no other substrate script changed (`method_gate.py`, `evidence_schema.py`, `memory_store.py`, `verify_insight.py` byte-identical). *(Final verification Step 2)*
- [ ] **No workflow file touched** — `git diff --name-only` shows zero `.js` workflow paths. *(Final verification Step 2)*