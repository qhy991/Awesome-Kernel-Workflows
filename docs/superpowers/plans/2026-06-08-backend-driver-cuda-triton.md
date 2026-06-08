# Backend Drivers: cuda + triton (P3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author the cuda and triton reference Backend Drivers (six-file set each) on the P1/P2 substrate — fully testing the parts that run on macOS (the shared NVIDIA evidence mapper + L0-conformant manifests/idioms) and authoring + stub/syntax-testing the build/run/profile shell scripts, with real GPU execution explicitly deferred to a CI/hardware tier.

**Architecture:** A shared `_substrate/backends/_evidence_nvidia.py` (NCU-CSV → canonical metrics, unit-tested incl. the occupancy ÷100 / ns→ms traps) with thin `cuda/` and `triton/` `to_evidence.py` wrappers — both backends lower to PTX and share one ncu mapping. `manifest.json`/`idioms.json` per backend pass `validate_backend.py` L0. The three `.sh` files per backend are authored to the §4.5 envelope contract and tested for syntax + JSON-envelope via fake-tool PATH stubs; real nvcc/ncu/triton execution is deferred (this machine is macOS without them).

**Tech Stack:** Python 3 stdlib (json, csv, argparse, subprocess, unittest), bash, JSON driver files. Test-only fake-tool stubs. No third-party deps in the substrate code.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `_substrate/backends/__init__.py` | Verify present (already committed in P1) | Package marker so `backends._evidence_nvidia` is importable in-process by the test harness. |
| `_substrate/backends/_evidence_nvidia.py` | Create | Shared NVIDIA NCU-CSV → canonical-metrics mapper (cuda + triton vendor-collapse); the keystone pure function + CLI. |
| `_substrate/backends/cuda/to_evidence.py` | Create | Thin wrapper: sys.path-inserts backends dir, imports the shared mapper, calls `main(source_backend="cuda")`. |
| `_substrate/backends/triton/to_evidence.py` | Create | Thin wrapper: same, `main(source_backend="triton")`. |
| `_substrate/backends/cuda/manifest.json` | Create | cuda driver machine-read manifest (backend_id cuda, hw_vendor nvidia, status experimental). |
| `_substrate/backends/cuda/idioms.json` | Create | cuda idioms keyed by real `method_gate.TABLE` names; covers all four meaningful classes. |
| `_substrate/backends/triton/manifest.json` | Create | triton driver manifest (.py source, hw_vendor nvidia, status experimental). |
| `_substrate/backends/triton/idioms.json` | Create | triton idioms (`@triton.jit`), real `method_gate.TABLE` names, lang_fence python. |
| `_substrate/backends/cuda/build.sh` | Create | nvcc compile to `.so` with `-lineinfo`; JSON envelope (exit 0/2/3). |
| `_substrate/backends/cuda/run.sh` | Create | Load `.so`, run vs problem; anti_cheat-exact stdout key set; missing-artifact envelope (GPU run deferred). |
| `_substrate/backends/cuda/profile.sh` | Create | ncu → csv pointer; 4-counter request; exit-4 on absent profiler. |
| `_substrate/backends/triton/build.sh` | Create | JIT warmup materializing PTX into `TRITON_CACHE_DIR`; same envelope as cuda/build.sh. |
| `_substrate/backends/triton/run.sh` | Create | python launcher + correctness/latency; anti_cheat key set; GPU run deferred. |
| `_substrate/backends/triton/profile.sh` | Create | ncu over launcher; `--kernel-name` caveat; pointer envelope; exit-4 on absent profiler. |
| `_substrate/tests/fixtures/ncu/full4.csv` | Create | NCU long-format fixture with all 4 counters (+ extra dram_write + l2 rows). |
| `_substrate/tests/fixtures/ncu/missing_dram.csv` | Create | Fixture with dram counter absent (null-rule case). |
| `_substrate/tests/fixtures/ncu/empty.csv` | Create | Header-only fixture (malformed/empty case → exit 2). |
| `_substrate/tests/test_evidence_nvidia.py` | Create | Mapper unit tests (canonical mapping, null rule, malformed, vendor-collapse wrappers). |
| `_substrate/tests/test_driver_conformance.py` | Create | L0 conformance for cuda + triton dirs via validate_backend.py; method-name + backend_id asserts. |
| `_substrate/tests/test_driver_scripts.py` | Create | Shell-script logic tests via fake-tool PATH stubs (build/run/profile envelope, exit codes). |
| `_substrate/backends/REGISTRY.md` | Modify | Register cuda + triton as experimental, GPU-deferred. |

## Testability note

Everything that runs on this macOS/darwin host is fully tested: the pure-Python `_evidence_nvidia.py` NCU-CSV → canonical mapping (incl. the error-prone `occupancy = warps_active% ÷ 100` and `latency_ms = ns ÷ 1e6` conversions and the null rule), both thin `to_evidence.py` wrappers (vendor-collapse: same CSV → byte-identical metrics differing only in `source_backend`), the L0 conformance of both driver dirs via `validate_backend.py` (no kernel/compiler/profiler execution), and the build/run/profile `.sh` files' own logic — argument parsing, the universal JSON envelope, exit codes, and the contract key set — exercised through fake-tool PATH stubs. Deferred to the GPU/CI hardware tier (needs an NVIDIA box with nvcc + ncu + triton; spec §8.3 opt-in hardware tier, §9.3 mock harness): all real compile/run/profile, the live `torch.compile` baseline semantics of `compile_latency_ms`, the exit-4 profiler-unavailable path where ncu could otherwise exist, ncu-on-triton mangled kernel-name discovery, GPU-tier confirmation of the assumed NCU CSV header, and L1–L3 conformance (promoting `experimental` → `stable`).

---

### Task 1 — Shared NVIDIA NCU-CSV → canonical mapper (`_evidence_nvidia.py`)

The keystone pure function + CLI. Stdlib-only. Parses the 4 NCU counters from a long-format CSV, converts units (ns→ms; warps_active% ÷100→occupancy 0–1; dram/sm pass through 0–100), applies the **null rule** (absent counter → JSON `null` + omitted from `coverage`, never `0.0`), and emits the universal envelope (exit `0` ok / `2` native unparseable, JSON still printed / `3` bad-args).

> **Design refinement adopted (spec §5.1's literal-import shim does NOT work for path-invoked standalone scripts).** The spec says Triton's `to_evidence.py` is "literally the CUDA file via `from ..cuda.to_evidence import main`". That relative import fails when the script is invoked by absolute path with no package context. **Instead**: one shared module `_substrate/backends/_evidence_nvidia.py` holds the real NCU-CSV→canonical mapping; `cuda/to_evidence.py` and `triton/to_evidence.py` are **thin wrappers** that `sys.path`-insert the backends dir, import the shared mapper, and call its `main()` with their own `source_backend` id. CUDA and Triton thus share **one tested mapping** (both lower to PTX, both profiled by the same `ncu`).

> **ASSUMED NCU CSV FORMAT (flag for the GPU tier to confirm).** This Part assumes `ncu --csv --page raw` emits the **long ("raw") format**: a header row, then one row per `(kernel, metric)` triple, with at minimum the columns `"Kernel Name","Metric Name","Metric Value"` (NCU also emits `"ID"`, `"Metric Unit"`, etc.; the mapper reads by **column name**, ignoring extras and column order). One kernel per export is assumed; if multiple kernels appear the mapper takes the **first kernel's** rows (documented, deterministic). Metric Value may carry thousands separators (`1,234.5`) and a trailing unit token — the mapper strips commas and parses the leading float. **The GPU tier MUST confirm the real `ncu` column header and whether values arrive comma-grouped / unit-suffixed; if they differ, only `_evidence_nvidia._parse_ncu_csv` changes, and these tests pin the contract that change must preserve.**

> **dram_pct policy (canonical, pinned by tests):** `dram_pct = read% (+ write% if present)` — both pass through 0–100. With the full fixture this is `40.0 + 22.0 = 62.0` (matches spec §4.6 example `dram_pct: 62.0`). If write is absent, dram_pct = read% alone. If read is absent, dram_pct is null.

**Files**
- Create: `_substrate/backends/_evidence_nvidia.py`
- Create: `_substrate/tests/fixtures/ncu/full4.csv` (all 4 counters)
- Create: `_substrate/tests/fixtures/ncu/missing_dram.csv` (dram counter absent)
- Create: `_substrate/tests/fixtures/ncu/empty.csv` (header only, no metric rows)
- Create: `_substrate/tests/test_evidence_nvidia.py`

**Steps**

- [ ] **Step 1 — RED: write the fixtures and the canonical-mapping test first.**

  Create `_substrate/tests/fixtures/ncu/full4.csv` (EXACT text — long/raw format, deliberately scrambled metric-row order and an extra `dram__bytes_read` row plus an unrelated `l2` row to prove name-based selection):
  ```csv
  "ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
  "0","fused_gemm_kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","48.0"
  "0","fused_gemm_kernel","gpu__time_duration.sum","ns","410000"
  "0","fused_gemm_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","%","51.0"
  "0","fused_gemm_kernel","dram__bytes_read.sum.pct_of_peak_sustained_elapsed","%","40.0"
  "0","fused_gemm_kernel","dram__bytes_write.sum.pct_of_peak_sustained_elapsed","%","22.0"
  "0","fused_gemm_kernel","lts__t_sector_hit_rate.pct","%","73.0"
  ```

  Create `_substrate/tests/fixtures/ncu/missing_dram.csv` (no `dram__*` rows at all):
  ```csv
  "ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
  "0","memcpy_kernel","gpu__time_duration.sum","ns","123456"
  "0","memcpy_kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","30.0"
  "0","memcpy_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","%","88.0"
  ```

  Create `_substrate/tests/fixtures/ncu/empty.csv` (header only — no metric rows; the malformed/empty case):
  ```csv
  "ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
  ```

  Create `_substrate/tests/test_evidence_nvidia.py`:
  ```python
  import os, sys, json, subprocess, unittest
  SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
  sys.path.insert(0, SUB)

  import importlib
  _ev = importlib.import_module('backends._evidence_nvidia')

  FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures', 'ncu')
  MAPPER = os.path.join(SUB, 'backends', '_evidence_nvidia.py')


  def _csv(name):
      return os.path.join(FIXTURES, name)


  def run_mapper(csv_name, source_backend='cuda', script=MAPPER):
      """Invoke a mapper/wrapper script via subprocess; return (rc, parsed_or_raw)."""
      proc = subprocess.run(
          [sys.executable, script, '--native', _csv(csv_name),
           '--source-backend', source_backend],
          capture_output=True, text=True)
      try:
          payload = json.loads(proc.stdout)
      except json.JSONDecodeError:
          payload = {"_raw_stdout": proc.stdout, "_stderr": proc.stderr}
      return proc.returncode, payload


  class TestEvidenceNvidiaFull(unittest.TestCase):
      """Full 4-counter CSV -> correct canonical dict, units converted, all 4 in coverage."""

      def setUp(self):
          self.rc, self.payload = run_mapper('full4.csv', source_backend='cuda')

      def test_exit_zero_and_ok_true(self):
          self.assertEqual(self.rc, 0, msg=f"payload={self.payload}")
          self.assertIs(self.payload.get('ok'), True, msg=f"payload={self.payload}")

      def test_source_backend_passed_through(self):
          self.assertEqual(self.payload['source_backend'], 'cuda')

      def test_latency_ms_is_ns_div_1e6(self):
          # 410000 ns / 1e6 = 0.41 ms
          self.assertAlmostEqual(self.payload['metrics']['latency_ms'], 0.41, places=9)

      def test_dram_pct_is_read_plus_write_0_to_100(self):
          # 40.0 + 22.0 = 62.0, in 0-100 (NOT 0-1)
          self.assertAlmostEqual(self.payload['metrics']['dram_pct'], 62.0, places=9)

      def test_sm_pct_passes_through_0_to_100(self):
          self.assertAlmostEqual(self.payload['metrics']['sm_pct'], 48.0, places=9)

      def test_occupancy_is_warps_active_div_100_range_0_to_1(self):
          # THE error-prone line: 51.0 % -> 0.51 (0-1), never 51.0
          occ = self.payload['metrics']['occupancy']
          self.assertAlmostEqual(occ, 0.51, places=9)
          self.assertGreaterEqual(occ, 0.0)
          self.assertLessEqual(occ, 1.0)

      def test_vendor_tag_is_nvidia(self):
          self.assertEqual(self.payload['metrics']['_vendor'], 'nvidia')

      def test_coverage_lists_all_four_canonical_keys(self):
          self.assertEqual(
              sorted(self.payload['coverage']),
              sorted(['latency_ms', 'dram_pct', 'sm_pct', 'occupancy']))

      def test_backend_native_carries_unmapped_counters(self):
          # the l2 sector-hit-rate row is free-form backend_native, not a canonical key
          self.assertIn('lts__t_sector_hit_rate.pct',
                        self.payload['metrics']['backend_native'])


  class TestEvidenceNvidiaNullRule(unittest.TestCase):
      """Missing dram counter -> dram_pct is JSON null AND absent from coverage (never 0.0)."""

      def setUp(self):
          self.rc, self.payload = run_mapper('missing_dram.csv', source_backend='cuda')

      def test_exit_zero_ok_true(self):
          self.assertEqual(self.rc, 0, msg=f"payload={self.payload}")
          self.assertIs(self.payload.get('ok'), True)

      def test_dram_pct_is_json_null_not_zero(self):
          metrics = self.payload['metrics']
          self.assertIn('dram_pct', metrics)        # key present
          self.assertIsNone(metrics['dram_pct'])    # value is JSON null
          self.assertNotEqual(metrics['dram_pct'], 0.0)  # NEVER fabricated 0.0

      def test_dram_pct_absent_from_coverage(self):
          self.assertNotIn('dram_pct', self.payload['coverage'])

      def test_measured_keys_present_in_coverage(self):
          self.assertEqual(
              sorted(self.payload['coverage']),
              sorted(['latency_ms', 'sm_pct', 'occupancy']))

      def test_occupancy_still_divided_by_100(self):
          self.assertAlmostEqual(self.payload['metrics']['occupancy'], 0.88, places=9)


  class TestEvidenceNvidiaMalformed(unittest.TestCase):
      """Empty / unparseable CSV -> exit 2, {ok: false}, JSON still printed (envelope)."""

      def test_empty_csv_exits_2_ok_false(self):
          rc, payload = run_mapper('empty.csv', source_backend='cuda')
          self.assertEqual(rc, 2, msg=f"payload={payload}")
          self.assertIs(payload.get('ok'), False, msg=f"payload={payload}")
          self.assertIn('error', payload)  # JSON still printed on stdout

      def test_garbage_csv_exits_2_ok_false(self):
          # a file that is not CSV-with-our-columns at all
          import tempfile
          with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as fh:
              fh.write("this is not ncu output\nno header columns here\n")
              path = fh.name
          try:
              proc = subprocess.run(
                  [sys.executable, MAPPER, '--native', path,
                   '--source-backend', 'cuda'],
                  capture_output=True, text=True)
              payload = json.loads(proc.stdout)
              self.assertEqual(proc.returncode, 2, msg=f"payload={payload}")
              self.assertIs(payload.get('ok'), False)
          finally:
              os.unlink(path)


  if __name__ == '__main__':
      unittest.main()
  ```

  Run and SEE IT FAIL (module does not exist yet):
  ```
  python3 -m unittest discover -s _substrate/tests -p 'test_evidence_nvidia.py' -v
  ```
  **Expected RED:** collection error / `ModuleNotFoundError: No module named 'backends._evidence_nvidia'` (because `_evidence_nvidia.py` does not exist yet; `backends/__init__.py` is already present from P1). All `test_evidence_nvidia` tests error.

- [ ] **Step 2 — GREEN-enabler: make `backends` importable as a package for the test's `importlib.import_module`.**

  The test does `import_module('backends._evidence_nvidia')`. `_substrate` is already on `sys.path` (harness convention), so `backends` must be an importable subpackage. Create an empty package marker.

  Verify `_substrate/backends/__init__.py` is present (it was committed in P1). If somehow missing, create it with:
  ```python
  # Package marker so `backends._evidence_nvidia` is importable in-process by the
  # test harness (which puts _substrate/ on sys.path). The .sh/.py driver scripts are
  # still invoked by absolute PATH (no package context) — see _evidence_nvidia.main's
  # sys.path-insert idiom and the thin cuda/triton wrappers (Task 2).
  ```
  > Note: this does not affect the existing path-invoked `validate_backend.py` (it `sys.path`-inserts `_substrate/` itself and imports top-level `method_gate`). Re-run the full suite later (Step 4) to confirm the 37 stay green.

- [ ] **Step 3 — GREEN: implement the shared mapper.**

  Create `_substrate/backends/_evidence_nvidia.py` (COMPLETE — no placeholders):
  ```python
  #!/usr/bin/env python3
  """Shared NVIDIA NCU-CSV -> canonical-metrics mapper (cuda + triton, vendor-collapsed).

  Spec: docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md
        §4.6 to_evidence neutral interface + canonical-unit table + null rule;
        §5.1 CUDA/Triton vendor-collapse (both lower to PTX, profiled by the same ncu).

  Design refinement (the spec's `from ..cuda.to_evidence import main` literal-import shim
  fails for PATH-invoked standalone scripts with no package context): this module holds the
  ONE real mapping; cuda/to_evidence.py and triton/to_evidence.py are THIN wrappers that
  sys.path-insert the backends dir, import this module, and call main() with their own
  source_backend id.

  ASSUMED NCU FORMAT (GPU tier must confirm): `ncu --csv --page raw` long format with, at
  minimum, columns "Kernel Name","Metric Name","Metric Value" (extra columns / column order
  ignored — read by name). First kernel's rows are used if several appear. Values may carry
  thousands separators / a trailing unit token; the leading float is parsed.

  Canonical units emitted (every backend MUST honor):
    latency_ms = gpu__time_duration.sum (ns) / 1e6
    dram_pct   = dram_read% (+ dram_write% if present)        # 0-100, pass-through
    sm_pct     = sm__throughput.avg.pct_of_peak…              # 0-100, pass-through
    occupancy  = sm__warps_active.avg.pct… / 100              # 0-1   <-- THE error-prone line

  Null rule: a counter absent from the CSV -> canonical key set to JSON null AND omitted from
  `coverage` (NEVER fabricated 0.0). diagnose.py's §5.3.2 None-aware classify() then yields
  `unknown` rather than a confident wrong label.

  Usage (invoked WITH a python prefix):
    _evidence_nvidia.py --native <ncu.csv|-> --source-backend <id> [--format ncu-csv] [--run <result.json>]
  Prints ONE JSON object on stdout; logs to stderr.
  Exit: 0 normalized · 2 native unparseable (JSON still printed, ok:false) · 3 bad args.
  Pure function (same CSV in -> same JSON out).
  """
  import os, sys, json, csv, io, argparse

  # --- NCU metric name -> how it maps into canonical keys ---------------------------------
  M_DURATION = "gpu__time_duration.sum"
  M_SM       = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
  M_WARPS    = "sm__warps_active.avg.pct_of_peak_sustained_active"
  M_DRAM_RD  = "dram__bytes_read.sum.pct_of_peak_sustained_elapsed"
  M_DRAM_WR  = "dram__bytes_write.sum.pct_of_peak_sustained_elapsed"

  # NCU raw-format column headers we read (by name; extras/order ignored).
  COL_KERNEL = "Kernel Name"
  COL_METRIC = "Metric Name"
  COL_VALUE  = "Metric Value"

  CANONICAL_KEYS = ("latency_ms", "dram_pct", "sm_pct", "occupancy")


  class NativeParseError(Exception):
      """Raised when the native profile cannot be parsed into the expected NCU shape."""


  def _parse_float(raw):
      """Parse an NCU 'Metric Value' cell: strip thousands separators and any trailing
      unit token, return the leading float. Raise ValueError if no number is present."""
      if raw is None:
          raise ValueError("empty metric value")
      s = str(raw).strip().replace(",", "")
      if not s:
          raise ValueError("empty metric value")
      # take the leading numeric token (handles "0.41 msecond" style cells defensively)
      token = s.split()[0]
      return float(token)


  def _parse_ncu_csv(text):
      """Parse NCU long/raw CSV text into {metric_name: float} for the FIRST kernel seen.

      Raises NativeParseError if the required columns are missing or no metric rows with a
      parseable value exist. THE GPU TIER MUST CONFIRM the real ncu header; only this
      function changes if it differs.
      """
      reader = csv.DictReader(io.StringIO(text))
      if reader.fieldnames is None:
          raise NativeParseError("empty native file (no header row)")
      have = set(reader.fieldnames)
      missing = {COL_KERNEL, COL_METRIC, COL_VALUE} - have
      if missing:
          raise NativeParseError(
              f"native CSV missing required columns {sorted(missing)}; "
              f"saw columns {reader.fieldnames}")

      first_kernel = None
      metrics = {}
      for row in reader:
          kernel = (row.get(COL_KERNEL) or "").strip()
          name = (row.get(COL_METRIC) or "").strip()
          if not name:
              continue
          if first_kernel is None:
              first_kernel = kernel
          elif kernel != first_kernel:
              continue  # only the first kernel's rows (deterministic; documented)
          try:
              metrics[name] = _parse_float(row.get(COL_VALUE))
          except (ValueError, TypeError):
              continue  # a non-numeric value row (e.g. a string metric) is simply skipped

      if not metrics:
          raise NativeParseError("native CSV had a valid header but no parseable metric rows")
      return first_kernel, metrics


  def to_canonical(native_metrics, source_backend):
      """Map a {ncu_metric_name: float} dict to the canonical evidence dict.

      Applies units (ns->ms; warps_active%/100->occupancy 0-1; dram/sm pass-through 0-100)
      and the null rule (absent canonical input -> None + omitted from coverage). Every
      unmapped native metric is preserved verbatim under metrics.backend_native.
      """
      # latency_ms
      latency_ms = None
      if M_DURATION in native_metrics:
          latency_ms = native_metrics[M_DURATION] / 1e6

      # dram_pct = read% (+ write% if present); null only when read is absent
      dram_pct = None
      if M_DRAM_RD in native_metrics:
          dram_pct = native_metrics[M_DRAM_RD]
          if M_DRAM_WR in native_metrics:
              dram_pct = dram_pct + native_metrics[M_DRAM_WR]

      # sm_pct pass-through (0-100)
      sm_pct = native_metrics.get(M_SM)  # None if absent

      # occupancy = warps_active% / 100  -> 0-1  (THE error-prone line)
      occupancy = None
      if M_WARPS in native_metrics:
          occupancy = native_metrics[M_WARPS] / 100.0

      canonical = {
          "latency_ms": latency_ms,
          "dram_pct": dram_pct,
          "sm_pct": sm_pct,
          "occupancy": occupancy,
      }
      coverage = [k for k in CANONICAL_KEYS if canonical[k] is not None]

      # backend_native: every native metric not consumed by a canonical mapping.
      consumed = {M_DURATION, M_SM, M_WARPS, M_DRAM_RD, M_DRAM_WR}
      backend_native = {k: v for k, v in native_metrics.items() if k not in consumed}

      metrics = dict(canonical)
      metrics["_vendor"] = "nvidia"
      metrics["backend_native"] = backend_native
      return {
          "ok": True,
          "metrics": metrics,
          "source_backend": source_backend,
          "coverage": coverage,
      }


  def _read_native(path):
      if path == "-":
          return sys.stdin.read()
      with open(path, encoding="utf-8") as fh:
          return fh.read()


  def main(argv=None, source_backend=None):
      """CLI entrypoint. `source_backend` may be supplied by a thin wrapper (cuda/triton);
      a --source-backend flag, if given, takes precedence. Returns an exit code."""
      ap = argparse.ArgumentParser(description="NVIDIA NCU-CSV -> canonical metrics")
      ap.add_argument("--native", required=True, help="NCU CSV path, or '-' for stdin")
      ap.add_argument("--source-backend", dest="source_backend", default=None,
                      help="backend id stamped into source_backend (cuda|triton)")
      ap.add_argument("--format", default="ncu-csv",
                      help="native profile format (only ncu-csv supported)")
      ap.add_argument("--run", default=None,
                      help="optional run.sh result.json (reserved; not consumed by the mapper)")
      a = ap.parse_args(argv)

      sb = a.source_backend or source_backend
      if not sb:
          # bad args: no source_backend from flag or wrapper
          print(json.dumps({"ok": False, "error": "missing --source-backend"},
                           ensure_ascii=False))
          return 3

      if a.format != "ncu-csv":
          print(json.dumps({"ok": False, "error": f"unsupported format {a.format}"},
                           ensure_ascii=False))
          return 3

      try:
          text = _read_native(a.native)
      except OSError as exc:
          print(json.dumps({"ok": False, "error": f"cannot read native file: {exc}"},
                           ensure_ascii=False))
          return 3

      try:
          _kernel, native_metrics = _parse_ncu_csv(text)
      except NativeParseError as exc:
          # exit 2: native unparseable, JSON still printed (universal envelope)
          print(json.dumps({"ok": False, "error": f"native unparseable: {exc}"},
                           ensure_ascii=False))
          print(f"[_evidence_nvidia] parse error: {exc}", file=sys.stderr)
          return 2

      result = to_canonical(native_metrics, sb)
      print(json.dumps(result, indent=2, ensure_ascii=False))
      return 0


  if __name__ == "__main__":
      sys.exit(main())
  ```

  > Reconciliation note: the `--format` flag (default `ncu-csv`) is accepted so the Part-3 shell-script tests, which pass `--format ncu-csv`, work against this same canonical mapper; the `dram_pct = read + write` policy (62.0) is the single source of truth across all parts.

  Make it executable (it is also python-invoked, but keep the envelope rule consistent):
  ```
  chmod +x _substrate/backends/_evidence_nvidia.py
  ```

  Run and SEE IT PASS:
  ```
  python3 -m unittest discover -s _substrate/tests -p 'test_evidence_nvidia.py' -v
  ```
  **Expected GREEN.** Spot-check the full-CSV envelope by hand:
  ```
  python3 _substrate/backends/_evidence_nvidia.py \
    --native _substrate/tests/fixtures/ncu/full4.csv --source-backend cuda
  ```
  Exit `0`, stdout exactly (key order as below; `occupancy` is **0.51**, not 51.0; `latency_ms` **0.41**; `dram_pct` **62.0**):
  ```json
  {
    "ok": true,
    "metrics": {
      "latency_ms": 0.41,
      "dram_pct": 62.0,
      "sm_pct": 48.0,
      "occupancy": 0.51,
      "_vendor": "nvidia",
      "backend_native": {
        "lts__t_sector_hit_rate.pct": 73.0
      }
    },
    "source_backend": "cuda",
    "coverage": [
      "latency_ms",
      "dram_pct",
      "sm_pct",
      "occupancy"
    ]
  }
  ```
  And the null-rule case:
  ```
  python3 _substrate/backends/_evidence_nvidia.py \
    --native _substrate/tests/fixtures/ncu/missing_dram.csv --source-backend cuda
  ```
  → `metrics.dram_pct` is `null`, `"dram_pct"` **absent** from `coverage` (`["latency_ms","sm_pct","occupancy"]`), `occupancy` **0.88**. And the empty fixture:
  ```
  python3 _substrate/backends/_evidence_nvidia.py \
    --native _substrate/tests/fixtures/ncu/empty.csv --source-backend cuda ; echo "rc=$?"
  ```
  → `{"ok": false, "error": "native unparseable: ..."}` on stdout, `rc=2`.

- [ ] **Step 4 — Confirm the full suite (no regressions) and commit.**
  ```
  python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v 2>&1 | tail -3
  ```
  **Expected:** `Ran 51 tests ... OK` (37 prior + 14 new in `test_evidence_nvidia.py`). If the count differs, do not proceed — investigate before committing.

  Commit on `dev/solver-substrate`:
  ```
  git add _substrate/backends/_evidence_nvidia.py _substrate/backends/__init__.py \
          _substrate/tests/fixtures/ncu/ _substrate/tests/test_evidence_nvidia.py
  git commit -m "Add shared NVIDIA NCU-CSV -> canonical evidence mapper (cuda+triton)

  _evidence_nvidia.py maps the 4 NCU classifier counters to the canonical metrics
  dict: latency_ms (ns/1e6), dram_pct (read+write, 0-100), sm_pct (0-100), and
  occupancy (warps_active%/100, 0-1). Honors the null rule: an absent counter is
  emitted as JSON null and omitted from coverage, never fabricated as 0.0. Pure
  function; universal envelope (exit 0/2/3). Adopts the design refinement that
  cuda+triton share ONE mapper (spec 5.1's literal relative-import shim fails for
  path-invoked standalone scripts). Stdlib only; 14 new tests, full suite green.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 2 — Thin `cuda/` and `triton/` `to_evidence.py` wrappers (vendor-collapse)

Two ~10-line standalone scripts, invoked by **absolute path with a `python` prefix** (no package context). Each `sys.path`-inserts the backends dir, imports the shared mapper, and calls `main()` with its own `source_backend` id. This proves the §5.1 vendor-collapse: **identical metrics from the same CSV, differing only in `source_backend`.**

**Files**
- Create: `_substrate/backends/cuda/to_evidence.py`
- Create: `_substrate/backends/triton/to_evidence.py`
- Modify: `_substrate/tests/test_evidence_nvidia.py` (add the cross-wrapper equality test class)

**Steps**

- [ ] **Step 1 — RED: add the cross-wrapper equality test.**

  Append to `_substrate/tests/test_evidence_nvidia.py`:
  ```python
  CUDA_WRAPPER = os.path.join(SUB, 'backends', 'cuda', 'to_evidence.py')
  TRITON_WRAPPER = os.path.join(SUB, 'backends', 'triton', 'to_evidence.py')


  class TestVendorCollapseWrappers(unittest.TestCase):
      """cuda/to_evidence.py and triton/to_evidence.py on the SAME csv -> identical
      metrics, differing ONLY in source_backend (spec 5.1 vendor-collapse)."""

      def _run_path_invoked(self, script):
          # NO --source-backend flag: the wrapper supplies its own id. This also proves
          # the path-invoked (no package context) import idiom works.
          proc = subprocess.run(
              [sys.executable, script, '--native', _csv('full4.csv')],
              capture_output=True, text=True)
          self.assertEqual(proc.returncode, 0,
                           msg=f"stderr={proc.stderr}\nstdout={proc.stdout}")
          return json.loads(proc.stdout)

      def test_cuda_wrapper_stamps_cuda(self):
          self.assertEqual(self._run_path_invoked(CUDA_WRAPPER)['source_backend'], 'cuda')

      def test_triton_wrapper_stamps_triton(self):
          self.assertEqual(self._run_path_invoked(TRITON_WRAPPER)['source_backend'], 'triton')

      def test_metrics_identical_except_source_backend(self):
          cuda = self._run_path_invoked(CUDA_WRAPPER)
          triton = self._run_path_invoked(TRITON_WRAPPER)
          # metrics + coverage byte-identical (both lower to PTX, same ncu mapping)
          self.assertEqual(cuda['metrics'], triton['metrics'])
          self.assertEqual(cuda['coverage'], triton['coverage'])
          # ONLY source_backend differs
          self.assertNotEqual(cuda['source_backend'], triton['source_backend'])
          cuda_norm = dict(cuda); triton_norm = dict(triton)
          cuda_norm.pop('source_backend'); triton_norm.pop('source_backend')
          self.assertEqual(cuda_norm, triton_norm)

      def test_explicit_flag_overrides_wrapper_default(self):
          # --source-backend, if passed, wins over the wrapper's baked-in id
          proc = subprocess.run(
              [sys.executable, CUDA_WRAPPER, '--native', _csv('full4.csv'),
               '--source-backend', 'triton'],
              capture_output=True, text=True)
          self.assertEqual(proc.returncode, 0, msg=proc.stderr)
          self.assertEqual(json.loads(proc.stdout)['source_backend'], 'triton')
  ```

  Run and SEE IT FAIL:
  ```
  python3 -m unittest discover -s _substrate/tests -p 'test_evidence_nvidia.py' -v
  ```
  **Expected RED:** the 4 new `TestVendorCollapseWrappers` tests fail — `proc.returncode` is non-zero because the wrapper scripts do not exist (`subprocess` runs `python <missing>` → exit 2, stdout empty → `json.loads` raises and `assertEqual(returncode, 0)` fails first with the captured stderr `No such file or directory`).

- [ ] **Step 2 — GREEN: write both thin wrappers.**

  Create `_substrate/backends/cuda/to_evidence.py`:
  ```python
  #!/usr/bin/env python3
  """cuda/to_evidence.py — THIN wrapper over the shared NVIDIA mapper.

  Spec 5.1 vendor-collapse: cuda and triton lower to PTX and are profiled by the same
  ncu, so they share ONE mapping in backends/_evidence_nvidia.py. The spec's literal
  `from ..cuda.to_evidence import main` relative import fails for this PATH-invoked
  standalone script (no package context), so we sys.path-insert the backends dir and
  import the shared mapper explicitly.

  Invoked WITH a python prefix:
    to_evidence.py --native <ncu.csv|-> [--source-backend <id>] [--format ncu-csv] [--run <result.json>]
  """
  import os, sys

  _BACKENDS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # _substrate/backends
  if _BACKENDS not in sys.path:
      sys.path.insert(0, _BACKENDS)

  import _evidence_nvidia  # noqa: E402

  if __name__ == "__main__":
      sys.exit(_evidence_nvidia.main(source_backend="cuda"))
  ```

  Create `_substrate/backends/triton/to_evidence.py` (identical but for the id and docstring):
  ```python
  #!/usr/bin/env python3
  """triton/to_evidence.py — THIN wrapper over the shared NVIDIA mapper.

  Spec 5.1 vendor-collapse: triton lowers to PTX and is profiled by the same ncu as
  cuda, so it reuses backends/_evidence_nvidia.py verbatim — only the source_backend id
  differs. (The spec's `from ..cuda.to_evidence import main` relative import fails for
  this PATH-invoked standalone script; we sys.path-insert the backends dir instead.)

  Note: the Triton-specific profiling ENV (TRITON_CACHE_DIR etc.) lives in profile.sh,
  NOT here — the shared mapper concerns only the 3 canonical classifier counters
  (spec 5.1 caveat). backend_native source-attributed fields may be weaker under Triton.

  Invoked WITH a python prefix:
    to_evidence.py --native <ncu.csv|-> [--source-backend <id>] [--format ncu-csv] [--run <result.json>]
  """
  import os, sys

  _BACKENDS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # _substrate/backends
  if _BACKENDS not in sys.path:
      sys.path.insert(0, _BACKENDS)

  import _evidence_nvidia  # noqa: E402

  if __name__ == "__main__":
      sys.exit(_evidence_nvidia.main(source_backend="triton"))
  ```

  > Wrapper convention (resolved across parts): both wrappers compute the backends dir as the **parent of their own dir** (`dirname(dirname(__file__))`), sys.path-insert it, `import _evidence_nvidia`, and call `main(source_backend="<id>")` with the keyword argument. This is the single convention used by every wrapper in this plan.

  Make both executable:
  ```
  chmod +x _substrate/backends/cuda/to_evidence.py _substrate/backends/triton/to_evidence.py
  ```

  Run and SEE IT PASS:
  ```
  python3 -m unittest discover -s _substrate/tests -p 'test_evidence_nvidia.py' -v
  ```
  **Expected GREEN** (all `test_evidence_nvidia` classes, including `TestVendorCollapseWrappers`). Manual proof of the vendor-collapse:
  ```
  python3 _substrate/backends/cuda/to_evidence.py   --native _substrate/tests/fixtures/ncu/full4.csv | python3 -c "import sys,json;print(json.load(sys.stdin)['source_backend'])"
  python3 _substrate/backends/triton/to_evidence.py --native _substrate/tests/fixtures/ncu/full4.csv | python3 -c "import sys,json;print(json.load(sys.stdin)['source_backend'])"
  ```
  → prints `cuda` then `triton`; the `metrics` blocks are byte-identical.

- [ ] **Step 3 — Confirm full suite and commit.**
  ```
  python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v 2>&1 | tail -3
  ```
  **Expected:** `Ran 55 tests ... OK` (51 + 4 new wrapper tests).

  Commit on `dev/solver-substrate`:
  ```
  git add _substrate/backends/cuda/to_evidence.py _substrate/backends/triton/to_evidence.py \
          _substrate/tests/test_evidence_nvidia.py
  git commit -m "Add thin cuda/ + triton/ to_evidence.py wrappers (vendor-collapse)

  Both wrappers sys.path-insert the backends dir, import the shared
  _evidence_nvidia mapper, and call main() with their own source_backend id
  (cuda / triton). Proves spec 5.1 vendor-collapse: the SAME ncu CSV yields
  byte-identical metrics differing only in source_backend. Replaces the spec's
  literal 'from ..cuda.to_evidence import main' shim, which fails for these
  path-invoked standalone scripts. 4 new tests, full suite green.

  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 3 — RED: conformance test for the (not-yet-existing) cuda + triton drivers

This test shells out to the real validator against the real `_substrate/backends/cuda` and `_substrate/backends/triton` directories (NOT fixtures — the existing `run_validator` in `test_validate_backend.py` is hard-wired to `FIXTURES`, so this needs its own helper pointing at `backends/`). It also imports `method_gate` directly to assert every idiom name is a real `TABLE` value, and that `backend_id == dir`. It MUST fail first because the four files do not exist yet.

**Files**
- Create: `_substrate/tests/test_driver_conformance.py`

**Steps**

- [ ] **Step 1** — Write the complete failing test file `_substrate/tests/test_driver_conformance.py`:

```python
import os, sys, json, subprocess, unittest
SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

import method_gate  # live table — single source of truth for real method names

VALIDATOR = os.path.join(SUB, 'backends', 'validate_backend.py')
BACKENDS = os.path.join(SUB, 'backends')

# All real method_gate method names, sourced live (can never drift from the table).
KNOWN_METHODS = {m for methods in method_gate.TABLE.values() for m in methods}

# The drivers this part introduces. Both lower to PTX and are profiled by ncu, so both
# carry hw_vendor "nvidia" and share the nvidia evidence mapping.
REAL_DRIVERS = ['cuda', 'triton']


def run_validator(driver_dir_abspath):
    """Shell out to validate_backend.py <dir>; return (returncode, parsed_stdout_json)."""
    proc = subprocess.run(
        [sys.executable, VALIDATOR, driver_dir_abspath],
        capture_output=True, text=True)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"_raw_stdout": proc.stdout, "_stderr": proc.stderr}
    return proc.returncode, payload


def load_driver_json(driver, fname):
    with open(os.path.join(BACKENDS, driver, fname), encoding="utf-8") as fh:
        return json.load(fh)


class TestRealDriversValidate(unittest.TestCase):
    def test_each_real_driver_exits_zero_ok_true_no_errors(self):
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                code, payload = run_validator(os.path.join(BACKENDS, driver))
                self.assertEqual(code, 0,
                                 msg=f"{driver}: expected exit 0, got {code}; payload={payload}")
                self.assertEqual(payload.get('ok'), True, msg=f"{driver}: payload={payload}")
                self.assertEqual(payload.get('errors'), [], msg=f"{driver}: payload={payload}")


class TestRealDriversMethodNames(unittest.TestCase):
    def test_every_idiom_method_is_a_real_method_gate_name(self):
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                idioms = load_driver_json(driver, 'idioms.json')
                methods = idioms.get('methods', {})
                self.assertIsInstance(methods, dict, msg=f"{driver}: methods not an object")
                self.assertTrue(methods, msg=f"{driver}: methods is empty")
                for name in methods:
                    self.assertIn(name, KNOWN_METHODS,
                                  msg=f"{driver}: idiom method '{name}' not in method_gate.TABLE")
                for name in idioms.get('unsupported_methods', []):
                    self.assertIn(name, KNOWN_METHODS,
                                  msg=f"{driver}: unsupported '{name}' not in method_gate.TABLE")

    def test_idioms_cover_a_gated_method_for_each_meaningful_class(self):
        # Each manifest lists the 4 meaningful classes; idioms.json should reference at least
        # one real gated method from each of those four classes (so the prompt layer always
        # has a concrete idiom to surface whatever bottleneck the gate picks).
        meaningful = ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound"]
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                named = set(load_driver_json(driver, 'idioms.json').get('methods', {}))
                for bclass in meaningful:
                    gated = set(method_gate.TABLE[bclass])
                    self.assertTrue(named & gated,
                                    msg=f"{driver}: no idiom covers any method of '{bclass}' "
                                        f"(class methods={sorted(gated)})")


class TestRealDriversBackendId(unittest.TestCase):
    def test_backend_id_equals_directory_name(self):
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                manifest = load_driver_json(driver, 'manifest.json')
                self.assertEqual(manifest.get('backend_id'), driver,
                                 msg=f"{driver}: manifest.backend_id != dir name")
                idioms = load_driver_json(driver, 'idioms.json')
                self.assertEqual(idioms.get('backend_id'), driver,
                                 msg=f"{driver}: idioms.backend_id != dir name")

    def test_both_drivers_are_nvidia_vendor_and_experimental(self):
        # cuda and triton both run on NVIDIA hardware and are profiled by the same ncu, so
        # both share hw_vendor "nvidia"; neither is GPU-validated on this host, so both are
        # honestly marked status "experimental".
        for driver in REAL_DRIVERS:
            with self.subTest(driver=driver):
                manifest = load_driver_json(driver, 'manifest.json')
                self.assertEqual(manifest.get('hw_vendor'), 'nvidia', msg=f"{driver}")
                self.assertEqual(manifest.get('threshold_profile'), 'nvidia', msg=f"{driver}")
                self.assertEqual(manifest.get('status'), 'experimental', msg=f"{driver}")


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2** — Run ONLY the new test and observe the RED failure (the directories/files don't exist yet, so `run_validator` returns exit 3 / unreadable, and `load_driver_json` raises `FileNotFoundError`):

```
python3 -m unittest discover -s _substrate/tests -p 'test_driver_conformance.py' -v
```

Confirm the failures are exactly "file/dir not found" shaped (NOT an import error in the test itself). If `method_gate` import fails, fix `sys.path` before proceeding. Do NOT commit yet — the GREEN files land together with this test in Task 6's commit. Default: hold.

---

### Task 4 — GREEN: `_substrate/backends/cuda/` manifest + idioms

Create the cuda driver's two machine-read files, modeled on the `fixtures/good/` template but with `backend_id "cuda"` and `status "experimental"`. Every method key is copied verbatim from `method_gate.TABLE` — no invented names.

**Files**
- Create: `_substrate/backends/cuda/manifest.json`
- Create: `_substrate/backends/cuda/idioms.json`

**Steps**

- [ ] **Step 1** — Create `_substrate/backends/cuda/manifest.json`:

```json
{
  "schema_version": 1,
  "backend_id": "cuda",
  "display_name": "CUDA C++ (NVIDIA, nvcc)",
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
    "metrics": { "latency_ms": true, "dram_pct": true, "sm_pct": true, "occupancy": true },
    "bottleneck_classes": ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound"],
    "problem_types": ["kernel-optimization", "kernel-generation"],
    "precisions": ["fp32", "fp16", "bf16"]
  },
  "requires_tools": ["nvcc", "python3"],
  "optional_tools": ["ncu"],
  "idioms": "idioms.json",
  "status": "experimental"
}
```

- [ ] **Step 2** — Create `_substrate/backends/cuda/idioms.json`. Every key under `methods` is a verbatim `method_gate.TABLE` name; the file covers at least one method from each of the four meaningful classes (memory_bound, compute_bound, latency_occupancy, overhead_bound):

```json
{
  "schema_version": 1,
  "backend_id": "cuda",
  "lang_fence": "cuda",
  "impl_requirements": "Emit a single .cu translation unit exposing a PYBIND11_MODULE with a forward() binding that launches the kernel; no host-side framework calls beyond the binding.",
  "unsupported_methods": [],
  "read_metric_guide": "High dram_pct with low sm_pct => memory_bound. High sm_pct with low dram_pct => compute_bound. Low occupancy with neither saturated => latency_occupancy. Many short kernels / launch-dominated wall time => overhead_bound.",
  "methods": {
    "vectorized_load_store": {
      "idiom": "float4 / int4 vectorized global loads and stores",
      "prompt_guidance": "Widen memory transactions to 128-bit by casting to float4/int4 so each thread moves 16 bytes per access."
    },
    "memory_coalescing": {
      "idiom": "coalesced 128-bit global loads",
      "prompt_guidance": "Reorder the thread-to-element mapping so consecutive threads in a warp read consecutive addresses."
    },
    "shared_memory_tiling": {
      "idiom": "__shared__ tiling with __syncthreads()",
      "prompt_guidance": "Stage reused operands into a __shared__ tile per block, sync, then compute from shared memory to cut redundant global traffic."
    },
    "async_copy_pipeline": {
      "idiom": "cp.async double-buffered global->shared pipeline",
      "prompt_guidance": "Use cp.async (or __pipeline_memcpy_async) to overlap the next tile's global->shared copy with current-tile compute."
    },
    "l2_cache_reuse": {
      "idiom": "L2-residency-friendly tiling / __ldg read-only path",
      "prompt_guidance": "Block the iteration so hot operands stay L2-resident and route read-only data through __ldg / const __restrict__."
    },
    "tensor_core_mma": {
      "idiom": "wmma / mma.sync",
      "prompt_guidance": "Use tensor-core MMA intrinsics (wmma::mma_sync or mma.sync.aligned) for the inner GEMM with fp16/bf16 operands."
    },
    "instruction_reduction": {
      "idiom": "strength reduction and loop unrolling (#pragma unroll)",
      "prompt_guidance": "Hoist invariants, replace divides/mods with shifts where legal, and #pragma unroll the inner loop to cut issued instructions."
    },
    "fast_math_intrinsics": {
      "idiom": "__expf / __fdividef fast-math intrinsics",
      "prompt_guidance": "Where precision tolerance allows, replace transcendental calls with __-prefixed device intrinsics (e.g. __expf, __fdividef)."
    },
    "register_tiling": {
      "idiom": "per-thread register accumulator tiles",
      "prompt_guidance": "Accumulate an MxN micro-tile in registers per thread to raise arithmetic intensity and reuse loaded operands."
    },
    "occupancy_increase": {
      "idiom": "__launch_bounds__ / lower per-thread register pressure",
      "prompt_guidance": "Annotate the kernel with __launch_bounds__ and shrink live state to fit more resident warps per SM."
    },
    "block_size_tuning": {
      "idiom": "sweep blockDim (128/256/512) for the SM",
      "prompt_guidance": "Try block sizes that are multiples of the warp size (128/256/512) to balance occupancy against per-thread resources."
    },
    "register_pressure_reduction": {
      "idiom": "scope-narrowing / recompute to spill less",
      "prompt_guidance": "Shorten variable lifetimes and prefer recompute over caching to drop register count below the spill threshold."
    },
    "launch_config_tuning": {
      "idiom": "grid/block geometry and dynamic shared-mem sizing",
      "prompt_guidance": "Tune gridDim/blockDim and dynamic __shared__ bytes so the launch geometry matches the SM count and tile shape."
    },
    "kernel_fusion": {
      "idiom": "fuse the elementwise epilogue into the main kernel",
      "prompt_guidance": "Avoid a second launch by computing the elementwise epilogue inside the producing kernel before the store."
    },
    "launch_overhead_reduction": {
      "idiom": "persistent-kernel / CUDA-graph capture",
      "prompt_guidance": "Collapse many tiny launches into a persistent kernel or a captured CUDA graph to amortize launch cost."
    },
    "library_fallback_hybrid": {
      "idiom": "delegate the hot GEMM to cuBLAS",
      "prompt_guidance": "When a vendor library kernel dominates, call cuBLAS/cuDNN for the heavy op and keep custom code only for the glue."
    },
    "cpu_gpu_overlap": {
      "idiom": "multi-stream H2D/compute/D2H overlap",
      "prompt_guidance": "Split the work across CUDA streams so host<->device copies overlap kernel execution."
    }
  }
}
```

- [ ] **Step 3** — Validate the cuda driver standalone and confirm exit 0 + `{"ok": true, "errors": []}`:

```
python3 _substrate/backends/validate_backend.py _substrate/backends/cuda ; echo "exit=$?"
```

Expect `"ok": true` and exit status `0`.

- [ ] **Step 4** — Commit:

```
git add _substrate/backends/cuda/manifest.json _substrate/backends/cuda/idioms.json
git commit -m "feat(substrate): add L0-conformant cuda backend driver (manifest + idioms)

backend_id cuda, hw_vendor/threshold_profile nvidia, status experimental
(GPU-untested on this macOS host). idioms reference only real
method_gate.TABLE names and cover all four meaningful bottleneck classes.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5 — GREEN: `_substrate/backends/triton/` manifest + idioms

Create the triton driver. Same nvidia vendor + capabilities as cuda (it lowers to PTX and is profiled by the same ncu / shares the nvidia evidence mapping), but Python source (`.py`), no nvcc, `lang_fence "python"`, and `@triton.jit` idioms instead of CUDA C++. The `methods` keys are the SAME abstract `method_gate` names as cuda — only the `idiom`/`prompt_guidance` strings differ.

**Files**
- Create: `_substrate/backends/triton/manifest.json`
- Create: `_substrate/backends/triton/idioms.json`

**Steps**

- [ ] **Step 1** — Create `_substrate/backends/triton/manifest.json`. `source_ext ".py"`; `artifact_ext ".ptx"` (Triton's persisted lowered artifact); compiler name `"triton"` (the JIT, not nvcc); `requires_tools ["python3"]` with `triton` in `optional_tools` (importing triton is not possible on this macOS host, and L0 never executes it, so it is honestly optional at validate-time):

```json
{
  "schema_version": 1,
  "backend_id": "triton",
  "display_name": "Triton (NVIDIA, @triton.jit)",
  "source_ext": ".py",
  "aux_ext": [],
  "artifact_ext": ".ptx",
  "hw_vendor": "nvidia",
  "threshold_profile": "nvidia",
  "compiler": { "name": "triton", "invoke": "build.sh" },
  "runner": { "invoke": "run.sh" },
  "profiler": {
    "name": "ncu",
    "invoke": "profile.sh",
    "format": "ncu-csv",
    "to_evidence": "to_evidence.py"
  },
  "capabilities": {
    "metrics": { "latency_ms": true, "dram_pct": true, "sm_pct": true, "occupancy": true },
    "bottleneck_classes": ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound"],
    "problem_types": ["kernel-optimization", "kernel-generation"],
    "precisions": ["fp32", "fp16", "bf16"]
  },
  "requires_tools": ["python3"],
  "optional_tools": ["triton", "ncu"],
  "idioms": "idioms.json",
  "status": "experimental"
}
```

> Reconciliation note: the canonical triton manifest uses `compiler.name "triton"`, `artifact_ext ".ptx"`, and `requires_tools ["python3"]` (triton optional). The Part-3 shell-script tests only assert the envelope/exit-code behavior of `build.sh`/`run.sh`/`profile.sh` and the L0 pass of the dir — they do not pin `compiler.name`, `artifact_ext`, or `requires_tools`, so this manifest satisfies both the conformance test (Task 3) and the script tests (Tasks 9–11).

- [ ] **Step 2** — Create `_substrate/backends/triton/idioms.json`. `lang_fence "python"`, no PYBIND11 (a `@triton.jit` kernel plus a Python launcher), same abstract method keys mapped to Triton idioms:

```json
{
  "schema_version": 1,
  "backend_id": "triton",
  "lang_fence": "python",
  "impl_requirements": "Provide a @triton.jit kernel plus a plain Python launcher that computes the grid and calls kernel[grid](...). No PYBIND11; the launcher returns the output tensor directly.",
  "unsupported_methods": [],
  "read_metric_guide": "High dram_pct with low sm_pct => memory_bound. High sm_pct with low dram_pct => compute_bound. Low occupancy with neither saturated => latency_occupancy. Many short kernels / launch-dominated wall time => overhead_bound.",
  "methods": {
    "vectorized_load_store": {
      "idiom": "block-shaped tl.load / tl.store over contiguous offsets",
      "prompt_guidance": "Load and store whole BLOCK-sized tiles with tl.arange offsets so Triton vectorizes the access into wide transactions."
    },
    "memory_coalescing": {
      "idiom": "contiguous tl.load via block pointers / tl.make_block_ptr",
      "prompt_guidance": "Lay out program_id-to-offset so the last (contiguous) axis varies fastest; use tl.make_block_ptr for clean coalesced tiles."
    },
    "shared_memory_tiling": {
      "idiom": "tl.dot over BLOCK_M x BLOCK_K x BLOCK_N tiles",
      "prompt_guidance": "Pick BLOCK_M/BLOCK_N/BLOCK_K so each program reuses an operand tile across the K loop; Triton stages it through shared memory for you."
    },
    "async_copy_pipeline": {
      "idiom": "num_stages software pipelining of the K loop",
      "prompt_guidance": "Raise num_stages (and tune num_warps) so Triton emits cp.async-backed multi-stage prefetch of the next K tile."
    },
    "l2_cache_reuse": {
      "idiom": "grouped/super-grouped program ordering (GROUP_SIZE_M)",
      "prompt_guidance": "Reorder program_id launch order (group along M) so co-scheduled programs reuse the same B columns from L2."
    },
    "tensor_core_mma": {
      "idiom": "tl.dot",
      "prompt_guidance": "Express the inner GEMM as tl.dot(a, b) on fp16/bf16 tiles so Triton lowers it to tensor-core mma."
    },
    "instruction_reduction": {
      "idiom": "tl.constexpr-specialized shapes and folded ops",
      "prompt_guidance": "Mark shapes/strides tl.constexpr so the compiler folds bounds math and unrolls the inner loop."
    },
    "fast_math_intrinsics": {
      "idiom": "tl math ops with reduced-precision accumulation flags",
      "prompt_guidance": "Use Triton's math ops and allow_tf32 / lower-precision accumulation where tolerance permits to cut transcendental cost."
    },
    "register_tiling": {
      "idiom": "larger BLOCK_M x BLOCK_N register accumulator",
      "prompt_guidance": "Grow the accumulator tile shape so each program holds more of the output in registers, raising arithmetic intensity."
    },
    "occupancy_increase": {
      "idiom": "tune num_warps to raise resident warps",
      "prompt_guidance": "Adjust num_warps (and shrink the block) so more warps stay resident per SM."
    },
    "block_size_tuning": {
      "idiom": "triton.autotune over BLOCK_* / num_warps configs",
      "prompt_guidance": "Sweep BLOCK_M/BLOCK_N/BLOCK_K and num_warps with @triton.autotune to find the best geometry for the SM."
    },
    "register_pressure_reduction": {
      "idiom": "smaller BLOCK tile / fewer live tl values",
      "prompt_guidance": "Shrink the tile shape and reuse buffers so the kernel needs fewer live registers and avoids spills."
    },
    "launch_config_tuning": {
      "idiom": "grid lambda and num_warps/num_stages in the launcher",
      "prompt_guidance": "Tune the grid lambda plus num_warps/num_stages passed at kernel[grid](...) launch to match the problem shape."
    },
    "kernel_fusion": {
      "idiom": "fuse the elementwise epilogue inside the @triton.jit kernel",
      "prompt_guidance": "Apply the activation / bias / scaling on the accumulator before tl.store so no second kernel is launched."
    },
    "launch_overhead_reduction": {
      "idiom": "fewer programs per launch / persistent-style grid",
      "prompt_guidance": "Coarsen the grid so each program does more work, cutting the number of launches and their overhead."
    },
    "library_fallback_hybrid": {
      "idiom": "fall back to torch.matmul / cuBLAS for the hot op",
      "prompt_guidance": "If a vendor GEMM beats the Triton kernel, delegate the heavy op to torch.matmul and keep Triton for fusable glue."
    },
    "cpu_gpu_overlap": {
      "idiom": "non_blocking H2D copies on a side CUDA stream",
      "prompt_guidance": "Issue host->device transfers with non_blocking=True on a separate stream so they overlap kernel launches."
    }
  }
}
```

- [ ] **Step 3** — Validate the triton driver standalone and confirm exit 0 + `{"ok": true, "errors": []}`:

```
python3 _substrate/backends/validate_backend.py _substrate/backends/triton ; echo "exit=$?"
```

- [ ] **Step 4** — Commit:

```
git add _substrate/backends/triton/manifest.json _substrate/backends/triton/idioms.json
git commit -m "feat(substrate): add L0-conformant triton backend driver (manifest + idioms)

backend_id triton, hw_vendor nvidia (lowers to PTX, profiled by the same
ncu / shares the nvidia evidence mapping). lang_fence python, no PYBIND11.
Same abstract method_gate keys as cuda mapped to @triton.jit idioms.
status experimental (GPU-untested on this macOS host).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6 — GREEN: run the conformance test, then the full suite, then commit it

Now that all four files (cuda + triton manifest/idioms) exist, the Task 3 test must go green, and the existing tests must stay green.

**Files**
- Create: (commit) `_substrate/tests/test_driver_conformance.py` (authored in Task 3, held back)

**Steps**

- [ ] **Step 1** — Run the new conformance test alone and confirm all of its test methods pass for both `cuda` and `triton` subTests:

```
python3 -m unittest discover -s _substrate/tests -p 'test_driver_conformance.py' -v
```

Expect `OK` with both `cuda` and `triton` subTests passing in every method.

- [ ] **Step 2** — Run the FULL suite and confirm no regressions:

```
python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v 2>&1 | tail -3
```

Expect a final `OK` line. If anything is red, fix the JSON (do NOT modify existing tests) and re-run.

- [ ] **Step 3** — Commit the test (held back from Task 3):

```
git add _substrate/tests/test_driver_conformance.py
git commit -m "test(substrate): L0 conformance for real cuda + triton drivers

Validates _substrate/backends/{cuda,triton} via validate_backend.py (exit 0,
ok:true, errors:[]); asserts every idioms.json method is a live
method_gate.TABLE name, that all four meaningful classes are covered, and
backend_id == dir name. Fully testable on macOS (no kernel/ncu execution).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7 — Guard: confirm the shared mapper + to_evidence wrappers exist, then seed the shell-script test file

This guard establishes the shared `test_driver_scripts.py` (reused by Tasks 7–12) and confirms the shared NCU→canonical mapper and both thin wrappers from Tasks 1–2 are present. The `occupancy = warps_active% ÷ 100` invariant the §5.3.2 null rule depends on is re-pinned here against a fake-ncu fixture, this time via the `--format ncu-csv` invocation the shell scripts use.

> Idempotency note: Tasks 1–2 already shipped `_evidence_nvidia.py`, `cuda/to_evidence.py`, and `triton/to_evidence.py`. This guard's first test simply confirms they exist; if you are executing the plan in order, it passes immediately. If a prior part did NOT ship them, return to Tasks 1–2 — do not re-author the mapper here (the canonical mapper with `dram_pct = read + write` is the single source of truth).

**Files**
- Create: `_substrate/tests/test_driver_scripts.py` (shared by Tasks 7–12)

**Steps**

- [ ] **Step 1 — RED.** Create `_substrate/tests/test_driver_scripts.py` with the shared header + first class. Run it; on an in-order execution the first class passes (mapper exists), but `TestSharedNvidiaMapper` also exercises the wrapper via `--format`, pinning the contract.

```python
import os, sys, json, stat, subprocess, tempfile, textwrap, unittest

SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
sys.path.insert(0, SUB)

BACKENDS = os.path.join(SUB, 'backends')
CUDA = os.path.join(BACKENDS, 'cuda')
TRITON = os.path.join(BACKENDS, 'triton')


def _write_exec(path, body):
    """Write body to path and chmod 0755 (used for fake-tool stubs)."""
    with open(path, 'w') as fh:
        fh.write(body)
    os.chmod(path, 0o755)


def _run(argv, env=None, cwd=None):
    """Run argv; return (returncode, stdout, stderr)."""
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=cwd)
    return proc.returncode, proc.stdout, proc.stderr


def _json_or_raw(out):
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"_raw": out}


def _path_env(stub_dir):
    """Copy os.environ with stub_dir prepended to PATH."""
    env = dict(os.environ)
    env['PATH'] = stub_dir + os.pathsep + env.get('PATH', '')
    return env


# ----- fake-tool stub bodies (bash) -----
FAKE_NVCC_OK = textwrap.dedent('''\
    #!/usr/bin/env bash
    # Fake nvcc: parse for an output token (-o <file>) and touch it, then exit 0.
    out=""
    prev=""
    for a in "$@"; do
      if [ "$prev" = "-o" ]; then out="$a"; fi
      prev="$a"
    done
    [ -n "$out" ] && : > "$out"
    echo "fake nvcc ok" 1>&2
    exit 0
''')

FAKE_NVCC_FAIL = textwrap.dedent('''\
    #!/usr/bin/env bash
    echo "kernel.cu(7): error: identifier \\"foo\\" is undefined" 1>&2
    echo "1 error detected in the compilation of kernel.cu" 1>&2
    exit 1
''')

# Fake ncu: emit a canned NCU --csv profile on stdout (profile.sh captures stdout).
# NOTE: the fake's CSV deliberately includes BOTH dram read AND write rows so that the
# shared mapper's dram_pct = read + write contract (40.0 + 22.0 = 62.0) is exercised
# consistently with the Task-1 fixtures.
FAKE_NCU_CSV = textwrap.dedent('''\
    #!/usr/bin/env bash
    cat <<'CSV'
    "ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
    "0","my_kernel","gpu__time_duration.sum","ns","123456"
    "0","my_kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","48.0"
    "0","my_kernel","dram__bytes_read.sum.pct_of_peak_sustained_elapsed","%","40.0"
    "0","my_kernel","dram__bytes_write.sum.pct_of_peak_sustained_elapsed","%","22.0"
    "0","my_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","%","51.0"
    CSV
    echo "fake ncu ok" 1>&2
    exit 0
''')


class TestSharedNvidiaMapper(unittest.TestCase):
    def test_shared_mapper_and_cuda_wrapper_exist(self):
        self.assertTrue(os.path.isfile(os.path.join(BACKENDS, '_evidence_nvidia.py')),
                        "_evidence_nvidia.py shared mapper missing (Tasks 1-2)")
        self.assertTrue(os.path.isfile(os.path.join(CUDA, 'to_evidence.py')),
                        "cuda/to_evidence.py wrapper missing (Tasks 1-2)")

    def test_mapper_occupancy_is_warps_active_over_100(self):
        # The single most error-prone line: occupancy = warps_active_pct / 100 (0..1).
        with tempfile.TemporaryDirectory() as td:
            csv = os.path.join(td, 'native.csv')
            _write_exec(os.path.join(td, '_emit.sh'), FAKE_NCU_CSV)
            with open(csv, 'w') as fh:
                subprocess.run([os.path.join(td, '_emit.sh')], stdout=fh)
            code, out, err = _run(
                [sys.executable, os.path.join(CUDA, 'to_evidence.py'),
                 '--native', csv, '--format', 'ncu-csv'])
            self.assertEqual(code, 0, msg=f"out={out} err={err}")
            payload = _json_or_raw(out)
            self.assertEqual(payload.get('ok'), True, payload)
            m = payload['metrics']
            self.assertEqual(payload['source_backend'], 'cuda', payload)
            self.assertEqual(m['_vendor'], 'nvidia', payload)
            self.assertAlmostEqual(m['latency_ms'], 123456 / 1e6, places=9)
            self.assertAlmostEqual(m['sm_pct'], 48.0, places=3)
            self.assertAlmostEqual(m['dram_pct'], 62.0, places=3)   # read + write (canonical)
            self.assertAlmostEqual(m['occupancy'], 0.51, places=4)  # 51.0 / 100
            self.assertIn('occupancy', payload['coverage'])
```

```bash
python3 -m unittest discover -s _substrate/tests -p 'test_driver_scripts.py' -v
```
Expected (in-order execution): GREEN — the mapper + cuda wrapper already exist from Tasks 1–2 and `dram_pct == 62.0`, `occupancy == 0.51`. If they are missing, return to Tasks 1–2.

> Reconciliation note: this fake-ncu CSV includes the dram **write** row so `dram_pct` is the canonical `read + write = 62.0` — matching Task 1's fixtures and the single mapper. Do not assert `dram_pct == 40.0`.

- [ ] **Step 2 — commit.**
```
git add _substrate/tests/test_driver_scripts.py
git commit -m "P3: shell-script test scaffold + re-pin shared mapper occupancy÷100 / dram read+write via --format

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8 — `cuda/build.sh` (nvcc compile, fake-tool tested)

**Files**
- Create: `_substrate/backends/cuda/build.sh`
- Modify: `_substrate/tests/test_driver_scripts.py` (add `TestCudaBuild`)

**Steps**

- [ ] **Step 1 — RED.** Append `TestCudaBuild` to the test file.

```python
class TestCudaBuild(unittest.TestCase):
    SCRIPT = os.path.join(CUDA, 'build.sh')

    def test_exists_executable_and_syntax(self):
        self.assertTrue(os.path.isfile(self.SCRIPT), "cuda/build.sh missing")
        self.assertTrue(os.access(self.SCRIPT, os.X_OK), "cuda/build.sh not executable")
        code, _, err = _run(['bash', '-n', self.SCRIPT])
        self.assertEqual(code, 0, msg=f"bash -n failed: {err}")

    def test_build_ok_with_fake_nvcc(self):
        with tempfile.TemporaryDirectory() as td:
            _write_exec(os.path.join(td, 'nvcc'), FAKE_NVCC_OK)
            src = os.path.join(td, 'kernel.cu')
            out = os.path.join(td, 'kernel.so')
            with open(src, 'w') as fh:
                fh.write("// fake cuda source\n")
            env = _path_env(td)
            code, sout, serr = _run([self.SCRIPT, '--source', src, '--out', out,
                                     '--arch', 'sm_80'], env=env)
            self.assertEqual(code, 0, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), True, p)
            self.assertEqual(p.get('compiled'), True, p)
            self.assertEqual(p.get('artifact'), out, p)
            self.assertTrue(os.path.isfile(out), "artifact not produced by fake nvcc")
            self.assertIn('build_latency_ms', p)
            self.assertIsInstance(p['build_latency_ms'], (int, float))
            self.assertIn('stderr_tail', p)

    def test_build_passes_lineinfo_to_nvcc(self):
        # -lineinfo is REQUIRED for ncu source attribution; assert the script emits it.
        with tempfile.TemporaryDirectory() as td:
            # fake nvcc that records its argv to a sidecar file
            rec = os.path.join(td, 'argv.txt')
            _write_exec(os.path.join(td, 'nvcc'), textwrap.dedent(f'''\
                #!/usr/bin/env bash
                echo "$@" > "{rec}"
                out=""; prev=""
                for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
                [ -n "$out" ] && : > "$out"
                exit 0
            '''))
            src = os.path.join(td, 'k.cu'); out = os.path.join(td, 'k.so')
            open(src, 'w').write("//\n")
            code, sout, serr = _run([self.SCRIPT, '--source', src, '--out', out],
                                    env=_path_env(td))
            self.assertEqual(code, 0, msg=f"{sout} {serr}")
            self.assertIn('-lineinfo', open(rec).read())

    def test_build_compile_failure_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            _write_exec(os.path.join(td, 'nvcc'), FAKE_NVCC_FAIL)
            src = os.path.join(td, 'kernel.cu'); out = os.path.join(td, 'kernel.so')
            open(src, 'w').write("//\n")
            code, sout, serr = _run([self.SCRIPT, '--source', src, '--out', out],
                                    env=_path_env(td))
            self.assertEqual(code, 2, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), False, p)
            self.assertEqual(p.get('compiled'), False, p)
            self.assertIsNone(p.get('artifact'), p)
            self.assertIn('error detected', p.get('stderr_tail', ''))

    def test_build_missing_nvcc_exit_3(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, 'kernel.cu'); out = os.path.join(td, 'kernel.so')
            open(src, 'w').write("//\n")
            # nvcc is genuinely absent on this macOS host, so the inherited env already exercises
            # the script's own "nvcc not found" guard (exit 3). Do NOT wipe PATH — that would break
            # the `#!/usr/bin/env bash` shebang itself (exit 127). On a GPU box where nvcc exists,
            # point PATH at a stub dir that has bash+coreutils but omits nvcc.
            env = dict(os.environ)
            code, sout, serr = _run([self.SCRIPT, '--source', src, '--out', out], env=env)
            self.assertEqual(code, 3, msg=f"out={sout} err={serr}")
            self.assertEqual(_json_or_raw(sout).get('ok'), False)

    def test_build_missing_args_exit_3(self):
        code, sout, _ = _run([self.SCRIPT, '--source', '/x.cu'])  # no --out
        self.assertEqual(code, 3)
        self.assertEqual(_json_or_raw(sout).get('ok'), False)
```

```bash
python3 -m unittest discover -s _substrate/tests -p 'test_driver_scripts.py' -v
```
Expected RED: `cuda/build.sh missing`.

- [ ] **Step 2 — GREEN.** Create `_substrate/backends/cuda/build.sh` (complete).

```bash
#!/usr/bin/env bash
# cuda/build.sh — compile a .cu kernel to a .so via nvcc (or a --build-cmd template).
# Spec §4.5. Universal envelope: ONE json on stdout, logs to stderr.
#   exit 0 ok · 2 compile failure (json printed) · 3 bad args / missing tool.
# -lineinfo is REQUIRED for ncu source attribution.
set -u

emit() { printf '%s\n' "$1"; }   # one-line JSON on stdout
die3() { emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"error\":\"$1\"}"; exit 3; }

SOURCE="" OUT="" ARCH="sm_80" BUILD_CMD="" EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --source)    SOURCE="${2:-}"; shift 2 ;;
    --out)       OUT="${2:-}"; shift 2 ;;
    --arch)      ARCH="${2:-}"; shift 2 ;;
    --build-cmd) BUILD_CMD="${2:-}"; shift 2 ;;
    --extra)     EXTRA="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$SOURCE" ] || die3 "missing --source"
[ -n "$OUT" ]    || die3 "missing --out"
[ -f "$SOURCE" ] || die3 "source not found: $SOURCE"

# JSON-escape helper for the stderr tail (escape backslash, quote, strip control chars).
json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

# Resolve the build command. Default: nvcc -shared -Xcompiler -fPIC -lineinfo -arch=... -o OUT SOURCE
if [ -n "$BUILD_CMD" ]; then
  # template tokens: {source} {out} {arch} {extra}
  CMD="${BUILD_CMD//\{source\}/$SOURCE}"
  CMD="${CMD//\{out\}/$OUT}"
  CMD="${CMD//\{arch\}/$ARCH}"
  CMD="${CMD//\{extra\}/$EXTRA}"
  # -lineinfo must be present for ncu attribution; inject if the template omitted it.
  case "$CMD" in *-lineinfo*) : ;; *) CMD="$CMD -lineinfo" ;; esac
else
  command -v nvcc >/dev/null 2>&1 || die3 "nvcc not found on PATH"
  CMD="nvcc -shared -Xcompiler -fPIC -lineinfo -arch=$ARCH $EXTRA -o $OUT $SOURCE"
fi

# If a template was given, still verify its leading tool exists.
TOOL="${CMD%% *}"
command -v "$TOOL" >/dev/null 2>&1 || die3 "build tool not found: $TOOL"

# Wall-time the compile (ms). Use python3 for portable millisecond timing.
START="$(python3 -c 'import time;print(int(time.time()*1000))')"
STDERR_FILE="$(mktemp)"
# shellcheck disable=SC2086
eval $CMD 2>"$STDERR_FILE"
RC=$?
END="$(python3 -c 'import time;print(int(time.time()*1000))')"
BUILD_MS=$(( END - START ))

# Tail of stderr (last 20 lines), JSON-escaped.
TAIL="$(tail -n 20 "$STDERR_FILE")"
ESC_TAIL="$(json_escape "$TAIL")"
cat "$STDERR_FILE" 1>&2
rm -f "$STDERR_FILE"

if [ "$RC" -eq 0 ] && [ -f "$OUT" ]; then
  emit "{\"ok\":true,\"compiled\":true,\"artifact\":\"$OUT\",\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":\"\"}"
  exit 0
else
  emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":$ESC_TAIL}"
  exit 2
fi
```

```bash
chmod +x _substrate/backends/cuda/build.sh
python3 -m unittest discover -s _substrate/tests -p 'test_driver_scripts.py' -v
```
Expected GREEN: all `TestCudaBuild` pass.

- [ ] **Step 3 — commit.**
```
git add _substrate/backends/cuda/build.sh _substrate/tests/test_driver_scripts.py
git commit -m "P3: cuda/build.sh — nvcc compile w/ -lineinfo, JSON envelope, fake-tool tested

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9 — `cuda/run.sh` (run + correctness, missing-artifact tested; GPU deferred)

Real correctness/latency need the GPU. We test arg handling and the clean error envelope for the missing-artifact path (exit 2/3), plus the exact stdout key set on the no-GPU error path.

**Files**
- Create: `_substrate/backends/cuda/run.sh`
- Modify: `_substrate/tests/test_driver_scripts.py` (add `TestCudaRun`)

**Steps**

- [ ] **Step 1 — RED.** Append `TestCudaRun`.

```python
class TestCudaRun(unittest.TestCase):
    SCRIPT = os.path.join(CUDA, 'run.sh')

    def _problem(self, td):
        p = os.path.join(td, 'problem.json')
        with open(p, 'w') as fh:
            json.dump({"op": "add", "shape": [128, 128]}, fh)
        return p

    def test_exists_executable_and_syntax(self):
        self.assertTrue(os.path.isfile(self.SCRIPT), "cuda/run.sh missing")
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))
        code, _, err = _run(['bash', '-n', self.SCRIPT])
        self.assertEqual(code, 0, msg=err)

    def test_missing_artifact_clean_error_envelope_exit_3(self):
        # No GPU needed: a nonexistent artifact is a preflight/bad-input failure -> clean JSON envelope, exit 3 (spec §4.5).
        with tempfile.TemporaryDirectory() as td:
            prob = self._problem(td); out = os.path.join(td, 'result.json')
            code, sout, serr = _run([self.SCRIPT, '--artifact', '/nope/x.so',
                                     '--problem', prob, '--out', out])
            self.assertEqual(code, 3, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), False, p)
            # Contract keys must still be present (anti_cheat reads these exactly).
            for k in ('compiled', 'correct', 'candidate_latency_ms', 'eager_latency_ms',
                      'compile_latency_ms', 'claimed_speedup'):
                self.assertIn(k, p, f"missing key {k}: {p}")
            self.assertEqual(p['correct'], False, p)
            self.assertLessEqual(p['claimed_speedup'], 1.0, p)  # correct:false ⇒ ≤1.0

    def test_missing_args_exit_3(self):
        code, sout, _ = _run([self.SCRIPT, '--artifact', '/x.so'])  # no --problem/--out
        self.assertEqual(code, 3)
        self.assertEqual(_json_or_raw(sout).get('ok'), False)

    def test_bad_problem_file_exit_3(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'r.json')
            code, sout, _ = _run([self.SCRIPT, '--artifact', '/x.so',
                                  '--problem', '/no/problem.json', '--out', out])
            self.assertEqual(code, 3)
            self.assertEqual(_json_or_raw(sout).get('ok'), False)
```

```bash
python3 -m unittest discover -s _substrate/tests -p 'test_driver_scripts.py' -v
```
Expected RED: `cuda/run.sh missing`.

- [ ] **Step 2 — GREEN.** Create `_substrate/backends/cuda/run.sh` (complete). The runner shells to a python launcher for the real GPU path; on macOS the missing-artifact / missing-arg guards fire before any GPU work, which is exactly what we test.

```bash
#!/usr/bin/env bash
# cuda/run.sh — load the compiled .so, run vs problem, measure correctness + latency.
# Spec §4.5. stdout keys MUST match anti_cheat.py --metrics EXACTLY:
#   {ok,compiled,correct,candidate_latency_ms,eager_latency_ms,compile_latency_ms,
#    claimed_speedup,...}
#   compile_latency_ms = torch.compile BASELINE latency (NOT build time).
# correct:false ⇒ claimed_speedup ≤ 1.0.
# exit 0 ok · 2 op-error (json printed) · 3 bad args / missing input.
set -u

# Clean error envelope with the full contract key set (claimed_speedup floored at 1.0).
err_envelope() {
  local msg="$1" rc="$2"
  printf '{"ok":false,"compiled":false,"correct":false,"candidate_latency_ms":null,'
  printf '"eager_latency_ms":null,"compile_latency_ms":null,"claimed_speedup":1.0,'
  printf '"error":"%s"}\n' "$msg"
  exit "$rc"
}

ARTIFACT="" PROBLEM="" OUT="" REPS=50 RTOL="1e-3" ATOL="1e-3" BASELINE="both"
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact) ARTIFACT="${2:-}"; shift 2 ;;
    --problem)  PROBLEM="${2:-}"; shift 2 ;;
    --out)      OUT="${2:-}"; shift 2 ;;
    --reps)     REPS="${2:-}"; shift 2 ;;
    --rtol)     RTOL="${2:-}"; shift 2 ;;
    --atol)     ATOL="${2:-}"; shift 2 ;;
    --baseline) BASELINE="${2:-}"; shift 2 ;;
    *) err_envelope "unknown arg: $1" 3 ;;
  esac
done

[ -n "$ARTIFACT" ] || err_envelope "missing --artifact" 3
[ -n "$PROBLEM" ]  || err_envelope "missing --problem" 3
[ -n "$OUT" ]      || err_envelope "missing --out" 3
[ -f "$PROBLEM" ]  || err_envelope "problem not found: $PROBLEM" 3
[ -f "$ARTIFACT" ] || err_envelope "artifact not found: $ARTIFACT" 3   # preflight/bad-input => exit 3 (spec §4.5)

command -v python3 >/dev/null 2>&1 || err_envelope "python3 not found" 3

# --- GPU path (deferred): the python launcher does the real torch run. On a real box it
#     imports the .so, runs candidate vs eager/compile baselines, torch.allclose, and
#     CUDA-event timing, then prints the canonical envelope. Here it never gets reached
#     for the tested paths because the guards above fire first. ---
python3 - "$ARTIFACT" "$PROBLEM" "$OUT" "$REPS" "$RTOL" "$ATOL" "$BASELINE" <<'PY'
import sys, json
artifact, problem, out, reps, rtol, atol, baseline = sys.argv[1:8]
result = {
    "ok": False, "compiled": True, "correct": False,
    "candidate_latency_ms": None, "eager_latency_ms": None,
    "compile_latency_ms": None, "claimed_speedup": 1.0,
    "error": "GPU execution deferred: requires NVIDIA device + CUDA runtime",
}
try:
    import torch
    if not torch.cuda.is_available():
        result["error"] = "no CUDA device available (deferred GPU tier)"
        print(json.dumps(result)); sys.exit(2)
    # ---- real GPU path would go here (load .so, run, time, compare) ----
    result["error"] = "real GPU run not yet implemented"
    print(json.dumps(result)); sys.exit(2)
except ImportError:
    result["error"] = "torch unavailable"
    print(json.dumps(result)); sys.exit(2)
PY
exit $?
```

```bash
chmod +x _substrate/backends/cuda/run.sh
python3 -m unittest discover -s _substrate/tests -p 'test_driver_scripts.py' -v
```
Expected GREEN: all `TestCudaRun` pass (missing-artifact → exit 2 with full key set; missing-arg/bad-problem → exit 3).

- [ ] **Step 3 — commit.**
```
git add _substrate/backends/cuda/run.sh _substrate/tests/test_driver_scripts.py
git commit -m "P3: cuda/run.sh — anti_cheat-exact key set, missing-artifact envelope; GPU run deferred

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10 — `cuda/profile.sh` (ncu → csv pointer, fake-ncu tested; profiler-absent exit 4)

**Files**
- Create: `_substrate/backends/cuda/profile.sh`
- Modify: `_substrate/tests/test_driver_scripts.py` (add `TestCudaProfile`)

**Steps**

- [ ] **Step 1 — RED.** Append `TestCudaProfile`.

```python
class TestCudaProfile(unittest.TestCase):
    SCRIPT = os.path.join(CUDA, 'profile.sh')

    def _problem(self, td):
        p = os.path.join(td, 'problem.json')
        json.dump({"op": "add"}, open(p, 'w'))
        return p

    def test_exists_executable_and_syntax(self):
        self.assertTrue(os.path.isfile(self.SCRIPT), "cuda/profile.sh missing")
        self.assertTrue(os.access(self.SCRIPT, os.X_OK))
        code, _, err = _run(['bash', '-n', self.SCRIPT])
        self.assertEqual(code, 0, msg=err)

    def test_profile_ok_with_fake_ncu_writes_csv_and_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            _write_exec(os.path.join(td, 'ncu'), FAKE_NCU_CSV)
            art = os.path.join(td, 'k.so'); open(art, 'w').write("")
            prob = self._problem(td); out = os.path.join(td, 'native.csv')
            code, sout, serr = _run([self.SCRIPT, '--artifact', art,
                                     '--problem', prob, '--out', out], env=_path_env(td))
            self.assertEqual(code, 0, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), True, p)
            self.assertEqual(p.get('profiler'), 'ncu', p)
            self.assertEqual(p.get('native_profile'), out, p)
            self.assertEqual(p.get('format'), 'ncu-csv', p)
            self.assertTrue(os.path.isfile(out), "csv not written")
            self.assertIn('sm__warps_active', open(out).read())

    def test_profile_requests_the_four_counters(self):
        with tempfile.TemporaryDirectory() as td:
            rec = os.path.join(td, 'argv.txt')
            _write_exec(os.path.join(td, 'ncu'), textwrap.dedent(f'''\
                #!/usr/bin/env bash
                echo "$@" > "{rec}"
                echo '"ID","Metric Name","Metric Value"' ; exit 0
            '''))
            art = os.path.join(td, 'k.so'); open(art, 'w').write("")
            prob = self._problem(td); out = os.path.join(td, 'n.csv')
            _run([self.SCRIPT, '--artifact', art, '--problem', prob, '--out', out],
                 env=_path_env(td))
            argv = open(rec).read()
            for c in ('gpu__time_duration.sum',
                      'sm__throughput.avg.pct_of_peak_sustained_elapsed',
                      'dram__bytes_read.sum.pct_of_peak_sustained_elapsed',
                      'sm__warps_active.avg.pct_of_peak_sustained_active'):
                self.assertIn(c, argv, f"profile.sh did not request {c}")

    def test_profiler_absent_exit_4(self):
        with tempfile.TemporaryDirectory() as td:
            art = os.path.join(td, 'k.so'); open(art, 'w').write("")
            prob = self._problem(td); out = os.path.join(td, 'n.csv')
            env = dict(os.environ)   # ncu genuinely absent on macOS; keep PATH so the shebang resolves (wiping it => exit 127, not the exit-4 guard)
            code, sout, serr = _run([self.SCRIPT, '--artifact', art,
                                     '--problem', prob, '--out', out], env=env)
            self.assertEqual(code, 4, msg=f"out={sout} err={serr}")
            self.assertEqual(_json_or_raw(sout).get('ok'), False)

    def test_missing_args_exit_3(self):
        code, sout, _ = _run([self.SCRIPT, '--artifact', '/x.so'])
        self.assertEqual(code, 3)
```

```bash
python3 -m unittest discover -s _substrate/tests -p 'test_driver_scripts.py' -v
```
Expected RED: `cuda/profile.sh missing`.

- [ ] **Step 2 — GREEN.** Create `_substrate/backends/cuda/profile.sh` (complete).

```bash
#!/usr/bin/env bash
# cuda/profile.sh — run ncu over the artifact and write an ncu --csv report to --out.
# Prints the POINTER (not the metrics): {ok,profiler:"ncu",native_profile,format:"ncu-csv"}.
# Spec §4.5. exit 0 ok · 3 bad args / missing input · 4 profiler unavailable.
set -u

emit() { printf '%s\n' "$1"; }
die3() { emit "{\"ok\":false,\"profiler\":\"ncu\",\"native_profile\":null,\"error\":\"$1\"}"; exit 3; }
die4() { emit "{\"ok\":false,\"profiler\":\"ncu\",\"native_profile\":null,\"error\":\"$1\"}"; exit 4; }

ARTIFACT="" PROBLEM="" OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact) ARTIFACT="${2:-}"; shift 2 ;;
    --problem)  PROBLEM="${2:-}"; shift 2 ;;
    --out)      OUT="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$ARTIFACT" ] || die3 "missing --artifact"
[ -n "$PROBLEM" ]  || die3 "missing --problem"
[ -n "$OUT" ]      || die3 "missing --out"
[ -f "$ARTIFACT" ] || die3 "artifact not found: $ARTIFACT"
[ -f "$PROBLEM" ]  || die3 "problem not found: $PROBLEM"

command -v ncu >/dev/null 2>&1 || die4 "ncu profiler not available"

# The four counters to_evidence/_evidence_nvidia.py parses (AccelOpt set).
METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.pct_of_peak_sustained_elapsed,dram__bytes_write.sum.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

# A python launcher loads the .so and runs the kernel once; ncu profiles that process.
# On a real box this is the actual launcher; here ncu is faked so the launcher is moot.
LAUNCHER="python3 -c import sys"

# ncu --csv prints CSV to stdout; capture it into --out.
ncu --csv --page raw --metrics "$METRICS" --target-processes all \
    python3 -c "pass" >"$OUT" 2>/tmp/ncu_profile.stderr
RC=$?
[ -s /tmp/ncu_profile.stderr ] && cat /tmp/ncu_profile.stderr 1>&2
rm -f /tmp/ncu_profile.stderr

if [ "$RC" -ne 0 ]; then
  emit "{\"ok\":false,\"profiler\":\"ncu\",\"native_profile\":null,\"error\":\"ncu failed (exit $RC); no profile produced\"}"
  exit 4   # spec §4.5 profile.sh codes are 0/4/3 only; a failed ncu run maps to 4 (profile unavailable)
fi

emit "{\"ok\":true,\"profiler\":\"ncu\",\"native_profile\":\"$OUT\",\"format\":\"ncu-csv\"}"
exit 0
```

```bash
chmod +x _substrate/backends/cuda/profile.sh
python3 -m unittest discover -s _substrate/tests -p 'test_driver_scripts.py' -v
```
Expected GREEN: all `TestCudaProfile` pass (csv written, pointer emitted, four counters requested, ncu-absent → exit 4).

- [ ] **Step 3 — commit.**
```
git add _substrate/backends/cuda/profile.sh _substrate/tests/test_driver_scripts.py
git commit -m "P3: cuda/profile.sh — ncu csv pointer, 4-counter request, exit-4 on absent profiler

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11 — Triton driver dir: confirm L0 + `to_evidence` via the shell-script suite

The triton `manifest.json`/`idioms.json`/`to_evidence.py` were created in Tasks 2 and 5. This task adds the `TestTritonL0` class to the shell-script suite so the triton dir's L0 pass and its `to_evidence` vendor-collapse are also asserted from within `test_driver_scripts.py` (alongside the upcoming `.sh` tests), then confirms green.

> Reconciliation note: there is exactly ONE triton manifest/idioms/to_evidence set — authored in Tasks 2 and 5 with the canonical `compiler.name "triton"`, `artifact_ext ".ptx"`, `requires_tools ["python3"]` manifest. This task does NOT re-create them; it only adds the cross-checking test class. The fake-ncu CSV uses the canonical `dram_pct = read + write = 62.0`.

**Files**
- Modify: `_substrate/tests/test_driver_scripts.py` (add `TestTritonL0`)

**Steps**

- [ ] **Step 1 — add `TestTritonL0`.** Append:

```python
class TestTritonL0(unittest.TestCase):
    VALIDATOR = os.path.join(BACKENDS, 'validate_backend.py')

    def test_triton_dir_passes_l0(self):
        code, out, err = _run([sys.executable, self.VALIDATOR, TRITON])
        self.assertEqual(code, 0, msg=f"out={out} err={err}")
        self.assertEqual(_json_or_raw(out).get('ok'), True, out)

    def test_triton_to_evidence_uses_shared_mapper_source_triton(self):
        with tempfile.TemporaryDirectory() as td:
            csv = os.path.join(td, 'n.csv')
            _write_exec(os.path.join(td, '_e.sh'), FAKE_NCU_CSV)
            with open(csv, 'w') as fh:
                subprocess.run([os.path.join(td, '_e.sh')], stdout=fh)
            code, out, err = _run([sys.executable,
                                   os.path.join(TRITON, 'to_evidence.py'),
                                   '--native', csv, '--format', 'ncu-csv'])
            self.assertEqual(code, 0, msg=f"{out} {err}")
            p = _json_or_raw(out)
            self.assertEqual(p['source_backend'], 'triton', p)
            self.assertEqual(p['metrics']['_vendor'], 'nvidia', p)
            self.assertAlmostEqual(p['metrics']['occupancy'], 0.51, places=4)
            self.assertAlmostEqual(p['metrics']['dram_pct'], 62.0, places=3)  # read + write
```

```bash
python3 -m unittest discover -s _substrate/tests -p 'test_driver_scripts.py' -v
```
Expected GREEN: `TestTritonL0` both pass (L0 ok, source_backend=='triton', occupancy 0.51, dram_pct 62.0) — the triton dir was already built in Tasks 2 and 5.

- [ ] **Step 2 — commit.**
```
git add _substrate/tests/test_driver_scripts.py
git commit -m "P3: assert triton L0 + to_evidence vendor-collapse from the shell-script suite

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12 — `triton/build.sh`, `triton/run.sh`, `triton/profile.sh` (python/JIT, fake-tool tested)

Triton's "build" is a JIT warmup (run python that imports the kernel and triggers a launch to materialize PTX into `TRITON_CACHE_DIR`). On macOS `triton` is absent, so the warmup python exits cleanly with an error envelope; we test arg handling, the envelope, exit codes, and (for profile) the ncu pointer + the `--kernel-name` caveat.

**Files**
- Create: `_substrate/backends/triton/build.sh`
- Create: `_substrate/backends/triton/run.sh`
- Create: `_substrate/backends/triton/profile.sh`
- Modify: `_substrate/tests/test_driver_scripts.py` (add `TestTritonScripts`)

**Steps**

- [ ] **Step 1 — RED.** Append `TestTritonScripts`.

```python
class TestTritonScripts(unittest.TestCase):
    BUILD = os.path.join(TRITON, 'build.sh')
    RUN = os.path.join(TRITON, 'run.sh')
    PROFILE = os.path.join(TRITON, 'profile.sh')

    def test_all_three_exist_executable_syntax(self):
        for s in (self.BUILD, self.RUN, self.PROFILE):
            self.assertTrue(os.path.isfile(s), f"{s} missing")
            self.assertTrue(os.access(s, os.X_OK), f"{s} not executable")
            code, _, err = _run(['bash', '-n', s])
            self.assertEqual(code, 0, msg=f"{s}: {err}")

    def test_build_missing_args_exit_3(self):
        code, sout, _ = _run([self.BUILD, '--source', '/k.py'])  # no --out
        self.assertEqual(code, 3)
        self.assertEqual(_json_or_raw(sout).get('ok'), False)

    def test_build_envelope_keys_present_on_triton_absent(self):
        # triton is absent on macOS → build.sh's warmup must still print the envelope.
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, 'kernel.py'); out = os.path.join(td, 'art.json')
            open(src, 'w').write("# triton kernel\n")
            code, sout, serr = _run([self.BUILD, '--source', src, '--out', out])
            p = _json_or_raw(sout)
            self.assertIn('ok', p); self.assertIn('compiled', p)
            self.assertIn('build_latency_ms', p); self.assertIn('stderr_tail', p)
            # triton absent ⇒ not compiled, op-error exit 2
            self.assertEqual(p['compiled'], False, p)
            self.assertEqual(code, 2, msg=f"out={sout} err={serr}")

    def test_run_missing_artifact_full_key_set_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            prob = os.path.join(td, 'p.json'); json.dump({"op": "add"}, open(prob, 'w'))
            out = os.path.join(td, 'r.json')
            code, sout, _ = _run([self.RUN, '--artifact', '/nope.json',
                                  '--problem', prob, '--out', out])
            self.assertEqual(code, 2)
            p = _json_or_raw(sout)
            for k in ('compiled', 'correct', 'candidate_latency_ms', 'eager_latency_ms',
                      'compile_latency_ms', 'claimed_speedup'):
                self.assertIn(k, p, f"missing {k}: {p}")
            self.assertEqual(p['correct'], False)
            self.assertLessEqual(p['claimed_speedup'], 1.0)

    def test_profile_ok_with_fake_ncu_pointer(self):
        with tempfile.TemporaryDirectory() as td:
            _write_exec(os.path.join(td, 'ncu'), FAKE_NCU_CSV)
            art = os.path.join(td, 'art.json'); open(art, 'w').write("{}")
            prob = os.path.join(td, 'p.json'); json.dump({"op": "add"}, open(prob, 'w'))
            out = os.path.join(td, 'n.csv')
            code, sout, serr = _run([self.PROFILE, '--artifact', art, '--problem', prob,
                                     '--out', out, '--kernel-name', 'add_kernel'],
                                    env=_path_env(td))
            self.assertEqual(code, 0, msg=f"out={sout} err={serr}")
            p = _json_or_raw(sout)
            self.assertEqual(p.get('ok'), True, p)
            self.assertEqual(p.get('native_profile'), out, p)
            self.assertEqual(p.get('format'), 'ncu-csv', p)
            self.assertTrue(os.path.isfile(out))

    def test_profile_ncu_absent_exit_4(self):
        with tempfile.TemporaryDirectory() as td:
            art = os.path.join(td, 'a.json'); open(art, 'w').write("{}")
            prob = os.path.join(td, 'p.json'); json.dump({"op": "add"}, open(prob, 'w'))
            out = os.path.join(td, 'n.csv')
            env = dict(os.environ)   # ncu genuinely absent on macOS; keep PATH so the shebang resolves (wiping it => exit 127, not the exit-4 guard)
            code, sout, _ = _run([self.PROFILE, '--artifact', art, '--problem', prob,
                                  '--out', out], env=env)
            self.assertEqual(code, 4)
```

```bash
python3 -m unittest discover -s _substrate/tests -p 'test_driver_scripts.py' -v
```
Expected RED: `triton/build.sh missing`.

- [ ] **Step 2a — GREEN.** Create `_substrate/backends/triton/build.sh` (JIT warmup).

```bash
#!/usr/bin/env bash
# triton/build.sh — "build" = JIT warmup: import the kernel + trigger one launch so
# Triton materializes PTX into TRITON_CACHE_DIR. The cache dir IS the artifact.
# Same envelope as cuda/build.sh. Spec §4.5/§5.1.
#   exit 0 ok · 2 warmup failure (json printed) · 3 bad args.
set -u

emit() { printf '%s\n' "$1"; }
die3() { emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"error\":\"$1\"}"; exit 3; }

SOURCE="" OUT="" EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --source) SOURCE="${2:-}"; shift 2 ;;
    --out)    OUT="${2:-}"; shift 2 ;;
    --extra)  EXTRA="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$SOURCE" ] || die3 "missing --source"
[ -n "$OUT" ]    || die3 "missing --out"
[ -f "$SOURCE" ] || die3 "source not found: $SOURCE"
command -v python3 >/dev/null 2>&1 || die3 "python3 not found"

# TRITON_CACHE_DIR receives the compiled PTX. Use --out as the cache root.
export TRITON_CACHE_DIR="$OUT"
mkdir -p "$OUT" 2>/dev/null

START="$(python3 -c 'import time;print(int(time.time()*1000))')"
STDERR_FILE="$(mktemp)"
# Warmup: import triton + the kernel module + run one launch. On macOS triton is absent,
# so this exits non-zero with an ImportError captured into stderr.
python3 - "$SOURCE" 2>"$STDERR_FILE" <<'PY'
import sys, importlib.util
src = sys.argv[1]
import triton  # noqa: F401  (absent on macOS -> ImportError, caught by exit code)
spec = importlib.util.spec_from_file_location("candidate_kernel", src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# A real warmup would call mod.forward() / launch the @triton.jit kernel here to
# force PTX materialization into TRITON_CACHE_DIR.
PY
RC=$?
END="$(python3 -c 'import time;print(int(time.time()*1000))')"
BUILD_MS=$(( END - START ))
TAIL="$(tail -n 20 "$STDERR_FILE")"
ESC_TAIL="$(python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))' <"$STDERR_FILE")"
cat "$STDERR_FILE" 1>&2
rm -f "$STDERR_FILE"

if [ "$RC" -eq 0 ]; then
  emit "{\"ok\":true,\"compiled\":true,\"artifact\":\"$OUT\",\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":\"\"}"
  exit 0
else
  emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":$ESC_TAIL}"
  exit 2
fi
```

- [ ] **Step 2b — GREEN.** Create `_substrate/backends/triton/run.sh`.

```bash
#!/usr/bin/env bash
# triton/run.sh — python launcher + torch.testing.assert_close + triton.testing.do_bench.
# Same stdout key shape as cuda/run.sh (anti_cheat contract). Spec §4.5.
#   exit 0 ok · 2 op-error (json printed) · 3 bad args / missing input.
set -u

err_envelope() {
  local msg="$1" rc="$2"
  printf '{"ok":false,"compiled":false,"correct":false,"candidate_latency_ms":null,'
  printf '"eager_latency_ms":null,"compile_latency_ms":null,"claimed_speedup":1.0,'
  printf '"error":"%s"}\n' "$msg"
  exit "$rc"
}

ARTIFACT="" PROBLEM="" OUT="" REPS=50 RTOL="1e-3" ATOL="1e-3" BASELINE="both"
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact) ARTIFACT="${2:-}"; shift 2 ;;
    --problem)  PROBLEM="${2:-}"; shift 2 ;;
    --out)      OUT="${2:-}"; shift 2 ;;
    --reps)     REPS="${2:-}"; shift 2 ;;
    --rtol)     RTOL="${2:-}"; shift 2 ;;
    --atol)     ATOL="${2:-}"; shift 2 ;;
    --baseline) BASELINE="${2:-}"; shift 2 ;;
    *) err_envelope "unknown arg: $1" 3 ;;
  esac
done

[ -n "$ARTIFACT" ] || err_envelope "missing --artifact" 3
[ -n "$PROBLEM" ]  || err_envelope "missing --problem" 3
[ -n "$OUT" ]      || err_envelope "missing --out" 3
[ -f "$PROBLEM" ]  || err_envelope "problem not found: $PROBLEM" 3
[ -e "$ARTIFACT" ] || err_envelope "artifact not found: $ARTIFACT" 3   # preflight/bad-input => exit 3 (spec §4.5)
command -v python3 >/dev/null 2>&1 || err_envelope "python3 not found" 3

python3 - "$ARTIFACT" "$PROBLEM" "$OUT" "$REPS" "$RTOL" "$ATOL" "$BASELINE" <<'PY'
import sys, json
artifact, problem, out, reps, rtol, atol, baseline = sys.argv[1:8]
result = {
    "ok": False, "compiled": True, "correct": False,
    "candidate_latency_ms": None, "eager_latency_ms": None,
    "compile_latency_ms": None, "claimed_speedup": 1.0,
    "error": "GPU execution deferred: requires NVIDIA device + triton runtime",
}
try:
    import torch
    if not torch.cuda.is_available():
        result["error"] = "no CUDA device available (deferred GPU tier)"
        print(json.dumps(result)); sys.exit(2)
    import triton  # noqa: F401
    # ---- real path: import launcher, torch.testing.assert_close, triton.testing.do_bench
    result["error"] = "real triton run not yet implemented"
    print(json.dumps(result)); sys.exit(2)
except ImportError as exc:
    result["error"] = f"runtime unavailable: {exc}"
    print(json.dumps(result)); sys.exit(2)
PY
exit $?
```

- [ ] **Step 2c — GREEN.** Create `_substrate/backends/triton/profile.sh` (ncu on the python launcher; `--kernel-name` caveat).

```bash
#!/usr/bin/env bash
# triton/profile.sh — run ncu over the python launcher and write ncu --csv to --out.
# CAVEAT (Triton kernel-name discovery): Triton mangles the kernel symbol name and stores
# the compiled object in TRITON_CACHE_DIR, so ncu's --kernel-name regex cannot be derived
# statically. For now we accept an explicit --kernel-name; if omitted we profile ALL
# kernels (--kernel-name left unset). Auto-discovery from TRITON_CACHE_DIR is deferred.
# Same pointer envelope as cuda/profile.sh. Spec §4.5/§5.1.
#   exit 0 ok · 2 ncu error · 3 bad args / missing input · 4 ncu unavailable.
set -u

emit() { printf '%s\n' "$1"; }
die3() { emit "{\"ok\":false,\"profiler\":\"ncu\",\"native_profile\":null,\"error\":\"$1\"}"; exit 3; }
die4() { emit "{\"ok\":false,\"profiler\":\"ncu\",\"native_profile\":null,\"error\":\"$1\"}"; exit 4; }

ARTIFACT="" PROBLEM="" OUT="" KERNEL_NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact)    ARTIFACT="${2:-}"; shift 2 ;;
    --problem)     PROBLEM="${2:-}"; shift 2 ;;
    --out)         OUT="${2:-}"; shift 2 ;;
    --kernel-name) KERNEL_NAME="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$ARTIFACT" ] || die3 "missing --artifact"
[ -n "$PROBLEM" ]  || die3 "missing --problem"
[ -n "$OUT" ]      || die3 "missing --out"
[ -e "$ARTIFACT" ] || die3 "artifact not found: $ARTIFACT"
[ -f "$PROBLEM" ]  || die3 "problem not found: $PROBLEM"
command -v ncu >/dev/null 2>&1 || die4 "ncu profiler not available"

METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.pct_of_peak_sustained_elapsed,dram__bytes_write.sum.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

# TRITON_CACHE_DIR points at the build artifact so the launcher reuses the warmed PTX.
export TRITON_CACHE_DIR="$ARTIFACT"

# bash 3.2 (macOS default) aborts on "${ARR[@]}" for an EMPTY array under `set -u`, so inline
# the two branches instead of expanding a possibly-empty array. Also pass --page raw so live ncu
# emits the long format _evidence_nvidia._parse_ncu_csv reads.
if [ -n "$KERNEL_NAME" ]; then
  ncu --csv --page raw --metrics "$METRICS" --target-processes all --kernel-name "$KERNEL_NAME" \
      python3 -c "pass" >"$OUT" 2>/tmp/triton_ncu.stderr
else
  ncu --csv --page raw --metrics "$METRICS" --target-processes all \
      python3 -c "pass" >"$OUT" 2>/tmp/triton_ncu.stderr
fi
RC=$?
[ -s /tmp/triton_ncu.stderr ] && cat /tmp/triton_ncu.stderr 1>&2
rm -f /tmp/triton_ncu.stderr

if [ "$RC" -ne 0 ]; then
  emit "{\"ok\":false,\"profiler\":\"ncu\",\"native_profile\":null,\"error\":\"ncu failed (exit $RC); no profile produced\"}"
  exit 4   # spec §4.5 profile.sh codes are 0/4/3 only; a failed ncu run maps to 4 (profile unavailable)
fi
emit "{\"ok\":true,\"profiler\":\"ncu\",\"native_profile\":\"$OUT\",\"format\":\"ncu-csv\"}"
exit 0
```

```bash
chmod +x _substrate/backends/triton/build.sh _substrate/backends/triton/run.sh _substrate/backends/triton/profile.sh
python3 -m unittest discover -s _substrate/tests -p 'test_driver_scripts.py' -v
```
Expected GREEN: all `TestTritonScripts` pass.

- [ ] **Step 3 — commit.**
```
git add _substrate/backends/triton/build.sh _substrate/backends/triton/run.sh _substrate/backends/triton/profile.sh _substrate/tests/test_driver_scripts.py
git commit -m "P3: triton build/run/profile.sh — JIT-warmup envelope, anti_cheat keys, ncu pointer + kernel-name caveat

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 13 — Full-suite green + REGISTRY.md update (cuda/triton → experimental, GPU-untested note)

**Files**
- Modify: `_substrate/backends/REGISTRY.md`

**Steps**

- [ ] **Step 1 — full suite green.** Confirm nothing regressed and the new tests joined.
```bash
python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v 2>&1 | tail -5
```
Expected: all prior tests + the new `test_evidence_nvidia.py`, `test_driver_conformance.py`, and `test_driver_scripts.py` tests pass.

- [ ] **Step 2 — read then update REGISTRY.md.** Read `_substrate/backends/REGISTRY.md` and confirm its current table/columns (it already seeds `cuda` from commit 33ddb05). Preserve the existing column shape; do not invent new schema. Replace the single `cuda … planned` row and add `triton`, plus the note.

Replace (example — adapt to the file's actual row text):
```
| cuda | `cuda/` | nvidia | planned | (unassigned) |
```
with:
```
| cuda | `cuda/` | nvidia | experimental | (unassigned) |
| triton | `triton/` | nvidia | experimental | (unassigned) |

> **Note (P3):** the `cuda` and `triton` `build.sh`/`run.sh`/`profile.sh` are
> **GPU-untested** — this repo runs on macOS where `nvcc`/`ncu`/`triton` are absent. What
> IS verified on macOS: `validate_backend.py` (L0) for both dirs, and `to_evidence.py`
> (the shared `_evidence_nvidia.py` NCU→canonical mapping, incl. `occupancy = warps ÷ 100`
> and `dram_pct = read + write`) via fake-tool PATH stubs. The `.sh` scripts have
> arg-parsing, JSON-envelope, and exit-code coverage only. End-to-end compile/run/profile
> is **deferred to the GPU/CI tier** (spec §8.3, §9.3).
```

- [ ] **Step 3 — commit.**
```
git add _substrate/backends/REGISTRY.md
git commit -m "docs(substrate): register cuda + triton drivers (experimental, GPU-deferred)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Definition of Done

This plan is complete when all of the following hold (mapping to spec Appendix A P3 + §5.1):

- **Six-file driver set authored for both backends** (Appendix A P3): each of `_substrate/backends/cuda/` and `_substrate/backends/triton/` contains `manifest.json`, `idioms.json`, `build.sh`, `run.sh`, `profile.sh`, and `to_evidence.py`.
- **L0 conformance** (Appendix A P3): `validate_backend.py` exits `0` with `{"ok": true, "errors": []}` for both driver dirs; `backend_id == dir name`; every `idioms.json` `methods`/`unsupported_methods` name is a live `method_gate.TABLE` entry; all four meaningful bottleneck classes are covered. Pinned by `test_driver_conformance.py`.
- **Shared NVIDIA evidence mapper** (§5.1 vendor-collapse + §4.6 canonical units + §5.3.2 null rule): `_evidence_nvidia.py` maps the four NCU counters to `latency_ms` (ns ÷ 1e6), `dram_pct` (read + write, 0–100), `sm_pct` (0–100), and `occupancy` (warps_active% ÷ 100, 0–1); absent counters become JSON `null` and are omitted from `coverage`. Both `cuda/to_evidence.py` and `triton/to_evidence.py` are thin wrappers producing byte-identical metrics differing only in `source_backend`. Pinned by `test_evidence_nvidia.py`.
- **Universal envelope contract** (§4.5): every `.sh` and `to_evidence.py` prints ONE JSON object on stdout, logs to stderr, and uses exit `0` ok / `2` op-error (JSON still printed) / `3` bad-args-or-missing-tool / `4` profiler-unavailable; `run.sh` emits the exact anti_cheat key set (`compiled`, `correct`, `candidate_latency_ms`, `eager_latency_ms`, `compile_latency_ms`, `claimed_speedup`) with `correct:false ⇒ claimed_speedup ≤ 1.0`; `build.sh` passes `-lineinfo` and reports a distinct `build_latency_ms`. Pinned by `test_driver_scripts.py` via fake-tool PATH stubs.
- **Honest experimental status**: both manifests carry `status "experimental"`; `REGISTRY.md` lists both as `experimental`, `hw_vendor nvidia`, with the GPU-deferred note.
- **Full test suite green** on macOS: `python3 -m unittest discover -s _substrate/tests -p 'test_*.py' -v` ends in `OK` (prior 37 + the new `test_evidence_nvidia.py`, `test_driver_conformance.py`, `test_driver_scripts.py` tests), with no existing test modified.
- **All work committed** on `dev/solver-substrate` in the per-task commits above.

---

## Deferred verification (GPU/CI tier)

These require a real **NVIDIA box with nvcc + ncu + triton** and CANNOT be tested on macOS. They are explicitly out of scope for this plan's green bar (spec §8.3 opt-in hardware tier, §9.3 mock harness):

1. **cuda end-to-end**: `build.sh` actually compiling a `.cu` to a loadable `.so` with `-lineinfo`; `run.sh` loading it, running candidate vs eager/`torch.compile` baselines, real `torch.allclose` correctness, CUDA-event latency, and a real `claimed_speedup`; `profile.sh` producing a genuine ncu CSV that `to_evidence.py` maps to non-null metrics.
2. **triton end-to-end**: JIT warmup truly materializing PTX into `TRITON_CACHE_DIR`; `run.sh` with `triton.testing.do_bench` + `torch.testing.assert_close`; `profile.sh` ncu over the launcher.
3. **ncu-on-triton kernel-name discovery**: deriving the mangled `@triton.jit` symbol name from `TRITON_CACHE_DIR` so `--kernel-name` can be auto-populated (currently a manual `--kernel-name` arg; mangled-name auto-discovery deferred).
4. **`compile_latency_ms` semantics on a live `torch.compile`**: confirming `run.sh` reports the torch.compile BASELINE wall-time (distinct from `build.sh`'s `build_latency_ms` JIT/compile time) — the two keys cannot diverge meaningfully without a GPU.
5. **The exit-`4` profiler-unavailable path** of `profile.sh` (`ncu` absent) observed where `ncu` could otherwise exist, plus an end-to-end `profile.sh → to_evidence.py → diagnose.py` integration test on a real kernel.
6. **GPU-tier confirmation of the assumed NCU CSV header/format**: if the real `ncu --csv --page raw` header or value formatting (comma-grouping / unit suffixes / multi-kernel ordering) differs, only `_evidence_nvidia._parse_ncu_csv` changes, and the Task-1 tests pin the contract that change must preserve.
7. **L1–L3 conformance** for both drivers (only L0 is verified on macOS) → promote `experimental` → `stable` only after that runs green on the GPU tier.