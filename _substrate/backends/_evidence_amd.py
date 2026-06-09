#!/usr/bin/env python3
"""Shared AMD rocprofv3 CSV -> canonical-metrics mapper (rocm backend).

Parallel to _evidence_nvidia.py (NCU mapper). Parses the CSV output of
`rocprofv3 --stats --csv` and maps native AMD counters to the four canonical
keys consumed by diagnose.py.

ASSUMED ROCPROF FORMAT: rocprofv3 --stats --csv long format with, at minimum,
columns "Name","Value" (or "Counter Name","Counter Value" for pmc mode).
First kernel's rows are used if several appear.

Canonical units emitted (every backend MUST honor):
  latency_ms = KernelDuration (ns) / 1e6
  dram_pct   = MemUnitBusy (0-100 %)
  sm_pct     = VALUBusy (0-100 %; VALU is the closest AMD analogue to NVIDIA SM throughput)
  occupancy  = Wavefronts / MaxWavefronts if available, else GPU_UTIL / 100  -> 0-1

Null rule: a counter absent from the CSV -> canonical key set to JSON null AND
omitted from `coverage` (NEVER fabricated 0.0).

Usage (invoked WITH a python prefix):
  _evidence_amd.py --native <rocprof.csv|-> --source-backend <id> [--format rocprof-csv] [--run <result.json>]
Prints ONE JSON object on stdout; logs to stderr.
Exit: 0 normalized · 2 native unparseable (JSON still printed, ok:false) · 3 bad args.
Pure function (same CSV in -> same JSON out).
"""
import os, sys, json, csv, io, argparse

# --- rocprofv3 metric names we look for (case-insensitive matching) ----------
# Duration counter (nanoseconds). KernelDuration is the rocprofv3 stats name;
# DurationNs is the pmc-mode name.
M_DURATION_KEYS = ("KernelDuration", "DurationNs", "Duration")
# Memory unit busy percentage — the AMD analogue of NVIDIA dram_pct.
# Strictly the memory-unit-busy counter; cache-hit ratios live in backend_native.
M_MEMBZ_KEYS = ("MemUnitBusy", "MEM_UNIT_BUSY")
# VALU busy percentage — the AMD analogue of NVIDIA sm_pct (SM throughput).
M_VALUBZ_KEYS = ("VALUBusy", "VALU_BUSY")
# Occupancy: prefer Wavefronts/MaxWavefronts ratio; falls back to a 0-100 pct.
M_OCC_KEYS = ("Wavefronts",)
M_MAXWV_KEYS = ("MaxWavefronts",)

# Alternative: rocprofv3 pmc counter names
M_FETCH_SIZE = "FETCH_SIZE"
M_WRITE_SIZE = "WRITE_SIZE"

# Canonical keys
CANONICAL_KEYS = ("latency_ms", "dram_pct", "sm_pct", "occupancy")

# Column name sets for header auto-detection (rocprofv3 varies across versions)
COL_NAME_ALTS = {"Name", "Counter Name", "Metric Name", "name"}
COL_VALUE_ALTS = {"Value", "Counter Value", "Metric Value", "value"}
COL_KERNEL_ALTS = {"KernelName", "Kernel Name", "kernel_name", "Name"}


class NativeParseError(Exception):
    """Raised when the native profile cannot be parsed into the expected shape."""


def _parse_float(raw):
    """Parse a rocprof 'Value' cell: strip thousands separators and any trailing
    unit token, return the leading float."""
    if raw is None:
        raise ValueError("empty metric value")
    s = str(raw).strip().replace(",", "")
    if not s:
        raise ValueError("empty metric value")
    token = s.split()[0]
    return float(token)


def _find_col(fieldnames, alts):
    """Return the first column name in fieldnames that matches one of the alts (case-insensitive)."""
    lower_map = {f.lower().strip(): f for f in fieldnames}
    for alt in alts:
        if alt.lower() in lower_map:
            return lower_map[alt.lower()]
    return None


def _parse_rocprof_csv(text):
    """Parse rocprofv3 CSV text into {metric_name: float} for the FIRST kernel seen.

    Handles both the --stats and --pmc output formats. Raises NativeParseError if
    no metric rows with a parseable value exist.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise NativeParseError("empty native file (no header row)")

    col_name = _find_col(reader.fieldnames, COL_NAME_ALTS)
    col_value = _find_col(reader.fieldnames, COL_VALUE_ALTS)

    if col_name is None or col_value is None:
        # Try single-row stats format: header is metric names, one data row
        # e.g. "KernelDuration,VALUBusy,MemUnitBusy,..."
        return _parse_rocprof_wide(text)

    col_kernel = _find_col(reader.fieldnames, COL_KERNEL_ALTS)
    if col_kernel == col_name:
        col_kernel = None

    first_kernel = None
    metrics = {}
    for row in reader:
        if col_kernel:
            kernel = (row.get(col_kernel) or "").strip()
            if first_kernel is None:
                first_kernel = kernel
            elif kernel != first_kernel:
                continue
        name = (row.get(col_name) or "").strip()
        if not name:
            continue
        try:
            metrics[name] = _parse_float(row.get(col_value))
        except (ValueError, TypeError):
            continue

    if not metrics:
        raise NativeParseError("native CSV had a valid header but no parseable metric rows")
    return first_kernel, metrics


def _parse_rocprof_wide(text):
    """Fallback for rocprofv3 wide/stats format where each column is a metric."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise NativeParseError("empty native file (no header row)")
    metrics = {}
    first_kernel = None
    for row in reader:
        for col_k in COL_KERNEL_ALTS:
            if col_k in row:
                kn = (row[col_k] or "").strip()
                if first_kernel is None:
                    first_kernel = kn
                elif kn != first_kernel:
                    continue
                break
        for key, val_str in row.items():
            key = key.strip()
            if key in COL_KERNEL_ALTS:
                continue
            try:
                metrics[key] = _parse_float(val_str)
            except (ValueError, TypeError):
                continue
        break  # only first data row
    if not metrics:
        raise NativeParseError("wide-format CSV had a header but no parseable row")
    return first_kernel, metrics


def _ci_get(metrics, key_set):
    """Case-insensitive lookup: return the value for the first key_set member found."""
    lower_map = {k.lower(): v for k, v in metrics.items()}
    for k in key_set:
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    return None


def to_canonical(native_metrics, source_backend):
    """Map a {rocprof_metric_name: float} dict to the canonical evidence dict."""
    # latency_ms
    latency_ms = None
    for dk in M_DURATION_KEYS:
        if dk in native_metrics:
            latency_ms = native_metrics[dk] / 1e6
            break
    if latency_ms is None:
        v = _ci_get(native_metrics, M_DURATION_KEYS)
        if v is not None:
            latency_ms = v / 1e6

    # dram_pct: MemUnitBusy is a 0-100 percentage on AMD
    dram_pct = _ci_get(native_metrics, M_MEMBZ_KEYS)

    # sm_pct: VALUBusy is the AMD analogue to NVIDIA SM throughput (0-100)
    sm_pct = _ci_get(native_metrics, M_VALUBZ_KEYS)

    # occupancy: Wavefronts/MaxWavefronts -> 0-1
    occupancy = None
    wf = _ci_get(native_metrics, M_OCC_KEYS)
    maxwf = _ci_get(native_metrics, M_MAXWV_KEYS)
    if wf is not None and maxwf is not None and maxwf > 0:
        occupancy = wf / maxwf
        if occupancy > 1.0:
            occupancy = occupancy / 100.0
    elif wf is not None:
        # If only wavefronts available and looks like a percentage (0-100)
        if wf <= 100.0:
            occupancy = wf / 100.0

    canonical = {
        "latency_ms": latency_ms,
        "dram_pct": dram_pct,
        "sm_pct": sm_pct,
        "occupancy": occupancy,
    }
    coverage = [k for k in CANONICAL_KEYS if canonical[k] is not None]

    # backend_native: every native metric not consumed by a canonical mapping
    consumed_lower = set()
    for ks in (M_DURATION_KEYS, M_MEMBZ_KEYS, M_VALUBZ_KEYS, M_OCC_KEYS, M_MAXWV_KEYS):
        for k in ks:
            consumed_lower.add(k.lower())
    backend_native = {k: v for k, v in native_metrics.items() if k.lower() not in consumed_lower}

    metrics = dict(canonical)
    metrics["_vendor"] = "amd"
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
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except UnicodeDecodeError as exc:
        raise NativeParseError(f"native file is not valid UTF-8: {exc}") from exc


def main(argv=None, source_backend=None):
    """CLI entrypoint. `source_backend` may be supplied by a thin wrapper;
    a --source-backend flag, if given, takes precedence. Returns an exit code."""
    ap = argparse.ArgumentParser(description="AMD rocprof CSV -> canonical metrics")
    ap.add_argument("--native", required=True, help="rocprof CSV path, or '-' for stdin")
    ap.add_argument("--source-backend", dest="source_backend", default=None,
                    help="backend id stamped into source_backend (rocm)")
    ap.add_argument("--format", default="rocprof-csv",
                    help="native profile format (only rocprof-csv supported)")
    ap.add_argument("--run", default=None,
                    help="optional run.sh result.json (reserved; not consumed by the mapper)")
    a = ap.parse_args(argv)

    sb = a.source_backend or source_backend
    if not sb:
        print(json.dumps({"ok": False, "error": "missing --source-backend"},
                         ensure_ascii=False))
        return 3

    if a.format != "rocprof-csv":
        print(json.dumps({"ok": False, "error": f"unsupported format {a.format}"},
                         ensure_ascii=False))
        return 3

    try:
        text = _read_native(a.native)
    except OSError as exc:
        print(json.dumps({"ok": False, "error": f"cannot read native file: {exc}"},
                         ensure_ascii=False))
        return 3
    except NativeParseError as exc:
        print(json.dumps({"ok": False, "error": f"native unparseable: {exc}"},
                         ensure_ascii=False))
        print(f"[_evidence_amd] parse error: {exc}", file=sys.stderr)
        return 2

    try:
        _kernel, native_metrics = _parse_rocprof_csv(text)
    except NativeParseError as exc:
        print(json.dumps({"ok": False, "error": f"native unparseable: {exc}"},
                         ensure_ascii=False))
        print(f"[_evidence_amd] parse error: {exc}", file=sys.stderr)
        return 2

    result = to_canonical(native_metrics, sb)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
