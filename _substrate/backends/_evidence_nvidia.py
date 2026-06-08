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
  dram_pct   = dram_read% + dram_write%;                     # each 0-100; SUM may reach ~200
               #   downstream must NOT assume dram_pct is bounded by [0,100]
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
        # TODO: GPU tier confirm ncu never emits blank-kernel-name rows with parseable metric values
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

    # dram_pct = read% + write%; each counter is 0-100 so the SUM may reach ~200 —
    # downstream must NOT assume [0,100] for this key.
    # NOTE: read+write summation convention is a GPU-tier-to-confirm assumption (deferred risk).
    # null only when read is absent.
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
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except UnicodeDecodeError as exc:
        raise NativeParseError(f"native file is not valid UTF-8: {exc}") from exc


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
    except NativeParseError as exc:
        # non-UTF-8 / undecodable file: exit 2 (same as parse errors, not a bad-arg exit 3)
        print(json.dumps({"ok": False, "error": f"native unparseable: {exc}"},
                         ensure_ascii=False))
        print(f"[_evidence_nvidia] parse error: {exc}", file=sys.stderr)
        return 2

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
