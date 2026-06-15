#!/usr/bin/env python3
"""MetaX mcProfiler-CSV / mcTracer-log -> canonical-metrics mapper.

Maps MetaX MACA SDK profiler output to the universal 4-key metric dict:
  latency_ms, dram_pct, sm_pct, occupancy

mcProfiler CSV format (MACA SDK, exact columns may drift per version):
  The mapper matches columns by normalised-name alias (lowercase, whitespace collapsed),
  not by exact header string, to tolerate SDK version drift.

mcTracer log format:
  Line-oriented runtime trace. Only kernel latency is reliably extractable;
  dram_pct/sm_pct/occupancy are always null (honest degradation, null rule).

Canonical units:
  latency_ms = kernel duration (microseconds / 1000, or nanoseconds / 1e6)
  dram_pct   = DRAM bandwidth utilisation (0–100, may exceed 100 on some MetaX SKUs)
  sm_pct     = SM / compute-unit utilisation (0–100)
  occupancy  = ALWAYS null — MetaX has no warp-occupancy counter

Null rule: a null canonical key is OMITTED from the 'coverage' set in the evidence
envelope, so diagnose.py yields 'unknown' rather than a wrong label.
"""
import os, sys, json, csv, io, argparse, re

MCPROF_COLUMN_ALIASES = {
    # latency
    "kernel_time_us": "latency_us",
    "duration_us": "latency_us",
    "elapsed_us": "latency_us",
    "gpu_time_us": "latency_us",
    "kernel_duration": "latency_us",
    "time": "latency_us",
    # DRAM bandwidth
    "dram_read_bw": "dram_read_pct",
    "dram_write_bw": "dram_write_pct",
    "dram_bandwidth_utilization": "dram_pct",
    "dram_bw_pct": "dram_pct",
    "global_memory_bandwidth_pct": "dram_pct",
    "hbm_bandwidth_pct": "dram_pct",
    # SM / compute
    "sm_utilization": "sm_pct",
    "sm_util_pct": "sm_pct",
    "compute_utilization": "sm_pct",
    "compute_util_pct": "sm_pct",
    "core_utilization": "sm_pct",
    "alu_utilization": "sm_pct",
}

MCTRACE_LATENCY_PATTERN = re.compile(
    r"(?:kernel|Kernel)\s+[\"']?(\S+?)[\"']?\s*[:=]\s*([\d.]+)\s*(us|ms|ns|µs)",
    re.IGNORECASE,
)


class NativeParseError(Exception):
    pass


def _normalise_name(raw):
    return re.sub(r"\s+", " ", raw.strip().lower().replace("_", " "))


def _parse_float(raw):
    if raw is None:
        return None
    s = str(raw).strip().strip('"').strip("'").replace(",", "")
    if s == "" or s.lower() in ("n/a", "null", "none", "nan"):
        return None
    # Accept trailing unit token (e.g. "123.45 us" -> 123.45)
    m = re.match(r"([\d.eE+-]+)", s)
    if m:
        return float(m.group(1))
    raise NativeParseError(f"cannot parse float from: {raw!r}")


def _parse_mcprof_csv(text):
    """Parse mcProfiler CSV output into raw metric dict."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise NativeParseError("mcprof CSV has no header row")

    # Build normalised column index
    col_map = {}
    for col in reader.fieldnames:
        norm = _normalise_name(col)
        if norm in MCPROF_COLUMN_ALIASES:
            col_map[col] = MCPROF_COLUMN_ALIASES[norm]

    # Take the first data row (primary kernel)
    for row in reader:
        result = {}
        for col, canonical in col_map.items():
            val = _parse_float(row.get(col, ""))
            if val is not None:
                result[canonical] = val
        return result

    raise NativeParseError("mcprof CSV has no data rows")


def _parse_mctrace_log(text):
    """Parse mcTracer runtime log. Only extracts kernel latency."""
    lines = text.splitlines()
    latencies_us = []
    for line in lines:
        m = MCTRACE_LATENCY_PATTERN.search(line)
        if m:
            value = float(m.group(2))
            unit = m.group(3).lower()
            if unit in ("ns",):
                value /= 1000.0
            elif unit in ("ms",):
                value *= 1000.0
            latencies_us.append(value)

    if not latencies_us:
        return {}

    # Use min latency (best observed kernel run)
    return {"latency_us": min(latencies_us)}


def to_canonical(native_metrics, source_backend, *, profiler_format="mcprof-csv"):
    """Map raw mcprof/mctrace metrics to canonical 4-key dict."""
    canonical = {
        "latency_ms": None,
        "dram_pct": None,
        "sm_pct": None,
        "occupancy": None,
    }

    # latency
    if "latency_us" in native_metrics:
        canonical["latency_ms"] = native_metrics["latency_us"] / 1000.0

    # DRAM
    dram_read = native_metrics.get("dram_read_pct")
    dram_write = native_metrics.get("dram_write_pct")
    dram_combined = native_metrics.get("dram_pct")
    if dram_combined is not None:
        canonical["dram_pct"] = dram_combined
    elif dram_read is not None and dram_write is not None:
        canonical["dram_pct"] = round(dram_read + dram_write, 2)
    elif dram_read is not None:
        canonical["dram_pct"] = dram_read
    elif dram_write is not None:
        canonical["dram_pct"] = dram_write

    # SM / compute
    canonical["sm_pct"] = native_metrics.get("sm_pct")

    per_key = {}
    for k in canonical:
        per_key[k] = {
            "value": canonical[k],
            "source_backend": source_backend,
            "profiler_format": profiler_format,
            "estimated": False,
        }

    return canonical, per_key


def _read_native(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def main(argv=None, source_backend=None):
    ap = argparse.ArgumentParser(description="MetaX mcProfiler/mcTracer -> canonical metrics")
    ap.add_argument("--native", required=True, help="path to mcprof CSV or mctrace log")
    ap.add_argument("--source-backend", default=source_backend or "metax")
    ap.add_argument("--format", default=None, help="mcprof-csv | mctrace-log (auto-detected)")
    ap.add_argument("--run", default=None, help="optional run result.json to merge")
    args = ap.parse_args(argv)

    text = _read_native(args.native)
    fmt = args.format
    if fmt is None:
        # Auto-detect: CSV header vs plain log
        if text.lstrip().startswith("kernel") or "," in (text.split("\n")[0] if text else ""):
            fmt = "mcprof-csv"
        else:
            fmt = "mctrace-log"

    try:
        if fmt == "mcprof-csv":
            native = _parse_mcprof_csv(text)
        elif fmt == "mctrace-log":
            native = _parse_mctrace_log(text)
        else:
            print(json.dumps({"ok": False, "error": f"unknown format: {fmt}"}))
            return 2
    except NativeParseError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 2

    canonical, per_key = to_canonical(native, args.source_backend, profiler_format=fmt)

    coverage = sorted(k for k, v in canonical.items() if v is not None)
    result = {
        "ok": True,
        "source_backend": args.source_backend,
        "profiler_format": fmt,
        "canonical": canonical,
        "coverage": coverage,
        "per_key": per_key,
    }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())