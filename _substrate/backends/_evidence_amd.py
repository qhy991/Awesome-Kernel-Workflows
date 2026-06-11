#!/usr/bin/env python3
"""Shared AMD rocprof-CSV -> canonical-metrics mapper (rocm backend).

Mirrors the spec of _evidence_nvidia.py (the cuda/triton mapper). Single source of
truth for AMD counter -> canonical-key mapping. Both rocprofv3 (ROCm 6+) and rocprof
(ROCm 5) emit a CSV with a "KernelName" column and one column per requested counter.

Canonical units emitted (every backend MUST honor):
  latency_ms = (EndNs - BeginNs) / 1e6                        # from rocprof timing
                # if absent, fall back to GRBM_GUI_ACTIVE / clock if available
  dram_pct   = read_bytes + write_bytes mapped to peak%,
               # null if no DRAM counters were collected
  sm_pct     = (SQ_INSTS_VALU / (SQ_WAVES * peak_valu_per_wave)) * 100
               # heuristic; documented in counter_guide.md
               # null when SQ_INSTS_VALU/SQ_WAVES is absent
  occupancy  = (SQ_WAVES / peak_waves_per_cu) clamped to [0,1]
               # null when SQ_WAVES is absent

Null rule: a counter absent from the CSV -> canonical key set to JSON null AND omitted
from `coverage` (NEVER fabricated 0.0).

Usage (invoked WITH a python prefix):
  _evidence_amd.py --native <rocprof.csv|-> --source-backend rocm [--format rocprof-csv]
Prints ONE JSON object on stdout; logs to stderr.
Exit: 0 normalized . 2 native unparseable . 3 bad args.
"""
import sys, json, csv, io, argparse

# Counter name -> canonical key mapping (the subset profile.sh requests by default).
M_SQ_WAVES   = "SQ_WAVES"
M_SQ_VALU    = "SQ_INSTS_VALU"
M_VALUINSTS  = "VALUInsts"
M_VFETCH     = "VFetchInsts"
M_VWRITE     = "VWriteInsts"
M_TCP_ACC    = "TCP_TOTAL_CACHE_ACCESSES_sum"
M_TCP_HIT    = "TCP_TOTAL_CACHE_HITS_sum"
M_GUI_ACTIVE = "GRBM_GUI_ACTIVE"
COL_KERNEL   = "KernelName"

# Conservative defaults; real silicon should override via env. These exist so the
# mapper can produce a *bounded* canonical value rather than null when only raw
# counters are available. None of these are silently inserted -- they only scale
# counters that ARE present in the CSV.
PEAK_VALU_PER_WAVE = 100_000   # arbitrary unit; only ratio matters for sm_pct heuristic
PEAK_WAVES_PER_CU  = 32        # gfx1151 RDNA3.5 wave32 ceiling

CANONICAL_KEYS = ("latency_ms", "dram_pct", "sm_pct", "occupancy")


class NativeParseError(Exception):
    pass


def _parse_float(raw):
    if raw is None:
        raise ValueError("empty value")
    s = str(raw).strip().replace(",", "")
    if not s:
        raise ValueError("empty value")
    token = s.split()[0]
    return float(token)


def _parse_rocprof_csv(text):
    """Parse rocprof/rocprofv3 CSV. Returns (first_kernel, metrics_dict).
    metrics_dict aggregates over all rows for the first kernel via summation
    (rocprof emits one row per dispatch; KerSor wants the per-kernel total).
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise NativeParseError("empty native file (no header row)")
    have = set(reader.fieldnames)
    if COL_KERNEL not in have:
        # rocprofv3 sometimes uses "Kernel_Name" or "kernel_name"; accept variants.
        alt = next((c for c in reader.fieldnames if c.lower().replace("_", "") == "kernelname"), None)
        if alt is None:
            raise NativeParseError(
                f"native CSV missing kernel-name column; saw columns {reader.fieldnames}")
        kernel_col = alt
    else:
        kernel_col = COL_KERNEL

    first_kernel = None
    metrics = {}
    begin_ns = None
    end_ns = None
    rows_seen = 0
    for row in reader:
        kernel = (row.get(kernel_col) or "").strip()
        if not kernel:
            continue
        if first_kernel is None:
            first_kernel = kernel
        elif kernel != first_kernel:
            continue
        rows_seen += 1
        # Optional timing columns (rocprofv3 'BeginNs' / 'EndNs').
        for ts_col, holder in (("BeginNs", "begin_ns"), ("EndNs", "end_ns")):
            v = row.get(ts_col)
            if v is None:
                continue
            try:
                t = _parse_float(v)
                if holder == "begin_ns":
                    begin_ns = t if begin_ns is None else min(begin_ns, t)
                else:
                    end_ns = t if end_ns is None else max(end_ns, t)
            except (ValueError, TypeError):
                pass
        for col, val in row.items():
            if col in (kernel_col, "BeginNs", "EndNs", "DispatchNs", "CompleteNs", "Index"):
                continue
            try:
                f = _parse_float(val)
            except (ValueError, TypeError):
                continue
            metrics[col] = metrics.get(col, 0.0) + f

    if first_kernel is None or rows_seen == 0:
        raise NativeParseError("native CSV had a valid header but no parseable kernel rows")

    if begin_ns is not None and end_ns is not None and end_ns > begin_ns:
        metrics["_latency_ns"] = end_ns - begin_ns
    return first_kernel, metrics


def to_canonical(native_metrics, source_backend):
    latency_ms = None
    if "_latency_ns" in native_metrics:
        latency_ms = native_metrics["_latency_ns"] / 1e6

    # sm_pct heuristic: SQ_INSTS_VALU / (SQ_WAVES * peak) ratio, scaled to 0-100.
    sm_pct = None
    waves = native_metrics.get(M_SQ_WAVES)
    valu = native_metrics.get(M_SQ_VALU) or native_metrics.get(M_VALUINSTS)
    if waves and waves > 0 and valu is not None:
        sm_pct = min(100.0, (valu / (waves * PEAK_VALU_PER_WAVE)) * 100.0)

    # occupancy: SQ_WAVES / peak (per kernel), clamped.
    occupancy = None
    if waves is not None:
        occupancy = max(0.0, min(1.0, waves / float(PEAK_WAVES_PER_CU)))

    # dram_pct: no native counter in the default set; require explicit DRAM counters.
    # If TCP cache accesses/hits are present, derive l1_hit_pct into backend_native.
    dram_pct = None

    canonical = {
        "latency_ms": latency_ms,
        "dram_pct": dram_pct,
        "sm_pct": sm_pct,
        "occupancy": occupancy,
    }
    coverage = [k for k in CANONICAL_KEYS if canonical[k] is not None]

    backend_native = dict(native_metrics)
    backend_native.pop("_latency_ns", None)
    # Derived (kept under backend_native, NOT promoted to canonical to avoid scope creep):
    acc = native_metrics.get(M_TCP_ACC)
    hit = native_metrics.get(M_TCP_HIT)
    if acc and acc > 0 and hit is not None:
        backend_native["l1_hit_pct"] = (hit / acc) * 100.0

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
    ap = argparse.ArgumentParser(description="AMD rocprof-CSV -> canonical metrics")
    ap.add_argument("--native", required=True, help="rocprof CSV path, or '-' for stdin")
    ap.add_argument("--source-backend", dest="source_backend", default=None,
                    help="backend id stamped into source_backend (rocm)")
    ap.add_argument("--format", default="rocprof-csv",
                    help="native profile format (only rocprof-csv supported)")
    ap.add_argument("--run", default=None, help="optional run.sh result.json (reserved)")
    a = ap.parse_args(argv)

    sb = a.source_backend or source_backend
    if not sb:
        print(json.dumps({"ok": False, "error": "missing --source-backend"}, ensure_ascii=False))
        return 3
    if a.format != "rocprof-csv":
        print(json.dumps({"ok": False, "error": f"unsupported format {a.format}"}, ensure_ascii=False))
        return 3

    try:
        text = _read_native(a.native)
    except OSError as exc:
        print(json.dumps({"ok": False, "error": f"cannot read native file: {exc}"}, ensure_ascii=False))
        return 3
    except NativeParseError as exc:
        print(json.dumps({"ok": False, "error": f"native unparseable: {exc}"}, ensure_ascii=False))
        print(f"[_evidence_amd] parse error: {exc}", file=sys.stderr)
        return 2

    try:
        _kernel, native_metrics = _parse_rocprof_csv(text)
    except NativeParseError as exc:
        print(json.dumps({"ok": False, "error": f"native unparseable: {exc}"}, ensure_ascii=False))
        print(f"[_evidence_amd] parse error: {exc}", file=sys.stderr)
        return 2

    result = to_canonical(native_metrics, sb)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
