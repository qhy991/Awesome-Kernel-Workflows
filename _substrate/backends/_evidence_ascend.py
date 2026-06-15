#!/usr/bin/env python3
"""Shared Ascend msprof-CSV -> canonical-metrics mapper (ascend backend).

Mirrors the spec of _evidence_nvidia.py (cuda) and _evidence_amd.py (rocm). Single source
of truth for Ascend msprof counter -> canonical-key mapping. msprof's op_summary /
kernel_details CSV carries one row per kernel/op with per-unit utilization columns.

IMPORTANT: the exact msprof CSV column identifiers live in the CANN SDK (not in this repo),
and they have drifted across CANN releases. So this mapper matches columns by a NORMALIZED
name (lowercased, stripped of spaces/underscores/%/parens) against an alias list, and applies
the decided null rule: a metric whose source column is absent -> canonical key set to JSON
null AND omitted from `coverage` (NEVER a fabricated 0.0).

Canonical units emitted (every backend MUST honor):
  latency_ms = aicore/task duration -> ms (us/1e3 by default; ns/1e6 if the column says ns)
  dram_pct   = GM/HBM (MTE) bandwidth utilization, 0-100; null if no bandwidth column
  sm_pct     = AI Core compute utilization = max(cube_util, vector_util), 0-100;
               null if neither cube nor vector utilization column is present
  occupancy  = ALWAYS null  -- Ascend has no warp-occupancy counter; latency_occupancy is
               therefore never concluded for this backend (manifest omits the capability)

Usage (invoked WITH a python prefix):
  _evidence_ascend.py --native <msprof.csv|-> --source-backend ascend [--format msprof-csv]
Prints ONE JSON object on stdout; logs to stderr.
Exit: 0 normalized . 2 native unparseable . 3 bad args.
"""
import sys, json, csv, io, argparse

CANONICAL_KEYS = ("latency_ms", "dram_pct", "sm_pct", "occupancy")

# Normalized-name alias lists. A column matches if its normalized form is in the set.
# Duration columns carry a unit suffix (e.g. "aicore_time(us)" -> "aicoretimeus"); the unit is
# stripped and detected separately in _pick_duration, so these are the UNIT-LESS base names.
DURATION_ALIASES = {"aicoretime", "taskduration", "duration", "totaltime", "aivtime",
                    "aictime", "tasktime", "time", "totalcycles"}
DRAM_ALIASES = {"gmbandwidth", "gmbandwidthpercent", "hbmbandwidth", "mtebandwidth",
                "mtebound", "memorybandwidth", "membound", "mte2bandwidth", "mte1bandwidth"}
CUBE_ALIASES = {"cubeutilization", "cuberatio", "macratio", "cubeutil", "mac",
                "cubeutilrate", "aicutilization"}
VEC_ALIASES = {"vectorutilization", "vecratio", "vectorratio", "vecutil", "aivvecratio",
               "vectorutil", "aivutilization"}


class NativeParseError(Exception):
    pass


def _norm(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _parse_float(raw):
    if raw is None:
        raise ValueError("empty value")
    s = str(raw).strip().replace(",", "").replace("%", "")
    if not s:
        raise ValueError("empty value")
    return float(s.split()[0])


def _pct(value):
    """Normalize a utilization figure to 0-100. Values in [0,1] are treated as ratios."""
    if value is None:
        return None
    v = float(value)
    if 0.0 <= v <= 1.0:
        v *= 100.0
    return max(0.0, min(100.0, v))


def _parse_msprof_csv(text):
    """Parse an msprof op_summary/kernel_details CSV. Returns (first_op, {norm_col: (value, raw_col)}).
    Aggregates duration by SUM and utilization columns by taking the first row's value for the
    first op (msprof emits one row per op; we report the dominant/first op)."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise NativeParseError("empty native file (no header row)")
    # locate an op/kernel name column
    name_col = None
    for c in reader.fieldnames:
        if _norm(c) in ("opname", "kernelname", "name", "optype", "opname0"):
            name_col = c
            break

    first_op = None
    cols = {}        # norm_col -> accumulated float
    raw_names = {}   # norm_col -> original column name
    rows_seen = 0
    for row in reader:
        op = (row.get(name_col) or "").strip() if name_col else f"row{rows_seen}"
        if name_col and not op:
            continue
        if first_op is None:
            first_op = op
        elif name_col and op != first_op:
            continue
        rows_seen += 1
        for col, val in row.items():
            if col == name_col:
                continue
            try:
                f = _parse_float(val)
            except (ValueError, TypeError):
                continue
            nc = _norm(col)
            cols[nc] = cols.get(nc, 0.0) + f
            raw_names.setdefault(nc, col)

    if rows_seen == 0:
        raise NativeParseError("native CSV had a header but no parseable rows")
    return first_op or "kernel", cols, raw_names


def _pick(cols, aliases):
    for nc, v in cols.items():
        if nc in aliases:
            return v, nc
    return None, None


def _pick_duration(cols):
    """Find a duration column (unit suffix tolerated) and return latency in MILLISECONDS.
    Unit precedence: a trailing 'ns'/'us'/'ms' on the column name; default 'us' (msprof norm)."""
    units = {"ns": 1e-6, "us": 1e-3, "ms": 1.0}
    for nc, v in cols.items():
        base, scale = nc, units["us"]   # default: microseconds
        for suffix, s in units.items():
            if nc.endswith(suffix) and nc[:-len(suffix)] in DURATION_ALIASES:
                base, scale = nc[:-len(suffix)], s
                break
        if base in DURATION_ALIASES:
            return v * scale
    return None


def to_canonical(cols, raw_names, source_backend):
    # latency (unit-suffix tolerant; returns ms directly)
    latency_ms = _pick_duration(cols)

    # dram_pct from MTE/GM/HBM bandwidth utilization
    dram_raw, _ = _pick(cols, DRAM_ALIASES)
    dram_pct = _pct(dram_raw)

    # sm_pct = AI Core compute utilization = max(cube, vector) when measured
    cube_raw, _ = _pick(cols, CUBE_ALIASES)
    vec_raw, _ = _pick(cols, VEC_ALIASES)
    cube_pct = _pct(cube_raw)
    vec_pct = _pct(vec_raw)
    sm_candidates = [x for x in (cube_pct, vec_pct) if x is not None]
    sm_pct = max(sm_candidates) if sm_candidates else None

    # occupancy: never measured on Ascend
    occupancy = None

    canonical = {
        "latency_ms": latency_ms,
        "dram_pct": dram_pct,
        "sm_pct": sm_pct,
        "occupancy": occupancy,
    }
    coverage = [k for k in CANONICAL_KEYS if canonical[k] is not None]

    backend_native = {raw_names.get(nc, nc): v for nc, v in cols.items()}
    if cube_pct is not None:
        backend_native["cube_util_pct"] = cube_pct
    if vec_pct is not None:
        backend_native["vector_util_pct"] = vec_pct

    metrics = dict(canonical)
    metrics["_vendor"] = "ascend"
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
    ap = argparse.ArgumentParser(description="Ascend msprof-CSV -> canonical metrics")
    ap.add_argument("--native", required=True, help="msprof CSV path, or '-' for stdin")
    ap.add_argument("--source-backend", dest="source_backend", default=None,
                    help="backend id stamped into source_backend (ascend)")
    ap.add_argument("--format", default="msprof-csv",
                    help="native profile format (only msprof-csv supported)")
    ap.add_argument("--run", default=None, help="optional run.sh result.json (reserved)")
    a = ap.parse_args(argv)

    sb = a.source_backend or source_backend
    if not sb:
        print(json.dumps({"ok": False, "error": "missing --source-backend"}, ensure_ascii=False))
        return 3
    if a.format != "msprof-csv":
        print(json.dumps({"ok": False, "error": f"unsupported format {a.format}"}, ensure_ascii=False))
        return 3

    try:
        text = _read_native(a.native)
    except OSError as exc:
        print(json.dumps({"ok": False, "error": f"cannot read native file: {exc}"}, ensure_ascii=False))
        return 3
    except NativeParseError as exc:
        print(json.dumps({"ok": False, "error": f"native unparseable: {exc}"}, ensure_ascii=False))
        print(f"[_evidence_ascend] parse error: {exc}", file=sys.stderr)
        return 2

    try:
        _op, cols, raw_names = _parse_msprof_csv(text)
    except NativeParseError as exc:
        print(json.dumps({"ok": False, "error": f"native unparseable: {exc}"}, ensure_ascii=False))
        print(f"[_evidence_ascend] parse error: {exc}", file=sys.stderr)
        return 2

    result = to_canonical(cols, raw_names, sb)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
