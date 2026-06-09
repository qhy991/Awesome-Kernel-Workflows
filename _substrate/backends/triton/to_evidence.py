#!/usr/bin/env python3
"""triton/to_evidence.py — Proton-hatchet -> canonical-metrics mapper (Triton-native).

Triton is NOT profiled by ncu here (ncu mangles the JIT kernel symbol as
`triton_<fn>_<hash>` and needs elevated perf-counter access). Instead the triton driver
uses **Proton** (`triton.profiler`), Triton's own profiler, which instruments at the
Triton level and emits a `.hatchet` JSON. This module maps that JSON to the same four
canonical keys diagnose.py speaks.

Hatchet shape (confirmed against triton 3.6 / proton cupti backend):
  [ <call-tree>, { "CUDA": { "<dev>": {arch, bus_width, clock_rate, memory_clock_rate,
                                       num_sms} } } ]
  call-tree node = { "frame": {"name","type"}, "metrics": {...}, "children": [...] }
    leaf kernel node carries  metrics["time (ns)"] + metrics["count"]
    scope node carries the user-annotated metrics["bytes"] / metrics["flops"]

Canonical mapping (every backend MUST honor the units + null rule):
  latency_ms = (time_ns / count) / 1e6                       # per-invocation device time
  dram_pct   = achieved_BW  / peak_BW   * 100   (>=0; MAY exceed 100, like the cuda path)
  sm_pct     = achieved_FLOP / peak_FLOP * 100
  occupancy  = ALWAYS None  (proton/CUPTI does not report achieved occupancy)

  achieved_BW   = bytes / latency_s         (bytes/flops are PER single invocation)
  achieved_FLOP = flops / latency_s
  peak_BW       = memory_clock_rate(kHz)*1e3 * 2(DDR) * (bus_width/8)        [device-derived]
  peak_FLOP     = num_sms * fp32_cores_per_sm(arch) * 2(FMA) * clock_rate(kHz)*1e3

Null rule: dram_pct / sm_pct are produced ONLY when the proton scope was annotated with
`bytes` / `flops` (the Triton-idiomatic roofline annotation) AND device peaks are derivable;
otherwise they are JSON null and omitted from `coverage` (NEVER fabricated 0.0), so
diagnose.py yields `unknown` rather than a confident wrong label. occupancy is always null,
hence triton declares no `latency_occupancy` capability.

Invoked WITH a python prefix:
  to_evidence.py --native <prof.hatchet|-> [--source-backend triton] [--format proton-hatchet]
Prints ONE JSON object on stdout; logs to stderr.
Exit: 0 normalized · 2 native unparseable (JSON still printed, ok:false) · 3 bad args.
Pure function (same hatchet in -> same JSON out).
"""
import sys, json, argparse

CANONICAL_KEYS = ("latency_ms", "dram_pct", "sm_pct", "occupancy")
SUPPORTED_FORMATS = ("proton-hatchet", "proton-json")

# fp32 ALUs per SM by compute-capability major*10+minor ("arch" string in the hatchet).
# Architectural constants (NOT vendor peak numbers); default 64 for unknown archs.
FP32_CORES_PER_SM = {
    "70": 64, "72": 64, "75": 64,        # Volta / Turing
    "80": 64,                            # A100 (Ampere GA100)
    "86": 128, "87": 128, "89": 128,     # Ampere GA10x / Ada
    "90": 128,                           # Hopper
    "100": 128, "120": 128,              # Blackwell
}


class NativeParseError(Exception):
    """Raised when the native profile cannot be parsed into the expected proton shape."""


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _walk(node, acc):
    """Accumulate kernel time/count leaves and summed bytes/flops over the call tree."""
    if not isinstance(node, dict):
        return
    m = node.get("metrics") or {}
    name = (node.get("frame") or {}).get("name")
    t = _to_float(m.get("time (ns)"))
    cnt = _to_float(m.get("count"))
    if t is not None and cnt and name and name != "ROOT":
        acc["kernels"].append({"name": name, "time_ns": t, "count": cnt})
    b = _to_float(m.get("bytes"))
    if b:
        acc["bytes"] += b
    f = _to_float(m.get("flops"))
    if f:
        acc["flops"] += f
    for child in node.get("children") or []:
        _walk(child, acc)


def _device_info(hatchet):
    """Pull the first CUDA device record from the trailing device block, if present."""
    if not isinstance(hatchet, list):
        return {}
    for entry in hatchet:
        if isinstance(entry, dict) and "CUDA" in entry and isinstance(entry["CUDA"], dict):
            for _dev_id, info in entry["CUDA"].items():
                if isinstance(info, dict):
                    return info
    return {}


def _peak_bw_bytes_per_s(dev):
    mck = _to_float(dev.get("memory_clock_rate"))  # kHz
    bw = _to_float(dev.get("bus_width"))           # bits
    if mck and bw:
        return mck * 1e3 * 2.0 * (bw / 8.0)        # DDR factor 2; bits -> bytes
    return None


def _peak_fp32_flops_per_s(dev):
    sms = _to_float(dev.get("num_sms"))
    clk = _to_float(dev.get("clock_rate"))         # kHz
    if sms and clk:
        cores = FP32_CORES_PER_SM.get(str(dev.get("arch", "")), 64)
        return sms * cores * 2.0 * (clk * 1e3)     # FMA factor 2
    return None


def _parse_proton_hatchet(text):
    """Parse a proton .hatchet JSON string into (latency_ms, dram_pct, sm_pct, native).

    Raises NativeParseError if the JSON is malformed or carries no kernel timing leaf.
    """
    try:
        hatchet = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise NativeParseError(f"native profile is not valid JSON: {exc}") from exc

    tree = hatchet[0] if isinstance(hatchet, list) and hatchet else hatchet
    acc = {"kernels": [], "bytes": 0.0, "flops": 0.0}
    _walk(tree, acc)
    if not acc["kernels"]:
        raise NativeParseError("proton profile carried no kernel timing leaf "
                               "(time (ns) + count); nothing attributed")

    # Dominant kernel = the one with the largest total device time.
    dom = max(acc["kernels"], key=lambda k: k["time_ns"])
    latency_ms = (dom["time_ns"] / dom["count"]) / 1e6
    latency_s = (dom["time_ns"] / dom["count"]) / 1e9

    dev = _device_info(hatchet)
    peak_bw = _peak_bw_bytes_per_s(dev)
    peak_fp = _peak_fp32_flops_per_s(dev)

    achieved_bw = (acc["bytes"] / latency_s) if (acc["bytes"] and latency_s) else None
    achieved_fp = (acc["flops"] / latency_s) if (acc["flops"] and latency_s) else None

    dram_pct = (achieved_bw / peak_bw * 100.0) if (achieved_bw and peak_bw) else None
    sm_pct = (achieved_fp / peak_fp * 100.0) if (achieved_fp and peak_fp) else None

    native = {
        "kernel_name": dom["name"],
        "count": dom["count"],
        "time_ns_total": dom["time_ns"],
        "bytes_per_call": acc["bytes"] or None,
        "flops_per_call": acc["flops"] or None,
        "achieved_gbps": (achieved_bw / 1e9) if achieved_bw else None,
        "achieved_gflops": (achieved_fp / 1e9) if achieved_fp else None,
        "peak_gbps": (peak_bw / 1e9) if peak_bw else None,
        "peak_fp32_gflops": (peak_fp / 1e9) if peak_fp else None,
        "arch": dev.get("arch"),
        "num_sms": dev.get("num_sms"),
        # dram_pct/sm_pct are device-derived roofline ESTIMATES, not hardware counters.
        "estimated": dram_pct is not None or sm_pct is not None,
    }
    return latency_ms, dram_pct, sm_pct, native


def to_canonical(latency_ms, dram_pct, sm_pct, native, source_backend):
    canonical = {
        "latency_ms": latency_ms,
        "dram_pct": dram_pct,
        "sm_pct": sm_pct,
        "occupancy": None,   # proton/CUPTI does not report achieved occupancy
    }
    coverage = [k for k in CANONICAL_KEYS if canonical[k] is not None]
    metrics = dict(canonical)
    metrics["_vendor"] = "nvidia"          # nvidia diagnose.py thresholds still apply
    metrics["backend_native"] = native
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


def main(argv=None, source_backend="triton"):
    ap = argparse.ArgumentParser(description="Proton-hatchet -> canonical metrics (triton)")
    ap.add_argument("--native", required=True, help="proton .hatchet path, or '-' for stdin")
    ap.add_argument("--source-backend", dest="source_backend", default=None,
                    help="backend id stamped into source_backend (default triton)")
    ap.add_argument("--format", default="proton-hatchet",
                    help="native profile format (proton-hatchet | proton-json)")
    ap.add_argument("--run", default=None,
                    help="optional run.sh result.json (reserved; not consumed by the mapper)")
    a = ap.parse_args(argv)

    sb = a.source_backend or source_backend
    if a.format not in SUPPORTED_FORMATS:
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
        latency_ms, dram_pct, sm_pct, native = _parse_proton_hatchet(text)
    except NativeParseError as exc:
        print(json.dumps({"ok": False, "error": f"native unparseable: {exc}"},
                         ensure_ascii=False))
        print(f"[triton/to_evidence] parse error: {exc}", file=sys.stderr)
        return 2

    result = to_canonical(latency_ms, dram_pct, sm_pct, native, sb)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
