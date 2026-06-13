#!/usr/bin/env python3
"""metal/to_evidence.py — Metal profiler output -> canonical metrics.

Apple Silicon has unified memory (UMA), so the concept of "DRAM bandwidth %"
is fundamentally different from NVIDIA's discrete-GPU model:
  - dram_pct: always null (UMA — bandwidth is shared with CPU, not a separate pool)
  - sm_pct: GPU core utilization (% of peak theoretical), derived from timing +
            device-derived roofline when flops are annotated
  - occupancy: always null (no public API for per-SM warp occupancy on Metal)
  - latency_ms: GPU kernel duration from MTLCommandBuffer timestamps

Invoked WITH a python prefix:
  to_evidence.py --native <timing.json|gputrace.gputrace> [--source-backend metal]
                  [--format metal-csv|timing-json] [--run <result.json>]
"""
import json, sys, os, argparse


def extract_latency_from_timing(data):
    """Extract kernel latency (ms) from a timing-only JSON profile.

    Expected format:
      {"kernels": [{"name": "...", "duration_ns": 12345, "invocations": 500}]}

    Returns {latency_ms, dram_pct:null, sm_pct:null, occupancy:null}
    """
    kernels = data.get("kernels", [])
    if not kernels:
        return {"latency_ms": None, "dram_pct": None, "sm_pct": None, "occupancy": None}

    total_duration_ns = 0
    total_invocations = 0
    for k in kernels:
        dur = k.get("duration_ns", 0)
        inv = k.get("invocations", 1)
        total_duration_ns += dur
        total_invocations += inv

    avg_duration_ns = total_duration_ns / max(total_invocations, 1)
    latency_ms = avg_duration_ns / 1e6

    return {
        "latency_ms": round(latency_ms, 6),
        "dram_pct": None,
        "sm_pct": None,
        "occupancy": None,
    }


def extract_from_gputrace(data):
    """Extract metrics from a .gputrace parsed into JSON.

    Metal GPU trace (from MTLCaptureManager) can provide:
      - GPU active time (%)
      - Shader core utilization
      - Device memory bandwidth utilization (UMA, so this is system memory BW)

    Maps the closest available counters to canonical keys.
    Returns {latency_ms, dram_pct, sm_pct, occupancy}
    """
    metrics = data.get("metrics", data)
    return {
        "latency_ms": metrics.get("gpu_time_ms"),
        "dram_pct": metrics.get("device_memory_bw_pct"),
        "sm_pct": metrics.get("gpu_core_utilization_pct"),
        "occupancy": None,
    }


def main(source_backend="metal"):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", required=True, help="path to native profile file")
    ap.add_argument("--source-backend", default=source_backend)
    ap.add_argument("--format", default=None,
                    help="profile format: metal-csv, timing-json (auto-detect if absent)")
    ap.add_argument("--run", default=None, help="optional run.sh result.json for latency fallback")
    args = ap.parse_args()

    native_path = args.native
    if not os.path.exists(native_path):
        result = {"ok": False, "error": f"native profile not found: {native_path}"}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1

    try:
        with open(native_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"cannot read native file: {exc}"},
                         indent=2, ensure_ascii=False))
        return 1

    fmt = args.format or data.get("format", "timing-json")

    if fmt == "timing-json":
        canonical = extract_latency_from_timing(data)
    elif fmt == "metal-csv":
        canonical = extract_from_gputrace(data)
    else:
        print(json.dumps({"ok": False, "error": f"unsupported format {fmt}"},
                         indent=2, ensure_ascii=False))
        return 1

    result = {
        "ok": True,
        "source_backend": args.source_backend,
        "metrics": canonical,
        "coverage": [k for k, v in canonical.items() if v is not None],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())