#!/usr/bin/env python3
"""generic/to_evidence.py — generic substrate "profile" -> canonical metrics.

There is NO vendor profiler on the generic fallback plane, so every hardware
counter is honestly null:
  - dram_pct:  null (no memory-bandwidth counter)
  - sm_pct:    null (no compute-utilization counter)
  - occupancy: null (no occupancy counter)
  - latency_ms: wall-clock latency, IF a run result is supplied via --run

The decided null rule (SDK: absent counter = null, never fabricated 0.0) makes
diagnose.py short-circuit to bottleneck 'unknown' (its all-null guard) rather
than misapplying vendor thresholds to non-vendor hardware. Note this to_evidence
does NOT stamp metrics["_vendor"]: with all counters null there is no threshold
branch to select, so no 'generic' diagnose profile is needed (and none exists).

Invoked WITH a python prefix:
  to_evidence.py --native <path> [--source-backend generic] [--format none]
                  [--run <result.json>]

--native is accepted for invocation-contract uniformity but profile.sh never
produces one (profiler "none"). Latency, when present, comes from --run.
"""
import json, sys, os, argparse


def _latency_from_run(run_path):
    """Best-effort wall-clock latency (ms) from a run.sh result.json."""
    if not run_path or not os.path.exists(run_path):
        return None
    try:
        with open(run_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    val = data.get("candidate_latency_ms")
    return val if isinstance(val, (int, float)) else None


def main(source_backend="generic"):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", required=False, default=None,
                    help="accepted for contract uniformity; unused (profiler 'none')")
    ap.add_argument("--source-backend", default=source_backend)
    ap.add_argument("--format", default=None)
    ap.add_argument("--run", default=None, help="run.sh result.json for latency")
    args = ap.parse_args()

    canonical = {
        "latency_ms": _latency_from_run(args.run),
        "dram_pct": None,
        "sm_pct": None,
        "occupancy": None,
    }
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
