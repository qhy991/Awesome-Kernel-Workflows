#!/usr/bin/env python3
"""Layer C — diagnosis: profile metrics -> shared bottleneck taxonomy.

Borrowed from AccelOpt / KernelBand (phi features) / KernelBlaster (state class)
/ cuPilot (roofline). Deterministic thresholds. The canonical bottleneck_class
enum is shared across all solvers so memory keys and routing features line up.

Usage:
  diagnose.py --metrics metrics.json   # {dram_pct, sm_pct, occupancy, latency_ms, problem_size?}
Prints: {bottleneck_class, evidence}
bottleneck_class in {memory_bound, compute_bound, latency_occupancy, overhead_bound, unknown}
"""
import sys, json, argparse

CLASSES = ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound", "unknown"]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True)
    a = ap.parse_args()
    m = json.loads(sys.stdin.read() if a.metrics == "-" else open(a.metrics).read())
    cls, ev = classify(m)
    print(json.dumps({"bottleneck_class": cls, "evidence": ev}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
