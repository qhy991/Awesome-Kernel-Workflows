#!/usr/bin/env python3
"""ncu_parse.py <details.csv> - curated NCU details-page CSV -> key-metrics JSON.

Consumes the CSV produced by `ncu --import <rep> --csv --page details` and
extracts a curated, canonical metric set. Output JSON:

  {
    "kernel_count": N,
    "total_duration_us": float,
    "key_metrics": { ...canonical keys for the DOMINANT (longest) kernel... },
    "per_kernel": [ {"id":.., "name":.., "metrics": {...}}, ... ]
  }

Canonical keys are the single naming source for the analyst agent, the
fingerprint, and the mock tools. Unknown metrics are ignored.
"""
import csv
import hashlib
import json
import sys

# (Metric Name, unit class) -> canonical key.  unit class: 'pct' (unit is %),
# 'time' (ns/us/ms -> us), 'bytes' (byte/Kbyte/Mbyte -> bytes),
# 'rate' (byte/second family -> Gbyte/second), None (take raw number).
MAP = {
    ("Compute (SM) Throughput", "pct"): "sm_pct",
    ("Memory Throughput", "pct"): "mem_pct",
    ("DRAM Throughput", "pct"): "dram_pct",
    ("L1/TEX Cache Throughput", "pct"): "l1_pct",
    ("L2 Cache Throughput", "pct"): "l2_pct",
    ("Duration", "time"): "duration_us",
    ("Elapsed Cycles", None): "elapsed_cycles",
    ("L2 Hit Rate", "pct"): "l2_hit_pct",
    ("Memory Throughput", "rate"): "dram_gbps",
    ("Eligible Warps Per Scheduler", None): "eligible_warps_per_scheduler",
    ("Issued Warp Per Scheduler", None): "issued_warp_per_scheduler",
    ("Active Warps Per Scheduler", None): "active_warps_per_scheduler",
    ("Warp Cycles Per Issued Instruction", None): "warp_cycles_per_issued_inst",
    ("Achieved Occupancy", "pct"): "occupancy_pct",
    ("Theoretical Occupancy", "pct"): "theoretical_occupancy_pct",
    ("Registers Per Thread", None): "regs_per_thread",
    ("Static Shared Memory Per Block", "bytes"): "smem_static_bytes",
    ("Dynamic Shared Memory Per Block", "bytes"): "smem_dynamic_bytes",
    ("Grid Size", None): "grid_size",
    ("Block Size", None): "block_size",
    ("Waves Per SM", None): "waves_per_sm",
}

TIME_TO_US = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0,
              "ms": 1e3, "msecond": 1e3, "s": 1e6, "second": 1e6}
BYTES_MULT = {"byte": 1, "byte/block": 1, "kbyte": 1024, "kbyte/block": 1024,
              "mbyte": 1024**2, "mbyte/block": 1024**2}
RATE_TO_GBPS = {"byte/second": 1e-9, "kbyte/second": 1e-6, "mbyte/second": 1e-3,
                "gbyte/second": 1.0, "tbyte/second": 1e3}


def unit_class(unit):
    u = (unit or "").strip().lower()
    if u == "%":
        return "pct"
    if u in TIME_TO_US:
        return "time"
    if u in BYTES_MULT:
        return "bytes"
    if u in RATE_TO_GBPS:
        return "rate"
    return None


def to_number(raw):
    s = str(raw).strip().replace(",", "")
    if not s:
        return None
    try:
        f = float(s)
        return int(f) if f.is_integer() and abs(f) < 1e15 else f
    except ValueError:
        return None


def convert(value, unit, cls):
    u = (unit or "").strip().lower()
    if cls == "time":
        return value * TIME_TO_US.get(u, 1.0)
    if cls == "bytes":
        return value * BYTES_MULT.get(u, 1)
    if cls == "rate":
        return value * RATE_TO_GBPS.get(u, 1.0)
    return value


def parse(path):
    kernels = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Metric Name") or "").strip()
            unit = (row.get("Metric Unit") or "").strip()
            cls = unit_class(unit)
            key = MAP.get((name, cls)) or MAP.get((name, None))
            if key is None:
                continue
            val = to_number(row.get("Metric Value"))
            if val is None:
                continue
            kid = (row.get("ID") or "0").strip()
            k = kernels.setdefault(kid, {"id": kid,
                                         "name": (row.get("Kernel Name") or "").strip(),
                                         "metrics": {}})
            k["metrics"][key] = convert(val, unit, cls if (name, cls) in MAP else None)
    return list(kernels.values())


def main(argv):
    if len(argv) != 1:
        sys.stderr.write("usage: ncu_parse.py <details.csv>\n")
        return 2
    per_kernel = parse(argv[0])
    if not per_kernel:
        json.dump({"kernel_count": 0, "total_duration_us": None,
                   "key_metrics": {}, "per_kernel": []}, sys.stdout)
        sys.stdout.write("\n")
        return 0
    durations = [(k["metrics"].get("duration_us") or 0.0) for k in per_kernel]
    dominant = per_kernel[max(range(len(per_kernel)), key=lambda i: durations[i])]
    km = dict(dominant["metrics"])
    km["smem_total_bytes"] = (km.get("smem_static_bytes", 0) or 0) + (km.get("smem_dynamic_bytes", 0) or 0)
    out = {
        "kernel_count": len(per_kernel),
        "total_duration_us": round(sum(durations), 3),
        "dominant_kernel": dominant["name"],
        "key_metrics": km,
        "per_kernel": per_kernel,
    }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
