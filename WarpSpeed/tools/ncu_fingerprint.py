#!/usr/bin/env python3
"""ncu_fingerprint.py <parsed.json> - quantize curated NCU metrics into a
stable fingerprint hash (dedup key #2 in the WarpSpeed checkpoint table).

Bucketing intentionally coarse: two kernels whose profiles land in the same
buckets AND whose latencies agree within noise are treated as the same
structural state. Latency itself is NOT part of the fingerprint (it is the
other half of the hard-dedup conjunction).

Output: {"fingerprint": "<sha1[:12]>", "buckets": {...}}
"""
import hashlib
import json
import sys

# key -> bucket width (None = exact value)
BUCKETS = {
    "sm_pct": 2.5,
    "mem_pct": 2.5,
    "dram_pct": 2.5,
    "l2_hit_pct": 5.0,
    "occupancy_pct": 5.0,
    "warp_cycles_per_issued_inst": 2.0,
    "eligible_warps_per_scheduler": 0.5,
    "regs_per_thread": 8,
    "smem_total_bytes": 8192,
    "grid_size": None,
    "block_size": None,
}


def main(argv):
    if len(argv) != 1:
        sys.stderr.write("usage: ncu_fingerprint.py <parsed.json>\n")
        return 2
    parsed = json.load(open(argv[0]))
    km = parsed.get("key_metrics", {})
    buckets = {"kernel_count": parsed.get("kernel_count")}
    for key, width in sorted(BUCKETS.items()):
        v = km.get(key)
        if v is None:
            buckets[key] = None
        elif width is None:
            buckets[key] = v
        else:
            buckets[key] = int(float(v) // width)
    blob = json.dumps(buckets, sort_keys=True).encode()
    json.dump({"fingerprint": hashlib.sha1(blob).hexdigest()[:12],
               "buckets": buckets}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
