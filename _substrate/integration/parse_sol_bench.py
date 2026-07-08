#!/usr/bin/env python3
"""Summarize a sol-execbench bench.jsonl: geomean of PASSED per-workload speedup_factor.

The file has non-JSON header/progress lines before the records; skip anything that
does not parse as a JSON object. Each record: evaluation.status ("PASSED"/other) and
evaluation.performance.speedup_factor. Verified schema (kersor-20260624-060316).
"""
import argparse
import json
import math
import sys


def summarize(path):
    speedups, passed, total = [], 0, 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = rec.get("evaluation")
            if not isinstance(ev, dict):
                continue
            total += 1
            if ev.get("status") == "PASSED":
                sf = ((ev.get("performance") or {}).get("speedup_factor"))
                if isinstance(sf, (int, float)) and sf > 0:
                    speedups.append(float(sf))
                    passed += 1
    if not speedups:
        return {"speedup": None, "status": "FAIL", "passed": 0, "total": total}
    geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    return {"speedup": geo, "status": "PASS", "passed": passed, "total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_jsonl")
    args = ap.parse_args()
    s = summarize(args.bench_jsonl)
    sp = f"{s['speedup']:.3f}" if s["speedup"] is not None else "null"
    print(f"SPEEDUP={sp} STATUS={s['status']} WORKLOADS={s['passed']}/{s['total']}")
    return 0 if s["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())