#!/usr/bin/env python3
"""Summarize a sol-execbench bench.jsonl: geomean of PASSED per-workload speedup_factor.

The file has non-JSON header/progress lines before the records; skip anything that
does not parse as a JSON object. Each record: evaluation.status ("PASSED"/other) and
evaluation.performance.speedup_factor. Verified schema (kersor-20260624-060316).
"""
import argparse
import json
import math
import os
import sys
import tempfile


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
    geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else None
    status = "PASS" if total > 0 and passed == total else "FAIL"
    return {"speedup": geo, "status": status, "passed": passed, "total": total}


def write_normalized(path, source, summary):
    payload = {
        "schema_version": 1,
        "source_bench_jsonl": os.path.abspath(source),
        "compiled": summary["total"] > 0,
        "correct": summary["status"] == "PASS",
        "geomean_speedup": summary["speedup"] if summary["speedup"] is not None else 0.0,
        "n_pass": summary["passed"],
        "n_total": summary["total"],
    }
    out = os.path.abspath(path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(out)}.", dir=os.path.dirname(out))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_jsonl")
    ap.add_argument("--out", help="atomically write canonical measurement JSON")
    args = ap.parse_args()
    s = summarize(args.bench_jsonl)
    if args.out:
        write_normalized(args.out, args.bench_jsonl, s)
    sp = f"{s['speedup']:.12g}" if s["speedup"] is not None else "null"
    print(f"SPEEDUP={sp} STATUS={s['status']} WORKLOADS={s['passed']}/{s['total']}")
    return 0 if s["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
