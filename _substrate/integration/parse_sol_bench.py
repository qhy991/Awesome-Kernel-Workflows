#!/usr/bin/env python3
"""Summarize a sol-execbench bench.jsonl using the task contract's reduction.

The file has non-JSON header/progress lines before the records; skip anything that
does not parse as a JSON object. Each record: evaluation.status ("PASSED"/other) and
evaluation.performance latency fields. Verified schema (kersor-20260624-060316).
"""
import argparse
import json
import math
import os
import sys
import tempfile


SUPPORTED_REDUCTIONS = {"sum", "mean", "geomean"}


def contract_reduction(path):
    if not path:
        return "geomean"
    values = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    reduction = values.get("aggregate_reduction", "geomean")
    if reduction not in SUPPORTED_REDUCTIONS:
        raise ValueError(
            f"unsupported aggregate_reduction={reduction!r}; "
            f"expected one of {sorted(SUPPORTED_REDUCTIONS)}"
        )
    return reduction


def summarize(path, reduction="geomean"):
    rows, passed, total = [], 0, 0
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
                perf = ev.get("performance") or {}
                sf = perf.get("speedup_factor")
                candidate = perf.get("latency_ms")
                reference = perf.get("reference_latency_ms")
                valid_speedup = isinstance(sf, (int, float)) and sf > 0
                valid_latencies = (
                    isinstance(candidate, (int, float)) and candidate > 0
                    and isinstance(reference, (int, float)) and reference > 0
                )
                if (reduction == "geomean" and valid_speedup) or (
                    reduction in {"sum", "mean"} and valid_latencies
                ):
                    rows.append((
                        float(sf) if valid_speedup else None,
                        float(reference) if valid_latencies else None,
                        float(candidate) if valid_latencies else None,
                    ))
                    passed += 1
    speedup = None
    reference_aggregate = None
    candidate_aggregate = None
    if rows:
        if reduction == "geomean":
            speedup = math.exp(sum(math.log(row[0]) for row in rows) / len(rows))
        else:
            reference_aggregate = sum(row[1] for row in rows)
            candidate_aggregate = sum(row[2] for row in rows)
            if reduction == "mean":
                reference_aggregate /= len(rows)
                candidate_aggregate /= len(rows)
            speedup = reference_aggregate / candidate_aggregate
    status = "PASS" if total > 0 and passed == total else "FAIL"
    return {
        "speedup": speedup,
        "status": status,
        "passed": passed,
        "total": total,
        "aggregate_reduction": reduction,
        "reference_latency_aggregate_ms": reference_aggregate,
        "candidate_latency_aggregate_ms": candidate_aggregate,
    }


def write_normalized(path, source, summary):
    payload = {
        "schema_version": 1,
        "source_bench_jsonl": os.path.abspath(source),
        "compiled": summary["total"] > 0,
        "correct": summary["status"] == "PASS",
        "metric_name": "speedup",
        "aggregate_reduction": summary["aggregate_reduction"],
        "speedup": summary["speedup"] if summary["speedup"] is not None else 0.0,
        "reference_latency_aggregate_ms": summary["reference_latency_aggregate_ms"],
        "candidate_latency_aggregate_ms": summary["candidate_latency_aggregate_ms"],
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
    ap.add_argument("--contract", help="contract.env containing aggregate_reduction")
    ap.add_argument("--out", help="atomically write canonical measurement JSON")
    args = ap.parse_args()
    try:
        reduction = contract_reduction(args.contract)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    s = summarize(args.bench_jsonl, reduction)
    if args.out:
        write_normalized(args.out, args.bench_jsonl, s)
    sp = f"{s['speedup']:.12g}" if s["speedup"] is not None else "null"
    print(
        f"SPEEDUP={sp} REDUCTION={s['aggregate_reduction']} "
        f"STATUS={s['status']} WORKLOADS={s['passed']}/{s['total']}"
    )
    return 0 if s["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
