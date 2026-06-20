#!/usr/bin/env python3
"""Degradation path under test: test-backend-ops perf  ->  profile_heuristic evidence.

Proves the claim from the analysis: NCU is just one evidence source. A test/perf
run is *also* analysis. We turn `test-backend-ops perf -o <OP>` stdout into a
bus-legal attempt-evidence record (evidence_schema.py shape) whose bottleneck
insight carries  evidence="profile_heuristic", confidence="inferred"  instead of
evidence="ncu", confidence="measured" -- no NCU anywhere.

Usage:
  perf_to_evidence.py --baseline base.txt [--candidate cand.txt]
                      --peak-gflops 9800 --peak-gbs 800
                      [--strategy-id vec_float4]
Emits the canonical per-attempt record on stdout. With --candidate it also
computes speedup and tags the attempt validated_win / failed_strategy, then
prints the channel-3 `attempt_evidence` object AccelOpt actually consumes.
"""
import sys, re, json, argparse

ANSI = re.compile(r"\033\[[0-9;]*m")
LINE = re.compile(r"^\s*(\w+)\(([^)]*)\):\s+(\d+)\s+runs\s+-\s+([\d.]+)\s+us/run")
FLOPS = re.compile(r"-\s+([\d.]+)\s+([kMGT])FLOPS")          # compute throughput
BW = re.compile(r"-\s+([\d.]+)\s+GB/s")                       # memory throughput
FLOP_MUL = {"k": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3}        # -> GFLOPS


def parse_perf(text):
    """Return list of {op, params, us, gflops|None, gbs|None}."""
    rows = []
    for raw in text.splitlines():
        s = ANSI.sub("", raw)
        m = LINE.search(s)
        if not m:
            continue
        op, params, _runs, us = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
        f, b = FLOPS.search(s), BW.search(s)
        rows.append({
            "op": op, "params": params, "us": us,
            "gflops": (float(f.group(1)) * FLOP_MUL[f.group(2)]) if f else None,
            "gbs": float(b.group(1)) if b else None,
        })
    return rows


def classify(row, peak_gflops, peak_gbs):
    """roofline-lite: achieved fraction of peak on whichever axis perf reported."""
    if row["gbs"] is not None:
        util = row["gbs"] / peak_gbs
        if util >= 0.50:
            return "memory_bound", {"dram_pct": round(util * 100, 1), "sm_pct": None}, \
                   f"achieves {row['gbs']:.0f} GB/s = {util*100:.0f}% of {peak_gbs:.0f} GB/s peak -> DRAM-bandwidth bound"
        return "latency_bound", {"dram_pct": round(util * 100, 1), "sm_pct": None}, \
               f"only {util*100:.0f}% of peak bandwidth at {row['us']:.1f}us/run -> latency/occupancy bound, not saturating DRAM"
    if row["gflops"] is not None:
        util = row["gflops"] / peak_gflops
        if util >= 0.50:
            return "compute_bound", {"dram_pct": None, "sm_pct": round(util * 100, 1)}, \
                   f"achieves {row['gflops']:.0f} GFLOPS = {util*100:.0f}% of {peak_gflops:.0f} GFLOPS peak -> compute bound"
        return "latency_bound", {"dram_pct": None, "sm_pct": round(util * 100, 1)}, \
               f"only {util*100:.0f}% of peak FLOPS -> latency/occupancy bound, not saturating SMs"
    return "latency_bound", {"dram_pct": None, "sm_pct": None}, \
           f"no throughput axis reported at {row['us']:.1f}us/run -> assume latency bound"


def evidence_record(row, peak_gflops, peak_gbs, *, attempt_id, parent_id,
                    speedup, correct, strategy_id, evaluated):
    bclass, metrics, claim = classify(row, peak_gflops, peak_gbs)
    metrics = {"latency_ms": round(row["us"] / 1000.0, 6), "occupancy": None, **metrics}
    insights = [{
        "kind": "bottleneck", "directive": "explore",
        "evidence": "profile_heuristic",   # <-- the whole point: NOT "ncu"
        "confidence": "inferred",          # <-- degraded from "measured", still bus-legal
        "claim": f"{row['op']}({row['params']}): {claim}",
    }]
    failed = []
    if evaluated:
        if correct and speedup is not None and speedup > 1.0:
            insights.append({"kind": "validated_win", "directive": "reuse",
                             "evidence": "benchmark", "confidence": "measured",
                             "claim": f"strategy '{strategy_id}' gave {speedup:.2f}x (test-backend-ops perf, correctness PASS)"})
        else:
            why = "correctness FAIL" if not correct else f"no speedup ({speedup:.2f}x)"
            insights.append({"kind": "failed_strategy", "directive": "avoid",
                             "evidence": "benchmark", "confidence": "measured",
                             "claim": f"strategy '{strategy_id}' rejected: {why}"})
            failed.append({"id": strategy_id, "kind": "failed_strategy",
                           "directive": "avoid", "evidence": "benchmark",
                           "confidence": "measured",
                           "claim": f"strategy '{strategy_id}' rejected: {why}"})
    return {
        "attempt_id": attempt_id, "parent_id": parent_id,
        "compiled": True, "correct": correct, "speedup": speedup,
        "metrics": metrics, "best_kernel": None,
        "convergence_status": "unknown", "insights": insights,
        "failed_strategies": failed,
    }


def to_channel3(rec, strategy_id):
    """AccelOpt channel-3 shape: transfer_items[] with kind='failed_strategy' -> hard constraint."""
    items = [{"id": (i.get("claim", "")[:40]), "kind": i["kind"], "directive": i["directive"],
              "evidence": i["evidence"], "claim": i["claim"]} for i in rec["insights"]]
    for fs in rec["failed_strategies"]:
        items.append({"id": fs["id"], "kind": "failed_strategy", "directive": "avoid",
                      "evidence": "benchmark", "claim": fs["claim"]})
    return {"round": 2, "source": "test-backend-ops(perf+test); no NCU",
            "metrics": rec["metrics"], "transfer_items": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate")
    ap.add_argument("--peak-gflops", type=float, required=True)
    ap.add_argument("--peak-gbs", type=float, required=True)
    ap.add_argument("--strategy-id", default="candidate_v1")
    ap.add_argument("--correct", choices=["pass", "fail"], default="pass",
                    help="verdict from the correctness_command (test-backend-ops test mode)")
    a = ap.parse_args()

    base = parse_perf(open(a.baseline).read())
    if not base:
        sys.exit("no perf rows parsed from baseline")
    brow = max(base, key=lambda r: r["us"])  # worst variant = the one worth optimizing

    speedup = correct = None
    crow = brow
    if a.candidate:
        cand = parse_perf(open(a.candidate).read())
        # match the same op/params if present, else worst
        crow = next((r for r in cand if r["op"] == brow["op"] and r["params"] == brow["params"]),
                    max(cand, key=lambda r: r["us"]))
        correct = (a.correct == "pass")  # real verdict from correctness_command (test mode)
        # a kernel that fails correctness has no meaningful speedup (bus contract:
        # an incorrect attempt may not claim speedup > 1.0)
        speedup = round(brow["us"] / crow["us"], 4) if correct else None

    rec = evidence_record(crow if a.candidate else brow, a.peak_gflops, a.peak_gbs,
                          attempt_id="a2" if a.candidate else "a1",
                          parent_id="a1" if a.candidate else None,
                          speedup=speedup, correct=(correct if a.candidate else True),
                          strategy_id=a.strategy_id, evaluated=bool(a.candidate))
    print(json.dumps(rec, indent=2))
    if a.candidate:
        # pure JSON to stderr so it can be redirected to a consumable file
        sys.stderr.write(json.dumps(to_channel3(rec, a.strategy_id), indent=2) + "\n")


if __name__ == "__main__":
    main()
