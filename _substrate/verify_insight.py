#!/usr/bin/env python3
"""Layer B+ — adversarial insight verification (deterministic confidence rules).

Insight *attribution* ("the bottleneck is long_scoreboard at M∈[144,192]") has no
executable ground truth, so the REFUTATION is an LLM refuter agent. This script
applies the deterministic CONFIDENCE rules on top of that verdict, so the final
confidence tag is reproducible (not LLM discretion):

  1. confidence:measured REQUIRES executable evidence (ncu/benchmark/compile/
     correctness/runtime). Non-executable evidence (profile_heuristic/llm_inferred)
     is capped at 'inferred'.
  2. a refuter verdict {refuted:true} downgrades one level
     (measured -> inferred -> hypothesized).
  3. otherwise unchanged.

Usage:
  verify_insight.py --insight insight.json [--refuted 0|1]
  verify_insight.py --insight - [--refuted 1]   # read insight JSON from stdin
Prints the insight with finalized confidence + a 'verification' trace.
"""
import sys, json, argparse

EXECUTABLE = {"ncu", "native_profiler", "benchmark", "compile", "correctness", "runtime"}
ORDER = ["hypothesized", "inferred", "measured"]  # index = strength


def _idx(conf):
    return ORDER.index(conf) if conf in ORDER else 0


def downgrade(conf, steps=1):
    return ORDER[max(0, _idx(conf) - steps)]


def cap(conf, max_conf):
    return ORDER[min(_idx(conf), _idx(max_conf))]


def verify(insight, refuted):
    trace = []
    conf = insight.get("confidence", "hypothesized")
    if conf not in ORDER:
        conf = "hypothesized"
    ev = insight.get("evidence", "llm_inferred")
    # rule 1: non-executable evidence cannot be 'measured'
    if ev not in EXECUTABLE:
        capped = cap(conf, "inferred")
        if capped != conf:
            trace.append(f"evidence '{ev}' non-executable -> cap at inferred")
            conf = capped
    # rule 2: refuted -> downgrade one level
    if refuted:
        new = downgrade(conf, 1)
        trace.append(f"refuted -> downgrade {conf}->{new}")
        conf = new
    out = dict(insight)
    out["confidence"] = conf
    out["verification"] = {"refuted": bool(refuted), "trace": trace, "final_confidence": conf}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--insight", required=True)
    ap.add_argument("--refuted", type=int, default=0)
    a = ap.parse_args()
    raw = sys.stdin.read() if a.insight == "-" else open(a.insight).read()
    insight = json.loads(raw)
    print(json.dumps(verify(insight, bool(a.refuted)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
