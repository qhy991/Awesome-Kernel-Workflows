#!/usr/bin/env python3
"""Layer A — canonical attempt-evidence schema (KerSor Solver SDK).

Every solver attempt emits one record in this schema. It is byte-compatible with
KerSor's result-analyzer / select-workflow inputs, so a conformant solver feeds
KerSor natively. Deterministic; stdlib only.

Usage:
  evidence_schema.py validate <file.json|->   # validate + normalize; prints JSON; exit 0 ok / 1 invalid
  evidence_schema.py template                  # print a blank template
"""
import sys, json, argparse

KIND = {"bottleneck", "failed_strategy", "search_constraint", "validated_win", "env_fact"}
DIRECTIVE = {"explore", "avoid", "constrain", "reuse", "gate"}
EVIDENCE = {"ncu", "native_profiler", "benchmark", "compile", "correctness", "runtime", "profile_heuristic", "llm_inferred", "user_provided"}
CONFIDENCE = {"measured", "inferred", "hypothesized"}
CONV = {"converged", "budget_exhausted", "stalled", "error", "unknown"}

TEMPLATE = {
    "attempt_id": "a1", "parent_id": None,
    "compiled": True, "correct": True, "speedup": 1.0,
    "metrics": {"latency_ms": None, "dram_pct": None, "sm_pct": None, "occupancy": None},
    "best_kernel": None, "convergence_status": "unknown",
    "insights": [{"kind": "bottleneck", "directive": "explore", "evidence": "ncu",
                  "confidence": "measured", "claim": "...",
                  "actionable_hint": "concrete next-step the next round can act on (e.g. 'tile to 128, stage A through shared memory')"}],
    "failed_strategies": [],
}


def _validate_item(it, errs, where):
    if not isinstance(it, dict):
        errs.append(f"{where}: not an object"); return
    for field, allowed in (("kind", KIND), ("directive", DIRECTIVE),
                           ("evidence", EVIDENCE), ("confidence", CONFIDENCE)):
        if it.get(field) not in allowed:
            errs.append(f"{where}.{field}={it.get(field)!r} not in {sorted(allowed)}")
    if not it.get("claim"):
        errs.append(f"{where}.claim missing/empty")
    # #48: an insight must carry an actionable_hint — a concrete next-step the next
    # round can act on (not just an observation). claim = what we saw; hint = what
    # to do about it. Without it, downstream rounds get a diagnosis with no action.
    if not it.get("actionable_hint"):
        errs.append(f"{where}.actionable_hint missing/empty")


def normalize(d):
    errs = []
    if not isinstance(d, dict):
        return None, ["root is not an object"]
    out = {
        "attempt_id": (str(d["attempt_id"]) if d.get("attempt_id") not in (None, "") else None),
        "parent_id": (str(d["parent_id"]) if d.get("parent_id") not in (None, "") else None),
        "compiled": bool(d.get("compiled", False)),
        "correct": bool(d.get("correct", False)),
        "speedup": d.get("speedup", None),
        "metrics": d.get("metrics", {}) or {},
        "best_kernel": d.get("best_kernel", None),
        "convergence_status": d.get("convergence_status", "unknown"),
        "insights": d.get("insights", []) or [],
        "failed_strategies": d.get("failed_strategies", []) or [],
    }
    if out["speedup"] is not None:
        try:
            out["speedup"] = float(out["speedup"])
        except (TypeError, ValueError):
            errs.append("speedup not numeric")
    if not isinstance(out["metrics"], dict):
        errs.append("metrics not an object")
    if out["convergence_status"] not in CONV:
        errs.append(f"convergence_status={out['convergence_status']!r} not in {sorted(CONV)}")
    for kind in ("insights", "failed_strategies"):
        if not isinstance(out[kind], list):
            errs.append(f"{kind} not a list"); continue
        for i, it in enumerate(out[kind]):
            _validate_item(it, errs, f"{kind}[{i}]")
    # contract: an incorrect attempt cannot claim a real speedup
    if not out["correct"] and isinstance(out["speedup"], (int, float)) and out["speedup"] > 1.0:
        errs.append("incorrect attempt claims speedup > 1.0 (contract violation)")
    return out, errs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("path")
    sub.add_parser("template")
    a = ap.parse_args()
    if a.cmd == "template":
        print(json.dumps(TEMPLATE, indent=2)); return 0
    raw = sys.stdin.read() if a.path == "-" else open(a.path).read()
    try:
        d = json.loads(raw)
    except Exception as e:
        print(json.dumps({"valid": False, "errors": [f"invalid json: {e}"]})); return 1
    out, errs = normalize(d)
    print(json.dumps({"valid": not errs, "errors": errs, "normalized": out}, indent=2, ensure_ascii=False))
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
