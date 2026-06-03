#!/usr/bin/env python3
"""Layer D — persistent, state-keyed optimization memory (KerSor Solver SDK).

Borrowed from KernelBlaster (cross-run DB, confidence blended with measured
outcome) + AKO4X (dead-ends with WHY + revalidate_if). Persists to a JSON file so
experience compounds across rounds, kernels, and sessions. Deterministic (no time
/ randomness): confidence is updated only from measured outcomes.

Usage:
  memory_store.py --db DB.json retrieve --class memory_bound
  memory_store.py --db DB.json update --class memory_bound --technique vectorized_load_store --speedup 1.2 --correct 1
  memory_store.py --db DB.json add-deadend --claim "split-K atomics" --why "atomic contention" --revalidate-if "N<512"
  memory_store.py --db DB.json list
Confidence update: conf <- clip(conf*DECAY + outcome, 0, 1);
  outcome = +0.2 if (correct and speedup>1.0) else -0.1
"""
import sys, json, argparse, os

DECAY = 0.9
START_CONF = 0.5


def load(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"schema_version": 1, "by_class": {}, "dead_ends": []}


def save(path, db):
    with open(path, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def cmd_retrieve(db, bclass):
    items = db["by_class"].get(bclass, [])
    ranked = sorted(items, key=lambda t: (-t["confidence"], -t.get("avg_speedup", 0.0), t["technique"]))
    return {"bottleneck_class": bclass, "techniques": ranked,
            "dead_ends": db.get("dead_ends", [])}


def cmd_update(db, bclass, technique, speedup, correct):
    bucket = db["by_class"].setdefault(bclass, [])
    entry = next((t for t in bucket if t["technique"] == technique), None)
    if entry is None:
        entry = {"technique": technique, "confidence": START_CONF, "usage_count": 0,
                 "correct_count": 0, "avg_speedup": 0.0}
        bucket.append(entry)
    outcome = 0.2 if (correct and speedup is not None and speedup > 1.0) else -0.1
    entry["confidence"] = round(clip(entry["confidence"] * DECAY + outcome), 4)
    entry["usage_count"] += 1
    if correct:
        entry["correct_count"] += 1
    if speedup is not None:
        n = entry["usage_count"]
        entry["avg_speedup"] = round((entry["avg_speedup"] * (n - 1) + speedup) / n, 4)
    return entry


def cmd_add_deadend(db, claim, why, revalidate_if):
    de = {"claim": claim, "why": why, "revalidate_if": revalidate_if}
    if de not in db["dead_ends"]:
        db["dead_ends"].append(de)
    return de


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieve"); r.add_argument("--class", dest="bclass", required=True)
    u = sub.add_parser("update")
    u.add_argument("--class", dest="bclass", required=True)
    u.add_argument("--technique", required=True)
    u.add_argument("--speedup", type=float, default=None)
    u.add_argument("--correct", type=int, default=0)
    d = sub.add_parser("add-deadend")
    d.add_argument("--claim", required=True); d.add_argument("--why", required=True)
    d.add_argument("--revalidate-if", dest="revalidate_if", default="")
    sub.add_parser("list")
    a = ap.parse_args()

    db = load(a.db)
    if a.cmd == "retrieve":
        print(json.dumps(cmd_retrieve(db, a.bclass), indent=2, ensure_ascii=False))
    elif a.cmd == "update":
        out = cmd_update(db, a.bclass, a.technique, a.speedup, bool(a.correct))
        save(a.db, db)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    elif a.cmd == "add-deadend":
        out = cmd_add_deadend(db, a.claim, a.why, a.revalidate_if)
        save(a.db, db)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    elif a.cmd == "list":
        print(json.dumps(db, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
