#!/usr/bin/env python3
"""Layer C — diagnosis: profile metrics -> shared bottleneck taxonomy.

Borrowed from AccelOpt / KernelBand (phi features) / KernelBlaster (state class)
/ cuPilot (roofline). Deterministic thresholds. The canonical bottleneck_class
enum is shared across all solvers so memory keys and routing features line up.

Usage:
  diagnose.py --metrics metrics.json   # {dram_pct, sm_pct, occupancy, latency_ms, problem_size?}
Prints: {bottleneck_class, evidence}
bottleneck_class in {memory_bound, compute_bound, latency_occupancy, overhead_bound, unknown}

Qualitative / perf_heuristic path (no NCU numerics):
  metrics may carry heuristic_bclass (canonical enum) or bottleneck_hint (free text).
  When evidence/evidence_source is profile_heuristic, these fields are honored before
  falling through to unknown — closing the perf_heuristic -> diagnose contract gap.
"""
import sys, json, argparse, re

CLASSES = ["memory_bound", "compute_bound", "latency_occupancy", "overhead_bound", "unknown"]

# Per-vendor decision thresholds (spec §5.3.1). Default/absent _vendor -> "nvidia"
# == today's literals, so every existing NVIDIA case stays byte-identical.
PROFILES = {
    "nvidia": dict(occ_lat=0.40, dram_mem=70, sm_mem=50, sm_comp=70, both_low=40),
    "apple": dict(occ_lat=0.30, dram_mem=65, sm_mem=55, sm_comp=65, both_low=35),
    # Ascend NPU (msprof): dram_pct=GM/MTE bandwidth, sm_pct=max(cube,vector) AI Core util.
    # occupancy is never measured (always null) so occ_lat is inert here. Memory-bound when
    # GM bandwidth is high while compute units are idle; compute-bound when cube/vector are
    # busy; overhead-bound when both are low (tiny launches / poor core occupancy).
    "ascend": dict(occ_lat=0.40, dram_mem=70, sm_mem=40, sm_comp=60, both_low=30),
    # MetaX GPU (mcProfiler via MACA SDK): dram_pct=DRAM bandwidth, sm_pct=SM compute util.
    # occupancy is never measured (always null) so occ_lat is inert here. Thresholds
    # set to mirror nvidia defaults until MetaX-specific calibration data is available.
    "metax": dict(occ_lat=0.40, dram_mem=70, sm_mem=50, sm_comp=70, both_low=40),
}

# Free-text hints from perf_heuristic agents / perf_to_evidence roofline claims.
# Order matters: first match wins.
_HINT_RULES = [
    ("memory_bound", re.compile(
        r"memory[_\s-]?bound|dram|bandwidth|l2[\s-]?bound|cache[\s-]?bound|"
        r"throughput[\s-]?bound|bandwidth[\s-]?bound|gm[\s/_-]?bound|mte[\s-]?bound", re.I)),
    ("compute_bound", re.compile(
        r"compute[_\s-]?bound|sm[\s/_-]?bound|tensor|flop|math[\s-]?bound|"
        r"cube[\s-]?bound|vector[\s-]?bound|ai[\s-]?core", re.I)),
    ("latency_occupancy", re.compile(
        r"latency[_\s-]?bound|latency[_\s-]?occupancy|occupancy|launch[\s-]?config|"
        r"not saturating|low util", re.I)),
    ("overhead_bound", re.compile(
        r"overhead[_\s-]?bound|kernel[\s-]?launch|launch[\s-]?overhead", re.I)),
]

_ALIAS = {
    "latency_bound": "latency_occupancy",
    "memory": "memory_bound",
    "compute": "compute_bound",
    "latency": "latency_occupancy",
    "overhead": "overhead_bound",
}


def _normalize_bclass(raw):
    if raw is None:
        return None
    s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if s in CLASSES:
        return s
    return _ALIAS.get(s)


def _heuristic_evidence_tag(m):
    return m.get("evidence") or m.get("evidence_source") or "profile_heuristic"


def _classify_from_hint(text):
    if not text:
        return None
    t = str(text).lower()
    # Explicit class phrases win over loose keyword hits (e.g. "latency bound, not saturating DRAM").
    explicit = [
        ("memory_bound", re.compile(r"memory[\s_-]?bound|dram[\s_-]?bound|bandwidth[\s_-]?bound")),
        ("compute_bound", re.compile(r"compute[\s_-]?bound|sm[\s_-]?bound|math[\s_-]?bound")),
        ("latency_occupancy", re.compile(r"latency[\s_-]?bound|latency[\s_-]?occupancy|occupancy[\s_-]?bound")),
        ("overhead_bound", re.compile(r"overhead[\s_-]?bound|launch[\s_-]?overhead")),
    ]
    for cls, pat in explicit:
        if pat.search(t):
            return cls, [f"heuristic hint {text!r} -> {cls}"]
    for cls, pat in _HINT_RULES:
        if pat.search(t):
            return cls, [f"heuristic hint {text!r} -> {cls}"]
    return None


def _classify_heuristic(m):
    """Qualitative path before numeric thresholds."""
    hb = _normalize_bclass(m.get("heuristic_bclass"))
    if hb and hb != "unknown":
        return hb, [f"heuristic_bclass={hb} (evidence={_heuristic_evidence_tag(m)})"]

    hint = m.get("bottleneck_hint")
    if hint is not None:
        got = _classify_from_hint(hint)
        if got:
            cls, ev = got
            ev.append(f"evidence={_heuristic_evidence_tag(m)}")
            return cls, ev

    # profile_heuristic with only a qualitative note in a generic field
    note = m.get("profile_note") or m.get("note")
    if note and (m.get("evidence") == "profile_heuristic"
                 or m.get("evidence_source") == "profile_heuristic"):
        got = _classify_from_hint(note)
        if got:
            cls, ev = got
            ev.append(f"evidence={_heuristic_evidence_tag(m)}")
            return cls, ev
    return None


def classify(m):
    got = _classify_heuristic(m)
    if got:
        return got

    dram = m.get("dram_pct")
    sm = m.get("sm_pct")
    occ = m.get("occupancy")
    prof = PROFILES.get(m.get("_vendor", "nvidia"), PROFILES["nvidia"])
    if dram is None and sm is None and occ is None:
        return "unknown", ["no profiler metrics available"]
    # latency_occupancy: single positive signal — occupancy measured and clearly low.
    if occ is not None and occ < prof["occ_lat"]:
        return "latency_occupancy", [f"occupancy {occ:.2f} < {prof['occ_lat']:.2f} (launch/occupancy limited)"]
    # compute_bound: single positive signal — sm measured and high.
    if sm is not None and sm >= prof["sm_comp"]:
        return "compute_bound", [f"sm {sm:.0f}% high"]
    # memory_bound / overhead_bound are two-sided: both dram AND sm must be measured.
    if dram is not None and sm is not None:
        if dram >= prof["dram_mem"] and sm < prof["sm_mem"]:
            return "memory_bound", [f"dram {dram:.0f}% high, sm {sm:.0f}% low"]
        if dram < prof["both_low"] and sm < prof["both_low"]:
            return "overhead_bound", [f"both utilizations low (dram {dram:.0f}%, sm {sm:.0f}%)"]
        return "unknown", [f"no dominant signal (dram {dram:.0f}%, sm {sm:.0f}%, occ {occ})"]
    # A required two-sided discriminator was unmeasured -> cannot conclude.
    return "unknown", ["no dominant signal (insufficient measured metrics)"]


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
