#!/usr/bin/env python3
"""Layer B — anti-cheat / validity gate + robust reward (KerSor Solver SDK).

Borrowed from CUDAAgent (robust reward), CUDALLM/TritorX (cheat detection),
AKO4X (pre-commit expectation). Deterministic. Static heuristics on the kernel
source + a dynamic robust reward from measured baselines. An invalid attempt
records speedup 0 and must never enter memory.

Usage:
  anti_cheat.py --source kernel.cu --metrics metrics.json
  anti_cheat.py --source-text "<code>" --metrics metrics.json
  # metrics.json: {compiled, correct, candidate_latency_ms, eager_latency_ms,
  #                compile_latency_ms, claimed_speedup}
Prints: {valid, reward, recorded_speedup, flags, reward_reason}
"""
import sys, json, argparse, re

# (regex, label) — library delegation where a custom kernel is expected
FALLBACK_PATTERNS = [
    (r"\btorch\.matmul\b", "torch.matmul fallback"),
    (r"\bF\.linear\b", "F.linear fallback"),
    (r"torch\.nn\.functional", "torch.nn.functional fallback"),
    (r"\bcublas\w*", "cuBLAS delegation"),
    (r"\bcudnn\w*", "cuDNN delegation"),
    (r"\bat::matmul\b", "at::matmul fallback"),
]
# (regex, label) — compute silently skipped / placeholder
SKIP_PATTERNS = [
    (r"return\s+input\b", "returns input unchanged"),
    (r"#\s*TODO", "TODO placeholder"),
    (r"raise\s+NotImplementedError", "NotImplementedError"),
    (r"^\s*pass\s*$", "empty pass body"),
]
HARDCODE_HINT = re.compile(r"\b(M|N|K|seq_len|hidden|batch|dim)\s*=\s*\d{2,}")

BLOCKING = {"library_fallback", "skipped_compute"}  # hardcoded_shape is a warning only


def load_vendor_patterns(path):
    """Parse a vendor cheat-pattern file into (fallback, skip) lists of (regex, label).

    Format (spec §5.3.3): two sections [fallback] and [skip]; one regex per line;
    optional "regex | label". Blank lines and lines starting with '#' are ignored.
    """
    fallback, skip = [], []
    section = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                continue
            if "|" in line:
                pat, label = line.split("|", 1)
                pat, label = pat.strip(), label.strip()
            else:
                pat, label = line, line
            try:
                re.compile(pat)
            except re.error as exc:
                raise ValueError(f"{path}: invalid regex {pat!r}: {exc}") from exc
            if section == "fallback":
                fallback.append((pat, label))
            elif section == "skip":
                skip.append((pat, label))
    return fallback, skip


def static_flags(src, extra_fallback=(), extra_skip=()):
    flags = []
    for pat, msg in list(FALLBACK_PATTERNS) + list(extra_fallback):
        if re.search(pat, src):
            flags.append({"type": "library_fallback", "detail": msg})
    for pat, msg in list(SKIP_PATTERNS) + list(extra_skip):
        if re.search(pat, src, re.M):
            flags.append({"type": "skipped_compute", "detail": msg})
    if len(HARDCODE_HINT.findall(src)) >= 2:
        flags.append({"type": "hardcoded_shape", "detail": "multiple hardcoded dimensions"})
    return flags


def robust_reward(m):
    if not m.get("compiled", False):
        return -1, "not compiled"
    if not m.get("correct", False):
        return -1, "incorrect"
    cand = m.get("candidate_latency_ms")
    eager = m.get("eager_latency_ms")
    comp = m.get("compile_latency_ms")
    if cand is None or eager is None:
        return 0, "missing latencies (cannot confirm speedup)"
    beats_eager = cand < eager
    beats_compile = (comp is not None) and (cand < comp)
    if beats_eager and beats_compile:
        return 3, "beats eager and torch.compile"
    if beats_eager:
        return 2, "beats eager only"
    return 0, "no speedup over baselines"


def evaluate(src, m, extra_fallback=(), extra_skip=()):
    flags = static_flags(src or "", extra_fallback, extra_skip)
    reward, reason = robust_reward(m)
    blocking = [f for f in flags if f["type"] in BLOCKING]
    valid = (reward >= 0) and (not blocking)
    # recorded_speedup: the reward-weighted speedup that may enter the beam/memory
    # (only when reward>=2, i.e. a real win over baseline). A valid compile+correct
    # run that merely doesn't beat the reward threshold (e.g. ~1.0x SMOKE, or a
    # sub-path win that loses on aggregate geomean) would record 0 here — which
    # misled consumers into reading a valid run as a failure. measured_speedup is
    # the actual measured value whenever the attempt is valid, independent of the
    # reward gate, so SMOKE/target-not-met-but-valid attempts are not misread as 0.
    claimed = float(m.get("claimed_speedup") or 0.0)
    recorded = claimed if (valid and reward >= 2) else 0.0
    measured = claimed if valid else 0.0
    return {
        "valid": valid,
        "reward": reward,
        "recorded_speedup": recorded,
        "measured_speedup": measured,
        "reward_reason": reason,
        "flags": flags,
        "blocking_flags": [f["type"] for f in blocking],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source"); ap.add_argument("--source-text")
    ap.add_argument("--vendor-patterns-file")
    ap.add_argument("--metrics", required=True)
    a = ap.parse_args()
    src = a.source_text if a.source_text is not None else (open(a.source).read() if a.source else "")
    m = json.loads(sys.stdin.read() if a.metrics == "-" else open(a.metrics).read())
    extra_fallback, extra_skip = ([], [])
    if a.vendor_patterns_file:
        extra_fallback, extra_skip = load_vendor_patterns(a.vendor_patterns_file)
    res = evaluate(src, m, extra_fallback, extra_skip)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
