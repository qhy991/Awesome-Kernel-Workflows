#!/usr/bin/env bash
# triton/run.sh — python launcher + torch.testing.assert_close + triton.testing.do_bench.
# Same stdout key shape as cuda/run.sh (anti_cheat contract). Spec §4.5.
#   exit 0 ok · 2 op-error (json printed) · 3 bad args / missing input.
set -u

# JSON-escape helper: returns a JSON string literal (with surrounding quotes).
json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

# Clean error envelope with the full contract key set (claimed_speedup floored at 1.0).
err_envelope() {
  local msg="$1" rc="$2"
  local _msg; _msg="$(json_escape "$msg")"
  printf '{"ok":false,"compiled":false,"correct":false,"candidate_latency_ms":null,'
  printf '"eager_latency_ms":null,"compile_latency_ms":null,"claimed_speedup":1.0,'
  printf '"error":%s}\n' "$_msg"
  exit "$rc"
}

ARTIFACT="" PROBLEM="" OUT="" REPS=50 RTOL="1e-3" ATOL="1e-3" BASELINE="both"
# BEGIN AUTO-GENERATED FLAG PARSER — regenerate from flags.yaml via _substrate/backends/_gen_flag_parser.py
# flags.yaml sha256=3a1f3a0082425666
# DO NOT EDIT BETWEEN SENTINELS — edit flags.yaml and re-run: python3 _substrate/backends/_gen_flag_parser.py --write triton
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact) ARTIFACT="${2:-}"; shift 2 ;;
    --problem)  PROBLEM="${2:-}"; shift 2 ;;
    --out)      OUT="${2:-}"; shift 2 ;;
    --reps)     REPS="${2:-}"; shift 2 ;;
    --rtol)     RTOL="${2:-}"; shift 2 ;;
    --atol)     ATOL="${2:-}"; shift 2 ;;
    --baseline) BASELINE="${2:-}"; shift 2 ;;
    *) err_envelope "unknown arg: $1" 3 ;;
  esac
done
# END AUTO-GENERATED FLAG PARSER

[ -n "$ARTIFACT" ] || err_envelope "missing --artifact" 3
[ -n "$PROBLEM" ]  || err_envelope "missing --problem" 3
[ -n "$OUT" ]      || err_envelope "missing --out" 3
[ -f "$PROBLEM" ]  || err_envelope "problem not found: $PROBLEM" 3
[ -e "$ARTIFACT" ] || err_envelope "artifact not found: $ARTIFACT" 3   # preflight/bad-input => exit 3 (spec §4.5)
command -v python3 >/dev/null 2>&1 || err_envelope "python3 not found" 3

# Reuse warmed PTX from the build artifact dir instead of re-JITting from scratch.
export TRITON_CACHE_DIR="$ARTIFACT"

python3 - "$ARTIFACT" "$PROBLEM" "$OUT" "$REPS" "$RTOL" "$ATOL" "$BASELINE" <<'PY'
import sys, json
artifact, problem, out, reps, rtol, atol, baseline = sys.argv[1:8]
result = {
    "ok": False, "compiled": True, "correct": False,
    "candidate_latency_ms": None, "eager_latency_ms": None,
    "compile_latency_ms": None, "claimed_speedup": 1.0,
    "error": "GPU execution deferred: requires NVIDIA device + triton runtime",
}
try:
    import torch
    if not torch.cuda.is_available():
        result["error"] = "no CUDA device available (deferred GPU tier)"
        print(json.dumps(result)); sys.exit(2)
    import triton  # noqa: F401
    # ---- real path: import launcher, torch.testing.assert_close, triton.testing.do_bench
    result["error"] = "real triton run not yet implemented"
    print(json.dumps(result)); sys.exit(2)
except ImportError as exc:
    result["error"] = f"runtime unavailable: {exc}"
    print(json.dumps(result)); sys.exit(2)
PY
exit $?
