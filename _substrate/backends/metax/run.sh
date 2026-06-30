#!/usr/bin/env bash
# metax/run.sh — load the compiled .so, run vs problem, measure correctness + latency
# on MetaX GPU via the MACA runtime (MXMGPU / mxgpu_llvm).
# Spec §4.5. stdout keys MUST match anti_cheat.py --metrics EXACTLY:
#   {ok,compiled,correct,candidate_latency_ms,eager_latency_ms,compile_latency_ms,
#    claimed_speedup,...}
#   correct:false ⇒ claimed_speedup ≤ 1.0.
# exit 0 ok · 2 op-error (json printed) · 3 bad args / missing input.
set -u

json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

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
# flags.yaml sha256=067a79294b2ca090
# DO NOT EDIT BETWEEN SENTINELS — edit flags.yaml and re-run: python3 _substrate/backends/_gen_flag_parser.py --write metax
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
[ -f "$ARTIFACT" ] || err_envelope "artifact not found: $ARTIFACT" 3

command -v python3 >/dev/null 2>&1 || err_envelope "python3 not found" 3

python3 - "$ARTIFACT" "$PROBLEM" "$OUT" "$REPS" "$RTOL" "$ATOL" "$BASELINE" <<'PY'
import sys, json
artifact, problem, out, reps, rtol, atol, baseline = sys.argv[1:8]
result = {
    "ok": False, "compiled": True, "correct": False,
    "candidate_latency_ms": None, "eager_latency_ms": None,
    "compile_latency_ms": None, "claimed_speedup": 1.0,
    "error": "GPU execution deferred: requires MetaX GPU + MACA runtime (MXMGPU)",
}
try:
    import torch
    # Check for MetaX device via torch (MACA SDK registers a custom device backend).
    # If torch.cuda.is_available() reports true with a MetaX GPU, the MACA runtime
    # has mapped it as a CUDA device.
    if not torch.cuda.is_available():
        result["error"] = "no GPU device available (deferred MetaX GPU tier)"
        print(json.dumps(result)); sys.exit(2)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
    result["error"] = f"real GPU run not yet implemented (device: {device_name})"
    print(json.dumps(result)); sys.exit(2)
except ImportError:
    result["error"] = "torch unavailable"
    print(json.dumps(result)); sys.exit(2)
PY
exit $?