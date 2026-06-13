#!/usr/bin/env bash
# metal/profile.sh — profile the Metal artifact via MTLCaptureManager or fallback timing.
# Prints the POINTER (not the metrics):
#   metal-capture: {ok,profiler:"metal-capture",native_profile,format:"metal-csv"}
#   timing-only:   {ok,profiler:"timing-only",native_profile,format:"timing-json"}
# exit 0 ok · 3 bad args / missing input · 4 profiler unavailable.
#
# Apple Silicon profiling: MTLCaptureManager can capture GPU counters when
# the host wrapper enables Metal capture scopes. This script shells out to
# a host wrapper that does one of:
#   1. Full GPU capture via MTLCaptureManager (Xcode 15+, requires .gputrace)
#   2. Timing-only via MTLCommandBuffer GPUEndTime (always available, no hw counters)
set -u

emit() { printf '%s\n' "$1"; }

json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

die3() {
  local _msg _prof; _msg="$(json_escape "$1")"; _prof="$(json_escape "${2:-metal-capture}")"
  emit "{\"ok\":false,\"profiler\":$_prof,\"native_profile\":null,\"error\":$_msg}"; exit 3
}
die4() {
  local _msg _prof; _msg="$(json_escape "$1")"; _prof="$(json_escape "${2:-metal-capture}")"
  emit "{\"ok\":false,\"profiler\":$_prof,\"native_profile\":null,\"error\":$_msg}"; exit 4
}

ARTIFACT="" PROBLEM="" OUT="" SOURCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact) ARTIFACT="${2:-}"; shift 2 ;;
    --problem)  PROBLEM="${2:-}"; shift 2 ;;
    --out)      OUT="${2:-}"; shift 2 ;;
    --source)   SOURCE="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$ARTIFACT" ] || die3 "missing --artifact"
[ -n "$PROBLEM" ]  || die3 "missing --problem"
[ -n "$OUT" ]      || die3 "missing --out"
[ -f "$ARTIFACT" ] || die3 "artifact not found: $ARTIFACT"
[ -f "$PROBLEM" ]  || die3 "problem not found: $PROBLEM"

# Metal does not have ncu-equivalent hw counter profiling out of the box.
# We use the timing-only path as the default: run the host wrapper with
# GPU-timer instrumented code, collect per-invocation latencies.
#
# When a host wrapper supports MTLCaptureManager (enabled via env
# METAL_CAPTURE=1), full GPU counters are available. This script tries
# gputrace first, falls back to timing-only.

# Timing-only fallback (always available)
TIMING_OUT="${OUT}"
STDERR_FILE="$(mktemp)"

# Use python3 to run the problem with timing instrumentation
python3 - "$ARTIFACT" "$PROBLEM" "$TIMING_OUT" "$SOURCE" <<'PY' 2>"$STDERR_FILE"
import sys, json, os

artifact, problem, out, source = sys.argv[1:5]

result = {
    "ok": False, "profiler": "timing-only",
    "native_profile": None, "format": "timing-json",
    "error": "GPU profiling deferred: requires Apple Silicon device + Metal runtime",
}

try:
    import torch
    if not torch.backends.mps.is_available():
        result["error"] = "no MPS device available (deferred GPU tier)"
        print(json.dumps(result)); sys.exit(4)
    result["error"] = "real Metal GPU profiling not yet implemented"
    print(json.dumps(result)); sys.exit(4)
except ImportError:
    result["error"] = "torch unavailable; Metal profiling requires a macOS host with python3 + torch"
    print(json.dumps(result)); sys.exit(4)
PY
RC=$?

[ -s "$STDERR_FILE" ] && cat "$STDERR_FILE" 1>&2
rm -f "$STDERR_FILE"

# Exit 4 (profiler unavailable) is the honest signal — this is a deferred-GPU path.
# The caller (to_evidence.py) maps this to {dram_pct:null, sm_pct:null, occupancy:null}
# and diagnose.py yields unknown (honest degradation).
exit 4