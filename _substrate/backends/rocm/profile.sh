#!/usr/bin/env bash
# rocm/profile.sh -- run rocprofv3 over the artifact (or test command) and write a CSV
# report to --out. Prints the POINTER, not the metrics:
#   {ok,profiler:"rocprofv3",native_profile,format:"rocprof-csv"}
# Spec SS4.5. exit 0 ok . 3 bad args / missing input . 4 profiler unavailable.
#
# Degraded mode: if no rocprof tool is present, exits 4 with a clear message. Callers
# that only need latency should use run.sh instead; profile.sh is the counter path.
set -u

emit() { printf '%s\n' "$1"; }

json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

die3() {
  local _msg; _msg="$(json_escape "$1")"
  emit "{\"ok\":false,\"profiler\":\"rocprofv3\",\"native_profile\":null,\"error\":$_msg}"; exit 3
}
die4() {
  local _msg; _msg="$(json_escape "$1")"
  emit "{\"ok\":false,\"profiler\":\"rocprofv3\",\"native_profile\":null,\"error\":$_msg}"; exit 4
}

ARTIFACT="" PROBLEM="" OUT="" TEST_CMD="" COUNTERS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact) ARTIFACT="${2:-}"; shift 2 ;;
    --problem)  PROBLEM="${2:-}"; shift 2 ;;
    --out)      OUT="${2:-}"; shift 2 ;;
    --test-cmd) TEST_CMD="${2:-}"; shift 2 ;;
    --counters) COUNTERS="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$OUT" ] || die3 "missing --out"

# Pick the available rocprof tool: rocprofv3 is preferred (ROCm 6+); fall back to rocprof.
ROCPROF=""
if command -v rocprofv3 >/dev/null 2>&1; then
  ROCPROF="rocprofv3"
elif command -v rocprof >/dev/null 2>&1; then
  ROCPROF="rocprof"
else
  die4 "no rocprof tool found on PATH (tried rocprofv3, rocprof)"
fi

# Default counter set: latency-oriented + memory + occupancy. Caller can override.
if [ -z "$COUNTERS" ]; then
  COUNTERS="SQ_WAVES,SQ_INSTS_VALU,VALUInsts,VFetchInsts,VWriteInsts,TCP_TOTAL_CACHE_ACCESSES_sum,TCP_TOTAL_CACHE_HITS_sum,GRBM_GUI_ACTIVE"
fi

# Build the target command. In-codebase mode uses --test-cmd verbatim; standalone mode
# launches a python harness against the .so. The standalone harness is the GPU-tier
# deferred path; tests cover only the wrapper.
TARGET_CMD=""
if [ -n "$TEST_CMD" ]; then
  TARGET_CMD="$TEST_CMD"
else
  [ -n "$ARTIFACT" ] || die3 "missing --artifact (or --test-cmd)"
  [ -n "$PROBLEM" ]  || die3 "missing --problem"
  [ -f "$ARTIFACT" ] || die3 "artifact not found: $ARTIFACT"
  [ -f "$PROBLEM" ]  || die3 "problem not found: $PROBLEM"
  TARGET_CMD="python3 \"$PROBLEM\" \"$ARTIFACT\""
fi

STDERR_FILE="$(mktemp)"
COUNTER_FILE="$(mktemp)"
# Write a counter spec file (rocprofv3 prefers --pmc CSV via -i input file or --pmc-mode).
# rocprofv3 syntax: rocprofv3 --pmc <list> -o <out_csv> -- <command>
# rocprof   syntax: rocprof   -i counters.txt -o out.csv <command>
case "$ROCPROF" in
  rocprofv3)
    # shellcheck disable=SC2086
    rocprofv3 --pmc "$COUNTERS" -o "$OUT" -- bash -c "$TARGET_CMD" 2>"$STDERR_FILE"
    RC=$?
    ;;
  rocprof)
    printf 'pmc : %s\n' "$(echo "$COUNTERS" | tr ',' ' ')" >"$COUNTER_FILE"
    # shellcheck disable=SC2086
    rocprof -i "$COUNTER_FILE" -o "$OUT" bash -c "$TARGET_CMD" 2>"$STDERR_FILE"
    RC=$?
    ;;
esac

STDERR_TAIL="$(tail -c 4096 "$STDERR_FILE" 2>/dev/null || true)"
STDERR_JSON="$(json_escape "$STDERR_TAIL")"
rm -f "$STDERR_FILE" "$COUNTER_FILE"

if [ "$RC" -ne 0 ]; then
  emit "{\"ok\":false,\"profiler\":\"$ROCPROF\",\"native_profile\":null,\"stderr_tail\":$STDERR_JSON,\"error\":\"$ROCPROF exited $RC\"}"
  exit 2
fi

OUT_JSON="$(json_escape "$OUT")"
emit "{\"ok\":true,\"profiler\":\"$ROCPROF\",\"native_profile\":$OUT_JSON,\"format\":\"rocprof-csv\"}"
exit 0
