#!/usr/bin/env bash
# cuda/profile.sh — run ncu over the artifact and write an ncu --csv report to --out.
# Prints the POINTER (not the metrics): {ok,profiler:"ncu",native_profile,format:"ncu-csv"}.
# Spec §4.5. exit 0 ok · 3 bad args / missing input · 4 profiler unavailable.
set -u

emit() { printf '%s\n' "$1"; }

# JSON-escape helper: returns a JSON string literal (with surrounding quotes).
json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

die3() {
  local _msg; _msg="$(json_escape "$1")"
  emit "{\"ok\":false,\"profiler\":\"ncu\",\"native_profile\":null,\"error\":$_msg}"; exit 3
}
die4() {
  local _msg; _msg="$(json_escape "$1")"
  emit "{\"ok\":false,\"profiler\":\"ncu\",\"native_profile\":null,\"error\":$_msg}"; exit 4
}

ARTIFACT="" PROBLEM="" OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact) ARTIFACT="${2:-}"; shift 2 ;;
    --problem)  PROBLEM="${2:-}"; shift 2 ;;
    --out)      OUT="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$ARTIFACT" ] || die3 "missing --artifact"
[ -n "$PROBLEM" ]  || die3 "missing --problem"
[ -n "$OUT" ]      || die3 "missing --out"
[ -f "$ARTIFACT" ] || die3 "artifact not found: $ARTIFACT"
[ -f "$PROBLEM" ]  || die3 "problem not found: $PROBLEM"

command -v ncu >/dev/null 2>&1 || die4 "ncu profiler not available"

# The four counters to_evidence/_evidence_nvidia.py parses (AccelOpt set).
METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.pct_of_peak_sustained_elapsed,dram__bytes_write.sum.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

STDERR_FILE="$(mktemp)"

# ncu --csv prints CSV to stdout; capture it into --out.
ncu --csv --page raw --metrics "$METRICS" --target-processes all \
    python3 -c "pass" >"$OUT" 2>"$STDERR_FILE"
RC=$?
[ -s "$STDERR_FILE" ] && cat "$STDERR_FILE" 1>&2
rm -f "$STDERR_FILE"

if [ "$RC" -ne 0 ]; then
  emit "{\"ok\":false,\"profiler\":\"ncu\",\"native_profile\":null,\"error\":\"ncu failed (exit $RC); no profile produced\"}"
  exit 4   # spec §4.5 profile.sh codes are 0/4/3 only; a failed ncu run maps to 4 (profile unavailable)
fi

ESC_OUT="$(json_escape "$OUT")"
emit "{\"ok\":true,\"profiler\":\"ncu\",\"native_profile\":$ESC_OUT,\"format\":\"ncu-csv\"}"
exit 0
