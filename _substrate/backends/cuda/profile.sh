#!/usr/bin/env bash
# cuda/profile.sh — profile the artifact with ncu (preferred) or nsys (fallback).
# Prints the POINTER (not the metrics):
#   ncu:  {ok,profiler:"ncu",native_profile,format:"ncu-csv"}
#   nsys: {ok,profiler:"nsys",native_profile,format:"nsys-sqlite"}
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
  local _msg _prof; _msg="$(json_escape "$1")"; _prof="$(json_escape "${2:-ncu}")"
  emit "{\"ok\":false,\"profiler\":$_prof,\"native_profile\":null,\"error\":$_msg}"; exit 3
}
die4() {
  local _msg _prof; _msg="$(json_escape "$1")"; _prof="$(json_escape "${2:-ncu}")"
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

# --- NCU path (hardware counters; preferred) --------------------------------
if command -v ncu >/dev/null 2>&1; then
  METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.pct_of_peak_sustained_elapsed,dram__bytes_write.sum.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"
  STDERR_FILE="$(mktemp)"
  ncu --csv --page raw --metrics "$METRICS" --target-processes all \
      python3 -c "pass" >"$OUT" 2>"$STDERR_FILE"
  RC=$?
  [ -s "$STDERR_FILE" ] && cat "$STDERR_FILE" 1>&2
  rm -f "$STDERR_FILE"
  if [ "$RC" -ne 0 ]; then
    emit "{\"ok\":false,\"profiler\":\"ncu\",\"native_profile\":null,\"error\":\"ncu failed (exit $RC); no profile produced\"}"
    exit 4
  fi
  ESC_OUT="$(json_escape "$OUT")"
  emit "{\"ok\":true,\"profiler\":\"ncu\",\"native_profile\":$ESC_OUT,\"format\":\"ncu-csv\"}"
  exit 0
fi

# --- nsys fallback (timeline + kernel latency; no dram/sm/occupancy counters) -
if ! command -v nsys >/dev/null 2>&1; then
  die4 "neither ncu nor nsys profiler available"
fi

# nsys writes <base>.sqlite (+ .nsys-rep). --out should name the .sqlite destination.
case "$OUT" in
  *.sqlite) SQLITE_OUT="$OUT"; BASE="${OUT%.sqlite}" ;;
  *) die3 "nsys fallback requires --out to end with .sqlite" "nsys" ;;
esac

if [ -n "$SOURCE" ] && [ -f "$SOURCE" ]; then
  RUN_CMD=(python3 "$SOURCE")
elif [ -x "$ARTIFACT" ]; then
  RUN_CMD=("$ARTIFACT")
else
  die4 "nsys fallback requires runnable --source launcher or executable --artifact" "nsys"
fi

STDERR_FILE="$(mktemp)"
nsys profile --trace=cuda --export=sqlite --force-overwrite=true -o "$BASE" "${RUN_CMD[@]}" >"$STDERR_FILE" 2>&1
RC=$?
[ -s "$STDERR_FILE" ] && cat "$STDERR_FILE" 1>&2
rm -f "$STDERR_FILE"

if [ "$RC" -ne 0 ] || [ ! -s "$SQLITE_OUT" ]; then
  emit "{\"ok\":false,\"profiler\":\"nsys\",\"native_profile\":null,\"error\":\"nsys failed (exit $RC); no sqlite profile produced\"}"
  exit 4
fi

ESC_OUT="$(json_escape "$SQLITE_OUT")"
emit "{\"ok\":true,\"profiler\":\"nsys\",\"native_profile\":$ESC_OUT,\"format\":\"nsys-sqlite\"}"
exit 0
