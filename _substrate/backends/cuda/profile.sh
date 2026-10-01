#!/usr/bin/env bash
# cuda/profile.sh — profile a caller-owned launch of the artifact.
# Prints the POINTER (not the metrics):
#   ncu:  {ok,profiler:"ncu",native_profile,format:"ncu-csv"}
#   nsys: {ok,profiler:"nsys",native_profile,format:"nsys-sqlite"}
# Spec §4.5. exit 0 ok · 3 bad args / missing input · 4 profiler unavailable.
set -u

ENVELOPE_OUT=""
emit() {
  printf '%s\n' "$1"
  [ -z "$ENVELOPE_OUT" ] || printf '%s\n' "$1" >"$ENVELOPE_OUT"
}

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

ARTIFACT="" PROBLEM="" OUT="" SOURCE="" NCU_BINARY="${NCU:-ncu}"
NSYS_BINARY="${NSYS:-nsys}" NSYS_OUT=""
METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.pct_of_peak_sustained_elapsed,dram__bytes_write.sum.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"
KERNEL_NAME="" LAUNCH_COUNT=""
RUN_CMD=()
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact) ARTIFACT="${2:-}"; shift 2 ;;
    --problem)  PROBLEM="${2:-}"; shift 2 ;;
    --out)      OUT="${2:-}"; shift 2 ;;
    --envelope-out) ENVELOPE_OUT="${2:-}"; shift 2 ;;
    --source)   SOURCE="${2:-}"; shift 2 ;;
    --ncu-binary) NCU_BINARY="${2:-}"; shift 2 ;;
    --nsys-binary) NSYS_BINARY="${2:-}"; shift 2 ;;
    --nsys-out) NSYS_OUT="${2:-}"; shift 2 ;;
    --metrics) METRICS="${2:-}"; shift 2 ;;
    --kernel-name) KERNEL_NAME="${2:-}"; shift 2 ;;
    --launch-count) LAUNCH_COUNT="${2:-}"; shift 2 ;;
    --) shift; RUN_CMD=("$@"); break ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$ARTIFACT" ] || die3 "missing --artifact"
[ -n "$PROBLEM" ]  || die3 "missing --problem"
[ -n "$OUT" ]      || die3 "missing --out"
[ -f "$ARTIFACT" ] || die3 "artifact not found: $ARTIFACT"
[ -f "$PROBLEM" ]  || die3 "problem not found: $PROBLEM"
[ -n "$METRICS" ] || die3 "empty --metrics"
case "$LAUNCH_COUNT" in ''|*[!0-9]*) [ -z "$LAUNCH_COUNT" ] || die3 "--launch-count must be a positive integer" ;; esac
[ -z "$LAUNCH_COUNT" ] || [ "$LAUNCH_COUNT" -gt 0 ] || die3 "--launch-count must be a positive integer"
if [ "${#RUN_CMD[@]}" -eq 0 ]; then
  if [ -n "$SOURCE" ] && [ -f "$SOURCE" ]; then
    RUN_CMD=(python3 "$SOURCE")
  elif [ -x "$ARTIFACT" ]; then
    RUN_CMD=("$ARTIFACT")
  else
    die3 "no runnable workload: pass launcher argv after --, --source, or an executable --artifact"
  fi
fi
command -v "${RUN_CMD[0]}" >/dev/null 2>&1 || [ -x "${RUN_CMD[0]}" ] || die3 "launcher not found: ${RUN_CMD[0]}"

run_nsys() {
  local why="$1" sqlite_out base stderr_file rc esc_out
  command -v "$NSYS_BINARY" >/dev/null 2>&1 || die4 "nsys unavailable after $why" "nsys"
  sqlite_out="${NSYS_OUT:-$OUT}"
  case "$sqlite_out" in
    *.sqlite) base="${sqlite_out%.sqlite}" ;;
    *) die3 "nsys fallback requires --nsys-out (or --out) ending in .sqlite" "nsys" ;;
  esac
  stderr_file="$(mktemp)"
  "$NSYS_BINARY" profile --trace=cuda --export=sqlite --force-overwrite=true \
      -o "$base" "${RUN_CMD[@]}" >"$stderr_file" 2>&1
  rc=$?
  [ -s "$stderr_file" ] && cat "$stderr_file" 1>&2
  rm -f "$stderr_file"
  if [ "$rc" -ne 0 ] || [ ! -s "$sqlite_out" ]; then
    emit "{\"ok\":false,\"profiler\":\"nsys\",\"native_profile\":null,\"error\":\"nsys failed (exit $rc); no sqlite profile produced\"}"
    exit 4
  fi
  esc_out="$(json_escape "$sqlite_out")"
  emit "{\"ok\":true,\"profiler\":\"nsys\",\"native_profile\":$esc_out,\"format\":\"nsys-sqlite\",\"degraded_from\":\"$why\"}"
  exit 0
}

# --- NCU path (hardware counters; preferred) --------------------------------
if command -v "$NCU_BINARY" >/dev/null 2>&1; then
  NCU_ARGS=(--csv --page raw --metrics "$METRICS" --target-processes all)
  [ -z "$KERNEL_NAME" ] || NCU_ARGS+=(--kernel-name "$KERNEL_NAME")
  [ -z "$LAUNCH_COUNT" ] || NCU_ARGS+=(--launch-count "$LAUNCH_COUNT")
  STDERR_FILE="$(mktemp)"
  "$NCU_BINARY" "${NCU_ARGS[@]}" \
      "${RUN_CMD[@]}" >"$OUT" 2>"$STDERR_FILE"
  RC=$?
  [ -s "$STDERR_FILE" ] && cat "$STDERR_FILE" 1>&2
  rm -f "$STDERR_FILE"
  if [ "$RC" -ne 0 ]; then
    if grep -q 'ERR_NVGPUCTRPERM' "$OUT"; then
      [ -z "$NSYS_OUT" ] || run_nsys "counter_permission_denied"
      emit '{"ok":false,"profiler":"ncu","native_profile":null,"error_code":"counter_permission_denied","error":"ERR_NVGPUCTRPERM: GPU performance counters denied"}'
      exit 4
    fi
    emit "{\"ok\":false,\"profiler\":\"ncu\",\"native_profile\":null,\"error_code\":\"ncu_failed\",\"error\":\"ncu failed (exit $RC); no profile produced\"}"
    exit 4
  fi
  # Use the same parser that normalizes the profile; a zero NCU exit with no
  # parseable target kernel is not a measurement (NCU 2026 uses wide CSV).
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if ! python3 "$SCRIPT_DIR/../_evidence_nvidia.py" --native "$OUT" \
       --source-backend cuda >/dev/null 2>&1
  then
    emit '{"ok":false,"profiler":"ncu","native_profile":null,"error":"ncu returned no parseable kernel metrics"}'
    exit 4
  fi
  ESC_OUT="$(json_escape "$OUT")"
  emit "{\"ok\":true,\"profiler\":\"ncu\",\"native_profile\":$ESC_OUT,\"format\":\"ncu-csv\"}"
  exit 0
fi

# --- nsys fallback (timeline + kernel latency; no dram/sm/occupancy counters) -
run_nsys "ncu_unavailable"
