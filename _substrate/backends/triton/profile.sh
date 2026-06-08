#!/usr/bin/env bash
# triton/profile.sh — run ncu over the python launcher and write ncu --csv to --out.
# TODO (GPU tier): CAVEAT (Triton kernel-name discovery): Triton mangles the kernel symbol name and stores
# the compiled object in TRITON_CACHE_DIR, so ncu's --kernel-name regex cannot be derived
# statically. For now we accept an explicit --kernel-name; if omitted we profile ALL
# kernels (--kernel-name left unset). Auto-discovery from TRITON_CACHE_DIR is deferred.
# Same pointer envelope as cuda/profile.sh. Spec §4.5/§5.1.
#   exit 0 ok · 2 ncu error · 3 bad args / missing input · 4 ncu unavailable.
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

ARTIFACT="" PROBLEM="" OUT="" KERNEL_NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact)    ARTIFACT="${2:-}"; shift 2 ;;
    --problem)     PROBLEM="${2:-}"; shift 2 ;;
    --out)         OUT="${2:-}"; shift 2 ;;
    --kernel-name) KERNEL_NAME="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$ARTIFACT" ] || die3 "missing --artifact"
[ -n "$PROBLEM" ]  || die3 "missing --problem"
[ -n "$OUT" ]      || die3 "missing --out"
[ -e "$ARTIFACT" ] || die3 "artifact not found: $ARTIFACT"
[ -f "$PROBLEM" ]  || die3 "problem not found: $PROBLEM"
command -v ncu >/dev/null 2>&1 || die4 "ncu profiler not available"

# The four counters to_evidence/_evidence_nvidia.py parses (AccelOpt set).
METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.pct_of_peak_sustained_elapsed,dram__bytes_write.sum.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

# TRITON_CACHE_DIR points at the build artifact so the launcher reuses the warmed PTX.
export TRITON_CACHE_DIR="$ARTIFACT"

STDERR_FILE="$(mktemp)"

# bash 3.2 (macOS default) aborts on "${ARR[@]}" for an EMPTY array under `set -u`, so inline
# the two branches instead of expanding a possibly-empty array. Also pass --page raw so live ncu
# emits the long format _evidence_nvidia._parse_ncu_csv reads.
if [ -n "$KERNEL_NAME" ]; then
  ncu --csv --page raw --metrics "$METRICS" --target-processes all --kernel-name "$KERNEL_NAME" \
      python3 -c "pass" >"$OUT" 2>"$STDERR_FILE"
else
  ncu --csv --page raw --metrics "$METRICS" --target-processes all \
      python3 -c "pass" >"$OUT" 2>"$STDERR_FILE"
fi
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
