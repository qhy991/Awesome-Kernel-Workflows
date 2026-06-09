#!/usr/bin/env bash
# rocm/profile.sh — run rocprofv3 over the artifact and write a CSV report to --out.
# Prints the POINTER (not the metrics): {ok,profiler:"rocprofv3",native_profile,format:"rocprof-csv"}.
# Spec §4.5. exit 0 ok · 3 bad args / missing input · 4 profiler unavailable.
#
# Prefers rocprofv3 (ROCm 6.x+). Falls back to rocprof (ROCm 5.x) if v3 is absent.
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
  emit "{\"ok\":false,\"profiler\":\"rocprofv3\",\"native_profile\":null,\"error\":$_msg}"; exit 3
}
die4() {
  local _msg; _msg="$(json_escape "$1")"
  emit "{\"ok\":false,\"profiler\":\"rocprofv3\",\"native_profile\":null,\"error\":$_msg}"; exit 4
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

# Detect profiler: prefer rocprofv3, fall back to rocprof
PROFILER=""
if command -v rocprofv3 >/dev/null 2>&1; then
  PROFILER="rocprofv3"
elif command -v rocprof >/dev/null 2>&1; then
  PROFILER="rocprof"
else
  die4 "rocprofv3/rocprof profiler not available (ROCm not installed?)"
fi

STDERR_FILE="$(mktemp)"

# AMD counters we need for to_evidence/_evidence_amd.py:
#   KernelDuration (ns), VALUBusy (%), MemUnitBusy (%), Wavefronts, MaxWavefronts
COUNTERS="KernelDuration,VALUBusy,MemUnitBusy,Wavefronts,MaxWavefronts"

if [ "$PROFILER" = "rocprofv3" ]; then
  # rocprofv3 --stats --csv writes a CSV to stdout
  rocprofv3 --stats --csv --counters "$COUNTERS" \
      python3 -c "pass" >"$OUT" 2>"$STDERR_FILE"
  RC=$?
else
  # rocprof (v1/v2) fallback: --stats flag + pmc counters file
  PMC_FILE="$(mktemp)"
  printf 'pmc: %s\n' "$COUNTERS" > "$PMC_FILE"
  rocprof --stats --csv -i "$PMC_FILE" \
      python3 -c "pass" >"$OUT" 2>"$STDERR_FILE"
  RC=$?
  rm -f "$PMC_FILE"
fi

[ -s "$STDERR_FILE" ] && cat "$STDERR_FILE" 1>&2
rm -f "$STDERR_FILE"

if [ "$RC" -ne 0 ]; then
  emit "{\"ok\":false,\"profiler\":\"$PROFILER\",\"native_profile\":null,\"error\":\"$PROFILER failed (exit $RC); no profile produced\"}"
  exit 4
fi

ESC_OUT="$(json_escape "$OUT")"
emit "{\"ok\":true,\"profiler\":\"$PROFILER\",\"native_profile\":$ESC_OUT,\"format\":\"rocprof-csv\"}"
exit 0
