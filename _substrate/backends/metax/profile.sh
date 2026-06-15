#!/usr/bin/env bash
# metax/profile.sh — profile the compiled .so with mcTracer + mcProfiler (MetaX toolchain).
# mcTracer collects runtime traces; mcProfiler produces CSV metric output.
# Prints the POINTER (not the raw metrics): {ok,profiler:"mcprof",native_profile,format:"mcprof-csv"}
# Spec §4.5. exit 0 ok · 3 bad args / missing input · 4 profiler unavailable.
set -u

emit() { printf '%s\n' "$1"; }

json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

die3() {
  local _msg _prof; _msg="$(json_escape "$1")"; _prof="$(json_escape "${2:-mcprof}")"
  emit "{\"ok\":false,\"profiler\":$_prof,\"native_profile\":null,\"error\":$_msg}"; exit 3
}
die4() {
  local _msg _prof; _msg="$(json_escape "$1")"; _prof="$(json_escape "${2:-mcprof}")"
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

# --- mcProfiler path (preferred) --------------------------------------------
if command -v mcProfiler >/dev/null 2>&1; then
  STDERR_FILE="$(mktemp)"
  # mcProfiler CSV output: kernel name, elapsed cycles, DRAM read/write bandwidth,
  # SM utilization, etc. The exact column set depends on the MACA SDK version;
  # _evidence_metax.py normalises column names before mapping to canonical keys.
  mcProfiler --csv --target "$ARTIFACT" >"$OUT" 2>"$STDERR_FILE"
  RC=$?
  [ -s "$STDERR_FILE" ] && cat "$STDERR_FILE" 1>&2
  rm -f "$STDERR_FILE"
  if [ "$RC" -ne 0 ]; then
    emit "{\"ok\":false,\"profiler\":\"mcprof\",\"native_profile\":null,\"error\":\"mcProfiler failed (exit $RC); no profile produced\"}"
    exit 4
  fi
  ESC_OUT="$(json_escape "$OUT")"
  emit "{\"ok\":true,\"profiler\":\"mcprof\",\"native_profile\":$ESC_OUT,\"format\":\"mcprof-csv\"}"
  exit 0
fi

# --- mcTracer fallback (runtime trace; limited metric set) ------------------
if command -v mcTracer >/dev/null 2>&1; then
  STDERR_FILE="$(mktemp)"
  # mcTracer produces a runtime trace that includes kernel launch latencies
  # but limited hardware counter detail. Accept as degraded profile.
  mcTracer --output "$OUT" -- "$ARTIFACT" >"$STDERR_FILE" 2>&1
  RC=$?
  [ -s "$STDERR_FILE" ] && cat "$STDERR_FILE" 1>&2
  rm -f "$STDERR_FILE"
  if [ "$RC" -ne 0 ] || [ ! -s "$OUT" ]; then
    emit "{\"ok\":false,\"profiler\":\"mctrace\",\"native_profile\":null,\"error\":\"mcTracer failed (exit $RC); no trace produced\"}"
    exit 4
  fi
  ESC_OUT="$(json_escape "$OUT")"
  emit "{\"ok\":true,\"profiler\":\"mctrace\",\"native_profile\":$ESC_OUT,\"format\":\"mctrace-log\"}"
  exit 0
fi

die4 "neither mcProfiler nor mcTracer available (is MACA SDK installed?)" "mcprof"