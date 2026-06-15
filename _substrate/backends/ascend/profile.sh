#!/usr/bin/env bash
# ascend/profile.sh — profile the kernel with Ascend msprof (CANN profiler).
# Prints the POINTER (not the metrics):
#   {ok,profiler:"msprof",native_profile,format:"msprof-csv"}
# msprof profiles a running application, so it needs a runnable --source launcher (a python
# module that builds inputs and calls the op) OR a --problem + --mkb-root to drive the op.
# Spec §4.5. exit 0 ok · 3 bad args / missing input · 4 profiler unavailable.
#
# NOTE: Ascend has no NPU on the macOS dev box; this path is GPU/NPU-tier and degrades to
# exit 4 (unknown) when msprof or the NPU runtime is absent — honest, never fabricated.
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
  emit "{\"ok\":false,\"profiler\":\"msprof\",\"native_profile\":null,\"error\":$_msg}"; exit 3
}
die4() {
  local _msg; _msg="$(json_escape "$1")"
  emit "{\"ok\":false,\"profiler\":\"msprof\",\"native_profile\":null,\"error\":$_msg}"; exit 4
}

ARTIFACT="" PROBLEM="" OUT="" SOURCE="" OP="${MKB_OP:-}" MKB_ROOT="${MULTIKERNELBENCH_ROOT:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact)     ARTIFACT="${2:-}"; shift 2 ;;
    --kernel)       ARTIFACT="${ARTIFACT:-${2:-}}"; shift 2 ;;  # workflow .js passes --kernel
    --problem)      PROBLEM="${2:-}"; shift 2 ;;
    --out|--result) OUT="${2:-}"; shift 2 ;;
    --source)       SOURCE="${2:-}"; shift 2 ;;
    --op)           OP="${2:-}"; shift 2 ;;
    --mkb-root)     MKB_ROOT="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$OUT" ] || die3 "missing --out/--result"

command -v msprof >/dev/null 2>&1 || die4 "msprof not on PATH (CANN toolkit absent — deferred NPU tier)"

# Build the application command msprof should profile.
if [ -n "$SOURCE" ] && [ -f "$SOURCE" ]; then
  APP_CMD="python3 $SOURCE"
elif [ -n "$ARTIFACT" ] && [ -f "$ARTIFACT" ] && [ -n "$OP" ] && [ -n "$MKB_ROOT" ] && [ -f "$MKB_ROOT/eval_single_runner.py" ]; then
  APP_CMD="python3 $MKB_ROOT/eval_single_runner.py -i $ARTIFACT -o $OP -l ascendc_direct_launch -r $OUT.mkbresult.json"
else
  die4 "msprof needs a runnable --source launcher, or --artifact+--op+--mkb-root; none usable"
fi

# msprof writes a profiling result directory; --output names its parent. We export the
# op-summary CSV as the native profile to_evidence.py will read.
OUT_DIR="${OUT%.csv}_msprof"
mkdir -p "$OUT_DIR"
STDERR_FILE="$(mktemp)"
# shellcheck disable=SC2086
msprof --output="$OUT_DIR" --application="$APP_CMD" >"$STDERR_FILE" 2>&1
RC=$?
[ -s "$STDERR_FILE" ] && cat "$STDERR_FILE" 1>&2
rm -f "$STDERR_FILE"

if [ "$RC" -ne 0 ]; then
  emit "{\"ok\":false,\"profiler\":\"msprof\",\"native_profile\":null,\"error\":\"msprof failed (exit $RC); no profile produced\"}"
  exit 4
fi

# Prefer an op_summary / kernel_details CSV from the msprof export tree.
SUMMARY="$(find "$OUT_DIR" -type f \( -iname '*op_summary*.csv' -o -iname '*kernel_details*.csv' -o -iname '*op_statistic*.csv' \) 2>/dev/null | head -n1)"
if [ -z "$SUMMARY" ]; then
  emit "{\"ok\":false,\"profiler\":\"msprof\",\"native_profile\":null,\"error\":\"msprof produced no op_summary/kernel_details CSV under $OUT_DIR\"}"
  exit 4
fi
cp "$SUMMARY" "$OUT"
ESC_OUT="$(json_escape "$OUT")"
emit "{\"ok\":true,\"profiler\":\"msprof\",\"native_profile\":$ESC_OUT,\"format\":\"msprof-csv\"}"
exit 0
