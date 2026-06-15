#!/usr/bin/env bash
# metax/build.sh — compile a .cu kernel to a .so via mxcc (MetaX mxgpu_llvm toolchain).
# mxcc is the MACA SDK compiler frontend; it wraps Clang/LLVM with MetaX GPU target
# lowering. Supports --arch to specify the MetaX GPU ISA target.
# Spec §4.5. Universal envelope: ONE json on stdout, logs to stderr.
#   exit 0 ok · 2 compile failure (json printed) · 3 bad args / missing tool.
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
  emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"error\":$_msg}"; exit 3
}

SOURCE="" OUT="" ARCH="" BUILD_CMD="" EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --source)    SOURCE="${2:-}"; shift 2 ;;
    --out)       OUT="${2:-}"; shift 2 ;;
    --arch)      ARCH="${2:-}"; shift 2 ;;
    --build-cmd) BUILD_CMD="${2:-}"; shift 2 ;;
    --extra)     EXTRA="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$SOURCE" ] || die3 "missing --source"
[ -n "$OUT" ]    || die3 "missing --out"
[ -f "$SOURCE" ] || die3 "source not found: $SOURCE"

# Resolve the build command. Default: mxcc -shared -fPIC -o OUT SOURCE
# ARCH is optional for mxcc (target GPU is detected or defaults to the installed MACA SDK arch).
# When present it is passed as -march=<arch>.
if [ -n "$BUILD_CMD" ]; then
  CMD="${BUILD_CMD//\{source\}/$SOURCE}"
  CMD="${CMD//\{out\}/$OUT}"
  CMD="${CMD//\{arch\}/${ARCH:-}}"
  CMD="${CMD//\{extra\}/${EXTRA:-}}"
  TOOL="${CMD%% *}"
  command -v "$TOOL" >/dev/null 2>&1 || die3 "build tool not found: $TOOL"
  START="$(python3 -c 'import time;print(int(time.time()*1000))')"
  STDERR_FILE="$(mktemp)"
  bash -c "$CMD" 2>"$STDERR_FILE"
  RC=$?
else
  command -v mxcc >/dev/null 2>&1 || die3 "mxcc not found on PATH (is MACA SDK installed?)"

  START="$(python3 -c 'import time;print(int(time.time()*1000))')"
  STDERR_FILE="$(mktemp)"
  if [ -n "$ARCH" ]; then
    mxcc -shared -fPIC -march="$ARCH" $EXTRA -o "$OUT" "$SOURCE" 2>"$STDERR_FILE"
  else
    mxcc -shared -fPIC $EXTRA -o "$OUT" "$SOURCE" 2>"$STDERR_FILE"
  fi
  RC=$?
fi

END="$(python3 -c 'import time;print(int(time.time()*1000))')"
BUILD_MS=$(( END - START ))

TAIL="$(tail -n 20 "$STDERR_FILE")"
ESC_TAIL="$(json_escape "$TAIL")"
cat "$STDERR_FILE" 1>&2
rm -f "$STDERR_FILE"

if [ "$RC" -eq 0 ] && [ -f "$OUT" ]; then
  ESC_OUT="$(json_escape "$OUT")"
  emit "{\"ok\":true,\"compiled\":true,\"artifact\":$ESC_OUT,\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":\"\"}"
  exit 0
else
  emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":$ESC_TAIL}"
  exit 2
fi