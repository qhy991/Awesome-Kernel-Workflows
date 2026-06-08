#!/usr/bin/env bash
# cuda/build.sh — compile a .cu kernel to a .so via nvcc (or a --build-cmd template).
# Spec §4.5. Universal envelope: ONE json on stdout, logs to stderr.
#   exit 0 ok · 2 compile failure (json printed) · 3 bad args / missing tool.
# -lineinfo is REQUIRED for ncu source attribution.
set -u

emit() { printf '%s\n' "$1"; }   # one-line JSON on stdout
die3() { emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"error\":\"$1\"}"; exit 3; }

SOURCE="" OUT="" ARCH="sm_80" BUILD_CMD="" EXTRA=""
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

# JSON-escape helper for the stderr tail (escape backslash, quote, strip control chars).
json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

# Resolve the build command. Default: nvcc -shared -Xcompiler -fPIC -lineinfo -arch=... -o OUT SOURCE
if [ -n "$BUILD_CMD" ]; then
  # template tokens: {source} {out} {arch} {extra}
  CMD="${BUILD_CMD//\{source\}/$SOURCE}"
  CMD="${CMD//\{out\}/$OUT}"
  CMD="${CMD//\{arch\}/$ARCH}"
  CMD="${CMD//\{extra\}/$EXTRA}"
  # -lineinfo must be present for ncu attribution; inject if the template omitted it.
  case "$CMD" in *-lineinfo*) : ;; *) CMD="$CMD -lineinfo" ;; esac
else
  command -v nvcc >/dev/null 2>&1 || die3 "nvcc not found on PATH"
  CMD="nvcc -shared -Xcompiler -fPIC -lineinfo -arch=$ARCH $EXTRA -o $OUT $SOURCE"
fi

# If a template was given, still verify its leading tool exists.
TOOL="${CMD%% *}"
command -v "$TOOL" >/dev/null 2>&1 || die3 "build tool not found: $TOOL"

# Wall-time the compile (ms). Use python3 for portable millisecond timing.
START="$(python3 -c 'import time;print(int(time.time()*1000))')"
STDERR_FILE="$(mktemp)"
# shellcheck disable=SC2086
eval $CMD 2>"$STDERR_FILE"
RC=$?
END="$(python3 -c 'import time;print(int(time.time()*1000))')"
BUILD_MS=$(( END - START ))

# Tail of stderr (last 20 lines), JSON-escaped.
TAIL="$(tail -n 20 "$STDERR_FILE")"
ESC_TAIL="$(json_escape "$TAIL")"
cat "$STDERR_FILE" 1>&2
rm -f "$STDERR_FILE"

if [ "$RC" -eq 0 ] && [ -f "$OUT" ]; then
  emit "{\"ok\":true,\"compiled\":true,\"artifact\":\"$OUT\",\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":\"\"}"
  exit 0
else
  emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":$ESC_TAIL}"
  exit 2
fi
