#!/usr/bin/env bash
# rocm/build.sh -- compile a .cu/.hip kernel via hipcc, OR run an arbitrary --build-cmd
# template for in-codebase kernels. Universal envelope (spec SS4.5):
#   one json on stdout, logs to stderr.
#   exit 0 ok . 2 compile failure (json printed) . 3 bad args / missing tool.
# -g (line tables) is REQUIRED for rocprofv3 source attribution.
set -u

emit() { printf '%s\n' "$1"; }

# JSON-escape helper.
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

SOURCE="" OUT="" ARCH="gfx1100" BUILD_CMD="" EXTRA=""
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

# In-codebase mode: a custom --build-cmd is given. Source/out may be omitted
# (e.g. cmake builds an entire project, no single artifact). Skip those checks.
if [ -n "$BUILD_CMD" ]; then
  # template tokens: {source} {out} {arch} {extra}
  CMD="${BUILD_CMD//\{source\}/$SOURCE}"
  CMD="${CMD//\{out\}/$OUT}"
  CMD="${CMD//\{arch\}/$ARCH}"
  CMD="${CMD//\{extra\}/$EXTRA}"

  TOOL="${CMD%% *}"
  command -v "$TOOL" >/dev/null 2>&1 || die3 "build tool not found: $TOOL"

  START="$(python3 -c 'import time;print(int(time.time()*1000))')"
  STDERR_FILE="$(mktemp)"
  # shellcheck disable=SC2086
  bash -c "$CMD" 2>"$STDERR_FILE"
  RC=$?
  END="$(python3 -c 'import time;print(int(time.time()*1000))')"
  COMPILE_MS=$((END - START))

  if [ "$RC" -ne 0 ]; then
    STDERR_TAIL="$(tail -c 4096 "$STDERR_FILE" 2>/dev/null || true)"
    STDERR_JSON="$(json_escape "$STDERR_TAIL")"
    rm -f "$STDERR_FILE"
    OUT_JSON_VAL="null"
    if [ -n "$OUT" ]; then OUT_JSON_VAL="$(json_escape "$OUT")"; fi
    emit "{\"ok\":false,\"compiled\":false,\"artifact\":$OUT_JSON_VAL,\"compile_ms\":$COMPILE_MS,\"stderr_tail\":$STDERR_JSON,\"error\":\"build command exited $RC\"}"
    exit 2
  fi
  rm -f "$STDERR_FILE"

  # For in-codebase mode, "artifact" may be the project's resulting binary or null.
  OUT_JSON_VAL="null"
  if [ -n "$OUT" ] && [ -e "$OUT" ]; then OUT_JSON_VAL="$(json_escape "$OUT")"; fi
  emit "{\"ok\":true,\"compiled\":true,\"artifact\":$OUT_JSON_VAL,\"compile_ms\":$COMPILE_MS}"
  exit 0
fi

# Standalone mode: hipcc on a single .cu/.hip file.
[ -n "$SOURCE" ] || die3 "missing --source"
[ -n "$OUT" ]    || die3 "missing --out"
[ -f "$SOURCE" ] || die3 "source not found: $SOURCE"

command -v hipcc >/dev/null 2>&1 || die3 "hipcc not found on PATH"

START="$(python3 -c 'import time;print(int(time.time()*1000))')"
STDERR_FILE="$(mktemp)"
# -fPIC for shared object; -g for rocprofv3 source attribution; --offload-arch for the GPU.
# shellcheck disable=SC2086
hipcc -shared -fPIC -g --offload-arch="$ARCH" $EXTRA -o "$OUT" "$SOURCE" 2>"$STDERR_FILE"
RC=$?
END="$(python3 -c 'import time;print(int(time.time()*1000))')"
COMPILE_MS=$((END - START))

if [ "$RC" -ne 0 ]; then
  STDERR_TAIL="$(tail -c 4096 "$STDERR_FILE" 2>/dev/null || true)"
  STDERR_JSON="$(json_escape "$STDERR_TAIL")"
  rm -f "$STDERR_FILE"
  OUT_JSON_VAL="$(json_escape "$OUT")"
  emit "{\"ok\":false,\"compiled\":false,\"artifact\":$OUT_JSON_VAL,\"compile_ms\":$COMPILE_MS,\"stderr_tail\":$STDERR_JSON,\"error\":\"hipcc exited $RC\"}"
  exit 2
fi
rm -f "$STDERR_FILE"

OUT_JSON_VAL="$(json_escape "$OUT")"
emit "{\"ok\":true,\"compiled\":true,\"artifact\":$OUT_JSON_VAL,\"compile_ms\":$COMPILE_MS}"
exit 0
