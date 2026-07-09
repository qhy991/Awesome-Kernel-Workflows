#!/usr/bin/env bash
# generic/build.sh — CPU-toolchain fallback compile: source -> shared object (.so).
# Universal envelope: ONE json on stdout, logs to stderr.
#   exit 0 ok · 2 compile failure (json printed) · 3 bad args / missing tool.
#
# Heuristic-by-extension, HONESTY-GATED (workflow-language decoupling design):
#   - A known source extension maps to a CPU toolchain ONLY when that tool is on
#     PATH (.cpp/.cc/.cxx -> g++/clang++; .c -> gcc/clang).
#   - If the extension is unknown OR the mapped tool is absent, an explicit
#     --build-cmd template is required (same mechanism the other drivers support).
#   - With neither, exit 3 (bad args / missing tool) — a clean, recorded failure,
#     never a crash and never a masked "no toolchain here" state.
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

# First tool on PATH from the candidate list; empty if none.
first_on_path() {
  local t
  for t in "$@"; do
    if command -v "$t" >/dev/null 2>&1; then printf '%s' "$t"; return 0; fi
  done
  return 1
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

command -v python3 >/dev/null 2>&1 || die3 "python3 not found on PATH"

if [ -n "$BUILD_CMD" ]; then
  # Explicit template: {source} {out} {arch} {extra} substitution, tool-on-PATH check.
  CMD="${BUILD_CMD//\{source\}/$SOURCE}"
  CMD="${CMD//\{out\}/$OUT}"
  CMD="${CMD//\{arch\}/$ARCH}"
  CMD="${CMD//\{extra\}/$EXTRA}"
  TOOL="${CMD%% *}"
  command -v "$TOOL" >/dev/null 2>&1 || die3 "build tool not found: $TOOL"
else
  # Heuristic-by-extension. Only proceed if the mapped tool actually resolves.
  case "$SOURCE" in
    *.cpp|*.cc|*.cxx|*.C)
      TOOL="$(first_on_path g++ clang++)" \
        || die3 "no C++ toolchain (g++/clang++) on PATH for '$SOURCE'; pass --build-cmd" ;;
    *.c)
      TOOL="$(first_on_path gcc clang)" \
        || die3 "no C toolchain (gcc/clang) on PATH for '$SOURCE'; pass --build-cmd" ;;
    *)
      die3 "unknown source extension for generic build: '$SOURCE' — pass an explicit --build-cmd template" ;;
  esac
  CMD="$TOOL -shared -fPIC -O2 $EXTRA \"$SOURCE\" -o \"$OUT\""
fi

START="$(python3 -c 'import time;print(int(time.time()*1000))')"
STDERR_FILE="$(mktemp)"
bash -c "$CMD" 2>"$STDERR_FILE"
RC=$?
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
