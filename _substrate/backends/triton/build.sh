#!/usr/bin/env bash
# triton/build.sh — "build" = JIT warmup: import the kernel + trigger one launch so
# Triton materializes PTX into TRITON_CACHE_DIR. The cache dir IS the artifact.
# Same envelope as cuda/build.sh. Spec §4.5/§5.1.
#   exit 0 ok · 2 warmup failure (json printed) · 3 bad args.
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
  emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"error\":$_msg}"; exit 3
}

SOURCE="" OUT="" EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --source) SOURCE="${2:-}"; shift 2 ;;
    --out)    OUT="${2:-}"; shift 2 ;;
    --extra)  EXTRA="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$SOURCE" ] || die3 "missing --source"
[ -n "$OUT" ]    || die3 "missing --out"
[ -f "$SOURCE" ] || die3 "source not found: $SOURCE"
command -v python3 >/dev/null 2>&1 || die3 "python3 not found"

# TRITON_CACHE_DIR receives the compiled PTX. Use --out as the cache root.
export TRITON_CACHE_DIR="$OUT"
mkdir -p "$OUT" 2>/dev/null

START="$(python3 -c 'import time;print(int(time.time()*1000))')"
STDERR_FILE="$(mktemp)"
# Warmup: import triton + the kernel module + run one launch. On macOS triton is absent,
# so this exits non-zero with an ImportError captured into stderr.
python3 - "$SOURCE" 2>"$STDERR_FILE" <<'PY'
import sys, importlib.util
src = sys.argv[1]
import triton  # noqa: F401  (absent on macOS -> ImportError, caught by exit code)
spec = importlib.util.spec_from_file_location("candidate_kernel", src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# TODO (GPU tier): a real warmup MUST launch the @triton.jit kernel (mod.forward(...) with real args) to force PTX materialization; importing the module alone leaves TRITON_CACHE_DIR empty. Until then a 'successful' build may point at an empty cache dir.
PY
RC=$?
END="$(python3 -c 'import time;print(int(time.time()*1000))')"
BUILD_MS=$(( END - START ))
ESC_TAIL="$(python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))' <"$STDERR_FILE")"
cat "$STDERR_FILE" 1>&2
rm -f "$STDERR_FILE"

if [ "$RC" -eq 0 ]; then
  ESC_OUT="$(json_escape "$OUT")"
  emit "{\"ok\":true,\"compiled\":true,\"artifact\":$ESC_OUT,\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":\"\"}"
  exit 0
else
  emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":$ESC_TAIL}"
  exit 2
fi
