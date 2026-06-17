# Shared helpers for WarpSpeed mock acceptance tests. Source me.
# Each test gets an isolated scratch dir (locks, mock state) and the mock
# tools prepended to PATH, so tests can run standalone or via the runner.

set -u

WS_ROOT=${WS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
GPU_RUN="$WS_ROOT/infra/gpu_run"
WSLOCK="$WS_ROOT/infra/wslock.py"

TDIR=$(mktemp -d "${TMPDIR:-/tmp}/ws-test.XXXXXX")
export WARPSPEED_LOCK_DIR="$TDIR/locks"
export WARPSPEED_MOCK_DIR="$TDIR/mock"
export WARPSPEED_WARMUP_REPS=2
export PATH="$WS_ROOT/tests/mock/bin:$PATH"
unset WARPSPEED_STATE WARPSPEED_LOCKED_GR WARPSPEED_LOCKED_MEM 2>/dev/null || true

cleanup() { rm -rf "$TDIR"; }
trap cleanup EXIT

fail() { echo "FAIL($(basename "$0")): $*" >&2; exit 1; }
note() { echo "  - $*"; }

# ensure_mockproj <var_prefix> <lat> <profile>: builds (or reuses) a mock
# project; sets <var_prefix>_DIR and <var_prefix>_BIN.
ensure_mockproj() {
  _pfx=$1; _lat=$2; _prof=$3
  _dir_var="${_pfx}_DIR"; _bin_var="${_pfx}_BIN"
  _existing=$(eval "echo \${${_dir_var}:-}")
  if [ -z "$_existing" ]; then
    _dir="$TDIR/proj_${_pfx}"
    "$WS_ROOT/tests/mock/mkmockproj.sh" "$_dir" "$_lat" "$_prof" >/dev/null
    (cd "$_dir" && ./build.sh >/dev/null)
    eval "$_dir_var=\"$_dir\""
  else
    eval "_dir=\"$_existing\""
  fi
  eval "$_bin_var=\"$_dir/bin/kernel_bench\""
}

# track <logfile> <cmd...>: run cmd, bracketing with start/end timestamps.
track() {
  _log=$1; shift
  echo "start $(python3 -c 'import time; print(time.time())')" >> "$_log"
  "$@"
  _rc=$?
  echo "end $(python3 -c 'import time; print(time.time())')" >> "$_log"
  return $_rc
}

# peak_concurrency <logfile> -> prints "jobs peak"
peak_concurrency() {
  python3 - "$1" <<'PY'
import sys
ev = []
for line in open(sys.argv[1]):
    kind, ts = line.split()
    ev.append((float(ts), 1 if kind == "start" else -1))
ev.sort(key=lambda e: (e[0], e[1]))   # at a tie, count the start first (conservative)
cur = peak = jobs = 0
for _, d in ev:
    cur += d
    peak = max(peak, cur)
    if d == 1:
        jobs += 1
print(jobs, peak)
PY
}

json_get() {  # json_get '<json>' key  (dot path, e.g. clocks.graphics_mhz)
  python3 - "$2" <<PY
import json, sys
o = json.loads('''$1''')
for part in sys.argv[1].split("."):
    o = o[part] if not isinstance(o, list) else o[int(part)]
print(o)
PY
}
