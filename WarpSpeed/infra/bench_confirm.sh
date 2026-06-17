#!/usr/bin/env bash
# bench_confirm.sh <binary> [--reps N] [--warmup W] [--shape S]
#
# Tier-2 confirmation benchmark: canonical absolute latency for the
# leaderboard. Must run inside `gpu_run bench` (device + clock locking are the
# bench role's job; this script just measures and records the actual clocks).

set -eu

if [ -z "${GPU_RUN_DEVICE:-}" ]; then
  echo '{"ok": false, "error": "bench_confirm.sh must run inside a gpu_run session. Use: gpu_run bench -- bench_confirm.sh ..."}' >&2
  exit 3
fi

BIN=${1:?usage: bench_confirm.sh <binary> [--reps N] [--warmup W] [--shape S]}
shift

REPS=${WARPSPEED_CONFIRM_REPS:-200}
WARMUP=${WARPSPEED_WARMUP_REPS:-10}
SHAPE=default
while [ $# -gt 0 ]; do
  case "$1" in
    --reps)   REPS=$2; shift 2 ;;
    --warmup) WARMUP=$2; shift 2 ;;
    --shape)  SHAPE=$2; shift 2 ;;
    *) echo "bench_confirm.sh: unknown arg $1" >&2; exit 2 ;;
  esac
done

[ -x "$BIN" ] || { echo "{\"ok\": false, \"error\": \"binary not executable: $BIN\"}" >&2; exit 2; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/ws-confirm.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

"$BIN" --reps "$REPS" --warmup "$WARMUP" --shape "$SHAPE" | grep '^LAT_US=' > "$TMP/lat.txt"
nvidia-smi -q -d CLOCK -i "$GPU_RUN_DEVICE" > "$TMP/clocks.txt" 2>/dev/null || true

python3 - "$TMP/lat.txt" "$TMP/clocks.txt" "$GPU_RUN_DEVICE" "$SHAPE" <<'PY'
import json, re, statistics, sys

vals = [float(l.split("=", 1)[1]) for l in open(sys.argv[1]) if l.startswith("LAT_US=")]
if not vals:
    print(json.dumps({"ok": False, "error": "no LAT_US lines captured"}))
    sys.exit(1)

clocks = {"graphics_mhz": None, "memory_mhz": None}
try:
    txt = open(sys.argv[2]).read()
    g = re.search(r"Graphics\s*:\s*(\d+)\s*MHz", txt)
    m = re.search(r"Memory\s*:\s*(\d+)\s*MHz", txt)
    if g: clocks["graphics_mhz"] = int(g.group(1))
    if m: clocks["memory_mhz"] = int(m.group(1))
except OSError:
    pass

mean = statistics.fmean(vals)
std = statistics.stdev(vals) if len(vals) > 1 else 0.0
print(json.dumps({
    "ok": True,
    "latency_us_mean": round(mean, 3),
    "latency_us_std": round(std, 4),
    "latency_us_std_pct": round(100 * std / mean, 4),
    "reps": len(vals), "device": sys.argv[3], "shape": sys.argv[4],
    "clocks": clocks,
}))
PY
