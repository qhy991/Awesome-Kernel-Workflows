#!/usr/bin/env bash
# bench_screen.sh <parent_binary> <candidate_binary> [--reps N] [--warmup W]
#                 [--blocks B] [--shape S]
#
# Tier-1 screening benchmark: parent and candidate run back-to-back on the
# SAME device in the same gpu_run session, in interleaved blocks
# (A B A B ... ), so device identity, thermal state, and neighbor interference
# hit both sides equally - the RELATIVE speedup is device-unbiased. That
# relative number (plus its noise) is all this script reports; routing policy
# lives in workflow.js, not here.
#
# Binary contract (see harness-template/README.md):
#   <binary> --reps N --warmup W [--shape S]   prints one "LAT_US=<float>" per rep
#
# REFUSES to run outside a gpu_run session (GPU_RUN_DEVICE must be set): this
# is the §9-replacement guard that makes bare-GPU benchmarking unrepresentable.

set -eu

if [ -z "${GPU_RUN_DEVICE:-}" ]; then
  echo '{"ok": false, "error": "bench_screen.sh must run inside a gpu_run session (GPU_RUN_DEVICE unset). Use: gpu_run pool -- bench_screen.sh ..."}' >&2
  exit 3
fi

PARENT=${1:?usage: bench_screen.sh <parent_binary> <candidate_binary> [--reps N] [--warmup W] [--blocks B] [--shape S]}
CAND=${2:?candidate binary required}
shift 2

REPS=${WARPSPEED_SCREEN_REPS:-50}
WARMUP=${WARPSPEED_WARMUP_REPS:-10}
BLOCKS=5
SHAPE=default
while [ $# -gt 0 ]; do
  case "$1" in
    --reps)   REPS=$2; shift 2 ;;
    --warmup) WARMUP=$2; shift 2 ;;
    --blocks) BLOCKS=$2; shift 2 ;;
    --shape)  SHAPE=$2; shift 2 ;;
    *) echo "bench_screen.sh: unknown arg $1" >&2; exit 2 ;;
  esac
done

[ -x "$PARENT" ] || { echo "{\"ok\": false, \"error\": \"parent binary not executable: $PARENT\"}" >&2; exit 2; }
[ -x "$CAND" ]   || { echo "{\"ok\": false, \"error\": \"candidate binary not executable: $CAND\"}" >&2; exit 2; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/ws-screen.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

PER_BLOCK=$(( REPS / BLOCKS ))
[ "$PER_BLOCK" -ge 1 ] || PER_BLOCK=1

b=0
while [ "$b" -lt "$BLOCKS" ]; do
  if [ "$b" -eq 0 ]; then W=$WARMUP; else W=2; fi
  "$PARENT" --reps "$PER_BLOCK" --warmup "$W" --shape "$SHAPE" | grep '^LAT_US=' >> "$TMP/parent.txt"
  "$CAND"   --reps "$PER_BLOCK" --warmup "$W" --shape "$SHAPE" | grep '^LAT_US=' >> "$TMP/cand.txt"
  b=$(( b + 1 ))
done

python3 - "$TMP/parent.txt" "$TMP/cand.txt" "$GPU_RUN_DEVICE" "$SHAPE" "$PER_BLOCK" "$BLOCKS" <<'PY'
import json, statistics, sys

def load(p):
    vals = []
    for line in open(p):
        if line.startswith("LAT_US="):
            vals.append(float(line.split("=", 1)[1]))
    return vals

parent, cand = load(sys.argv[1]), load(sys.argv[2])
if not parent or not cand:
    print(json.dumps({"ok": False, "error": "no LAT_US lines captured"}))
    sys.exit(1)

pm, cm = statistics.fmean(parent), statistics.fmean(cand)
ps = statistics.stdev(parent) if len(parent) > 1 else 0.0
cs = statistics.stdev(cand) if len(cand) > 1 else 0.0
out = {
    "ok": True,
    "parent_mean_us": round(pm, 3), "parent_std_pct": round(100 * ps / pm, 4),
    "cand_mean_us": round(cm, 3), "cand_std_pct": round(100 * cs / cm, 4),
    "rel_speedup_pct": round((pm / cm - 1.0) * 100, 4),
    "within_device_std_pct": round(max(100 * ps / pm, 100 * cs / cm), 4),
    "reps_each": len(parent), "blocks": int(sys.argv[6]),
    "device": sys.argv[3], "shape": sys.argv[4],
}
print(json.dumps(out))
PY
