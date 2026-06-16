#!/usr/bin/env bash
# calibrate.sh <binary> [--reps N] [--warmup W] [--shape S]
#
# One-time cross-device noise measurement: runs the SAME prebuilt binary on
# every device this session holds (run it as `gpu_run cal -- calibrate.sh ...`
# so all locks are held), computes per-device means and the cross-device sigma
# that defines the screening significance margin.
#
# Output JSON is what `wsdb.py calibration-set` ingests.

set -eu

if [ -z "${GPU_RUN_DEVICE:-}" ]; then
  echo '{"ok": false, "error": "calibrate.sh must run inside a gpu_run session. Use: gpu_run cal -- calibrate.sh ..."}' >&2
  exit 3
fi

BIN=${1:?usage: calibrate.sh <binary> [--reps N] [--warmup W] [--shape S]}
shift

REPS=${WARPSPEED_CALIBRATE_REPS:-50}
WARMUP=${WARPSPEED_WARMUP_REPS:-10}
SHAPE=default
while [ $# -gt 0 ]; do
  case "$1" in
    --reps)   REPS=$2; shift 2 ;;
    --warmup) WARMUP=$2; shift 2 ;;
    --shape)  SHAPE=$2; shift 2 ;;
    *) echo "calibrate.sh: unknown arg $1" >&2; exit 2 ;;
  esac
done

[ -x "$BIN" ] || { echo "{\"ok\": false, \"error\": \"binary not executable: $BIN\"}" >&2; exit 2; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/ws-cal.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

GR=${WARPSPEED_LOCKED_GR:-}
MEM=${WARPSPEED_LOCKED_MEM:-}
CP=${WARPSPEED_CLOCK_PREFIX:-}

DEVICES=$(echo "$GPU_RUN_DEVICE" | tr ',' ' ')
for d in $DEVICES; do
  if [ -n "$GR" ];  then ${CP}nvidia-smi -i "$d" -lgc "$GR"  >/dev/null 2>&1 || echo "calibrate: WARN: -lgc failed on $d" >&2; fi
  if [ -n "$MEM" ]; then ${CP}nvidia-smi -i "$d" -lmc "$MEM" >/dev/null 2>&1 || echo "calibrate: WARN: -lmc failed on $d" >&2; fi
  CUDA_VISIBLE_DEVICES=$d "$BIN" --reps "$REPS" --warmup "$WARMUP" --shape "$SHAPE" | grep '^LAT_US=' > "$TMP/dev_$d.txt"
  if [ -n "$GR" ];  then ${CP}nvidia-smi -i "$d" -rgc >/dev/null 2>&1 || true; fi
  if [ -n "$MEM" ]; then ${CP}nvidia-smi -i "$d" -rmc >/dev/null 2>&1 || true; fi
done

python3 - "$TMP" "$BIN" "$REPS" "$SHAPE" <<'PY'
import glob, hashlib, json, os, statistics, sys

tmp, binary, reps, shape = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
means = {}
for p in sorted(glob.glob(os.path.join(tmp, "dev_*.txt"))):
    dev = os.path.basename(p)[4:-4]
    vals = [float(l.split("=", 1)[1]) for l in open(p) if l.startswith("LAT_US=")]
    if vals:
        means[dev] = round(statistics.fmean(vals), 3)

if len(means) < 2:
    print(json.dumps({"ok": False, "error": "need >=2 devices for cross-device sigma", "per_device_means_us": means}))
    sys.exit(1)

vals = list(means.values())
sigma_pct = 100.0 * statistics.stdev(vals) / statistics.fmean(vals)
sha = hashlib.sha256(open(binary, "rb").read()).hexdigest()[:16]
print(json.dumps({
    "ok": True,
    "per_device_means_us": means,
    "cross_device_sigma_pct": round(sigma_pct, 4),
    "reps": reps, "shape": shape, "binary_sha256": sha,
}))
PY
