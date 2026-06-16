#!/usr/bin/env bash
# ncu_profile.sh <binary> <out_json> [--full] [--shape S] [--reps N]
#
# Curated-section NCU profiling. Runs inside `gpu_run ncu` (device 7). The
# curated sections keep profile cost bounded; `--full` (analyst escalation
# only) switches to `--set full`.
#
# Produces <out_json> = parsed key metrics + fingerprint + artifact paths, and
# prints the same JSON to stdout. Checkpoint caching is the CALLER's job:
# check <state>/ncu_cache/<commit>.json before invoking this.

set -eu

if [ -z "${GPU_RUN_DEVICE:-}" ]; then
  echo '{"ok": false, "error": "ncu_profile.sh must run inside a gpu_run session. Use: gpu_run ncu -- ncu_profile.sh ..."}' >&2
  exit 3
fi
if [ "${WARPSPEED_GPU_ROLE:-}" != "ncu" ] && [ "${WARPSPEED_ALLOW_ANY_ROLE:-0}" != "1" ]; then
  echo '{"ok": false, "error": "ncu_profile.sh must run under the ncu role (gpu_run ncu -- ...)"}' >&2
  exit 3
fi

BIN=${1:?usage: ncu_profile.sh <binary> <out_json> [--full] [--shape S] [--reps N]}
OUT=${2:?out_json path required}
shift 2

FULL=0
SHAPE=default
REPS=3
while [ $# -gt 0 ]; do
  case "$1" in
    --full)  FULL=1; shift ;;
    --shape) SHAPE=$2; shift 2 ;;
    --reps)  REPS=$2; shift 2 ;;
    *) echo "ncu_profile.sh: unknown arg $1" >&2; exit 2 ;;
  esac
done

[ -x "$BIN" ] || { echo "{\"ok\": false, \"error\": \"binary not executable: $BIN\"}" >&2; exit 2; }

SELF_DIR=$(cd "$(dirname "$0")" && pwd)
TOOLS="$SELF_DIR/../tools"
SECTIONS=${WARPSPEED_NCU_SECTIONS:-SpeedOfLight,MemoryWorkloadAnalysis,SchedulerStats,WarpStateStats,Occupancy,LaunchStats}

REP=${OUT%.json}
mkdir -p "$(dirname "$OUT")"

set --
if [ "$FULL" = "1" ]; then
  set -- --set full
else
  for s in $(echo "$SECTIONS" | tr ',' ' '); do
    set -- "$@" --section "$s"
  done
fi

ncu "$@" -o "$REP" -f --target-processes all \
  "$BIN" --reps "$REPS" --warmup 2 --shape "$SHAPE" >&2

ncu --import "$REP.ncu-rep" --csv --page details > "$REP.csv"

python3 "$TOOLS/ncu_parse.py" "$REP.csv" > "$REP.parsed.json"
FP_JSON=$(python3 "$TOOLS/ncu_fingerprint.py" "$REP.parsed.json")

python3 - "$REP.parsed.json" "$OUT" "$REP" "$SECTIONS" "$FULL" <<PY
import json, sys
parsed = json.load(open(sys.argv[1]))
fp = json.loads('''$FP_JSON''')
parsed["fingerprint"] = fp["fingerprint"]
parsed["fingerprint_buckets"] = fp["buckets"]
parsed["ncu_rep_path"] = sys.argv[3] + ".ncu-rep"
parsed["csv_path"] = sys.argv[3] + ".csv"
parsed["sections"] = sys.argv[4].split(",") if sys.argv[5] == "0" else ["full"]
parsed["full_set"] = sys.argv[5] == "1"
parsed["ok"] = True
json.dump(parsed, open(sys.argv[2], "w"))
print(json.dumps(parsed))
PY
