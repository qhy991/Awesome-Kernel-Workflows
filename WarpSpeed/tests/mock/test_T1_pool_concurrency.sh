#!/usr/bin/env bash
# T1: 12 concurrent `gpu_run pool` jobs -> at most 6 run simultaneously, all complete.
. "$(dirname "$0")/lib.sh"

LOG="$TDIR/conc.log"
for i in $(seq 1 12); do
  "$GPU_RUN" pool --label "t1-$i" -- sh -c \
    "echo start \$(python3 -c 'import time; print(time.time())') >> '$LOG'; sleep 1.2; echo end \$(python3 -c 'import time; print(time.time())') >> '$LOG'" \
    2>/dev/null &
done
wait

read -r JOBS PEAK <<EOF
$(peak_concurrency "$LOG")
EOF
note "jobs=$JOBS peak=$PEAK"
[ "$JOBS" = "12" ] || fail "expected 12 completed jobs, got $JOBS"
[ "$PEAK" -le 6 ] || fail "pool concurrency exceeded 6 (peak=$PEAK)"
[ "$PEAK" -ge 4 ] || fail "suspiciously low concurrency (peak=$PEAK) - pool not parallel?"
echo "PASS T1"
