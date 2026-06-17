#!/usr/bin/env bash
# T3: ncu role serializes strictly on the NCU device.
. "$(dirname "$0")/lib.sh"

LOG="$TDIR/ncu.log"
for i in 1 2 3; do
  "$GPU_RUN" ncu --label "t3-$i" -- sh -c \
    "echo start \$(python3 -c 'import time; print(time.time())') >> '$LOG'; sleep 0.5; echo end \$(python3 -c 'import time; print(time.time())') >> '$LOG'" \
    2>/dev/null &
done
wait

read -r JOBS PEAK <<EOF
$(peak_concurrency "$LOG")
EOF
note "jobs=$JOBS peak=$PEAK"
[ "$JOBS" = "3" ] || fail "expected 3 completed ncu jobs, got $JOBS"
[ "$PEAK" = "1" ] || fail "ncu jobs overlapped (peak=$PEAK)"

# pool jobs must NOT contend with the ncu device: a pool job during an ncu hold
# should still start immediately.
"$GPU_RUN" ncu -- sleep 2 2>/dev/null &
NCU_PID=$!
sleep 0.3
START=$(python3 -c 'import time; print(time.time())')
"$GPU_RUN" pool -- true 2>/dev/null
TOOK=$(python3 -c "import time; print(time.time() - $START)")
wait $NCU_PID
note "pool-during-ncu took ${TOOK}s"
python3 -c "exit(0 if $TOOK < 1.0 else 1)" || fail "pool job blocked behind ncu device hold"
echo "PASS T3"
