#!/usr/bin/env bash
# T2: bench role serializes strictly; clocks are locked before and restored
# after EVERY job (asserted via the mock nvidia-smi call log + state files).
. "$(dirname "$0")/lib.sh"

export WARPSPEED_LOCKED_GR=1500 WARPSPEED_LOCKED_MEM=2000
LOG="$TDIR/bench.log"
for i in 1 2 3; do
  "$GPU_RUN" bench --label "t2-$i" -- sh -c \
    "echo start \$(python3 -c 'import time; print(time.time())') >> '$LOG'; sleep 0.6; echo end \$(python3 -c 'import time; print(time.time())') >> '$LOG'" \
    2>/dev/null &
done
wait

read -r JOBS PEAK <<EOF
$(peak_concurrency "$LOG")
EOF
note "jobs=$JOBS peak=$PEAK"
[ "$JOBS" = "3" ] || fail "expected 3 completed bench jobs, got $JOBS"
[ "$PEAK" = "1" ] || fail "bench jobs overlapped (peak=$PEAK)"

SMI_LOG="$WARPSPEED_MOCK_DIR/nvidia-smi.log"
[ -f "$SMI_LOG" ] || fail "mock nvidia-smi never invoked"
LGC=$(grep -c '"-lgc"' "$SMI_LOG" || true)
RGC=$(grep -c '"-rgc"' "$SMI_LOG" || true)
LMC=$(grep -c '"-lmc"' "$SMI_LOG" || true)
RMC=$(grep -c '"-rmc"' "$SMI_LOG" || true)
note "lgc=$LGC rgc=$RGC lmc=$LMC rmc=$RMC"
[ "$LGC" = "3" ] && [ "$RGC" = "3" ] || fail "clock lock/restore not paired per job (lgc=$LGC rgc=$RGC)"
[ "$LMC" = "3" ] && [ "$RMC" = "3" ] || fail "mem clock lock/restore not paired (lmc=$LMC rmc=$RMC)"
# After all jobs, no device may be left in a locked-clock state.
if ls "$WARPSPEED_MOCK_DIR/state/"gpu*_lgc >/dev/null 2>&1; then fail "graphics clocks left locked after bench jobs"; fi
if ls "$WARPSPEED_MOCK_DIR/state/"gpu*_lmc >/dev/null 2>&1; then fail "memory clocks left locked after bench jobs"; fi
echo "PASS T2"
