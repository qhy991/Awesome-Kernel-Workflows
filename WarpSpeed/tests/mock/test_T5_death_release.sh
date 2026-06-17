#!/usr/bin/env bash
# T5: flock is released by the kernel when the holder dies (kill -9) - the
# property that makes orphaned-lock cleanup unnecessary by construction.
. "$(dirname "$0")/lib.sh"

WARPSPEED_POOL_DEVICES=2 "$GPU_RUN" pool --label t5-holder -- sleep 600 2>/dev/null &
HOLDER=$!
sleep 0.8

PROBE=$(python3 "$WSLOCK" probe --lock-dir "$WARPSPEED_LOCK_DIR" --devices 2)
echo "$PROBE" | grep -q '"state": "held"' || fail "holder did not take the lock: $PROBE"

kill -9 "$HOLDER" 2>/dev/null
sleep 0.5
pkill -9 -f "sleep 600" 2>/dev/null || true
sleep 0.3

PROBE=$(python3 "$WSLOCK" probe --lock-dir "$WARPSPEED_LOCK_DIR" --devices 2)
echo "$PROBE" | grep -q '"state": "free"' || fail "lock not released after kill -9: $PROBE"
echo "PASS T5"
