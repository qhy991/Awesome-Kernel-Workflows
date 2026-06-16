#!/usr/bin/env bash
# T4: --timeout kills the whole child process group, exits 124, and the lock
# is immediately re-acquirable. Nonzero child exits propagate unchanged.
. "$(dirname "$0")/lib.sh"

MARK="ws-t4-$$"
START=$(python3 -c 'import time; print(time.time())')
WARPSPEED_POOL_DEVICES=0 "$GPU_RUN" pool --timeout 2 -- sh -c "sleep 600 & exec sleep 600 # $MARK" 2>/dev/null
RC=$?
TOOK=$(python3 -c "import time; print(time.time() - $START)")
note "rc=$RC took=${TOOK}s"
[ "$RC" = "124" ] || fail "expected exit 124 on timeout, got $RC"
python3 -c "exit(0 if $TOOK < 8 else 1)" || fail "timeout took too long (${TOOK}s)"

PROBE=$(python3 "$WSLOCK" probe --lock-dir "$WARPSPEED_LOCK_DIR" --devices 0)
echo "$PROBE" | grep -q '"state": "free"' || fail "lock not released after timeout: $PROBE"

sleep 0.5
if pgrep -f "$MARK" >/dev/null 2>&1; then
  pkill -9 -f "$MARK" || true
  fail "child process group survived the timeout kill"
fi

WARPSPEED_POOL_DEVICES=0 "$GPU_RUN" pool -- sh -c 'exit 9' 2>/dev/null
[ "$?" = "9" ] || fail "nonzero child exit not propagated"
echo "PASS T4"
