#!/usr/bin/env bash
# T10: tier-2 confirm runs on the bench device with locked clocks, records the
# actual clocks in its output, and clocks are restored afterwards.
. "$(dirname "$0")/lib.sh"
ensure_mockproj P800 800.0 membound

export WARPSPEED_LOCKED_GR=1500 WARPSPEED_LOCKED_MEM=2000
OUT=$("$GPU_RUN" bench -- "$WS_ROOT/infra/bench_confirm.sh" "$P800_BIN" --reps 30 2>/dev/null)
note "$OUT"
[ "$(json_get "$OUT" ok)" = "True" ] || fail "confirm not ok"
[ "$(json_get "$OUT" reps)" = "30" ] || fail "wrong rep count"
[ "$(json_get "$OUT" device)" = "6" ] || fail "confirm not on bench device 6"
[ "$(json_get "$OUT" clocks.graphics_mhz)" = "1500" ] || fail "locked graphics clock not observed during run"
[ "$(json_get "$OUT" clocks.memory_mhz)" = "2000" ] || fail "locked memory clock not observed during run"
MEAN=$(json_get "$OUT" latency_us_mean)
python3 -c "exit(0 if 780 <= $MEAN <= 820 else 1)" || fail "mean $MEAN not near 800us"

# Clocks restored after the session
if ls "$WARPSPEED_MOCK_DIR/state/"gpu6_l* >/dev/null 2>&1; then fail "bench clocks left locked after confirm"; fi

# Refusal outside gpu_run
if "$WS_ROOT/infra/bench_confirm.sh" "$P800_BIN" >/dev/null 2>&1; then
  fail "bench_confirm ran without GPU_RUN_DEVICE (guard missing)"
fi
echo "PASS T10"
