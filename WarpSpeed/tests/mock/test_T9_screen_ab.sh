#!/usr/bin/env bash
# T9: tier-1 screening. A/A must report ~0 delta (within noise); A/B with a
# known 6.25% mock delta must recover it; refuses to run outside gpu_run.
. "$(dirname "$0")/lib.sh"
ensure_mockproj P850 850.0 balanced
ensure_mockproj P800 800.0 balanced

SCREEN="$WS_ROOT/infra/bench_screen.sh"

# Refusal outside a gpu_run session
if "$SCREEN" "$P850_BIN" "$P850_BIN" >/dev/null 2>&1; then
  fail "bench_screen ran without GPU_RUN_DEVICE (guard missing)"
fi

AA=$("$GPU_RUN" pool -- "$SCREEN" "$P850_BIN" "$P850_BIN" --reps 20 --blocks 4 2>/dev/null)
note "A/A: $AA"
DELTA=$(json_get "$AA" rel_speedup_pct)
NOISE=$(json_get "$AA" within_device_std_pct)
python3 -c "exit(0 if abs($DELTA) <= 3 * max($NOISE, 0.05) else 1)" \
  || fail "A/A delta $DELTA% exceeds 3x noise $NOISE%"

AB=$("$GPU_RUN" pool -- "$SCREEN" "$P850_BIN" "$P800_BIN" --reps 20 --blocks 4 2>/dev/null)
note "A/B: $AB"
DELTA=$(json_get "$AB" rel_speedup_pct)
python3 -c "exit(0 if 4.5 <= $DELTA <= 8.0 else 1)" \
  || fail "A/B delta $DELTA% not within [4.5, 8.0] (true: 6.25%)"
echo "PASS T9"
