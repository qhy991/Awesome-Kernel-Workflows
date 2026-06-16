#!/usr/bin/env bash
# T7: calibrate.sh measures all devices and reports a plausible cross-device
# sigma (the mock binary has known per-device offsets of -0.4%..+0.4%).
. "$(dirname "$0")/lib.sh"
ensure_mockproj P850 850.0 balanced

OUT=$("$GPU_RUN" cal -- "$WS_ROOT/infra/calibrate.sh" "$P850_BIN" --reps 12 2>/dev/null)
note "$OUT"
[ "$(json_get "$OUT" ok)" = "True" ] || fail "calibrate not ok"
N=$(python3 -c "import json; print(len(json.loads('''$OUT''')['per_device_means_us']))")
[ "$N" = "8" ] || fail "expected 8 device means, got $N"
SIGMA=$(json_get "$OUT" cross_device_sigma_pct)
python3 -c "exit(0 if 0.05 <= $SIGMA <= 1.0 else 1)" || fail "sigma out of plausible range: $SIGMA"
echo "PASS T7"
