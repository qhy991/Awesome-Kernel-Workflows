#!/usr/bin/env bash
# WarpSpeed REAL-GPU acceptance suite. Runs ON the GPU node:
#   ssh H100-lsh 'warpspeed-accept/WarpSpeed/tests/node/run_node_tests.sh'
#
# Part 1 reuses the lock-semantics tests verbatim (T1/T3/T4/T5/T6 need no GPU
# tools at all - pure flock/timeout mechanics on real Linux).
# Part 2 runs T2 (serialization with mock clock logging) + a REAL clock-lock
# probe (skipped without permission).
# Part 3 builds the toy CUDA kernel and runs real calibrate / screen A-A /
# confirm / NCU-section acceptance against real devices.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export WS_ROOT=$(cd "$HERE/../.." && pwd)
# CUDA toolkit bins (nvcc, compute-sanitizer) are often installed but not on
# PATH for non-login shells - pick up the standard location.
[ -d /usr/local/cuda/bin ] && export PATH="$PATH:/usr/local/cuda/bin"
PASS=0; FAIL=0; SKIP=0; FAILED=""

say()  { printf '%s\n' "$*"; }
res() { # res <name> <rc> ; rc 0 pass, 77 skip, else fail
  case "$2" in
    0)  PASS=$((PASS+1)); say "PASS  $1" ;;
    77) SKIP=$((SKIP+1)); say "SKIP  $1" ;;
    *)  FAIL=$((FAIL+1)); FAILED="$FAILED $1"; say "FAIL  $1"; [ -f "/tmp/wsnode-$1.log" ] && tail -8 "/tmp/wsnode-$1.log" | sed 's/^/      /' ;;
  esac
}

say "== WarpSpeed node acceptance on $(hostname) =="
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0)
say "GPUs visible: $NGPU"
[ "$NGPU" -ge 2 ] || { say "FATAL: need >=2 GPUs"; exit 1; }
command -v ncu >/dev/null && NCU_OK=1 || { NCU_OK=0; say "WARN: ncu not on PATH - NCU tests will be skipped"; }
command -v compute-sanitizer >/dev/null || say "WARN: compute-sanitizer not on PATH"
command -v codex >/dev/null && say "codex: present" || say "WARN: codex CLI not found (required before real searches, not for these tests)"

# ---- Part 1: lock semantics (identical test files; no GPU tools involved) ----
for t in T1_pool_concurrency T3_ncu_serialize T4_timeout_kill_unlock T5_death_release T6_device_report; do
  "$WS_ROOT/tests/mock/test_$t.sh" > "/tmp/wsnode-$t.log" 2>&1; res "$t" $?
done

# ---- Part 2: bench serialization (mock clocks) + real clock-permission probe --
PATH_SAVE=$PATH
"$WS_ROOT/tests/mock/test_T2_bench_serialize.sh" > /tmp/wsnode-T2.log 2>&1; res "T2(serialize,mock-clocks)" $?
(
  set -e
  BD=$(( NGPU > 6 ? 6 : NGPU - 1 ))
  if nvidia-smi -i "$BD" -lgc 1500 >/dev/null 2>&1; then
    nvidia-smi -i "$BD" -rgc >/dev/null 2>&1
    exit 0
  fi
  exit 77
) ; res "T2r(real-clock-permission)" $?

# ---- Part 3: real-GPU measurement chain --------------------------------------
TOY="$HERE/toy"
( cd "$TOY" && make -s kernel_bench ) > /tmp/wsnode-build.log 2>&1 || { res "toy-build" 1; say "cannot continue real-GPU tests"; exit 1; }
res "toy-build" 0
BIN="$TOY/kernel_bench"

export WARPSPEED_LOCK_DIR=/tmp/warpspeed-nodetest/locks
export WARPSPEED_WARMUP_REPS=5
export WARPSPEED_LOCK_WAIT=120   # fail fast in tests instead of queueing an hour
rm -rf /tmp/warpspeed-nodetest
GPU_RUN="$WS_ROOT/infra/gpu_run"
LAST=$(( NGPU - 1 ))
POOL_ALL=$(seq 0 "$LAST" | tr '\n' ' ')

# T7r: real cross-device calibration (bench/ncu devices overlap the pool on
# purpose - the dedupe in gpu_run/wslock must handle it)
(
  set -e
  OUT=$(WARPSPEED_POOL_DEVICES="$POOL_ALL" WARPSPEED_BENCH_DEVICE=0 WARPSPEED_NCU_DEVICE="$LAST" \
        "$GPU_RUN" cal -- "$WS_ROOT/infra/calibrate.sh" "$BIN" --reps 20 2>/dev/null)
  echo "$OUT"
  python3 - "$OUT" <<'PY'
import json, sys
o = json.loads(sys.argv[1])
assert o["ok"], o
assert len(o["per_device_means_us"]) >= 2
assert 0 <= o["cross_device_sigma_pct"] <= 5.0, o["cross_device_sigma_pct"]
PY
) > /tmp/wsnode-T7r.log 2>&1; res "T7r(real-calibrate)" $?

# T9r: A/A screening on a real device -> |delta| within 3x noise
(
  set -e
  OUT=$("$GPU_RUN" pool -- "$WS_ROOT/infra/bench_screen.sh" "$BIN" "$BIN" --reps 20 --blocks 4 2>/dev/null)
  echo "$OUT"
  python3 - "$OUT" <<'PY'
import json, sys
o = json.loads(sys.argv[1])
assert o["ok"], o
assert abs(o["rel_speedup_pct"]) <= 3 * max(o["within_device_std_pct"], 0.2), o
PY
) > /tmp/wsnode-T9r.log 2>&1; res "T9r(real-screen-AA)" $?

# T10r: confirm on the bench device, clocks recorded
(
  set -e
  OUT=$(WARPSPEED_BENCH_DEVICE=$(( NGPU > 6 ? 6 : 0 )) "$GPU_RUN" bench -- "$WS_ROOT/infra/bench_confirm.sh" "$BIN" --reps 40 2>/dev/null)
  echo "$OUT"
  python3 - "$OUT" <<'PY'
import json, sys
o = json.loads(sys.argv[1])
assert o["ok"] and o["reps"] == 40 and o["latency_us_mean"] > 0, o
assert o["clocks"]["graphics_mhz"] is not None, "clock readback failed"
PY
) > /tmp/wsnode-T10r.log 2>&1; res "T10r(real-confirm)" $?

# T11r: real NCU curated sections -> parse -> stable fingerprint
if [ "$NCU_OK" = "1" ]; then
(
  set -e
  ND=$(( NGPU > 7 ? 7 : 0 ))
  A=$(WARPSPEED_NCU_DEVICE=$ND "$GPU_RUN" ncu --timeout 600 -- "$WS_ROOT/infra/ncu_profile.sh" "$BIN" /tmp/wsnode-prof-a.json --reps 1 2>/dev/null)
  B=$(WARPSPEED_NCU_DEVICE=$ND "$GPU_RUN" ncu --timeout 600 -- "$WS_ROOT/infra/ncu_profile.sh" "$BIN" /tmp/wsnode-prof-b.json --reps 1 2>/dev/null)
  echo "$A"
  python3 - "$A" "$B" <<'PY'
import json, sys
a, b = json.loads(sys.argv[1]), json.loads(sys.argv[2])
km = a["key_metrics"]
assert a["kernel_count"] >= 1, a
for key in ("sm_pct", "mem_pct", "occupancy_pct", "duration_us"):
    assert isinstance(km.get(key), (int, float)), "missing metric %s" % key
assert a["fingerprint"] == b["fingerprint"], "fingerprint unstable: %s vs %s" % (a["fingerprint"], b["fingerprint"])
PY
) > /tmp/wsnode-T11r.log 2>&1; res "T11r(real-ncu)" $?
else
  res "T11r(real-ncu)" 77
fi

# T8: harness contract (python mockproj harness; works anywhere)
"$WS_ROOT/tests/mock/test_T8_harness.sh" > /tmp/wsnode-T8.log 2>&1; res "T8(harness-contract)" $?

# T12/T13: db + integration loop (mock tools fine on the node too)
"$WS_ROOT/tests/mock/test_T12_wsdb.sh" > /tmp/wsnode-T12.log 2>&1; res "T12(wsdb)" $?
"$WS_ROOT/tests/mock/test_T13_mock_loop.sh" > /tmp/wsnode-T13.log 2>&1; res "T13(mock-loop)" $?

say "----------------------------------------"
say "node acceptance: $PASS passed, $SKIP skipped, $FAIL failed"
[ "$FAIL" = "0" ] || { say "failed:$FAILED"; exit 1; }
