#!/usr/bin/env bash
# T6: GPU_RUN_DEVICE=<i> on stderr; CUDA_VISIBLE_DEVICES exported to the child;
# stdout passes through untouched.
. "$(dirname "$0")/lib.sh"

OUT="$TDIR/out.txt"; ERR="$TDIR/err.txt"
WARPSPEED_POOL_DEVICES=3 "$GPU_RUN" pool -- sh -c 'echo "DEV=$CUDA_VISIBLE_DEVICES ROLE=$WARPSPEED_GPU_ROLE"' >"$OUT" 2>"$ERR"
grep -q '^GPU_RUN_DEVICE=3$' "$ERR" || fail "GPU_RUN_DEVICE=3 not on stderr: $(cat "$ERR")"
grep -q '^DEV=3 ROLE=pool$' "$OUT" || fail "child env wrong: $(cat "$OUT")"

# bench role reports its fixed device
WARPSPEED_BENCH_DEVICE=6 "$GPU_RUN" bench -- true >"$OUT" 2>"$ERR"
grep -q '^GPU_RUN_DEVICE=6$' "$ERR" || fail "bench device not reported: $(cat "$ERR")"

# cal role exposes the full device list
"$GPU_RUN" cal -- sh -c 'echo "ALL=$CUDA_VISIBLE_DEVICES"' >"$OUT" 2>"$ERR"
grep -q '^ALL=0,1,2,3,4,5,6,7$' "$OUT" || fail "cal device list wrong: $(cat "$OUT")"

# cal role DEDUPES when bench/ncu overlap the pool (a duplicate device would
# deadlock the all-mode acquirer against itself)
WARPSPEED_BENCH_DEVICE=3 WARPSPEED_NCU_DEVICE=5 "$GPU_RUN" cal -- sh -c 'echo "ALL=$CUDA_VISIBLE_DEVICES"' >"$OUT" 2>"$ERR"
grep -q '^ALL=0,1,2,3,4,5$' "$OUT" || fail "cal dedupe wrong: $(cat "$OUT")"
echo "PASS T6"
