#!/usr/bin/env bash
# WarpSpeed mock acceptance suite: runs every tests/mock/test_T*.sh with the
# mock GPU tools on PATH. Needs no GPUs; runs on macOS and Linux.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export WS_ROOT=$(cd "$HERE/../.." && pwd)

PASS=0; FAIL=0; FAILED=""
for t in "$HERE"/test_*.sh; do
  name=$(basename "$t")
  if "$t" > "/tmp/ws-$name.out" 2>&1; then
    PASS=$((PASS + 1)); echo "PASS  $name"
  else
    FAIL=$((FAIL + 1)); FAILED="$FAILED $name"; echo "FAIL  $name"
    sed 's/^/      /' "/tmp/ws-$name.out" | tail -15
  fi
done
echo "----------------------------------------"
echo "mock acceptance: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ] || { echo "failed:$FAILED"; exit 1; }
