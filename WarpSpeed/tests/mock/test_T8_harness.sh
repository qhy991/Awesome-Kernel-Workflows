#!/usr/bin/env bash
# T8: correctness harness contract - exit 0/1 + JSON report over all shapes,
# deterministic across runs, and enforceable as read-only.
. "$(dirname "$0")/lib.sh"
ensure_mockproj P850 850.0 balanced

H="$P850_DIR/harness/correctness.py"
A=$(python3 "$H" --impl "$P850_BIN"); RC_A=$?
B=$(python3 "$H" --impl "$P850_BIN"); RC_B=$?
[ "$RC_A" = "0" ] && [ "$RC_B" = "0" ] || fail "harness exit nonzero on correct impl"
[ "$A" = "$B" ] || fail "harness output not deterministic"
echo "$A" | python3 -c 'import json,sys; r=json.load(sys.stdin); assert r["ok"] and len(r["shapes"])==2' || fail "bad report: $A"

# Broken kernel -> exit 1, ok=false
BROKEN="$TDIR/proj_broken"
"$WS_ROOT/tests/mock/mkmockproj.sh" "$BROKEN" 850.0 balanced >/dev/null
echo "// MOCK_BROKEN" >> "$BROKEN/src/kernel.cu"
(cd "$BROKEN" && ./build.sh >/dev/null)
OUT=$(python3 "$BROKEN/harness/correctness.py" --impl "$BROKEN/bin/kernel_bench"); RC=$?
[ "$RC" = "1" ] || fail "broken impl should exit 1, got $RC"
echo "$OUT" | grep -q '"ok": false' || fail "broken impl report should be ok=false: $OUT"

# Single-shape mode
python3 "$H" --impl "$P850_BIN" --shape small >/dev/null || fail "--shape mode failed"

# Read-only enforcement (chmod a-w, as WarpSpeed init applies)
chmod -R a-w "$P850_DIR/harness"
if echo x >> "$H" 2>/dev/null; then
  chmod -R u+w "$P850_DIR/harness"
  fail "harness writable after chmod a-w"
fi
chmod -R u+w "$P850_DIR/harness"
echo "PASS T8"
