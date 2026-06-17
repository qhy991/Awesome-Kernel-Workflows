#!/usr/bin/env bash
# T11: NCU profiling uses ONLY the curated sections by default, escalates to
# --set full only with --full, parses to canonical metrics, and the
# fingerprint is stable across re-profiles.
. "$(dirname "$0")/lib.sh"
ensure_mockproj P800 800.0 membound

PROF="$WS_ROOT/infra/ncu_profile.sh"

# Role guard: must refuse under a non-ncu role
if "$GPU_RUN" pool -- "$PROF" "$P800_BIN" "$TDIR/x.json" >/dev/null 2>&1; then
  fail "ncu_profile ran under pool role (role guard missing)"
fi

A=$("$GPU_RUN" ncu -- "$PROF" "$P800_BIN" "$TDIR/a.json" 2>/dev/null)
B=$("$GPU_RUN" ncu -- "$PROF" "$P800_BIN" "$TDIR/b.json" 2>/dev/null)
FP_A=$(json_get "$A" fingerprint)
FP_B=$(json_get "$B" fingerprint)
note "fingerprint=$FP_A"
[ -n "$FP_A" ] && [ "$FP_A" = "$FP_B" ] || fail "fingerprint not stable across re-profiles ($FP_A vs $FP_B)"
[ "$(json_get "$A" key_metrics.dram_pct)" = "84" ] || fail "parsed metrics wrong"
[ "$(json_get "$A" kernel_count)" = "1" ] || fail "kernel_count wrong"

# Default invocations: curated sections only, never --set full
NCU_LOG="$WARPSPEED_MOCK_DIR/ncu.log"
python3 - "$NCU_LOG" <<'PY' || fail "section usage violated curation"
import json, sys
curated = {"SpeedOfLight", "MemoryWorkloadAnalysis", "SchedulerStats",
           "WarpStateStats", "Occupancy", "LaunchStats"}
for line in open(sys.argv[1]):
    argv = json.loads(line)["argv"]
    if "--import" in argv or "--version" in argv:
        continue
    if "--set" in argv:
        raise SystemExit(1)  # default run must not use --set
    secs = {argv[i + 1] for i, a in enumerate(argv) if a == "--section"}
    assert secs and secs.issubset(curated), secs
PY

# --full escalation reaches the tool
"$GPU_RUN" ncu -- "$PROF" "$P800_BIN" "$TDIR/full.json" --full >/dev/null 2>&1
grep -q '"--set", "full"' "$NCU_LOG" || fail "--full did not escalate to --set full"

# A different mock profile must produce a different fingerprint
ensure_mockproj PCB 800.0 computebound
C=$("$GPU_RUN" ncu -- "$PROF" "$PCB_BIN" "$TDIR/c.json" 2>/dev/null)
FP_C=$(json_get "$C" fingerprint)
[ "$FP_C" != "$FP_A" ] || fail "distinct profiles produced identical fingerprints"
echo "PASS T11"
