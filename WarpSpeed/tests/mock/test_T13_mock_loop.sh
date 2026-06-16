#!/usr/bin/env bash
# T13: live mock search loop. Mechanically replays the EXACT command sequences
# the orchestrator's prompts instruct agents to run - rendered config commands,
# git worktree per experiment, parent-build cache, screen/confirm/profile via
# gpu_run, wsdb records, blocked counting, and a forced rewind - against a real
# mock project. This is the integration gate for prompt<->infra<->db fit.
. "$(dirname "$0")/lib.sh"

PROJ="$TDIR/proj"
"$WS_ROOT/tests/mock/mkmockproj.sh" "$PROJ" 850.0 balanced >/dev/null
(cd "$PROJ" && ./build.sh >/dev/null)

# --- Init (as the init agent does): render config with fast-test overrides ---
python3 "$WS_ROOT/tools/render_config.py" \
  --warpspeed-dir "$WS_ROOT" --project-dir "$PROJ" \
  --set target_gpu='"H100"' --set BUDGET_GPU_MINUTES=60 --set K_BLOCKED=1 \
  --set SCREEN_REPS=16 --set CONFIRM_REPS=24 --set WARMUP_REPS=2 --set SCREEN_BLOCKS=4 \
  --set LOCK_DIR="\"$TDIR/locks\"" > "$TDIR/render.json" || fail "render_config failed"

STATE="$PROJ/.warpspeed"
[ -f "$STATE/config.json" ] || fail "materialized config missing"
mkdir -p "$STATE/worktrees" "$STATE/review" "$STATE/builds" "$STATE/ncu_cache" "$STATE/logs" "$STATE/results"

getcmd() { python3 -c "import json; print(json.load(open('$STATE/config.json'))['commands']['$1'])"; }
getpath() { python3 -c "import json; print(json.load(open('$STATE/config.json'))['paths']['$1'])"; }
WSDB=$(getcmd wsdb)
SCREEN_CMD=$(getcmd screen)
CONFIRM_CMD=$(getcmd confirm)
NCU_CMD=$(getcmd ncu_profile)
CORR_CMD=$(getcmd correctness)
BIN=bin/kernel_bench

$WSDB init >/dev/null || fail "wsdb init via rendered command failed"
$WSDB config-set --json @"$STATE/config.json" >/dev/null || fail "config-set failed"

jget() { python3 -c "
import json, sys
o = json.loads(sys.argv[1])
for p in sys.argv[2].split('.'):
    o = o[p] if not isinstance(o, list) else o[int(p)]
print(o)" "$1" "$2"; }

# --- Seed (as the seed agent does; eval = the shell parse agents get) ---------
BASE=$(git -C "$PROJ" rev-parse HEAD)
git -C "$PROJ" tag -f ckpt/baseline "$BASE" >/dev/null
CONF=$(eval "WARPSPEED_EXP_ID=seed $CONFIRM_CMD '$PROJ/$BIN' --shape default" 2>/dev/null)
BASE_LAT=$(jget "$CONF" latency_us_mean)
PROF=$(eval "WARPSPEED_EXP_ID=seed $NCU_CMD '$PROJ/$BIN' '$STATE/ncu_cache/$BASE.json' --shape default" 2>/dev/null)
BASE_FP=$(jget "$PROF" fingerprint)
python3 - "$BASE" "$BASE_LAT" "$BASE_FP" > "$TDIR/seed.json" <<'PY'
import json, sys
print(json.dumps({"commit": sys.argv[1], "latency_us": float(sys.argv[2]),
                  "assumptions": [], "ncu_fingerprint": sys.argv[3]}))
PY
$WSDB seed-baseline --json @"$TDIR/seed.json" >/dev/null || fail "seed-baseline failed"
note "baseline $BASE_LAT us fp=$BASE_FP"

# run_exp <exp_id> <round> <parent_commit> <new_lat> <profile> <tags_json>
# Replays the implementor+screen+confirm+profile+record path; echoes "status candLat commit fp".
run_exp() {
  EXP=$1 ROUND=$2 PARENT=$3 NEWLAT=$4 PROFILE=$5 TAGS=$6
  WT="$STATE/worktrees/$EXP"
  git -C "$PROJ" worktree add -q -b "exp/r$ROUND/$EXP-mock" "$WT" "$PARENT" || fail "worktree add failed for $EXP"
  python3 - "$EXP" "$ROUND" "$PARENT" "$TAGS" > "$TDIR/$EXP.start.json" <<'PY'
import json, sys
print(json.dumps({"exp_id": sys.argv[1], "round": int(sys.argv[2]), "type": "exploit",
                  "parent_commit": sys.argv[3], "hypothesis": "mock hypothesis",
                  "direction_tags": json.loads(sys.argv[4]), "predicted_gain_pct": 5.0,
                  "predicted_mechanism": "mock mechanism",
                  "worktree_path": "", "branch": ""}))
PY
  $WSDB exp-start --json @"$TDIR/$EXP.start.json" >/dev/null || fail "exp-start failed"

  # implementor: edit MOCK_LAT_US (+ profile), rebuild, prove correctness, commit
  python3 - "$WT/src/kernel.cu" "$NEWLAT" "$PROFILE" <<'PY'
import re, sys
p, lat, prof = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(p).read()
t = re.sub(r"// MOCK_LAT_US: \S+", "// MOCK_LAT_US: %s" % lat, t)
t = re.sub(r"// MOCK_PROFILE: \S+", "// MOCK_PROFILE: %s" % prof, t)
open(p, "w").write(t)
PY
  (cd "$WT" && ./build.sh >/dev/null)
  eval "WARPSPEED_EXP_ID=$EXP $CORR_CMD '$WT/$BIN'" >/dev/null 2>&1 || fail "correctness failed in $EXP"
  git -C "$WT" -c user.name=mock -c user.email=m@x commit -qam "[$EXP] mock hypothesis"
  COMMIT=$(git -C "$WT" rev-parse HEAD)

  # screen: parent-build cache exactly as the screen prompt specifies
  PBIN="$STATE/builds/$PARENT/$BIN"
  if [ ! -x "$PBIN" ]; then
    git -C "$PROJ" worktree add -q --detach "$STATE/builds/wt-$EXP" "$PARENT"
    (cd "$STATE/builds/wt-$EXP" && ./build.sh >/dev/null)
    mkdir -p "$(dirname "$PBIN")" && cp "$STATE/builds/wt-$EXP/$BIN" "$PBIN"
    git -C "$PROJ" worktree remove --force "$STATE/builds/wt-$EXP"; git -C "$PROJ" worktree prune
  fi
  SCR=$(eval "WARPSPEED_EXP_ID=$EXP $SCREEN_CMD '$PBIN' '$WT/$BIN' --shape default" 2>/dev/null) || fail "screen failed in $EXP"
  REL=$(jget "$SCR" rel_speedup_pct)
  CONFJ=$(eval "WARPSPEED_EXP_ID=$EXP $CONFIRM_CMD '$WT/$BIN' --shape default" 2>/dev/null) || fail "confirm failed in $EXP"
  CLAT=$(jget "$CONFJ" latency_us_mean)
  PROFJ=$(eval "WARPSPEED_EXP_ID=$EXP $NCU_CMD '$WT/$BIN' '$STATE/ncu_cache/exp-$EXP.json' --shape default" 2>/dev/null) || fail "profile failed in $EXP"
  FP=$(jget "$PROFJ" fingerprint)

  SIG=$(python3 -c "print(1 if float('$REL') > max(1.0, 2*0.35) else 0)")
  STATUS=$([ "$SIG" = "1" ] && echo correct_faster || echo correct_slower)
  python3 - "$EXP" "$STATUS" "$COMMIT" "$REL" "$CLAT" > "$TDIR/$EXP.finish.json" <<'PY'
import json, sys
print(json.dumps({"exp_id": sys.argv[1], "status": sys.argv[2], "commit_hash": sys.argv[3],
                  "speedup_vs_parent": float(sys.argv[4]), "achieved_gain_pct": float(sys.argv[4]),
                  "prediction_gap_pct": 5.0 - float(sys.argv[4]),
                  "latency_us_mean": float(sys.argv[5]), "bench_tier": "confirm",
                  "review_iterations": 1, "sanitizer_clean": 1}))
PY
  $WSDB exp-finish --json @"$TDIR/$EXP.finish.json" >/dev/null || fail "exp-finish failed"
  echo "$STATUS $CLAT $COMMIT $FP"
}

# --- Round 1: real improvement 850 -> 800 ------------------------------------
$WSDB next-round >/dev/null
read -r ST1 LAT1 C1 FP1 <<EOF
$(run_exp r1e1 1 "$BASE" 800.0 membound '["tma"]')
EOF
note "r1e1: $ST1 $LAT1 us"
[ "$ST1" = "correct_faster" ] || fail "expected correct_faster in round 1, got $ST1"
python3 - "$C1" "$BASE" "$LAT1" "$FP1" > "$TDIR/ck1.json" <<'PY'
import json, sys
print(json.dumps({"commit": sys.argv[1], "parent_commit": sys.argv[2], "round": 1,
                  "latency_us": float(sys.argv[3]), "assumptions": ["tma"],
                  "strategy_set_hash": "th1", "ncu_fingerprint": sys.argv[4]}))
PY
OUT=$($WSDB checkpoint-upsert --json @"$TDIR/ck1.json")
[ "$(jget "$OUT" new_best)" = "True" ] || fail "round-1 winner should be new best: $OUT"
$WSDB blocked-mark --json '{"round": 1}' >/dev/null

# --- Round 2: insignificant change -> blocked -> postmortem due (K=1) --------
$WSDB next-round >/dev/null
read -r ST2 LAT2 C2 FP2 <<EOF
$(run_exp r2e1 2 "$C1" 799.5 membound '["swizzle"]')
EOF
note "r2e1: $ST2 $LAT2 us"
[ "$ST2" = "correct_slower" ] || fail "expected correct_slower in round 2, got $ST2"
OUT=$($WSDB blocked-mark --json '{"round": 2}')
echo "$OUT" | grep -q "\"postmortem_due\": \[\"$C1\"\]" || fail "K_BLOCKED=1 should flag $C1: $OUT"

# --- Dedup sanity: same fingerprint + latency within noise merges -------------
OUT=$($WSDB dedup-check --json "{\"noise_pct\": 1.0, \"candidates\": [{\"exp_id\": \"r2e1\", \"commit\": \"$C2\", \"latency_us\": $LAT2, \"fingerprint\": \"$FP2\", \"strategy_set_hash\": \"zz\"}]}")
echo "$OUT" | grep -q "\"into_commit\": \"$C1\"" || fail "r2e1 should hard-dedup into $C1 (same profile, same latency): $OUT"

# --- Rewind (as the postmortem rewind agent does) ------------------------------
LSN=$($WSDB lessons-append --json '{"type":"assumption_invalidated","claim":"tma at c1 poisons descendants (mock)","mechanism":"m","scope":{"arch":"sm90"},"tags":["tma"],"confidence":0.7}')
LID=$(python3 -c "import json; print(json.loads('''$LSN''')['ids'][0])")
OUT=$($WSDB rewind --json "{\"suspect_checkpoint\": \"$C1\", \"suspect_tag\": \"tma\", \"lesson_id\": \"$LID\"}")
[ "$(jget "$OUT" rewound_to)" = "$BASE" ] || fail "rewind should land on baseline: $OUT"
$WSDB queue-spec --json "{\"spec\": {\"type\": \"replay\", \"parent_commit\": \"$BASE\", \"hypothesis\": \"replay\", \"direction_tags\": [\"swizzle\"], \"predicted_gain_pct\": 4, \"predicted_mechanism\": \"m\"}}" >/dev/null

# --- Post-rewind snapshot: frontier=baseline, lessons survive, replay queued ---
SNAP=$($WSDB round-snapshot)
[ "$(jget "$SNAP" frontier_commit)" = "$BASE" ] || fail "frontier not rewound"
N=$(python3 -c "import json; s=json.loads('''$SNAP'''); print(len(s['frontier']))")
[ "$N" = "1" ] || fail "retired subtree still live in frontier"
NL=$(python3 -c "import json; s=json.loads('''$SNAP'''); print(len(s['recent_lessons']))")
[ "$NL" -ge 1 ] || fail "lessons must survive the rewind"
NQ=$(python3 -c "import json; s=json.loads('''$SNAP'''); print(len(s['queued_specs']))")
[ "$NQ" = "1" ] || fail "replay spec must be queued"

# --- Budget: gpu_run accounting flowed into the ledger -------------------------
OUT=$($WSDB budget-ingest)
USED=$(jget "$OUT" gpu_minutes_used)
python3 -c "exit(0 if float('$USED') > 0 else 1)" || fail "gpu-minutes ledger empty despite gpu_run usage: $OUT"

# --- Worktree cleanup as the maintain agent does -------------------------------
git -C "$PROJ" worktree remove --force "$STATE/worktrees/r1e1" 2>/dev/null
git -C "$PROJ" worktree remove --force "$STATE/worktrees/r2e1" 2>/dev/null
git -C "$PROJ" worktree prune
git -C "$PROJ" rev-parse --verify "exp/r1/r1e1-mock" >/dev/null || fail "experiment branch must survive cleanup (evidence)"

$WSDB status >/dev/null 2>&1 || fail "status command failed"
echo "PASS T13"
