#!/usr/bin/env bash
# T12: wsdb.py - concurrent writers without lock failures, orphan recovery,
# rewind/retire semantics, hard dedup, blocked counting, budget ingest.
. "$(dirname "$0")/lib.sh"

DB="$TDIR/state/search.sqlite"
WSDB="$WS_ROOT/tools/wsdb.py"
W() { python3 "$WSDB" --db "$DB" "$@"; }

W init >/dev/null || fail "init failed"
W config-set --json '{"K_BLOCKED": 2, "BUDGET_GPU_MINUTES": 100}' >/dev/null

# --- 1. contention: 8 parallel writers x 50 ops, zero failures ---------------
python3 - "$WSDB" "$DB" <<'PY' || fail "contention test failed"
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

wsdb, db = sys.argv[1], sys.argv[2]

def worker(w):
    for i in range(50):
        eid = "rc%de%d" % (w, i)
        for cmd, payload in [
            ("exp-start", {"exp_id": eid, "round": 0, "type": "exploit",
                           "parent_commit": "seed", "hypothesis": "h",
                           "direction_tags": ["t"], "predicted_gain_pct": 1}),
            ("exp-finish", {"exp_id": eid, "status": "correct_slower",
                            "speedup_vs_parent": 0.1}),
        ]:
            r = subprocess.run(["python3", wsdb, "--db", db, cmd, "--json", json.dumps(payload)],
                               capture_output=True, text=True)
            out = json.loads(r.stdout)
            assert r.returncode == 0 and out.get("ok"), (cmd, eid, r.stdout, r.stderr)
    return True

with ThreadPoolExecutor(max_workers=8) as ex:
    assert all(ex.map(worker, range(8)))
print("contention: 800 writes ok")
PY

N=$(python3 -c "
import sqlite3; con = sqlite3.connect('$DB')
print(con.execute(\"SELECT COUNT(*) FROM experiments WHERE status='correct_slower'\").fetchone()[0])")
[ "$N" = "400" ] || fail "expected 400 finished experiments, got $N"

# --- 2. orphan recovery -------------------------------------------------------
for i in 1 2 3; do
  W exp-start --json "{\"exp_id\":\"orph$i\",\"round\":1,\"type\":\"explore\",\"worktree_path\":\"/tmp/wt$i\"}" >/dev/null
done
W exp-finish --json '{"exp_id":"orph1","status":"incorrect"}' >/dev/null
OUT=$(W recover-orphans)
N=$(python3 -c "import json; print(len(json.loads('''$OUT''')['orphaned']))")
[ "$N" = "2" ] || fail "expected 2 orphans, got $N ($OUT)"

# --- 3. rewind / retire -------------------------------------------------------
W seed-baseline --json '{"commit":"base","latency_us":1000,"assumptions":[]}' >/dev/null
W checkpoint-upsert --json '{"commit":"ck1","parent_commit":"base","round":1,"latency_us":950,"assumptions":["persistent_kernel"]}' >/dev/null
W checkpoint-upsert --json '{"commit":"ck2","parent_commit":"ck1","round":2,"latency_us":920,"assumptions":["persistent_kernel","tma"]}' >/dev/null
W checkpoint-upsert --json '{"commit":"ck3","parent_commit":"ck2","round":3,"latency_us":910,"assumptions":["persistent_kernel","tma","pingpong"]}' >/dev/null
OUT=$(W rewind --json '{"suspect_checkpoint":"ck1","suspect_tag":"persistent_kernel"}')
[ "$(json_get "$OUT" rewound_to)" = "base" ] || fail "rewind target wrong: $OUT"
RETIRED=$(python3 -c "import json; print(sorted(json.loads('''$OUT''')['retired']))")
[ "$RETIRED" = "['ck1', 'ck2', 'ck3']" ] || fail "retired subtree wrong: $RETIRED"
FRONT=$(W frontier)
[ "$(json_get "$FRONT" frontier_commit)" = "base" ] || fail "frontier pointer not rewound"
N=$(python3 -c "import json; print(len(json.loads('''$FRONT''')['frontier']))")
[ "$N" = "1" ] || fail "retired checkpoints still live: $FRONT"

# --- 4. hard dedup ------------------------------------------------------------
W checkpoint-upsert --json '{"commit":"ck4","parent_commit":"base","round":4,"latency_us":900,"assumptions":["tma"],"ncu_fingerprint":"fpX"}' >/dev/null
W exp-start --json '{"exp_id":"dup1","round":5,"type":"explore","parent_commit":"base"}' >/dev/null
OUT=$(W dedup-check --json '{"noise_pct":1.0,"candidates":[{"exp_id":"dup1","commit":"ck5","latency_us":905,"fingerprint":"fpX","strategy_set_hash":"hZ"}]}')
M=$(python3 -c "import json; m=json.loads('''$OUT''')['merge']; print(m[0]['into_commit'] if m else 'none')")
[ "$M" = "ck4" ] || fail "hard dedup did not merge into ck4: $OUT"
OUT=$(W dedup-check --json '{"noise_pct":1.0,"candidates":[{"exp_id":"dup1","commit":"ck6","latency_us":905,"fingerprint":"fpY"}]}')
M=$(python3 -c "import json; print(len(json.loads('''$OUT''')['merge']))")
[ "$M" = "0" ] || fail "dedup merged despite different fingerprint"

# --- 5. blocked counting + postmortem trigger ----------------------------------
for r in 6 7; do
  W exp-start  --json "{\"exp_id\":\"blk$r\",\"round\":$r,\"type\":\"exploit\",\"parent_commit\":\"ck4\"}" >/dev/null
  W exp-finish --json "{\"exp_id\":\"blk$r\",\"status\":\"correct_slower\"}" >/dev/null
  OUT=$(W blocked-mark --json "{\"round\":$r}")
done
echo "$OUT" | grep -q '"postmortem_due": \["ck4"\]' || fail "ck4 should be postmortem_due after 2 blocked rounds (K=2): $OUT"
# a good child resets the counter
W exp-start  --json '{"exp_id":"good1","round":8,"type":"exploit","parent_commit":"ck4"}' >/dev/null
W exp-finish --json '{"exp_id":"good1","status":"correct_faster","speedup_vs_parent":3.0}' >/dev/null
OUT=$(W blocked-mark --json '{"round":8}')
BC=$(python3 -c "
import sqlite3; con = sqlite3.connect('$DB')
print(con.execute(\"SELECT blocked_count FROM checkpoints WHERE commit_hash='ck4'\").fetchone()[0])")
[ "$BC" = "0" ] || fail "good child should reset blocked_count, got $BC"

# --- 6. budget ingest (incremental by byte offset) ------------------------------
mkdir -p "$TDIR/state/logs"
printf '%s\n' '{"wall_s": 60, "role": "pool"}' '{"wall_s": 120, "role": "bench"}' >> "$TDIR/state/logs/gpu_minutes.jsonl"
OUT=$(W budget-ingest)
[ "$(json_get "$OUT" added_gpu_minutes)" = "3.0" ] || fail "budget ingest wrong: $OUT"
printf '%s\n' '{"wall_s": 30}' >> "$TDIR/state/logs/gpu_minutes.jsonl"
OUT=$(W budget-ingest)
[ "$(json_get "$OUT" added_gpu_minutes)" = "0.5" ] || fail "incremental ingest re-read old lines: $OUT"
[ "$(json_get "$OUT" gpu_minutes_used)" = "3.5" ] || fail "budget total wrong: $OUT"

# --- 7. lessons append/query/supersede -----------------------------------------
W lessons-append --json '{"type":"technique_works","claim":"c1","mechanism":"m","scope":{"arch":"sm90"},"tags":["tma"],"confidence":0.5}' >/dev/null
OUT=$(W lessons-append --json '{"type":"technique_fails","claim":"c1 refined","mechanism":"m2","scope":{"arch":"sm90"},"tags":["tma"],"confidence":0.8,"supersedes":"L0001"}')
Q=$(W lessons-query --tags tma --arch sm90 --max 10)
python3 - <<PY || fail "superseded lesson still returned: $Q"
import json
ls = json.loads('''$Q''')["lessons"]
assert all(l["id"] != "L0001" for l in ls), ls
assert any(l["claim"] == "c1 refined" for l in ls), ls
PY
[ -f "$TDIR/state/bitlessons.jsonl" ] || fail "bitlessons.jsonl missing"
[ -f "$TDIR/state/bitlessons.md" ] || fail "bitlessons.md mirror missing"

echo "PASS T12"
