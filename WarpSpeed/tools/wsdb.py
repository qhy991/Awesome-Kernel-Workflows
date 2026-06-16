#!/usr/bin/env python3
"""wsdb.py - WarpSpeed search-database CLI (python3 stdlib only).

Every WarpSpeed agent reads/writes search state exclusively through this tool:

    python3 wsdb.py --db <state>/search.sqlite <subcommand> [--json '<payload>'] [flags]

One JSON object on stdout per invocation; exit 0 on success, 1 on failure
(with {"ok": false, "error": ...}). Safe under concurrent writers: WAL,
busy_timeout=30s, BEGIN IMMEDIATE for every write transaction.

BitLessons live next to the DB as append-only bitlessons.jsonl (+ a
bitlessons.md human mirror); lessons_index in SQLite is the queryable mirror.
"""
import argparse
import json
import os
import sqlite3
import sys

SCHEMA_VERSION = 1
FINAL_STATUSES = ("gen_failed", "compile_error", "incorrect", "review_rejected",
                  "correct_slower", "correct_faster", "new_best")
GOOD_STATUSES = ("correct_faster", "new_best")


def connect(db_path):
    con = sqlite3.connect(db_path, timeout=35, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def state_dir(args):
    return os.path.dirname(os.path.abspath(args.db))


def meta_get(con, key, default=None):
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def meta_set(con, key, value):
    con.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def meta_get_json(con, key, default):
    raw = meta_get(con, key)
    return json.loads(raw) if raw else default


def jload(args):
    raw = args.json
    if raw is None:
        return {}
    if raw == "-":
        raw = sys.stdin.read()
    elif raw.startswith("@"):
        with open(raw[1:]) as f:
            raw = f.read()
    return json.loads(raw)


def write_tx(con):
    con.execute("BEGIN IMMEDIATE")


def cfg(con):
    return meta_get_json(con, "config", {})


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_init(con, args):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "schema.sql")) as f:
        schema = f.read()
    con.executescript(schema)  # manages its own transaction; idempotent
    write_tx(con)
    created = meta_get(con, "schema_version") is None
    meta_set(con, "schema_version", str(SCHEMA_VERSION))
    if meta_get(con, "current_round") is None:
        meta_set(con, "current_round", "0")
    con.execute("COMMIT")
    return {"ok": True, "created": created, "schema_version": SCHEMA_VERSION}


def cmd_config_set(con, args):
    payload = jload(args)
    write_tx(con)
    meta_set(con, "config", json.dumps(payload))
    con.execute("COMMIT")
    return {"ok": True}


def cmd_config_get(con, args):
    return {"ok": True, "config": cfg(con)}


def cmd_calibration_set(con, args):
    payload = jload(args)
    write_tx(con)
    meta_set(con, "calibration", json.dumps(payload))
    con.execute("COMMIT")
    return {"ok": True}


def cmd_calibration_get(con, args):
    cal = meta_get_json(con, "calibration", None)
    return {"ok": True, "calibration": cal, "fresh": cal is not None}


def cmd_seed_baseline(con, args):
    p = jload(args)
    write_tx(con)
    existing = con.execute("SELECT commit_hash FROM checkpoints WHERE parent_commit IS NULL").fetchone()
    if existing:
        con.execute("COMMIT")
        return {"ok": True, "already": True, "commit": existing["commit_hash"]}
    con.execute(
        "INSERT INTO checkpoints(commit_hash,parent_commit,round_created,latency_us,"
        "assumptions_json,strategy_set_hash,ncu_fingerprint,key_metrics_json) "
        "VALUES(?,NULL,0,?,?,?,?,?)",
        (p["commit"], p.get("latency_us"),
         json.dumps(sorted(p.get("assumptions", []))),
         p.get("strategy_set_hash"), p.get("ncu_fingerprint"),
         json.dumps(p.get("key_metrics")) if p.get("key_metrics") else None))
    meta_set(con, "frontier_commit", p["commit"])
    meta_set(con, "baseline_commit", p["commit"])
    con.execute("COMMIT")
    return {"ok": True, "already": False, "commit": p["commit"]}


def cmd_next_round(con, args):
    write_tx(con)
    r = int(meta_get(con, "current_round", "0")) + 1
    meta_set(con, "current_round", str(r))
    con.execute("INSERT OR IGNORE INTO budget_ledger(round) VALUES(?)", (r,))
    con.execute("COMMIT")
    return {"ok": True, "round": r}


def _headroom(km):
    if not km:
        return None
    vals = [km.get(k) for k in ("sm_pct", "mem_pct", "dram_pct")]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(100 - max(vals), 1) if vals else None


def _frontier_rows(con):
    rows = con.execute(
        "SELECT * FROM checkpoints WHERE retired=0 ORDER BY latency_us ASC NULLS LAST").fetchall()
    frontier_commit = meta_get(con, "frontier_commit")
    out = []
    for r in rows:
        km = json.loads(r["key_metrics_json"]) if r["key_metrics_json"] else None
        out.append({
            "commit": r["commit_hash"], "parent": r["parent_commit"],
            "round_created": r["round_created"], "latency_us": r["latency_us"],
            "assumptions": json.loads(r["assumptions_json"]),
            "strategy_set_hash": r["strategy_set_hash"],
            "ncu_fingerprint": r["ncu_fingerprint"],
            "blocked_count": r["blocked_count"],
            "headroom_pct": _headroom((km or {}).get("key_metrics", km) if km else None),
            "is_frontier": r["commit_hash"] == frontier_commit,
        })
    return out


def _budget_status(con):
    c = cfg(con)
    row = con.execute("SELECT COALESCE(SUM(gpu_minutes_used),0) AS m, "
                      "COALESCE(SUM(tokens_used),0) AS t FROM budget_ledger").fetchone()
    total = c.get("BUDGET_GPU_MINUTES")
    used = round(row["m"], 2)
    remaining = round(total - used, 2) if total else None
    quartile = min(3, int(4 * used / total)) if total else 0
    return {"gpu_minutes_used": used, "gpu_minutes_total": total,
            "gpu_minutes_remaining": remaining, "tokens_used": row["t"],
            "quartile": quartile}


def _prediction_calibration(con):
    rows = con.execute(
        "SELECT direction_tags, prediction_gap_pct FROM experiments "
        "WHERE prediction_gap_pct IS NOT NULL").fetchall()
    agg = {}
    for r in rows:
        try:
            tags = json.loads(r["direction_tags"] or "[]")
        except ValueError:
            tags = []
        tag = tags[0] if tags else "(untagged)"
        a = agg.setdefault(tag, [0.0, 0])
        a[0] += r["prediction_gap_pct"]
        a[1] += 1
    return [{"tag": t, "mean_gap_pct": round(s / n, 2), "n": n}
            for t, (s, n) in sorted(agg.items(), key=lambda kv: -kv[1][1])]


def cmd_round_snapshot(con, args):
    c = cfg(con)
    k_blocked = int(c.get("K_BLOCKED", 4))
    frontier = _frontier_rows(con)
    leaderboard = [
        {"commit": r["commit"], "latency_us": r["latency_us"],
         "assumptions": r["assumptions"], "round_created": r["round_created"]}
        for r in sorted([f for f in frontier if f["latency_us"] is not None],
                        key=lambda f: f["latency_us"])[:5]]
    lessons = con.execute(
        "SELECT id,type,tags,claim,confidence FROM lessons_index "
        "WHERE superseded_by IS NULL ORDER BY created_at DESC, id DESC LIMIT 20").fetchall()
    queued = con.execute(
        "SELECT id, spec_json FROM queued_specs WHERE consumed=0 ORDER BY id").fetchall()
    running = con.execute("SELECT COUNT(*) AS n FROM experiments WHERE status='running'").fetchone()["n"]
    return {
        "ok": True,
        "round": int(meta_get(con, "current_round", "0")),
        "baseline_commit": meta_get(con, "baseline_commit"),
        "frontier_commit": meta_get(con, "frontier_commit"),
        "frontier": frontier,
        "leaderboard": leaderboard,
        "recent_lessons": [dict(l, tags=json.loads(l["tags"] or "[]")) for l in map(dict, lessons)],
        "lessons_digest": meta_get(con, "lessons_digest", ""),
        "budget": _budget_status(con),
        "calibration": meta_get_json(con, "calibration", None),
        "prediction_calibration": _prediction_calibration(con),
        "queued_specs": [{"queued_id": q["id"], "spec": json.loads(q["spec_json"])} for q in queued],
        "postmortem_due": [f["commit"] for f in frontier if f["blocked_count"] >= k_blocked],
        "running_experiments": running,
    }


def cmd_frontier(con, args):
    return {"ok": True, "frontier": _frontier_rows(con),
            "frontier_commit": meta_get(con, "frontier_commit")}


def cmd_leaderboard(con, args):
    snap = cmd_round_snapshot(con, args)
    return {"ok": True, "leaderboard": snap["leaderboard"]}


def cmd_exp_start(con, args):
    p = jload(args)
    write_tx(con)
    con.execute(
        "INSERT INTO experiments(exp_id,round,type,parent_commit,hypothesis,direction_tags,"
        "predicted_gain_pct,predicted_mechanism,status,worktree_path,branch) "
        "VALUES(?,?,?,?,?,?,?,?,'running',?,?) "
        "ON CONFLICT(exp_id) DO UPDATE SET status='running', worktree_path=excluded.worktree_path, "
        "branch=excluded.branch",
        (p["exp_id"], p.get("round"), p.get("type"), p.get("parent_commit"),
         p.get("hypothesis"), json.dumps(p.get("direction_tags", [])),
         p.get("predicted_gain_pct"), p.get("predicted_mechanism"),
         p.get("worktree_path"), p.get("branch")))
    con.execute("COMMIT")
    return {"ok": True, "exp_id": p["exp_id"]}


def cmd_exp_finish(con, args):
    p = jload(args)
    exp_id = p.pop("exp_id")
    if p.get("status") not in FINAL_STATUSES:
        return {"ok": False, "error": "invalid final status %r" % p.get("status")}
    cols = ["status", "review_iterations", "sanitizer_clean", "device_id", "bench_tier",
            "latency_us_mean", "latency_us_std", "speedup_vs_parent", "achieved_gain_pct",
            "prediction_gap_pct", "ncu_path", "diagnosis", "failure_reason", "lesson_ids",
            "commit_hash"]
    sets, vals = ["finished_at=datetime('now')"], []
    for c in cols:
        if c in p:
            sets.append("%s=?" % c)
            vals.append(p[c])
    if "key_metrics" in p:
        sets.append("key_metrics_json=?")
        vals.append(json.dumps(p["key_metrics"]))
    vals.append(exp_id)
    write_tx(con)
    cur = con.execute("UPDATE experiments SET %s WHERE exp_id=?" % ",".join(sets), vals)
    con.execute("COMMIT")
    if cur.rowcount == 0:
        return {"ok": False, "error": "unknown exp_id %r (call exp-start first)" % exp_id}
    return {"ok": True, "exp_id": exp_id, "status": p["status"]}


def cmd_exp_get(con, args):
    row = con.execute("SELECT * FROM experiments WHERE exp_id=?", (args.exp_id,)).fetchone()
    if not row:
        return {"ok": False, "error": "unknown exp_id %r" % args.exp_id}
    d = dict(row)
    for k in ("direction_tags",):
        d[k] = json.loads(d[k] or "[]")
    if d.get("key_metrics_json"):
        d["key_metrics"] = json.loads(d.pop("key_metrics_json"))
    return {"ok": True, "experiment": d}


def cmd_recover_orphans(con, args):
    write_tx(con)
    rows = con.execute(
        "SELECT exp_id, worktree_path, branch FROM experiments WHERE status='running'").fetchall()
    con.execute("UPDATE experiments SET status='gen_failed', "
                "failure_reason='orchestrator_restart', finished_at=datetime('now') "
                "WHERE status='running'")
    con.execute("COMMIT")
    return {"ok": True, "orphaned": [dict(r) for r in rows]}


def cmd_checkpoint_upsert(con, args):
    p = jload(args)
    write_tx(con)
    best = con.execute(
        "SELECT MIN(latency_us) AS m FROM checkpoints WHERE retired=0 AND latency_us IS NOT NULL"
    ).fetchone()["m"]
    lat = p.get("latency_us")
    new_best = best is None or (lat is not None and lat < best)
    con.execute(
        "INSERT INTO checkpoints(commit_hash,parent_commit,round_created,latency_us,"
        "assumptions_json,strategy_set_hash,ncu_fingerprint,key_metrics_json) "
        "VALUES(?,?,?,?,?,?,?,?) "
        "ON CONFLICT(commit_hash) DO UPDATE SET latency_us=excluded.latency_us, "
        "key_metrics_json=COALESCE(excluded.key_metrics_json, key_metrics_json)",
        (p["commit"], p.get("parent_commit"), p.get("round"), lat,
         json.dumps(sorted(p.get("assumptions", []))),
         p.get("strategy_set_hash"), p.get("ncu_fingerprint"),
         json.dumps(p.get("key_metrics")) if p.get("key_metrics") else None))
    if new_best:
        meta_set(con, "frontier_commit", p["commit"])
    con.execute("COMMIT")
    return {"ok": True, "commit": p["commit"], "new_best": bool(new_best)}


def cmd_blocked_mark(con, args):
    p = jload(args)
    rnd = int(p["round"])
    k_blocked = int(p.get("k_blocked") or cfg(con).get("K_BLOCKED", 4))
    write_tx(con)
    parents = con.execute(
        "SELECT DISTINCT parent_commit FROM experiments WHERE round=? AND parent_commit IS NOT NULL",
        (rnd,)).fetchall()
    blocked, due = [], []
    for row in parents:
        parent = row["parent_commit"]
        good = con.execute(
            "SELECT COUNT(*) AS n FROM experiments WHERE round=? AND parent_commit=? "
            "AND status IN (%s)" % ",".join("?" * len(GOOD_STATUSES)),
            (rnd, parent) + GOOD_STATUSES).fetchone()["n"]
        if good:
            con.execute("UPDATE checkpoints SET blocked_count=0 WHERE commit_hash=?", (parent,))
            continue
        cur = con.execute(
            "UPDATE checkpoints SET blocked_count=blocked_count+1 WHERE commit_hash=? AND retired=0",
            (parent,))
        if cur.rowcount:
            bc = con.execute("SELECT blocked_count FROM checkpoints WHERE commit_hash=?",
                             (parent,)).fetchone()["blocked_count"]
            blocked.append({"commit": parent, "blocked_count": bc})
            if bc >= k_blocked:
                due.append(parent)
    con.execute("COMMIT")
    return {"ok": True, "blocked": blocked, "postmortem_due": due}


def cmd_dedup_check(con, args):
    p = jload(args)
    noise = float(p.get("noise_pct", 1.0))
    cands = p.get("candidates", [])
    existing = con.execute(
        "SELECT commit_hash, latency_us, ncu_fingerprint, strategy_set_hash "
        "FROM checkpoints WHERE retired=0").fetchall()
    merge, soft = [], []

    def close(a, b):
        return a is not None and b is not None and b > 0 and abs(a - b) / b * 100.0 <= noise

    seen = [(e["commit_hash"], e["latency_us"], e["ncu_fingerprint"]) for e in existing]
    for c in cands:
        hit = None
        for commit, lat, fp in seen:
            if commit == c.get("commit"):
                continue
            if fp and c.get("fingerprint") and fp == c["fingerprint"] and close(c.get("latency_us"), lat):
                hit = commit
                break
        if hit:
            merge.append({"exp_id": c["exp_id"], "into_commit": hit})
        else:
            seen.append((c.get("commit"), c.get("latency_us"), c.get("fingerprint")))
        for e in existing:
            if e["strategy_set_hash"] and c.get("strategy_set_hash") == e["strategy_set_hash"] \
               and e["commit_hash"] != c.get("commit"):
                soft.append({"exp_id": c["exp_id"], "existing_commit": e["commit_hash"]})
                break
    if merge:
        write_tx(con)
        for m in merge:
            con.execute("UPDATE experiments SET merged_into=? WHERE exp_id=?",
                        (m["into_commit"], m["exp_id"]))
        con.execute("COMMIT")
    return {"ok": True, "merge": merge, "soft": soft}


def cmd_trajectory(con, args):
    chain = []
    commit = args.commit
    seen = set()
    while commit and commit not in seen:
        seen.add(commit)
        ck = con.execute("SELECT * FROM checkpoints WHERE commit_hash=?", (commit,)).fetchone()
        if not ck:
            break
        exp = con.execute(
            "SELECT exp_id,hypothesis,direction_tags,predicted_gain_pct,achieved_gain_pct,"
            "prediction_gap_pct,diagnosis,status FROM experiments WHERE commit_hash=? "
            "ORDER BY id DESC LIMIT 1", (commit,)).fetchone()
        chain.append({
            "commit": commit, "parent": ck["parent_commit"],
            "latency_us": ck["latency_us"],
            "assumptions": json.loads(ck["assumptions_json"]),
            "blocked_count": ck["blocked_count"],
            "key_metrics": json.loads(ck["key_metrics_json"]) if ck["key_metrics_json"] else None,
            "created_by": dict(exp, direction_tags=json.loads(exp["direction_tags"] or "[]")) if exp else None,
        })
        commit = ck["parent_commit"]
    chain.reverse()
    return {"ok": True, "trajectory": chain}


def _descendants(con, root):
    kids = {}
    for r in con.execute("SELECT commit_hash, parent_commit FROM checkpoints").fetchall():
        kids.setdefault(r["parent_commit"], []).append(r["commit_hash"])
    out, stack = [], [root]
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(kids.get(c, []))
    return out


def cmd_rewind(con, args):
    p = jload(args)
    suspect = p["suspect_checkpoint"]
    tag = p.get("suspect_tag")
    write_tx(con)
    # newest ancestor whose assumption set lacks the suspect tag
    target, commit, seen = None, suspect, set()
    while commit and commit not in seen:
        seen.add(commit)
        ck = con.execute("SELECT parent_commit, assumptions_json FROM checkpoints "
                         "WHERE commit_hash=?", (commit,)).fetchone()
        if not ck:
            break
        if commit != suspect and (not tag or tag not in json.loads(ck["assumptions_json"])):
            target = commit
            break
        commit = ck["parent_commit"]
    if target is None:
        con.execute("COMMIT")
        return {"ok": False, "error": "no ancestor lacking tag %r above %s" % (tag, suspect)}
    retired = _descendants(con, suspect)
    con.executemany("UPDATE checkpoints SET retired=1 WHERE commit_hash=?",
                    [(c,) for c in retired])
    meta_set(con, "frontier_commit", target)
    con.execute("COMMIT")
    return {"ok": True, "rewound_to": target, "retired": retired,
            "lesson_id": p.get("lesson_id")}


def cmd_queue_spec(con, args):
    p = jload(args)
    write_tx(con)
    cur = con.execute("INSERT INTO queued_specs(round_queued, spec_json) VALUES(?,?)",
                      (p.get("round"), json.dumps(p["spec"])))
    con.execute("COMMIT")
    return {"ok": True, "queued_id": cur.lastrowid}


def cmd_queued_consume(con, args):
    p = jload(args)
    write_tx(con)
    con.executemany("UPDATE queued_specs SET consumed=1 WHERE id=?",
                    [(i,) for i in p.get("ids", [])])
    con.execute("COMMIT")
    return {"ok": True}


# --- lessons ----------------------------------------------------------------

def _lessons_paths(args):
    sd = state_dir(args)
    return os.path.join(sd, "bitlessons.jsonl"), os.path.join(sd, "bitlessons.md")


def _render_lessons_md(jsonl_path, md_path):
    lines = ["# BitLessons (append-only; superseded entries struck through)", ""]
    superseded = set()
    lessons = []
    if os.path.exists(jsonl_path):
        for raw in open(jsonl_path):
            if raw.strip():
                l = json.loads(raw)
                lessons.append(l)
                if l.get("supersedes"):
                    superseded.add(l["supersedes"])
    for l in lessons:
        mark = "~~" if l["id"] in superseded else ""
        scope = l.get("scope", {})
        lines.append("- %s**%s** [%s] (conf %.2f) %s%s" % (
            mark, l["id"], l.get("type", "?"), l.get("confidence", 0),
            l.get("claim", ""), mark))
        if l.get("mechanism"):
            lines.append("  - why: %s" % l["mechanism"])
        if scope:
            lines.append("  - scope: %s" % json.dumps(scope))
        if l.get("evidence"):
            lines.append("  - evidence: %s" % ", ".join(l["evidence"]))
        if l.get("supersedes"):
            lines.append("  - supersedes: %s" % l["supersedes"])
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def cmd_lessons_append(con, args):
    p = jload(args)
    lessons = p if isinstance(p, list) else [p]
    jsonl_path, md_path = _lessons_paths(args)
    write_tx(con)
    n = con.execute("SELECT COUNT(*) AS n FROM lessons_index").fetchone()["n"]
    ids = []
    with open(jsonl_path, "a") as f:
        for l in lessons:
            n += 1
            lid = l.get("id") or "L%04d" % n
            l["id"] = lid
            f.write(json.dumps(l) + "\n")
            con.execute(
                "INSERT OR REPLACE INTO lessons_index(id,type,tags,scope_json,confidence,claim) "
                "VALUES(?,?,?,?,?,?)",
                (lid, l.get("type"), json.dumps(l.get("tags", [])),
                 json.dumps(l.get("scope", {})), l.get("confidence", 0.5), l.get("claim", "")))
            if l.get("supersedes"):
                con.execute("UPDATE lessons_index SET superseded_by=? WHERE id=?",
                            (lid, l["supersedes"]))
            ids.append(lid)
    con.execute("COMMIT")
    _render_lessons_md(jsonl_path, md_path)
    return {"ok": True, "ids": ids}


def cmd_lessons_query(con, args):
    tags = set((args.tags or "").split(",")) - {""}
    arch = args.arch
    rows = con.execute("SELECT * FROM lessons_index WHERE superseded_by IS NULL "
                       "ORDER BY confidence DESC, created_at DESC").fetchall()
    selected = []
    for r in rows:
        rtags = set(json.loads(r["tags"] or "[]"))
        scope = json.loads(r["scope_json"] or "{}")
        if tags and not (rtags & tags):
            continue
        if arch and scope.get("arch") and arch not in str(scope["arch"]):
            continue
        selected.append(r["id"])
        if len(selected) >= args.max:
            break
    jsonl_path, _ = _lessons_paths(args)
    bodies = {}
    if os.path.exists(jsonl_path):
        for raw in open(jsonl_path):
            if raw.strip():
                l = json.loads(raw)
                bodies[l["id"]] = l   # last write wins (append-only updates)
    return {"ok": True, "lessons": [bodies[i] for i in selected if i in bodies]}


def cmd_digest_set(con, args):
    p = jload(args)
    write_tx(con)
    meta_set(con, "lessons_digest", p.get("digest", ""))
    con.execute("COMMIT")
    return {"ok": True}


# --- budget -----------------------------------------------------------------

def cmd_budget_ingest(con, args):
    sd = state_dir(args)
    path = os.path.join(sd, "logs", "gpu_minutes.jsonl")
    write_tx(con)
    offset = int(meta_get(con, "gpu_minutes_offset", "0"))
    added_s = 0.0
    new_offset = offset
    if os.path.exists(path):
        with open(path) as f:
            f.seek(offset)
            for line in f:
                new_offset += len(line.encode())
                line = line.strip()
                if not line:
                    continue
                try:
                    added_s += float(json.loads(line).get("wall_s", 0))
                except ValueError:
                    continue
    rnd = int(meta_get(con, "current_round", "0"))
    con.execute("INSERT OR IGNORE INTO budget_ledger(round) VALUES(?)", (rnd,))
    con.execute("UPDATE budget_ledger SET gpu_minutes_used=gpu_minutes_used+? WHERE round=?",
                (added_s / 60.0, rnd))
    meta_set(con, "gpu_minutes_offset", str(new_offset))
    con.execute("COMMIT")
    return {"ok": True, "added_gpu_minutes": round(added_s / 60.0, 3),
            **_budget_status(con)}


def cmd_budget_add(con, args):
    p = jload(args)
    rnd = int(p.get("round") or meta_get(con, "current_round", "0"))
    write_tx(con)
    con.execute("INSERT OR IGNORE INTO budget_ledger(round) VALUES(?)", (rnd,))
    if p.get("tokens_used"):
        con.execute("UPDATE budget_ledger SET tokens_used=tokens_used+? WHERE round=?",
                    (int(p["tokens_used"]), rnd))
    if p.get("gpu_minutes"):
        con.execute("UPDATE budget_ledger SET gpu_minutes_used=gpu_minutes_used+? WHERE round=?",
                    (float(p["gpu_minutes"]), rnd))
    con.execute("COMMIT")
    return {"ok": True, **_budget_status(con)}


def cmd_budget_status(con, args):
    return {"ok": True, **_budget_status(con)}


def cmd_status(con, args):
    snap = cmd_round_snapshot(con, args)
    lines = ["WarpSpeed status - round %d" % snap["round"],
             "frontier: %s" % (snap["frontier_commit"] or "(none)"),
             "", "leaderboard:"]
    for i, e in enumerate(snap["leaderboard"]):
        lines.append("  %d. %s  %.1f us  %s" % (i + 1, (e["commit"] or "")[:10],
                                                e["latency_us"] or -1, ",".join(e["assumptions"])))
    lines.append("")
    lines.append("live checkpoints: %d  (postmortem due: %s)" % (
        len(snap["frontier"]), ", ".join(c[:10] for c in snap["postmortem_due"]) or "none"))
    b = snap["budget"]
    lines.append("budget: %.1f gpu-min used%s, %d tokens" % (
        b["gpu_minutes_used"],
        " / %s" % b["gpu_minutes_total"] if b["gpu_minutes_total"] else "",
        b["tokens_used"]))
    lines.append("recent lessons:")
    for l in snap["recent_lessons"][:8]:
        lines.append("  [%s] %s" % (l["id"], l["claim"]))
    if args.full:
        lines.append("")
        lines.append("recent experiments:")
        for r in con.execute("SELECT exp_id,status,speedup_vs_parent,hypothesis FROM experiments "
                             "ORDER BY id DESC LIMIT 15").fetchall():
            lines.append("  %s %-16s %s%% | %s" % (
                r["exp_id"], r["status"],
                round(r["speedup_vs_parent"], 2) if r["speedup_vs_parent"] is not None else "-",
                (r["hypothesis"] or "")[:70]))
    text = "\n".join(lines)
    sys.stderr.write(text + "\n")
    return {"ok": True, "text": text, "snapshot": snap}


COMMANDS = {
    "init": cmd_init,
    "config-set": cmd_config_set, "config-get": cmd_config_get,
    "calibration-set": cmd_calibration_set, "calibration-get": cmd_calibration_get,
    "seed-baseline": cmd_seed_baseline,
    "next-round": cmd_next_round,
    "round-snapshot": cmd_round_snapshot,
    "frontier": cmd_frontier, "leaderboard": cmd_leaderboard,
    "exp-start": cmd_exp_start, "exp-finish": cmd_exp_finish, "exp-get": cmd_exp_get,
    "recover-orphans": cmd_recover_orphans,
    "checkpoint-upsert": cmd_checkpoint_upsert,
    "blocked-mark": cmd_blocked_mark,
    "dedup-check": cmd_dedup_check,
    "trajectory": cmd_trajectory,
    "rewind": cmd_rewind,
    "queue-spec": cmd_queue_spec, "queued-consume": cmd_queued_consume,
    "lessons-append": cmd_lessons_append, "lessons-query": cmd_lessons_query,
    "digest-set": cmd_digest_set,
    "budget-ingest": cmd_budget_ingest, "budget-add": cmd_budget_add,
    "budget-status": cmd_budget_status,
    "status": cmd_status,
}


def main(argv):
    ap = argparse.ArgumentParser(prog="wsdb.py")
    ap.add_argument("--db", required=True)
    ap.add_argument("cmd", choices=sorted(COMMANDS))
    ap.add_argument("--json", default=None)
    ap.add_argument("--exp-id", dest="exp_id", default=None)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--tags", default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--max", type=int, default=15)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
    con = connect(args.db)
    try:
        if args.cmd != "init":
            try:
                needs_init = meta_get(con, "schema_version") is None
            except sqlite3.OperationalError:
                needs_init = True
            if needs_init:  # auto-init so any command works on a fresh db
                cmd_init(con, args)
        out = COMMANDS[args.cmd](con, args)
    except Exception as e:
        try:
            con.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        out = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    finally:
        con.close()
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
