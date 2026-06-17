-- WarpSpeed search database schema (applied idempotently by `wsdb.py init`).
-- Lives at <state_dir>/search.sqlite, i.e. inside the TARGET project's state
-- dir - never inside the WarpSpeed workflow directory.

CREATE TABLE IF NOT EXISTS checkpoints (
  commit_hash       TEXT PRIMARY KEY,
  parent_commit     TEXT,
  round_created     INTEGER,
  latency_us        REAL,              -- confirmed (tier-2) number when available
  assumptions_json  TEXT NOT NULL DEFAULT '[]',  -- sorted strategy tags baked in
  strategy_set_hash TEXT,              -- dedup key #1 (hash of sorted assumptions)
  ncu_fingerprint   TEXT,              -- dedup key #2 (bucketed key metrics)
  key_metrics_json  TEXT,              -- cached parsed NCU metrics (headroom source)
  blocked_count     INTEGER NOT NULL DEFAULT 0,
  retired           INTEGER NOT NULL DEFAULT 0,
  created_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiments (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  exp_id             TEXT UNIQUE NOT NULL,   -- r<round>e<i>
  round              INTEGER,
  type               TEXT,                   -- explore|exploit|wildcard|replay|ablation
  parent_commit      TEXT,
  commit_hash        TEXT,                   -- candidate commit (recorded even on failure)
  hypothesis         TEXT,
  direction_tags     TEXT,                   -- JSON array
  predicted_gain_pct REAL,                   -- MANDATORY in spec
  predicted_mechanism TEXT,
  status             TEXT NOT NULL DEFAULT 'running',
                     -- running | gen_failed | compile_error | incorrect |
                     -- review_rejected | correct_slower | correct_faster | new_best
  review_iterations  INTEGER,
  sanitizer_clean    INTEGER,
  device_id          TEXT,
  bench_tier         TEXT,                   -- screen | confirm
  latency_us_mean    REAL,
  latency_us_std     REAL,
  speedup_vs_parent  REAL,                   -- relative speedup pct from screening
  achieved_gain_pct  REAL,
  prediction_gap_pct REAL,                   -- predicted - achieved
  key_metrics_json   TEXT,
  ncu_path           TEXT,
  diagnosis          TEXT,
  failure_reason     TEXT,
  lesson_ids         TEXT,
  worktree_path      TEXT,
  branch             TEXT,
  merged_into        TEXT,                   -- commit this candidate was hard-deduped into
  created_at         TEXT DEFAULT (datetime('now')),
  finished_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_exp_round_parent ON experiments(round, parent_commit);
CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);

CREATE TABLE IF NOT EXISTS lessons_index (   -- queryable mirror of bitlessons.jsonl
  id            TEXT PRIMARY KEY,
  type          TEXT,
  tags          TEXT,                        -- JSON array
  scope_json    TEXT,
  confidence    REAL,
  superseded_by TEXT,
  claim         TEXT,
  created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS budget_ledger (
  round            INTEGER PRIMARY KEY,
  gpu_minutes_used REAL    NOT NULL DEFAULT 0,
  tokens_used      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS queued_specs (    -- replay/ablation specs queued across rounds
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  round_queued INTEGER,
  spec_json    TEXT NOT NULL,
  consumed     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (            -- frontier pointer, round counter, calibration, ...
  key   TEXT PRIMARY KEY,
  value TEXT
);
