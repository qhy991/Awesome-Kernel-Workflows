#!/usr/bin/env python3
"""render_config.py - materialize a WarpSpeed run config (deterministic).

This is the decoupling mechanism: the workflow script never resolves paths or
composes shell commands; this tool merges config.defaults.json with caller
overrides, resolves every path to ABSOLUTE form rooted at the caller-supplied
project/workflow dirs, renders the command templates, scans the wiki tag
index, and writes <state_dir>/config.json. The Init agent runs it once per
run and relays the JSON.

Usage:
  render_config.py --warpspeed-dir W --project-dir P [--state-dir S]
                   [--harness-dir H] [--set KEY=JSONVALUE]...
"""
import argparse
import json
import os
import re
import sys


def parse_frontmatter(path):
    fm = {}
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return fm
    m = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    if not m:
        return fm
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if val.startswith("[") and val.endswith("]"):
            fm[key] = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
        else:
            fm[key] = val.strip("'\"")
    return fm


def scan_wiki(wiki_dir, arch):
    """tag -> [absolute page paths], filtered to pages whose arch list
    includes this arch (or declares no arch scope)."""
    index = {}
    pages = []
    if not os.path.isdir(wiki_dir):
        return index, pages
    for name in sorted(os.listdir(wiki_dir)):
        if not name.endswith(".md") or name == "INDEX.md":
            continue
        path = os.path.join(wiki_dir, name)
        fm = parse_frontmatter(path)
        archs = fm.get("arch", [])
        if archs and arch not in archs:
            continue
        entry = {"path": path, "title": fm.get("title", name[:-3]),
                 "tags": fm.get("tags", []),
                 "expected_gain_pct": fm.get("expected_gain_pct", [])}
        pages.append(entry)
        for tag in entry["tags"]:
            index.setdefault(tag, []).append(path)
    return index, pages


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--warpspeed-dir", required=True)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--harness-dir", default=None)
    ap.add_argument("--set", dest="sets", action="append", default=[])
    args = ap.parse_args(argv)

    W = os.path.abspath(args.warpspeed_dir)
    P = os.path.abspath(args.project_dir)
    S = os.path.abspath(args.state_dir or os.path.join(P, ".warpspeed"))
    H = os.path.abspath(args.harness_dir or os.path.join(P, "harness"))

    defaults_path = os.path.join(W, "config", "config.defaults.json")
    with open(defaults_path) as f:
        cfg = json.load(f)

    for kv in args.sets:
        if "=" not in kv:
            raise SystemExit("--set expects KEY=VALUE, got %r" % kv)
        key, raw = kv.split("=", 1)
        try:
            cfg[key] = json.loads(raw)
        except ValueError:
            cfg[key] = raw

    target = str(cfg.get("target_gpu", "sm90"))
    arch = cfg.get("ARCH_MAP", {}).get(target, target if target.startswith("sm") else "sm90")
    cfg["arch"] = arch

    cfg["paths"] = {
        "warpspeed_dir": W, "project_dir": P, "state_dir": S, "harness_dir": H,
        "db": os.path.join(S, "search.sqlite"),
        "worktrees": os.path.join(S, "worktrees"),
        "review": os.path.join(S, "review"),
        "builds": os.path.join(S, "builds"),
        "ncu_cache": os.path.join(S, "ncu_cache"),
        "logs": os.path.join(S, "logs"),
        "results": os.path.join(S, "results"),
        "bitlessons": os.path.join(S, "bitlessons.jsonl"),
        "hardware_facts": os.path.join(W, "config", "hardware-facts-%s.md" % arch),
        "wiki_dir": os.path.join(W, "wiki"),
    }

    clock_env = ""
    if cfg.get("LOCKED_GRAPHICS_CLOCK"):
        clock_env += " WARPSPEED_LOCKED_GR=%s" % cfg["LOCKED_GRAPHICS_CLOCK"]
    if cfg.get("LOCKED_MEM_CLOCK"):
        clock_env += " WARPSPEED_LOCKED_MEM=%s" % cfg["LOCKED_MEM_CLOCK"]
    if cfg.get("CLOCK_CMD_PREFIX"):
        clock_env += " WARPSPEED_CLOCK_PREFIX='%s'" % cfg["CLOCK_CMD_PREFIX"]

    subs = {
        "warpspeed": W, "project": P, "state": S, "harness": H,
        "lock_dir": cfg.get("LOCK_DIR", "/tmp/warpspeed/locks"),
        "pool_devices": ",".join(str(d) for d in cfg.get("POOL_DEVICES", [0, 1, 2, 3, 4, 5])),
        "bench_device": cfg.get("BENCH_DEVICE", 6),
        "ncu_device": cfg.get("NCU_DEVICE", 7),
        "screen_reps": cfg.get("SCREEN_REPS", 50),
        "confirm_reps": cfg.get("CONFIRM_REPS", 200),
        "warmup_reps": cfg.get("WARMUP_REPS", 10),
        "ncu_sections": cfg.get("NCU_SECTIONS", ""),
        "clock_env": clock_env,
    }
    env_prefix = cfg.get("env_prefix_template", "").format(**subs)
    subs["env_prefix"] = env_prefix

    cfg["commands"] = {name: tpl.format(**subs)
                       for name, tpl in cfg.get("command_templates", {}).items()}
    cfg["env_prefix"] = env_prefix

    index, pages = scan_wiki(cfg["paths"]["wiki_dir"], arch)
    cfg["wiki_index"] = index
    cfg["wiki_pages"] = pages

    os.makedirs(S, exist_ok=True)
    out_path = os.path.join(S, "config.json")
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
    json.dump({"ok": True, "config_path": out_path, "config": cfg}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
