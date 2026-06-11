#!/usr/bin/env python3
"""scripts/patch_arg_guard.py

Insert the arg_guard import + unwrap call at the top of every workflow .js
in the catalog. Idempotent.

Run from repo root:
    python scripts/patch_arg_guard.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BLOCKLIST = {
    "_manifests", "_meta", "_substrate", "_templates", "_tools",
    "scripts", "docs", "badges", "node_modules", ".git",
}


def find_workflow_js():
    out = []
    for entry in sorted(REPO_ROOT.iterdir()):
        if not entry.is_dir() or entry.name in BLOCKLIST:
            continue
        for f in sorted(entry.iterdir()):
            if f.suffix == ".js":
                out.append(f)
    tpl = REPO_ROOT / "_templates"
    if tpl.is_dir():
        for f in sorted(tpl.iterdir()):
            if f.suffix == ".js":
                out.append(f)
    return out


def build_patch(file_path: Path) -> str:
    guard = REPO_ROOT / "_substrate" / "arg_guard.js"
    rel = os.path.relpath(guard, file_path.parent).replace("\\", "/")
    if not rel.startswith("."):
        rel = "./" + rel
    return (
        "\n"
        "// --- BEGIN arg_guard (auto-inserted by scripts/patch_arg_guard.py) ---\n"
        f"import {{ unwrapArgs as __unwrapArgs }} from '{rel}'\n"
        "// eslint-disable-next-line no-global-assign\n"
        "args = __unwrapArgs(typeof args === 'undefined' ? undefined : args)\n"
        "// --- END arg_guard ---\n"
        "\n"
    )


def find_meta_end(src: str) -> int:
    start = src.find("export const meta")
    if start < 0:
        return -1
    brace = src.find("{", start)
    if brace < 0:
        return -1
    depth = 0
    for i in range(brace, len(src)):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                j = i + 1
                while j < len(src) and src[j] in ";\t ":
                    j += 1
                if j < len(src) and src[j] == "\n":
                    j += 1
                return j
    return -1


def patch_one(file_path: Path) -> str:
    src = file_path.read_text(encoding="utf-8")
    if "arg_guard.js" in src:
        return "already_patched"
    if "export const meta" not in src:
        return "skipped_no_meta"
    insert_at = find_meta_end(src)
    if insert_at < 0:
        return "skipped_meta_parse_failed"
    patch = build_patch(file_path)
    file_path.write_text(src[:insert_at] + patch + src[insert_at:], encoding="utf-8")
    return "patched"


def main():
    files = find_workflow_js()
    counts = {"patched": 0, "already_patched": 0, "skipped_no_meta": 0, "skipped_meta_parse_failed": 0}
    for f in files:
        s = patch_one(f)
        counts[s] = counts.get(s, 0) + 1
        rel = f.relative_to(REPO_ROOT)
        print(f"{s:<28} {rel}")
    print()
    print("Summary: " + " ".join(f"{k}={v}" for k, v in counts.items()) + f" total={len(files)}")


if __name__ == "__main__":
    main()
