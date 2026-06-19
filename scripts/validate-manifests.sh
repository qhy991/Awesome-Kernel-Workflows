#!/usr/bin/env bash
# validate-manifests.sh — every routable workflow .js must have manifest coverage.
#
# Coverage rule (per .js file):
#   1. Same directory contains manifest.yaml whose workflow.name or entrypoint
#      matches the .js basename, OR
#   2. Parent directory manifest.yaml lists the file under variants[].
#
# Usage: ./scripts/validate-manifests.sh [REPO_ROOT]
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="$(cd "$ROOT" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 required" >&2
  exit 1
fi

python3 - "$ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
skip_parts = {
    "_templates", "_tools", "_substrate", "_manifests", "_meta",
    "experiments", "kernel_cache", "candidates", "__pycache__",
    "scripts", "node_modules",
}

def collect_js_files():
    out = []
    for p in sorted(root.rglob("*.js")):
        rel = p.relative_to(root)
        if len(rel.parts) > 2:
            continue
        if any(part in skip_parts for part in rel.parts):
            continue
        out.append(p)
    return out

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required (pip install pyyaml)", file=sys.stderr)
    sys.exit(1)

def load_manifest(path: Path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def js_covered(js_path: Path, siblings: list[Path]) -> tuple[bool, str]:
    js_name = js_path.name
    stem = js_path.stem
    manifest = js_path.parent / "manifest.yaml"
    if not manifest.is_file():
        return False, ""
    if len(siblings) == 1:
        return True, str(manifest)
    data = load_manifest(manifest)
    wf = data.get("workflow") or {}
    primary = (data.get("entrypoint") or wf.get("output_filename") or "").strip()
    wf_name = (wf.get("name") or data.get("name") or "").strip()
    if primary == js_name or wf_name == stem:
        return True, str(manifest)
    variants = data.get("variants") or []
    for v in variants:
        if not isinstance(v, dict):
            continue
        ep = (v.get("entrypoint") or v.get("output_filename") or "").strip()
        name = (v.get("name") or "").strip()
        if ep == js_name or name == stem:
            return True, f"{manifest}#variants"
    return False, ""

errors = []
all_js = collect_js_files()
by_dir: dict[Path, list[Path]] = {}
for js in all_js:
    by_dir.setdefault(js.parent, []).append(js)

for js in all_js:
    ok, where = js_covered(js, by_dir[js.parent])
    if not ok:
        errors.append(f"  - {js.relative_to(root)} (no manifest.yaml coverage)")

if errors:
    print("validate-manifests: FAIL — uncovered workflow .js files:", file=sys.stderr)
    print("\n".join(errors), file=sys.stderr)
    print(
        "\nFix: add manifest.yaml or variants[] entry in the workflow directory.",
        file=sys.stderr,
    )
    sys.exit(1)

n = len(all_js)
print(f"validate-manifests: OK — {n} workflow .js files covered by manifest.yaml")
PY
