#!/usr/bin/env python3
"""Merge a legacy KerSor workflow_metadata.json into AKW manifest.yaml files.

One-time / idempotent: writes or updates the top-level `routing:` block per workflow.
For multi-js directories, adds `variants[]` when needed.

This archived migration tool is intentionally not wired to a current default
metadata path; pass the legacy JSON path explicitly when replaying the migration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

AKW = Path(__file__).resolve().parents[1]

SKIP_PARTS = {
    "_templates", "_tools", "_substrate", "_manifests", "_meta",
    "experiments", "kernel_cache", "candidates", "__pycache__",
    "scripts", "node_modules",
}

ROUTING_KEYS = {
    "method_category", "topology", "languages", "backends", "requires_ncu",
    "requires_harness", "role", "fidelity_boundary", "speedup_field",
    "best_kernel_field", "required_one_of", "integration_patterns",
    "known_broken", "max_kernel_lines", "supports_partial_patch",
    "requires_embedded_registration", "selection_tags", "required_args",
    "all_args", "framework_integrator_fallback",
}


def collect_js(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*.js")):
        rel = p.relative_to(root)
        if len(rel.parts) > 2:
            continue
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        out.append(p)
    return out


def meta_name(js: Path) -> str:
    text = js.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if "name:" in line and ("'" in line or '"' in line):
            for q in ("'", '"'):
                if f"name:{q}" in line.replace(" ", "") or f"name: {q}" in line:
                    start = line.index(q) + 1
                    end = line.index(q, start)
                    return line[start:end]
    return js.stem


def js_to_manifest(js: Path) -> Path:
    return js.parent / "manifest.yaml"


def build_routing(entry: dict) -> dict:
    routing = {}
    for k in ROUTING_KEYS:
        if k not in entry:
            continue
        v = entry[k]
        if v is None:
            routing[k] = None
        elif v == "":
            continue
        else:
            routing[k] = v
    return routing


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: scripts/merge-routing-into-manifests.py /path/to/legacy/workflow_metadata.json",
            file=sys.stderr,
        )
        return 2
    meta_path = Path(sys.argv[1])
    if not meta_path.is_file():
        print(f"ERROR: metadata not found: {meta_path}", file=sys.stderr)
        return 1

    workflows = json.loads(meta_path.read_text(encoding="utf-8"))["workflows"]
    js_files = collect_js(AKW)
    by_name = {meta_name(j): j for j in js_files}

    missing_meta = []
    updated = []

    # Group js by manifest directory
    by_dir: dict[Path, list[Path]] = {}
    for js in js_files:
        by_dir.setdefault(js.parent, []).append(js)

    for directory, js_list in sorted(by_dir.items()):
        manifest_path = directory / "manifest.yaml"
        if not manifest_path.is_file():
            for js in js_list:
                missing_meta.append(js)
            continue

        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        variants = list(data.get("variants") or [])
        primary_js = js_list[0] if len(js_list) == 1 else None

        if len(js_list) > 1:
            primary_name = (data.get("workflow") or {}).get("name") or data.get("name")
            for js in js_list:
                name = meta_name(js)
                if name == primary_name:
                    primary_js = js
                    break
            if primary_js is None:
                primary_js = js_list[0]

        for js in js_list:
            name = meta_name(js)
            if name not in workflows:
                missing_meta.append(js)
                continue
            routing = build_routing(workflows[name])
            if js == primary_js or len(js_list) == 1:
                data["routing"] = routing
            else:
                found = False
                for v in variants:
                    if isinstance(v, dict) and (
                        v.get("name") == name
                        or v.get("entrypoint") == js.name
                    ):
                        v["routing"] = routing
                        found = True
                        break
                if not found:
                    variants.append({
                        "name": name,
                        "entrypoint": js.name,
                        "routing": routing,
                    })
            updated.append(name)

        if variants:
            data["variants"] = variants

        manifest_path.write_text(
            yaml.dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

    if missing_meta:
        print("WARN: no legacy routing metadata entry for:", file=sys.stderr)
        for js in missing_meta:
            print(f"  {js.relative_to(AKW)}", file=sys.stderr)

    print(f"Updated routing in {len(by_dir)} manifest(s); {len(updated)} workflow(s)")
    return 0 if not missing_meta else 1


if __name__ == "__main__":
    raise SystemExit(main())
