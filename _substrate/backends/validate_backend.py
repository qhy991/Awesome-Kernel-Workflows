#!/usr/bin/env python3
"""L0 structural validator for a backend driver directory.

Spec: docs/superpowers/specs/2026-06-05-backend-driver-axis-design.md §4.4/§4.7/§4.9 (L0).
Stdlib only; hand-rolled checks (no YAML, no JSON-Schema, no Node). Machine-read driver
files are JSON (manifest.json/idioms.json) — see BACKEND-DRIVER-SDK.md "Deviations".

Usage:
  validate_backend.py <driver_dir>
Prints: {"ok": bool, "errors": [str, ...]}
Exit:   0 ok · 1 L0 errors · 3 validator's own bad-args guard (argparse missing-positional is exit 2)
"""
import os, sys, json, argparse

# Import the live method_gate so the idiom-reference check uses the REAL method names.
_SUB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # _substrate/
if _SUB not in sys.path:
    sys.path.insert(0, _SUB)
import method_gate  # noqa: E402

# The four canonical metric keys diagnose.py / to_evidence.py speak.
CANONICAL_METRICS = {"latency_ms", "dram_pct", "sm_pct", "occupancy"}
# The 4 meaningful bottleneck classes. Per spec §4.9 the allowed set a driver MAY list is
# these 4 PLUS the implicit `unknown` (listing `unknown` is permitted, not required).
MEANINGFUL_CLASSES = {"memory_bound", "compute_bound", "latency_occupancy", "overhead_bound"}
ALLOWED_BCLASSES = MEANINGFUL_CLASSES | {"unknown"}
# All real method_gate method names (sourced live so this can never drift from the table).
KNOWN_METHODS = {m for methods in method_gate.TABLE.values() for m in methods}
# A driver dir must not contain a copy of any _substrate/*.py script. Updating this set
# requires a substrate-scope change (spec §3.1).
SUBSTRATE_SCRIPTS = {'evidence_schema.py', 'anti_cheat.py', 'diagnose.py',
                     'method_gate.py', 'memory_store.py', 'verify_insight.py'}


def _load_json(path, errors):
    """Read a JSON file; append a message to errors and return None on failure."""
    if not os.path.isfile(path):
        errors.append(f"missing file: {os.path.basename(path)}")
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"unreadable {os.path.basename(path)}: {exc}")
        return None


def validate(driver_dir):
    """Run all L0 checks on driver_dir. Return {'ok': bool, 'errors': [str, ...]}."""
    errors = []
    dir_name = os.path.basename(os.path.normpath(driver_dir))

    manifest = _load_json(os.path.join(driver_dir, "manifest.json"), errors)
    idioms = _load_json(os.path.join(driver_dir, "idioms.json"), errors)

    # --- manifest checks ---
    if manifest is None:
        pass  # _load_json already recorded the missing/unreadable error
    elif not isinstance(manifest, dict):
        errors.append("manifest.json top-level value is not an object")
    else:
        # backend_id present and == dir name (the dispatch key)
        backend_id = manifest.get("backend_id")
        if not isinstance(backend_id, str) or not backend_id:
            errors.append("manifest.backend_id missing or not a string")
        elif backend_id != dir_name:
            errors.append(
                f"manifest.backend_id '{backend_id}' != directory name '{dir_name}'")

        # invoke fields must equal the fixed filenames (spec §4.2 footnote)
        for role, fixed in [('compiler', 'build.sh'), ('runner', 'run.sh')]:
            block = manifest.get(role)
            if isinstance(block, dict):
                inv = block.get('invoke')
                if inv is not None and inv != fixed:
                    errors.append(
                        f"manifest.{role}.invoke '{inv}' must equal '{fixed}'")
        prof = manifest.get('profiler')
        if isinstance(prof, dict):
            inv = prof.get('invoke')
            if inv is not None and inv != 'profile.sh':
                errors.append(
                    f"manifest.profiler.invoke '{inv}' must equal 'profile.sh'")
            te = prof.get('to_evidence')
            if te is not None and te != 'to_evidence.py':
                errors.append(
                    f"manifest.profiler.to_evidence '{te}' must equal 'to_evidence.py'")
        if 'threshold_profile' not in manifest:
            errors.append("manifest.threshold_profile missing (required §4.4)")

        caps = manifest.get("capabilities")
        if not isinstance(caps, dict):
            errors.append("manifest.capabilities missing or not an object")
        else:
            # capabilities.metrics keys subset of canonical keys
            metrics = caps.get("metrics")
            if not isinstance(metrics, dict):
                errors.append("manifest.capabilities.metrics missing or not an object")
            else:
                for key in metrics:
                    if key not in CANONICAL_METRICS:
                        errors.append(
                            f"capabilities.metrics key '{key}' is not a canonical metric "
                            f"(allowed: {sorted(CANONICAL_METRICS)})")
            # bottleneck_classes subset of the 4 meaningful classes
            bclasses = caps.get("bottleneck_classes")
            if not isinstance(bclasses, list):
                errors.append("manifest.capabilities.bottleneck_classes missing or not a list")
            else:
                for bc in bclasses:
                    if bc not in ALLOWED_BCLASSES:
                        errors.append(
                            f"capabilities.bottleneck_classes entry '{bc}' is not an "
                            f"allowed class (allowed: {sorted(ALLOWED_BCLASSES)})")

    # --- idioms checks ---
    if idioms is None:
        pass  # _load_json already recorded the missing/unreadable error
    elif not isinstance(idioms, dict):
        errors.append("idioms.json top-level value is not an object")
    else:
        # every methods key and every unsupported_methods entry is a real method_gate name
        methods = idioms.get("methods")
        if not isinstance(methods, dict):
            errors.append("idioms.methods missing or not an object")
        else:
            for name in methods:
                if name not in KNOWN_METHODS:
                    errors.append(
                        f"idioms.methods references unknown method_gate method '{name}'")
        unsupported = idioms.get("unsupported_methods", [])
        if not isinstance(unsupported, list):
            errors.append("idioms.unsupported_methods is not a list")
        else:
            for name in unsupported:
                if name not in KNOWN_METHODS:
                    errors.append(
                        f"idioms.unsupported_methods references unknown method_gate "
                        f"method '{name}'")

    # A driver dir must not contain a copy of any _substrate/*.py script.
    if os.path.isdir(driver_dir):
        for fname in os.listdir(driver_dir):
            if fname in SUBSTRATE_SCRIPTS:
                errors.append(
                    f"driver dir must not contain a copy of _substrate/{fname} "
                    f"(driver is data; never imports/shadows substrate)")

    return {"ok": len(errors) == 0, "errors": errors}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("driver_dir", help="path to the backend driver directory")
    a = ap.parse_args()
    if not os.path.isdir(a.driver_dir):
        print(json.dumps({"ok": False, "errors": [f"not a directory: {a.driver_dir}"]},
                         indent=2, ensure_ascii=False))
        return 3
    result = validate(a.driver_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
