#!/usr/bin/env python3
"""WarpSpeed correctness harness - PROJECT-OWNED, READ-ONLY TO AGENTS.

Scaffolded once by WarpSpeed init into <project>/harness/. A human completes
the two TODO functions below, reviews tolerances.json, then makes the whole
directory read-only (chmod -R a-w harness/). After that, agents only ever RUN
it via the fixed invocation:

    gpu_run pool -- python3 harness/correctness.py --impl <binary> [--shape S]

Contract (relied on by every WarpSpeed agent and script - do not change):
  - iterates ALL shapes in problem_shapes.json (or just --shape S)
  - reference and implementation see IDENTICAL inputs, deterministic seeds
  - per-dtype tolerances come from tolerances.json (human-reviewed, explicit)
  - exit 0 iff every shape passes; exit 1 otherwise; exit 2 on usage error
  - single-line JSON report on stdout:
      {"ok": bool, "seed": int, "shapes": {"<name>": {"correct": bool, ...}}}
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260612  # deterministic; never randomize


def load_json(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TODO (human, once): implement these two functions for YOUR kernel.
# ---------------------------------------------------------------------------

def make_inputs(shape, rng):
    """Build deterministic inputs for one shape (e.g. numpy arrays from rng).

    `shape` is one entry from problem_shapes.json; `rng` is a seeded
    numpy.random.Generator (or random.Random if numpy is unavailable).
    Return whatever run_reference/run_impl need.
    """
    raise NotImplementedError(
        "harness/correctness.py: make_inputs() not implemented - complete the "
        "harness (see harness/README.md) before running WarpSpeed")


def run_reference(shape, inputs):
    """Compute the reference output (e.g. numpy / torch eager / cuBLAS)."""
    raise NotImplementedError("harness/correctness.py: run_reference() not implemented")


def run_impl(impl_path, shape, inputs):
    """Run the candidate binary on `inputs` and return its output.

    Define your own I/O bridge (e.g. write inputs to a temp .npz, invoke
    `<impl> --io <in> <out> --shape <name>`, read outputs back). The binary
    must not choose its own inputs.
    """
    raise NotImplementedError("harness/correctness.py: run_impl() not implemented")


def compare(shape, ref, out, tolerances):
    """Tolerance-aware comparison. Default: allclose per dtype from
    tolerances.json. Override if your kernel needs a custom metric."""
    import numpy as np
    dtype = shape.get("dtype", "fp32")
    tol = tolerances[dtype]
    ok = bool(np.allclose(out, ref, rtol=tol["rtol"], atol=tol["atol"]))
    err = float(np.max(np.abs(np.asarray(out, dtype="float64") - np.asarray(ref, dtype="float64")))) if not ok else 0.0
    return ok, {"rtol": tol["rtol"], "atol": tol["atol"], "max_abs_err": err}


# ---------------------------------------------------------------------------
# Fixed driver - no project-specific edits needed below.
# ---------------------------------------------------------------------------

def main(argv):
    impl, only_shape = None, None
    i = 0
    while i < len(argv):
        if argv[i] == "--impl":
            impl = argv[i + 1]; i += 2
        elif argv[i] == "--shape":
            only_shape = argv[i + 1]; i += 2
        else:
            sys.stderr.write("usage: correctness.py --impl <binary> [--shape S]\n")
            return 2
        continue
    if not impl:
        sys.stderr.write("usage: correctness.py --impl <binary> [--shape S]\n")
        return 2

    shapes = load_json("problem_shapes.json")["shapes"]
    tolerances = load_json("tolerances.json")
    if only_shape:
        shapes = [s for s in shapes if s["name"] == only_shape]
        if not shapes:
            print(json.dumps({"ok": False, "error": "unknown shape %r" % only_shape}))
            return 2

    report = {"ok": True, "seed": SEED, "shapes": {}}
    for shape in shapes:
        try:
            import numpy as np
            rng = np.random.default_rng(SEED)
        except ImportError:
            import random
            rng = random.Random(SEED)
        try:
            inputs = make_inputs(shape, rng)
            ref = run_reference(shape, inputs)
            out = run_impl(impl, shape, inputs)
            ok, detail = compare(shape, ref, out, tolerances)
        except NotImplementedError as e:
            print(json.dumps({"ok": False, "error": str(e)}))
            return 1
        except Exception as e:  # any crash = incorrect, never silently passed
            ok, detail = False, {"exception": "%s: %s" % (type(e).__name__, e)}
        report["shapes"][shape["name"]] = {"correct": ok, **detail}
        if not ok:
            report["ok"] = False

    print(json.dumps(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
