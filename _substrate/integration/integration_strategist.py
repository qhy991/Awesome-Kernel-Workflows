#!/usr/bin/env python3
"""integration-strategist: the build/integration-mode entry point — the THIRD axis.

Same general technique as profiling-strategist / verification-strategist, applied
to the COMPILATION/INTEGRATION axis: routes HOW a candidate kernel is built+tested.
For a kernel that compiles as its own translation unit -> standalone. For a kernel
embedded in a project it cannot compile without (llama.cpp fattn.cu / mmq.cuh) ->
embedded_inplace / embedded_dispatch / registry_dispatch, reusing the existing
embedded/embedded_eval.js + ADAPTER_CONTRACT execution mechanism.

This is what makes ALL workflows able to handle llama.cpp embedded kernels, not
just the 6 that hardcode an integration_pattern: a standalone-default workflow
calls this, and if the kernel can't compile standalone, it switches to the
embedded eval sequence instead of the failing single-TU build.sh.

ASYMMETRIC AUTHORITY (same as the other two strategists):
  * AGENT decides the FUZZY input: classify can_compile_standalone (read the
    kernel's #includes / template deps / dispatch-symbol refs) and probe the host.
  * method SELECTION is a DETERMINISTIC function of declared inputs (resolve()).
  * build_fidelity (isolated/project_native/production) + reversible are stamped
    by the method's registry row -- never by the agent.
  * the actual build/test/revert still belongs to the workflow + adapter; this
    component only routes the mode and points at the eval mechanism.

Usage:
  integration_strategist.py resolve --kernel <path> --project-root <path> \
        --can-standalone <yes|no|uncertain> \
        [--host-probe '{"compiler":true,"project_build":true,"register_script":false,"runtime_registry":false,"reversibility_net":true}'] \
        [--registry integration_registry.json] [--cache cache.json] [--trajectory traj.jsonl]
"""
import sys, os, json, argparse, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REGISTRY = os.path.join(HERE, "integration_registry.json")
HOST_CAPS = ("compiler", "project_build", "register_script", "runtime_registry", "reversibility_net", "sol_execbench_cli")


def probe_host(override):
    """host capability probe; override wins so a test can pin an env deterministically.
    compiler presence is probed live; project_build/register_script/runtime_registry/
    reversibility_net are project-specific and default to False unless overridden."""
    live = {"compiler": shutil.which("nvcc") is not None or shutil.which("python3") is not None,
            "project_build": False, "register_script": False,
            "runtime_registry": False, "reversibility_net": False,
            "sol_execbench_cli": (os.environ.get("SOL_EXECBENCH") not in (None, "")) or shutil.which("sol-execbench") is not None}
    if override:
        live.update(override)
    return live


def _gate_ok(md, host):
    for req in md.get("requires", []):
        if not host.get(req):
            return False, f"requires '{req}' (absent)"
    return True, "host supports " + ",".join(md.get("requires", []))


def resolve(can_standalone, host, registry):
    """DETERMINISTIC. (can_standalone, host) -> routing decision."""
    methods = registry["methods"]
    host_sig = "+".join(sorted(c for c in HOST_CAPS if host.get(c))) or "none"

    # A kernel that can compile standalone needs no project coupling.
    if can_standalone == "yes" and host.get("compiler"):
        md = methods["standalone"]
        return _decision("standalone", md, host_sig, [], "kernel compiles as its own TU -> standalone (isolated)")

    # uncertain + compiler: optimistically try the cheap standalone path; a build
    # failure cleanly reveals the kernel is actually embedded (and the workflow can
    # re-resolve with can_standalone=no). uncertain without a compiler can't try
    # standalone -> fall through to the embedded ladder / derive.
    if can_standalone == "uncertain" and host.get("compiler"):
        md = methods["standalone"]
        return _decision("standalone", md, host_sig, [], "uncertain standalone-ness + compiler -> try standalone; build failure reveals embedded-ness")

    # Confirmed non-standalone (or uncertain-without-compiler): walk the embedded
    # preference ladder to the highest-fidelity reversible mode the host supports.
    # standalone is intentionally NOT in this ladder (a doomed single-TU compile of
    # a confirmed-embedded kernel is dishonest).
    tried = []
    for name in registry["preference"]["non_standalone"]:
        md = methods.get(name)
        if not md:
            continue
        ok, why = _gate_ok(md, host)
        if ok:
            return _decision(name, md, host_sig, tried, why)
        tried.append((name, why))

    # Nothing embedded is supported on this host -> autonomy: the agent must derive
    # an adapter (framework-integrator style) for this project, then re-resolve.
    return {
        "cache_key": f"{can_standalone}|{host_sig}", "method": "derive_adapter",
        "method_kind": "derive", "build_fidelity": None, "reversible": None,
        "requires": [], "eval_mechanism": None,
        "abstained_from": [t for (t, _w) in tried],
        "rationale": "kernel cannot compile standalone and no embedded mode is host-supported "
                     "-> agent must derive a project adapter (framework-integrator), then cache it back",
        "autonomy_directive": {"action": "derive_integration_adapter",
                               "must_emit": ["kind", "register_script", "build_command", "reversibility_net"],
                               "stamp_rule": "build_fidelity + reversible are assigned by method kind, not by you"},
    }


def _decision(name, md, host_sig, tried, why):
    return {
        "cache_key": f"{'yes' if name == 'standalone' else 'no'}|{host_sig}",
        "method": name,
        "method_kind": md["kind"],
        "build_fidelity": md["build_fidelity"],      # stamped by registry, not agent
        "reversible": md["reversible"],              # stamped by registry, not agent
        "requires": md.get("requires", []),
        "eval_mechanism": md.get("eval_mechanism"),
        "abstained_from": [t for (t, _w) in tried],
        "rationale": f"{name}: {why}",
    }


def _load(p):
    return json.load(open(p)) if p and os.path.exists(p) else {}


def get_decision(a):
    registry = _load(a.registry or DEFAULT_REGISTRY)
    host = probe_host(json.loads(a.host_probe) if a.host_probe else None)
    fresh = resolve(a.can_standalone, host, registry)
    if a.cache:
        cache = _load(a.cache)
        if fresh["cache_key"] in cache:
            if a.trajectory:
                open(a.trajectory, "a").write(json.dumps({"event": "integration_decision", "cache": "hit", "decision": cache[fresh["cache_key"]]}) + "\n")
            return cache[fresh["cache_key"]], True
        cache[fresh["cache_key"]] = fresh
        json.dump(cache, open(a.cache, "w"), indent=2)
    if a.trajectory:
        open(a.trajectory, "a").write(json.dumps({"event": "integration_decision", "cache": "miss", "decision": fresh}) + "\n")
    return fresh, False


def main():
    ap = argparse.ArgumentParser()
    p = ap.add_subparsers(dest="cmd", required=True)
    r = p.add_parser("resolve")
    r.add_argument("--kernel")
    r.add_argument("--project-root")
    r.add_argument("--can-standalone", default="uncertain", choices=["yes", "no", "uncertain"])
    r.add_argument("--registry")
    r.add_argument("--host-probe")
    r.add_argument("--cache")
    r.add_argument("--trajectory")
    a = ap.parse_args()
    decision, hit = get_decision(a)
    decision = {**decision, "_cache": "hit" if hit else "miss"}
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
