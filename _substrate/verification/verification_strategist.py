#!/usr/bin/env python3
"""verification-strategist: the verification-depth entry point for GENERATOR
workflows (TritorX, KernelAgent, ...) that have no performance signal.

Same general technique as profiling-strategist, applied to a different routed
quantity: instead of "which profiler", it routes "how deeply to verify a generated
candidate". An agent picks the FUZZY input -- the verification NEED (prune vs
confirm) and a host capability probe; the substrate DETERMINISTICALLY maps
(need x host) to a method and STAMPS confidence by method. The model must NOT
assign confidence itself.

ASYMMETRIC AUTHORITY (same as the profiling twin):
  * agent decides the FUZZY inputs: classify need (prune/confirm), probe host.
  * method SELECTION is a DETERMINISTIC function of declared inputs (resolve()):
    same (need, host) -> same route, every time.
  * CONFIDENCE is stamped by the method's registry row -- never by the agent.
  * the actual correctness verdict still belongs to the workflow's test harness;
    this component only routes WHICH depth to invoke and how to tag its evidence.

The ladder (deepest first):
  reference_test (correctness, measured)  -- run vs reference, numerical compare
  smoke_test     (runtime,    measured)   -- compile + run, shape/finiteness only
  compile_lint   (compile,    inferred)   -- compile + lint, no execution
  static         (llm_inferred,hypothesized) -- source read only
`prune` skips the execution rungs (no reference_test/smoke_test) so early
screening stays cheap; `confirm` walks the full ladder to the deepest method the
host supports.

Usage:
  verification_strategist.py resolve --need prune \
        [--host-probe '{"compiler":true,"harness_runnable":true,"reference_available":false}'] \
        [--registry verification_registry.json] [--cache cache.json] [--trajectory traj.jsonl]
"""
import sys, os, json, argparse, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REGISTRY = os.path.join(HERE, "verification_registry.json")
HOST_CAPS = ("compiler", "harness_runnable", "reference_available")


def probe_host(override):
    """host capability probe; override wins so a test can pin an env deterministically.
    Live probes are best-effort (compiler presence); harness/reference availability is
    inherently task/host-specific and defaults to True (optimistic) unless overridden."""
    live = {"compiler": shutil.which("python3") is not None,  # triton/cuda compile via python host
            "harness_runnable": True, "reference_available": True}
    if override:
        live.update(override)
    return live


def _gate_ok(md, host):
    for req in md.get("requires", []):
        if not host.get(req):
            return False, f"requires '{req}' (absent)"
    return True, "host supports " + ",".join(md.get("requires", [])) + " -> " + md["kind"]


def resolve(need, host, registry):
    """DETERMINISTIC. (need, host) -> routing decision via the need's ladder."""
    need = need if need in registry["need_ladder"] else "default"
    methods = registry["methods"]
    host_sig = "+".join(sorted(c for c in HOST_CAPS if host.get(c))) or "none"
    tried = []
    for name in registry["need_ladder"][need]:
        md = methods.get(name)
        if not md:
            continue
        ok, why = _gate_ok(md, host)
        if ok:
            return {
                "cache_key": f"{need}|{host_sig}",
                "need": need,
                "method": name,
                "method_kind": md["kind"],
                "evidence_source": md["evidence"],        # stamped by registry, not agent
                "confidence": md["confidence"],           # stamped by registry, not agent
                "requires": md.get("requires", []),
                "abstained_from": [t for (t, _w) in tried],
                "rationale": f"{name}: {why}",
            }
        tried.append((name, why))
    return {"cache_key": f"{need}|{host_sig}", "need": need, "method": "static",
            "method_kind": "static", "evidence_source": "llm_inferred",
            "confidence": "hypothesized", "requires": [],
            "abstained_from": [t for (t, _w) in tried],
            "rationale": "no runnable verification method; source read only"}


def _load(p):
    return json.load(open(p)) if p and os.path.exists(p) else {}


def get_decision(a):
    registry = _load(a.registry or DEFAULT_REGISTRY)
    host = probe_host(json.loads(a.host_probe) if a.host_probe else None)
    fresh = resolve(a.need, host, registry)
    if a.cache:
        cache = _load(a.cache)
        if fresh["cache_key"] in cache:
            if a.trajectory:
                open(a.trajectory, "a").write(json.dumps({"event": "verification_decision", "cache": "hit", "decision": cache[fresh["cache_key"]]}) + "\n")
            return cache[fresh["cache_key"]], True
        cache[fresh["cache_key"]] = fresh
        json.dump(cache, open(a.cache, "w"), indent=2)
    if a.trajectory:
        open(a.trajectory, "a").write(json.dumps({"event": "verification_decision", "cache": "miss", "decision": fresh}) + "\n")
    return fresh, False


def main():
    ap = argparse.ArgumentParser()
    p = ap.add_subparsers(dest="cmd", required=True)
    r = p.add_parser("resolve")
    r.add_argument("--need", default="default", choices=["prune", "confirm", "default"])
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
