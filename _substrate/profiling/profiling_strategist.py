#!/usr/bin/env python3
"""profiling-strategist: the ONE profiling entry point every AKW workflow calls.

Picks a profiling METHOD per (backend x task x host) from a declarative registry,
then lets the substrate stamp how much to trust it. Measurement-side twin of
KerSor's framework-integrator.

THREE PROPERTIES the user asked for:
  * knows common backend methods  -> profiler_registry.json (ncu/nsys/rocprof/
    msprof/metal-capture/xpu-vtune/perf_heuristic/static). Add a backend = add a
    registry row, NOT strategist code.
  * generality                    -> resolve() is registry-driven; no per-backend
    branches live in code.
  * autonomy                      -> an unknown backend routes to 'derive_adapter':
    the agent derives a method at runtime (framework-integrator style) and the
    decision is cached back, so it is paid once.

ASYMMETRIC AUTHORITY (preserved):
  * an AGENT decides the FUZZY inputs: classify the task (op_class/size), probe the
    host, and -- only for unknown backends -- derive a new adapter.
  * method SELECTION over known backends is a DETERMINISTIC function of declared
    inputs (resolve()): same (backend, task, host) -> same route, every time.
  * CONFIDENCE is stamped by the registry row of the chosen METHOD -- never by the
    agent. classification of metrics into a bottleneck still belongs to diagnose.py.

FINDING (resolved): evidence_schema.py's EVIDENCE enum now includes 'native_profiler'.
The strategist stamps literal Nsight Compute as evidence='ncu' and every other native
hw profiler (rocprof/msprof/vtune/metal-capture) as evidence='native_profiler', so
provenance is honest instead of借用 the ncu tag.

Usage:
  profiling_strategist.py resolve --backend-manifest m.json --task attention --size large
        [--registry profiler_registry.json] [--host-probe '{"ncu":true}']
        [--cache cache.json] [--trajectory traj.jsonl]
  profiling_strategist.py run ...same... --baseline base.txt [--candidate cand.txt]
        --peak-gflops 5200 --peak-gbs 200
"""
import sys, os, json, argparse, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REGISTRY = os.path.join(HERE, "profiler_registry.json")
PROBE_TOOLS = ("ncu", "nsys", "rocprof", "msprof", "xcrun", "vtune")

# --- task knowledge: what a task most needs, and the full set it would ask for ---
PRIMARY_METRIC = {
    "attention": "dram_pct", "gemm": "sm_pct", "elementwise": "dram_pct",
    "reduction": "dram_pct", "default": "latency_ms",
}
METRIC_SETS = {
    "attention":   ["latency_ms", "dram_pct", "sm_pct"],
    "gemm":        ["latency_ms", "sm_pct", "dram_pct", "occupancy"],
    "elementwise": ["latency_ms", "dram_pct", "occupancy"],
    "reduction":   ["latency_ms", "dram_pct", "occupancy"],
    "default":     ["latency_ms"],
}


def desired_metrics(task, size):
    ms = list(METRIC_SETS.get(task, METRIC_SETS["default"]))
    if size == "tiny":  # tiny kernels: skip full roofline, latency+occ suffices
        ms = [m for m in ms if m in ("latency_ms", "occupancy")] or ["latency_ms"]
    return ms


def native_tool_for(manifest, registry):
    """the host binary to probe for this backend's native profiler (manifest is SSOT)."""
    name = (manifest.get("profiler") or {}).get("name")
    if not name:
        return None, None
    alias = registry.get("probe_tool_alias", {})
    return name, alias.get(name, name)


def probe_host(override, extra_tools=()):
    """live capability probe; override wins so a test can pin an env deterministically."""
    tools = set(PROBE_TOOLS) | {t for t in extra_tools if t}
    live = {t: (shutil.which(t) is not None) for t in tools}
    live["throughput_runnable"] = True  # run.sh / test-backend-ops perf is ~always runnable
    if override:
        live.update(override)
    return live


def _build_native(manifest, registry):
    """construct the native method def FROM the manifest (no per-backend table)."""
    name, tool = native_tool_for(manifest, registry)
    if not name:
        return None
    caps = manifest.get("capabilities", {}).get("metrics", {})
    # honest provenance: literal Nsight Compute -> evidence 'ncu'; any other native
    # hw profiler -> 'native_profiler' (its own enum value, not a borrowed 'ncu').
    ev = "ncu" if name == "ncu" else registry["native_stamp"]["evidence"]
    return {**registry["native_stamp"], "evidence": ev,
            "profiler": name, "tool": tool,
            "metrics": [m for m, ok in caps.items() if ok],
            "normalizer": (manifest.get("profiler") or {}).get("to_evidence", "to_evidence.py")}


def _gate_ok(md, host, primary):
    kind = md.get("kind")
    if kind == "native_profiler":
        if not host.get(md.get("tool")):
            return False, f"tool '{md.get('tool')}' absent"
        # hardware-agnostic gate: a native profiler is useful only if it yields
        # latency AND at least one bottleneck axis (dram/sm). A latency-only native
        # (e.g. Metal counters) gives no bottleneck signal -> fall to perf_heuristic,
        # whose throughput (GB/s, GFLOPS) DOES carry a bottleneck axis. We do NOT gate
        # on the exact task `primary` (sm_pct/dram_pct are vendor-named; a backend that
        # lacks one is not thereby useless -- e.g. Intel VTune gives latency+dram).
        metrics = md.get("metrics", [])
        if "latency_ms" not in metrics or not any(m in metrics for m in ("dram_pct", "sm_pct")):
            return False, f"native '{md.get('tool')}' is latency-only (no bottleneck axis)"
        return True, f"'{md.get('tool')}' present, covers latency + a bottleneck axis"
    if kind == "throughput":
        req = md.get("requires")
        if req and not host.get(req):
            return False, f"requires '{req}'"
        return True, "throughput runnable"
    if kind == "static":
        return True, "always available (source read)"
    return False, "unknown method kind"


def resolve(manifest, task, size, host, registry):
    """DETERMINISTIC for any backend whose manifest declares a profiler. (backend,
    task, host) -> routing decision via a GENERIC ladder; native comes from manifest."""
    backend_id = manifest.get("backend_id")
    want = desired_metrics(task, size)
    primary = PRIMARY_METRIC.get(task, "latency_ms")
    host_sig = "+".join(sorted(t for t in host if t != "throughput_runnable" and host.get(t))) or "none"

    if not backend_id:
        # AUTONOMY: no manifest for this backend -> agent derives an adapter at
        # runtime (framework-integrator style); cacheable so it is paid once.
        return {
            "cache_key": f"unknown|{task}|{size}|{host_sig}", "method": "derive_adapter",
            "method_kind": "derive", "evidence_source": None, "confidence": None,
            "normalizer": None, "requested_metrics": want, "primary_metric": primary,
            "profiler_name": None, "profile_invoke": None, "abstained_from": [],
            "rationale": "no backend manifest -> agent must derive a profiling adapter, then cache it back",
            "autonomy_directive": {"action": "derive_profiling_adapter",
                                   "must_emit": ["tool", "invoke", "metrics", "normalizer"],
                                   "stamp_rule": "confidence is assigned by method kind, not by you"},
        }

    native = _build_native(manifest, registry)
    fb = registry["fallback_methods"]
    tried = []
    for step in registry["ladder"]:
        md = native if step == "native" else fb.get(step)
        if not md:
            tried.append((step, "not declared")); continue
        ok, why = _gate_ok(md, host, primary)
        if ok:
            return {
                "cache_key": f"{backend_id}|{task}|{size}|{host_sig}",
                "method": "native_profiler" if step == "native" else step,
                "method_kind": md["kind"],
                "evidence_source": md["evidence"],        # stamped by registry/native_stamp, not agent
                "confidence": md["confidence"],           # stamped by registry/native_stamp, not agent
                "normalizer": md["normalizer"],
                "profiler_name": md.get("profiler"),
                "requested_metrics": want,
                "coverage_expected": [m for m in want if m in md.get("metrics", [])],
                "primary_metric": primary,
                "profile_invoke": (manifest.get("profiler") or {}).get("invoke") if step == "native" else None,
                "abstained_from": [t for (t, _w) in tried],
                "rationale": f"{step}: {why}",
            }
        tried.append((step, why))
    return {"cache_key": f"{backend_id}|{task}|{size}|{host_sig}", "method": "static",
            "method_kind": "static", "evidence_source": "llm_inferred", "confidence": "hypothesized",
            "normalizer": None, "requested_metrics": want, "primary_metric": primary,
            "profiler_name": "source-read", "abstained_from": [t for (t, _w) in tried],
            "rationale": "no runnable method; source read only"}


def _load(p):
    return json.load(open(p)) if p and os.path.exists(p) else {}


def get_decision(a):
    manifest = _load(a.backend_manifest)
    registry = _load(a.registry or DEFAULT_REGISTRY)
    _name, native_tool = native_tool_for(manifest, registry)
    host = probe_host(json.loads(a.host_probe) if a.host_probe else None, extra_tools=(native_tool,))
    fresh = resolve(manifest, a.task, a.size, host, registry)
    if a.cache:
        cache = _load(a.cache)
        if fresh["cache_key"] in cache:
            if a.trajectory:
                open(a.trajectory, "a").write(json.dumps({"event": "profiling_decision", "cache": "hit", "decision": cache[fresh["cache_key"]]}) + "\n")
            return cache[fresh["cache_key"]], True
        cache[fresh["cache_key"]] = fresh
        json.dump(cache, open(a.cache, "w"), indent=2)
    if a.trajectory:
        open(a.trajectory, "a").write(json.dumps({"event": "profiling_decision", "cache": "miss", "decision": fresh}) + "\n")
    return fresh, False


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("resolve", "run"):
        p = sub.add_parser(name)
        p.add_argument("--backend-manifest", required=True)
        p.add_argument("--registry")
        p.add_argument("--task", default="default")
        p.add_argument("--size", default="large", choices=["tiny", "small", "large"])
        p.add_argument("--host-probe")
        p.add_argument("--cache")
        p.add_argument("--trajectory")
        if name == "run":
            p.add_argument("--baseline"); p.add_argument("--candidate")
            p.add_argument("--peak-gflops", type=float, default=5200)
            p.add_argument("--peak-gbs", type=float, default=200)
            p.add_argument("--strategy-id", default="candidate_v1")
            p.add_argument("--correct", default="pass", choices=["pass", "fail"])
    a = ap.parse_args()

    decision, hit = get_decision(a)
    decision = {**decision, "_cache": "hit" if hit else "miss"}

    if a.cmd == "resolve":
        print(json.dumps(decision, indent=2)); return

    if decision["method"] == "perf_heuristic":
        if not a.baseline:
            sys.exit("perf_heuristic chosen but no --baseline given")
        cmd = [sys.executable, os.path.join(HERE, "perf_to_evidence.py"),
               "--baseline", a.baseline, "--peak-gflops", str(a.peak_gflops),
               "--peak-gbs", str(a.peak_gbs), "--strategy-id", a.strategy_id, "--correct", a.correct]
        if a.candidate:
            cmd += ["--candidate", a.candidate]
        out = subprocess.run(cmd, capture_output=True, text=True)
        sys.stderr.write(json.dumps({"routed_to": "perf_to_evidence.py", "decision": decision}, indent=2) + "\n")
        sys.stdout.write(out.stdout)
        if out.returncode:
            sys.stderr.write(out.stderr); sys.exit(out.returncode)
    elif decision["method"] == "native_profiler" or decision.get("method_kind") == "native_profiler":
        print(json.dumps({"action": "would_invoke", "profile_sh": decision.get("profile_invoke"),
                          "request_metrics": decision["requested_metrics"], "decision": decision}, indent=2))
    elif decision["method"] == "derive_adapter":
        print(json.dumps({"action": "agent_derive_adapter", "directive": decision["autonomy_directive"],
                          "decision": decision}, indent=2))
    else:
        print(json.dumps({"action": "static_analysis_stub", "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
