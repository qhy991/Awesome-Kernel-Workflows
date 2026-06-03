#!/usr/bin/env python3
"""Layer E — deterministic method gate (KerSor Solver SDK).

Borrowed from KernelSkill: normalized evidence -> decision table -> allowed_methods.
The LLM/controller may only pick optimization methods WITHIN the gated set; this
keeps "evidence-guided" deterministic instead of LLM discretion. Stdlib only.

Usage:
  method_gate.py --class memory_bound
  method_gate.py --class compute_bound --metrics metrics.json
Prints: {bottleneck_class, allowed_methods, rationale}
"""
import sys, json, argparse

TABLE = {
    "memory_bound": ["vectorized_load_store", "memory_coalescing", "async_copy_pipeline",
                     "shared_memory_tiling", "l2_cache_reuse"],
    "compute_bound": ["tensor_core_mma", "instruction_reduction", "fast_math_intrinsics",
                      "register_tiling", "occupancy_increase"],
    "latency_occupancy": ["occupancy_increase", "block_size_tuning",
                          "register_pressure_reduction", "launch_config_tuning"],
    "overhead_bound": ["kernel_fusion", "launch_overhead_reduction",
                       "library_fallback_hybrid", "cpu_gpu_overlap"],
    "unknown": ["profile_first", "conservative_tiling"],
}


def gate(bclass, metrics=None):
    allowed = list(TABLE.get(bclass, TABLE["unknown"]))
    rationale = f"decision table for bottleneck_class={bclass}"
    # refinement: if occupancy is already high, drop occupancy_increase as redundant
    if metrics and metrics.get("occupancy") is not None and metrics["occupancy"] >= 0.8:
        allowed = [m for m in allowed if m != "occupancy_increase"]
        rationale += "; dropped occupancy_increase (occupancy already >= 0.80)"
    return {"bottleneck_class": bclass, "allowed_methods": allowed, "rationale": rationale}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="bclass", required=True)
    ap.add_argument("--metrics")
    a = ap.parse_args()
    metrics = None
    if a.metrics:
        metrics = json.loads(sys.stdin.read() if a.metrics == "-" else open(a.metrics).read())
    print(json.dumps(gate(a.bclass, metrics), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
