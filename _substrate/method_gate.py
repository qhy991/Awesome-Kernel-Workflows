#!/usr/bin/env python3
"""Layer E — deterministic method gate (KerSor Solver SDK).

Borrowed from KernelSkill: normalized evidence -> decision table -> allowed_methods.
The LLM/controller may only pick optimization methods WITHIN the gated set; this
keeps "evidence-guided" deterministic instead of LLM discretion. Stdlib only.

Usage:
  method_gate.py --class memory_bound
  method_gate.py --class compute_bound --metrics metrics.json
  method_gate.py --class unknown --op-description "confirm baseline noop"
Prints: {bottleneck_class, allowed_methods, rationale}
"""
import sys, json, argparse, re

TABLE = {
    "memory_bound": ["vectorized_load_store", "memory_coalescing", "async_copy_pipeline",
                     "shared_memory_tiling", "l2_cache_reuse"],
    "compute_bound": ["tensor_core_mma", "instruction_reduction", "fast_math_intrinsics",
                      "register_tiling", "occupancy_increase"],
    "latency_occupancy": ["occupancy_increase", "block_size_tuning",
                          "register_pressure_reduction", "launch_config_tuning"],
    "overhead_bound": ["kernel_fusion", "launch_overhead_reduction",
                       "library_fallback_hybrid", "cpu_gpu_overlap"],
    "unknown": ["profile_first", "baseline_confirm", "noop_validate", "conservative_tiling"],
}

_CONFIRM_INTENT = re.compile(
    r"\b(confirm|validation|validate|noop|baseline[\s_-]?only|smoke|"
    r"do[\s_-]?not[\s_-]?optimi|no[\s_-]?optimi|re[\s_-]?verify|harness[\s_-]?confirm)\b",
    re.I)


def detect_confirm_intent(op_description=None):
    if not op_description:
        return False
    return bool(_CONFIRM_INTENT.search(str(op_description)))


def gate(bclass, metrics=None, op_description=None):
    allowed = list(TABLE.get(bclass, TABLE["unknown"]))
    rationale = f"decision table for bottleneck_class={bclass}"
    # refinement: if occupancy is already high, drop occupancy_increase as redundant
    if metrics and metrics.get("occupancy") is not None and metrics["occupancy"] >= 0.8:
        allowed = [m for m in allowed if m != "occupancy_increase"]
        rationale += "; dropped occupancy_increase (occupancy already >= 0.80)"
    if bclass == "unknown" and detect_confirm_intent(op_description):
        priority = ["baseline_confirm", "noop_validate"]
        allowed = priority + [m for m in allowed if m not in priority]
        rationale += "; confirm/noop intent in op_description — prioritized baseline_confirm, noop_validate"
    return {"bottleneck_class": bclass, "allowed_methods": allowed, "rationale": rationale}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--class", dest="bclass", required=True)
    ap.add_argument("--metrics")
    ap.add_argument("--op-description")
    a = ap.parse_args()
    metrics = None
    if a.metrics:
        metrics = json.loads(sys.stdin.read() if a.metrics == "-" else open(a.metrics).read())
    print(json.dumps(gate(a.bclass, metrics, a.op_description), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
