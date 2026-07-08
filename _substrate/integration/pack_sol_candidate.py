#!/usr/bin/env python3
"""Deterministically wrap a candidate kernel into a sol-execbench solution.json.

No LLM, no network. Schema mirrors CutlassGEMM/solution_example.json (verified).
If the kernel already carries a PYBIND11_MODULE it is packed as a single source;
otherwise the packer exits non-zero with a clear error (bare kernels cannot be
run by sol-execbench — a bound entry point is required).
"""
import argparse
import json
import os
import sys


def parse_contract(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()  # strip trailing comments (contract.env has them)
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def build_solution(kernel_src_text, kernel_filename, contract):
    task = contract.get("task_name") or contract.get("op") or "candidate"
    has_binding = "PYBIND11_MODULE" in kernel_src_text
    sources = [{"path": kernel_filename, "content": kernel_src_text}]
    if has_binding:
        entry_point = f"{kernel_filename}::run"
    else:
        # A bare kernel with no PYBIND11_MODULE cannot be packaged into a runnable
        # sol-execbench solution: there is no bound entry point. Fail loudly rather
        # than emit a solution.json whose entry_point resolves to nothing.
        raise SystemExit(
            "pack_sol_candidate: kernel has no PYBIND11_MODULE binding; "
            "sol-execbench needs a bound run() entry point. The proposal must emit "
            "a kernel + torch binding (see SOL_SOLUTION_CONTRACT), not a bare kernel."
        )
    return {
        "name": f"{task}_candidate",
        "definition": task,
        "author": "kersor-workflow",
        "spec": {
            "languages": ["cuda"],
            "target_hardware": ["LOCAL"],
            "entry_point": entry_point,
            "dependencies": [],
            "destination_passing_style": False,
            "binding": "torch",
            "compile_options": {
                "cflags": ["-std=c++17", "-O3"],
                "cuda_cflags": ["-std=c++17", "-O3", "--use_fast_math"],
                "ld_flags": [],
            },
        },
        "sources": sources,
        "description": f"sol-execbench candidate for {task} (backend={contract.get('backend', 'cuda')})",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.kernel) as fh:
        kernel_text = fh.read()
    contract = parse_contract(args.contract)
    sol = build_solution(kernel_text, os.path.basename(args.kernel), contract)
    with open(args.out, "w") as fh:
        json.dump(sol, fh, indent=2)
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
