#!/usr/bin/env python3
"""Deterministically wrap a candidate into a sol-execbench solution.json.

No LLM, no network.  The source representation owns the transport contract:
Python/Triton candidates expose a module-level ``run`` function, while CUDA C++
candidates expose ``run`` through ``PYBIND11_MODULE``.  A failed pack removes a
stale output so a later stage cannot accidentally evaluate an older candidate.
"""
import argparse
import ast
import json
import os
import sys


def parse_contract(path):
    d = {}
    # Task-directory dispatches have an authoritative definition.json but no
    # synthesized contract.env. The contract only contributes descriptive
    # solution metadata, so its absence must not block candidate packaging.
    if not path or not os.path.isfile(path):
        return d
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()  # strip trailing comments (contract.env has them)
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def normalize_cuda_filename(kernel_filename):
    """Return a source filename accepted by the cuda_cpp solution schema.

    Some generator workflows stage structured-output code in a generic
    ``kernel.py`` path even when the content is CUDA C++.  The solution language
    is authoritative here, so carrying that staging suffix into solution.json
    makes sol-execbench reject the package before compilation.
    """
    stem, suffix = os.path.splitext(kernel_filename)
    if suffix.lower() in {".cu", ".cc", ".cpp", ".cxx"}:
        return kernel_filename
    return f"{stem or 'kernel'}.cu"


def python_language(kernel_src_text):
    """Return the sol-execbench Python language, or fail on no public run()."""
    try:
        tree = ast.parse(kernel_src_text)
    except SyntaxError as exc:
        raise SystemExit(
            f"pack_sol_candidate: Python candidate is not valid syntax: {exc}"
        ) from exc
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run"
        for node in tree.body
    ):
        raise SystemExit(
            "pack_sol_candidate: Python candidate has no module-level run() entry point"
        )
    uses_triton = any(
        (isinstance(node, ast.Import) and any(alias.name == "triton" or alias.name.startswith("triton.") for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module and (node.module == "triton" or node.module.startswith("triton.")))
        for node in tree.body
    )
    return "triton" if uses_triton else "pytorch"


def build_solution(kernel_src_text, kernel_filename, contract):
    task = contract.get("task_name") or contract.get("op") or "candidate"
    has_binding = "PYBIND11_MODULE" in kernel_src_text
    if has_binding:
        kernel_filename = normalize_cuda_filename(kernel_filename)
        sources = [{"path": kernel_filename, "content": kernel_src_text}]
        entry_point = f"{kernel_filename}::run"
        spec = {
            # sol-execbench validates this against its public SolutionSpec enum.
            # "cuda" is a backend name, not a supported source-language value.
            "languages": ["cuda_cpp"],
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
        }
    elif os.path.splitext(kernel_filename)[1].lower() == ".py":
        language = python_language(kernel_src_text)
        sources = [{"path": kernel_filename, "content": kernel_src_text}]
        spec = {
            "languages": [language],
            "target_hardware": ["LOCAL"],
            "entry_point": f"{kernel_filename}::run",
            "dependencies": [],
            "destination_passing_style": False,
        }
    else:
        raise SystemExit(
            "pack_sol_candidate: CUDA/C++ candidate has no PYBIND11_MODULE binding; "
            "Python/Triton candidates must use a .py source with module-level run()."
        )
    return {
        "name": f"{task}_candidate",
        "definition": task,
        "author": "kersor-workflow",
        "spec": spec,
        "sources": sources,
        "description": f"sol-execbench candidate for {task} (backend={contract.get('backend', 'cuda')})",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        with open(args.kernel) as fh:
            kernel_text = fh.read()
        contract = parse_contract(args.contract)
        sol = build_solution(kernel_text, os.path.basename(args.kernel), contract)
    except BaseException:
        try:
            os.unlink(args.out)
        except FileNotFoundError:
            pass
        raise
    tmp = f"{args.out}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            json.dump(sol, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, args.out)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
