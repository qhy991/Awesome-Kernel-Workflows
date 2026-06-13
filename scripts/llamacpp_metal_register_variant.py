#!/usr/bin/env python3
"""llamacpp_metal_register_variant.py

Register a new Metal kernel variant into llama.cpp's ggml-metal dispatch system
so AKW fan-out workflows can compile/test/bench many candidates in parallel.

What a "variant" is for Metal:
  - a new kernel function added to ggml/src/ggml-metal/ggml-metal.metal (or separate .metal)
  - a small ENV-gated edit to ggml/src/ggml-metal/ggml-metal-ops.cpp dispatch

Supported kernel families:
  - mul_matvec   quantized GEMM (kernel_mul_mv_q4_0/q4_1/...), bottleneck #1
  - flash_attn   flash attention (kernel_flash_attn_ext_blk/pad), bottleneck #2

What gates which variant the binary runs:
  An ENV var KERSOR_VARIANT=<name>. If set and the variant is registered, the
  Metal dispatch routes to that variant; otherwise dispatch falls through to the
  normal best-kernel logic. Registering a variant does NOT change default
  behavior -- registered-inactive code is safe to leave in place.

Every insertion is wrapped in marker comments so unregister is byte-exact.

Usage:
  register   --variant v3 --source /path/to/variant.metal
             --project-root /path/to/llama.cpp/ggml --kernel-family mul_matvec
  unregister --variant v3 --project-root /path/to/llama.cpp/ggml
  list       --project-root /path/to/llama.cpp/ggml

Generic adapter flags (see ADAPTER_CONTRACT.md):
  --source / --project-root are the canonical flag names.
  --metal-src / --ggml-root are backward-compatible aliases.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER_BEGIN = "// >>> KERSOR_VARIANT_BEGIN {name} <<<"
MARKER_END   = "// <<< KERSOR_VARIANT_END {name} >>>"

# ==============================================================================
# Anchors in ggml-metal-ops.cpp (dispatch file)
# ==============================================================================
# The dispatch file is ggml/src/ggml-metal/ggml-metal-ops.cpp.
# Each kernel family has its own encode function. We insert ENV-gated dispatch
# at the top of each function, before the normal pipeline selection logic.

# --- mul_matvec anchors ---
# Anchor: function signature of ggml_metal_op_mul_mat() -- we insert the ENV
# gate right after the opening brace and local variable declarations.
ANCHOR_MUL_MAT_FN = "int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {"

# Anchor: inside the final else block that dispatches to kernel_mul_mv_q4_0_f32.
# This is the path for ne11=1 (batch-1 decode) — the primary bottleneck.
# We insert the ENV gate right after the pipeline is retrieved but before encoding.
ANCHOR_MUL_MAT_GATE = (
    'auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);'
)

# Second anchor: for variants that replace the ext-mv path (small-batch mat-mv).
ANCHOR_MUL_MAT_EXT_GATE = (
    "// find the break-even point where the matrix-matrix kernel becomes more efficient compared"
)

# --- flash_attn anchors ---
ANCHOR_FLASH_ATTN_FN = "int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {"

# Anchor: right before the "if (ggml_metal_op_flash_attn_ext_use_vec(op))" check
ANCHOR_FLASH_ATTN_GATE = (
    "if (ggml_metal_op_flash_attn_ext_use_vec(op))"
)

# ==============================================================================
# Anchors in ggml-metal.metal (kernel shader file)
# ==============================================================================
# The kernel file is ggml/src/ggml-metal/ggml-metal.metal.
# We append variant kernel functions at the end of the file (before any trailing
# newlines), wrapped in marker comments.

# ==============================================================================
# Helper functions
# ==============================================================================

def _ggml_metal_dir(ggml_root: Path) -> Path:
    """Resolve the ggml-metal source directory (handles both old and new layouts)."""
    # New layout: ggml/src/ggml-metal/
    new_path = ggml_root / "src" / "ggml-metal"
    if new_path.is_dir():
        return new_path
    # Old layout fallback: ggml-metal/ directly in repo root
    old_path = ggml_root / "ggml-metal"
    if old_path.is_dir():
        return old_path
    return new_path  # Return the expected path; let the caller error


def _dispatch_path(ggml_root: Path) -> Path:
    """Return path to the dispatch .cpp file (modern layout)."""
    return _ggml_metal_dir(ggml_root) / "ggml-metal-ops.cpp"


def _kernel_metal_path(ggml_root: Path) -> Path:
    """Return path to the monolithic kernel .metal file."""
    return _ggml_metal_dir(ggml_root) / "ggml-metal.metal"


def _variant_metal_path(ggml_root: Path, name: str, kernel_family: str) -> Path:
    """Return path for the variant .metal file in the ggml-metal directory."""
    prefix = "fattn" if kernel_family == "flash_attn" else "mulmv"
    return _ggml_metal_dir(ggml_root) / f"{prefix}-{name}.metal"


def _existing_variants(text: str) -> set:
    """Return set of currently-registered variant names in the dispatch file."""
    return set(re.findall(r"KERSOR_VARIANT_BEGIN ([A-Za-z0-9_-]+) <<<", text))


def _get_anchors(kernel_family: str) -> dict:
    """Return the anchor strings for the given kernel family."""
    if kernel_family == "mul_matvec":
        return {
            "fn_anchor": ANCHOR_MUL_MAT_FN,
            "gate_anchor": ANCHOR_MUL_MAT_GATE,
            "gate_anchor_ext": ANCHOR_MUL_MAT_EXT_GATE,
            "dispatch_fn": "ggml_metal_op_mul_mat",
            "kernel_prefix": "kernel_mul_mv",
            "pipeline_func": "ggml_metal_library_get_pipeline_mul_mv_ext",
        }
    elif kernel_family == "flash_attn":
        return {
            "fn_anchor": ANCHOR_FLASH_ATTN_FN,
            "gate_anchor": ANCHOR_FLASH_ATTN_GATE,
            "dispatch_fn": "ggml_metal_op_flash_attn_ext",
            "kernel_prefix": "kernel_flash_attn",
            "pipeline_func": "ggml_metal_library_get_pipeline_flash_attn_ext",
        }
    else:
        sys.exit(f"ERROR: unknown kernel_family '{kernel_family}'. Use 'mul_matvec' or 'flash_attn'.")


# ==============================================================================
# Commands
# ==============================================================================

def cmd_register(args):
    ggml_root = Path(args.project_root).resolve()
    metal_src = Path(args.source).resolve()
    name = args.variant
    kernel_family = args.kernel_family

    if not re.fullmatch(r"[a-z0-9_]{1,32}", name):
        sys.exit(f"ERROR: variant name '{name}' must match [a-z0-9_]{{1,32}}")
    if not metal_src.exists():
        sys.exit(f"ERROR: source not found: {metal_src}")

    anchors = _get_anchors(kernel_family)

    dispatch_file = _dispatch_path(ggml_root)
    kernel_file = _kernel_metal_path(ggml_root)

    if not dispatch_file.exists():
        sys.exit(f"ERROR: dispatch file not found at {dispatch_file}")
    if not kernel_file.exists():
        sys.exit(f"ERROR: kernel file not found at {kernel_file}")

    dispatch_text = dispatch_file.read_text(encoding="utf-8")

    if name in _existing_variants(dispatch_text):
        sys.exit(f"ERROR: variant '{name}' is already registered. Run unregister first.")

    # ---- 1. Copy the variant .metal file into ggml-metal directory ----
    dst_metal = _variant_metal_path(ggml_root, name, kernel_family)
    dst_metal.write_text(metal_src.read_text(encoding="utf-8"), encoding="utf-8")

    # ---- 2. Insert ENV-gated dispatch in ggml-metal-ops.cpp ----
    # We insert the gate BEFORE the normal pipeline selection logic.
    # The gate checks KERSOR_VARIANT env var and, if matching, loads the variant
    # pipeline from ggml-metal.metal and dispatches to it directly.
    #
    # For mul_matvec: the variant kernel is a complete replacement for the
    #   kernel_mul_mv_q*_f32 family. The gate compiles a pipeline from the
    #   variant kernel name and dispatches with the same args struct.
    #
    # For flash_attn: the variant kernel replaces kernel_flash_attn_ext_*.

    # Gate for the final else block (batch=1 path). Inserted after pipeline is retrieved,
    # the gate overrides the pipeline when KERSOR_VARIANT is set.
    # The generic compile_pipeline returns nr0/nsg=0, so we must set these explicitly
    # to match what the original pipeline would have had.
    gate_block = (
        MARKER_BEGIN.format(name=name) + "\n"
        f"        const char * kersor_variant = getenv(\"KERSOR_VARIANT\");\n"
        f"        if (kersor_variant && strcmp(kersor_variant, \"{name}\") == 0) {{\n"
        f"            pipeline = ggml_metal_library_compile_pipeline(lib, \"kernel_{name}\", \"kernel_{name}\", ggml_metal_cv_init());\n"
        f"            // compile_pipeline does not set nr0/nsg — set them explicitly\n"
        f"            pipeline.nr0 = 4;\n"
        f"            pipeline.nsg = 2;\n"
        f"            pipeline.nr1 = 4;\n"
        f"            pipeline.smem = 0;\n"
        f"        }}\n"
        + MARKER_END.format(name=name) + "\n"
    )

    # Insert the gate block right before the gate_anchor line
    if anchors["gate_anchor"] not in dispatch_text:
        # Try the ext-mv gate anchor as fallback
        if anchors.get("gate_anchor_ext") and anchors["gate_anchor_ext"] in dispatch_text:
            anchors["gate_anchor"] = anchors["gate_anchor_ext"]
        else:
            sys.exit(
                f"ERROR: gate anchor not found in {dispatch_file}.\n"
                f"Expected: '{anchors['gate_anchor'][:80]}...'\n"
                f"The llama.cpp version may be incompatible with this adapter."
            )

    dispatch_text = dispatch_text.replace(
        anchors["gate_anchor"],
        anchors["gate_anchor"] + "\n" + gate_block,
        1,
    )

    dispatch_file.write_text(dispatch_text, encoding="utf-8")

    # ---- 3. Append variant kernel function to ggml-metal.metal ----
    # Read the variant .metal content and extract just the kernel function(s).
    # The variant file should contain complete kernel function definitions.
    variant_source = metal_src.read_text(encoding="utf-8")

    kernel_text = kernel_file.read_text(encoding="utf-8")

    # Append the variant kernel at the end of ggml-metal.metal with markers
    kernel_append = (
        "\n"
        + MARKER_BEGIN.format(name=name) + "\n"
        + "// KERSOR_VARIANT: " + name + " (" + kernel_family + ")\n"
        + "// Source: " + str(metal_src) + "\n"
        + variant_source.rstrip("\n") + "\n"
        + MARKER_END.format(name=name) + "\n"
    )

    kernel_file.write_text(kernel_text.rstrip("\n") + kernel_append, encoding="utf-8")

    print(f"registered variant '{name}' ({kernel_family})")
    print(f"  metal:       {dst_metal}")
    print(f"  kernel file: {kernel_file} (+ variant appended)")
    print(f"  dispatch:    {dispatch_file} (ENV gate inserted)")
    print(f"  ENV:         KERSOR_VARIANT={name}")


def cmd_unregister(args):
    ggml_root = Path(args.project_root).resolve()
    name = args.variant

    dispatch_file = _dispatch_path(ggml_root)
    kernel_file = _kernel_metal_path(ggml_root)

    any_change = False

    # ---- 1. Remove ENV gate from dispatch file ----
    if dispatch_file.exists():
        text = dispatch_file.read_text(encoding="utf-8")
        if name in _existing_variants(text):
            pattern = re.compile(
                r"[ \t]*// >>> KERSOR_VARIANT_BEGIN " + re.escape(name) + r" <<<\n"
                r".*?"
                r"[ \t]*// <<< KERSOR_VARIANT_END " + re.escape(name) + r" >>>\n",
                re.DOTALL,
            )
            text = pattern.sub("", text)
            dispatch_file.write_text(text, encoding="utf-8")
            print(f"removed dispatch entries for '{name}' from {dispatch_file}")
            any_change = True
        else:
            print(f"variant '{name}' not found in dispatch (nothing to remove)")
    else:
        print(f"dispatch file not found at {dispatch_file}")

    # ---- 2. Remove variant kernel from ggml-metal.metal ----
    if kernel_file.exists():
        text = kernel_file.read_text(encoding="utf-8")
        pattern = re.compile(
            r"\n[ \t]*// >>> KERSOR_VARIANT_BEGIN " + re.escape(name) + r" <<<\n"
            r".*?"
            r"[ \t]*// <<< KERSOR_VARIANT_END " + re.escape(name) + r" >>>\n",
            re.DOTALL,
        )
        if pattern.search(text):
            text = pattern.sub("", text)
            kernel_file.write_text(text, encoding="utf-8")
            print(f"removed kernel variant '{name}' from {kernel_file}")
            any_change = True
        else:
            print(f"variant '{name}' not found in kernel file (nothing to remove)")

    # ---- 3. Remove variant .metal file from ggml-metal directory ----
    for kf in ("mul_matvec", "flash_attn"):
        metal_file = _variant_metal_path(ggml_root, name, kf)
        if metal_file.exists():
            metal_file.unlink()
            print(f"deleted {metal_file}")

    if not any_change:
        print(f"variant '{name}' was not registered (nothing to do)")


def cmd_list(args):
    ggml_root = Path(args.project_root).resolve()
    dispatch_file = _dispatch_path(ggml_root)

    if not dispatch_file.exists():
        print(f"(dispatch file not found at {dispatch_file})")
        return

    text = dispatch_file.read_text(encoding="utf-8")
    names = sorted(_existing_variants(text))

    if not names:
        print("(no variants registered)")
        return

    for n in names:
        # Determine kernel family from context
        before = text.split(f"KERSOR_VARIANT_BEGIN {n} <<<")[0]
        family = "unknown"
        if "ggml_metal_op_mul_mat" in before.split("KERSOR_VARIANT_BEGIN")[-1]:
            family = "mul_matvec"
        elif "ggml_metal_op_flash_attn_ext" in before.split("KERSOR_VARIANT_BEGIN")[-1]:
            family = "flash_attn"
        print(f"  {n:24s}  family={family}  ENV=KERSOR_VARIANT={n}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register")
    r.add_argument("--variant", required=True)
    r.add_argument("--source", "--metal-src", dest="source", required=True,
                   help="path to the new variant .metal file (alias: --metal-src)")
    r.add_argument("--project-root", "--ggml-root", dest="project_root", required=True,
                   help="path to llama.cpp/ggml project root (alias: --ggml-root)")
    r.add_argument("--kernel-family", dest="kernel_family", required=True,
                   choices=["mul_matvec", "flash_attn"],
                   help="which kernel family to target: mul_matvec (quantized GEMM) or flash_attn")
    r.set_defaults(func=cmd_register)

    u = sub.add_parser("unregister")
    u.add_argument("--variant", required=True)
    u.add_argument("--project-root", "--ggml-root", dest="project_root", required=True,
                   help="path to llama.cpp/ggml project root (alias: --ggml-root)")
    u.set_defaults(func=cmd_unregister)

    l = sub.add_parser("list")
    l.add_argument("--project-root", "--ggml-root", dest="project_root", required=True,
                   help="path to llama.cpp/ggml project root (alias: --ggml-root)")
    l.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()