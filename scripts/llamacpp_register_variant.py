#!/usr/bin/env python3
"""llamacpp_register_variant.py

Register a new fattn kernel variant into llama.cpp's ggml-cuda dispatch system
so AKW fan-out workflows can compile/test/bench many candidates in parallel.

What a "variant" is:
  - a new .cuh file at  ggml/src/ggml-cuda/<basename>.cuh
  - a new template-instances .cu that instantiates DECL_FATTN_RDNA_APU_CASE
  - 5 small edits to ggml/src/ggml-cuda/fattn.cu  (include, enum, ENV gate,
    alloc-size case, dispatch case)

What gates which variant the binary runs:
  An ENV var KERSOR_VARIANT=<name>. If set and the variant is registered, the
  selector returns that variant; otherwise dispatch falls through to the
  normal best-kernel logic. Registering a variant does NOT change default
  behavior - line code is safe to leave registered.

Every insertion is wrapped in marker comments so `unregister` is byte-exact.

Usage:
  register   --variant v3 --cuh-src /path/to/your_variant.cuh \
             --ggml-root /path/to/llama.cpp/ggml [--dkq 256 --dv 256]
  unregister --variant v3 --ggml-root /path/to/llama.cpp/ggml
  list       --ggml-root /path/to/llama.cpp/ggml
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

MARKER_BEGIN = "// >>> KERSOR_VARIANT_BEGIN {name} <<<"
MARKER_END   = "// <<< KERSOR_VARIANT_END {name} >>>"
ENV_GATE_BEGIN = "    // >>> KERSOR_VARIANT_GATE_BEGIN <<<"
ENV_GATE_END   = "    // <<< KERSOR_VARIANT_GATE_END >>>"

# Anchors we insert relative to. These are stable lines in fattn.cu.
ANCHOR_INCLUDE_TAIL  = '#include "fattn.cuh"'
ANCHOR_ENUM_LASTLINE = "BEST_FATTN_KERNEL_RDNA_APU = 500,"
ANCHOR_SELECTOR_TOP  = (
    "static best_fattn_kernel ggml_cuda_get_best_fattn_kernel"
    "(const int device, const ggml_tensor * dst) {"
)
# We place the ENV gate immediately BEFORE the original RDNA_APU return so the
# variant inherits ALL of the upstream selector's conditions (RDNA3.5 cc, head
# dims, GQA ratio, non-quantized K/V, etc.). Anything broader would fire the
# variant on cases the original kernel doesn't support.
ANCHOR_RDNA_APU_RETURN = "        return BEST_FATTN_KERNEL_RDNA_APU;"
ANCHOR_ALLOC_CASE    = "case BEST_FATTN_KERNEL_RDNA_APU:\n            need_f16_K = K->type == GGML_TYPE_F32;"
ANCHOR_DISPATCH_CASE = (
    "case BEST_FATTN_KERNEL_RDNA_APU:\n"
    "            ggml_cuda_flash_attn_ext_rdna_apu(ctx, dst);\n"
    "            break;"
)

ENUM_BASE = 500  # RDNA_APU base; variants get 501, 502, ...


def variant_paths(ggml_root: Path, name: str):
    cuda_dir = ggml_root / "src" / "ggml-cuda"
    cuh = cuda_dir / f"fattn-rdna-apu-{name}.cuh"
    entry = cuda_dir / f"fattn-rdna-apu-{name}.cu"
    inst = cuda_dir / "template-instances" / f"fattn-rdna-apu-{name}-instance-dkq256-dv256.cu"
    fattn_cu = cuda_dir / "fattn.cu"
    return cuh, entry, inst, fattn_cu


def existing_variants(fattn_cu_text: str):
    """Return set of currently-registered variant names in fattn.cu."""
    return set(re.findall(r"KERSOR_VARIANT_BEGIN ([A-Za-z0-9_]+) <<<", fattn_cu_text))


def _rename_symbols(text: str, name: str, upper: str) -> str:
    """Suffix every identifier containing rdna_apu / RDNA_APU with the variant
    name to avoid ODR/redefinition clashes. fattn.cu transitively includes both
    the original fattn-rdna-apu.cuh and the variant .cuh in the same TU, so even
    `static` helpers must have distinct names. Generic so we don't have to
    enumerate every symbol the kernel author chose."""
    # Match the largest identifier that contains the token, then splice the
    # suffix in right after the token.
    text = re.sub(
        r"\b([A-Za-z0-9_]*)rdna_apu([A-Za-z0-9_]*)\b",
        lambda m: f"{m.group(1)}rdna_apu_{name}{m.group(2)}",
        text,
    )
    text = re.sub(
        r"\b([A-Za-z0-9_]*)RDNA_APU([A-Za-z0-9_]*)\b",
        lambda m: f"{m.group(1)}RDNA_APU_{upper}{m.group(2)}",
        text,
    )
    return text


def next_enum_id(registered_names, name_to_id):
    used = set(name_to_id.values())
    cand = ENUM_BASE + 1
    while cand in used:
        cand += 1
    return cand


def cmd_register(args):
    ggml_root = Path(args.ggml_root).resolve()
    cuh_src = Path(args.cuh_src).resolve()
    name = args.variant

    if not re.fullmatch(r"[a-z0-9_]{1,32}", name):
        sys.exit(f"ERROR: variant name '{name}' must match [a-z0-9_]{{1,32}}")
    if not cuh_src.exists():
        sys.exit(f"ERROR: cuh source not found: {cuh_src}")

    cuh_dst, entry_dst, inst_dst, fattn_cu = variant_paths(ggml_root, name)
    if not fattn_cu.exists():
        sys.exit(f"ERROR: fattn.cu not found at {fattn_cu}")

    text = fattn_cu.read_text(encoding="utf-8")

    if name in existing_variants(text):
        sys.exit(f"ERROR: variant '{name}' is already registered. Run unregister first.")

    # Allocate a fresh enum id (look at any already-inserted enum lines)
    name_to_id = {
        m.group(1): int(m.group(2))
        for m in re.finditer(
            r"BEST_FATTN_KERNEL_RDNA_APU_([A-Z0-9_]+)\s*=\s*(\d+)", text
        )
    }
    enum_id = next_enum_id(existing_variants(text), name_to_id)
    upper = name.upper()

    # ---- 1. write the .cuh ----
    # The caller-provided source uses the canonical RDNA_APU symbol names
    # (matching fattn-rdna-apu.cuh). We rewrite them to <name>-suffixed symbols
    # so multiple variants can coexist without link-time duplicate definitions.
    cuh_text = cuh_src.read_text(encoding="utf-8")
    # Defensive: if the candidate already pre-suffixed any rdna_apu symbol with
    # the variant name, the rename would double-suffix (e.g. rdna_apu_r1v1_r1v1).
    # That's a contract violation by the proposer.
    if re.search(r"\brdna_apu_" + re.escape(name) + r"\b", cuh_text) or \
       re.search(r"\bRDNA_APU_" + re.escape(upper) + r"\b", cuh_text):
        sys.exit(
            f"ERROR: candidate .cuh already contains pre-suffixed symbols "
            f"(rdna_apu_{name}/RDNA_APU_{upper}). The register script owns the "
            f"suffixing - candidate must use canonical rdna_apu/RDNA_APU names."
        )
    cuh_text = _rename_symbols(cuh_text, name, upper)
    cuh_dst.write_text(cuh_text, encoding="utf-8")

    # ---- 1b. write the entry .cu (small dispatch stub that calls the templated case) ----
    # The .cuh declares ggml_cuda_flash_attn_ext_rdna_apu_<name>() and defines the
    # _case<DKQ,DV> template; this stub provides the non-templated entry symbol
    # that fattn.cu's dispatch switch calls. Mirrors fattn-rdna-apu.cu (12 lines).
    entry_dst.write_text(
        f'#include "common.cuh"\n'
        f'#include "fattn-rdna-apu-{name}.cuh"\n'
        f'\n'
        f'void ggml_cuda_flash_attn_ext_rdna_apu_{name}'
        f'(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {{\n'
        f'    const ggml_tensor * K = dst->src[1];\n'
        f'    const ggml_tensor * V = dst->src[2];\n'
        f'    GGML_ASSERT(K->ne[0] == {args.dkq});\n'
        f'    GGML_ASSERT(V->ne[0] == {args.dv});\n'
        f'    ggml_cuda_flash_attn_ext_rdna_apu_{name}_case<{args.dkq}, {args.dv}>(ctx, dst);\n'
        f'}}\n',
        encoding="utf-8",
    )

    # ---- 2. write the template instance .cu ----
    inst_dst.parent.mkdir(parents=True, exist_ok=True)
    inst_dst.write_text(
        f'#include "../fattn-rdna-apu-{name}.cuh"\n'
        f"DECL_FATTN_RDNA_APU_{upper}_CASE({args.dkq}, {args.dv});\n",
        encoding="utf-8",
    )

    # ---- 3. insert include ----
    include_marker = MARKER_BEGIN.format(name=name) + "\n" \
        f'#include "fattn-rdna-apu-{name}.cuh"\n' \
        + MARKER_END.format(name=name) + "\n"
    text = text.replace(
        ANCHOR_INCLUDE_TAIL + "\n",
        ANCHOR_INCLUDE_TAIL + "\n" + include_marker,
        1,
    )

    # ---- 4. insert enum entry ----
    enum_line = f"    {MARKER_BEGIN.format(name=name)}\n" \
        f"    BEST_FATTN_KERNEL_RDNA_APU_{upper} = {enum_id},\n" \
        f"    {MARKER_END.format(name=name)}\n"
    text = text.replace(
        ANCHOR_ENUM_LASTLINE + "\n",
        ANCHOR_ENUM_LASTLINE + "\n" + enum_line,
        1,
    )

    # ---- 5. insert / extend ENV gate at top of selector ----
    text = _upsert_env_gate(text, name, upper)

    # ---- 6. alloc-size case (reuse RDNA_APU semantics: f16 K/V if input is f32) ----
    alloc_line = f"        {MARKER_BEGIN.format(name=name)}\n" \
        f"        case BEST_FATTN_KERNEL_RDNA_APU_{upper}:\n" \
        f"        {MARKER_END.format(name=name)}\n"
    text = text.replace(
        "        case BEST_FATTN_KERNEL_RDNA_APU:\n",
        "        case BEST_FATTN_KERNEL_RDNA_APU:\n" + alloc_line,
        1,
    )

    # ---- 7. dispatch case ----
    dispatch_block = (
        f"        {MARKER_BEGIN.format(name=name)}\n"
        f"        case BEST_FATTN_KERNEL_RDNA_APU_{upper}:\n"
        f"            ggml_cuda_flash_attn_ext_rdna_apu_{name}(ctx, dst);\n"
        f"            break;\n"
        f"        {MARKER_END.format(name=name)}\n"
    )
    text = text.replace(
        "        case BEST_FATTN_KERNEL_RDNA_APU:\n"
        "            ggml_cuda_flash_attn_ext_rdna_apu(ctx, dst);\n"
        "            break;\n",
        "        case BEST_FATTN_KERNEL_RDNA_APU:\n"
        "            ggml_cuda_flash_attn_ext_rdna_apu(ctx, dst);\n"
        "            break;\n" + dispatch_block,
        1,
    )

    fattn_cu.write_text(text, encoding="utf-8")
    print(f"registered variant '{name}' (enum={enum_id})")
    print(f"  cuh : {cuh_dst}")
    print(f"  inst: {inst_dst}")
    print(f"  ENV : KERSOR_VARIANT={name}")
    print(f"  expected entry symbol: ggml_cuda_flash_attn_ext_rdna_apu_{name}")

    # CMake file(GLOB) lacks CONFIGURE_DEPENDS, so the new .cu/.cuh files are
    # invisible to the next incremental build until reconfigure runs. Without
    # this the build links against a stale source list and reports
    # "undefined symbol" for the variant entry. Register owns this invariant.
    if args.cmake_build_dir:
        bd = Path(args.cmake_build_dir).resolve()
        if not bd.exists():
            sys.exit(f"ERROR: --cmake-build-dir does not exist: {bd}")
        print(f"  reconfigure: cmake {bd}")
        rc = subprocess.call(["cmake", str(bd)], stdout=subprocess.DEVNULL)
        if rc != 0:
            sys.exit(f"ERROR: cmake reconfigure failed (rc={rc})")


def _upsert_env_gate(text: str, name: str, upper: str) -> str:
    """Insert (or extend) the single ENV-gate block IMMEDIATELY BEFORE the
    original RDNA_APU return. By living inside the same if-block, the variant
    fires only when the original kernel would have fired, so we never route to
    a variant for cases it doesn't support."""
    gate_inner = (
        f"        {MARKER_BEGIN.format(name=name)}\n"
        f"        {{\n"
        f'            const char * kersor_variant_{name} = std::getenv("KERSOR_VARIANT");\n'
        f"            static auto kersor_streq_{name} = [](const char * a, const char * b) {{\n"
        f"                if (!a || !b) return false;\n"
        f"                while (*a && *a == *b) {{ ++a; ++b; }}\n"
        f"                return *a == 0 && *b == 0;\n"
        f"            }};\n"
        f'            if (kersor_streq_{name}(kersor_variant_{name}, "{name}")) '
        f"return BEST_FATTN_KERNEL_RDNA_APU_{upper};\n"
        f"        }}\n"
        f"        {MARKER_END.format(name=name)}\n"
    )

    needle = ANCHOR_RDNA_APU_RETURN + "\n"
    if needle not in text:
        sys.exit("ERROR: could not locate RDNA_APU return anchor in fattn.cu")
    return text.replace(needle, gate_inner + needle, 1)


def cmd_unregister(args):
    ggml_root = Path(args.ggml_root).resolve()
    name = args.variant
    cuh_dst, entry_dst, inst_dst, fattn_cu = variant_paths(ggml_root, name)
    text = fattn_cu.read_text(encoding="utf-8")

    if name not in existing_variants(text):
        print(f"variant '{name}' was not registered (nothing to do)")
    else:
        # Strip every "BEGIN name ... END name" block (including its own lines).
        pattern = re.compile(
            r"[ \t]*// >>> KERSOR_VARIANT_BEGIN " + re.escape(name) + r" <<<\n"
            r".*?"
            r"[ \t]*// <<< KERSOR_VARIANT_END " + re.escape(name) + r" >>>\n",
            re.DOTALL,
        )
        text = pattern.sub("", text)

        # No global scaffold any more (each variant carries its own gate inline).

        fattn_cu.write_text(text, encoding="utf-8")
        print(f"removed dispatch entries for '{name}' from {fattn_cu}")

    for p in (cuh_dst, entry_dst, inst_dst):
        if p.exists():
            p.unlink()
            print(f"deleted {p}")

    # Same reglob requirement on unregister: build.ninja still references the
    # files we just deleted, so the next build will fail compiling phantom files.
    if args.cmake_build_dir:
        bd = Path(args.cmake_build_dir).resolve()
        if not bd.exists():
            sys.exit(f"ERROR: --cmake-build-dir does not exist: {bd}")
        print(f"  reconfigure: cmake {bd}")
        rc = subprocess.call(["cmake", str(bd)], stdout=subprocess.DEVNULL)
        if rc != 0:
            sys.exit(f"ERROR: cmake reconfigure failed (rc={rc})")


def cmd_list(args):
    ggml_root = Path(args.ggml_root).resolve()
    fattn_cu = ggml_root / "src" / "ggml-cuda" / "fattn.cu"
    text = fattn_cu.read_text(encoding="utf-8")
    names = sorted(existing_variants(text))
    if not names:
        print("(no variants registered)")
        return
    for n in names:
        m = re.search(rf"BEST_FATTN_KERNEL_RDNA_APU_{n.upper()}\s*=\s*(\d+)", text)
        eid = m.group(1) if m else "?"
        print(f"  {n:24s}  enum={eid}  ENV=KERSOR_VARIANT={n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register")
    r.add_argument("--variant", required=True)
    # `--source` / `--project-root` are the generic adapter-contract flag names
    # (see _substrate/embedded/ADAPTER_CONTRACT.md); `--cuh-src` / `--ggml-root`
    # are kept as backward-compatible aliases for existing callers.
    r.add_argument("--cuh-src", "--source", dest="cuh_src", required=True,
                   help="path to the new variant .cuh to copy in (generic alias: --source)")
    r.add_argument("--ggml-root", "--project-root", dest="ggml_root", required=True,
                   help="path to llama.cpp/ggml project root (generic alias: --project-root)")
    r.add_argument("--dkq", type=int, default=256)
    r.add_argument("--dv", type=int, default=256)
    r.add_argument("--cmake-build-dir", default=None,
                   help="if set, run `cmake <dir>` after writing files so file(GLOB) re-globs the new .cu sources")
    r.set_defaults(func=cmd_register)

    u = sub.add_parser("unregister")
    u.add_argument("--variant", required=True)
    u.add_argument("--ggml-root", "--project-root", dest="ggml_root", required=True,
                   help="path to llama.cpp/ggml project root (generic alias: --project-root)")
    u.add_argument("--cmake-build-dir", default=None,
                   help="if set, run `cmake <dir>` after deleting files so the build no longer references phantoms")
    u.set_defaults(func=cmd_unregister)

    l = sub.add_parser("list")
    l.add_argument("--ggml-root", "--project-root", dest="ggml_root", required=True,
                   help="path to llama.cpp/ggml project root (generic alias: --project-root)")
    l.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
