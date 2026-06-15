#!/usr/bin/env python3
"""sgl_register_variant.py  --  DEPRECATED

2026-06-14: This hand-written per-framework adapter is deprecated in favor of the
KerSor `framework-integrator` agent (same migration as `llamacpp_register_variant.py`).
A general LLM agent can derive the SGLang integration at runtime by reading
`common_extension.cc` (TORCH_LIBRARY bindings), CMakeLists.txt (explicit SOURCES
list), and the `benchmark/kernels/` harnesses. The key inference step — the ENV
gate at library load to handle static-init torch binding — is now part of the
integrator playbook, not locked in this script.

This script remains as a **reference implementation**. See `ADAPTER_CONTRACT.md`.

Original docstring follows.
--

Reference adapter #2 for the embedded-dispatch contract
(_substrate/embedded/ADAPTER_CONTRACT.md), targeting sglang's sgl-kernel — a
PyTorch custom-op extension (cmake + csrc/ + TORCH_LIBRARY, built into one .so).

This demonstrates the contract generalizing from llama.cpp's C++ runtime dispatch
to a torch-op extension library. Where llamacpp_register_variant.py edits a C++
selector in fattn.cu, this edits two project files, byte-exact and marker-wrapped:

  1. CMakeLists.txt  -- inserts the variant .cu into the explicit `set(SOURCES ...)`.
  2. csrc/common_extension.cc -- inserts an ENV-gated branch around the op's
     `m.impl("<op>", torch::kCUDA, &<orig>)` registration.

Why a load-time gate (not per-call): torch binds an op's impl once at TORCH_LIBRARY
static-init. The benchmark harness launches a fresh process per variant with
KERSOR_VARIANT set, so binding the variant at load is sufficient and keeps default
dispatch byte-for-byte unchanged when the env var is absent.

Variant authoring contract (the sgl flavor of EMBEDDING_CONTRACT): the variant .cu
MUST define, in addition to its kernel, a fixed-signature registrar:

    void kersor_register_<variant>(torch::Library& m);   // binds its kernel to <op>

This is what lets the adapter wire a variant in WITHOUT parsing the op's C++
signature — the variant file owns its own signature.

Usage:
  register   --variant v1 --source /path/new_rmsnorm.cu --project-root /path/sgl-kernel --op rmsnorm
  unregister --variant v1 --project-root /path/sgl-kernel
  list       --project-root /path/sgl-kernel
  self-test  --project-root /path/sgl-kernel --op rmsnorm   # round-trip on a temp copy; asserts byte-exact

`--cuh-src`/`--ggml-root` aliases are accepted for cross-adapter symmetry.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile

VARIANT_SUBDIR = os.path.join("csrc", "_kersor_variants")
CMAKE = "CMakeLists.txt"
DISPATCH = os.path.join("csrc", "common_extension.cc")

SRC_BEGIN = "    # >>> KERSOR_VARIANT_SRC_BEGIN {name} <<<"
SRC_END = "    # <<< KERSOR_VARIANT_SRC_END {name} >>>"
DECL_BEGIN = "// >>> KERSOR_VARIANT_DECL_BEGIN {name} <<<"
DECL_END = "// <<< KERSOR_VARIANT_DECL_END {name} >>>"
GATE_BEGIN = "  // >>> KERSOR_VARIANT_GATE_BEGIN {name} <<<"
GATE_END = "  // <<< KERSOR_VARIANT_GATE_END {name} >>>"


def _read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def _write(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def _md5(p):
    return hashlib.md5(_read(p).encode("utf-8")).hexdigest()


def _fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def registered_variants(root):
    disp = _read(os.path.join(root, DISPATCH))
    return set(re.findall(r"KERSOR_VARIANT_GATE_BEGIN (\S+) <<<", disp))


def cmd_register(a):
    root = a.project_root
    name = a.variant
    op = a.op
    if not op:
        _fail("register requires --op (the torch op name, e.g. rmsnorm)")
    src = a.source
    if not os.path.isfile(src):
        _fail(f"variant source not found: {src}")
    cmake_path = os.path.join(root, CMAKE)
    disp_path = os.path.join(root, DISPATCH)
    for p in (cmake_path, disp_path):
        if not os.path.isfile(p):
            _fail(f"not an sgl-kernel project root (missing {p})")
    if name in registered_variants(root):
        _fail(f"variant '{name}' already registered; unregister first")

    # 1. copy variant .cu into csrc/_kersor_variants/<op>__<name>.cu
    vdir = os.path.join(root, VARIANT_SUBDIR)
    os.makedirs(vdir, exist_ok=True)
    rel_cu = os.path.join(VARIANT_SUBDIR, f"{op}__{name}.cu").replace(os.sep, "/")
    shutil.copyfile(src, os.path.join(root, rel_cu))

    # 2. insert into set(SOURCES ...) — marker-wrapped, just before the closing ')'
    cmake = _read(cmake_path)
    m = re.search(r"\nset\(SOURCES\b.*?\n\)", cmake, re.DOTALL)
    if not m:
        _fail("could not locate `set(SOURCES ... )` block in CMakeLists.txt")
    block = m.group(0)
    insert = (
        SRC_BEGIN.format(name=name) + "\n"
        + f'    "{rel_cu}"\n'
        + SRC_END.format(name=name) + "\n"
    )
    # block ends with "\n)"; keep that "\n", insert full marker lines, re-add ")".
    new_block = block[:-1] + insert + ")"
    cmake = cmake[:m.start()] + new_block + cmake[m.end():]
    _write(cmake_path, cmake)

    # 3. ENV-gate in common_extension.cc: forward-declare the registrar, and wrap
    #    the original `m.impl("<op>", ..., &<orig>);` line in a load-time branch.
    disp = _read(disp_path)
    impl_re = re.compile(
        r'^([ \t]*)m\.impl\(\s*"' + re.escape(op) + r'"\s*,[^\n;]*\);[ \t]*$',
        re.MULTILINE,
    )
    im = impl_re.search(disp)
    if not im:
        _fail(f'could not find `m.impl("{op}", ...)` in {DISPATCH}')
    orig_line = im.group(0)
    indent = im.group(1)
    gate = (
        GATE_BEGIN.format(name=name) + "\n"
        + indent + 'if (const char* __kv = std::getenv("KERSOR_VARIANT")) {\n'
        + indent + f'  if (std::string(__kv) == "{name}") {{ kersor_register_{name}(m); }}\n'
        + indent + "  else {\n"
        + indent + "  " + orig_line.strip() + "\n"
        + indent + "  }\n"
        + indent + "} else {\n"
        + indent + "  " + orig_line.strip() + "\n"
        + indent + "}\n"
        + GATE_END.format(name=name)
    )
    disp = disp[:im.start()] + gate + disp[im.end():]

    # forward declaration just after the last top-of-file #include
    decl = (
        DECL_BEGIN.format(name=name) + "\n"
        + "#include <string>\n#include <cstdlib>\n"
        + f"void kersor_register_{name}(torch::Library& m);\n"
        + DECL_END.format(name=name) + "\n"
    )
    includes = list(re.finditer(r"^#include .*$", disp, re.MULTILINE))
    pos = includes[-1].end() + 1 if includes else 0
    disp = disp[:pos] + "\n" + decl + disp[pos:]
    _write(disp_path, disp)

    print(f"registered variant '{name}' for op '{op}'")
    print(f"  src : {rel_cu}")
    print(f"  ENV : KERSOR_VARIANT={name}")
    print("  NOTE: variant .cu must define void kersor_register_%s(torch::Library&)" % name)


def _strip_markers(text, begin_tag, end_tag, name):
    b = begin_tag.format(name=name)
    e = end_tag.format(name=name)
    pat = re.compile(r"\n?" + re.escape(b) + r".*?" + re.escape(e) + r"\n?", re.DOTALL)
    return pat.sub("", text, count=1)


def _restore_gate(text, name):
    """Replace the gate block with the single original impl line it wrapped."""
    b = GATE_BEGIN.format(name=name)
    e = GATE_END.format(name=name)
    pat = re.compile(r"[ \t]*" + re.escape(b) + r".*?" + re.escape(e) + r"\n?", re.DOTALL)
    m = pat.search(text)
    if not m:
        return text
    inner = m.group(0)
    # the original line is the (de-indented) `m.impl(...)` that appears in the else
    impl = re.search(r'm\.impl\(\s*"[^"]+"\s*,[^\n;]*\);', inner)
    orig_indent = re.match(r"([ \t]*)", inner).group(1)
    replacement = (orig_indent + impl.group(0) + "\n") if impl else ""
    return text[:m.start()] + replacement + text[m.end():]


def cmd_unregister(a):
    root = a.project_root
    name = a.variant
    cmake_path = os.path.join(root, CMAKE)
    disp_path = os.path.join(root, DISPATCH)

    cmake = _read(cmake_path)
    # SOURCES markers were inserted as full lines (each ending in "\n") before the
    # closing ")", so remove the block plus its trailing newline exactly.
    _b, _e = SRC_BEGIN.format(name=name), SRC_END.format(name=name)
    cmake = re.sub(re.escape(_b) + r".*?" + re.escape(_e) + r"\n", "", cmake, count=1, flags=re.DOTALL)
    _write(cmake_path, cmake)

    disp = _read(disp_path)
    disp = _restore_gate(disp, name)
    disp = _strip_markers(disp, DECL_BEGIN, DECL_END, name)
    _write(disp_path, disp)

    cu = os.path.join(root, VARIANT_SUBDIR)
    for f in os.listdir(cu) if os.path.isdir(cu) else []:
        if f.endswith(f"__{name}.cu"):
            os.remove(os.path.join(cu, f))
    print(f"unregistered variant '{name}' (byte-exact)")


def cmd_list(a):
    for n in sorted(registered_variants(a.project_root)):
        print(n)


def cmd_self_test(a):
    """Round-trip register->unregister on a temp copy; assert byte-exact restore."""
    root = a.project_root
    op = a.op or "rmsnorm"
    with tempfile.TemporaryDirectory() as tmp:
        # copy just the two touched files + a csrc dir into a fake root
        fake = os.path.join(tmp, "sgl")
        os.makedirs(os.path.join(fake, "csrc"))
        shutil.copyfile(os.path.join(root, CMAKE), os.path.join(fake, CMAKE))
        shutil.copyfile(os.path.join(root, DISPATCH), os.path.join(fake, DISPATCH))
        before = {CMAKE: _md5(os.path.join(fake, CMAKE)),
                  DISPATCH: _md5(os.path.join(fake, DISPATCH))}
        # a throwaway variant source
        vsrc = os.path.join(tmp, "v.cu")
        _write(vsrc, "// test variant\nvoid kersor_register_selftest(torch::Library& m){}\n")

        ns = argparse.Namespace(project_root=fake, variant="selftest",
                                source=vsrc, op=op)
        cmd_register(ns)
        assert "selftest" in registered_variants(fake), "variant not registered"
        assert os.path.isdir(os.path.join(fake, VARIANT_SUBDIR)), "variant dir missing"
        cmd_unregister(argparse.Namespace(project_root=fake, variant="selftest"))

        after = {CMAKE: _md5(os.path.join(fake, CMAKE)),
                 DISPATCH: _md5(os.path.join(fake, DISPATCH))}
        leftovers = registered_variants(fake)
        ok = before == after and not leftovers
        for f in (CMAKE, DISPATCH):
            tag = "OK " if before[f] == after[f] else "DIFF"
            print(f"  [{tag}] byte-exact {f}: {before[f][:8]} -> {after[f][:8]}")
        print(f"  leftover variants after unregister: {sorted(leftovers) or 'none'}")
        if not ok:
            _fail("self-test FAILED: not byte-exact reversible")
        print("self-test PASSED: register/unregister is byte-exact reversible")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register")
    r.add_argument("--variant", required=True)
    r.add_argument("--source", "--cuh-src", dest="source", required=True)
    r.add_argument("--project-root", "--ggml-root", dest="project_root", required=True)
    r.add_argument("--op", required=True, help="torch op name to gate, e.g. rmsnorm")
    r.set_defaults(func=cmd_register)

    u = sub.add_parser("unregister")
    u.add_argument("--variant", required=True)
    u.add_argument("--project-root", "--ggml-root", dest="project_root", required=True)
    u.set_defaults(func=cmd_unregister)

    l = sub.add_parser("list")
    l.add_argument("--project-root", "--ggml-root", dest="project_root", required=True)
    l.set_defaults(func=cmd_list)

    s = sub.add_parser("self-test")
    s.add_argument("--project-root", "--ggml-root", dest="project_root", required=True)
    s.add_argument("--op", default="rmsnorm")
    s.set_defaults(func=cmd_self_test)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
