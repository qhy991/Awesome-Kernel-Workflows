#!/usr/bin/env python3
"""Layer B — code-integrity gate (KerSor Solver SDK).

Distinct from anti_cheat.py (which asks "is this delegating to a library?"):
this gate asks "is the generated kernel COMPLETE and non-stub?" It catches the
two failure modes observed in the B200 archive:
  - 018 KSearch 6/6 truncation — the model stopped mid-kernel, leaving
    unbalanced braces / a dangling opener.
  - L2-054 whole-file Write stub — a syntactically-balanced skeleton whose
    kernel body is empty or a placeholder.

Deterministic, static heuristics on the kernel source. An integrity-flagged
attempt must be rejected before its "speedup" can enter memory.

Usage:
  code_integrity.py --source kernel.cu
  code_integrity.py --source-text "<code>" [--lang cuda|triton|python|auto] [--min-lines 50]
Prints: {valid, flags}
"""
import sys, json, argparse, re

# Placeholders that signal "the model did not actually implement the body".
PLACEHOLDER_PATTERNS = [
    (re.compile(r"your\s+code\s+goes?\s+here", re.I), "your-code-here placeholder"),
    (re.compile(r"implement\s+(?:this|the\s+body|your\s+kernel)\s+here", re.I), "implement-here placeholder"),
    (re.compile(r"__placeholder__", re.I), "__placeholder__ marker"),
    (re.compile(r"//\s*TODO\s*:?\s*implement", re.I), "TODO-implement comment"),
    (re.compile(r"/\*\s*TODO\s*:?\s*implement[^*]*\*/", re.I), "TODO-implement block"),
]
# A kernel body that is only these (after stripping) is an empty/stub body.
STUB_BODY_HINTS = re.compile(r"^\s*(?:return\s*;?|pass\s*;?|/\*.*\*/|//.*)?\s*$")


def _detect_lang(src):
    if re.search(r"@triton\.jit|^\s*def\s+\w+\s*\(|^\s*import\s+\w|^\s*from\s+\w+\s+import", src, re.M):
        return "python"
    if re.search(r"__global__|__device__|__host__|#include\b|#pragma\b", src):
        return "cuda"
    return "python"  # safe default for the '#' comment rule (cuda preprocessor is the only '#'' risk)


def strip_strings_and_comments(src, lang):
    """Remove string literals and comments so brace/paren counting is not fooled
    by characters inside them. Conservative — leaves code structure intact."""
    out = []
    i, n = 0, len(src)
    py = (lang == "python")
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        # Block comment /* */ (C-family)
        if c == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            i = (j + 2) if j != -1 else n
            continue
        # Line comment // (C-family)
        if c == "/" and nxt == "/":
            j = src.find("\n", i + 2)
            i = j if j != -1 else n
            continue
        # Python line comment '#'
        if py and c == "#":
            # Don't strip shebang lines oddly; just skip to EOL.
            j = src.find("\n", i + 1)
            i = j if j != -1 else n
            continue
        # String literals: ", ', and triple-quoted
        if c in ("'", '"'):
            quote = c
            triple = src[i:i + 3] == quote * 3
            if triple:
                j = src.find(quote * 3, i + 3)
                i = (j + 3) if j != -1 else n
                continue
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == quote:
                    j += 1
                    break
                if src[j] == "\n":
                    break  # unterminated string line — stop at EOL
                j += 1
            out.append('""')
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


def check_balance(stripped):
    """Return list of flag dicts for unbalanced () [] {} or dangling openers."""
    flags = []
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    openers = "([{"
    for ch in stripped:
        if ch in openers:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                flags.append({"type": "unbalanced", "detail": f"mismatched '{ch}'"})
                return flags  # one is enough to know it's broken
            stack.pop()
    if stack:
        # Unclosed opener at EOF → the kernel was truncated mid-body.
        flags.append({
            "type": "truncated_code",
            "detail": f"unclosed {''.join(stack)} at EOF — kernel appears truncated",
        })
    return flags


def check_empty_body(src, stripped):
    """Detect a kernel/function definition whose body is empty or a stub.

    Catches the L2-054 whole-file-stub pattern: the skeleton compiles and
    balances, but the kernel never does the work.
    """
    flags = []
    # CUDA: __global__|__device__ ... Name(...) { <body> }
    for m in re.finditer(
        r"(__global__|__device__|__host__)[^{]*?\b\w+\s*\([^)]*\)\s*\{([^{}]*)\}",
        stripped,
    ):
        body = m.group(2)
        if STUB_BODY_HINTS.match(body) or re.fullmatch(r"\s*", body):
            flags.append({"type": "empty_body", "detail": "CUDA kernel with empty/stub body"})
            break
    # Triton/Python: @triton.jit def name(...): <only pass / docstring / ellipsis>
    for m in re.finditer(
        r"def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*",
        src,
    ):
        rest = src[m.end():m.end() + 120]
        rest_stripped = re.sub(r"#.*", "", rest)
        first_real = next((ln.strip() for ln in rest_stripped.splitlines() if ln.strip()), "")
        if first_real in ("pass", "...", '"""', "'''") or first_real.startswith(("pass", "...")):
            flags.append({"type": "empty_body", "detail": "Python/Triton def with pass/ellipsis body"})
            break
    return flags


def check_placeholders(src):
    flags = []
    for pat, msg in PLACEHOLDER_PATTERNS:
        if pat.search(src):
            flags.append({"type": "placeholder", "detail": msg})
    return flags


def check_min_lines(src, min_lines):
    if min_lines and len(src.splitlines()) < min_lines:
        return [{"type": "below_min_lines",
                 "detail": f"{len(src.splitlines())} lines < required {min_lines}"}]
    return []


def evaluate(src, lang="auto", min_lines=0):
    if not src or not src.strip():
        return {"valid": False, "flags": [{"type": "empty_body", "detail": "empty source"}]}
    if lang == "auto":
        lang = _detect_lang(src)
    stripped = strip_strings_and_comments(src, lang)
    flags = []
    flags += check_balance(stripped)
    flags += check_empty_body(src, stripped)
    flags += check_placeholders(src)
    flags += check_min_lines(src, min_lines)
    blocking = {"truncated_code", "empty_body"}
    valid = not any(f["type"] in blocking for f in flags)
    return {"valid": valid, "flags": flags}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source")
    ap.add_argument("--source-text")
    ap.add_argument("--lang", choices=["cuda", "triton", "python", "auto"], default="auto")
    ap.add_argument("--min-lines", type=int, default=0)
    a = ap.parse_args()
    if a.source_text is not None:
        src = a.source_text
    elif a.source:
        with open(a.source) as f:
            src = f.read()
    else:
        ap.error("provide --source or --source-text")
    res = evaluate(src, lang=a.lang, min_lines=a.min_lines)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
