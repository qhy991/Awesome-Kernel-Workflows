#!/usr/bin/env bash
# ascend/build.sh — validate the MultiKernelBench AscendC direct-launch JSON submission
# and stage it as the artifact. The real bisheng+cmake compile is COUPLED with correctness
# and timing inside MultiKernelBench's eval_single (invoked by run.sh), so "build" here is a
# structural-contract preflight (JIT-like: no separate .so artifact at the substrate layer).
# Spec §4.5 universal envelope: ONE json on stdout, logs to stderr.
#   exit 0 ok · 2 compile/validation failure (json printed) · 3 bad args / missing tool.
set -u

emit() { printf '%s\n' "$1"; }

json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

die3() {
  local _msg; _msg="$(json_escape "$1")"
  emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"error\":$_msg}"; exit 3
}

SOURCE="" OUT="" ARCH="dav-2201" BUILD_CMD="" EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --source)    SOURCE="${2:-}"; shift 2 ;;
    --out)       OUT="${2:-}"; shift 2 ;;
    --arch)      ARCH="${2:-}"; shift 2 ;;
    --build-cmd) BUILD_CMD="${2:-}"; shift 2 ;;   # accepted for envelope parity; compile is deferred to run
    --extra)     EXTRA="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$SOURCE" ] || die3 "missing --source"
[ -n "$OUT" ]    || die3 "missing --out"
[ -f "$SOURCE" ] || die3 "source not found: $SOURCE"
command -v python3 >/dev/null 2>&1 || die3 "python3 not found on PATH"

START="$(python3 -c 'import time;print(int(time.time()*1000))')"

# Structural validation of the JSON submission contract (see idioms.json impl_requirements).
VALIDATION="$(python3 - "$SOURCE" "$OUT" <<'PY'
import json, re, sys, shutil
src_path, out_path = sys.argv[1], sys.argv[2]
raw = open(src_path, encoding="utf-8").read().strip()
# tolerate a ```json ... ``` fence around the submission
m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
if m:
    raw = m.group(1).strip()
errs = []
try:
    spec = json.loads(raw)
except Exception as e:
    print(json.dumps({"ok": False, "error": f"submission is not valid JSON: {e}"})); sys.exit(0)
if not isinstance(spec, dict):
    print(json.dumps({"ok": False, "error": "submission top-level is not a JSON object"})); sys.exit(0)

entry = (spec.get("entry") or {}).get("model")
if not entry or "::" not in str(entry):
    errs.append("entry.model must be 'ModelNew.py::ModelNew'")
sources = spec.get("sources")
if not isinstance(sources, list) or not sources:
    errs.append("sources[] missing or empty")
    print(json.dumps({"ok": False, "error": "; ".join(errs)})); sys.exit(0)

texts = {}
for s in sources:
    p = (s or {}).get("path"); c = (s or {}).get("content", "")
    if not p:
        errs.append("a source entry has no path"); continue
    texts[p] = c

has_modelnew = any(p.endswith(".py") and "class ModelNew" in c for p, c in texts.items())
has_pybind   = any("PYBIND11_MODULE" in c for c in texts.values())
has_aicore   = any("__aicore__" in c for c in texts.values())
if not has_modelnew:
    errs.append("no ModelNew.py source defining 'class ModelNew'")
if not has_pybind:
    errs.append("no source containing PYBIND11_MODULE (pybind binding)")
if not has_aicore:
    errs.append("no Ascend C kernel source containing '__aicore__'")

# Anti-cheat preflight: forbid obvious torch compute ops in reachable ModelNew code.
for p, c in texts.items():
    if p.endswith(".py") and "class ModelNew" in c:
        for bad in ("torch.matmul", "torch.nn.functional", "F.linear", "F.conv", "nn.Linear", "nn.Conv"):
            if bad in c:
                errs.append(f"ModelNew.py uses forbidden torch compute op '{bad}' (forward must delegate to the NPU op)")

if errs:
    print(json.dumps({"ok": False, "error": "; ".join(errs)})); sys.exit(0)

# Stage the (de-fenced) canonical submission as the artifact for run.sh.
open(out_path, "w", encoding="utf-8").write(raw)
print(json.dumps({"ok": True, "artifact": out_path}))
PY
)"

END="$(python3 -c 'import time;print(int(time.time()*1000))')"
BUILD_MS=$(( END - START ))

OK="$(printf '%s' "$VALIDATION" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("ok"))')"
if [ "$OK" = "True" ]; then
  ESC_OUT="$(json_escape "$OUT")"
  emit "{\"ok\":true,\"compiled\":true,\"artifact\":$ESC_OUT,\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":\"\",\"note\":\"structural-contract preflight; bisheng compile deferred to run via eval_single\"}"
  exit 0
else
  ERR="$(printf '%s' "$VALIDATION" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("error",""))')"
  ESC_ERR="$(json_escape "$ERR")"
  emit "{\"ok\":false,\"compiled\":false,\"artifact\":null,\"build_latency_ms\":$BUILD_MS,\"stderr_tail\":$ESC_ERR}"
  exit 2
fi
