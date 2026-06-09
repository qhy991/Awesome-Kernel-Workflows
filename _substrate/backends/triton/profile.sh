#!/usr/bin/env bash
# triton/profile.sh — profile the Triton launcher with PROTON (triton.profiler), NOT ncu.
#
# Why not ncu: ncu mangles the JIT kernel symbol (`triton_<fn>_<hash>`) so `ncu -k` cannot
# target it statically, and ncu's hardware counters need elevated perf-counter access
# (perf_event_paranoid / sudo). Proton instruments inside Triton, names kernels directly,
# and uses CUPTI device timing — no kernel-name regex and no sudo. It emits a `.hatchet`
# JSON that to_evidence.py maps to canonical metrics (latency_ms always; dram_pct/sm_pct as
# device-derived roofline estimates when the launcher annotates bytes/flops).
#
# Prints the POINTER (not the metrics): {ok,profiler:"proton",native_profile,format:"proton-hatchet"}.
# Spec §4.5. exit 0 ok · 3 bad args / missing input · 4 profiler/runtime unavailable (degrade to unknown).
#
# Launcher contract (--source): a plain Python module exposing
#   make_inputs()            -> tuple of args, OR make_inputs_from_problem(problem_dict)
#   forward(*inputs)         -> runs the @triton.jit kernel once (or launch(*inputs))
#   optional BYTES / FLOPS   -> per-invocation byte/flop counts (else read from problem.json:
#                               problem["bytes"], problem["flops"]) for roofline dram_pct/sm_pct.
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
  emit "{\"ok\":false,\"profiler\":\"proton\",\"native_profile\":null,\"error\":$_msg}"; exit 3
}
die4() {
  local _msg; _msg="$(json_escape "$1")"
  emit "{\"ok\":false,\"profiler\":\"proton\",\"native_profile\":null,\"error\":$_msg}"; exit 4
}

ARTIFACT="" PROBLEM="" OUT="" SOURCE="" KERNEL_NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact)    ARTIFACT="${2:-}"; shift 2 ;;
    --problem)     PROBLEM="${2:-}"; shift 2 ;;
    --out)         OUT="${2:-}"; shift 2 ;;
    --source)      SOURCE="${2:-}"; shift 2 ;;
    --kernel-name) KERNEL_NAME="${2:-}"; shift 2 ;;
    *) die3 "unknown arg: $1" ;;
  esac
done

[ -n "$ARTIFACT" ] || die3 "missing --artifact"
[ -n "$PROBLEM" ]  || die3 "missing --problem"
[ -n "$OUT" ]      || die3 "missing --out"
[ -e "$ARTIFACT" ] || die3 "artifact not found: $ARTIFACT"
[ -f "$PROBLEM" ]  || die3 "problem not found: $PROBLEM"
command -v python3 >/dev/null 2>&1 || die3 "python3 not found"

# Proton is a Python module (triton.profiler), not a PATH binary. Absent => exit 4 (degrade).
python3 -c "import triton.profiler" >/dev/null 2>&1 || die4 "proton (triton.profiler) not available"

# The launcher import + input construction + proton session run inside this harness, which
# prints the pointer envelope and sets the exit code. profile.sh passes that code through.
python3 - "$ARTIFACT" "$PROBLEM" "$OUT" "$SOURCE" "$KERNEL_NAME" <<'PY'
import sys, os, json, tempfile, shutil, importlib.util

artifact, problem_p, out, source, kname = sys.argv[1:6]
os.environ.setdefault("TRITON_CACHE_DIR", artifact)  # reuse warmed PTX from build.sh


def fail(msg, code):
    print(json.dumps({"ok": False, "profiler": "proton", "native_profile": None, "error": msg}))
    sys.exit(code)


try:
    import torch
    if not torch.cuda.is_available():
        fail("no CUDA device available (proton profiling deferred to GPU tier)", 4)
    import triton  # noqa: F401
    import triton.profiler as proton
except Exception as exc:  # noqa: BLE001
    fail(f"proton/triton runtime unavailable: {exc}", 4)

if not source or not os.path.isfile(source):
    fail("no runnable --source launcher provided; proton needs a launcher contract "
         "(make_inputs()+forward()); profiling deferred", 4)

try:
    spec = importlib.util.spec_from_file_location("candidate_launcher", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except Exception as exc:  # noqa: BLE001
    fail(f"failed to import launcher {source}: {exc}", 4)

try:
    problem = json.load(open(problem_p, encoding="utf-8"))
except Exception:  # noqa: BLE001
    problem = {}

# --- inputs ---
if hasattr(mod, "make_inputs"):
    inputs = mod.make_inputs()
elif hasattr(mod, "make_inputs_from_problem"):
    inputs = mod.make_inputs_from_problem(problem)
else:
    fail("launcher exposes no make_inputs()/make_inputs_from_problem(); cannot build inputs", 4)
if not isinstance(inputs, (tuple, list)):
    inputs = (inputs,)

fn = getattr(mod, "forward", None) or getattr(mod, "launch", None)
if fn is None:
    fail("launcher exposes no forward()/launch() entrypoint", 4)

# --- roofline annotation (optional; null dram_pct/sm_pct without it) ---
metrics = {}
b = getattr(mod, "BYTES", None) or problem.get("bytes")
f = getattr(mod, "FLOPS", None) or problem.get("flops")
if b:
    metrics["bytes"] = int(b)
if f:
    metrics["flops"] = int(f)

reps = int(problem.get("profile_reps", 20))
scope_name = kname or getattr(mod, "KERNEL_NAME", None) or "kernel"

try:
    fn(*inputs)              # warmup (force JIT + cache)
    torch.cuda.synchronize()
except Exception as exc:  # noqa: BLE001
    fail(f"launcher forward() raised during warmup: {exc}", 4)

tmp = tempfile.mkdtemp()
cwd = os.getcwd()
try:
    os.chdir(tmp)            # proton writes <name>.hatchet into CWD
    proton.start("prof", context="shadow", backend="cupti")
    with proton.scope(scope_name, metrics=metrics):
        for _ in range(reps):
            fn(*inputs)
    torch.cuda.synchronize()
    proton.finalize()
except Exception as exc:  # noqa: BLE001
    os.chdir(cwd)
    fail(f"proton session failed: {exc}", 4)
finally:
    os.chdir(cwd)

hatchet = os.path.join(tmp, "prof.hatchet")
if not os.path.isfile(hatchet):
    fail("proton produced no .hatchet profile", 4)
shutil.copyfile(hatchet, out)
print(json.dumps({"ok": True, "profiler": "proton", "native_profile": out,
                  "format": "proton-hatchet"}))
sys.exit(0)
PY
exit $?
