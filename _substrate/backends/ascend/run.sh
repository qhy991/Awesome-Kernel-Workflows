#!/usr/bin/env bash
# ascend/run.sh — run a staged AscendC direct-launch submission by DELEGATING to
# MultiKernelBench's eval_single_runner.py (which compiles via bisheng+cmake, checks
# correctness against the reference, and times with NPU events — all coupled). We map its
# result JSON onto the universal envelope.
#
# stdout keys MUST match anti_cheat.py --metrics:
#   {ok,compiled,correct,candidate_latency_ms,eager_latency_ms,compile_latency_ms,
#    claimed_speedup,...}
#   correct:false ⇒ claimed_speedup ≤ 1.0 (we floor at 1.0).
# exit 0 ok · 2 op-error / deferred NPU tier (json printed) · 3 bad args / missing input.
set -u

json_escape() {
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

err_envelope() {
  local msg="$1" rc="$2"
  local _msg; _msg="$(json_escape "$msg")"
  printf '{"ok":false,"compiled":false,"correct":false,"candidate_latency_ms":null,'
  printf '"eager_latency_ms":null,"compile_latency_ms":null,"claimed_speedup":1.0,'
  printf '"error":%s}\n' "$_msg"
  exit "$rc"
}

ARTIFACT="" KERNEL_SRC="" PROBLEM="" OUT="" OP="${MKB_OP:-}" LANGUAGE="ascendc_direct_launch"
MKB_ROOT="${MULTIKERNELBENCH_ROOT:-}" EVAL_RUNNER="" BASELINE_LATENCY=""
# BEGIN AUTO-GENERATED FLAG PARSER — regenerate from flags.yaml via _substrate/backends/_gen_flag_parser.py
# flags.yaml sha256=78e1f056565cbbe1
# DO NOT EDIT BETWEEN SENTINELS — edit flags.yaml and re-run: python3 _substrate/backends/_gen_flag_parser.py --write ascend
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact)                      ARTIFACT="${2:-}"; shift 2 ;;
    --kernel)                        KERNEL_SRC="${2:-}"; shift 2 ;;
    --problem)                       PROBLEM="${2:-}"; shift 2 ;;
    --out|--result)                  OUT="${2:-}"; shift 2 ;;
    --op)                            OP="${2:-}"; shift 2 ;;
    --language)                      LANGUAGE="${2:-}"; shift 2 ;;
    --mkb-root)                      MKB_ROOT="${2:-}"; shift 2 ;;
    --eval-runner)                   EVAL_RUNNER="${2:-}"; shift 2 ;;
    --baseline-latency)              BASELINE_LATENCY="${2:-}"; shift 2 ;;
    --reps|--rtol|--atol|--baseline) shift 2 ;;
    *) err_envelope "unknown arg: $1" 3 ;;
  esac
done
# END AUTO-GENERATED FLAG PARSER

# The submission to evaluate is the build.sh artifact; fall back to the raw --kernel source.
[ -n "$ARTIFACT" ] || ARTIFACT="$KERNEL_SRC"
# Default the output path next to the artifact when neither --out nor --result was given.
[ -n "$OUT" ] || { [ -n "$ARTIFACT" ] && OUT="${ARTIFACT}.run.json"; }

[ -n "$ARTIFACT" ] || err_envelope "missing --artifact/--kernel (staged submission json)" 3
[ -n "$OUT" ]      || err_envelope "missing --out/--result" 3
[ -f "$ARTIFACT" ] || err_envelope "artifact not found: $ARTIFACT" 3
command -v python3 >/dev/null 2>&1 || err_envelope "python3 not found" 3

# Resolve the op name: --op / $MKB_OP win, else read an "op"/"name" field from --problem,
# else try the submission artifact itself (it may carry a "problem"/"op"/"name" field).
if [ -z "$OP" ] && [ -n "$PROBLEM" ] && [ -f "$PROBLEM" ]; then
  OP="$(python3 - "$PROBLEM" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    print(d.get("op") or d.get("name") or d.get("problem") or "")
except Exception:
    print("")
PY
)"
fi
# Last resort: the submission artifact may name its own op.
if [ -z "$OP" ] && [ -f "$ARTIFACT" ]; then
  OP="$(python3 - "$ARTIFACT" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    print(d.get("op") or d.get("problem") or d.get("name") or "")
except Exception:
    print("")
PY
)"
fi
[ -n "$OP" ] || err_envelope "missing op name: pass --op, set \$MKB_OP, or add op/name to --problem or the submission" 3

# Resolve the eval runner / MultiKernelBench root.
if [ -z "$EVAL_RUNNER" ] && [ -n "$MKB_ROOT" ]; then
  EVAL_RUNNER="$MKB_ROOT/eval_single_runner.py"
fi
if [ -z "$EVAL_RUNNER" ] || [ ! -f "$EVAL_RUNNER" ]; then
  err_envelope "eval_single_runner.py not found (pass --mkb-root <MultiKernelBench dir>, --eval-runner <path>, or set MULTIKERNELBENCH_ROOT); deferred NPU tier" 2
fi
[ -n "$MKB_ROOT" ] || MKB_ROOT="$(dirname "$EVAL_RUNNER")"

# Preflight: torch_npu must be importable, else this is the deferred (no-NPU) tier.
if ! python3 -c "import torch_npu" >/dev/null 2>&1; then
  err_envelope "torch_npu unavailable — no Ascend NPU runtime on this host (deferred NPU tier)" 2
fi

RESULT_JSON="$(mktemp)"
STDERR_FILE="$(mktemp)"
( cd "$MKB_ROOT" && python3 eval_single_runner.py -i "$ARTIFACT" -o "$OP" -l "$LANGUAGE" -r "$RESULT_JSON" ) >"$STDERR_FILE" 2>&1
RC=$?
[ -s "$STDERR_FILE" ] && cat "$STDERR_FILE" 1>&2

if [ "$RC" -ne 0 ] || [ ! -s "$RESULT_JSON" ]; then
  TAIL="$(tail -c 2048 "$STDERR_FILE" 2>/dev/null || true)"
  rm -f "$RESULT_JSON" "$STDERR_FILE"
  err_envelope "eval_single_runner failed (exit $RC): $TAIL" 2
fi
rm -f "$STDERR_FILE"

# Map MultiKernelBench result {compiled, correctness, performance.mean, cheating} -> envelope.
python3 - "$RESULT_JSON" "$OUT" "${BASELINE_LATENCY:-}" <<'PY'
import json, sys
res_path, out_path, baseline = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(res_path, encoding="utf-8"))
compiled = bool(r.get("compiled"))
cheating = bool(r.get("cheating"))
correct = bool(r.get("correctness")) and not cheating
perf = r.get("performance") or {}
cand = perf.get("mean")  # NPU-event milliseconds
base = float(baseline) if baseline not in (None, "",) else None
speedup = 1.0
if correct and cand and base:
    speedup = base / cand
env = {
    "ok": correct,
    "compiled": compiled,
    "correct": correct,
    "candidate_latency_ms": cand if correct else None,
    "eager_latency_ms": base,
    "compile_latency_ms": base,
    "claimed_speedup": speedup if correct else 1.0,
    "hardware": r.get("hardware"),
    "cheating": cheating,
}
for k in ("compile_info", "correctness_info", "cheating_info"):
    if r.get(k):
        env[k] = r[k]
open(out_path, "w", encoding="utf-8").write(json.dumps(env) + "\n")
print(json.dumps(env))
PY
RC=$?
rm -f "$RESULT_JSON"
exit $RC
