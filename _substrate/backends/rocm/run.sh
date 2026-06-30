#!/usr/bin/env bash
# rocm/run.sh -- run a candidate (either a compiled .so or a project test command)
# and emit the universal envelope. Two modes:
#
#   1. Standalone: --artifact <.so> --problem <problem.py> [...]
#      Same shape as cuda/run.sh, but uses HIP instead of CUDA in the python launcher.
#
#   2. In-codebase: --test-cmd "<cmd>" [--bench-cmd "<cmd>"] [--correct-regex <re>] [--latency-regex <re>]
#      Runs the project's own test/benchmark commands verbatim, parses pass/fail and
#      latency from their stdout. This is what KerSor's in-place patch loop uses.
#
# Universal envelope:
#   {ok,compiled,correct,candidate_latency_ms,eager_latency_ms,compile_latency_ms,
#    claimed_speedup,...}
#   correct:false => claimed_speedup floored at 1.0.
# exit 0 ok . 2 op-error (json printed) . 3 bad args / missing input.
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

ARTIFACT="" PROBLEM="" OUT="" REPS=50 RTOL="1e-3" ATOL="1e-3" BASELINE="both"
TEST_CMD="" BENCH_CMD="" CORRECT_REGEX="" LATENCY_REGEX="" BASELINE_LATENCY=""
# BEGIN AUTO-GENERATED FLAG PARSER — regenerate from flags.yaml via _substrate/backends/_gen_flag_parser.py
# flags.yaml sha256=467574f6ccee2a96
# DO NOT EDIT BETWEEN SENTINELS — edit flags.yaml and re-run: python3 _substrate/backends/_gen_flag_parser.py --write rocm
while [ $# -gt 0 ]; do
  case "$1" in
    --artifact)         ARTIFACT="${2:-}"; shift 2 ;;
    --problem)          PROBLEM="${2:-}"; shift 2 ;;
    --out)              OUT="${2:-}"; shift 2 ;;
    --reps)             REPS="${2:-}"; shift 2 ;;
    --rtol)             RTOL="${2:-}"; shift 2 ;;
    --atol)             ATOL="${2:-}"; shift 2 ;;
    --baseline)         BASELINE="${2:-}"; shift 2 ;;
    --test-cmd)         TEST_CMD="${2:-}"; shift 2 ;;
    --bench-cmd)        BENCH_CMD="${2:-}"; shift 2 ;;
    --correct-regex)    CORRECT_REGEX="${2:-}"; shift 2 ;;
    --latency-regex)    LATENCY_REGEX="${2:-}"; shift 2 ;;
    --baseline-latency) BASELINE_LATENCY="${2:-}"; shift 2 ;;
    *) err_envelope "unknown arg: $1" 3 ;;
  esac
done
# END AUTO-GENERATED FLAG PARSER

# Mode 2: in-codebase test/bench commands.
if [ -n "$TEST_CMD" ] || [ -n "$BENCH_CMD" ]; then
  TEST_OUT_FILE="$(mktemp)"
  CORRECT="true"
  CORRECT_DETAIL=""
  if [ -n "$TEST_CMD" ]; then
    # shellcheck disable=SC2086
    bash -c "$TEST_CMD" >"$TEST_OUT_FILE" 2>&1
    TEST_RC=$?
    if [ "$TEST_RC" -ne 0 ]; then
      CORRECT="false"
      CORRECT_DETAIL="test command exited $TEST_RC"
    elif [ -n "$CORRECT_REGEX" ]; then
      if ! grep -E -q "$CORRECT_REGEX" "$TEST_OUT_FILE"; then
        CORRECT="false"
        CORRECT_DETAIL="correct-regex did not match"
      fi
    fi
  fi

  CAND_LAT_MS="null"
  BENCH_OUT_FILE="$(mktemp)"
  if [ -n "$BENCH_CMD" ] && [ "$CORRECT" = "true" ]; then
    # shellcheck disable=SC2086
    bash -c "$BENCH_CMD" >"$BENCH_OUT_FILE" 2>&1
    BENCH_RC=$?
    if [ "$BENCH_RC" -ne 0 ]; then
      CORRECT_DETAIL="benchmark command exited $BENCH_RC"
    elif [ -n "$LATENCY_REGEX" ]; then
      # Capture the first numeric token matching the regex (us or ms). Caller's regex
      # MUST include exactly one captured group containing the number.
      LAT_RAW="$(grep -E -o "$LATENCY_REGEX" "$BENCH_OUT_FILE" | head -n1 | grep -E -o '[0-9]+\.?[0-9]*' | head -n1 || true)"
      if [ -n "$LAT_RAW" ]; then CAND_LAT_MS="$LAT_RAW"; fi
    fi
  fi

  CLAIMED="1.0"
  if [ "$CORRECT" = "true" ] && [ "$CAND_LAT_MS" != "null" ] && [ -n "$BASELINE_LATENCY" ]; then
    CLAIMED="$(python3 -c "print(${BASELINE_LATENCY}/${CAND_LAT_MS})")"
  fi

  TEST_TAIL="$(tail -c 4096 "$TEST_OUT_FILE" 2>/dev/null || true)"
  BENCH_TAIL="$(tail -c 4096 "$BENCH_OUT_FILE" 2>/dev/null || true)"
  TEST_JSON="$(json_escape "$TEST_TAIL")"
  BENCH_JSON="$(json_escape "$BENCH_TAIL")"
  DETAIL_JSON="$(json_escape "$CORRECT_DETAIL")"
  rm -f "$TEST_OUT_FILE" "$BENCH_OUT_FILE"

  BASE_VAL="${BASELINE_LATENCY:-null}"
  printf '{"ok":true,"compiled":true,"correct":%s,"candidate_latency_ms":%s,' "$CORRECT" "$CAND_LAT_MS"
  printf '"eager_latency_ms":%s,"compile_latency_ms":%s,"claimed_speedup":%s,' "$BASE_VAL" "$BASE_VAL" "$CLAIMED"
  printf '"mode":"in_codebase","test_tail":%s,"bench_tail":%s,"detail":%s}\n' "$TEST_JSON" "$BENCH_JSON" "$DETAIL_JSON"
  exit 0
fi

# Mode 1: standalone .so / problem.py.
[ -n "$ARTIFACT" ] || err_envelope "missing --artifact (or use --test-cmd for in-codebase mode)" 3
[ -n "$PROBLEM" ]  || err_envelope "missing --problem" 3
[ -n "$OUT" ]      || err_envelope "missing --out" 3
[ -f "$PROBLEM" ]  || err_envelope "problem not found: $PROBLEM" 3
[ -f "$ARTIFACT" ] || err_envelope "artifact not found: $ARTIFACT" 3
command -v python3 >/dev/null 2>&1 || err_envelope "python3 not found" 3

python3 - "$ARTIFACT" "$PROBLEM" "$OUT" "$REPS" "$RTOL" "$ATOL" "$BASELINE" <<'PY'
import sys, json
artifact, problem, out, reps, rtol, atol, baseline = sys.argv[1:8]
result = {
    "ok": False, "compiled": True, "correct": False,
    "candidate_latency_ms": None, "eager_latency_ms": None,
    "compile_latency_ms": None, "claimed_speedup": 1.0,
    "error": "GPU execution deferred: requires AMD ROCm device + HIP runtime",
}
try:
    import torch
    if not torch.cuda.is_available():
        # On ROCm, torch.cuda.is_available() returns True when HIP is present.
        result["error"] = "no HIP/ROCm device available (deferred GPU tier)"
        print(json.dumps(result)); sys.exit(2)
    result["error"] = "real ROCm GPU run not yet implemented"
    print(json.dumps(result)); sys.exit(2)
except ImportError:
    result["error"] = "torch (with ROCm build) unavailable"
    print(json.dumps(result)); sys.exit(2)
PY
exit $?
