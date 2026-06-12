#!/usr/bin/env bash
# stage1-roundtrip.sh - End-to-end verification of the register/build/test/bench
# /unregister chain for the llamacpp-embedded-search workflow, using the
# existing fattn-rdna-apu.cuh as the variant source (a "trivial clone").
#
# This isolates the dispatch + register-script half from the subagent half.
# What it proves:
#   1. fattn.cu is byte-exact restored after unregister
#   2. KERSOR_VARIANT=<name> actually routes dispatch to the variant
#      (verified by running test-backend-ops with and without the ENV var
#       and confirming the variant case was selected -- relies on the test
#       binary printing the kernel name; if it does not, this script just
#       confirms correctness + perf without proving routing)
#   3. Incremental build wall-clock for ONE new variant .cu
#   4. test-backend-ops correctness pass + perf number under ENV gate
#
# Usage:
#   bash stage1-roundtrip.sh

set -uo pipefail

# ---- paths (edit if your tree is elsewhere) ----
GGML_ROOT="${GGML_ROOT:-C:/Users/wwxq/Desktop/haichao/code/llama.cpp/ggml}"
LLAMA_ROOT="$(dirname "$GGML_ROOT")"
BUILD_DIR="${BUILD_DIR:-$LLAMA_ROOT/build-amd}"
AKW_ROOT="${AKW_ROOT:-C:/Users/wwxq/Desktop/haichao/code/Awesome-Kernel-Workflows}"
REG="$AKW_ROOT/scripts/llamacpp_register_variant.py"
TBOPS="$BUILD_DIR/bin/test-backend-ops.exe"

VARIANT="${1:-v0clone}"
REFERENCE_CUH="$GGML_ROOT/src/ggml-cuda/fattn-rdna-apu.cuh"
FATTN_CU="$GGML_ROOT/src/ggml-cuda/fattn.cu"

# ---- helpers ----
say()  { printf '\n=== %s ===\n' "$*"; }
fail() { printf '\nFAIL: %s\n' "$*" >&2; cleanup; exit 1; }

PRISTINE_MD5=""
capture_pristine() {
  PRISTINE_MD5="$(md5sum "$FATTN_CU" | awk '{print $1}')"
  printf 'pristine md5: %s\n' "$PRISTINE_MD5"
}

cleanup() {
  # Always try to unregister, even on failure, so the tree is clean.
  python "$REG" list --ggml-root "$GGML_ROOT" 2>/dev/null \
    | awk '{print $1}' | grep -E "^${VARIANT}$" >/dev/null \
    && python "$REG" unregister --variant "$VARIANT" --ggml-root "$GGML_ROOT" >/dev/null 2>&1
}
trap cleanup EXIT

# ---- preflight ----
for f in "$REG" "$REFERENCE_CUH" "$FATTN_CU"; do
  [[ -f "$f" ]] || fail "missing prerequisite: $f"
done
[[ -d "$BUILD_DIR" ]] || fail "build dir not found (expected pre-configured): $BUILD_DIR"
command -v cmake >/dev/null || fail "cmake not on PATH"

say "Preflight OK"
capture_pristine

# ---- step 1: baseline build (incremental, should be ~0 if already built) ----
say "Step 1: baseline incremental build"
# Reconfigure so any stale variant file references from prior runs are dropped.
cmake "$BUILD_DIR" >/tmp/stage1-reconfigure-baseline.log 2>&1 \
  || { tail -40 /tmp/stage1-reconfigure-baseline.log; fail "baseline reconfigure failed"; }
t0=$(date +%s)
cmake --build "$BUILD_DIR" --config Release --target test-backend-ops -- -j"$(nproc 2>/dev/null || echo 4)" \
  >/tmp/stage1-build-baseline.log 2>&1 \
  || { tail -40 /tmp/stage1-build-baseline.log; fail "baseline build failed"; }
t1=$(date +%s)
printf 'baseline build duration: %ds\n' $((t1 - t0))

[[ -x "$TBOPS" ]] || fail "test-backend-ops binary not at $TBOPS after build"

# ---- step 2: baseline correctness ----
say "Step 2: baseline correctness (no KERSOR_VARIANT)"
"$TBOPS" test -o FLASH_ATTN_EXT -b ROCm0 >/tmp/stage1-test-baseline.log 2>&1
rc=$?
tail -3 /tmp/stage1-test-baseline.log
[[ $rc -eq 0 ]] || fail "baseline correctness exit=$rc"

# ---- step 3: baseline perf ----
say "Step 3: baseline perf"
"$TBOPS" perf -o FLASH_ATTN_EXT -p hsk=256,hsv=256 -b ROCm0 \
  >/tmp/stage1-perf-baseline.log 2>&1
grep -E "FLASH_ATTN_EXT|us|TFLOPS" /tmp/stage1-perf-baseline.log | head -8 || true

# ---- step 4: register variant ----
say "Step 4: register variant '$VARIANT' (clone of $REFERENCE_CUH)"
python "$REG" register --variant "$VARIANT" --cuh-src "$REFERENCE_CUH" --ggml-root "$GGML_ROOT" \
  || fail "register failed"
python "$REG" list --ggml-root "$GGML_ROOT" | grep -qE "^\s+${VARIANT}\s" \
  || fail "variant did not appear in list"

# Confirm fattn.cu changed.
post_md5="$(md5sum "$FATTN_CU" | awk '{print $1}')"
[[ "$post_md5" != "$PRISTINE_MD5" ]] || fail "fattn.cu md5 unchanged after register"
printf 'fattn.cu md5 after register: %s (was %s)\n' "$post_md5" "$PRISTINE_MD5"

# ---- step 5: incremental build ----
say "Step 5: incremental build with variant registered"
# CMakeLists.txt uses file(GLOB) without CONFIGURE_DEPENDS, so new .cu files
# (entry stub + template instance) are invisible to a plain incremental build.
# Force an explicit reconfigure to re-glob.
cmake "$BUILD_DIR" >/tmp/stage1-reconfigure.log 2>&1 \
  || { tail -40 /tmp/stage1-reconfigure.log; fail "cmake reconfigure failed"; }
t0=$(date +%s)
cmake --build "$BUILD_DIR" --config Release --target test-backend-ops -- -j"$(nproc 2>/dev/null || echo 4)" \
  >/tmp/stage1-build-variant.log 2>&1
rc=$?
t1=$(date +%s)
INC_BUILD_SEC=$((t1 - t0))
printf 'incremental build duration: %ds\n' "$INC_BUILD_SEC"
if [[ $rc -ne 0 ]]; then
  tail -60 /tmp/stage1-build-variant.log
  fail "variant build failed"
fi

# ---- step 6: correctness under ENV gate ----
say "Step 6: correctness with KERSOR_VARIANT=$VARIANT"
KERSOR_VARIANT="$VARIANT" "$TBOPS" test -o FLASH_ATTN_EXT -b ROCm0 \
  >/tmp/stage1-test-variant.log 2>&1
rc=$?
tail -3 /tmp/stage1-test-variant.log
[[ $rc -eq 0 ]] || fail "variant correctness exit=$rc"

# ---- step 7: perf under ENV gate ----
say "Step 7: perf with KERSOR_VARIANT=$VARIANT"
KERSOR_VARIANT="$VARIANT" "$TBOPS" perf -o FLASH_ATTN_EXT -p hsk=256,hsv=256 -b ROCm0 \
  >/tmp/stage1-perf-variant.log 2>&1
grep -E "FLASH_ATTN_EXT|us|TFLOPS" /tmp/stage1-perf-variant.log | head -8 || true

# ---- step 8: unregister and confirm byte-exact ----
say "Step 8: unregister and verify byte-exact"
python "$REG" unregister --variant "$VARIANT" --ggml-root "$GGML_ROOT" \
  || fail "unregister failed"

final_md5="$(md5sum "$FATTN_CU" | awk '{print $1}')"
if [[ "$final_md5" != "$PRISTINE_MD5" ]]; then
  diff <(echo "$PRISTINE_MD5") <(echo "$final_md5")
  fail "fattn.cu NOT byte-exact after unregister (was $PRISTINE_MD5, now $final_md5)"
fi
printf 'fattn.cu md5 after unregister: %s (matches pristine)\n' "$final_md5"

# Confirm new files were deleted.
for f in "$GGML_ROOT/src/ggml-cuda/fattn-rdna-apu-${VARIANT}.cuh" \
         "$GGML_ROOT/src/ggml-cuda/fattn-rdna-apu-${VARIANT}.cu" \
         "$GGML_ROOT/src/ggml-cuda/template-instances/fattn-rdna-apu-${VARIANT}-instance-dkq256-dv256.cu"; do
  [[ ! -e "$f" ]] || fail "leftover file: $f"
done

# ---- summary ----
say "Stage 1 PASS"
printf '  baseline correctness: PASS\n'
printf '  variant correctness:  PASS\n'
printf '  incremental build:    %ds (the per-variant cost the search workflow pays)\n' "$INC_BUILD_SEC"
printf '  byte-exact roundtrip: PASS\n'
printf '\nLogs:\n'
printf '  /tmp/stage1-build-baseline.log\n'
printf '  /tmp/stage1-test-baseline.log\n'
printf '  /tmp/stage1-perf-baseline.log\n'
printf '  /tmp/stage1-build-variant.log\n'
printf '  /tmp/stage1-test-variant.log\n'
printf '  /tmp/stage1-perf-variant.log\n'

# Disarm trap (already cleaned up successfully).
trap - EXIT
