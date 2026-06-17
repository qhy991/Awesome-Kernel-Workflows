---
title: FP8 accumulation strategies
tags: [fp8, fp8_reg_accum, accuracy, mma]
arch: [sm90, sm100]
expected_gain_pct: [5, 20]
---

# FP8 MMA + controlled accumulation

## What
Run the hot MMA loop in FP8 (e4m3 operands) for 2× tensor throughput while
controlling accuracy: periodic promotion of partial sums into full-FP32
register (sm90) or TMEM (sm100) accumulators every C k-steps — the
`fp8_reg_accum` family of strategies.

## When
- Kernel is tensor-throughput-bound at FP16/BF16 (SOL-SM% ≥ 80 dominated by
  HMMA) and the model/tolerance budget admits FP8 inputs.
- harness tolerances.json has an explicit, human-reviewed fp8 entry — if it
  does not, STOP: that is a harness gap, not an implementation choice.

## How
1. Quantize operands to e4m3 with per-tile (or per-channel) scales staged in
   SMEM; dequantize once at epilogue: out = acc × sa × sb.
2. sm90: `wgmma.mma_async...k32` FP8 shapes; hardware accumulates with
   limited intermediate precision over long K. Chunk: every C k-tiles
   (C≈4-8), add the group's accumulator into a persistent FP32 register
   accumulator and zero the group accumulator.
3. Pick C by measured error against the harness across ALL shapes (long-K
   shapes are the accuracy-critical ones), then bake C as an assumption tag
   (`fp8_reg_accum_c4`).
4. sm100: tcgen05 FP8 paths accumulate in TMEM; the chunking trick becomes a
   TMEM-to-TMEM promotion or epilogue-side compensation — re-derive C, do
   not copy the sm90 value.

## Pitfalls
- Accuracy failures appear ONLY on large-K shapes; a green run on the default
  shape proves nothing — the harness's shape sweep is the gate.
- Scale staging in registers bloats consumer register budgets; put scales in
  SMEM and load per-tile ([register-pressure-occupancy]).
- e5m2 for activations is rarely worth it (precision cliff); default e4m3
  both sides unless dynamic range demonstrably requires otherwise.
- Throughput gain ≈ 2× only if the kernel was MMA-bound; memory-bound FP8
  mostly saves bandwidth, not time — predict the mechanism accordingly.

## Interactions
- Doubles effective K-rate, which usually requires +1 pipeline stage in
  [warp-specialization-pingpong] to keep fed.
- Classic rewind suspect: if downstream strategies stall on accuracy-driven
  constraints, ablate `fp8_reg_accum` before abandoning the subtree.
