---
title: WGMMA SMEM-operand pipelines
tags: [wgmma, pipeline_depth, smem_staging, mma]
arch: [sm90]
expected_gain_pct: [5, 30]
---

# WGMMA with SMEM operands (sm90)

## What
Drive the Hopper tensor cores with `wgmma.mma_async.m64nNk16` reading B (and
usually A) directly from swizzled SMEM via matrix descriptors, overlapping MMA
groups with the next tile's TMA loads.

## When (NCU signals)
- SOL-SM% well below peak with tensor instructions present: MMA issue gaps.
- `mma.sync` (Ampere-style) or small-shape HMMA in the SASS — legacy path.
- High register pressure from register-operand MMA forcing low occupancy.

## How
1. Pick the largest N that fits: accumulator regs/thread = N/2 for FP32
   accum (m64n128 → 64 regs, m64n256 → 128). Bigger N amortizes issue and
   descriptor overhead; balance vs [register-pressure-occupancy].
2. Encode SMEM descriptors: base addr (matrix start), leading/stride byte
   offsets, swizzle mode matching the TMA descriptor's swizzle.
3. Issue pattern per K-tile:
   `wgmma.fence.sync.aligned` → repeat `wgmma.mma_async` over K fragments →
   `wgmma.commit_group.sync.aligned` → consume when `wgmma.wait_group.sync.aligned N`
   admits (N = groups allowed in flight, typically 1-2).
4. Keep ≥2 K-tiles in flight: while group g computes, TMA fills stage g+1
   ([warp-specialization-pingpong] owns that handoff).
5. Epilogue: accumulators are in registers; stage through SMEM for coalesced
   TMA store rather than per-thread global stores.

## Pitfalls
- Forgetting `wgmma.fence` before first accumulator use or after register
  epilogue edits → garbage accumulators only under specific schedules.
- `wait_group 0` after every commit serializes MMA with loads — the most
  common silent perf bug; keep one group outstanding.
- Descriptor swizzle must equal the actual SMEM layout written by TMA; a
  mismatch is numerically wrong only for some tile coordinates (test all
  shapes, not just the default).
- k-major vs mn-major fragment layouts differ for FP8 (k32); transposed
  operands need different descriptors, not pointer arithmetic.

## Interactions
- Feeds from [tma-bulk-async-copy]; register budget set via setmaxnreg in
  [warp-specialization-pingpong]. On sm100 prefer [tcgen05-tmem-basics]
  (wgmma runs but underuses Blackwell tensor cores).
