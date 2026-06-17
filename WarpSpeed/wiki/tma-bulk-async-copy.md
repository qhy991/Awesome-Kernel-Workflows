---
title: TMA bulk async copy
tags: [tma, async_copy, mbarrier, smem_staging]
arch: [sm90, sm100]
expected_gain_pct: [3, 15]
---

# TMA bulk async copy

## What
Replace per-thread `ld.global`/`cp.async.cg` tiling with descriptor-driven
`cp.async.bulk.tensor` transfers issued by ONE thread. The TMA engine handles
address generation, bounds, and swizzle; completion is signaled to an mbarrier
with expect-tx byte counts.

## When (NCU signals)
- High `Issued Warp Per Scheduler` with large fraction of LSU instructions;
  address-generation integer instructions crowding the pipe.
- `Stall Long Scoreboard` dominant while DRAM throughput is below roofline —
  the copy loop, not the memory system, is the bottleneck.
- Kernel already tiles via SMEM with predictable (affine) layouts.

## How
1. Host: build `CUtensorMap` via `cuTensorMapEncodeTiled` (global shape,
   strides, box size ≤256 in the inner dim, swizzle 32/64/128B matched to the
   SMEM consumer layout).
2. Device: `elect.one.sync` thread issues
   `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`,
   then `mbarrier.arrive.expect_tx` with the exact byte count.
3. Consumers `mbarrier.try_wait.parity` on the stage's barrier; flip phase
   bit per buffer slot.
4. Stores back: `fence.proxy.async.shared::cta` BEFORE
   `cp.async.bulk.tensor` store, then `commit_group` + `wait_group`.

## Pitfalls
- Missing the async-proxy fence before TMA stores reads stale SMEM — races
  that compute-sanitizer often misses; reviewers must check fences manually.
- expect-tx byte count must equal the actual transfer exactly or the barrier
  never (or early) trips — hangs show up as timeout, not wrong answers.
- 128B SMEM alignment for boxes; misaligned boxes silently fall back slower.
- One barrier per pipeline stage; reusing one barrier across stages with
  >1 in-flight transfer corrupts phase tracking.

## Interactions
- Pairs with [warp-specialization-pingpong] (producer warps own TMA issue).
- Enables [smem-swizzle-bank-conflicts] fixes for free (swizzle in descriptor).
- On sm100, identical model; combine with TMEM staging for weight-stationary.
