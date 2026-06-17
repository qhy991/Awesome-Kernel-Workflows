---
title: L2 persistence policy & prefetch
tags: [l2_persist, prefetch, cp_async_bulk, cache_policy]
arch: [sm90, sm100]
expected_gain_pct: [1, 8]
---

# L2 persistence & prefetch

## What
Steer the L2: pin hot, reused regions (weights, lookup tables, work queues)
with persistence windows; bias eviction with cache hints on loads/stores;
prefetch upcoming tiles into L2 ahead of the pipeline.

## When (NCU signals)
- L2 Hit Rate low (<50%) while the working set of the REUSED operand would
  fit in a fraction of L2 (50 MB on H100): thrashing by streaming data.
- DRAM% near peak with significant read amplification of one operand.
- Multi-kernel sequences re-reading what the previous kernel just wrote.

## How
1. Persistence window (host): `cudaStreamSetAttribute` with
   `cudaStreamAttrValue.accessPolicyWindow` {base_ptr, num_bytes,
   hitRatio≈0.6-1.0, hitProp=Persisting, missProp=Streaming}; size ≤
   `persistingL2CacheMaxSize` (set via `cudaDeviceSetLimit`).
2. Per-access hints (device): `__ldg`/`__stcs` style or PTX `ld.global.L2::
   evict_last` for the reused operand, `evict_first` for streamed output;
   `cp.async.bulk` accepts an L2 cache-policy descriptor operand.
3. Prefetch: `cp.async.bulk.prefetch.L2.global` for the NEXT work item's
   tiles during the current item's MMA (persistent kernels make the "next
   item" known early).
4. Reset between phases: `cudaCtxResetPersistingL2Cache()` or the streaming
   data inherits the pinned region.

## Pitfalls
- hitRatio=1.0 on a window larger than the persisting carve-out silently
  thrashes the pinned set itself; undersize the window, not oversize.
- Pinning helps only when the OTHER traffic was evicting it: confirm the
  baseline L2 hit-rate story in NCU before predicting a gain (this tag has
  historically the largest predicted-vs-achieved gaps).
- Hints are requests, not guarantees; SASS may drop them at -O3 reorders —
  verify with `l2_persist` counters, not source inspection.

## Interactions
- Prefetch slots naturally into [persistent-kernel-scheduling] work loops.
- On B200, cross-die L2 partitioning means windows are per-die; pin the
  operand near its consumers ([tcgen05-tmem-basics] weight-stationary).
