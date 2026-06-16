---
title: Persistent kernels & work-queue scheduling
tags: [persistent_kernel, work_queue, wave_quant, occupancy]
arch: [sm90, sm100]
expected_gain_pct: [2, 12]
---

# Persistent kernels & wave quantization

## What
Launch exactly (#SMs × blocks-per-SM) CTAs once; each loops over a queue of
tile work items (static stride or atomic counter) instead of launching one
CTA per tile. Kills per-CTA launch/epilogue overhead and tail-wave waste.

## When (NCU signals)
- `Waves Per SM` fractional part is large (e.g. 2.3 waves: the 0.3-wave tail
  idles most SMs); or many small launches dominate (LaunchStats kernel count).
- Grid >> SM count with short-lived CTAs; achieved occupancy fine but
  duration dominated by ramp/tail.
- Inter-tile reuse exists (weights stay resident across items).

## How
1. Grid = SM count × max co-resident CTAs (from occupancy calc, NOT hardcoded
   — read SM count at runtime; H100 SXM=132, B200 differs per die/part).
2. Work distribution: `for (item = blockIdx.x; item < n_items; item += gridDim.x)`
   for uniform costs; atomic ticket counter for skewed costs.
3. Keep stage-invariant data (scales, descriptors, weight tiles) loaded once
   before the work loop.
4. Re-arm the pipeline between items: reset/advance mbarrier phases, do NOT
   re-init barriers mid-flight (init only at kernel start).
5. For multi-shape queues, order items large→small to minimize tail skew.

## Pitfalls
- Hardcoded SM counts silently break on other parts (and on B200 dies);
  derive from `cudaDevAttrMultiProcessorCount`.
- Atomic counters in L2 become a hotspot past ~thousands of items/ms; switch
  to strided or hierarchical distribution.
- Persistent CTAs pin the GPU: a deadlocked pipeline hangs the whole device —
  always run under gpu_run timeouts.
- Occupancy regression risk: persistent body unions the register/SMEM needs
  of ALL phases; check `Block Limit` lines in Occupancy section after.

## Interactions
- Baked-assumption heavy: tag `persistent_kernel` on every descendant — this
  is a classic suspect for post-mortem ablation when later strategies stall.
- Combines with [warp-specialization-pingpong]; drain/re-arm logic is the
  interaction bug site. L2-resident queues pair with [l2-policy-and-prefetch].
