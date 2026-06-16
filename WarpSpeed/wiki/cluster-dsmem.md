---
title: Thread-block clusters & distributed SMEM
tags: [cluster, dsmem, multicast, tma]
arch: [sm90, sm100]
expected_gain_pct: [3, 12]
---

# Clusters + DSMEM + TMA multicast

## What
Group 2-8 CTAs into a cluster; share tiles through distributed shared memory
(DSMEM) and load shared operands ONCE via TMA multicast into all members'
SMEM, cutting DRAM traffic for operand reuse patterns (e.g. B tile shared
across N-split CTAs).

## When (NCU signals)
- DRAM% near roofline while the same global tiles are demonstrably loaded by
  multiple CTAs (L2 hit rate high but DRAM still saturated → L2 not enough).
- SMEM capacity, not bandwidth, limits tile size per CTA — pooling across the
  cluster effectively doubles/quadruples usable SMEM.

## How
1. Launch: `cudaLaunchKernelEx` with `cudaLaunchAttributeClusterDimension`
   (e.g. 2×1×1); `__cluster_dims__` for static. Portable max 8 CTAs.
2. TMA multicast: set the multicast CTA mask on `cp.async.bulk.tensor` so one
   issue fills every member's stage buffer; each member's mbarrier gets the
   expect-tx.
3. DSMEM access: `cluster.map_shared_rank(ptr, rank)` to ld/st/atomic a peer
   CTA's SMEM; synchronize with `cluster.sync()` or peer mbarriers (arrive on
   remote barrier via mapped address).
4. Partition work so peers consume disjoint slices of the shared tile
   (split-N or split-K across cluster ranks).

## Pitfalls
- Cluster occupancy: the whole cluster must co-reside on one GPC; big SMEM ×
  8 CTAs may simply not schedule (launch failure or serialization) — check
  Occupancy `Block Limit Cluster`.
- `cluster.sync()` is expensive (~µs scale); prefer mbarrier handshakes per
  stage over full-cluster syncs in inner loops.
- DSMEM atomics to a hot peer address serialize across the GPC fabric.
- Exiting CTAs early (tail items) while peers still read their SMEM is UB —
  persistent-kernel drains must keep all ranks alive until cluster-wide done.

## Interactions
- Extends [tma-bulk-async-copy]; halves the producer traffic in
  [warp-specialization-pingpong] for shared-operand GEMMs.
- On sm100, 2-CTA MMA pairs make cluster=2 the natural unit; see
  [tcgen05-tmem-basics].
