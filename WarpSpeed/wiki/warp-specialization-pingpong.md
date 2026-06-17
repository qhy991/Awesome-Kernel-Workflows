---
title: Warp specialization with ping-pong buffers
tags: [warp_spec, pingpong, producer_consumer, pipeline_depth, setmaxnreg]
arch: [sm90, sm100]
expected_gain_pct: [5, 25]
---

# Warp specialization + ping-pong (producer/consumer pipelines)

## What
Split the CTA into producer warps (issue TMA loads) and consumer warpgroups
(run MMA + epilogue), communicating through a ring of K SMEM buffer stages
guarded by paired full/empty mbarriers. "Ping-pong" = 2 stages; depth 3-4
hides longer latencies at SMEM cost.

## When (NCU signals)
- `Stall Long Scoreboard` or `Stall Wait` dominant in WarpStateStats while
  SOL-SM% is mid-range: compute waits on memory it could overlap.
- Tensor-pipe utilization notably below SOL-SM%; MMA bubbles between tiles.
- Enough SMEM headroom for ≥2 stages of (A tile + B tile).

## How
1. Partition: warpgroup 0 = producer (often just 1 elected thread issues TMA),
   warpgroups 1..n = consumers. Branch on `threadIdx / 128`.
2. Two mbarrier arrays: `full[K]` (producer arrives with expect-tx, consumers
   wait) and `empty[K]` (consumers arrive after use, producer waits).
3. Producer loop: wait `empty[i]` (phase p), issue TMA into stage i, arrive
   `full[i]`. Consumer loop: wait `full[i]`, MMA from stage i, arrive `empty[i]`.
4. Rebalance registers with `setmaxnreg`: producers `.dec` to 24-40 regs,
   consumers `.inc` to 224-240 — this is what makes big-N WGMMA accumulators
   fit (see [register-pressure-occupancy]).
5. Choose depth: stages = ceil(load_latency / mma_time_per_tile), capped by
   SMEM. Tag the choice explicitly (e.g. `pingpong_depth3`) — it is a baked
   assumption later experiments may need to ablate.

## Pitfalls
- Phase-parity bugs: each side tracks its own phase bit per stage; off-by-one
  = deadlock (timeout) or silent reuse (wrong results under load only).
- `setmaxnreg` requires the launch-bounds register budget to cover the SUM of
  warpgroup allocations; violating it silently caps occupancy.
- Producer must be excluded from MMA math entirely or register rebalancing
  breaks its spills.
- Tail tiles: drain loop must arrive on `empty` for unused stages or the next
  kernel launch's producer hangs (persistent kernels especially).

## Interactions
- Requires [tma-bulk-async-copy]. Pairs with [wgmma-smem-pipelines] (sm90)
  or tcgen05 single-thread MMA issue (sm100), where the consumer side becomes
  an MMA-issue warp + TMEM drain warps ([tcgen05-tmem-basics]).
- With [persistent-kernel-scheduling], pipeline state must be re-armed per
  work item; the empty-barrier drain is the common bug site.
