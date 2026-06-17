---
title: Register pressure & occupancy tuning
tags: [regs, occupancy, spills, setmaxnreg]
arch: [sm90, sm100]
expected_gain_pct: [2, 15]
---

# Register pressure / occupancy

## What
Trade registers, SMEM, and co-resident warps deliberately instead of letting
the compiler pick: `__launch_bounds__`/`-maxrregcount` to cap, `setmaxnreg`
to rebalance between warpgroups, accumulator-shape choices to control demand.

## When (NCU signals)
- LaunchStats `Registers Per Thread` ≥ 168 with Occupancy `Block Limit
  Registers` the binding limit, AND WarpStateStats shows latency-bound stalls
  (`Long Scoreboard`) that more warps would hide.
- OR the inverse: spills (`Local Memory` traffic in MemoryWorkloadAnalysis,
  LDL/STL in SASS) after forcing occupancy too high.
- Eligible warps/scheduler < 1 while issued ≈ eligible: starved schedulers.

## How
1. Decide the regime first (from NCU, not intuition): latency-bound kernels
   want more warps; tensor-throughput kernels often run BEST at 1-2 big
   warpgroups/SM with 224-240 regs each and occupancy ~12-25%. Do not
   "fix" low occupancy on an MMA-saturated kernel.
2. Cap with `__launch_bounds__(threads, minBlocksPerSM)` — prefer this over
   global -maxrregcount (per-kernel, survives in source, reviewable).
3. Warp-specialized kernels: `setmaxnreg.dec` producers to 24-40,
   `.inc` consumers to the WGMMA accumulator need (N/2 + working set).
   Total per CTA must fit the 64K-reg file or launch fails/occupancy drops.
4. Reduce demand structurally: smaller N accumulator tiles, re-materialize
   cheap values, hoist invariant scales to SMEM/constant, split mega-kernels
   into phases. On sm100, move accumulators to TMEM ([tcgen05-tmem-basics]).
5. Re-measure occupancy AND latency: report both; occupancy is a means.

## Pitfalls
- Spill cliff: each spilled 128B costs L1/L2 round-trips inside the hot loop;
  a 2% reg cut that introduces spills is a net loss — check Local Memory
  traffic after every cap change.
- `setmaxnreg` totals violating launch bounds fail at launch only on some
  driver versions — assert at init.
- Occupancy gains capped by another limiter (SMEM, blocks) do nothing; read
  ALL `Block Limit *` rows before predicting a gain mechanism.

## Interactions
- Couples tightly with [wgmma-smem-pipelines] (accumulator N) and
  [warp-specialization-pingpong] (per-warpgroup budgets). Pipeline-depth
  increases consume SMEM that may force tile shrink — co-tune, and record the
  chosen point as an explicit assumption tag (e.g. `regs224_occ2cta`).
