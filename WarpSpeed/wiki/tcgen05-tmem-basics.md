---
title: tcgen05 + TMEM fundamentals
tags: [tcgen05, tmem, mma, cta_pair]
arch: [sm100]
expected_gain_pct: [10, 40]
---

# tcgen05 / Tensor Memory (Blackwell)

## What
Port the hot MMA loop from wgmma (register accumulators, per-warpgroup issue)
to `tcgen05.mma` (TMEM accumulators, single-thread issue) — the native
Blackwell tensor-core path. Optionally widen to 2-CTA MMA pairs.

## When (NCU signals)
- Running an sm90-style WGMMA kernel on sm100: tensor SOL% plateaus well
  below Blackwell peak; register pressure limits tiles ([register-pressure-occupancy]).
- Epilogue-light GEMM-like kernels with large accumulator footprints.

## How
1. Allocate TMEM: `tcgen05.alloc.cta_group::1.sync.aligned` (columns sized to
   the accumulator tile); plan capacity like SMEM — it is the new scarce
   resource (~256 KB/SM, verify per part).
2. Issue: ONE elected thread per CTA(-pair) runs the
   `tcgen05.mma.cta_group::N.kind::*` loop over K; operands stream from SMEM
   (TMA-fed, same [tma-bulk-async-copy] machinery); completion via
   `tcgen05.commit` → mbarrier.
3. Restructure warp roles: TMA producer warps unchanged; the old MMA
   warpgroups become (a) one MMA-issue warp and (b) epilogue warps that
   `tcgen05.ld` TMEM → registers → SMEM → TMA store, overlapped with the next
   tile's MMA.
4. 2-CTA pairs (`cta_group::2`): cluster=2, peer CTA supplies half the
   operands; doubles effective tile width for the same per-CTA SMEM.
5. Free registers: drop consumer `setmaxnreg.inc` budgets; producers can
   often grow instead (more in-flight descriptors).

## Pitfalls
- TMEM is allocate/deallocate managed — leaking allocations across persistent
  work items exhausts it after N items (works in short tests, dies in long
  runs: run the full-shape sweep).
- `tcgen05.ld` epilogue bandwidth can become the new bottleneck for small
  tiles; if epilogue% grows in the profile, the port was premature.
- Mixed wgmma+tcgen05 in one kernel is legal but schedules poorly; convert
  the whole hot loop or none.
- Verify instruction availability for your CUDA version at init (compile
  probe), not at round 5.

## Interactions
- Supersedes [wgmma-smem-pipelines] on sm100. Reshapes
  [warp-specialization-pingpong] roles. Pairs with [cluster-dsmem] for
  cta_group::2. FP8/FP4 variants follow [fp8-register-accumulation] with
  TMEM-side promotion.
