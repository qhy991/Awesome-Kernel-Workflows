---
title: SMEM swizzle & bank-conflict elimination
tags: [swizzle, bank_conflict, layout, smem_staging]
arch: [sm90, sm100]
expected_gain_pct: [2, 10]
---

# Shared-memory swizzling / bank conflicts

## What
Reorder SMEM layout (XOR-swizzle or padding) so that the access pattern of
both the writer (TMA/cp.async) and the reader (MMA fragment loads, epilogue)
hits all 32 banks per phase.

## When (NCU signals)
- MemoryWorkloadAnalysis: `Shared Memory` bank conflicts counter > ~5% of
  wavefronts; L1/TEX throughput high while SM% stalls on `Stall Short Scoreboard`.
- Transposed reads (column access of row-major tiles) or strided epilogues.

## How
1. Prefer hardware swizzle via descriptors: pick 32/64/128B swizzle in the
   `CUtensorMap` AND the matching mode in the WGMMA/ldmatrix descriptor —
   then no manual index math at all.
2. Manual fallback: `smem[row][col ^ ((row & 7) << s)]` XOR patterns; choose
   s so a 128B phase covers all banks for your element size.
3. Padding (`[rows][cols+1]`) costs SMEM capacity (hurts pipeline depth);
   use only when XOR breaks vectorized 16B accesses.
4. Verify with NCU: conflicts counter → ~0, and check `ldmatrix`/`stmatrix`
   throughput recovered.

## Pitfalls
- Swizzle must be consistent across EVERY producer and consumer of the same
  buffer (TMA write, ldmatrix read, epilogue store) — a single unswizzled
  reader is numerically wrong, not slow.
- 16B vector alignment: XOR patterns must preserve 16B groups or vectorized
  loads decompose (slower, still correct — easy to misattribute).
- Changing tile sizes changes which swizzle mode is conflict-free; re-derive
  when [wgmma-smem-pipelines] N/K tiles change.

## Interactions
- Free when adopted together with [tma-bulk-async-copy] (descriptor swizzle).
- Bank-conflict fixes often unlock the gains predicted (but not achieved) by
  pipeline-depth experiments — check this first when prediction gaps cluster
  on pipeline tags.
