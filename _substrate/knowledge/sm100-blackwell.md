# Blackwell sm_100 — generator knowledge pack

Single source of truth for CUDA/Triton kernel generators targeting **sm_100 /
Blackwell (B200)**. Consolidated from WarpSpeed's `config/hardware-facts-sm100.md`
+ `wiki/tcgen05-tmem-basics.md` so every generator (not just WarpSpeed) emits
the correct native intrinsics. The 017 advisor blacklisted every CUDA generator
on B200 because they emitted Hopper `wgmma` here (AWK #53).

> **Note (architecture):** arch-mismatch *gating* is enforced vendor-neutrally
> at the KerSor injection layer (KerSor #70, `resolve-compute-lever.sh`), not by
> a workflow-side lint. This pack is reference material consumed by KerSor's
> dispatch and by humans — the rule below is the NVIDIA instance of that
> vendor-neutral gate.

## The one rule that matters
On sm_100 the native tensor-core MMA path is **`tcgen05.mma`** (5th-gen tensor
core, TMEM-backed accumulators) — NOT Hopper's `wgmma`. `wgmma` still *executes*
on sm_100 but leaves Blackwell tensor-core throughput on the table; treat it as
a porting smell, not a strategy. Mirrors going the other way: `tcgen05.*` does
NOT exist on sm_90/Hopper — emitting it there is a hard miscompile.

| target arch | native MMA family | do NOT emit |
|-------------|-------------------|-------------|
| sm_100 (Blackwell) | `tcgen05.mma` (+ TMEM `tcgen05.alloc/ld/st/dealloc`, mbarrier commit) | `wgmma.*`, `mma.async` (Hopper) |
| sm_90 (Hopper)     | `wgmma.*`, `mma.async` | `tcgen05.*` (Blackwell-only) |

## tcgen05 / TMEM essentials
- `tcgen05.mma` sources accumulators (and optionally A) from **TMEM** — a
  dedicated 256 KB/SM on-chip memory (~reverify on your part), NOT registers.
- Allocate/free: `tcgen05.alloc` / `tcgen05.dealloc`; load/store: `tcgen05.ld` / `tcgen05.st`.
- `tcgen05.mma` is issued by ONE thread (single-thread MMA issue); completion is
  signaled through an mbarrier (`tcgen05.commit` → mbarrier arrive).
- 2-SM MMA pairs (CTA-pair MMA): two CTAs in a cluster cooperate on one MMA,
  doubling effective MMA width — requires an even cluster + peer-CTA sync.

## SM / memory facts (B200)
- Dual-die, 192 GB HBM3e, ~8 TB/s aggregate HBM (~). L2 ~126 MB total; cross-die
  traffic is NOT free — data locality per die matters for persistent kernels.
- Shared memory per SM: 228 KB (same carve-out model as Hopper). 64K 32-bit
  registers/SM, 255 regs/thread max, 2048 threads/SM, 64 warps/SM.
- FP4/FP6/FP8 tensor formats; dense FP8 ~4.5 PFLOPS/GPU (~).

## Migration notes (sm_90 → sm_100)
- Porting the hot MMA loop to `tcgen05`+TMEM is its own strategy tag, not an
  automatic win — TMEM ld/st epilogue costs can eat the gain for small tiles.
- With TMEM accumulators, consumer warpgroups need fewer registers; rebalance
  producer/consumer `setmaxnreg` splits vs the Hopper layout.
- Recompute persistent-kernel grid sizing (SM count differs per part, ~148/die)
  rather than copying H100 numbers.

Numbers marked (~) should be re-verified against the CUDA 12.8+ PTX ISA /
tuning guide for the exact part before being load-bearing in a hypothesis.
