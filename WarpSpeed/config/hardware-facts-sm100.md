# Hardware facts — sm100 (B200 / Blackwell)

Constants for the implementor. NOTE: numbers marked (~) should be re-verified
against the CUDA 12.8+ PTX ISA / tuning guide for your exact part before being
load-bearing in a hypothesis.

## SM / memory
- B200: dual-die, 192 GB HBM3e, ~8 TB/s aggregate HBM bandwidth (~).
- Shared memory per SM: 228 KB (same carve-out model as Hopper); max 227 KB/block.
- 64K 32-bit registers per SM, 255 regs/thread max, 2048 threads/SM, 64 warps/SM.
- L2 ~126 MB total across dies (~); cross-die traffic is NOT free - data
  locality per die matters for persistent kernels.
- FP4/FP6/FP8 tensor formats; dense FP8 ~4.5 PFLOPS per GPU (~).

## tcgen05 (5th-gen tensor core) + TMEM
- New `tcgen05.*` instruction family replaces wgmma as the preferred MMA path:
  `tcgen05.mma` sources accumulators (and optionally A) from TMEM (Tensor
  Memory), a dedicated 256 KB per-SM on-chip memory (~), NOT registers.
- TMEM is allocated via `tcgen05.alloc`, accessed via `tcgen05.ld/st`,
  freed via `tcgen05.dealloc`. Accumulating in TMEM frees the register file -
  the Hopper register-pressure wall around big-N accumulation largely moves
  to TMEM capacity planning instead.
- `tcgen05.mma` is issued by ONE thread (single-thread MMA issue), completion
  signaled through mbarrier (`tcgen05.commit` -> mbarrier arrive). Warp
  specialization shifts from "feed wgmma per warpgroup" to "one MMA-issue
  warp + TMA warps + epilogue warpgroups draining TMEM".
- 2-SM MMA pairs (CTA-pair MMA): two CTAs in a cluster cooperate on one MMA
  with shared operands - doubles effective MMA width; requires even cluster
  and peer-CTA synchronization.

## TMA / clusters
- TMA model carries over from sm90 (cp.async.bulk.tensor + mbarrier expect-tx),
  plus weight-stationary patterns benefit from larger L2 and TMEM staging.
- Clusters: portable max 8 CTAs as on Hopper (verify 16 opt-in support on your
  driver); DSMEM and TMA multicast unchanged in model.

## Migration notes from sm90 kernels
- wgmma still executes on sm100 but leaves tensor-core throughput on the
  table vs tcgen05; treat "port hot MMA loop to tcgen05+TMEM" as its own
  strategy tag (tcgen05), not an automatic win - TMEM ld/st epilogue costs
  can eat the gain for small tiles.
- setmaxnreg interacts differently: consumer warpgroups need fewer registers
  when accumulating in TMEM; rebalance producer/consumer register splits.
- Wave quantization: SM count differs per part (~148/die, verify); recompute
  persistent-kernel grid sizing rather than copying H100 numbers.
