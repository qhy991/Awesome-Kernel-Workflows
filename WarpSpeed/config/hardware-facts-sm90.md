# Hardware facts — sm90 (H100 / Hopper)

Authoritative constants the implementor must respect. (SXM5 80GB variant;
PCIe parts have fewer SMs and lower clocks.)

## SM / occupancy limits
- 132 SMs (SXM5). 4 SM sub-partitions (warp schedulers) per SM.
- 64K 32-bit registers per SM; max 255 regs/thread; allocation granularity 8 regs (warp-level 256).
- Max 2048 threads/SM (64 warps), 32 thread blocks/SM, 1024 threads/block.
- Shared memory: 228 KB per SM; max 227 KB per thread block (must opt in via
  `cudaFuncAttributeMaxDynamicSharedMemorySize`). SMEM/L1 carve-out is configurable.
- L2: 50 MB, partitioned (two halves); L2 persistence windows available.
- HBM3: ~3.35 TB/s peak. BF16/FP16 tensor: 989 TFLOPS dense; FP8: 1979 TFLOPS dense.
- Boost clock ~1980 MHz; lockable via `nvidia-smi -lgc` (root).

## Thread block clusters
- Cluster dims up to 8 CTAs portable (16 with
  `cudaFuncAttributeNonPortableClusterSizeAllowed` opt-in, at occupancy cost).
- Distributed shared memory (DSMEM): any CTA in the cluster can ld/st/atomics
  another CTA's SMEM via `cluster.map_shared_rank()`; requires
  `cluster.sync()` or mbarrier-based arrival for safety.
- TMA multicast: one TMA load can broadcast to SMEM of multiple CTAs in a cluster.

## TMA (Tensor Memory Accelerator)
- `cp.async.bulk.tensor.{1..5}d` global<->shared, descriptor-driven
  (`CUtensorMap`, created host-side via `cuTensorMapEncodeTiled`).
- Global address must be 16B aligned; SMEM destination 128B aligned;
  box inner dimension ≤ 256 elements; swizzle modes: 32B/64B/128B.
- Completion: loads signal an mbarrier with expect-tx (`mbarrier.arrive.expect_tx`
  + transaction bytes); stores complete via `cp.async.bulk.commit_group` /
  `cp.async.bulk.wait_group N`.
- TMA stores from SMEM written by the generic proxy REQUIRE
  `fence.proxy.async.shared::cta` before the store is issued (async proxy).

## mbarrier
- 64-bit object in SMEM, `mbarrier.init` with arrival count; phase-parity
  wait (`mbarrier.try_wait.parity`) - track the phase bit per pipeline stage.
- One thread should issue expect-tx per TMA load; all consumers arrive.

## WGMMA (warpgroup MMA)
- `wgmma.mma_async.sync.aligned.m64nNk16` for FP16/BF16 (k32 for FP8);
  N ∈ {8,16,...,256}. Operates per warpgroup (4 contiguous warps, 128 threads).
- A from SMEM or registers; B from SMEM (descriptor-encoded, swizzled layouts).
- Async: `wgmma.fence` before first use of accumulator registers,
  `wgmma.commit_group`, `wgmma.wait_group N`. Accumulators stay in registers -
  watch register pressure (m64n256k16 FP32 accum = 128 regs/thread).
- FP8 accumulate: hardware accumulates in FP32 but with limited intermediate
  precision on long K - register-level re-accumulation chunking is a known
  accuracy/perf trade (tag fp8_reg_accum).

## setmaxnreg
- `setmaxnreg.inc/dec.sync.aligned.u32 N`: warpgroup-level dynamic register
  reallocation (producer warps shrink to 24-40 regs, consumer warpgroups grow
  to 232-240). Requires launch bounds compatible with the total budget.

## Caveats
- `elect.one.sync` for single-thread TMA issue inside a warp.
- Persistent kernels must respect 132-SM wave quantization (grid = #SMs × blocks/SM).
- compute-sanitizer racecheck does NOT model mbarrier phase semantics fully;
  a clean racecheck does not prove pipeline correctness - reviewer must check
  arrive/wait pairing manually.
