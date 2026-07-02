---
name: gemmptx-instruction-evidence
description: Use when GemmPTX plans, implements, or verifies CUDA/CuTe/CUTLASS GEMM instruction paths with PTX/SASS evidence, especially mma.sync, wgmma, TMA, mbarrier, tcgen05, tensor-core, disassembly, or hypothesis_not_realized decisions.
---

# GemmPTX Instruction Evidence

## Core Rule

GemmPTX optimizes GEMM instruction paths, not generic compute-bound code. Do not optimize from profiler counters until compile and correctness pass and PTX/SASS proves the intended path exists. If the candidate is correct but the expected instruction regex is absent, record `hypothesis_not_realized`.

## Architecture Map

| Arch | Prefer | Evidence regex | Main risks |
|---|---|---|---|
| `sm80` | `mma.sync` + `cp.async` | `mma\.sync\.aligned`, SASS `HMMA|MMA`; `cp\.async` | alignment, stage count vs occupancy, register pressure |
| `sm90` | `wgmma.mma_async` + TMA (`cp.async.bulk.tensor`) | `wgmma\.mma_async`, `cp\.async\.bulk\.tensor`, SASS `WGMMA|TMA|CP_ASYNC_BULK` | descriptor swizzle, `wgmma.fence/commit/wait`, mbarrier phase |
| `sm100` | `tcgen05` + TMEM | `tcgen05`, SASS `TCGEN05|MMA` | TMEM capacity, CTA-pair sync, small-tile overhead |

Use exact hardware facts when available. If SM count, L2, SMEM, or compute capability are unmeasured, keep the hypothesis conservative.

## GEMM Triage

- Small M or low `waves_per_sm`: try split-K, StreamK, persistent scheduling, or smaller tiles before chasing bigger MMA shapes.
- High tensor-pipe headroom with scalar/SIMT disassembly: move toward tensor-core MMA.
- Low L2 hit with reused operand fitting L2: try launch swizzle, operand residency, or TMA/cache policy.
- High registers/thread or spills: shrink tile, reduce accumulator width, shorten live ranges, or use TMEM/setmaxnreg only when the arch supports it.
- Good SM% and tensor-pipe utilization: avoid large instruction rewrites unless the mechanism is sharply defined.

## Evidence Gate

For every candidate:

1. Compile.
2. Run correctness.
3. Disassemble PTX/SASS.
4. Match expected `ptx_regex` or `sass_regex`.
5. Only then run benchmark/profile.

Required disassembly fields: `observed_instructions`, `ptx_path`, `sass_path`, `registers_per_thread`, `shared_mem_bytes`, `local_mem_bytes`, `spill_loads`, `spill_stores`, `missing_expected_instructions`.

## Pitfalls

- WGMMA: missing `wgmma.fence`, serializing with `wait_group 0`, or mismatched SMEM descriptor swizzle can be correct for one shape and wrong for another.
- TMA: `mbarrier.arrive.expect_tx` byte count must match the transfer. Missing async proxy fence before TMA stores can read stale SMEM.
- `mma.sync`: source-level WMMA/CUTLASS code may lower differently than intended; trust disassembly, not source shape names.
- `tcgen05`: do not assume Blackwell wins automatically; TMEM and CTA-pair overhead can erase gains on small tiles.
- Register caps: do not blindly use `maxrregcount`; spills often convert a tensor-core win into a memory-bound loss.

## Candidate Discipline

One major hypothesis per candidate. Do not combine tile shape, pipeline depth, layout swizzle, epilogue fusion, and instruction-family migration in the same attempt unless the previous evidence demands the coupling.
