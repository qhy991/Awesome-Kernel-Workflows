# WarpSpeed Kernel Wiki

Curated technique pages, read-only at run time. Each page declares
frontmatter consumed by `tools/render_config.py` to build the tag→page index
(`tags`, `arch` scope, `expected_gain_pct` range). The planner emits
`direction_tags`; the implementor receives the 2–3 matching pages — never the
whole wiki.

| Page | Tags | Arch |
|---|---|---|
| tma-bulk-async-copy.md | tma, async_copy, mbarrier, smem_staging | sm90, sm100 |
| warp-specialization-pingpong.md | warp_spec, pingpong, producer_consumer, pipeline_depth, setmaxnreg | sm90, sm100 |
| persistent-kernel-scheduling.md | persistent_kernel, work_queue, wave_quant, occupancy | sm90, sm100 |
| wgmma-smem-pipelines.md | wgmma, pipeline_depth, smem_staging, mma | sm90 |
| smem-swizzle-bank-conflicts.md | swizzle, bank_conflict, layout, smem_staging | sm90, sm100 |
| cluster-dsmem.md | cluster, dsmem, multicast, tma | sm90, sm100 |
| register-pressure-occupancy.md | regs, occupancy, spills, setmaxnreg | sm90, sm100 |
| l2-policy-and-prefetch.md | l2_persist, prefetch, cp_async_bulk, cache_policy | sm90, sm100 |
| fp8-register-accumulation.md | fp8, fp8_reg_accum, accuracy, mma | sm90, sm100 |
| tcgen05-tmem-basics.md | tcgen05, tmem, mma, cta_pair | sm100 |

Adding a page: copy the frontmatter shape, keep the What / When (NCU
signals) / How / Pitfalls / Interactions structure, scope `arch` honestly,
and state an honest `expected_gain_pct` range — the planner's prediction
calibration depends on it.
