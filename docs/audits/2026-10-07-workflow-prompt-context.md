# Workflow prompt-context audit — 2026-10-07

Scope: all35 catalog workflow scripts, excluding templates/helpers as separate methods. This is a source audit with executable regressions for the five active task-mode methods, not GPU qualification of all35. A source interpolation can be a necessary first materialization, intentional code analysis, or repeated history; a regex match alone is not a defect.

## Completed changes

- Task-owned source/result/trace files are the detailed authority. Prompt projections retain selection/reward scalars and exact file references; complete diagnostics and source stay in existing runtime records/files.
- CUDAAgent, AccelOpt, KernelFoundry and KSearch now use existing file evidence in additional downstream paths. CUDALLM was repaired in058eebf. Search loops, history windows, budgets and reward definitions are unchanged.
- KSearch previously lost failure detail when no candidate passed because cycleBestEval remains null; the latest rejected evaluation is now available by path for repair/backtracking, without promoting it.
- Xe-Forge printed a misleading “truncated” suffix although the entire source was present; removed that suffix. No character caps introduced.
- Missing authority paths retain prior content rather than silently discard evidence or invent a file. The follow-up migration below implements file lifecycle and per-method tests. Initial materialization may still contain one complete new source; downstream histories use existing files.

## Per-workflow review

| Method | Script | Finding/disposition | Candidate scan lines |
|---|---|---|---|
| AKO4X | `AKO4X/ako4x-kernel-optimizer.js` | Fixed: existing authoritative variant_path is used in smoke/benchmark/archive prompts; raw code remains a compatibility output. | 1239,1295,1511,1543,1602,1824 |
| ARGUS | `ARGUS/argus-kernel-optimization.js` | No targeted repeated-source/history pattern found; static scan is not native-runtime or context-limit qualification. | none |
| AccelOpt | `AccelOpt/accelopt-kernel-optimization.js` | Fixed measured beam and slow/fast learning source handoff. Initial unmaterialized code and method-owned knowledge remain inline. | 745,1701 |
| AdaExplore | `AdaExplore/adaexplore-kernel-optimization.js` | Fixed: diverse-pool, MCTS path, tuner/report context uses authoritative kernelPath; Sol nodes retain actual Sol materialization path. | 994,1055,1065,1100,1618 |
| AscendC | `AscendC/ascendc-kernel-optimization.js` | Candidate summaries explicitly select scalar metrics; no source-body history match. | 464,507 |
| Astra | `Astra/astra-kernel-optimization.js` | Generated candidate materialization found; not evidence of repeated history overflow. | none |
| Atrex | `Atrex/atrex-kernel-optimization.js` | No targeted repeated-source/history pattern found; static scan is not native-runtime or context-limit qualification. | none |
| AutoMegaKernel | `AutoMegaKernel/automegakernel-megakernel-optimization.js` | No targeted repeated-source/history pattern found; static scan is not native-runtime or context-limit qualification. | none |
| CUDAAgent | `CUDAAgent/cuda-agent-kernel-optimization.js` | Fixed task-mode prior error feedback: retain raw history, pass result-file references to Implement. | 1146,1382 |
| CUDALLM | `CUDALLM/cudallm-fsr-kernel-generation.js` | Fixed task-mode selection/reinforcement/report file indices (058eebf). Catalog and feature-score strategy state remains inline. | 1106,1482 |
| CutlassGEMM | `CutlassGEMM/cutlass-gemm-optimization.js` | No targeted repeated-source/history pattern found; static scan is not native-runtime or context-limit qualification. | none |
| FACT | `FACT/fact-kernel-optimization.js` | Source passed to evaluator for materialization; not by itself a repeated-history defect. | none |
| GPUForecasters | `GPUForecasters/gpuforecasters-kernel-optimization.js` | No targeted repeated-source/history pattern found; static scan is not native-runtime or context-limit qualification. | none |
| GemmPTX | `GemmPTX/gemmptx-gemm-optimization.js` | Fixed: compile/correctness/history diagnostics reference command-owned evidence or a complete saved assessment; scalar instruction/score history remains. | 891 |
| Generalist | `Generalist/generalist-kernel-optimization.js` | Candidate exploration uses code_path; no change required for known-kernel handoff. | 1088,1161,1210 |
| HarnessEngineering | `HarnessEngineering/harness-engineering-kernel-optimization.js` | Fixed: decision/implementation/evidence files keep complete records; history and incumbent summaries refer to these files. | 381 |
| InPlacePatch | `InPlacePatch/in-place-patch-optimization.js` | No targeted repeated-source/history pattern found; static scan is not native-runtime or context-limit qualification. | none |
| KDA | `KDA/kda-kernel-workflow.js` | No targeted repeated-source/history pattern found; static scan is not native-runtime or context-limit qualification. | none |
| KEET | `KEET/keet-kernel-explanation.js` | Code evidence is the analysis subject; file ranges may be appropriate but no automatic removal. | 640,853 |
| KSearch | `KSearch/ksearch-kernel-optimization.js` | Fixed repair/backtrack evidence projection; last rejected result now remains reachable even when no candidate passed. | 1655,1696,1980 |
| KernelAgent | `KernelAgent/kernelagent-triton-synthesis.js` | Fixed: per-candidate/per-refinement source files; verification materializes exact bytes, downstream refinement/composition carries paths; Sol rounds no longer share one source filename. | 757,1212,1407,1514,1623 |
| KernelBand | `KernelBand/kernelband-kernel-optimization.js` | Task/file candidate path already used for representative and selected kernels; not proven bounded for every other input. | none |
| KernelBlaster | `KernelBlaster/kernelblaster-kernel-optimization.js` | Fixed: variant-specific materialization and incumbent source-path lineage across rollout steps; initial generation remains a first write. | 1182 |
| KernelFoundry | `KernelFoundry/kernelfoundry-kernel-optimization.js` | Fixed tested parent/report source and rejected-candidate evidence handoff; raw transitions retained. | none |
| KernelFoundryDx | `KernelFoundryDx/kernelfoundrydx-kernel-optimization.js` | Fixed: existing variant_path is used for parent, diagnosis and report instead of duplicate code. Initial seeds remain first-materialization input. | 737,930,1090 |
| KernelSkill | `KernelSkill/kernelskill-kernel-optimization.js` | Fixed: seed/review persistence and repair/optimization output files; downstream diagnosis/report reads the corresponding current/best path. | 929 |
| LlamacppEmbeddedSearch | `LlamacppEmbeddedSearch/llamacpp-embedded-search.js` | No targeted repeated-source/history pattern found; static scan is not native-runtime or context-limit qualification. | none |
| LlamacppEmbeddedSearch | `LlamacppEmbeddedSearch/llamacpp-metal-embedded-search.js` | No targeted repeated-source/history pattern found; static scan is not native-runtime or context-limit qualification. | none |
| ReGraphT | `ReGraphT/regrapht-kernel-optimization.js` | Fixed: per-candidate source/result files, per-round graph snapshots and selected-example files; selection/update/report use file indices. | 805 |
| STARK | `STARK/stark-kernel-optimization.js` | Fixed: per-node source and result paths; code/debug/sibling/report contexts read complete files. Existing source and evaluator traces retained. | 607,612,621,637,646,1360 |
| StitchCUDA | `StitchCUDA/stitchcuda-kernel-optimization.js` | CodeResult materialization found; first write is not automatically redundant historical context. | 1049 |
| TritorX | `TritorX/tritorx-operator-generation.js` | No targeted repeated-source/history pattern found; static scan is not native-runtime or context-limit qualification. | none |
| WarpSpeed | `WarpSpeed/warpspeed-kernel-search.js` | Candidates passed to a subprocess statistical helper; distinguish command input from LLM prompt. | none |
| Xe-Forge | `Xe-Forge/xe-forge-kernel-optimization.js` | Fixed: initial Analyze and optimized Verify persist per-cycle sources; subsequent cycles reference exact previous files. | 443,628 |
| cuPilot | `cuPilot/cupilot-kernel-optimization.js` | Fixed: revision activation persists final code to a unique epoch/generation/candidate path; later translation/report reads it. | 715,1018 |

## Verification and limits

Task-mode regression tests execute actual workflow bodies under the native runtime seam with large diagnostics and retained file pointers. Negative controls retain missing-source/unknown-result checks. No checkpoint, final artifact, raw trace or workflow outcome is rewritten by projection. Runtime context may still overflow from unusually large task specifications, generated strategy state, or initial code materialization; no guarantee for all providers is made. For file-first adoption in remaining methods: persist each candidate once, return declared file identities, preserve raw evaluation files, and index those records in history. Do not add a model turn that recopies the entire growing history.

Validation at this patch:87 task-result/workflow regressions and7 shared-scaffolding guards pass; all35 Host parses pass. Existing handoff static scan still reports the same5 bare-parse sites on both bf071db and this patch; no new sites. The earlier CUDALLM legacy golden3 failures also reproduce on bf071db; this change does not claim the full legacy suite is green.

Follow-up: source persistence and method-context regression details are in `2026-10-07-workspace-context-migration.md`; the8-theorem Lean model is in `../formal/WorkspaceHandoff.lean`. Static scan line numbers above describe the original audit, not immutable source locations.
