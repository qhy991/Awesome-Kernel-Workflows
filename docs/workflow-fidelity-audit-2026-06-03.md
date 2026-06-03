# Workflow Fidelity Audit - 2026-06-03

Scope: top-level method workflows in `Awesome-Kernel-Workflows`. `_meta/`, `_templates/`, and `experiments/` are excluded because they are generator/template/runtime artifacts rather than paper-method workflows.

Verdict scale:

- Strict / high-fidelity: preserves the source method's decision loop and evidence contract.
- Faithful but simplified: preserves core method, but compresses implementation details or system runtime.
- Idea-preserving only: useful workflow inspired by the paper, but not a strict reproduction.
- High-risk: claims source fidelity while missing a load-bearing mechanism.

## Summary

| Workflow | Verdict | Main reason |
|---|---|---|
| `AKO4X/ako4x-kernel-optimizer.js` | Strict / high-fidelity | Directly mirrors local AKO4X round/iteration/archive contract, including pre-bench expectation, smoke/full bench, archive gates, two-layer lessons, TRAPS, and Mode 3 retrospective. |
| `KDA/kda-kernel-workflow.js` | Strict / high-fidelity | Directly mirrors local `kernel-design-agents/docs/agent-flow.md`: inspect, draft, executable plan, one candidate at a time, validate, record, promote/reject. |
| `AdaExplore/adaexplore-kernel-optimization.js` | High-fidelity method translation | Correctly preserves Adapt/Explore split, failure-grounded "You cannot..." memory, MCTS, large/small steps, UCB/expand-UCB, diversity pool, and evaluator authority. It intentionally does not call the repo entrypoint, which matches the prior design requirement. |
| `KSearch/ksearch-kernel-optimization.js` | High-fidelity method translation | Preserves co-evolving world-model tree, frontier action selection, generate/improve attempts, evaluation, success refine, failure backtrack, and persisted solution/tree state. |
| `KernelAgent/kernelagent-triton-synthesis.js` | Faithful but simplified | Preserves KernelAgent/Fuser shape: route, parallel Triton seeds, strict verification, refinement, optional subgraph composition. Simplifies real repo worker/process/runtime details into agent prompts. |
| `KEET/keet-kernel-explanation.js` | Faithful but simplified | Preserves source-inspection before profile-inspection, metric selection, grounded NCU explanation, aggregation, and hypothesis review. Not an optimizer; correct as an explanation workflow. |
| `AccelOpt/accelopt-kernel-optimization.js` | Faithful but shifted target | Preserves Plan/Execute/Profile/Summarize/Experience loop, beam, slow-fast learning, thresholds, per-branch dedup. Main drift: source repo is accelerator-general and Trainium/NKI plus NVIDIA; workflow narrows heavily to CUDA/NCU. |
| `CUDAAgent/cuda-agent-kernel-optimization.js` | Inference-loop faithful, paper-level partial | Correctly states the paper's RL/PPO training is not reproduced and implements only the inference-time SKILL loop. Strictly following the paper would require dataset/RL training/reward rollout machinery, which this workflow omits by design. |
| `ReGraphT/regrapht-kernel-optimization.js` | Inference-loop faithful, paper-level partial | Preserves reasoning graph plus Monte Carlo Graph Search and evaluator-grounded update. It does not train/fine-tune a small model, so it follows the training-free inference/use phase, not the full paper pipeline. |
| `STARK/stark-kernel-optimization.js` | Faithful but simplified | Preserves Plan/Code/Debug separation, epsilon-greedy tree search, grounded instruction anchors, dynamic context windows, and leaderboard. Real model-temperature/runtime separation is represented as args/prompts, not guaranteed by workflow runtime. |
| `KernelBand/kernelband-kernel-optimization.js` | Faithful algorithm sketch, execution risk | Includes feature vectors, K-means, representative profiling, hardware masks, masked UCB, reward update. Risk: much of phi/profile extraction and clustering is delegated to LLM prompts, so strictness depends on real profiler outputs and deterministic clustering evidence. |
| `KernelFoundry/kernelfoundry-kernel-optimization.js` | Faithful algorithm sketch, execution risk | Includes MAP-Elites 4x4x4 archive, behavioral descriptors, gradient-informed selection, meta-prompt evolution, and templated tuning idea. Risk: descriptor classification and meta-prompt edits are prompt-mediated; strict MAP-Elites behavior needs stronger deterministic archive/update checks. |
| `cuPilot/cupilot-kernel-optimization.js` | Faithful algorithm sketch, execution risk | Preserves strategy-level crossover, strategy translator, roofline guidance, RAG init, tournament selection, elitism, strategy alignment. Risk: roofline classification/RAG/NCU feedback are agent-produced unless commands and strategy corpus are concrete. |
| `TritorX/tritorx-operator-generation.js` | Faithful FSM sketch, not full-scale strict | Preserves coverage-over-speed objective, FSM, Triton linter, compile/test, OpInfo, debug feedback, summarization, batch operator loop. Strict paper fidelity would require real target ASIC dialect, real linter, OpInfo harness, and scale/coverage accounting. |
| `Astra/astra-kernel-optimization.js` | Idea-preserving only unless local Astra repo exists | Captures existing-kernel production optimization with testing/profiling/planning/coding agents and reintegration notes. I did not find a local Astra source repo under `~/Research` or `/data/.../Research`; current workflow is paper-summary-derived. |
| `CUDALLM/cudallm-fsr-kernel-generation.js` | Idea-preserving only | Explicitly implements a Feature Search and Reinforcement loop and says it does not train. If the CUDA-LLM paper's main claim is model/training/data methodology, this is not strict; it is an executable search heuristic inspired by CUDA features. No local CUDA-LLM repo found. |
| `ARGUS/argus-kernel-optimization.js` | High-risk partial | It names the real load-bearing ARGUS idea: data-flow invariants with tag functions/assertions and ICRL. But the workflow appears to ask agents to create/check invariants rather than providing an actual DSL/invariant checker. Without executable tag assertion validation, the dense feedback mechanism is not strict. |

## Main Architectural Findings

1. The workflows split into two families:
   - Direct local workflow translations: `AKO4X`, `KDA`, `AdaExplore`, `KSearch`, `KernelAgent`.
   - Paper-derived method sketches: `ARGUS`, `KernelFoundry`, `KernelBand`, `cuPilot`, `TritorX`, `Astra`, `CUDALLM`, etc.

2. The strongest workflows have a real evidence owner:
   - `AKO4X`: benchmark/smoke/archive gates and TRAPS.
   - `AdaExplore`: evaluator JSON is authoritative and memory updates are explicit.
   - `KSearch`: world-model state is updated on measured success/failure.
   - `KDA`: validation/evaluation commands and `candidates.jsonl` equivalent.

3. The weaker workflows often encode paper mechanisms as natural-language agent instructions:
   - `ARGUS`: invariant checker is not a real executable validator.
   - `KernelBand`: phi/K-means/profile signatures may be LLM-produced if no concrete NCU command exists.
   - `KernelFoundry`: behavioral descriptors and prompt evolution are mostly prompt-mediated.
   - `cuPilot`: roofline/RAG/SCE can degrade to self-reported strategy text.
   - `TritorX`: strictness requires real linter plus OpInfo harness; otherwise it is a finite-state prompt loop.

4. "Does not train" boundaries are correctly stated in `CUDAAgent`, `CUDALLM`, and `ReGraphT`, but those should be labeled as inference-time adaptations rather than strict full-paper reproductions.

## Recommended Fix Priority

1. Fix or relabel `ARGUS` first.
   - Either add a concrete invariant DSL/checker contract with executable tag assertion validation, or downgrade README wording from "implements ARGUS methodology" to "ARGUS-inspired planning workflow".

2. Add required evidence contracts to `KernelBand`, `KernelFoundry`, and `cuPilot`.
   - Make profiler output / clustering / archive updates deterministic where possible.
   - Require command outputs or JSON artifacts for feature vectors, descriptors, masks, and rewards.

3. Relabel `CUDALLM`, `CUDAAgent`, and `ReGraphT` precisely.
   - Use "inference-time adaptation" or "workflow adaptation" where training/fine-tuning is omitted.

4. For `TritorX`, make the linter/OpInfo harness mandatory if claiming strict method adherence.
   - Without those, keep it as "TritorX-style FSM".

5. For source coverage, clone or link local repos for paper-only workflows if strict auditing is expected.
   - Missing locally: `Astra`, `CUDA-LLM`, `ARGUS`, `KernelBand`, `KernelFoundry`, `TritorX`, `cuPilot`, `STARK`, `ReGraphT`, `KEET`.

## Bottom Line

No, the current set does not all strictly follow the corresponding repositories or papers.

The strictest are `AKO4X`, `KDA`, `AdaExplore`, and `KSearch`. A middle group faithfully preserves the method shape but simplifies runtime machinery. The main architectural risk is the high-complexity paper workflows where the paper's core feedback signal is not an executable artifact but a prompt instruction. For these, the right repair is not more prose; it is to promote the paper's load-bearing signal into a workflow-owned contract: concrete commands, JSON artifacts, deterministic update rules, and hard failure behavior when evidence is missing.
