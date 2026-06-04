<h1 align="center">Awesome Kernel Workflows</h1>

<p align="center">
  Curated <b>GPU kernel optimization</b> methods from academia and industry, packaged as reusable <a href="https://docs.anthropic.com/en/docs/claude-code">Claude Code</a> Workflows
</p>

<p align="center">
  <a href="https://github.com/qhy991/Awesome-Kernel-Workflows"><img src="https://img.shields.io/badge/GitHub-Awesome--Kernel--Workflows-blue?logo=github" alt="GitHub"></a>
  <a href="#catalog"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/qhy991/Awesome-Kernel-Workflows/main/badges/workflows.json" alt="workflow count"></a>
  <img src="https://img.shields.io/badge/Claude%20Code-Workflows-7C3AED" alt="Claude Code Workflows">
  <img src="https://img.shields.io/badge/GPU-CUDA%20%7C%20Triton%20%7C%20DSL-green" alt="GPU Kernels">
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

---

## What is this?

GPU kernel optimization produces a steady stream of strong work: self-improving loops (e.g. AccelOpt), agentic frameworks (e.g. AKO), and engineering playbooks behind industrial SOTA (FlashInfer, CUTLASS, etc.). These contributions usually ship as **papers plus code repos**. Handing them to a coding agent as-is rarely reproduces the **decision process**—when to profile, how to read NCU, how to accumulate experience, and so on.

This repository aims to:

1. **Collect** — Methods and systems from academia and industry that are suitable for agent-driven kernel optimization.
2. **Distill** — Abstract each method’s core loop (plan / execute / profile / learn, etc.) into explicit phases.
3. **Ship** — Implement them as [Claude Code Workflows](https://docs.anthropic.com/en/docs/claude-code/workflows) (`.js`) you can drop into your kernel project and invoke.

Each subdirectory is **one method or one workflow**, with the workflow definition, parameter notes, and (over time) source links and examples.

---

## Repository layout

```
Awesome-Kernel-Workflows/
├── README.md                    # English (this file)
├── README.zh-CN.md              # 简体中文
├── AccelOpt/                    # Self-improving loop with NCU profiling
│   ├── accelopt-kernel-optimization.js
│   └── README.md
├── KEET/                        # NCU profile → explanation pipeline
│   ├── keet-kernel-explanation.js
│   └── README.md
├── ARGUS/                       # Data-flow invariant guided optimization
│   ├── argus-kernel-optimization.js
│   └── README.md
├── AKO4X/                       # Multi-round closed-loop with experience accumulation
│   ├── ako4x-kernel-optimizer.js
│   └── README.md
├── KDA/                         # Evidence-driven draft-plan-implement-validate-decide
│   ├── kda-kernel-workflow.js
│   └── README.md
├── AdaExplore/                  # MCTS tree search + failure-driven skill memory
│   ├── adaexplore-kernel-optimization.js
│   └── README.md
├── KernelFoundry/               # MAP-Elites evolutionary + meta-prompt co-evolution
│   ├── kernelfoundry-kernel-optimization.js
│   └── README.md
├── KSearch/                     # World-model-guided tree search
│   └── ksearch-kernel-optimization.js
├── KernelBand/                  # MAB with hardware-aware pruning + clustering
│   ├── kernelband-kernel-optimization.js
│   └── README.md
├── KernelAgent/                 # Multi-agent Triton synthesis with parallel verification
│   ├── kernelagent-triton-synthesis.js
│   ├── manifest.yaml
│   └── README.md
├── STARK/                       # Multi-agent collaboration + tree search + grounded instruction
│   ├── stark-kernel-optimization.js
│   ├── manifest.yaml
│   └── README.md
├── ReGraphT/                    # CUDA reasoning graph + Monte Carlo Graph Search
│   ├── regrapht-kernel-optimization.js
│   ├── manifest.yaml
│   └── README.md
├── Astra/                       # Multi-agent optimization for existing CUDA kernels
│   ├── astra-kernel-optimization.js
│   ├── manifest.yaml
│   └── README.md
├── CUDALLM/                     # Feature Search and Reinforcement for CUDA generation
│   ├── cudallm-fsr-kernel-generation.js
│   ├── manifest.yaml
│   └── README.md
├── CutlassGEMM/                 # CUTLASS GEMM multi-config dispatch tuning
│   ├── cutlass-gemm-optimization.js
│   └── README.md
├── ArchAgent/                   # Evolutionary cache replacement policy optimization
│   ├── archagent-cache-policy-optimization.js
│   └── README.md
├── FACT/                        # Compositional CUTLASS pattern synthesis
│   ├── fact-kernel-optimization.js
│   └── README.md
├── GPUForecasters/              # PUCT search with learned speedup forecasting
│   ├── gpuforecasters-kernel-optimization.js
│   └── README.md
├── KernelBlaster/               # Memory-augmented in-context RL for CUDA kernels
│   ├── kernelblaster-kernel-optimization.js
│   └── README.md
├── KernelFoundryDx/             # Diagnosis-driven multi-island Triton evolution
│   ├── kernelfoundrydx-kernel-optimization.js
│   └── README.md
├── KernelSkill/                 # Dual-memory multi-agent CUDA optimization
│   ├── kernelskill-kernel-optimization.js
│   └── README.md
├── StitchCUDA/                  # Planner/Coder/Verifier CUDA synthesis
│   ├── stitchcuda-kernel-optimization.js
│   └── README.md
├── Xe-Forge/                    # Multi-stage CoVeR optimization for Intel XPU
│   ├── xe-forge-kernel-optimization.js
│   └── README.md
├── _meta/                       # Meta-workflow: paper → workflow generation
│   ├── README.md
│   ├── tools/
│   │   ├── generate-workflow.js
│   │   └── validate-workflow.js
│   ├── templates/
│   │   ├── iterative-loop.js
│   │   ├── search-based.js
│   │   ├── single-pass.js
│   │   └── tree-exploration.js
│   └── manifests/
│       ├── schema.yaml
│       ├── accelopt.yaml
│       ├── keet.yaml
│       ├── argus.yaml
│       └── ksearch.yaml
└── <MoreMethods>/               # PRs welcome
    └── *.js
```

Workflow files follow Claude Code conventions: export `meta` (name, description, phases) and orchestrate multi-step agent work via `phase()` and `agent()`.

---

## Workflow suitability

Every top-level workflow declares `WORKFLOW_SUITABILITY` after `meta`. If you explicitly pass an unsupported `args.language` or `args.problem_type`, the workflow fails before doing any work and reports the supported values plus the reason. The check is deliberately conservative: workflows do not infer language or problem type from natural-language `problem_definition`.

| Workflow | Supported language/backend | Supported `problem_type` values | Good fit | Avoid when |
|----------|----------------------------|---------------------------------|----------|------------|
| [AccelOpt](AccelOpt/) | CUDA | `cuda-kernel-optimization`, `cuda-kernel-generation` | Existing CUDA kernels, or CUDA seed generation followed by NCU/benchmark-driven optimization | Triton/SYCL/XPU tasks or missing benchmark/profile contract |
| [KEET](KEET/) | CUDA | `performance-explanation` | Explaining CUDA source plus Nsight Compute profile evidence | Kernel generation or optimization without profile artifacts |
| [ARGUS](ARGUS/) | Argus DSL, CUDA, ROCm, Triton | `invariant-guided-kernel-optimization`, `gpu-kernel-optimization` | GPU kernel optimization with invariant checker/test feedback | Tasks without invariant or validation evidence |
| [AKO4X](AKO4X/) | Triton, CUDA, CuTe DSL, TileLang, C++, PyTorch | `gpu-kernel-optimization`, `kernel-generation` | Multi-round benchmark-driven GPU kernel optimization across supported DSLs | Non-kernel application code or unsupported backend toolchains |
| [KDA](KDA/) | CUDA | `cuda-kernel-optimization`, `cuda-kernel-generation` | Evidence-driven CUDA implementation, validation, and optimization loops | Non-CUDA backends until the KDA skill flow is generalized |
| [K-Search](KSearch/) | Triton, CUDA, Python | `gpu-kernel-optimization`, `kernel-search` | World-model tree search with an evaluator/benchmark contract | Tasks without executable evaluator feedback |
| [AdaExplore](AdaExplore/) | Triton | `triton-kernel-optimization`, `triton-kernel-generation` | PyTorch operator spec to Triton using MCTS and failure memory | Direct CUDA/CUTLASS/SYCL optimization |
| [KernelFoundry](KernelFoundry/) | SYCL, CUDA, Triton | `gpu-kernel-optimization`, `kernel-generation`, `kernel-search` | MAP-Elites quality-diversity search with descriptor/archive feedback | Single deterministic patch workflows without archive state |
| [CUDA Agent](CUDAAgent/) | CUDA | `cuda-kernel-generation`, `cuda-kernel-optimization` | PyTorch model/operator to custom CUDA ops and bindings | Triton/SYCL/CUTLASS-only tasks |
| [cuPilot](cuPilot/) | CUDA | `cuda-kernel-optimization` | Strategy-level CUDA evolution with roofline/profiler evidence | Non-CUDA kernels |
| [TritorX](TritorX/) | Triton | `aten-triton-operator-generation`, `operator-generation` | ATen/Triton operator coverage generation and compile/lint/test loops | Performance-first CUDA tuning |
| [KernelBand](KernelBand/) | Triton, CUDA | `gpu-kernel-optimization`, `kernel-search` | Bandit-guided search using hardware signatures and profiling | Backends without comparable profiling/evaluator evidence |
| [KernelAgent](KernelAgent/) | Triton | `triton-kernel-generation`, `operator-generation` | Triton synthesis with PyTorch-style verification harnesses | CUDA/C++/CUTLASS kernels |
| [STARK](STARK/) | CUDA | `cuda-kernel-optimization`, `kernel-search` | CUDA tree-search refinement with multi-agent planning/debugging | Non-CUDA backends until code-context adapters exist |
| [ReGraphT](ReGraphT/) | CUDA | `cuda-kernel-optimization`, `kernel-search` | CUDA reasoning graph search and Monte Carlo graph search | Non-CUDA optimization traces |
| [Astra](Astra/) | CUDA | `cuda-kernel-optimization` | Existing production CUDA/PyBind kernels with tests and profiling | From-scratch Triton/SYCL generation |
| [CUDA-LLM](CUDALLM/) | CUDA | `cuda-kernel-generation`, `cuda-kernel-optimization` | CUDA feature search/reinforcement from task specs | Non-CUDA output languages |
| [CutlassGEMM](CutlassGEMM/) | CUTLASS, CUDA, C++ | `cutlass-gemm-optimization` | CUTLASS GEMM/SOL-ExecBench dispatch tuning | General elementwise/attention kernels outside CUTLASS GEMM |
| [ArchAgent](ArchAgent/) | C++ | `cache-policy-search` | ChampSim-style CPU cache replacement policy search | GPU kernel optimization |
| [FACT](FACT/) | CUTLASS, CUDA, C++ | `cutlass-pattern-synthesis`, `cutlass-gemm-optimization` | CUTLASS pattern discovery, realization, composition, and ablation | Standalone Triton/SYCL kernels |
| [GPU Forecasters](GPUForecasters/) | CUDA | `cuda-kernel-optimization`, `kernel-search` | CUDA/GPU search with speedup forecaster and execute/abstain feedback | Tasks without GPU execution or forecast calibration |
| [KernelBlaster](KernelBlaster/) | CUDA | `cuda-kernel-optimization` | NCU elapsed-cycle CUDA optimization with persistent memory | Non-CUDA backends |
| [KernelFoundryDx](KernelFoundryDx/) | Triton | `triton-kernel-optimization`, `triton-kernel-generation` | PyTorch reference to Triton multi-island evolution | CUDA/CUTLASS/SYCL kernels |
| [KernelSkill](KernelSkill/) | CUDA | `cuda-kernel-optimization`, `cuda-kernel-generation` | CUDA custom kernels from PyTorch reference with memory/profiler guidance | Triton/SYCL/Metal tasks |
| [StitchCUDA](StitchCUDA/) | CUDA | `cuda-kernel-generation`, `cuda-kernel-optimization` | Planner/Coder/Verifier CUDA synthesis and replanning | Non-CUDA backends |
| [Xe-Forge](Xe-Forge/) | Triton, SYCL, Intel XPU | `xpu-kernel-optimization`, `triton-kernel-optimization` | Intel XPU CoVeR staged refinement | NVIDIA CUDA-only tuning |
| [Generalist](Generalist/) | CUDA | `cuda-kernel-generation`, `cuda-kernel-optimization` | CUDA benchmark-driven default solver substrate | Backend-neutral solving; not generalized yet |
| [Meta-Workflow](_meta/) | Tooling | N/A | Generating and validating workflow definitions | Direct kernel optimization |

---

## Catalog {#catalog}

| Method | Tags | Core loop | Paper / project |
|--------|------|-----------|-----------------|
| [AccelOpt](AccelOpt/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Experience](https://img.shields.io/badge/experience--memory-orange?style=flat) | Plan → Execute → Profile → Learn → Iterate | [arXiv:2511.15915](https://arxiv.org/abs/2511.15915) (MLSys 2026) |
| [KEET](KEET/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![Explanation](https://img.shields.io/badge/explanation-teal?style=flat) | Source Inspection → Profile Inspection → Aggregation → Review | [arXiv:2605.04467](https://arxiv.org/abs/2605.04467) (UMD/NVIDIA/LLNL 2026) |
| [ARGUS](ARGUS/) | ![ROCm](https://img.shields.io/badge/ROCm-ED1C24?style=flat&logo=amd&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![DSL](https://img.shields.io/badge/DSL-darkgreen?style=flat) ![ICRL](https://img.shields.io/badge/ICRL-blue?style=flat) ![Invariants](https://img.shields.io/badge/invariants-red?style=flat) | Plan → Select → Lower → Validate → Learn | [arXiv:2604.18616](https://arxiv.org/abs/2604.18616) (CausalFlow/HKUST/Stanford 2026) |
| [AKO4X](AKO4X/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![CuTe](https://img.shields.io/badge/CuTe-darkgreen?style=flat) ![TileLang](https://img.shields.io/badge/TileLang-darkgreen?style=flat) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Experience](https://img.shields.io/badge/experience--memory-orange?style=flat) | Round-Init → Iterate → Archive → Retrospect | [AKO Project](https://tongminglaic.github.io/AKO) |
| [KDA](KDA/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Evidence](https://img.shields.io/badge/evidence--driven-green?style=flat) | Inspect → Plan → Implement → Validate → Decide | [MIT HAN Lab](https://github.com/mit-han-lab/kernel-design-agents) |
| [K-Search](KSearch/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![WorldModel](https://img.shields.io/badge/world--model-orange?style=flat) | Init Tree → Select → Generate → Evaluate → Refine/Backtrack | [arXiv:2602.19128](https://arxiv.org/abs/2602.19128) |
| [AdaExplore](AdaExplore/) | ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![MCTS](https://img.shields.io/badge/MCTS-darkblue?style=flat) ![SkillMemory](https://img.shields.io/badge/skill--memory-orange?style=flat) | Select → Expand → Evaluate → Backpropagate | [arXiv:2604.16625](https://arxiv.org/abs/2604.16625) |
| [KernelFoundry](KernelFoundry/) | ![SYCL](https://img.shields.io/badge/SYCL-0071C5?style=flat&logo=intel&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![Evolutionary](https://img.shields.io/badge/MAP--Elites-darkblue?style=flat) ![MetaPrompt](https://img.shields.io/badge/meta--prompt-orange?style=flat) | Select → Vary → Evaluate → Insert → Evolve-Prompts | [arXiv:2603.12440](https://arxiv.org/abs/2603.12440) (Intel 2026) |
| [CUDA Agent](CUDAAgent/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![RL-trained](https://img.shields.io/badge/RL--trained-red?style=flat) | Profile → Implement → Verify → Refine | [arXiv:2602.24286](https://arxiv.org/abs/2602.24286) (ByteDance/Tsinghua 2026) |
| [cuPilot](cuPilot/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![Evolutionary](https://img.shields.io/badge/evolutionary-darkblue?style=flat) ![Roofline](https://img.shields.io/badge/roofline-teal?style=flat) ![RAG](https://img.shields.io/badge/RAG-orange?style=flat) | Strategize → Translate → Revise → Evolve | [arXiv:2512.16465](https://arxiv.org/abs/2512.16465) (SEU/Tsinghua 2025) |
| [TritorX](TritorX/) | ![ASIC](https://img.shields.io/badge/ASIC%2FNPU-333?style=flat) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![FSM](https://img.shields.io/badge/FSM-blue?style=flat) ![Linter](https://img.shields.io/badge/linter-green?style=flat) ![Coverage](https://img.shields.io/badge/coverage--first-teal?style=flat) | Generate → Lint → Compile/Test → Debug (loop) | [arXiv:2512.10977](https://arxiv.org/abs/2512.10977) (Meta 2025) |
| [KernelBand](KernelBand/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![MAB](https://img.shields.io/badge/MAB--UCB-darkblue?style=flat) ![Clustering](https://img.shields.io/badge/clustering-orange?style=flat) ![HW-Pruning](https://img.shields.io/badge/HW--pruning-red?style=flat) | Profile → Cluster → Select(UCB) → Generate → Evaluate → Update | [arXiv:2511.18868](https://arxiv.org/abs/2511.18868) (PKU 2026) |
| [KernelAgent](KernelAgent/) | ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![Verification](https://img.shields.io/badge/verification-green?style=flat) | Route → Generate(parallel) → Verify → Refine → Compose | [PyTorch Blog](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) (PyTorch Labs 2025) |
| [STARK](STARK/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![Grounded](https://img.shields.io/badge/grounded--instruction-green?style=flat) ![DynamicContext](https://img.shields.io/badge/dynamic--context-orange?style=flat) | Setup → Select(ε-greedy) → Plan/Code/Debug → Evaluate → Update | [arXiv:2510.16996](https://arxiv.org/abs/2510.16996) (Meta/Duke 2025) |
| [ReGraphT](ReGraphT/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![RAG](https://img.shields.io/badge/RAG-orange?style=flat) ![ReasoningGraph](https://img.shields.io/badge/reasoning--graph-teal?style=flat) ![MCGS](https://img.shields.io/badge/MCGS-darkblue?style=flat) | BuildGraph → Select(MCGS) → Generate → Evaluate → UpdateGraph | [arXiv:2510.19873](https://arxiv.org/abs/2510.19873) (CAS/SCUT 2025) |
| [Astra](Astra/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![Profiling](https://img.shields.io/badge/profiling-green?style=flat) ![SGLang](https://img.shields.io/badge/SGLang-orange?style=flat) | Setup → Test/Profile → Plan → Code → Evaluate → Record | [arXiv:2509.07506](https://arxiv.org/abs/2509.07506) (Stanford/SJTU/NJU 2025) |
| [CUDA-LLM](CUDALLM/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![FeatureSearch](https://img.shields.io/badge/feature--search-teal?style=flat) ![Reinforcement](https://img.shields.io/badge/reinforcement-red?style=flat) | Catalog → SelectFeatures → Generate → Evaluate → Reinforce | [arXiv:2506.09092](https://arxiv.org/abs/2506.09092) (2025) |
| [CutlassGEMM](CutlassGEMM/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat&logo=c%2B%2B&logoColor=white) ![CUTLASS](https://img.shields.io/badge/CUTLASS-76B900?style=flat) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Dispatch](https://img.shields.io/badge/multi--config--dispatch-teal?style=flat) | Analyze → GenerateConfigs → Profile(NCU) → TuneDispatch → Validate | [CUTLASS](https://github.com/NVIDIA/cutlass) / [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) |
| [ArchAgent](ArchAgent/) | ![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat&logo=c%2B%2B&logoColor=white) ![Cache](https://img.shields.io/badge/cache--policy-333?style=flat) ![Evolutionary](https://img.shields.io/badge/evolutionary-darkblue?style=flat) ![Cascade](https://img.shields.io/badge/short--long-green?style=flat) | Seed → ShortEval → Select → LongEval → Evolve | [arXiv:2602.22425](https://arxiv.org/abs/2602.22425) |
| [FACT](FACT/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat&logo=c%2B%2B&logoColor=white) ![CUTLASS](https://img.shields.io/badge/CUTLASS-76B900?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![Ablation](https://img.shields.io/badge/ablation-green?style=flat) | Discover → Realize → Compose → Ablate | [FACT Project](https://github.com/Project-FACT/FACT) |
| [GPU Forecasters](GPUForecasters/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![PUCT](https://img.shields.io/badge/PUCT-darkblue?style=flat) ![Forecasting](https://img.shields.io/badge/speedup--forecaster-teal?style=flat) ![Abstain](https://img.shields.io/badge/abstain-orange?style=flat) | TrainForecaster → Select(PUCT) → Forecast/Execute → Update | [arXiv:2605.31464](https://arxiv.org/abs/2605.31464) |
| [KernelBlaster](KernelBlaster/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Experience](https://img.shields.io/badge/experience--memory-orange?style=flat) ![ICRL](https://img.shields.io/badge/in--context--RL-red?style=flat) | Profile/Classify → Retrieve → Apply → Evaluate → Reward/Update | [arXiv:2602.14293](https://arxiv.org/abs/2602.14293) |
| [KernelFoundryDx](KernelFoundryDx/) | ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![Evolutionary](https://img.shields.io/badge/evolutionary-darkblue?style=flat) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![RAG](https://img.shields.io/badge/RAG-orange?style=flat) ![Diagnosis](https://img.shields.io/badge/diagnosis-teal?style=flat) | RAG-Seed → Evolve(Islands) → Evaluate → Diagnose → Migrate | [arXiv:2605.30359](https://arxiv.org/abs/2605.30359) (CUHK/Huawei 2026) |
| [KernelSkill](KernelSkill/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![SkillMemory](https://img.shields.io/badge/skill--memory-orange?style=flat) ![Verification](https://img.shields.io/badge/verification-green?style=flat) | Seed → Review → Repair/Optimize → Profile → UpdateMemory | [arXiv:2603.10085](https://arxiv.org/abs/2603.10085) |
| [StitchCUDA](StitchCUDA/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![Verification](https://img.shields.io/badge/verification-green?style=flat) ![Replanning](https://img.shields.io/badge/adaptive--replanning-orange?style=flat) | Plan → Code → Verify → Replan → Iterate | [arXiv:2603.02637](https://arxiv.org/abs/2603.02637) |
| [Xe-Forge](Xe-Forge/) | ![XPU](https://img.shields.io/badge/Intel--XPU-0071C5?style=flat&logo=intel&logoColor=white) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![CoVeR](https://img.shields.io/badge/CoVeR-purple?style=flat) ![VTune](https://img.shields.io/badge/VTune-555?style=flat) | Stage → Generate → Verify → Refine → Promote | [Xe-Forge Project](https://github.com/intel/Xe-Forge) |
| [Meta-Workflow](_meta/) | ![Tooling](https://img.shields.io/badge/tooling-gray?style=flat) | Research → Model → Assemble → Generate → Validate | — |

### Tag Legend

| Category | Tags |
|----------|------|
| **Backend** | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![ROCm](https://img.shields.io/badge/ROCm-ED1C24?style=flat&logo=amd&logoColor=white) ![SYCL](https://img.shields.io/badge/SYCL-0071C5?style=flat&logo=intel&logoColor=white) ![XPU](https://img.shields.io/badge/Intel--XPU-0071C5?style=flat&logo=intel&logoColor=white) |
| **Kernel Language** | ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![DSL](https://img.shields.io/badge/DSL-darkgreen?style=flat) ![CuTe](https://img.shields.io/badge/CuTe-darkgreen?style=flat) ![TileLang](https://img.shields.io/badge/TileLang-darkgreen?style=flat) ![CUTLASS](https://img.shields.io/badge/CUTLASS-76B900?style=flat) ![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat&logo=c%2B%2B&logoColor=white) |
| **Search Strategy** | ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![MCTS](https://img.shields.io/badge/MCTS-darkblue?style=flat) ![PUCT](https://img.shields.io/badge/PUCT-darkblue?style=flat) ![Evolutionary](https://img.shields.io/badge/MAP--Elites-darkblue?style=flat) ![MAB](https://img.shields.io/badge/MAB--UCB-darkblue?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![CoVeR](https://img.shields.io/badge/CoVeR-purple?style=flat) |
| **Profiler** | ![NCU](https://img.shields.io/badge/NCU-555?style=flat) |
| **Learning Mechanism** | ![Experience](https://img.shields.io/badge/experience--memory-orange?style=flat) ![SkillMemory](https://img.shields.io/badge/skill--memory-orange?style=flat) ![WorldModel](https://img.shields.io/badge/world--model-orange?style=flat) ![MetaPrompt](https://img.shields.io/badge/meta--prompt-orange?style=flat) ![Clustering](https://img.shields.io/badge/clustering-orange?style=flat) ![ICRL](https://img.shields.io/badge/ICRL-blue?style=flat) ![RL-trained](https://img.shields.io/badge/RL--trained-red?style=flat) |
| **Special** | ![Invariants](https://img.shields.io/badge/invariants-red?style=flat) ![Evidence](https://img.shields.io/badge/evidence--driven-green?style=flat) ![Explanation](https://img.shields.io/badge/explanation-teal?style=flat) ![HW-Pruning](https://img.shields.io/badge/HW--pruning-red?style=flat) |

### Methodology Classification Matrix

Use this matrix as the primary taxonomy when adding or reviewing workflows. Backend and language badges are useful, but the more important dimensions are **loop topology**, **authoritative feedback**, and **state carried across attempts**.

| Workflow | Primary category | Search topology | Authoritative feedback | Carried state / memory | Fidelity boundary |
|----------|------------------|-----------------|------------------------|------------------------|-------------------|
| [AccelOpt](AccelOpt/) | `iterative_self_improving` | Beam-style Plan/Execute/Profile/Learn loop | NCU metrics, latency, slow/fast pairs | `experienceMemory`, candidate beam | Faithful CUDA/NCU adaptation of a broader accelerator method |
| [KEET](KEET/) | `single_pass_pipeline` | Source/profile analysis pipeline | NCU profile plus source-grounded hypotheses | Hypothesis verdicts and explanation report | Explanation workflow, not an optimizer |
| [ARGUS](ARGUS/) | `iterative_self_improving` | ICRL Plan/Select/Lower/Validate/Learn loop | Invariant violations, tests, throughput | Planner policy, invariant violation log, candidate beam | Strict use requires executable invariant checker artifacts |
| [AKO4X](AKO4X/) | `iterative_self_improving` | Round-level loop with inner iterations | Smoke/full benchmark, optional NCU | Experience headers, `TRAPS.md`, archive | High-fidelity to AKO4X round/archive protocol |
| [KDA](KDA/) | `iterative_self_improving` | One-candidate-at-a-time evidence loop | Validation command and target metric | Draft/plan docs, candidate records | High-fidelity to KDA agent-flow contract |
| [K-Search](KSearch/) | `tree_exploration` | Co-evolving world-model tree with backtracking | Evaluator speedup and pass/fail result | Decision tree, solution DB, best solution | High-fidelity inference/search translation |
| [AdaExplore](AdaExplore/) | `tree_exploration` | MCTS with large/small expansion iterations | Compile/correctness/performance evaluator | MCTS statistics, diversity pool, skill memory | High-fidelity standalone method translation |
| [KernelFoundry](KernelFoundry/) | `search_based` | MAP-Elites quality-diversity evolution | Compile, correctness, benchmark, descriptor evidence | Elite archive, descriptors, meta-prompts | Needs deterministic archive/descriptor artifacts for strictness |
| [CUDA Agent](CUDAAgent/) | `iterative_self_improving` | Profile/Implement/Verify/Refine loop | Compile, correctness, speedup reward | Interaction history and skill-loop notes | Inference-time loop only; no RL/PPO training reproduction |
| [cuPilot](cuPilot/) | `search_based` | Strategy-level evolutionary loop | NVCC/function check, NCU, roofline evidence | Strategy pool, population, RAG corpus | Strictness depends on concrete roofline/RAG/NCU artifacts |
| [TritorX](TritorX/) | `multi_stage_refinement` | FSM over Generate/Lint/Compile/Test/Debug | Linter, compiler, OpInfo correctness tests | Failure summaries, operator coverage stats | Coverage-first FSM unless real target harness is present |
| [KernelBand](KernelBand/) | `search_based` | Clustered multi-armed bandit with masked UCB | NCU feature vectors, latency reward | Cluster assignments, bandit statistics | Strictness requires real feature-vector and mask artifacts |
| [KernelAgent](KernelAgent/) | `multi_stage_refinement` | Route, parallel seeds, verify, refine, compose | Sandboxed verification pass/fail | Candidate pool, refinement history | Faithful but simplified runtime/process model |
| [STARK](STARK/) | `tree_exploration` | Epsilon-greedy tree with Plan/Code/Debug roles | Compile, correctness, runtime | Tree memory, leaderboard, dynamic contexts | Faithful but runtime role separation is prompt-mediated |
| [ReGraphT](ReGraphT/) | `tree_exploration` | Monte Carlo Graph Search over reasoning graph | Evaluator JSON with speedup/correctness | CUDA reasoning graph, selected paths | Training-free inference/search phase only |
| [Astra](Astra/) | `multi_stage_refinement` | Multi-agent production-kernel optimization loop | Tests, profiling, speedup | Run log, reintegration notes, best result | Idea-preserving unless source repo/runtime is available |
| [CUDA-LLM](CUDALLM/) | `iterative_self_improving` | Feature Search and Reinforcement loop | Compile/correctness/latency reward | Feature catalog, feature scores, candidates | Workflow adaptation; no model training reproduction |
| [CutlassGEMM](CutlassGEMM/) | `iterative_self_improving` | NCU-guided multi-config CUTLASS dispatch tuning loop | SOL-ExecBench correctness/speedup, NCU metrics, MFU | Tile configs, dispatch thresholds, per-M performance regimes | Engineering workflow adaptation for CUTLASS GEMM tuning |
| [ArchAgent](ArchAgent/) | `search_based` | Population evolution with short-to-long evaluation cascade | ChampSim IPC/MPKI and speedup vs baseline | Policy population, fitness history, diversity records | Workflow adaptation for CPU-cache policy search, outside CUDA-kernel execution |
| [FACT](FACT/) | `multi_stage_refinement` | Pattern discovery, realization, composition, and ablation | CUTLASS compile/correctness/performance and ablation speedup | Pattern registry, dependency graph, composed candidates | Faithful but simplified compositional synthesis workflow |
| [GPU Forecasters](GPUForecasters/) | `tree_exploration` | PUCT tree search guided by learned forecaster with abstain | GPU speedup evaluator plus forecast/abstain calibration | Forecaster model, search tree, GPU iterations ledger | Workflow adaptation; surrogate training/calibration must be concrete for strictness |
| [KernelBlaster](KernelBlaster/) | `iterative_self_improving` | MAIC-RL rollouts keyed by hardware performance state | NCU Elapsed Cycles, correctness, reward | Optimization database, trajectories, replay buffer | Faithful in-context RL adaptation with persistent memory |
| [KernelFoundryDx](KernelFoundryDx/) | `search_based` | Multi-island evolutionary Triton search with diagnosis hints | Compile/correctness/speedup plus anti-cheating checks | Island populations, elite archives, hint library | Faithful paper adaptation; no public runtime/source repo available |
| [KernelSkill](KernelSkill/) | `iterative_self_improving` | Seed/review plus repair-or-optimize refinement loop | Compiler/verifier/profiler, speedup, NCU/nsys evidence | Long-term skill library, optimize history, repair chain | Faithful decision-process adaptation; gate is prompt/workflow mediated |
| [StitchCUDA](StitchCUDA/) | `multi_stage_refinement` | Planner/Coder/Verifier with adaptive replanning | Compile, correctness, benchmark speedup | Plan history, failure counters, best candidate | Faithful but simplified three-agent orchestration |
| [Xe-Forge](Xe-Forge/) | `multi_stage_refinement` | Hard-ordered 11-stage CoVeR loops | Intel Triton compile, correctness, speedup, optional VTune | Best-in-stage kernels, promotion history | Workflow adaptation for Intel XPU project pipeline |
| [Meta-Workflow](_meta/) | `tooling` | Research/Model/Assemble/Generate/Validate | Manifest schema and static/semantic checks | Templates, manifests, validation reports | Repository infrastructure, not a paper method |

> Industrial practices (e.g. tiled attention as in FlashInfer, CUTLASS tuning workflows) will be added when they are concrete enough to agentize. Open an issue or PR to nominate the next entry.

---

## Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Workflows-enabled build)
- Per-workflow backend/toolchain. CUDA workflows need NVIDIA GPU + CUDA; Triton/SYCL/XPU/C++ workflows need their corresponding runtime and compiler stack.
- **Recommended for CUDA profiling workflows:** NVIDIA Nsight Compute (`ncu`) — workflows such as AccelOpt are evidence-driven; they avoid guessing bottlenecks
- Per-workflow extras may include: standalone benchmark harness, compiler/build command, experiment output directory (see each file’s header comments)

Run inside a container or dedicated virtual environment when possible; agents may compile code or run `pip install`.

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/qhy991/Awesome-Kernel-Workflows.git
```

### 2. Install as global Claude Code workflows (recommended)

Install once; available in **every project** via `~/.claude/workflows/`:

```bash
git clone https://github.com/qhy991/Awesome-Kernel-Workflows.git
cd Awesome-Kernel-Workflows
./scripts/install-global-workflows.sh
```

Restart Claude Code, browse with `/workflows`, or invoke by `meta.name` (e.g. `/ako4x-kernel-optimizer`).

### 3. Or install into a single kernel project only

Copy into the project’s `.claude/workflows/` directory:

```bash
mkdir -p /path/to/your-kernel-project/.claude/workflows
cp Awesome-Kernel-Workflows/AccelOpt/accelopt-kernel-optimization.js \
   /path/to/your-kernel-project/.claude/workflows/
```

### 4. Invoke from Claude Code

From your kernel project root, start Claude Code and call the workflow (parameters are documented in each `.js` file). AccelOpt example:

Problem-definition entry:

```javascript
Workflow({
  name: 'accelopt-kernel-optimization',
  args: {
    problem_definition: 'Implement y = gelu(x) for a contiguous fp32 tensor',
    language: 'cuda',
    problem_type: 'cuda-kernel-generation',
    target_gpu: 'H100',
    test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
    benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
    iterations: 5,
    seed_candidates: 3,
    exp_dir: '/tmp/kernel_workflow_exp',
  },
})
```

Existing-kernel entry:

```javascript
Workflow({
  name: 'accelopt-kernel-optimization',
  args: {
    kernel_path: '/path/to/kernel.cu',
    op_description: 'Quantized GEMM Q4_0 weight × FP32 activation',
    language: 'cuda',
    problem_type: 'cuda-kernel-optimization',
    target_gpu: 'H100',
    test_command: '<user-provided correctness command with {kernel_path}/{result_path}>',
    benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
    harness_path: '/path/to/harness.cu',
    harness_build_cmd: '<user-provided harness build command>',
    harness_run_args: '',
    kernel_name_regex: 'forward_kernel',
    ncu_binary: '<user-provided ncu binary path>',
    exp_dir: '/path/to/experiment/output',
    iterations: 3,
    breadth: 3,
    samples_per_plan: 2,
  },
})
```

**AccelOpt highlights** (see [`AccelOpt/accelopt-kernel-optimization.js`](AccelOpt/accelopt-kernel-optimization.js)):

- Implements **Plan → Execute → Profile → Summarize → Accumulate Experience → Repeat** from the paper
- Embeds **NCU** in Setup / Evaluate / Learn (`--set full`, source-level analysis, etc.) — profile first, then diagnose, then edit code
- Maintains `experienceMemory`: extracts reusable patterns from slow/fast kernel pairs and NCU metric diffs

Without NCU, pass `test_command` / `benchmark_command` for custom test and benchmark fallbacks.

---

## Related projects

This repo does **not** replace full optimization frameworks. It provides **pluggable Claude Code workflow snippets** that complement:

| Project | Relationship |
|---------|----------------|
| [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) | General agentic kernel optimization (TASK.md + skills); complementary to workflows here |
| [AccelOpt](https://arxiv.org/abs/2511.15915) | Source paper for the first catalogued workflow |
| [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) | Standardized kernel optimization benchmark suite |
| [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) | Industrial operator-level benchmarking and scoring |

---

## Contributing

PRs and issues are welcome, especially:

- **New workflows** tied to a paper, open-source system, or reproducible industrial process — include the source link and a short note on why it fits agents
- **Improvements** to existing workflows: NCU parsing, error handling, defaults, comments
- **Comparisons** on the same kernel across workflows (speedup, iteration count) — `examples/` may be added later

Checklist for contributors:

1. Add `<MethodName>/` with the `*.js` workflow
2. Document **source paper/repo** and **required args** in the file header
3. Update the catalog table in **both** `README.md` and `README.zh-CN.md`
4. Run `scripts/count-workflows.sh` to refresh the header **workflows** badge (`badges/workflows.json`)

---

## License

Per-workflow files should state license terms if derived from upstream projects. The repository default license applies when `LICENSE` is present at the repo root.

---

If this collection helps you operationalize kernel-optimization agents, consider starring the repo and nominating the next method to port.
