<h1 align="center">Awesome Kernel Workflows</h1>

<p align="center">
  将学术界与工业界的 <b>GPU Kernel 优化</b> 方法，沉淀为可复用的 <a href="https://docs.anthropic.com/en/docs/claude-code">Claude Code</a> Workflows
</p>

<p align="center">
  <a href="https://github.com/qhy991/Awesome-Kernel-Workflows"><img src="https://img.shields.io/badge/GitHub-Awesome--Kernel--Workflows-blue?logo=github" alt="GitHub"></a>
  <a href="#catalog"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/qhy991/Awesome-Kernel-Workflows/main/badges/workflows.json" alt="workflow 数量"></a>
  <img src="https://img.shields.io/badge/Claude%20Code-Workflows-7C3AED" alt="Claude Code Workflows">
  <img src="https://img.shields.io/badge/GPU-CUDA%20%7C%20Triton%20%7C%20DSL-green" alt="GPU Kernels">
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

---

## 这是什么？

GPU kernel 优化领域每年都有大量高质量工作：自改进循环（如 AccelOpt）、agentic 优化框架（如 AKO）、工业界 SOTA 实现（FlashInfer、CUTLASS 等）背后的工程方法论。这些工作往往以**论文 + 代码仓库**的形式存在，直接交给 coding agent 时，agent 很难稳定复现其**决策流程**（何时 profile、如何读 NCU、怎样积累 experience 等）。

本仓库的目标是：

1. **收集** — 梳理学术界与工业界与 GPU kernel 优化相关、且适合 agent 执行的方法论与系统；
2. **提炼** — 把每种方法的核心循环（plan / execute / profile / learn 等）抽象成清晰阶段；
3. **落地** — 写成 [Claude Code Workflow](https://docs.anthropic.com/en/docs/claude-code/workflows)（`.js`），可在你的 kernel 工程目录中直接调用。

每个子目录对应**一种方法或一条工作流**，内含 workflow 定义、参数说明，以及（逐步补充的）原文链接与使用示例。

---

## 目录结构

```
Awesome-Kernel-Workflows/
├── README.md                    # English
├── README.zh-CN.md              # 简体中文（本文件）
├── AccelOpt/                    # 自改进循环 + NCU 性能分析
│   ├── accelopt-kernel-optimization.js
│   └── README.md
├── KEET/                        # NCU profile → 性能解释流水线
│   ├── keet-kernel-explanation.js
│   └── README.md
├── ARGUS/                       # 数据流不变量引导优化
│   ├── argus-kernel-optimization.js
│   └── README.md
├── AKO4X/                       # 多轮闭环 + 经验累积
│   ├── ako4x-kernel-optimizer.js
│   └── README.md
├── KDA/                         # 证据驱动的草案-计划-实现-验证-决策
│   ├── kda-kernel-workflow.js
│   └── README.md
├── AdaExplore/                  # MCTS 树搜索 + 失败驱动技能记忆
│   ├── adaexplore-kernel-optimization.js
│   └── README.md
├── KernelFoundry/               # MAP-Elites 进化 + 元提示词共进化
│   ├── kernelfoundry-kernel-optimization.js
│   └── README.md
├── KSearch/                     # 世界模型引导的树搜索
│   └── ksearch-kernel-optimization.js
├── KernelBand/                  # 多臂老虎机 + 硬件感知剪枝 + 聚类
│   ├── kernelband-kernel-optimization.js
│   └── README.md
├── KernelAgent/                 # 多 Agent Triton 合成 + 并行验证
│   ├── kernelagent-triton-synthesis.js
│   ├── manifest.yaml
│   └── README.md
├── STARK/                       # 多 Agent 协作 + 树搜索 + Grounded Instruction
│   ├── stark-kernel-optimization.js
│   ├── manifest.yaml
│   └── README.md
├── ReGraphT/                    # CUDA Reasoning Graph + Monte Carlo Graph Search
│   ├── regrapht-kernel-optimization.js
│   ├── manifest.yaml
│   └── README.md
├── Astra/                       # 多 Agent 优化已有 CUDA kernel
│   ├── astra-kernel-optimization.js
│   ├── manifest.yaml
│   └── README.md
├── AutoMegaKernel/              # AMK megakernel schedule search adapter
│   ├── automegakernel-megakernel-optimization.js
│   ├── manifest.yaml
│   └── README.md
├── CUDALLM/                     # CUDA 生成的 Feature Search and Reinforcement
│   ├── cudallm-fsr-kernel-generation.js
│   ├── manifest.yaml
│   └── README.md
├── CutlassGEMM/                 # CUTLASS GEMM 多配置调度调优
│   ├── cutlass-gemm-optimization.js
│   └── README.md
├── GemmPTX/                     # 基于 PTX/SASS 证据的 GEMM 指令路径优化
│   ├── gemmptx-gemm-optimization.js
│   ├── manifest.yaml
│   └── README.md
├── FACT/                        # CUTLASS 模式组合式合成
│   ├── fact-kernel-optimization.js
│   └── README.md
├── GPUForecasters/              # 带加速预测器的 PUCT 搜索
│   ├── gpuforecasters-kernel-optimization.js
│   └── README.md
├── KernelBlaster/               # CUDA 内核的记忆增强上下文内 RL
│   ├── kernelblaster-kernel-optimization.js
│   └── README.md
├── KernelFoundryDx/             # 诊断驱动的多岛 Triton 进化
│   ├── kernelfoundrydx-kernel-optimization.js
│   └── README.md
├── KernelSkill/                 # 双层记忆的多 Agent CUDA 优化
│   ├── kernelskill-kernel-optimization.js
│   └── README.md
├── StitchCUDA/                  # Planner/Coder/Verifier CUDA 合成
│   ├── stitchcuda-kernel-optimization.js
│   └── README.md
├── WarpSpeed/                   # 多 GPU 节点上可回退的并行内核搜索
│   ├── warpspeed-kernel-search.js
│   ├── infra/                   # gpu_run 互斥启动器、两级基准、NCU 剖析
│   ├── tools/                   # wsdb CLI（SQLite 检查点树）、配置渲染、NCU 解析
│   ├── wiki/                    # 技术页（按标签索引、按架构限定）
│   ├── harness-template/        # 工程自有正确性 harness 脚手架
│   ├── tests/                   # mock GPU 套件 + vm 干跑 + 节点验收
│   └── README.md
├── Xe-Forge/                    # Intel XPU 多阶段 CoVeR 优化
│   ├── xe-forge-kernel-optimization.js
│   └── README.md
├── _meta/                       # Meta-workflow：论文 → 工作流生成
│   ├── README.md
│   ├── tools/                   # 生成器 + 验证器
│   ├── templates/               # 拓扑模板（迭代/搜索/单遍/树）
│   └── manifests/               # 结构化方法描述 (YAML)
└── <MoreMethods>/               # 欢迎 PR 补充
    └── *.js
```

Workflow 文件遵循 Claude Code 约定：导出 `meta`（名称、描述、阶段），并通过 `phase()` / `agent()` 编排多步 agent 任务。

---

## Workflow 适配矩阵

每个顶层 workflow 在 `<Workflow>/manifest.yaml` 中声明其 backend 能力（唯一权威来源）。下表展示 **driver 路径能力**——workflow 在传入 `args.backend` + `args.backend_dir` 时支持的后端。在无 driver 目录时（legacy 路径），`.js` 的 `WORKFLOW_SUITABILITY` 守卫可能接受更窄的子集；差异详见各 workflow 的 `manifest.yaml` `notes` 字段。该检查是保守的：workflow 不会从自然语言 `problem_definition` 推断语言或问题类型。

| Workflow | 支持语言 / 后端 | 支持的 `problem_type` | 适合处理 | 不适合 |
|----------|-----------------|-----------------------|----------|--------|
| [AscendC](AscendC/) | AscendC / Ascend（msprof + `ascendc_direct_launch`） | `ascend-kernel-optimization`, `ascend-kernel-generation` | 昇腾 910B NPU 上的 kernel 优化/生成，经 msprof 与 substrate ascend backend | CUDA/Triton/ROCm/Metal 目标——请用各自的 workflow |
| [AccelOpt](AccelOpt/) | CUDA（默认）· Triton via driver（vendor-locked: ncu） | `cuda-kernel-optimization`, `cuda-kernel-generation` | 已有 CUDA kernel，或先生成 CUDA seed 再用 NCU/benchmark 优化 | Triton/SYCL/XPU 任务，或没有 benchmark/profile 契约 |
| [KEET](KEET/) | CUDA（vendor-locked: ncu） | `performance-explanation` | 基于 CUDA 源码和 Nsight Compute profile 做性能解释 | 生成或优化 kernel，尤其是没有 profile artifact 时 |
| [ARGUS](ARGUS/) | ROCm/CUDA/Triton/ARGUS-DSL（legacy；driver 待迁移） | `invariant-guided-kernel-optimization`, `gpu-kernel-optimization` | 带 invariant checker / 测试反馈的 GPU kernel 优化 | 没有不变量或验证证据的任务 |
| [AKO4X](AKO4X/) | Triton（默认）· CUDA/CuTe/TileLang/C++/PyTorch via driver | `gpu-kernel-optimization`, `kernel-generation` | 多轮 benchmark 驱动的 GPU kernel/DSL 优化 | 非 kernel 应用代码，或不支持的后端工具链 |
| [KDA](KDA/) | CUDA（默认）· Triton via driver（experimental）· Ascend via substrate（faithful but simplified） | `cuda-kernel-optimization`, `cuda-kernel-generation`, `ascend-kernel-optimization`, `ascend-kernel-generation` | 证据驱动的 CUDA 实现、验证和优化循环；Ascend 经 substrate ascend backend | 没有 substrate driver 或验证契约的后端 |
| [K-Search](KSearch/) | Triton（默认）· CUDA/Python via driver | `gpu-kernel-optimization`, `kernel-search` | 带 evaluator/benchmark 契约的 world-model tree search | 没有可执行 evaluator 反馈的任务 |
| [AdaExplore](AdaExplore/) | CUDA · Triton via driver（experimental） | `triton-kernel-optimization`, `triton-kernel-generation` | 从 PyTorch 算子规格生成/优化 Triton kernel，使用 MCTS 和失败记忆 | 直接 CUDA/CUTLASS/SYCL 优化 |
| [KernelFoundry](KernelFoundry/) | CUDA（默认）· Triton via driver（experimental） | `gpu-kernel-optimization`, `kernel-generation`, `kernel-search` | MAP-Elites 质量-多样性 kernel 搜索，需要 descriptor/archive 反馈 | 没有 archive 状态的单次确定性 patch |
| [CUDA Agent](CUDAAgent/) | CUDA（vendor-locked: ncu） | `cuda-kernel-generation`, `cuda-kernel-optimization` | PyTorch model/operator 到自定义 CUDA ops 和 bindings | Triton/SYCL/CUTLASS-only 任务 |
| [cuPilot](cuPilot/) | CUDA（vendor-locked: ncu） | `cuda-kernel-optimization` | 带 roofline/profiler 证据的策略级 CUDA 进化 | 非 CUDA kernels |
| [TritorX](TritorX/) | Triton（vendor-locked: linter） | `aten-triton-operator-generation`, `operator-generation` | ATen/Triton operator coverage 生成，以及 compile/lint/test/debug 循环 | 以性能为首要目标的 CUDA 调优 |
| [KernelBand](KernelBand/) | Triton（默认）· CUDA via driver | `gpu-kernel-optimization`, `kernel-search` | 使用硬件特征和 profiling 的 bandit-guided search | 缺少类似 profiling/evaluator 证据的后端 |
| [KernelAgent](KernelAgent/) | Triton（默认）· CUDA via driver（experimental） | `triton-kernel-generation`, `operator-generation` | 带 PyTorch-style verification harness 的 Triton 合成 | CUDA/C++/CUTLASS kernels |
| [STARK](STARK/) | CUDA（默认）· Triton via driver（experimental） | `cuda-kernel-optimization`, `kernel-search` | 多 Agent 规划/调试的 CUDA tree-search refinement | 非 CUDA 后端，除非有对应 code-context adapter |
| [ReGraphT](ReGraphT/) | CUDA（默认）· Triton via driver（experimental） | `cuda-kernel-optimization`, `kernel-search` | CUDA reasoning graph 和 Monte Carlo graph search | 非 CUDA 优化轨迹 |
| [Astra](Astra/) | CUDA（默认）· Triton via driver（experimental） | `cuda-kernel-optimization` | 已有生产 CUDA/PyBind kernel，具备 tests 和 profiling | 从零生成 Triton/SYCL |
| [AutoMegaKernel](AutoMegaKernel/) | CUDA via AutoMegaKernel（method-intrinsic；外部 harness） | `amk-schedule-search`, `megakernel-synthesis`, `llama-megakernel-optimization` | 已有 AMK checkout；Llama-family CUDA megakernel 的 `ScheduleConfig` + `kernel_knobs` schedule search | 普通 standalone CUDA/Triton/CUTLASS kernel；缺少 AMK checkout；high-batch serving throughput |
| [CUDA-LLM](CUDALLM/) | CUDA（默认）· Triton via driver（experimental） | `cuda-kernel-generation`, `cuda-kernel-optimization` | 基于 task spec 的 CUDA feature search / reinforcement | 非 CUDA 输出语言 |
| [CutlassGEMM](CutlassGEMM/) | CUTLASS / C++（method-intrinsic） | `cutlass-gemm-optimization` | CUTLASS GEMM / SOL-ExecBench dispatch 调优 | CUTLASS GEMM 之外的通用 elementwise/attention kernel |
| [GemmPTX](GemmPTX/) | CUDA/CuTe/CUTLASS（vendor-locked: PTX/SASS 反汇编） | `gemm-ptx-optimization`, `cuda-gemm-ptx-optimization`, `gemm-instruction-optimization` | 已有 GEMM kernel，且需要用 PTX/SASS 证据证明指令路径选择 | 通用 compute-bound kernel；缺少反汇编/benchmark/正确性契约 |
| [FACT](FACT/) | CUTLASS / C++（method-intrinsic） | `cutlass-pattern-synthesis`, `cutlass-gemm-optimization` | CUTLASS pattern 发现、实现、组合和消融 | 独立 Triton/SYCL kernels |
| [GPU Forecasters](GPUForecasters/) | CUDA（vendor-locked: ncu） | `cuda-kernel-optimization`, `kernel-search` | 带 speedup forecaster 和 execute/abstain 反馈的 CUDA/GPU 搜索 | 没有 GPU 执行或预测器校准反馈的任务 |
| [InPlacePatch](InPlacePatch/) | CUDA/ROCm（vendor-locked: nvcc/hipcc） | `embedded-kernel-optimization` | 基于项目原生 build/test/benchmark 的字节精确 in-place kernel patch | 基于 exp_dir 的独立 kernel 工作流 |
| [KernelBlaster](KernelBlaster/) | CUDA（vendor-locked: ncu） | `cuda-kernel-optimization` | 基于 NCU elapsed cycles 和持久记忆的 CUDA 优化 | 非 CUDA 后端 |
| [KernelFoundryDx](KernelFoundryDx/) | Triton（method-intrinsic） | `triton-kernel-optimization`, `triton-kernel-generation` | PyTorch reference 到 Triton 的多岛进化 | CUDA/CUTLASS/SYCL kernels |
| [LlamacppEmbeddedSearch](LlamacppEmbeddedSearch/) | llama.cpp ggml-cuda（method-intrinsic） | `embedded-kernel-search` | 嵌入 llama.cpp ggml-cuda 的 kernel 的多变体并行搜索 | 独立 CUDA/Triton kernel 优化 |
| [KernelSkill](KernelSkill/) | CUDA（默认）· Triton via driver（experimental） | `cuda-kernel-optimization`, `cuda-kernel-generation` | 从 PyTorch reference 生成/优化 CUDA custom kernels，并使用 memory/profiler guidance | Triton/SYCL/Metal 任务 |
| [StitchCUDA](StitchCUDA/) | CUDA（默认）· Triton via driver（experimental） | `cuda-kernel-generation`, `cuda-kernel-optimization` | Planner/Coder/Verifier CUDA 合成和重规划 | 非 CUDA 后端 |
| [WarpSpeed](WarpSpeed/) | CUDA（vendor-locked：需要 NCU/compute-sanitizer） | `cuda-kernel-optimization`, `kernel-search` | 独占多 GPU 节点上的持续优化战役：检查点树、GPU 互斥、两级基准、可回退 | 一次性补丁、共享 GPU、缺少 NCU/sanitizer 的主机 |
| [Xe-Forge](Xe-Forge/) | Intel XPU（vendor-locked: xpu） | `xpu-kernel-optimization`, `triton-kernel-optimization` | Intel XPU CoVeR staged refinement | NVIDIA CUDA-only 调优 |
| [Generalist](Generalist/) | CUDA/Triton/MetaX（默认）· Ascend via substrate（faithful but simplified） | `cuda-kernel-generation`, `cuda-kernel-optimization`, `ascend-kernel-generation`, `ascend-kernel-optimization` | 后端无关的 substrate solver（beam）；默认单 solver 基线；Ascend 经 substrate ascend backend | 需要非 substrate 测量信号的任务 |
| [Meta-Workflow](_meta/) | Tooling | N/A | 生成和验证 workflow 定义 | 直接 kernel 优化 |

---

## 已收录 Workflows {#catalog}

| 方法 | 标签 | 核心循环 | 论文 / 项目 |
|------|------|----------|-------------|
| [AscendC](AscendC/) | ![AscendC](https://img.shields.io/badge/AscendC-0066CC?style=flat) ![msprof](https://img.shields.io/badge/msprof-555?style=flat) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) | Setup → Generate → Evaluate → Optimize（msprof）→ Report | 经验证的 session-local 变体（[AscendC/README.md](AscendC/README.md)）；已提升 upstream |
| [AccelOpt](AccelOpt/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Experience](https://img.shields.io/badge/experience--memory-orange?style=flat) | Plan → Execute → Profile → Learn → Iterate | [arXiv:2511.15915](https://arxiv.org/abs/2511.15915)（MLSys 2026） |
| [KEET](KEET/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![Explanation](https://img.shields.io/badge/explanation-teal?style=flat) | 源码检查 → Profile 检查 → 聚合 → 审查 | [arXiv:2605.04467](https://arxiv.org/abs/2605.04467)（UMD/NVIDIA/LLNL 2026） |
| [ARGUS](ARGUS/) | ![ROCm](https://img.shields.io/badge/ROCm-ED1C24?style=flat&logo=amd&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![DSL](https://img.shields.io/badge/DSL-darkgreen?style=flat) ![ICRL](https://img.shields.io/badge/ICRL-blue?style=flat) ![Invariants](https://img.shields.io/badge/invariants-red?style=flat) | Plan → Select → Lower → Validate → Learn | [arXiv:2604.18616](https://arxiv.org/abs/2604.18616)（CausalFlow/港科大/斯坦福 2026） |
| [AKO4X](AKO4X/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![CuTe](https://img.shields.io/badge/CuTe-darkgreen?style=flat) ![TileLang](https://img.shields.io/badge/TileLang-darkgreen?style=flat) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Experience](https://img.shields.io/badge/experience--memory-orange?style=flat) | Round-Init → Iterate → Archive → Retrospect | [AKO 项目](https://tongminglaic.github.io/AKO) |
| [KDA](KDA/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Evidence](https://img.shields.io/badge/evidence--driven-green?style=flat) | Inspect → Plan → Implement → Validate → Decide | [MIT HAN Lab](https://github.com/mit-han-lab/kernel-design-agents) |
| [K-Search](KSearch/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![WorldModel](https://img.shields.io/badge/world--model-orange?style=flat) | Init Tree → Select → Generate → Evaluate → Refine/Backtrack | [arXiv:2602.19128](https://arxiv.org/abs/2602.19128) |
| [AdaExplore](AdaExplore/) | ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![MCTS](https://img.shields.io/badge/MCTS-darkblue?style=flat) ![SkillMemory](https://img.shields.io/badge/skill--memory-orange?style=flat) | Select → Expand → Evaluate → Backpropagate | [arXiv:2604.16625](https://arxiv.org/abs/2604.16625) |
| [KernelFoundry](KernelFoundry/) | ![SYCL](https://img.shields.io/badge/SYCL-0071C5?style=flat&logo=intel&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![Evolutionary](https://img.shields.io/badge/MAP--Elites-darkblue?style=flat) ![MetaPrompt](https://img.shields.io/badge/meta--prompt-orange?style=flat) | Select → Vary → Evaluate → Insert → Evolve-Prompts | [arXiv:2603.12440](https://arxiv.org/abs/2603.12440)（Intel 2026） |
| [CUDA Agent](CUDAAgent/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![RL-trained](https://img.shields.io/badge/RL--trained-red?style=flat) | Profile → Implement → Verify → Refine | [arXiv:2602.24286](https://arxiv.org/abs/2602.24286)（字节跳动/清华 2026） |
| [cuPilot](cuPilot/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![Evolutionary](https://img.shields.io/badge/evolutionary-darkblue?style=flat) ![Roofline](https://img.shields.io/badge/roofline-teal?style=flat) ![RAG](https://img.shields.io/badge/RAG-orange?style=flat) | Strategize → Translate → Revise → Evolve | [arXiv:2512.16465](https://arxiv.org/abs/2512.16465)（东南大学/清华 2025） |
| [TritorX](TritorX/) | ![ASIC](https://img.shields.io/badge/ASIC%2FNPU-333?style=flat) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![FSM](https://img.shields.io/badge/FSM-blue?style=flat) ![Linter](https://img.shields.io/badge/linter-green?style=flat) ![Coverage](https://img.shields.io/badge/coverage--first-teal?style=flat) | Generate → Lint → Compile/Test → Debug（循环） | [arXiv:2512.10977](https://arxiv.org/abs/2512.10977)（Meta 2025） |
| [KernelBand](KernelBand/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![MAB](https://img.shields.io/badge/MAB--UCB-darkblue?style=flat) ![Clustering](https://img.shields.io/badge/clustering-orange?style=flat) ![HW-Pruning](https://img.shields.io/badge/HW--pruning-red?style=flat) | Profile → Cluster → Select(UCB) → Generate → Evaluate → Update | [arXiv:2511.18868](https://arxiv.org/abs/2511.18868)（北大 2026） |
| [KernelAgent](KernelAgent/) | ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![Verification](https://img.shields.io/badge/verification-green?style=flat) | 路由 → 并行生成 → 验证 → 修复 → 组合 | [PyTorch Blog](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/)（PyTorch Labs 2025） |
| [STARK](STARK/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![Grounded](https://img.shields.io/badge/grounded--instruction-green?style=flat) ![DynamicContext](https://img.shields.io/badge/dynamic--context-orange?style=flat) | Setup → Select(ε-greedy) → Plan/Code/Debug → Evaluate → Update | [arXiv:2510.16996](https://arxiv.org/abs/2510.16996)（Meta/Duke 2025） |
| [ReGraphT](ReGraphT/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![RAG](https://img.shields.io/badge/RAG-orange?style=flat) ![ReasoningGraph](https://img.shields.io/badge/reasoning--graph-teal?style=flat) ![MCGS](https://img.shields.io/badge/MCGS-darkblue?style=flat) | BuildGraph → Select(MCGS) → Generate → Evaluate → UpdateGraph | [arXiv:2510.19873](https://arxiv.org/abs/2510.19873)（中科院/华南理工 2025） |
| [Astra](Astra/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![Profiling](https://img.shields.io/badge/profiling-green?style=flat) ![SGLang](https://img.shields.io/badge/SGLang-orange?style=flat) | Setup → Test/Profile → Plan → Code → Evaluate → Record | [arXiv:2509.07506](https://arxiv.org/abs/2509.07506)（Stanford/SJTU/NJU 2025） |
| [AutoMegaKernel](AutoMegaKernel/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Megakernel](https://img.shields.io/badge/megakernel-333?style=flat) ![ScheduleSearch](https://img.shields.io/badge/schedule--search-darkblue?style=flat) ![ExternalHarness](https://img.shields.io/badge/external--harness-teal?style=flat) ![Verification](https://img.shields.io/badge/verification-green?style=flat) | Doctor → ProposeSurface → EvalBaseline → Loop/Autoresearch → Audit | [arXiv:2606.09682](https://arxiv.org/abs/2606.09682) / [AMK](https://github.com/RightNow-AI/AutoMegaKernel) |
| [CUDA-LLM](CUDALLM/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![FeatureSearch](https://img.shields.io/badge/feature--search-teal?style=flat) ![Reinforcement](https://img.shields.io/badge/reinforcement-red?style=flat) | Catalog → SelectFeatures → Generate → Evaluate → Reinforce | [arXiv:2506.09092](https://arxiv.org/abs/2506.09092)（2025） |
| [CutlassGEMM](CutlassGEMM/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat&logo=c%2B%2B&logoColor=white) ![CUTLASS](https://img.shields.io/badge/CUTLASS-76B900?style=flat) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Dispatch](https://img.shields.io/badge/multi--config--dispatch-teal?style=flat) | 分析 → 生成配置 → Profile(NCU) → 调优调度 → 验证 | [CUTLASS](https://github.com/NVIDIA/cutlass) / [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) |
| [GemmPTX](GemmPTX/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![PTX](https://img.shields.io/badge/PTX%2FSASS-555?style=flat) ![GEMM](https://img.shields.io/badge/GEMM-333?style=flat) ![Evidence](https://img.shields.io/badge/disassembly--evidence-green?style=flat) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) | HardwareCensus → GEMMSignature → InstructionPlan → DisassembleVerify → Profile → Decide | AKW 工程 workflow |
| [FACT](FACT/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat&logo=c%2B%2B&logoColor=white) ![CUTLASS](https://img.shields.io/badge/CUTLASS-76B900?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![Ablation](https://img.shields.io/badge/ablation-green?style=flat) | 发现 → 实现 → 组合 → 消融 | [FACT 项目](https://github.com/Project-FACT/FACT) |
| [GPU Forecasters](GPUForecasters/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![PUCT](https://img.shields.io/badge/PUCT-darkblue?style=flat) ![Forecasting](https://img.shields.io/badge/speedup--forecaster-teal?style=flat) ![Abstain](https://img.shields.io/badge/abstain-orange?style=flat) | 训练预测器 → Select(PUCT) → 预测/执行 → 更新 | [arXiv:2605.31464](https://arxiv.org/abs/2605.31464) |
| [KernelBlaster](KernelBlaster/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Experience](https://img.shields.io/badge/experience--memory-orange?style=flat) ![ICRL](https://img.shields.io/badge/in--context--RL-red?style=flat) | Profile/分类 → 检索 → 应用 → 评估 → 奖励/更新 | [arXiv:2602.14293](https://arxiv.org/abs/2602.14293) |
| [KernelFoundryDx](KernelFoundryDx/) | ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![Evolutionary](https://img.shields.io/badge/evolutionary-darkblue?style=flat) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![RAG](https://img.shields.io/badge/RAG-orange?style=flat) ![Diagnosis](https://img.shields.io/badge/diagnosis-teal?style=flat) | RAG 种子 → 岛屿进化 → 评估 → 诊断 → 迁移 | [arXiv:2605.30359](https://arxiv.org/abs/2605.30359)（CUHK/Huawei 2026） |
| [KernelSkill](KernelSkill/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![SkillMemory](https://img.shields.io/badge/skill--memory-orange?style=flat) ![Verification](https://img.shields.io/badge/verification-green?style=flat) | Seed → Review → 修复/优化 → Profile → 更新记忆 | [arXiv:2603.10085](https://arxiv.org/abs/2603.10085) |
| [StitchCUDA](StitchCUDA/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![MultiAgent](https://img.shields.io/badge/multi--agent-teal?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![Verification](https://img.shields.io/badge/verification-green?style=flat) ![Replanning](https://img.shields.io/badge/adaptive--replanning-orange?style=flat) | 规划 → 编码 → 验证 → 重规划 → 迭代 | [arXiv:2603.02637](https://arxiv.org/abs/2603.02637) |
| [WarpSpeed](WarpSpeed/) | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![NCU](https://img.shields.io/badge/NCU-555?style=flat) ![Experience](https://img.shields.io/badge/experience--memory-orange?style=flat) ![Rewind](https://img.shields.io/badge/rewind-red?style=flat) ![CrossReview](https://img.shields.io/badge/cross--model--review-green?style=flat) | 规划 → 并行生成 → A/B 初筛 → 锁频确认 → NCU 剖析 → 记录 → 验尸/回退 | [设计说明](WarpSpeed/README.md) |
| [Xe-Forge](Xe-Forge/) | ![XPU](https://img.shields.io/badge/Intel--XPU-0071C5?style=flat&logo=intel&logoColor=white) ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![CoVeR](https://img.shields.io/badge/CoVeR-purple?style=flat) ![VTune](https://img.shields.io/badge/VTune-555?style=flat) | 阶段 → 生成 → 验证 → 优化 → 提升 | [Xe-Forge 项目](https://github.com/intel/Xe-Forge) |
| [Meta-Workflow](_meta/) | ![Tooling](https://img.shields.io/badge/tooling-gray?style=flat) | Research → Model → Assemble → Generate → Validate | — |

### 标签说明

| 分类 | 标签 |
|------|------|
| **后端** | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![ROCm](https://img.shields.io/badge/ROCm-ED1C24?style=flat&logo=amd&logoColor=white) ![SYCL](https://img.shields.io/badge/SYCL-0071C5?style=flat&logo=intel&logoColor=white) ![XPU](https://img.shields.io/badge/Intel--XPU-0071C5?style=flat&logo=intel&logoColor=white) |
| **内核语言** | ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![DSL](https://img.shields.io/badge/DSL-darkgreen?style=flat) ![CuTe](https://img.shields.io/badge/CuTe-darkgreen?style=flat) ![TileLang](https://img.shields.io/badge/TileLang-darkgreen?style=flat) ![CUTLASS](https://img.shields.io/badge/CUTLASS-76B900?style=flat) ![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat&logo=c%2B%2B&logoColor=white) |
| **搜索策略** | ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![MCTS](https://img.shields.io/badge/MCTS-darkblue?style=flat) ![PUCT](https://img.shields.io/badge/PUCT-darkblue?style=flat) ![Evolutionary](https://img.shields.io/badge/MAP--Elites-darkblue?style=flat) ![MAB](https://img.shields.io/badge/MAB--UCB-darkblue?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) ![CoVeR](https://img.shields.io/badge/CoVeR-purple?style=flat) |
| **分析工具** | ![NCU](https://img.shields.io/badge/NCU-555?style=flat) |
| **学习机制** | ![Experience](https://img.shields.io/badge/experience--memory-orange?style=flat) ![SkillMemory](https://img.shields.io/badge/skill--memory-orange?style=flat) ![WorldModel](https://img.shields.io/badge/world--model-orange?style=flat) ![MetaPrompt](https://img.shields.io/badge/meta--prompt-orange?style=flat) ![Clustering](https://img.shields.io/badge/clustering-orange?style=flat) ![ICRL](https://img.shields.io/badge/ICRL-blue?style=flat) ![RL-trained](https://img.shields.io/badge/RL--trained-red?style=flat) |
| **特殊** | ![Invariants](https://img.shields.io/badge/invariants-red?style=flat) ![Evidence](https://img.shields.io/badge/evidence--driven-green?style=flat) ![Explanation](https://img.shields.io/badge/explanation-teal?style=flat) ![HW-Pruning](https://img.shields.io/badge/HW--pruning-red?style=flat) |

### 方法学分类矩阵

新增或审查 workflow 时，优先使用这张矩阵。CUDA / Triton / ROCm 等后端标签很有用，但真正决定方法差异的是 **循环拓扑**、**权威反馈信号** 和 **跨轮携带的状态/记忆**。

| Workflow | 主要类别 | 搜索拓扑 | 权威反馈信号 | 状态 / 记忆 | Fidelity 边界 |
|----------|----------|----------|--------------|-------------|---------------|
| [AscendC](AscendC/) | `iterative_self_improving` | plan→generate→evaluate→optimize（msprof）循环 | msprof 瓶颈类型 + eval_single_runner 计时（ascendc_direct_launch） | 候选提交、最优路径、停滞计数 | Substrate reference——substrate ascend backend 拥有测量信号 |
| [AccelOpt](AccelOpt/) | `iterative_self_improving` | Beam 式 Plan/Execute/Profile/Learn 循环 | NCU 指标、延迟、慢/快对 | `experienceMemory`、candidate beam | 对更广义 accelerator 方法的 CUDA/NCU 适配 |
| [KEET](KEET/) | `single_pass_pipeline` | 源码/Profile 分析流水线 | NCU profile + 源码假设 | 假设裁决、解释报告 | 解释 workflow，不是优化器 |
| [ARGUS](ARGUS/) | `iterative_self_improving` | ICRL Plan/Select/Lower/Validate/Learn 循环 | 不变量违规、单测、吞吐量 | planner policy、不变量违规日志、候选 beam | 严格复现需要可执行 invariant checker 证据 |
| [AKO4X](AKO4X/) | `iterative_self_improving` | 外层 round + 内层 iteration | smoke/full benchmark、可选 NCU | 经验 header、`TRAPS.md`、archive | 高保真对应 AKO4X round/archive 协议 |
| [KDA](KDA/) | `iterative_self_improving` | 单候选证据驱动循环 | validation command、目标指标 | draft/plan 文档、候选记录 | 高保真对应 KDA agent-flow 契约 |
| [K-Search](KSearch/) | `tree_exploration` | 共演化 world-model tree + 回溯 | evaluator speedup、pass/fail | decision tree、solution DB、best solution | 高保真 inference/search 翻译 |
| [AdaExplore](AdaExplore/) | `tree_exploration` | MCTS + large/small expansion | 编译/正确性/性能 evaluator | MCTS 统计、多样性池、skill memory | 高保真 standalone 方法翻译 |
| [KernelFoundry](KernelFoundry/) | `search_based` | MAP-Elites 质量-多样性进化 | 编译、正确性、benchmark、descriptor 证据 | elite archive、descriptor、meta-prompt | 严格性依赖确定性 archive/descriptor 证据 |
| [CUDA Agent](CUDAAgent/) | `iterative_self_improving` | Profile/Implement/Verify/Refine 循环 | 编译、正确性、speedup reward | 交互历史、skill-loop notes | 只覆盖推理时循环，不复现 RL/PPO 训练 |
| [cuPilot](cuPilot/) | `search_based` | strategy-level evolution | NVCC/function check、NCU、roofline 证据 | strategy pool、population、RAG corpus | 严格性依赖真实 roofline/RAG/NCU 产物 |
| [TritorX](TritorX/) | `multi_stage_refinement` | Generate/Lint/Compile/Test/Debug FSM | linter、compiler、OpInfo 正确性测试 | failure summary、operator coverage stats | 无真实目标 harness 时只能称 TritorX-style FSM |
| [KernelBand](KernelBand/) | `search_based` | 聚类多臂老虎机 + Masked UCB | NCU feature vector、latency reward | cluster assignment、bandit statistics | 严格性需要真实 feature-vector 和 mask 产物 |
| [KernelAgent](KernelAgent/) | `multi_stage_refinement` | Route、并行 seed、verify、refine、compose | sandboxed verification pass/fail | candidate pool、refinement history | 方法形状保真，但 runtime/process 有简化 |
| [STARK](STARK/) | `tree_exploration` | epsilon-greedy tree + Plan/Code/Debug 角色 | 编译、正确性、runtime | tree memory、leaderboard、dynamic contexts | 方法形状保真，角色分离主要由 prompt 表达 |
| [ReGraphT](ReGraphT/) | `tree_exploration` | reasoning graph 上的 MCGS | evaluator JSON：speedup/correctness | CUDA reasoning graph、selected paths | 只覆盖 training-free inference/search 阶段 |
| [Astra](Astra/) | `multi_stage_refinement` | 多 Agent 生产内核优化循环 | tests、profiling、speedup | run log、reintegration notes、best result | 无源 repo/runtime 时属于 idea-preserving |
| [AutoMegaKernel](AutoMegaKernel/) | `search_based` | 委托给 AMK loop/autoresearch 的结构化 ScheduleConfig + kernel_knobs 搜索 | AMK validate-before-launch、全模型正确性、latency/roofline verdict | best config、rows/results.tsv、flywheel corpus | 严格 adapter 到外部 AMK harness；不是独立重写 |
| [CUDA-LLM](CUDALLM/) | `iterative_self_improving` | Feature Search and Reinforcement 循环 | 编译/正确性/latency reward | feature catalog、feature scores、candidates | workflow adaptation，不复现模型训练 |
| [CutlassGEMM](CutlassGEMM/) | `iterative_self_improving` | NCU 引导的 CUTLASS 多配置 dispatch 调优循环 | SOL-ExecBench 正确性/speedup、NCU 指标、MFU | tile configs、dispatch thresholds、per-M performance regimes | 面向 CUTLASS GEMM 调优的工程 workflow adaptation |
| [GemmPTX](GemmPTX/) | `iterative_self_improving` | 单候选指令假设循环，profile 前必须先通过 PTX/SASS 验证 | compile/test/benchmark/disassemble 命令，可选 NCU/profile 指标 | candidate history、verified instruction paths、hypothesis_not_realized dead ends | 面向 GEMM 指令路径调优的 AKW 原创 workflow adaptation |
| [FACT](FACT/) | `multi_stage_refinement` | 模式发现、实现、组合与消融 | CUTLASS 编译/正确性/性能和消融 speedup | pattern registry、dependency graph、composed candidates | 组合式合成 workflow，方法形状保真但有简化 |
| [GPU Forecasters](GPUForecasters/) | `tree_exploration` | learned forecaster + abstain 引导的 PUCT tree search | GPU speedup evaluator、forecast/abstain 校准 | forecaster model、search tree、GPU iterations ledger | workflow adaptation；严格性依赖真实 surrogate 训练/校准 |
| [KernelBlaster](KernelBlaster/) | `iterative_self_improving` | 按硬件性能状态索引的 MAIC-RL rollout | NCU Elapsed Cycles、正确性、reward | optimization database、trajectory、replay buffer | 带持久记忆的 in-context RL adaptation |
| [KernelFoundryDx](KernelFoundryDx/) | `search_based` | 诊断提示驱动的多岛 Triton 进化 | 编译/正确性/speedup + 反作弊检查 | island populations、elite archives、hint library | 依据论文的 faithful adaptation；无公开 runtime/source repo |
| [KernelSkill](KernelSkill/) | `iterative_self_improving` | seed/review 后的 repair-or-optimize 精炼循环 | compiler/verifier/profiler、speedup、NCU/nsys 证据 | 长期 skill library、optimize history、repair chain | 决策流程保真；gate 由 prompt/workflow 表达 |
| [StitchCUDA](StitchCUDA/) | `multi_stage_refinement` | Planner/Coder/Verifier + 自适应重规划 | 编译、正确性、benchmark speedup | plan history、failure counters、best candidate | 三智能体编排形状保真但有简化 |
| [WarpSpeed](WarpSpeed/) | `tree_exploration` | 带假设标签的检查点树上做预注册实验；验尸消融后前沿回退 | 显著性门控的 A/B 初筛 + 锁频确认、精选 NCU 段、sanitizer、codex 跨模型评审 | SQLite 树 + GPU 分钟台账、仅追加 BitLessons、NCU 缓存、证据分支 | AKW 原创工程 workflow（无论文）；GPU 纪律由互斥锁强制 |
| [Xe-Forge](Xe-Forge/) | `multi_stage_refinement` | 硬序 11 阶段 CoVeR 循环 | Intel Triton 编译、正确性、speedup、可选 VTune | best-in-stage kernels、promotion history | 面向 Intel XPU 项目流程的 workflow adaptation |
| [Meta-Workflow](_meta/) | `tooling` | Research/Model/Assemble/Generate/Validate | manifest schema、静态/语义检查 | templates、manifests、validation reports | 仓库基础设施，不是论文方法 |

> 工业界实践（如 FlashInfer 式分块 attention、CUTLASS 调参流程）会按「可 agent 化程度」陆续收录；欢迎提交 issue / PR 提名。

---

## 环境要求

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)（支持 Workflows 的版本）
- 按 workflow 后端准备对应工具链。CUDA workflow 需要 NVIDIA GPU + CUDA；Triton/SYCL/XPU/C++ workflow 需要对应 runtime 和 compiler stack。
- CUDA profiling workflow 推荐：**NVIDIA Nsight Compute**（`ncu`）— AccelOpt / KEET 等 workflow 以 profiling 证据驱动优化
- 各 workflow 可能额外要求：独立 benchmark harness、编译/构建命令、实验输出目录等（见各 workflow 文件头部注释）

建议在容器或独立虚拟环境中运行；agent 可能会执行编译、`pip install` 等操作。

---

## 快速开始

### 1. 克隆本仓库

```bash
git clone https://github.com/qhy991/Awesome-Kernel-Workflows.git
```

### 2. 安装为 Claude Code 全局 workflow（推荐）

一次安装，**所有项目**均可调用（`~/.claude/workflows/`）：

```bash
git clone https://github.com/qhy991/Awesome-Kernel-Workflows.git
cd Awesome-Kernel-Workflows
./scripts/install-global-workflows.sh
```

重启 Claude Code 后，用 `/workflows` 浏览，或按 `meta.name` 调用，例如 `/ako4x-kernel-optimizer`。

### 3. 或仅安装到单个 kernel 项目

在**待优化的 kernel 工程**根目录下复制到项目级 `.claude/workflows/`：

```bash
mkdir -p /path/to/your-kernel-project/.claude/workflows
cp Awesome-Kernel-Workflows/AccelOpt/accelopt-kernel-optimization.js \
   /path/to/your-kernel-project/.claude/workflows/
```

### 4. 在 Claude Code 中启动

进入 kernel 项目目录，启动 Claude Code，按 workflow 名称调用（参数以 workflow 文件内注释为准）。以 AccelOpt 为例：

只提供问题定义：

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

优化已有 kernel：

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
    kernel_name_regex: 'forward_kernel',
    ncu_binary: '<user-provided ncu binary path>',
    exp_dir: '/path/to/experiment/output',
    iterations: 3,
    breadth: 3,
    samples_per_plan: 2,
  },
})
```

---

## Workflow 方法论对比

| 方法 | 反馈信号 | 学习机制 | 目标精度 | 适用场景 |
|------|----------|----------|----------|----------|
| AccelOpt | NCU 指标 + 慢/快对 | 经验记忆累积 | 通用加速 | 任意 CUDA 内核 |
| KEET | NCU → 自然语言解释 | 无（单遍） | 理解性能 | 需要可解释分析 |
| ARGUS | 不变量违规反例 | ICRL 文本梯度 | 99%+ 库级 | GEMM/Attention/MoE |
| AKO4X | Benchmark + NCU | 5-段 header 经验 + TRAPS.md | 多轮渐进 | 多 DSL（Triton/CUDA/CuTe） |
| KDA | 验证命令 + 指标 | candidates.jsonl 记录 | 达标晋升 | 竞赛/基准测试场景 |
| K-Search | 世界模型评分 | 树搜索 + 回溯 | 最优路径 | 优化路径不确定时 |
| ReGraphT | evaluator 加速比 + 图奖励 | CUDA Reasoning Graph + MCGS | 小/本地模型图引导 | 有优化轨迹可复用时 |
| Astra | 正确性测试 + profiling + speedup | Testing/Profiling/Planning/Coding 多 Agent | 生产 CUDA kernel 迭代优化 | 已有 SGLang/CUDA kernel 时 |
| CUDA-LLM | 编译/正确性/latency reward | Feature Search + Reinforcement | 显式 CUDA feature 选择 | 从 task/spec 生成 CUDA kernel |
| AdaExplore | 编译/运行错误 + speedup | 技能记忆 + MCTS | Triton 内核生成 | PyTorch 算子替换 |
| KernelBand | NCU φ(k) + 延迟 | 动态聚类 + Masked UCB | 1.91× 几何均值 | 多策略大空间搜索 |

---

## 与其它项目的关系

本仓库**不替代**完整优化框架，而是提供**可插拔的 Claude Code workflow 片段**。你可能还会用到：

| 项目 | 关系 |
|------|------|
| [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) | 通用 agentic kernel 优化；可用 TASK.md + skills 驱动，与本仓库 workflow 可互补 |
| [AccelOpt](https://arxiv.org/abs/2511.15915) | 首个收录 workflow 的论文来源 |
| [Kernel Design Agents](https://github.com/mit-han-lab/kernel-design-agents) | KDA workflow 的来源项目（MIT HAN Lab） |
| [AdaExplore](https://stiglidu.github.io/AdaExplore/) | MCTS + 技能记忆方法论来源 |
| [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) | 标准化 kernel 优化评测集 |
| [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) | 工业界 operator 级 benchmark 与评分 |

---

## 如何贡献

欢迎 PR / Issue，尤其是：

- **新 workflow**：对应一篇论文、一个开源系统或一条可复现的工业界优化流程；请附原文链接与简短「为何适合 agent」说明；
- **改进现有 workflow**：更稳健的 NCU 解析、错误处理、参数默认值、注释；
- **对照实验**：在同一 kernel 上对比不同 workflow 的 speedup / 迭代次数（可放在 `examples/` 子目录，后续可加）。

贡献时请：

1. 新建 `<MethodName>/` 目录，放入 `*.js` workflow + `README.md`；
2. 在 workflow 顶部用注释写明**来源论文/仓库**与**必需参数**；
3. 更新 **`README.md` 与 `README.zh-CN.md`** 中的收录表格；
4. 运行 `scripts/count-workflows.sh`，刷新页眉 **workflows** 数量徽章（`badges/workflows.json`）。

---

## 许可证

各 workflow 文件请注明其衍生自的开源项目许可证；仓库默认许可证以根目录 `LICENSE` 为准（若尚未添加，贡献时可一并提交）。

---

如果本仓库对你整理 kernel 优化 agent 工作流有帮助，欢迎 Star，也欢迎提名下一个要收录的方法。
