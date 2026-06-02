<h1 align="center">Awesome Kernel Workflows</h1>

<p align="center">
  将学术界与工业界的 <b>GPU Kernel 优化</b> 方法，沉淀为可复用的 <a href="https://docs.anthropic.com/en/docs/claude-code">Claude Code</a> Workflows
</p>

<p align="center">
  <a href="https://github.com/qhy991/Awesome-Kernel-Workflows"><img src="https://img.shields.io/badge/GitHub-Awesome--Kernel--Workflows-blue?logo=github" alt="GitHub"></a>
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

## 已收录 Workflows

| 方法 | 标签 | 核心循环 | 论文 / 项目 |
|------|------|----------|-------------|
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
| [Meta-Workflow](_meta/) | ![Tooling](https://img.shields.io/badge/tooling-gray?style=flat) | Research → Model → Assemble → Generate → Validate | — |

### 标签说明

| 分类 | 标签 |
|------|------|
| **后端** | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![ROCm](https://img.shields.io/badge/ROCm-ED1C24?style=flat&logo=amd&logoColor=white) ![SYCL](https://img.shields.io/badge/SYCL-0071C5?style=flat&logo=intel&logoColor=white) |
| **内核语言** | ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![DSL](https://img.shields.io/badge/DSL-darkgreen?style=flat) ![CuTe](https://img.shields.io/badge/CuTe-darkgreen?style=flat) ![TileLang](https://img.shields.io/badge/TileLang-darkgreen?style=flat) |
| **搜索策略** | ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![MCTS](https://img.shields.io/badge/MCTS-darkblue?style=flat) ![Evolutionary](https://img.shields.io/badge/MAP--Elites-darkblue?style=flat) ![MAB](https://img.shields.io/badge/MAB--UCB-darkblue?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) |
| **分析工具** | ![NCU](https://img.shields.io/badge/NCU-555?style=flat) |
| **学习机制** | ![Experience](https://img.shields.io/badge/experience--memory-orange?style=flat) ![SkillMemory](https://img.shields.io/badge/skill--memory-orange?style=flat) ![WorldModel](https://img.shields.io/badge/world--model-orange?style=flat) ![MetaPrompt](https://img.shields.io/badge/meta--prompt-orange?style=flat) ![Clustering](https://img.shields.io/badge/clustering-orange?style=flat) ![ICRL](https://img.shields.io/badge/ICRL-blue?style=flat) ![RL-trained](https://img.shields.io/badge/RL--trained-red?style=flat) |
| **特殊** | ![Invariants](https://img.shields.io/badge/invariants-red?style=flat) ![Evidence](https://img.shields.io/badge/evidence--driven-green?style=flat) ![Explanation](https://img.shields.io/badge/explanation-teal?style=flat) ![HW-Pruning](https://img.shields.io/badge/HW--pruning-red?style=flat) |

> 工业界实践（如 FlashInfer 式分块 attention、CUTLASS 调参流程）会按「可 agent 化程度」陆续收录；欢迎提交 issue / PR 提名。

---

## 环境要求

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)（支持 Workflows 的版本）
- NVIDIA GPU + CUDA 工具链（或 AMD GPU + ROCm，视 workflow 而定）
- 推荐：**NVIDIA Nsight Compute**（`ncu`）— AccelOpt / KEET 等 workflow 以 profiling 证据驱动优化
- 各 workflow 可能额外要求：独立 benchmark harness、`nvcc` 编译命令、实验输出目录等（见各 workflow 文件头部注释）

建议在容器或独立虚拟环境中运行；agent 可能会执行编译、`pip install` 等操作。

---

## 快速开始

### 1. 克隆本仓库

```bash
git clone https://github.com/qhy991/Awesome-Kernel-Workflows.git
```

### 2. 将 workflow 接入你的 kernel 项目

在**待优化的 CUDA 工程**根目录下，把对应 workflow 复制到 Claude Code 工作流目录：

```bash
mkdir -p /path/to/your-kernel-project/.claude/workflows
cp Awesome-Kernel-Workflows/AccelOpt/accelopt-kernel-optimization.js \
   /path/to/your-kernel-project/.claude/workflows/
```

### 3. 在 Claude Code 中启动

进入 kernel 项目目录，启动 Claude Code，按 workflow 名称调用（参数以 workflow 文件内注释为准）。以 AccelOpt 为例：

```javascript
Workflow({
  name: 'accelopt-kernel-optimization',
  args: {
    kernel_path: '/path/to/kernel.cu',
    op_description: 'Quantized GEMM Q4_0 weight × FP32 activation',
    harness_path: '/path/to/harness.cu',
    harness_build_cmd: 'nvcc -O3 -lineinfo -arch=sm_90 ...',
    kernel_name_regex: 'forward_kernel',
    ncu_binary: 'ncu',
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
3. 更新 **`README.md` 与 `README.zh-CN.md`** 中的收录表格。

---

## 许可证

各 workflow 文件请注明其衍生自的开源项目许可证；仓库默认许可证以根目录 `LICENSE` 为准（若尚未添加，贡献时可一并提交）。

---

如果本仓库对你整理 kernel 优化 agent 工作流有帮助，欢迎 Star，也欢迎提名下一个要收录的方法。
