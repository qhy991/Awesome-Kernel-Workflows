<h1 align="center">Awesome Kernel Workflows</h1>

<p align="center">
  Curated <b>GPU kernel optimization</b> methods from academia and industry, packaged as reusable <a href="https://docs.anthropic.com/en/docs/claude-code">Claude Code</a> Workflows
</p>

<p align="center">
  <a href="https://github.com/qhy991/Awesome-Kernel-Workflows"><img src="https://img.shields.io/badge/GitHub-Awesome--Kernel--Workflows-blue?logo=github" alt="GitHub"></a>
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

## Catalog

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
| [Meta-Workflow](_meta/) | ![Tooling](https://img.shields.io/badge/tooling-gray?style=flat) | Research → Model → Assemble → Generate → Validate | — |

### Tag Legend

| Category | Tags |
|----------|------|
| **Backend** | ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white) ![ROCm](https://img.shields.io/badge/ROCm-ED1C24?style=flat&logo=amd&logoColor=white) ![SYCL](https://img.shields.io/badge/SYCL-0071C5?style=flat&logo=intel&logoColor=white) |
| **Kernel Language** | ![Triton](https://img.shields.io/badge/Triton-6C3483?style=flat) ![DSL](https://img.shields.io/badge/DSL-darkgreen?style=flat) ![CuTe](https://img.shields.io/badge/CuTe-darkgreen?style=flat) ![TileLang](https://img.shields.io/badge/TileLang-darkgreen?style=flat) |
| **Search Strategy** | ![Iterative](https://img.shields.io/badge/iterative-blue?style=flat) ![Tree](https://img.shields.io/badge/tree--search-darkblue?style=flat) ![MCTS](https://img.shields.io/badge/MCTS-darkblue?style=flat) ![Evolutionary](https://img.shields.io/badge/MAP--Elites-darkblue?style=flat) ![Pipeline](https://img.shields.io/badge/pipeline-purple?style=flat) |
| **Profiler** | ![NCU](https://img.shields.io/badge/NCU-555?style=flat) |
| **Learning Mechanism** | ![Experience](https://img.shields.io/badge/experience--memory-orange?style=flat) ![SkillMemory](https://img.shields.io/badge/skill--memory-orange?style=flat) ![WorldModel](https://img.shields.io/badge/world--model-orange?style=flat) ![MetaPrompt](https://img.shields.io/badge/meta--prompt-orange?style=flat) ![ICRL](https://img.shields.io/badge/ICRL-blue?style=flat) ![RL-trained](https://img.shields.io/badge/RL--trained-red?style=flat) |
| **Special** | ![Invariants](https://img.shields.io/badge/invariants-red?style=flat) ![Evidence](https://img.shields.io/badge/evidence--driven-green?style=flat) ![Explanation](https://img.shields.io/badge/explanation-teal?style=flat) |

> Industrial practices (e.g. tiled attention as in FlashInfer, CUTLASS tuning workflows) will be added when they are concrete enough to agentize. Open an issue or PR to nominate the next entry.

---

## Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Workflows-enabled build)
- NVIDIA GPU + CUDA toolchain
- **Recommended:** NVIDIA Nsight Compute (`ncu`) — workflows such as AccelOpt are evidence-driven; they avoid guessing bottlenecks
- Per-workflow extras may include: standalone benchmark harness, `nvcc` build line, experiment output directory (see each file’s header comments)

Run inside a container or dedicated virtual environment when possible; agents may compile code or run `pip install`.

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/qhy991/Awesome-Kernel-Workflows.git
```

### 2. Install into your kernel project

Copy the workflow into your project’s Claude Code workflows directory:

```bash
mkdir -p /path/to/your-kernel-project/.claude/workflows
cp Awesome-Kernel-Workflows/AccelOpt/accelopt-kernel-optimization.js \
   /path/to/your-kernel-project/.claude/workflows/
```

### 3. Invoke from Claude Code

From your kernel project root, start Claude Code and call the workflow (parameters are documented in each `.js` file). AccelOpt example:

```javascript
Workflow({
  name: 'accelopt-kernel-optimization',
  args: {
    kernel_path: '/path/to/kernel.cu',
    op_description: 'Quantized GEMM Q4_0 weight × FP32 activation',
    harness_path: '/path/to/harness.cu',
    harness_build_cmd: 'nvcc -O3 -lineinfo -arch=sm_90 ...',
    harness_run_args: '',
    kernel_name_regex: 'forward_kernel',
    ncu_binary: 'ncu',
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

---

## License

Per-workflow files should state license terms if derived from upstream projects. The repository default license applies when `LICENSE` is present at the repo root.

---

If this collection helps you operationalize kernel-optimization agents, consider starring the repo and nominating the next method to port.
