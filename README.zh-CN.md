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
├── README.md              # English
├── README.zh-CN.md        # 简体中文（本文件）
├── AccelOpt/
│   └── accelopt-kernel-optimization.js   # AccelOpt 自改进 + NCU 闭环
└── <MoreMethods>/                        # 欢迎 PR 补充
    └── *.js
```

Workflow 文件遵循 Claude Code 约定：导出 `meta`（名称、描述、阶段），并通过 `phase()` / `agent()` 编排多步 agent 任务。

---

## 已收录 Workflows

| 方法 | 类型 | Workflow | 核心循环 | 论文 / 项目 |
|------|------|----------|----------|-------------|
| [AccelOpt](AccelOpt/) | 学术 | [`accelopt-kernel-optimization`](AccelOpt/accelopt-kernel-optimization.js) | Setup → Plan → Execute → Evaluate → Learn → Iterate（**NCU 驱动**） | [arXiv:2511.15915](https://arxiv.org/abs/2511.15915)（MLSys 2026） |

> 工业界实践（如 FlashInfer 式分块 attention、CUTLASS 调参流程）会按「可 agent 化程度」陆续收录；欢迎提交 issue / PR 提名。

---

## 环境要求

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)（支持 Workflows 的版本）
- NVIDIA GPU + CUDA 工具链
- 推荐：**NVIDIA Nsight Compute**（`ncu`）— AccelOpt 等 workflow 以 profiling 证据驱动优化，而非凭猜测改代码
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

**AccelOpt workflow 要点**（详见 [`AccelOpt/accelopt-kernel-optimization.js`](AccelOpt/accelopt-kernel-optimization.js)）：

- 实现论文中的 **Plan → Execute → Profile → Summarize → Accumulate Experience → Repeat**
- 在 Setup / Evaluate / Learn 阶段嵌入 **NCU**（`--set full`、源码级分析等），遵循「先 profile、再诊断、再改代码」
- 维护 `experienceMemory`，把慢/快 kernel 对与 NCU 指标差异提炼为可复用 pattern

若暂无 NCU，workflow 支持通过 `test_command` / `benchmark_command` 回退到自定义测试与 benchmark 命令。

---

## 与其它项目的关系

本仓库**不替代**完整优化框架，而是提供**可插拔的 Claude Code workflow 片段**。你可能还会用到：

| 项目 | 关系 |
|------|------|
| [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) | 通用 agentic kernel 优化；可用 TASK.md + skills 驱动，与本仓库 workflow 可互补 |
| [AccelOpt](https://arxiv.org/abs/2511.15915) | 首个收录 workflow 的论文来源 |
| [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) | 标准化 kernel 优化评测集 |
| [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) | 工业界 operator 级 benchmark 与评分 |

---

## 如何贡献

欢迎 PR / Issue，尤其是：

- **新 workflow**：对应一篇论文、一个开源系统或一条可复现的工业界优化流程；请附原文链接与简短「为何适合 agent」说明；
- **改进现有 workflow**：更稳健的 NCU 解析、错误处理、参数默认值、注释；
- **对照实验**：在同一 kernel 上对比不同 workflow 的 speedup / 迭代次数（可放在 `examples/` 子目录，后续可加）。

贡献时请：

1. 新建 `<MethodName>/` 目录，放入 `*.js` workflow；
2. 在 workflow 顶部用注释写明**来源论文/仓库**与**必需参数**；
3. 更新 **`README.md` 与 `README.zh-CN.md`** 中的收录表格。

---

## 许可证

各 workflow 文件请注明其衍生自的开源项目许可证；仓库默认许可证以根目录 `LICENSE` 为准（若尚未添加，贡献时可一并提交）。

---

如果本仓库对你整理 kernel 优化 agent 工作流有帮助，欢迎 Star，也欢迎提名下一个要收录的方法。
