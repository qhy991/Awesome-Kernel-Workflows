# Kernel Foundry（诊断驱动 / 多专家）工作流

岛屿模型进化式 Triton 内核优化器，结合专家引导的 RAG 初始化与自我精炼的诊断驱动提示库，改编自：

- 论文：[Kernel Foundry: A Diagnosis-driven Evolutionary Kernel Optimizer with Multi-Experts](https://arxiv.org/abs/2605.30359)（arXiv:2605.30359）
- 工作流文件：[`kernelfoundrydx-kernel-optimization.js`](kernelfoundrydx-kernel-optimization.js)

[English](README.md) · **简体中文**

---

> **命名说明。** 本工作流对应的论文与仓库中已有的 Intel [`KernelFoundry/`](../KernelFoundry/)
> （arXiv:2603.12440，SYCL，MAP-Elites）**是两篇不同的工作**。二者同名但作者、arXiv 编号、
> 方法和目标后端都不同。本篇（香港中文大学 + 华为诺亚方舟实验室，**Triton**）放在
> `KernelFoundryDx/` 目录下（*Dx* 表示 **D**iagnosis-driven，诊断驱动），以避免与已有目录冲突。

---

## 功能简介

`kernelfoundrydx-kernel-optimization` 将内核合成视为一个**迭代进化优化**问题，而非一次性生成。
它并行运行若干**角色专精的岛屿**，每个岛屿进化各自的 Triton 内核种群，并维护一个集中式的
**经验 / 提示库**，学习哪些自然语言优化提示真正与实测加速相关。

循环：

1. **Setup** — 读取 PyTorch 参考实现，基准测试 eager 基线，初始化提示库。
2. **Init** — 专家引导的 RAG 初始化：检索相似的已验证 PyTorch→Triton 对，由领域专家模型生成正确种子，反作弊校验。
3. 每次迭代，对每个岛屿：
   - **Evolve** — LLM 基于岛屿角色、检索到的提示和历史对父代进行变异。
   - **Evaluate** — 在真实硬件上编译 + 运行，测量正确性与加速比（仅轻量信号，不用 ncu/nsys），反作弊检查。
   - **Diagnose** — Result Analyzer 对每个候选分类：失败模式（不正确）或粗粒度瓶颈
     `memory-bound / latency-bound / instruction-bound`（正确），然后生成提示。
   - **Evolve-Pop** — 更新种群与精英归档，按实测加速强化 / 弱化提示，停滞时在岛屿间概率性迁移精英。
4. **Report** — 所有岛屿中最优的有效内核、提示库演化、岛屿轨迹。

---

## 核心思想（忠于原文）

- **诊断驱动的提示**：三阶段 Result Analyzer（信号提取 → 诊断引擎 → 提示生成）将每次评估转化为可复用、带标签的自然语言提示。提示在与加速相关时被强化，否则被弱化。
- **多专家**：两层含义——(1) 一个小型领域专精的**专家模型**产出经 RAG 引导的正确初始内核；(2) **角色专精的岛屿**各自拥有不同的系统提示和提示子集（算子融合 / 访存 / 参数调优 / 指令优化）。
- **岛屿模型进化**：相互独立、持久化的种群，配合本地精英归档；岛屿间通过概率性精英迁移协调，而非投票或合并。
- **反作弊**：两级——提示约束（合法 `@triton.jit`，全部计算在 Triton 内）加上一个 LLM 校验器对作弊可能性打分，超过 `cheating_threshold` 的候选被丢弃。
- **不依赖剖析器**：瓶颈类别由运行时统计 + 启动配置推断，方法刻意避免重度剖析。

---

## 主要参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `ref_path`（必填） | — | 要移植到 Triton 的 PyTorch 参考模块（.py） |
| `op_description` | `'PyTorch operator'` | 算子的可读描述 |
| `target_gpu` | `'RTX5090'` | 目标 GPU（论文：RTX 5090，访存密集用 A800-80G） |
| `num_islands` | `4` | 角色专精岛屿数量 |
| `iterations` | `5` | 进化迭代次数（论文用 30） |
| `population_size` | `3` | 每个岛屿的种群规模 |
| `migration_stagnation` | `2` | 触发精英迁移前的非改进迭代数 |
| `benchmark_command` | `''` | 编译 + 基准测试命令 |
| `rtol` / `atol` | `0.01` | 正确性容差 |
| `retrieval_corpus_path` | `''` | 用于 RAG 初始化的已验证 PyTorch→Triton 对 |
| `hint_library_path` | `''` | 持久化的提示库 JSON（跨任务复用） |
| `exp_dir` | `/tmp/kernelfoundrydx_exp` | 输出目录 |
| `cheating_threshold` | `0.5` | 作弊可能性超过此值的候选被丢弃 |

---

## 调用示例

```javascript
Workflow({
  name: 'kernelfoundrydx-kernel-optimization',
  args: {
    ref_path: '/path/to/KernelBench/level2/95_Matmul_Add_Swish.py',
    op_description: 'Matmul + Add + Swish + Tanh + GELU + Hardtanh 融合',
    target_gpu: 'RTX5090',
    num_islands: 4,
    iterations: 30,
    population_size: 4,
    benchmark_command: '<user-provided benchmark command with {kernel_path}/{result_path}>',
    retrieval_corpus_path: '/path/to/verified_triton_pairs.jsonl',
    hint_library_path: '/path/to/kernelfoundrydx_hints.json',
    exp_dir: '/tmp/kernelfoundrydx_exp',
  },
})
```

请在带 Triton 工具链的容器 / 虚拟环境中运行——agent 会在 GPU 上编译并运行内核。

---

## 输出

- `best_speedup`、`best_kernel_code`
- `iterations_completed`、`kernels_evaluated`
- `islands_summary`（每个岛屿的角色 + 最佳加速）
- `hint_library`（带置信统计的最终提示——可通过 `hint_library_path` 持久化）
- `report`

---

## 关于保真度

截至该预印本，本论文没有公开代码仓库，因此本工作流是依据论文正文与附录（Evolver 系统提示、
按 正确性 / 访存受限 / 指令受限 / 延迟受限 分组的专家建议提示分类，以及进化示例）对方法的忠实改编。
检索语料库与提示库以路径形式暴露，便于你提供自己的 KernelBook 式数据并跨运行累积提示。

---

## 引用

```bibtex
@article{huang2026kernelfoundry,
  title={Kernel Foundry: A Diagnosis-driven Evolutionary Kernel Optimizer with Multi-Experts},
  author={Huang, Zixuan and Chen, Da and Huang, Kecheng and Yin, Lihao and Li, Xing and Zhen, Huiling and Yuan, Mingxuan and Shao, Zili},
  journal={arXiv preprint arXiv:2605.30359},
  year={2026},
  url={https://arxiv.org/abs/2605.30359}
}
```
