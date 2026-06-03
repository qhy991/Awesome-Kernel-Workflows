# STARK — Strategic Team of Agents for Refining Kernels

**English** · [简体中文](#简体中文)

A Claude Code Workflow implementing the **STARK** methodology (arXiv:2510.16996, Meta / Duke University 2025): a multi-agent framework for GPU kernel optimization that addresses three key limitations of monolithic LLM agents — naive exploration, monolithic design, and the planning-implementation gap.

## Source

- **Paper**: [STARK: Strategic Team of Agents for Refining Kernels](https://arxiv.org/abs/2510.16996) (arXiv:2510.16996, Meta Ranking AI Research / Duke University, 2025)
- **Benchmark**: [KernelBench](https://github.com/ScalingIntelligence/KernelBench) — 33-level GPU kernel optimization tasks (L1 single-op, L2 fused-op, L3 full architectures)

## Architecture

```
PyTorch Reference Kernel
        │
        ▼
┌───────────────────┐
│      Setup         │  Evaluate baseline, init tree + leaderboard
└─────────┬─────────┘
          │
   ┌──────┴──────────────────────────────────────┐
   │  Main Loop (attempt = 1 … B)                │
   │                                              │
   │  ┌───────────────────┐                     │
   │  │      Select        │  ε-greedy from tree │
   │  │  (root throttle)   │  (leaf-biased exp.) │
   │  └─────────┬─────────┘                     │
   │            │                               │
   │     HasBug(selected)?                      │
   │      /          \                          │
   │   yes            no                        │
   │    │              │                        │
   │    ▼              ▼                        │
   │ ┌─────────┐  ┌─────────────┐              │
   │ │  Debug   │  │     Plan     │  (τ=0.8)   │
   │ │ (τ=0.1) │  │ + Anchors    │            │
   │ └────┬────┘  └──────┬──────┘              │
   │      │              │                     │
   │      └──────┬───────┘                     │
   │             ▼                              │
   │  ┌───────────────────┐                   │
   │  │      Code          │  (τ=0.1)          │
   │  │  Resolve anchors   │                  │
   │  └─────────┬─────────┘                   │
   │            │                               │
   │            ▼                               │
   │  ┌───────────────────┐                   │
   │  │     Evaluate       │  compile + test   │
   │  │  (runtime_ms)      │  + profile        │
   │  └─────────┬─────────┘                   │
   │            │                               │
   │            ▼                               │
   │  ┌───────────────────┐                   │
   │  │      Update        │  tree ← child    │
   │  │  leaderboard ← best │                  │
   │  └───────────────────┘                   │
   └──────────────────────────────────────────┘
          │
          ▼
┌───────────────────┐
│      Report        │  Return best kernel + trajectory
└───────────────────┘
```

## Key Design Decisions

### 1. Multi-Agent Collaboration (MAD)

Three specialized agents replace the monolithic design:

| Agent | Temperature | Role | Context Window |
|-------|------------|------|----------------|
| **Plan** | τ = 0.8 | Propose optimization strategy with grounded instruction anchors | Selected node + its children + global leaderboard top-r |
| **Code** | τ = 0.1 | Translate anchored scaffold into executable CUDA | Selected node + its children + sibling patches |
| **Debug** | τ = 0.1 | Repair failing kernels with minimal changes | Failing node + sibling kernels (local fix patterns) |

### 2. Grounded Instruction

Plan Agent must insert explicit span anchors:

```cuda
<<<IMPROVE BEGINS: shared_memory_tiling>>>
// original code to be modified
<<<IMPROVE ENDS: shared_memory_tiling>>>
```

Code Agent resolves each anchor into concrete CUDA, ensuring plan-code alignment and reducing hallucination.

### 3. Strategic Search (ε-greedy over Tree Memory)

Domain-specific adaptations:

- **Root throttling**: Cap root children at `n_root` (default 5) to avoid redundant first-hop edits
- **Dead-branch pruning**: If a node has > `n_child_max` children and all fail, mark ineligible
- **High exploration rate**: ε = 0.3–0.4 (counteracts local optima in kernel space)
- **Leaf-biased exploration**: With probability ε, sample uniformly from expandable leaves

### 4. Dynamic Context Windows

Each agent receives a tailored view of history:

- **Plan Agent**: `{selected, root} ∪ children(selected) ∪ Top-r(leaderboard)` — reflection + ambition calibration + capability estimation
- **Code Agent**: `{selected, root} ∪ children(selected) ∪ {cousin nodes}` — patch transfer from successful siblings
- **Debug Agent**: `{failing, root} ∪ siblings(failing)` — local fix patterns

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Workflows-enabled build)
- NVIDIA GPU + CUDA toolchain (`nvcc`)
- Python test harness that compiles and benchmarks the kernel
- Optional: PyTorch for reference implementation

## Usage

### Basic — Optimize a reference kernel with default budget (30 attempts)

```javascript
Workflow({name: 'stark-kernel-optimization', args: {
  reference_kernel_path: '/path/to/reference.cu',
  test_harness_path: '/path/to/test.py',
}})
```

### Tuned — Higher budget, custom temperatures

```javascript
Workflow({name: 'stark-kernel-optimization', args: {
  reference_kernel_path: '/path/to/reference.cu',
  test_harness_path: '/path/to/test.py',
  budget: 50,
  epsilon: 0.4,
  plan_temperature: 0.9,
  code_temperature: 0.05,
  n_root: 8,
  leaderboard_size: 10,
  compile_command: 'nvcc -O3 -arch=sm_90 -lineinfo -o kernel reference.cu',
  benchmark_command: 'python test.py --profile',
}})
```

### Minimal — Just a kernel file, let STARK build the harness

```javascript
Workflow({name: 'stark-kernel-optimization', args: {
  reference_kernel_path: '/path/to/kernel.cu',
  test_harness_path: '',
  budget: 20,
}})
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `reference_kernel_path` | string | — | Path to reference CUDA kernel (**required**) |
| `test_harness_path` | string | — | Path to Python test harness (**required**) |
| `budget` | number | `30` | Max optimization attempts (tree nodes) |
| `epsilon` | number | `0.35` | ε-greedy exploration rate (0 = pure exploitation, 1 = pure random) |
| `n_root` | number | `5` | Root throttling cap (max direct children of root) |
| `n_child_max` | number | `3` | Dead-branch threshold (if all children fail) |
| `leaderboard_size` | number | `5` | Number of top kernels in global leaderboard |
| `plan_model` | string | `claude-sonnet-4-20250514` | LLM for plan agent |
| `code_model` | string | `claude-sonnet-4-20250514` | LLM for code agent |
| `debug_model` | string | `claude-sonnet-4-20250514` | LLM for debug agent |
| `plan_temperature` | number | `0.8` | Temperature for plan agent (high = creative) |
| `code_temperature` | number | `0.1` | Temperature for code agent (low = precise) |
| `debug_temperature` | number | `0.1` | Temperature for debug agent (low = precise) |
| `compile_command` | string | `''` | Custom nvcc compile command |
| `benchmark_command` | string | `''` | Custom benchmark command |
| `exp_dir` | string | `/tmp/stark_exp` | Experiment output directory |

## Phases

| Phase | Description |
|-------|-------------|
| **Setup** | Read reference kernel, compile + benchmark baseline, initialize search tree |
| **Select** | ε-greedy node selection with root throttling and dead-branch pruning |
| **Plan** | High-temperature plan agent proposes optimization + grounded instruction anchors |
| **Code** | Low-temperature code agent resolves anchors into anchor-free executable CUDA |
| **Debug** | Debug agent repairs compilation/correctness failures using sibling patterns |
| **Evaluate** | Compile, run correctness test, measure wall-clock runtime |
| **Update** | Append child to tree, update leaderboard with best performers |
| **Report** | Return best kernel + full optimization trajectory |

## Return Value

```json
{
  "outcome": "success",
  "summary": "Achieved 3.2x speedup over baseline with shared-memory tiling and vectorized loads",
  "best_runtime_ms": 12.5,
  "baseline_runtime_ms": 40.1,
  "speedup": 3.2,
  "total_attempts": 30,
  "correct_kernels": 8,
  "failed_kernels": 22,
  "best_kernel_id": "node_17",
  "best_kernel_code": "__global__ void optimized_kernel(...) { ... }",
  "best_kernel_plan": "Apply shared-memory tiling with 2D block layout",
  "tree_summary": [...],
  "leaderboard": [...],
  "optimization_trajectory": [...]
}
```

## Comparison with Other Workflows

| Workflow | Search Strategy | Multi-Agent? | Key Innovation |
|----------|----------------|--------------|----------------|
| **STARK** | ε-greedy tree search | Yes (Plan/Code/Debug) | Grounded instruction + dynamic context windows |
| AdaExplore | MCTS | No | Failure-driven skill memory |
| KSearch | World-model tree search | No | Tree-of-thought with backtracking |
| KernelBand | MAB-UCB | No | Hardware-aware pruning + clustering |
| AccelOpt | Iterative self-improving | No | NCU profiling + experience accumulation |

## Performance Claims (from Paper)

On KernelBench (33 tasks, 3 difficulty levels, 30 attempts each):

- **L1** (single-op): 100% success, up to **3.0×** speedup over torch baseline
- **L2** (fused-op): 100% success, up to **2.7×** speedup (Reflexion often slower than baseline)
- **L3** (full architectures): 100% success, up to **1.6×** speedup (baselines often fail entirely)

Sampling Agent achieves only 57% L1 success; Reflexion Agent succeeds but often produces slower-than-baseline kernels.

---

# 简体中文

实现 **STARK** 方法论的 Claude Code 工作流（arXiv:2510.16996，Meta / Duke University 2025）：通过多 Agent 协作、 grounded instruction 锚点和动态上下文窗口，解决单 Agent 的幼稚探索、单体设计和计划-实现鸿沟三大缺陷。

## 核心设计

1. **多 Agent 协作 (MAD)**：Plan（高温度 τ=0.8，探索策略）+ Code（低温度 τ=0.1，精确实现）+ Debug（低温度 τ=0.1，修复错误）
2. **Grounded Instruction**：Plan Agent 在代码中插入 `<<<IMPROVE BEGINS/ENDS>>>` 锚点，Code Agent 据此精确实现，消除幻觉
3. **策略搜索**：基于搜索树的 ε-greedy，包含根节点限流、死枝剪枝、叶节点偏置探索
4. **动态上下文窗口**：三个 Agent 各自获得不同历史视图（全局排行榜 / 堂兄弟节点 / 兄弟节点）

## 使用方法

```javascript
Workflow({name: 'stark-kernel-optimization', args: {
  reference_kernel_path: '/path/to/reference.cu',
  test_harness_path: '/path/to/test.py',
  budget: 30,
  epsilon: 0.35,
}})
```

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `reference_kernel_path` | string | — | 参考 CUDA 内核文件路径（必填） |
| `test_harness_path` | string | — | Python 测试 harness 路径（必填） |
| `budget` | number | `30` | 最大优化尝试次数 |
| `epsilon` | number | `0.35` | ε-greedy 探索率 |
| `n_root` | number | `5` | 根节点子节点上限（根节点限流） |
| `n_child_max` | number | `3` | 死枝剪枝阈值 |
| `leaderboard_size` | number | `5` | 全局排行榜保留数量 |
| `plan_temperature` | number | `0.8` | Plan Agent 温度（高=创造性） |
| `code_temperature` | number | `0.1` | Code Agent 温度（低=精确） |
| `debug_temperature` | number | `0.1` | Debug Agent 温度（低=精确） |

## 阶段

| 阶段 | 说明 |
|------|------|
| **Setup** | 读取参考内核，编译+基准测试，初始化搜索树 |
| **Select** | ε-greedy 节点选择（含根节点限流和死枝剪枝） |
| **Plan** | Plan Agent 提出优化策略 + grounded instruction 锚点 |
| **Code** | Code Agent 将锚点解析为可执行 CUDA 代码 |
| **Debug** | Debug Agent 用兄弟节点模式修复失败内核 |
| **Evaluate** | 编译、正确性测试、运行时测量 |
| **Update** | 将子节点加入树，更新排行榜 |
| **Report** | 返回最佳内核 + 完整优化轨迹 |

## 性能（论文结果）

在 KernelBench 上（33 个任务，3 个难度等级，每个 30 次尝试）：

- **L1**（单算子）：100% 成功率，最高 **3.0×** 加速
- **L2**（融合算子）：100% 成功率，最高 **2.7×** 加速
- **L3**（完整架构）：100% 成功率，最高 **1.6×** 加速

Sampling Agent L1 成功率仅 57%；Reflexion Agent 虽能通过测试但常生成比 baseline 更慢的内核。
