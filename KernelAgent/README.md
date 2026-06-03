# KernelAgent — Multi-Agent Triton Kernel Synthesis

**English** · [简体中文](#简体中文)

A Claude Code Workflow implementing the [KernelAgent](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) methodology from PyTorch Labs: generate verified Triton kernels from PyTorch problem descriptions using parallel seed generation, sandboxed verification, and iterative refinement.

## Source

- **Blog post**: [PyTorch KernelFalcon — Autonomous GPU Kernel Generation via Deep Agents](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/)
- **Repository**: [github.com/pytorch-labs/KernelAgent](https://github.com/pytorch-labs/KernelAgent)
- **License**: Apache-2.0

## Architecture

```
Problem (PyTorch code or description)
        │
        ▼
┌───────────────────┐
│      Setup         │  Parse problem + generate test harness
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│      Route         │  Static complexity analysis
│  ┌─────────────┐   │
│  │ Direct Path  │───┼──→ Single-kernel synthesis
│  └─────────────┘   │
│  ┌─────────────┐   │
│  │Pipeline Path │───┼──→ Extract subgraphs → parallel dispatch → compose
│  └─────────────┘   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│     Generate       │  N parallel seeds (diverse temperatures/strategies)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│      Verify        │  Sandboxed subprocess execution (PASS/FAIL)
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    │ all pass?  │
    └─────┬─────┘
      no  │  yes ──────────────────────────┐
          ▼                                │
┌───────────────────┐                      │
│      Refine        │  Error-driven fix    │
│  (up to N rounds)  │  → re-verify         │
└─────────┬─────────┘                      │
          │                                │
          ├────────────────────────────────┘
          │
          ▼
┌───────────────────┐
│     Compose        │  (Pipeline path) Stitch subgraph kernels
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│      Report        │  Summary + best kernel + artifacts
└───────────────────┘
```

## Key Design Decisions

1. **Auto-routing**: Static analysis of the problem decides between direct synthesis (simple problems) and the Fuser pipeline (complex multi-op problems with attention, convolutions, control flow)

2. **Parallel diverse seeds**: Multiple kernel candidates generated with varied temperatures and strategies (straightforward → different tiling → vectorized loads → alternative algorithms)

3. **Strict verification**: Sandboxed subprocess execution with hard bans on `torch.nn` fallbacks — a kernel is only verified when the test prints "PASS" and exits with code 0

4. **Error-driven refinement**: Failed candidates get targeted LLM-guided fixes based on stderr/stdout, with history tracking to avoid repeating failed approaches

5. **Composable pipeline**: For complex problems, subgraphs are extracted and dispatched to parallel agents, then composed into a single Triton program

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Workflows-enabled build)
- Python 3.8+
- [Triton](https://github.com/triton-lang/triton) (`pip install triton`)
- [PyTorch](https://pytorch.org/) with CUDA support
- LLM provider API key (OpenAI, Anthropic, or custom endpoint)

## Usage

### Basic — Generate a kernel from a problem file

```javascript
Workflow({name: 'kernelagent-triton-synthesis', args: {
  problem_path: '/path/to/KernelBench/level1/19_ReLU.py',
  num_workers: 4,
  max_rounds: 10,
  verify: true,
}})
```

### Direct description — No file needed

```javascript
Workflow({name: 'kernelagent-triton-synthesis', args: {
  problem_description: 'Implement a fused multiply-add kernel: C = A * B + bias, where A is [M, K], B is [K, N], bias is [N]. Use Triton with 2D tiling.',
  num_workers: 4,
  model_name: 'gpt-5',
}})
```

### Complex problem — Force pipeline path

```javascript
Workflow({name: 'kernelagent-triton-synthesis', args: {
  problem_path: '/path/to/multi_head_attention.py',
  num_workers: 6,
  max_rounds: 15,
  compose: true,
  fuser_extract_model: 'gpt-5',
  fuser_dispatch_model: 'o4-mini',
}})
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `problem_path` | string | — | Path to Python file with PyTorch problem (**required** if no `problem_description`) |
| `problem_description` | string | `''` | Direct text description of the kernel to generate |
| `num_workers` | number | `4` | Number of parallel verification workers |
| `max_rounds` | number | `10` | Maximum refinement rounds per failed candidate |
| `max_seeds` | number | `4` | Number of diverse kernel seeds per target |
| `model_name` | string | `claude-sonnet-4-20250514` | LLM model for generation and refinement |
| `verify` | boolean | `true` | Run verification tests |
| `compose` | boolean | `true` | Compose subgraph kernels (pipeline path) |
| `exp_dir` | string | `/tmp/kernelagent_exp` | Experiment output directory |
| `temperature_base` | number | `0.8` | Base temperature for seed diversity |

## Phases

| Phase | Description |
|-------|-------------|
| **Setup** | Parse problem from file/description, generate test harness with PyTorch reference |
| **Route** | Static analysis: detect attention/conv/control-flow patterns → choose direct vs pipeline path |
| **Generate** | N parallel agents each produce a candidate Triton kernel with distinct strategy |
| **Verify** | Execute each candidate in sandboxed subprocess, check for PASS + exit 0 |
| **Refine** | Iterative error-driven repair: analyze stderr, apply minimal fixes, re-verify (up to max_rounds) |
| **Compose** | (Pipeline path) Stitch verified subgraph kernels into single Triton program |
| **Report** | Summary: outcome, best kernel, verification status, recommendations |

## Comparison with Original KernelAgent

| Aspect | Original KernelAgent | This Workflow |
|--------|---------------------|---------------|
| Language | Python (TritonKernelAgent class) | Claude Code Workflow (.js) |
| Verification | Subprocess with `DISALLOWED_TORCH_PATTERNS` | Same policy checks + sandboxed agent execution |
| Parallelism | `multiprocessing.Process` workers | `parallel()` agent fan-out |
| Routing | `Fuser/auto_agent.py` AST analysis | Agent-based static analysis |
| Composition | `Fuser/compose_end_to_end.py` | Agent-based composition |
| LLM calls | Direct API (OpenAI/Anthropic providers) | Claude Code agents |

## Return Value

```json
{
  "outcome": "success",
  "summary": "Generated verified Triton kernel for ReLU...",
  "routing_path": "direct",
  "total_candidates": 4,
  "verified_count": 2,
  "refinement_rounds": 1,
  "best_kernel_code": "@triton.jit\ndef relu_kernel...",
  "session_directory": "/tmp/kernelagent_exp/session_..."
}
```

---

# 简体中文

实现 [KernelAgent](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/)（PyTorch Labs）方法论的 Claude Code 工作流：通过并行种子生成、沙箱验证和迭代修复，从 PyTorch 问题描述生成经过验证的 Triton 内核。

## 核心设计

1. **自动路由**：静态分析问题复杂度，决定使用直接合成（简单问题）还是 Fuser 流水线（复杂多算子问题）
2. **并行多样化种子**：多个内核候选使用不同温度和策略生成（直接实现 → 不同分块 → 向量化加载 → 替代算法）
3. **严格验证**：沙箱子进程执行，硬性禁止 `torch.nn` 回退 — 仅当测试打印 "PASS" 且退出码为 0 时才算验证通过
4. **错误驱动修复**：失败的候选根据 stderr/stdout 获得针对性的 LLM 修复，并跟踪历史以避免重复失败的方法
5. **可组合流水线**：对于复杂问题，提取子图并分派给并行代理，然后组合成单个 Triton 程序

## 使用方法

```javascript
Workflow({name: 'kernelagent-triton-synthesis', args: {
  problem_path: '/path/to/problem.py',
  num_workers: 4,
  max_rounds: 10,
  verify: true,
}})
```

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `problem_path` | string | — | PyTorch 问题文件路径（与 `problem_description` 二选一必填） |
| `problem_description` | string | `''` | 直接文本描述要生成的内核 |
| `num_workers` | number | `4` | 并行验证 worker 数量 |
| `max_rounds` | number | `10` | 每个失败候选的最大修复轮数 |
| `model_name` | string | `claude-sonnet-4-20250514` | 用于生成和修复的 LLM 模型 |
| `verify` | boolean | `true` | 是否运行验证测试 |
| `compose` | boolean | `true` | 是否组合子图内核（流水线路径） |
