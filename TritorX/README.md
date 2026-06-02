# TritorX: Agentic Operator Generation for ML ASICs

**English** · [简体中文](README.zh-CN.md)

FSM-based agentic generation of Triton ATen kernels at scale for emerging accelerator platforms. Implements the [TritorX](https://arxiv.org/abs/2512.10977) methodology (Hammond, Markosyan et al., Meta FAIR / Meta Superintelligence Labs, 2025).

## Overview

TritorX solves a different problem from the other workflows in this repository: instead of optimizing one kernel for peak performance, it generates **functionally correct kernels for an entire PyTorch ATen backend** on a new accelerator platform — overnight.

### Key Distinction: Coverage over Performance

| Aspect | Most workflows in this repo | TritorX |
|--------|----------------------------|---------|
| Goal | Maximize speedup on one kernel | Maximize operator coverage (481 ops, 20K+ tests) |
| Target | NVIDIA/AMD GPUs | Emerging ASICs (Meta MTIA, future chips) |
| Feedback | NCU profiling, speedup measurement | Linter + compiler + OpInfo correctness tests |
| Scale | 1 kernel, many iterations | Hundreds of operators in parallel |

### Results (from paper)

- **481** ATen operators generated with 100% OpInfo test pass rate (84.7% coverage)
- **20,000+** total tests passed across dtypes, shapes, and argument patterns
- 95% of operators complete in 2 hours on 200 MTIA devices
- End-to-end model enablement: NanoGPT, DLRM, Meta internal models

## Architecture: Finite State Machine

```
                    ┌─────────┐
                    │  Init   │
                    └────┬────┘
                         │
              ┌──────────▼──────────┐
              │  Generate Kernel    │◄──── feedback (compiler/runtime/accuracy)
              │  (LLM → kernel +   │
              │   wrapper pair)     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
         ┌───►│  Triton Linter      │
         │    │  (anti-cheat +      │
         │    │   dialect check)    │
         │    └──────────┬──────────┘
         │               │ pass
         │    ┌──────────▼──────────┐
         │    │  Compile/Execute/   │
         │    │  Test (OpInfo)      │
         │    └──────────┬──────────┘
         │               │
         │    ┌──────────▼──────────┐
    lint  │    │  Process Results    │
  violation   └──────────┬──────────┘
         │               │
         │    ┌──────────▼──────────┐
         │    │  Debug              │
         └────┤  (analyze failure,  │
              │   build feedback)   │
              └──────────┬──────────┘
                    ┌────┴────┐
              ┌─────▼──┐  ┌──▼─────┐
              │Failure │  │Success │
              │(max    │  │(all    │
              │retries)│  │tests   │
              └────────┘  │pass)   │
                          └────────┘
```

### Key Mechanisms

1. **Custom Triton Linter**: Prevents "cheating" — catches wrapper code that dispatches to host PyTorch ops instead of using the generated kernel
2. **In-context distillation**: Learns hardware-specific Triton semantics from compiler/runtime error feedback, not from upfront documentation
3. **OpInfo test harness**: Tests across ALL supported dtypes, shapes, and argument patterns (100x more tests than KernelBench)
4. **Feedback summarization**: Secondary LLM condenses long compiler logs to fit in context
5. **LLDB debugger integration**: For runtime crashes, extracts backtraces and register state

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| **Claude Code** | Workflow runtime |
| **Target accelerator** | MTIA, GPU, or hardware simulator (QEMU) |
| **Python + PyTorch + Triton** | Kernel JIT compilation and testing |
| **OpInfo test framework** | PyTorch's operator test infrastructure |

## Usage

### Single operator

```javascript
Workflow({name: 'tritorx-operator-generation', args: {
  operator_name: 'aten::softmax',
  operator_docstring: 'Applies softmax(x, dim) = exp(x_i) / sum(exp(x_j)) over dim',
  target_platform: 'MTIA',
  triton_dialect: 'triton-mtia',
  test_command: 'python run_opinfo_tests.py --op softmax',
  lint_command: 'python triton_linter.py --check kernel.py wrapper.py',
  max_attempts: 3,
  max_llm_calls_per_attempt: 15,
}})
```

### Batch operators

```javascript
Workflow({name: 'tritorx-operator-generation', args: {
  operator_list: [
    {name: 'aten::relu', docstring: '...'},
    {name: 'aten::gelu', docstring: '...'},
    {name: 'aten::softmax', docstring: '...'},
  ],
  target_platform: 'GPU',
  max_attempts: 3,
}})
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `operator_name` | `''` | ATen operator name (single mode) |
| `operator_docstring` | `''` | Operator documentation/signature |
| `operator_list` | `[]` | Batch mode: array of {name, docstring} |
| `target_platform` | `'GPU'` | Target: MTIA, GPU, or simulator |
| `triton_dialect` | `'triton'` | Triton dialect (triton-mtia for MTIA) |
| `test_command` | `''` | OpInfo test command |
| `lint_command` | `''` | Custom linter command |
| `max_attempts` | `3` | Max generation attempts per operator |
| `max_llm_calls_per_attempt` | `15` | Max LLM iterations per attempt |
| `supported_dtypes` | `['bf16','f16','f32','i32','i64']` | Dtypes to test |
| `few_shot_examples` | `[]` | Example kernel/wrapper pairs |

## References

- [Agentic Operator Generation for ML ASICs](https://arxiv.org/abs/2512.10977) — Hammond, Markosyan et al. (Meta, 2025)

---

# TritorX：面向 ML ASIC 的 Agent 算子生成

**English** · 简体中文

基于有限状态机的 Triton ATen 内核批量生成，面向新兴加速器平台。实现了 [TritorX](https://arxiv.org/abs/2512.10977) 方法论（Hammond, Markosyan 等人，Meta FAIR / Meta 超智研究院，2025）。

## 概述

TritorX 解决的问题与仓库中其他 workflow 不同：它不是优化单个内核的性能，而是为新加速器平台**批量生成整个 PyTorch ATen 后端的正确内核**。

### 论文结果

- **481** 个 ATen 算子生成并通过全部 OpInfo 测试（84.7% 覆盖率）
- **20,000+** 个测试通过（跨数据类型、形状、参数模式）
- 200 台 MTIA 设备上 95% 算子 2 小时内完成

### 核心机制

1. **自定义 Linter**：防止"作弊"（wrapper 调用宿主 PyTorch 算子）
2. **上下文蒸馏**：从编译器/运行时错误反馈中学习硬件特定语义
3. **OpInfo 测试框架**：全数据类型、形状、参数模式测试
4. **反馈摘要**：次级 LLM 压缩编译器日志以适配上下文窗口

## 使用方法

```javascript
Workflow({name: 'tritorx-operator-generation', args: {
  operator_name: 'aten::softmax',
  operator_docstring: '...',
  target_platform: 'MTIA',
  max_attempts: 3,
}})
```

## 参考文献

- [Agentic Operator Generation for ML ASICs](https://arxiv.org/abs/2512.10977) — Hammond, Markosyan 等 (Meta, 2025)
