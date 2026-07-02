# AutoMegaKernel：Megakernel Schedule Search Adapter

[English](README.md) · **简体中文**

这是面向 [AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel) 的严格 adapter workflow，来源论文为 [arXiv:2606.09682](https://arxiv.org/abs/2606.09682)。

## 概述

AutoMegaKernel（AMK）把 HuggingFace Llama-family 模型编译成一个 persistent cooperative CUDA megakernel，并优化 schedule，而不是让 agent 随意改写原始 CUDA 源码。Agent 可见的编辑面是结构化的 `ScheduleConfig` 与可选 `kernel_knobs`；AMK 自己负责 lowering、launch 前 deadlock/race 静态验证、全模型正确性检查、latency/roofline 测量，以及 keep/revert。

这个 AKW workflow 是 AMK harness 的**严格 adapter**。它不是 AMK 的独立重写，也**不是通用 CUDA kernel optimizer**。

## 适用场景

适合：

- 依赖已有的 AutoMegaKernel 仓库，并且 AMK 环境可运行。
- 目标是 AMK 支持的 Llama-family CUDA megakernel。
- 想用 Claude Code Workflow 编排 `amk propose`、`amk eval`、`amk loop` 或 `amk autoresearch`。

不适合：

- 优化普通 standalone CUDA/Triton/CUTLASS kernel。
- 没有安装或 checkout AMK。
- 目标是 high-batch serving throughput，而不是 AMK 的 single-stream decode 场景。
- 希望 AKW 内部复刻 AMK 的 VM、schedule validator、lowering 或 instruction ABI。

## 依赖边界

依赖已有的 AutoMegaKernel 仓库。权威证据全部来自 AMK：

| 证据 | Owner |
| --- | --- |
| 可编辑搜索面 | `amk propose` |
| validate-before-launch schedule 安全性 | AMK schedule validator |
| 正确性 | AMK full-model oracle |
| Latency 与 roofline | AMK eval/bench path |
| Keep/revert 与 flywheel 日志 | `amk loop` / `amk autoresearch` |

## 核心循环

```text
Setup AMK
→ Surface: amk propose
→ Baseline: amk eval incumbent ScheduleConfig
→ Search: amk loop 或 amk autoresearch
→ Audit: 没有 valid=true 且 correct=true 时不得报告 latency
→ Report: best_config、best_verdict、speedup、results_tsv/report
```

## 使用方法

```javascript
Workflow({name: 'automegakernel-megakernel-optimization', args: {
  amk_root: '/abs/path/AutoMegaKernel',
  model: 'toy',
  target_gpu: 'rtx5090',
  mode: 'loop',
  iterations: 16,
  device: 'auto',
  min_speedup: 1.01,
  exp_dir: '/tmp/amk-akw-run',
}})
```

长时间 AMK campaign：

```javascript
Workflow({name: 'automegakernel-megakernel-optimization', args: {
  amk_root: '/abs/path/AutoMegaKernel',
  model: 'toy-2L',
  target_gpu: 'rtx5090',
  mode: 'autoresearch',
  minutes: 480,
  overnight: true,
  device: 'cuda',
}})
```

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `amk_root` | 必填 | 已有 AutoMegaKernel checkout。 |
| `model` | 必填 | AMK model id，例如 `toy`、`toy-2L` 或受支持的 HuggingFace id。 |
| `target_gpu` | 必填 | AMK `GpuTarget`，例如 `rtx5090`、`b200`、`h100`、`a100`。 |
| `mode` | `loop` | `loop` 或 `autoresearch`。 |
| `iterations` | `8` | AMK loop trial 数；未设置 `minutes` 时也作为 autoresearch iters。 |
| `minutes` | 未设置 | AMK autoresearch 的 wall-clock 分钟数。 |
| `device` | `auto` | AMK device selector。 |
| `amk_command` | `amk` | AMK executable 名称或路径。 |
| `overnight` | `false` | autoresearch 时传给 AMK 的 overnight 模式。 |
| `cold` | `false` | autoresearch 时传给 AMK 的 cold-start 模式。 |
| `min_speedup` | `1.01` | `ok=true` 所需的最小加速比。 |
| `exp_dir` | `.` | adapter scratch 目录，用于 genome lines 与 incumbent config。 |

## Taxonomy

| 维度 | 值 |
| --- | --- |
| Category | `search_based` |
| Topology | iterative schedule-space search |
| Backend | CUDA via AutoMegaKernel |
| Feedback | AMK correctness-gated latency and roofline verdict |
| State | AMK best config、rows/results TSV、flywheel corpus |
| Fidelity | strict adapter to external AMK harness |

## 参考

- [AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis](https://arxiv.org/abs/2606.09682)
- [RightNow-AI/AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel)
