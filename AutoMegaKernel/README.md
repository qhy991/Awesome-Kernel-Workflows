# AutoMegaKernel: Megakernel Schedule Search Adapter

**English** · [简体中文](README.zh-CN.md)

Strict adapter workflow for [AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel), based on [arXiv:2606.09682](https://arxiv.org/abs/2606.09682).

## Overview

AutoMegaKernel (AMK) compiles a HuggingFace Llama-family model into one persistent cooperative CUDA megakernel and optimizes the schedule, not arbitrary raw CUDA source. The agent-visible edit surface is a structured `ScheduleConfig` plus optional `kernel_knobs`; AMK lowers that object through its own VM, validates deadlock/race safety before launch, checks full-model correctness, measures latency/roofline distance, and keeps or reverts candidates.

This AKW workflow is a **strict adapter** to that harness. It is **not a standalone reimplementation** of AMK and **not a general CUDA kernel optimizer**.

## When To Use

Use this workflow when:

- You have an AutoMegaKernel checkout and a working AMK environment.
- The target is an AMK-supported Llama-family CUDA megakernel.
- You want Claude Code Workflow orchestration around `amk propose`, `amk eval`, `amk loop`, or `amk autoresearch`.

Avoid this workflow when:

- You need to optimize an ordinary standalone CUDA/Triton/CUTLASS kernel.
- You do not have AMK installed.
- You need high-batch serving throughput rather than AMK's single-stream decode regime.
- You want AKW to reproduce AMK's VM, schedule validator, lowering, or instruction ABI internally.

## Dependency Boundary

Requires an existing AutoMegaKernel checkout. The authoritative evidence owner is AMK itself:

| Evidence | Owner |
| --- | --- |
| Editable search surface | `amk propose` |
| validate-before-launch schedule safety | AMK schedule validator |
| Correctness | AMK full-model oracle |
| Latency and roofline | AMK eval/bench path |
| Keep/revert and flywheel logs | `amk loop` / `amk autoresearch` |

## Core Loop

```text
Setup AMK
→ Surface: amk propose
→ Baseline: amk eval incumbent ScheduleConfig
→ Search: amk loop or amk autoresearch
→ Audit: no latency without valid=true and correct=true
→ Report: best_config, best_verdict, speedup, results_tsv/report
```

## Usage

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

For a longer AMK campaign:

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

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `amk_root` | required | Existing AutoMegaKernel checkout. |
| `model` | required | AMK model id, such as `toy`, `toy-2L`, or a supported HuggingFace id. |
| `target_gpu` | required | AMK `GpuTarget`, such as `rtx5090`, `b200`, `h100`, or `a100`. |
| `mode` | `loop` | `loop` or `autoresearch`. |
| `iterations` | `8` | AMK loop trials, or autoresearch iters when `minutes` is unset. |
| `minutes` | unset | Wall-clock minutes for AMK autoresearch. |
| `device` | `auto` | AMK device selector. |
| `amk_command` | `amk` | AMK executable name or path. |
| `overnight` | `false` | Pass AMK overnight mode for autoresearch. |
| `cold` | `false` | Pass AMK cold-start mode for autoresearch. |
| `min_speedup` | `1.01` | Required speedup for `ok=true`. |
| `exp_dir` | `.` | Adapter scratch directory for genome lines and incumbent config. |

## Taxonomy

| Dimension | Value |
| --- | --- |
| Category | `search_based` |
| Topology | iterative schedule-space search |
| Backend | CUDA via AutoMegaKernel |
| Feedback | AMK correctness-gated latency and roofline verdict |
| State | AMK best config, rows/results TSV, flywheel corpus |
| Fidelity | strict adapter to external AMK harness |

## References

- [AutoMegaKernel: A Statically-Checked Agent Harness for Self-Retargeting Megakernel Synthesis](https://arxiv.org/abs/2606.09682)
- [RightNow-AI/AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel)
