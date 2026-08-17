# Atrex 内核优化适配器

[English](README.md) · **简体中文**

这是 [Atrex Kernel Agent](https://github.com/alibaba/atrex-kernel-agent) 的严格适配器；对应论文为 [Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent](https://arxiv.org/abs/2607.14541)。

## 为什么采用严格适配器

当前 Atrex 仓库只有一个受支持的优化入口：`orchestrator/optimize.py`。它的 supervisor 已经拥有这些关键责任：

- 每个 framework/target 独立 Git workspace；
- 先建立正确性通过的 V0，生产模式下默认再建立 framework-native V1；
- 在完整 workload 集合上运行 Long Horizon clean-session episodes；
- profile/research/plan/edit/repair 循环与 optimization dropout；
- 分层 GPU wiki、参考 kernel 和上游源码检索；
- 显式预算、恢复状态；
- 终局验证、同 allocation 的 ABBA 验证与 squash promotion；
- 规范 `memory/v<N>.json`、journal、profiler 产物和聚合溯源。

若在 AKW 里重写这些控制逻辑，就会产生第二个权威源。因此本 workflow 只做四件事：检查 checkout 与命令、只启动一次官方 orchestrator、审计规范证据、返回官方晋升结果。

## 权威边界

| 证据 / 决策 | 权威所有者 |
| --- | --- |
| 公共问题推导与隐藏 workload 隔离 | Atrex orchestrator |
| Profiling、optimization dropout、知识检索 | Atrex episode engine |
| 完整 workload 正确性 | Atrex terminal validator |
| 严格性能提升 | Atrex same-allocation ABBA verifier |
| 晋升 | Atrex supervisor 与 squash promotion |
| Adapter 就绪检查与最终证据审计 | 本 AKW workflow |

`memory/live.json` 只用于可观测性，不是晋升证据。只有规范 `memory/v<N>.json` 与官方 terminal/ABBA 记录可以支持成功结论。

## 核心流程

```text
检查已有 Atrex checkout 与命令
→ 只启动一次 orchestrator/optimize.py
→ 所有内层循环由官方 Atrex campaign 拥有
→ 审计 canonical memory + journal + terminal validation + ABBA + promotion
→ 返回精确的晋升产物与证据路径
```

## 用法

```javascript
Workflow({name: 'atrex-kernel-optimization', args: {
  atrex_root: '/abs/path/atrex-kernel-agent',
  operator_input: 'atrex-bench/attention_forward',
  atrex_command: 'python3 {atrex_root}/orchestrator/optimize.py --operator {operator_input} --workspace {exp_dir} --platform {platform} --framework {framework} --mode {mode} --max-iters {max_iters}',
  platform: 'H20',
  framework: 'cuda',
  mode: 'production',
  max_iters: 300,
  exp_dir: '/tmp/atrex-akw',
  min_speedup: 1.01,
  turn_timeout_min: 720,
}})
```

workflow 不会根据猜测拼装命令。`atrex_command` 由调用方提供；若它没有调用当前 checkout 的官方 `orchestrator/optimize.py`，或没有把 campaign 状态限制在 `exp_dir` 下，doctor 阶段会拒绝执行。

## 适用范围

- SOL-ExecBench operator 优化；
- 原生 Atrex-Bench operators；
- 官方仓库支持的 NVIDIA CUDA 或 AMD ROCm campaign；
- Atrex-Bench 支持的 Triton、Gluon、FlyDSL、CuteDSL 候选。

若缺少 Atrex checkout、GPU gateway/platform 配置或官方命令，则不应使用。普通项目自有 harness 应使用 [HarnessEngineering](../HarnessEngineering/)。

## Fidelity

`strict_high_fidelity`：真正执行方法的是官方 Atrex checkout；AKW 不重写 Long Horizon engine、profile-driven optimizer、optimization dropout、隐藏 evaluator 或 promotion policy。

## 参考资料

- [Atrex 论文](https://arxiv.org/abs/2607.14541)
- [alibaba/atrex-kernel-agent](https://github.com/alibaba/atrex-kernel-agent)
- [alibaba/atrex-bench](https://github.com/alibaba/atrex-bench)
