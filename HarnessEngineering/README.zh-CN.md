# Harness Engineering 内核优化

[English](README.md) · **简体中文**

这是对论文 [Harness Engineering for LLM-Driven GPU Kernel Generation](https://arxiv.org/abs/2607.17979) 及其公开 [MLSys 2026 FlashInfer 竞赛产物](https://github.com/syhya/mlsys26-flashinfer-contest) 的证据优先 workflow 适配。

## 保留的关键边界

| 责任 | 权威所有者 |
| --- | --- |
| 基于 profile 选择假设 | workflow controller |
| 实现候选 | workflow implementer，只能写 `exp_dir` |
| 编译与正确性裁决 | 用户提供的 `test_command` |
| 更深的安全/可移植性裁决 | 可选的 `verification_command` |
| 计时与 speedup | 用户提供的 `benchmark_command` |
| 候选保留与晋升 | workflow 的确定性 gate |

优化开始前先冻结评测契约。候选只有在编译成功、正确性通过，并且（配置时）通过所选深度验证后才允许计时；只有实测严格优于当前最优，才会被晋升。

这是 **workflow adaptation**，不是对竞赛基础设施、隐藏测试或论文模型配置的重写。

## 核心循环

```text
冻结 harness 契约
→ 建立正确性门控的 baseline
→ 可选 profile + 一个有界假设
→ 在 exp_dir 内实现候选
→ 官方式正确性 / 验证 / 计时 gate
→ 严格更快则保留，否则拒绝
→ 审计候选与证据绑定
```

## 共享评测配置

配置名只说明应收集什么证据，不能代替可执行 verifier：

| 配置 | 重点 |
| --- | --- |
| `contract-grade` | 受 [Kernel Contracts](https://arxiv.org/abs/2608.12700) 启发的编译、数值、安全、确定性和可移植性义务 |
| `kernelbench-verified` | 受 [KernelBench-Verified](https://arxiv.org/abs/2607.16241) 启发的隐藏分布正确性、内存效率与真实 TF32 baseline |
| `kernelgenbench` | 受 [KernelGenBench](https://arxiv.org/abs/2607.27231) 启发的多来源、多硬件与成本/质量溯源 |
| `custom` | 项目自有证据契约 |

细节见 [`_substrate/verification/README.md`](../_substrate/verification/README.md)。一旦提供 `verification_command`，晋升必须看到它返回 `verified=true`。

## 用法

```javascript
Workflow({name: 'harness-engineering-kernel-optimization', args: {
  harness_root: '/abs/path/mlsys26-flashinfer-contest',
  kernel_path: '/abs/path/solution.py',
  problem_path: '/abs/path/config.toml',
  backend: 'cuda',
  test_command: './project-test --candidate {candidate_path} --json {result_path}',
  benchmark_command: './project-bench --candidate {candidate_path} --json {result_path}',
  profile_command: './project-profile --candidate {candidate_path} --out {artifact_path}',
  verification_command: './project-verify --candidate {candidate_path} --profile kernelbench-verified --json {result_path}',
  verification_profile: 'kernelbench-verified',
  iterations: 6,
  min_speedup: 1.05,
  exp_dir: '/tmp/harness-engineering-run',
}})
```

这些命令只是契约形状示例，不是 workflow 内置工具；必须替换成目标 harness 自己拥有的命令。

## 证据契约

规范化结果包含 `candidate_path`、`compiled`、`correct`、`verified`、`latency_ms`、`speedup`、`evidence_path` 与原始 `artifact_paths`。缺失或失败证据必须保持为 `false`/`null`，不能把文字说明升级成通过。

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `harness_root` | 必填 | 权威评测 harness checkout。 |
| `kernel_path` | 必填 | 已有 baseline solution。 |
| `test_command` | 必填 | 编译与正确性命令模板。 |
| `benchmark_command` | 必填 | 正确性门控的计时命令模板。 |
| `profile_command` | 空 | 可选 profile 命令。 |
| `verification_command` | 空 | 可选深度 verifier。 |
| `verification_profile` | `contract-grade` | 共享评测配置之一。 |
| `iterations` | `4` | 候选预算。 |
| `min_speedup` | `1.01` | 提前成功阈值。 |
| `exp_dir` | `/tmp/harness-engineering` | workflow 唯一写入边界。 |

## 参考资料

- [Harness Engineering 论文](https://arxiv.org/abs/2607.17979)
- [公开竞赛产物](https://github.com/syhya/mlsys26-flashinfer-contest)
- [Kernel Contracts](https://arxiv.org/abs/2608.12700)
- [KernelBench-Verified](https://arxiv.org/abs/2607.16241)
- [KernelGenBench](https://arxiv.org/abs/2607.27231)
